from __future__ import annotations

import json
from pathlib import Path

from swaag.communication import CommunicationService, CommunicationStore
from swaag.embedding_index import DerivedEmbeddingIndex
from swaag.history import HistoryStore
from swaag.mcp import McpAdapter
from swaag.runtime import AgentRuntime


class _FakeEmbeddingProvider:
    def embed(self, texts):
        vectors = []
        for text in texts:
            lower = text.casefold()
            vectors.append([
                float(lower.count("cache")),
                float(lower.count("release")),
                float(lower.count("network")),
            ])
        return vectors


def test_history_archive_is_exact_read_only_fallback(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint="cfg", model_base_url="http://model", session_name="archive-demo", session_name_source="explicit")
    store.record_event(state, "message_added", {"message": {"role": "user", "content": "archive-marker-712", "created_at": "2026-01-01T00:00:00+00:00", "name": None, "metadata": {}}})
    original = store.read_history(state.session_id)
    archived = store.archive_session(state.session_id, remove_active=True)
    assert archived["event_count"] == len(original)
    assert not store.history_path(state.session_id).exists()
    restored = store.read_history(state.session_id)
    assert [(e.sequence, e.event_type, e.hash) for e in restored] == [(e.sequence, e.event_type, e.hash) for e in original]
    details = store.query_history_details("archive-demo", "archive-marker-712", max_results=4)
    assert details["search_backend"] == "archive_fts5"
    assert details["matches"]
    shard = Path(archived["shard_path"])
    assert shard.exists()
    assert shard.stat().st_mode & 0o222 == 0


def test_derived_embedding_index_is_non_authoritative_and_ranked(tmp_path: Path) -> None:
    index = DerivedEmbeddingIndex(tmp_path, _FakeEmbeddingProvider())
    count = index.rebuild_session(
        "session_x",
        [
            (1, "situation", "cache miss cache miss"),
            (2, "action", "prepare release"),
            (3, "reason", "network unavailable"),
        ],
    )
    assert count == 3
    assert index.complete_through("session_x") == 3
    matches = index.search("cache regression", session_id="session_x", limit=2)
    assert matches[0].sequence == 1
    assert matches[0].field == "situation"
    assert matches[0].score > matches[1].score


def test_runtime_can_enable_background_derived_embedding_index(
    make_config, tmp_path: Path, monkeypatch
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.embedding_index.enabled = True
    config.embedding_index.base_url = "http://embedding.invalid"
    config.embedding_index.model = "fake-embedding"
    monkeypatch.setattr(
        "swaag.runtime.OpenAICompatibleEmbeddingProvider",
        lambda *_args, **_kwargs: _FakeEmbeddingProvider(),
    )

    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    runtime.history.record_event(
        state,
        "agent_status",
        {
            "action_index": 1,
            "situation": "cache regression",
            "action": "inspect release",
            "reason": "network evidence",
            "importance": "normal",
            "importance_rank": 1,
        },
    )
    assert runtime._embedding_indexer is not None
    runtime._embedding_indexer.flush(timeout=2)
    matches = runtime._embedding_indexer.index.search(
        "cache regression", session_id=state.session_id, limit=1
    )
    runtime._embedding_indexer.close()

    assert matches[0].field == "situation"


def test_communication_store_prioritizes_stop_and_preserves_correlation(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    service = CommunicationService(runtime)
    ordinary = service.submit(state.session_id, "inspect status")
    pause = service.submit(state.session_id, "pause")
    stop = service.submit(state.session_id, "stop now")
    assert ordinary.correlation_id != pause.correlation_id != stop.correlation_id
    assert service.store.next_pending(state.session_id).correlation_id == stop.correlation_id
    payload = json.loads(service.answer_status_question(state.session_id, "What is happening?"))
    assert payload["session_id"] == state.session_id
    service.store.set_status(stop.correlation_id, "completed", reply="stopped")
    completed = service.status(stop.correlation_id)
    assert completed.status == "completed"
    assert completed.reply == "stopped"
    assert completed.completed_at


def test_mcp_adapter_lists_and_calls_canonical_tools(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    mcp = McpAdapter(runtime)
    initialized = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert initialized["result"]["protocolVersion"] == "2026-07-28"
    listed = mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = {item["name"] for item in listed["result"]["tools"]}
    assert "calculator" in names
    called = mcp.handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "calculator", "arguments": {"expression": "6 * 7"}, "session": state.session_id},
    })
    assert called["result"]["isError"] is False
    assert called["result"]["session_id"] == state.session_id
    assert called["result"]["structuredContent"]["result"] == 42


def test_control_tools_read_status_and_queue_priority(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config)
    source = runtime.create_or_load_session()
    target = runtime.history.create(
        config_fingerprint="cfg",
        model_base_url="http://model",
        session_name="target-agent",
        session_name_source="explicit",
    )
    runtime.history.record_event(
        target,
        "message_added",
        {"message": {"role": "user", "content": "Deploy release 12", "created_at": "2026-01-01T00:00:00+00:00", "name": None, "metadata": {}}},
    )
    status_run = runtime.execute_tool_once(
        "agent_status_lookup",
        {"session_ref": "target-agent"},
        session_id=source.session_id,
    )
    assert status_run.tool_result.output["session_id"] == target.session_id
    assert status_run.tool_result.output["active_goal"] == "Deploy release 12"

    control_run = runtime.execute_tool_once(
        "agent_control",
        {"session_ref": "target-agent", "message": "pause"},
        session_id=source.session_id,
    )
    assert control_run.tool_result.output["queued"] is True
    pending = runtime.history.list_pending_control_messages(target.session_id)
    assert pending[0]["message"] == "pause"
    assert pending[0]["priority"] == 80
