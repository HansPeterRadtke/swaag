from __future__ import annotations

import json
import sqlite3
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
    with sqlite3.connect(f"file:{shard}?mode=ro&immutable=1", uri=True) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1


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


def test_communication_store_preserves_delivery_order_and_correlation(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    service = CommunicationService(runtime)
    ordinary = service.submit(state.session_id, "inspect status")
    pause = service.submit(state.session_id, "pause")
    stop = service.submit(state.session_id, "stop now")
    assert ordinary.correlation_id != pause.correlation_id != stop.correlation_id
    assert service.store.next_pending(state.session_id).correlation_id == ordinary.correlation_id
    assert {ordinary.priority, pause.priority, stop.priority} == {0}
    runtime.generate_communication_status = lambda **kwargs: {  # type: ignore[method-assign]
        "answer": "The session is idle.",
        "source_event_references": [],
        "evidence_projected": False,
    }
    answer = service.answer_status_question(state.session_id, "What is happening?")
    assert answer == "The session is idle."
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
    metadata = {
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"},
            "io.modelcontextprotocol/clientCapabilities": {},
        }
    }
    discovered = mcp.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": metadata}
    )
    assert discovered["result"]["supportedVersions"] == ["2026-07-28"]
    assert discovered["result"]["resultType"] == "complete"
    assert discovered["result"]["ttlMs"] == 0
    assert discovered["result"]["cacheScope"] == "private"
    listed = mcp.handle(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": metadata}
    )
    ordered_names = [item["name"] for item in listed["result"]["tools"]]
    names = set(ordered_names)
    assert "calculator" in names
    assert ordered_names == sorted(ordered_names)
    assert listed["result"]["ttlMs"] == 0
    assert listed["result"]["cacheScope"] == "private"
    called = mcp.handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {
            "_meta": {
                **metadata["_meta"],
                "com.swaag/sessionId": state.session_id,
            },
            "name": "calculator",
            "arguments": {"expression": "6 * 7"},
        },
    })
    assert called["result"]["isError"] is False
    assert called["result"]["_meta"]["com.swaag/sessionId"] == state.session_id
    assert called["result"]["structuredContent"]["result"] == 42


def test_mcp_distinguishes_invalid_calls_from_tool_execution_errors(
    make_config, tmp_path: Path
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    mcp = McpAdapter(AgentRuntime(config))
    metadata = {
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {},
        }
    }

    unknown = mcp.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {**metadata, "name": "not_registered", "arguments": {}},
        }
    )
    invalid = mcp.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {**metadata, "name": "calculator", "arguments": {}},
        }
    )
    failed = mcp.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                **metadata,
                "name": "calculator",
                "arguments": {"expression": "1 / 0"},
            },
        }
    )

    assert unknown["error"]["code"] == -32602
    assert invalid["error"]["code"] == -32602
    assert failed["result"]["isError"] is True
    assert failed["result"]["structuredContent"]["error"]["error_type"] == "ZeroDivisionError"
    assert failed["result"]["_meta"]["com.swaag/sessionId"]


def test_mcp_2026_rejects_missing_per_request_metadata(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=object())
    response = McpAdapter(runtime).handle(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    )

    assert response["error"]["code"] == -32602
    assert "requires params._meta" in response["error"]["message"]


def test_mcp_2026_allows_omitted_client_identity_but_rejects_malformed_identity(
    make_config,
) -> None:
    adapter = McpAdapter(AgentRuntime(make_config(), model_client=object()))
    metadata = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
    }

    anonymous = adapter.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": {"_meta": metadata},
        }
    )
    malformed = adapter.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "server/discover",
            "params": {
                "_meta": {
                    **metadata,
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "client-without-version"
                    },
                }
            },
        }
    )

    assert anonymous["result"]["supportedVersions"] == ["2026-07-28"]
    assert malformed["error"]["code"] == -32602
    assert "clientInfo" in malformed["error"]["message"]


def test_control_tools_read_status_and_queue_exact_message(make_config, tmp_path: Path) -> None:
    config = make_config(
        tools__enabled=["agent_status_lookup", "agent_control"],
        tools__allow_side_effect_tools=True,
    )
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
    for action_index in range(7):
        runtime.history.record_event(
            target,
            "agent_status",
            {
                "action_index": action_index,
                "situation": f"status {action_index}",
                "action": "inspect",
                "reason": "test status provenance",
                "importance": "normal",
                "importance_rank": 1,
            },
        )
    status_run = runtime.execute_tool_once(
        "agent_status_lookup",
        {"session_ref": "target-agent"},
        session_id=source.session_id,
    )
    assert status_run.tool_result.output["session_id"] == target.session_id
    assert status_run.tool_result.output["active_goal"] == "Deploy release 12"
    assert [item["situation"] for item in status_run.tool_result.output["status_history"]] == [
        f"status {index}" for index in range(7)
    ]

    control_run = runtime.execute_tool_once(
        "agent_control",
        {"session_ref": "target-agent", "message": "pause"},
        session_id=source.session_id,
    )
    assert control_run.tool_result.output["queued"] is True
    pending = runtime.history.list_pending_control_messages(target.session_id)
    assert pending[0]["message"] == "pause"
    assert pending[0]["priority"] == 0


def test_direct_runtime_tool_call_cannot_bypass_configured_enablement(
    make_config, tmp_path: Path
) -> None:
    config = make_config(tools__enabled=["echo"])
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config)

    run = runtime.execute_tool_once("calculator", {"expression": "1 + 1"})

    assert run.tool_result is None
    assert run.error is not None
    assert run.error["error_type"] == "PermissionError"
    assert "not enabled by configuration" in run.error["error"]
