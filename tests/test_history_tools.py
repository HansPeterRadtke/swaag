from __future__ import annotations

from pathlib import Path

import pytest

from swaag.history import HistoryStore
from swaag.runtime import AgentRuntime
from swaag.tools.base import ToolValidationError
from swaag.tools.registry import ToolRegistry
from swaag.types import Message
from swaag.utils import utc_now_iso


def _state_with_history(config, session_id: str = "session_history_tools"):
    store = HistoryStore(config.sessions.root)
    state = store.create(
        config_fingerprint=config.config_fingerprint(),
        model_base_url=config.model.base_url,
        session_id=session_id,
    )
    store.record_event(
        state,
        "message_added",
        {"message": {"role": "user", "content": "Deployment codename is Blue Heron.", "created_at": utc_now_iso(), "name": None, "metadata": {}}},
    )
    store.record_event(
        state,
        "tool_result",
        {"tool_name": "echo", "raw_input": {"text": "artifact-marker-73"}, "validated_input": {"text": "artifact-marker-73"}, "output": {"text": "artifact-marker-73"}, "display_text": "artifact-marker-73"},
    )
    store.record_event(
        state,
        "message_added",
        {"message": {"role": "assistant", "content": "Remembered Blue Heron.", "created_at": utc_now_iso(), "name": None, "metadata": {}}},
    )
    return store, state


def test_history_search_tool_finds_ranked_exact_history(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config)
    registry = ToolRegistry()

    invocation, result = registry.dispatch(
        "history_search",
        {"query": '"Blue Heron"', "max_results": 5},
        config,
        state,
    )

    assert invocation.validated_input["query"] == '"Blue Heron"'
    assert result.output["session_id"] == state.session_id
    assert result.output["match_count"] >= 1
    assert any("Blue Heron" in item["preview"] for item in result.output["matches"])
    assert all("payload" not in item for item in result.output["matches"])
    assert [event.event_type for event in result.generated_events] == ["history_retrieved"]


def test_history_search_tool_defaults_to_current_session(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config, "session_current")
    other = store.create(config_fingerprint="cfg", model_base_url="http://model", session_id="session_other")
    store.record_event(other, "message_added", {"message": {"role": "user", "content": "Blue Heron only in other", "created_at": utc_now_iso(), "name": None, "metadata": {}}})

    _, result = ToolRegistry().dispatch("history_search", {"query": "artifact-marker-73"}, config, state)

    assert result.output["session_id"] == "session_current"
    assert result.output["match_count"] == 1


def test_history_search_tool_caps_results_to_config(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.history_search.max_results = 2
    store, state = _state_with_history(config)
    for index in range(6):
        store.record_event(state, "message_added", {"message": {"role": "user", "content": f"needle {index}", "created_at": utc_now_iso(), "name": None, "metadata": {}}})

    _, result = ToolRegistry().dispatch("history_search", {"query": "needle", "max_results": 99}, config, state)

    assert result.output["match_count"] == 2


def test_history_search_validation_rejects_empty_query(make_config) -> None:
    tool = ToolRegistry().get("history_search")
    with pytest.raises(ToolValidationError):
        tool.validate({"query": "   "})


def test_history_window_returns_exact_events(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config)

    _, result = ToolRegistry().dispatch(
        "history_window",
        {"start_sequence": 2, "limit": 2},
        config,
        state,
    )

    events = result.output["events"]
    assert [event["sequence"] for event in events] == [2, 3]
    assert events[0]["payload"]["message"]["content"] == "Deployment codename is Blue Heron."
    assert events[1]["payload"]["output"]["text"] == "artifact-marker-73"
    assert [event.event_type for event in result.generated_events] == ["history_window_read"]


def test_history_window_can_read_named_other_session(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config, "session_primary")
    other = store.create(config_fingerprint="cfg", model_base_url="http://model", session_id="session_secondary", session_name="secondary")
    store.record_event(other, "message_added", {"message": {"role": "user", "content": "secondary fact", "created_at": utc_now_iso(), "name": None, "metadata": {}}})

    _, result = ToolRegistry().dispatch(
        "history_window",
        {"session_ref": "secondary", "start_sequence": 1, "limit": 2},
        config,
        state,
    )

    assert result.output["session_id"] == "session_secondary"
    assert result.output["events"][-1]["payload"]["message"]["content"] == "secondary fact"


def test_history_window_validation_bounds(make_config) -> None:
    tool = ToolRegistry().get("history_window")
    with pytest.raises(ToolValidationError):
        tool.validate({"start_sequence": 0})
    with pytest.raises(ToolValidationError):
        tool.validate({"start_sequence": 1, "limit": 21})


def test_history_tools_are_enabled_by_default(make_config) -> None:
    config = make_config()
    names = set(ToolRegistry().tool_names(config))
    assert {"history_search", "history_window"} <= names


def test_sqlite_wal_fts_is_durable_index_for_exact_history(make_config, tmp_path: Path) -> None:
    import sqlite3

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config, session_id="session_sqlite_index")
    assert store.sqlite_history_path().exists()
    with sqlite3.connect(store.sqlite_history_path()) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].casefold() == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        count = connection.execute("SELECT COUNT(*) FROM events WHERE session_id=?", (state.session_id,)).fetchone()[0]
        assert count == len(store.read_history(state.session_id))
        fts_count = connection.execute("SELECT COUNT(*) FROM events_fts WHERE session_id=?", (state.session_id,)).fetchone()[0]
        assert fts_count == count
    details = store.query_history_details(state.session_id, "Blue Heron", max_results=4)
    assert details["search_backend"] == "sqlite_fts5"
    assert details["matches"]
    assert any("Blue Heron" in item["preview"] for item in details["matches"])


def test_sqlite_control_priority_and_processed_idempotency(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint=config.config_fingerprint(), model_base_url=config.model.base_url)
    ordinary = store.enqueue_control_message(state.session_id, "please inspect logs", control_id="control_normal")
    pause = store.enqueue_control_message(state.session_id, "pause", control_id="control_pause")
    stop = store.enqueue_control_message(state.session_id, "stop now", control_id="control_stop")
    pending = store.list_pending_control_messages(state.session_id)
    assert [item["control_id"] for item in pending] == [stop["control_id"], pause["control_id"], ordinary["control_id"]]
    store.mark_control_message_processed(state.session_id, stop["control_id"])
    store.enqueue_control_message(state.session_id, "stop now", control_id=stop["control_id"])
    assert stop["control_id"] not in {item["control_id"] for item in store.list_pending_control_messages(state.session_id)}


def test_history_analyze_is_grounded_in_exact_candidate_sequences(make_config, tmp_path: Path, monkeypatch) -> None:
    import json

    from swaag.environment.environment import AgentEnvironment
    from swaag.tools.base import ToolContext
    from swaag.types import CompletionResult

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config, session_id="session_history_analyze")
    source_sequence = next(event.sequence for event in store.read_history(state.session_id) if "artifact-marker-73" in str(event.payload))

    def fake_complete(self, prompt, *, max_tokens, contract, temperature=None, kind=None, live_mode=False):
        assert contract.name == "history_analysis"
        assert str(source_sequence) in prompt
        return CompletionResult(
            text=json.dumps(
                {
                    "goal_constraints": ["Recover the exact prior artifact marker."],
                    "failure_evidence": ["The marker exists in a prior tool result."],
                    "candidate_root_causes": ["The previous strategy did not retrieve the exact historical tool result."],
                    "source_sequences": [source_sequence],
                    "wrong_strategy": "Relying on incomplete current context.",
                    "recommended_strategy": "Retrieve the exact history event and use its value.",
                    "uncertainties": [],
                }
            ),
            raw_request={}, raw_response={}, prompt_tokens=None, completion_tokens=None, finish_reason="stop",
        )

    monkeypatch.setattr("swaag.tools.history.LlamaCppClient.complete", fake_complete)
    env = AgentEnvironment(config, state)
    registry = ToolRegistry()
    invocation, result = registry.dispatch(
        "history_analyze",
        {"query": "Why did we miss artifact-marker-73?", "session_ref": state.session_id, "max_events": 8},
        config,
        state,
    )
    assert invocation.tool_name == "history_analyze"
    assert result.output["source_sequences"] == [source_sequence]
    assert result.output["recommended_strategy"].startswith("Retrieve the exact history event")
    assert [event.event_type for event in result.generated_events] == ["history_analyzed"]


def test_history_tools_accept_active_session_aliases(make_config, tmp_path: Path) -> None:
    from swaag.environment.environment import AgentEnvironment
    from swaag.tools.base import ToolContext

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint=config.config_fingerprint(), model_base_url=config.model.base_url, session_name="seed_23", session_name_source="explicit")
    store.record_event(
        state,
        "message_added",
        {"message": {"role": "user", "content": "alias-marker-91", "created_at": "2026-01-01T00:00:00+00:00", "name": None, "metadata": {}}},
    )
    env = AgentEnvironment(config, state)
    context = ToolContext(config=config, session_state=state, environment=env)
    registry = ToolRegistry()
    for session_ref in ("current", "seed_23", state.session_id):
        _invocation, result = registry.dispatch(
            "history_search",
            {"query": "alias-marker-91", "topic_hint": None, "session_ref": session_ref, "max_results": 4},
            config,
            state,
        )
        assert result.output["session_id"] == state.session_id
        assert result.output["matches"]


def test_history_search_exposes_search_backend_and_current_session_schema(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store, state = _state_with_history(config, session_id="session_backend_output")
    registry = ToolRegistry()
    _invocation, result = registry.dispatch(
        "history_search",
        {"query": "Blue Heron", "topic_hint": None, "session_ref": None, "max_results": 4},
        config,
        state,
    )
    assert result.output["search_backend"] == "sqlite_fts5"
    from swaag.tools.history import HistorySearchTool
    assert "session_ref=null" in HistorySearchTool.usage_guidance
    assert "never invent a session label" in HistorySearchTool.usage_guidance


def test_history_analyze_bounds_large_candidate_payloads(make_config, tmp_path: Path, monkeypatch) -> None:
    import json
    from swaag.types import CompletionResult

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint=config.config_fingerprint(), model_base_url=config.model.base_url)
    huge = "X" * 200_000 + " root-cause-marker-77"
    store.record_event(state, "assistant_progress", {"action_index": 1, "assistant_text": huge})
    captured: dict[str, str] = {}

    def fake_complete(self, prompt, *, max_tokens, contract, temperature=None, kind=None, live_mode=False):
        captured["prompt"] = prompt
        sequence = next(event.sequence for event in store.read_history(state.session_id) if event.event_type == "assistant_progress")
        return CompletionResult(
            text=json.dumps({
                "goal_constraints": ["diagnose"],
                "failure_evidence": ["bounded evidence"],
                "candidate_root_causes": ["cause"],
                "source_sequences": [sequence],
                "wrong_strategy": "old",
                "recommended_strategy": "new",
                "uncertainties": [],
            }),
            raw_request={}, raw_response={}, prompt_tokens=None, completion_tokens=None, finish_reason="stop",
        )

    monkeypatch.setattr("swaag.tools.history.LlamaCppClient.complete", fake_complete)
    registry = ToolRegistry()
    _invocation, result = registry.dispatch(
        "history_analyze",
        {"query": "root-cause-marker-77", "session_ref": None, "max_events": 12},
        config,
        state,
    )
    assert result.output["candidate_root_causes"] == ["cause"]
    assert len(captured["prompt"]) < 25_000
    assert "X" * 5000 not in captured["prompt"]
    assert "Bounded exact candidate excerpts" in captured["prompt"]


def test_history_search_excludes_its_own_current_invocation(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    runtime.history.record_event(
        state,
        "message_added",
        {"message": {"role": "assistant", "content": "Durable indexed-history marker: cobalt-history-fts-531.", "created_at": "2026-01-01T00:00:00+00:00", "name": None, "metadata": {}}},
    )
    run = runtime.execute_tool_once(
        "history_search",
        {"query": "durable indexed-history marker", "topic_hint": "history marker", "session_ref": None, "max_results": 1},
        session_id=state.session_id,
    )
    assert run.tool_result is not None
    match = run.tool_result.output["matches"][0]
    assert match["event_type"] == "message_added"
    assert "cobalt-history-fts-531" in match["preview"]
    assert match["sequence"] < max(
        event.sequence for event in runtime.history.read_history(state.session_id)
        if event.event_type == "tool_called" and event.payload.get("tool_name") == "history_search"
    )
