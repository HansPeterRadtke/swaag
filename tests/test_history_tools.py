from __future__ import annotations

from pathlib import Path

import pytest

from swaag.history import HistoryStore
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
