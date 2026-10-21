import json

import pytest

from swaag.history import HistoryCorruptionError
from swaag.orchestrator_stream import OrchestratorEventCursor
from swaag.runtime import AgentRuntime


def setup_history(make_config):
    runtime = AgentRuntime(make_config(), model_client=object())
    return runtime, runtime.create_or_load_session()


def test_filtered_page_never_decodes_unrelated_model_payloads(make_config, monkeypatch):
    runtime, state = setup_history(make_config)
    history = runtime.history
    history.record_event(state, "error", {"operation": "test", "error_type": "Fixture",
                                         "error": "UNNEEDED_LARGE_PAYLOAD" * 10000})
    selected = history.record_event(state, "turn_started", {"turn_index": 1, "user_text": "Hello."})
    history._ensure_session_indexed(state.session_id)
    original = json.loads
    def decode(text, *args, **kwargs):
        assert "UNNEEDED_LARGE_PAYLOAD" not in text
        return original(text, *args, **kwargs)
    monkeypatch.setattr(json, "loads", decode)
    events, next_sequence = history.read_history_event_page(state.session_id, start_sequence=1,
        limit=128, event_types=("turn_started",))
    assert events == [selected] and next_sequence == selected.sequence + 1
    assert not history.has_history_at_or_after(state.session_id, next_sequence)


def test_empty_filtered_pages_advance_and_later_appends_are_observed(make_config):
    runtime, state = setup_history(make_config)
    history = runtime.history
    end = state.event_count
    events, next_sequence = history.read_history_event_page(state.session_id, start_sequence=1,
        limit=128, event_types=("assistant_progress",))
    assert events == [] and next_sequence == end + 1
    assert not history.has_history_at_or_after(state.session_id, next_sequence)
    added = history.record_event(state, "turn_started", {"turn_index": 1, "user_text": "Hello."})
    assert history.has_history_at_or_after(state.session_id, next_sequence)
    events, following = history.read_history_event_page(state.session_id, start_sequence=next_sequence,
        limit=128, event_types=("turn_started",))
    assert events == [added] and following == added.sequence + 1


def test_append_racing_filtered_read_is_left_for_next_page(make_config, monkeypatch):
    runtime, state = setup_history(make_config)
    history = runtime.history
    observed_end = state.event_count
    read = history._iter_history_sqlite
    added = []
    def racing(*args, **kwargs):
        if not added:
            added.append(history.record_event(state, "turn_started", {"turn_index": 1, "user_text": "New turn."}))
        return read(*args, **kwargs)
    monkeypatch.setattr(history, "_iter_history_sqlite", racing)
    events, next_sequence = history.read_history_event_page(state.session_id, start_sequence=1,
        limit=128, event_types=("turn_started",))
    assert events == [] and next_sequence == observed_end + 1
    events, _ = history.read_history_event_page(state.session_id, start_sequence=next_sequence,
        limit=128, event_types=("turn_started",))
    assert events == added


def test_filtered_reads_still_detect_external_journal_corruption(make_config):
    runtime, state = setup_history(make_config)
    history = runtime.history
    path = history.history_path(state.session_id)
    content = path.read_text()
    assert state.session_id in content
    path.write_text(content.replace(state.session_id, "session_corrupt", 1))
    with pytest.raises(HistoryCorruptionError):
        history.read_history_event_page(state.session_id, start_sequence=1,
            limit=128, event_types=("assistant_progress",))


def test_cursor_crosses_ignored_pages_without_replaying_another_turn(make_config):
    runtime, state = setup_history(make_config)
    history = runtime.history
    for index in range(260):
        history.record_event(state, "error", {"operation": "test", "error_type": "Fixture", "error": str(index)})
    history.record_event(state, "turn_started", {"turn_index": 1, "user_text": "Hello."})
    history.record_event(state, "control_message_processed", {"control_id": "wanted", "session_id": state.session_id,
        "message": "Hello.", "decision": {}})
    final = history.record_event(state, "turn_finished", {"turn_index": 1, "assistant_text": "Hello.",
        "tool_steps": 0, "budget_reports": []})
    history.record_event(state, "turn_started", {"turn_index": 2, "user_text": "Unrelated."})
    cursor = OrchestratorEventCursor(history, {"session_id": state.session_id, "control_id": "wanted"})
    pages = 0
    while not cursor.caught_up():
        assert cursor.poll() == []
        pages += 1
        assert pages < 10
    assert pages == 3 and cursor.final_sequence == final.sequence


def test_delivery_anchor_uses_exact_control_event_and_preserves_legacy_replay(make_config):
    runtime, state = setup_history(make_config)
    history = runtime.history
    for index in range(275):
        history.record_event(state, "error", {
            "operation": "test", "error_type": "Fixture", "error": str(index),
        })
    next_sequence = state.event_count + 1
    assert history.orchestrator_delivery_anchor(state.session_id, "new-control") == next_sequence
    assert history.orchestrator_delivery_anchor(
        state.session_id, "legacy-control", replay_existing=True,
    ) == 1
    progress = history.record_event(state, "assistant_progress", {
        "control_id": "new-control", "action_index": 0,
        "assistant_text": "I have received the instruction.",
    })
    history.record_event(state, "control_message_processed", {
        "control_id": "new-control", "session_id": state.session_id,
        "message": "Hello.", "decision": {},
    })
    assert history.orchestrator_delivery_anchor(state.session_id, "new-control") == progress.sequence
    assert history.orchestrator_delivery_anchor(
        state.session_id, "new-control", replay_existing=True,
    ) == progress.sequence
    cursor = OrchestratorEventCursor(history, {
        "session_id": state.session_id, "control_id": "new-control",
        "history_anchor_sequence": progress.sequence,
    })
    items = cursor.poll()
    assert items[0]["assistant_text"] == "I have received the instruction."
    assert cursor.active is True


def test_stream_cursor_rejects_invalid_receipt_anchor(make_config):
    runtime, state = setup_history(make_config)
    for bad in (False, True, 0, -1, "42", 1.2):
        with pytest.raises(ValueError, match="history_anchor_sequence"):
            OrchestratorEventCursor(runtime.history, {
                "session_id": state.session_id,
                "control_id": "only-this-control",
                "history_anchor_sequence": bad,
            })
