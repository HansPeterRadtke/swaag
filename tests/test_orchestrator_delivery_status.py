"""Replay-safe live-status note after an independently verified handoff.

These tests require neither inference nor an Android phone; they exercise the
real append-only HistoryStore and SWAAG worker state in an isolated sandbox.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from swaag.communication import CommunicationService
from swaag.orchestrator_delivery import current_started_worker_status, status_suffix
from swaag.runtime import AgentRuntime


def _event(kind, **payload):
    return SimpleNamespace(event_type=kind, payload=payload)


def _worker(worker_id, status):
    return SimpleNamespace(worker_id=worker_id, status=status)


def test_only_real_successful_same_turn_worker_start_receipt_qualifies():
    records = {"worker_abc": _worker("worker_abc", "completed")}
    source = [
        _event("tool_error", tool_name="orchestration_control", output={"started_worker_ids": ["worker_abc"]}),
        _event("agent_action_selected", action={"started_worker_ids": ["worker_abc"]}),
        _event("turn_started"),
        _event("tool_result", tool_name="orchestration_control", output={"started_worker_ids": ["worker_abc"]}),
    ]
    assert current_started_worker_status(source, records.__getitem__) == []
    good = [_event("tool_result", tool_name="orchestration_control", output={"started_worker_ids": ["worker_abc"]}), _event("turn_started")]
    assert current_started_worker_status(good, records.__getitem__) == [{"worker_id": "worker_abc", "status": "completed"}]
    assert current_started_worker_status(good[:-1], records.__getitem__) == []
    assert current_started_worker_status([_event("tool_result", tool_name="read_file", output={"started_worker_ids": ["worker_abc"]}), _event("turn_started")], records.__getitem__) == []


@pytest.mark.parametrize("status,tail", [
    ("queued", ""), ("working", ""), ("created", ""),
    ("completed", "the delegated worker has completed."),
    ("failed", "the delegated worker has failed."),
    ("canceled", "the delegated worker was canceled."),
    ("input_required", "the delegated worker is waiting for input."),
])
def test_status_suffix_never_promotes_queued_or_unverified_work(status, tail):
    updates = current_started_worker_status(
        [_event("tool_result", tool_name="orchestration_control", output={"started_worker_ids": ["worker_abc"]}), _event("turn_started")],
        lambda key: _worker(key, status),
    )
    text = status_suffix(updates)
    if not tail:
        assert updates == [] and text == ""
    else:
        assert text.endswith(tail)
        assert "external effects" not in text


def _service(make_config):
    from tests.test_agent_action_loop import FakeModelClient
    config = make_config(model__context_limit=32000, runtime__completion_evaluation_enabled=False)
    runtime = AgentRuntime(config, model_client=FakeModelClient([]))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    return service, service._orchestrator_state()


def _turn(service, state, *, started: list[str] | None = None, answer: str = "I queued a worker.", strict: bool = False):
    history = service.orchestrator_runtime.history
    history.record_event(state, "turn_started", {"turn_index": 1, "user_text": "Start the requested worker."})
    if started is not None:
        history.record_event(state, "tool_result", {"tool_name": "orchestration_control", "raw_input": {}, "validated_input": {}, "output": {"started_worker_ids": started}})
    if strict:
        history.record_event(state, "agent_action_selected", {"action_index": 1, "occurrence": 1, "action": {"response_constraints": {"exact_word_count": 4}, "silent_completion": False}})
    return history.record_event(state, "turn_finished", {"turn_index": 1, "assistant_text": answer, "tool_steps": 0, "budget_reports": []})


def test_durable_appendix_and_replay_preserve_original_reply_and_no_tool_reexecution(make_config):
    service, state = _service(make_config)
    try:
        worker = service.workers.create("Read-only verification")
        service.workers.store.transition(worker.worker_id, "completed", result="DONE")
        terminal = _turn(service, state, started=[worker.worker_id])
        initial = service._deliver_orchestrator_reply(state, "I queued a worker.", fresh=True)
        assert initial["answer"].startswith("I queued a worker.")
        assert initial["answer"].endswith("the delegated worker has completed.")
        events = service.orchestrator_runtime.history.read_history(state.session_id)
        audits = [item for item in events if item.event_type == "orchestrator_worker_status_delivery"]
        assert len(audits) == 1
        assert audits[0].payload["turn_finished_sequence"] == terminal.sequence
        assert audits[0].payload["observations"] == [{"worker_id": worker.worker_id, "status": "completed"}]
        recovered = service._deliver_orchestrator_reply(state, "I queued a worker.", turn_finished_sequence=terminal.sequence, fresh=False)
        assert recovered == initial
        assert sum(e.event_type == "orchestrator_worker_status_delivery" for e in service.orchestrator_runtime.history.read_history(state.session_id)) == 1
    finally:
        service.close()


def test_no_appendix_when_no_worker_started_or_exact_words_required(make_config):
    for strict, started in ((False, None), (True, ["worker_test"])):
        service, state = _service(make_config)
        try:
            _turn(service, state, strict=strict, started=started)
            result = service._deliver_orchestrator_reply(state, "Exactly four words only.", fresh=True)
            assert result["answer"] == "Exactly four words only."
            assert not [e for e in service.orchestrator_runtime.history.read_history(state.session_id) if e.event_type == "orchestrator_worker_status_delivery"]
        finally:
            service.close()


def test_completed_worker_from_previous_turn_cannot_leak_into_followup(make_config):
    service, state = _service(make_config)
    try:
        worker = service.workers.create("A task")
        service.workers.store.transition(worker.worker_id, "completed", result="DONE")
        _turn(service, state, started=[worker.worker_id])
        state = service.orchestrator_runtime.history.rebuild_from_history(state.session_id, write_projections=False)
        _turn(service, state, started=None, answer="Hello.")
        latest = service._deliver_orchestrator_reply(state, "Hello.", fresh=True)
        assert latest["answer"] == "Hello."
    finally:
        service.close()
