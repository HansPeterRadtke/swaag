"""Reject fast spoken status drafts if significant worker-world facts change."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from swaag.fast_worker_state import build_fast_worker_state, meaningful_worker_state_revision
from tests.test_fast_worker_state import snapshot


def world_with_question(message: str):
    source = snapshot(with_progress=False)
    source["worker_question_inventory"] = {
        "questions": [
            {"worker_id": "worker-1", "criticality": "normal", "question": message}
        ],
        "complete": True,
        "scope": "all workers",
    }
    return build_fast_worker_state(source)


def test_question_replacement_at_constant_count_invalidates_fast_snapshot():
    before = world_with_question("Which warehouse texture should be installed?")
    after = world_with_question("The build failed. Which database version is available?")
    assert before["open_worker_question_count"] == after["open_worker_question_count"] == 1
    assert before["worker_question_summaries"] != after["worker_question_summaries"]
    assert meaningful_worker_state_revision(before) != meaningful_worker_state_revision(after)


def test_question_inventory_becoming_incomplete_invalidates_fast_snapshot():
    before = world_with_question("Choose a render resolution.")
    after = deepcopy(before)
    after["worker_question_inventory_complete"] = False
    assert meaningful_worker_state_revision(before) != meaningful_worker_state_revision(after)


def test_worker_model_route_change_invalidates_fast_snapshot():
    before = world_with_question("Choose a render resolution.")
    after = deepcopy(before)
    after["workers"][0]["model_route"] = "different"
    assert meaningful_worker_state_revision(before) != meaningful_worker_state_revision(after)


def test_fast_model_cannot_publish_answer_after_question_replaced(make_config):
    from tests.test_agent_action_loop import FakeModelClient, _action
    from tests.test_fast_foreground_reply import fast
    from swaag.communication import CommunicationService
    from swaag.runtime import AgentRuntime

    cfg = make_config(
        model__context_limit=32000,
        runtime__completion_evaluation_enabled=False,
        communication__fast_initial_reply=True,
        communication__poll_seconds=0.01,
    )
    holder = {}

    def changed_question(request):
        service = holder["service"]
        with service._foreground_snapshot_lock:
            updated = deepcopy(service._fast_foreground_state)
            updated["worker_question_summaries"][0]["question"] = (
                "The build failed. Which database version is available?"
            )
            service._fast_foreground_state = updated
        return fast("answer", "The warehouse worker still needs the texture choice.")

    client = FakeModelClient([
        changed_question,
        _action(message="The pending question changed; checking current state."),
    ])
    runtime = AgentRuntime(cfg, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    holder["service"] = service
    try:
        live = snapshot(with_progress=False)
        live["captured_at"] = datetime.now(timezone.utc).isoformat()
        live["worker_question_inventory"] = {
            "questions": [
                {"worker_id": "worker-1", "criticality": "normal",
                 "question": "Which warehouse texture should be installed?"}
            ],
            "complete": True,
            "scope": "all workers", "workers": [], "pending_revisions": [],
        }
        service._build_orchestrator_runtime_snapshot = lambda: live
        service._foreground_task_revision = lambda: (1,)
        service._refresh_orchestrator_snapshot()
        answer = service.orchestrator_message("What is the current worker question?")
        assert answer["answer"] == "The pending question changed; checking current state."
        assert [r["contract"] for r in client.requests] == [
            "fast_foreground_response", "agent_action"
        ]
    finally:
        service.close()


def test_explicit_progress_value_change_invalidates_status_even_if_assessed_at_unchanged():
    before = build_fast_worker_state(snapshot(continuous=True))
    after = deepcopy(before)
    after["workers"][0]["progress"]["cycle_percent"] = 50
    assert meaningful_worker_state_revision(before) != meaningful_worker_state_revision(after)


def test_polling_clock_and_heartbeat_do_not_reject_unchanged_status():
    before = world_with_question("Choose a render resolution.")
    after = deepcopy(before)
    after["captured_at"] = datetime.now(timezone.utc).isoformat()
    after["workers"][0]["supervisor_heartbeat_age_seconds_at_capture"] = 50
    after["workers"][0]["supervisor_phase"] = "inference"
    after["workers"][0]["supervisor_substate"] = "queued"
    assert meaningful_worker_state_revision(before) == meaningful_worker_state_revision(after)


def test_reported_next_action_change_requires_fresh_status():
    before = world_with_question("Choose a render resolution.")
    after = deepcopy(before)
    after["workers"][0]["reported_next_action"] = "Test a different renderer"
    assert meaningful_worker_state_revision(before) != meaningful_worker_state_revision(after)


def test_durable_semantic_worker_change_without_cache_refresh_rejects_fast_answer(make_config):
    """The background snapshot loop may not have run before first speech ends."""
    from tests.test_agent_action_loop import FakeModelClient, _action
    from tests.test_fast_foreground_reply import fast
    from swaag.communication import CommunicationService
    from swaag.runtime import AgentRuntime

    cfg = make_config(model__context_limit=32000,
                      runtime__completion_evaluation_enabled=False,
                      communication__fast_initial_reply=True,
                      communication__poll_seconds=0.01)
    holder = {}
    stable_prefix = (1, "unchanged", 1, "unchanged", 1)
    def changed_semantic_cursor(request):
        holder["service"]._foreground_task_revision = lambda: (
            *stable_prefix, (("worker-history", 12),), (), (("worker", "agent_question", 12),)
        )
        return fast("answer", "The worker has no outstanding questions.")
    client = FakeModelClient([changed_semantic_cursor,
                              _action(message="New worker evidence requires a fresh check.")])
    runtime = AgentRuntime(cfg, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    holder["service"] = service
    try:
        live = snapshot(with_progress=False)
        live["captured_at"] = datetime.now(timezone.utc).isoformat()
        service._build_orchestrator_runtime_snapshot = lambda: live
        service._foreground_task_revision = lambda: (
            *stable_prefix, (("worker-history", 11),), (), (("worker", "agent_question", 11),)
        )
        service._refresh_orchestrator_snapshot()
        response = service.orchestrator_message("Does the worker have any pending questions?")
        assert response["answer"] == "New worker evidence requires a fresh check."
        assert [item["contract"] for item in client.requests] == [
            "fast_foreground_response", "agent_action"
        ]
    finally:
        service.close()


def test_indexed_semantic_cursor_changes_only_for_meaningful_history(make_config):
    """Use actual durable worker events, not just a monkeypatched revision tuple."""
    from tests.test_preemption import _ImmediateClient
    from swaag.communication import CommunicationService
    from swaag.runtime import AgentRuntime

    runtime = AgentRuntime(make_config(model__context_limit=32000),
                           model_client=_ImmediateClient("unused"))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        worker = service.workers.create("Audit warehouse rendering")
        owner = service.workers.runtime
        worker_state = owner.create_or_load_session(session_id=worker.session_id)
        first = service._foreground_task_revision()
        assert len(first) == 9
        assert len(first[-2]) == 0  # No semantic worker-history changes.
        assert len(first[-1]) == 1  # Exact per-worker lifecycle identity.
        owner.history.record_event(worker_state, "model_tokenize_result", {
            "text_hash": "test-message", "tokens": 50, "exact": True,
        })
        telemetry = service._foreground_task_revision()
        assert telemetry[5] != first[5]
        assert service._foreground_status_revision(telemetry) == service._foreground_status_revision(first)
        owner.history.record_event(worker_state, "agent_progress", {
            "steps": [{"id": "one", "label": "Audit", "state": "in_progress", "weight": 1}],
            "reason": "Auditing current materials",
            "estimated_remaining_seconds": None,
            "estimate_conditions": "",
        })
        meaningful = service._foreground_task_revision()
        assert meaningful[-2] != telemetry[-2]
        assert any(entry[1] == "agent_progress" for entry in meaningful[-2])
        assert service._foreground_status_revision(meaningful) != service._foreground_status_revision(telemetry)
    finally:
        service.close()
