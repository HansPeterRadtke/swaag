"""A worker transition can commit out of timestamp order under concurrency."""
from __future__ import annotations

from tests.test_preemption import _ImmediateClient
from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime


def test_older_worker_update_under_global_max_invalidates_fast_world(make_config, monkeypatch):
    runtime = AgentRuntime(make_config(model__context_limit=32000),
                           model_client=_ImmediateClient("unused"))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        # Creation and status timestamps can be generated before acquiring the
        # worker DB write lock. A concurrent later timestamp can commit first.
        monkeypatch.setattr("swaag.workers.utc_now_iso",
                            lambda: "2026-10-22T01:00:00+00:00")
        first_worker = service.workers.create("Inspect source files")
        monkeypatch.setattr("swaag.workers.utc_now_iso",
                            lambda: "2026-10-22T01:05:00+00:00")
        second_worker = service.workers.create("Review tests")

        prior = service._foreground_task_revision()
        assert prior[0] == 2
        assert prior[1] == "2026-10-22T01:05:00+00:00"
        assert len(prior) >= 8

        # This timestamp is NEWER for worker one, but OLDER than the other
        # worker's existing timestamp: COUNT and MAX(updated_at) do not change.
        monkeypatch.setattr("swaag.workers.utc_now_iso",
                            lambda: "2026-10-22T01:03:00+00:00")
        service.workers.store.transition(
            first_worker.worker_id, "working", expected={"created"}
        )
        after = service._foreground_task_revision()
        assert after[:5] == prior[:5]
        assert service.workers.store.get(first_worker.worker_id).status == "working"
        assert service.workers.store.get(second_worker.worker_id).status == "created"

        # A fast answer already generated against the old worker status MUST
        # not be admitted after the durable status transition.
        assert service._foreground_status_revision(after) != service._foreground_status_revision(prior)
    finally:
        service.close()


def test_fast_speech_rejects_real_worker_transition_under_unchanged_global_max(
    make_config, monkeypatch,
):
    """Do not rely on the periodic background snapshot refreshing first."""
    from tests.test_agent_action_loop import FakeModelClient, _action
    from tests.test_fast_foreground_reply import fast

    cfg = make_config(
        model__context_limit=32000,
        runtime__completion_evaluation_enabled=False,
        communication__fast_initial_reply=True,
        communication__poll_seconds=0.01,
    )
    holder = {}

    def late_transition(request):
        service = holder["service"]
        monkeypatch.setattr(
            "swaag.workers.utc_now_iso", lambda: "2026-10-22T01:03:00+00:00",
        )
        service.workers.store.transition(
            holder["first"].worker_id, "working", expected={"created"}
        )
        # The quick reply was prepared from the old world, with both workers
        # created. Background refresh has not run in this deterministic test.
        return fast("answer", "Both workers are still only created.")

    client = FakeModelClient([
        late_transition,
        _action(message="Worker state changed, so a new status inspection is required."),
    ])
    runtime = AgentRuntime(cfg, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    holder["service"] = service
    try:
        monkeypatch.setattr(
            "swaag.workers.utc_now_iso", lambda: "2026-10-22T01:00:00+00:00",
        )
        holder["first"] = service.workers.create("Inspect the rendering source")
        monkeypatch.setattr(
            "swaag.workers.utc_now_iso", lambda: "2026-10-22T01:05:00+00:00",
        )
        service.workers.create("Review test evidence")
        service._refresh_orchestrator_snapshot()
        initial_revision = service._foreground_status_revision(
            service._foreground_task_revision()
        )
        outcome = service.orchestrator_message("Which workers are working?")
        assert outcome["answer"] == (
            "Worker state changed, so a new status inspection is required."
        )
        assert [call["contract"] for call in client.requests] == [
            "fast_foreground_response", "agent_action",
        ]
        assert service._foreground_status_revision(
            service._foreground_task_revision()
        ) != initial_revision
    finally:
        service.close()


def test_repeated_same_status_with_new_run_count_invalidates_worker_revision(
    make_config, monkeypatch,
):
    runtime = AgentRuntime(
        make_config(model__context_limit=32000),
        model_client=_ImmediateClient("unused"),
    )
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        monkeypatch.setattr(
            "swaag.workers.utc_now_iso", lambda: "2026-10-22T01:02:00+00:00"
        )
        first_worker = service.workers.create("Execute project step")
        before = service._foreground_task_revision()
        # Identical time and status, but a durable new execution run happened.
        service.workers.store.transition(
            first_worker.worker_id, "created",
            expected={"created"}, increment_run_count=True,
        )
        after = service._foreground_task_revision()
        assert after[:5] == before[:5]
        assert service.workers.store.get(first_worker.worker_id).run_count == 1
        assert service._foreground_status_revision(after) != service._foreground_status_revision(before)
    finally:
        service.close()
