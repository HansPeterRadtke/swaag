"""Final receipt commit cannot knowingly publish a stale fast worker status.

The foreground session-control lock protects the user's authority receipt,
but workers can change in a separate SQLite store while that lock is acquired.
"""
from __future__ import annotations

from contextlib import contextmanager

from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from tests.test_agent_action_loop import FakeModelClient, _action
from tests.test_fast_foreground_reply import fast
from swaag.session_lock import session_control_lock as real_control_lock


def test_worker_transition_after_initial_revision_check_rejects_stale_first_speech(
    make_config, monkeypatch,
):
    cfg = make_config(
        model__context_limit=32000,
        runtime__completion_evaluation_enabled=False,
        communication__fast_initial_reply=True,
        communication__poll_seconds=0.01,
    )
    client = FakeModelClient([
        fast("answer", "The worker is created and has not started executing."),
        _action(message="The worker has changed state; I need its current status."),
    ])
    runtime = AgentRuntime(cfg, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    entered_final_fence = []
    try:
        worker = service.workers.create("Inspect the repository for defects")
        assert worker.status == "created"
        service._refresh_orchestrator_snapshot()
        assert service._fast_foreground_state is not None
        initial_revision = service._foreground_status_revision(
            service._foreground_task_revision()
        )

        @contextmanager
        def racing_lock(history, session_id):
            # The worker changes *after* the earlier pre-commit revision test
            # but before receipt consumption in the final control lock.
            with real_control_lock(history, session_id) as acquired:
                if not entered_final_fence:
                    entered_final_fence.append(True)
                    service.workers.store.transition(
                        worker.worker_id, "working", expected={"created"},
                    )
                yield acquired

        monkeypatch.setattr("swaag.communication.session_control_lock", racing_lock)
        response = service.orchestrator_message("Has the repository worker started?")
        assert entered_final_fence
        assert service.workers.store.get(worker.worker_id).status == "working"
        assert service._foreground_status_revision(
            service._foreground_task_revision()
        ) != initial_revision
        # Neither the old speech nor a committed fast direct turn may survive.
        assert response["answer"] == (
            "The worker has changed state; I need its current status."
        )
        assert [call["contract"] for call in client.requests] == [
            "fast_foreground_response", "agent_action",
        ]
        history = runtime.history.read_history(response["session_id"])
        assert not any(
            event.event_type == "turn_finished"
            and event.payload.get("assistant_text")
            == "The worker is created and has not started executing."
            for event in history
        )
    finally:
        service.close()
