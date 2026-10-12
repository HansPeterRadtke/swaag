from __future__ import annotations

from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from tests.test_preemption import _ImmediateClient


def test_orchestrator_snapshot_keeps_terminal_results_and_failures(make_config):
    config = make_config(model__context_limit=32000)
    client = _ImmediateClient("The completed result is available.")
    runtime = AgentRuntime(config, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        completed = service.task_api.execute("create", {"objective":"verify package", "start":False})["worker"]
        failed = service.task_api.execute("create", {"objective":"check restored backup", "start":False})["worker"]
        service.workers.store.transition(completed["worker_id"], "completed", result="VERIFIER_EVIDENCE_579 passed, exit code 0")
        service.workers.store.transition(failed["worker_id"], "failed", error="RESTORE_EVIDENCE_863 archive mismatch")
        service.orchestrator_message("Did the package verifier and restore check finish?")
        prompt = client.requests[0]["prompt"]
        assert "VERIFIER_EVIDENCE_579" in prompt
        assert "RESTORE_EVIDENCE_863" in prompt
        assert completed["worker_id"] in prompt
        assert failed["worker_id"] in prompt
    finally:
        service.close()


def test_orchestrator_snapshot_preserves_history_progress(make_config):
    config = make_config(model__context_limit=32000)
    client = _ImmediateClient("Verification is half complete.")
    runtime = AgentRuntime(config, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        worker = service.workers.create("Verify source and installation")
        state = runtime.history.rebuild_from_history(worker.session_id)
        runtime.history.record_event(state, "agent_progress", {
            "steps": [
                {"id":"source","label":"SOURCE_VERIFIED_241","state":"completed","weight":1},
                {"id":"install","label":"INSTALL_PENDING_763","state":"pending","weight":1},
            ],
            "reason":"Source verified; installation still pending",
            "estimated_remaining_seconds":None,
            "estimate_conditions":"",
        })
        service.orchestrator_message("How far along is verification?")
        prompt = client.requests[0]["prompt"]
        assert "SOURCE_VERIFIED_241" in prompt and "INSTALL_PENDING_763" in prompt
        assert '"overall_percent":50.0' in prompt.replace(" ", "").replace("\n", "")
    finally:
        service.close()


def test_snapshot_is_collected_after_preemption_wait(make_config, monkeypatch):
    config = make_config(model__context_limit=32000)
    client = _ImmediateClient("Verification completed while the request waited.")
    runtime = AgentRuntime(config, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        worker = service.workers.create("Verify while inference is occupied")
        service.workers.store.transition(worker.worker_id, "working")
        def finish_during_preemption(*args, **kwargs):
            service.workers.store.transition(worker.worker_id, "completed", result="COMPLETED_DURING_WAIT_937")
            return None
        monkeypatch.setattr(service, "_preempt_runtime_call", finish_during_preemption)
        service.orchestrator_message("Has verification finished?")
        assert "COMPLETED_DURING_WAIT_937" in client.requests[0]["prompt"]
    finally:
        service.close()
