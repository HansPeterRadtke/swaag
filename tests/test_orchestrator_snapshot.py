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


def test_direct_reply_is_regenerated_after_worker_completion(make_config, monkeypatch):
    config = make_config(model__context_limit=32000)
    runtime = AgentRuntime(config, model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    snapshots = []
    try:
        worker = service.workers.create('Verification changes during generation')
        service.workers.store.transition(worker.worker_id, 'working')
        def generate(**kwargs):
            snapshot = kwargs['runtime_snapshot']
            snapshots.append(snapshot)
            if len(snapshots) == 1:
                service.workers.store.transition(worker.worker_id, 'completed', result='FRESH_COMPLETION_529')
                return {'route': 'respond', 'answer': 'Verification is still running.', 'reason': 'snapshot'}
            assert any(w.get('result') == 'FRESH_COMPLETION_529' for w in snapshot['terminal_workers'])
            return {'route': 'respond', 'answer': 'Verification has completed.', 'reason': 'fresh snapshot'}
        monkeypatch.setattr(runtime, 'generate_orchestrator_interaction', generate)
        result = service.orchestrator_message('Has verification finished?')
        assert result['answer'] == 'Verification has completed.'
        assert len(snapshots) == 2
        state = service._load_orchestrator_state()
        assert not any(m.role == 'assistant' and m.content == 'Verification is still running.' for m in state.messages)
    finally:
        service.close()


def test_continuously_changing_snapshot_does_not_publish_a_stale_reply(make_config, monkeypatch):
    import pytest
    config = make_config(model__context_limit=32000)
    runtime = AgentRuntime(config, model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    calls = []
    try:
        def generate(**kwargs):
            calls.append(kwargs)
            service.workers.create('New work observed during response ' + str(len(calls)))
            return {'route': 'respond', 'answer': 'There is no new work.', 'reason': 'old snapshot'}
        monkeypatch.setattr(runtime, 'generate_orchestrator_interaction', generate)
        with pytest.raises(RuntimeError, match='Task state changed during both response attempts'):
            service.orchestrator_message('What is running?')
        assert len(calls) == 2
        state = service._load_orchestrator_state()
        assert not any(m.role == 'assistant' for m in state.messages)
        rejected = [e for e in runtime.history.read_history(state.session_id) if e.event_type == 'orchestrator_direct_reply_rejected']
        assert len(rejected) == 2
    finally:
        service.close()


def test_snapshot_refresh_observes_progress_without_worker_transition(make_config, monkeypatch):
    config = make_config(model__context_limit=32000)
    runtime = AgentRuntime(config, model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    calls = []
    try:
        worker = service.workers.create('Report a material progress change')
        def generate(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                state = runtime.history.rebuild_from_history(worker.session_id)
                runtime.history.record_event(state, 'agent_progress', {
                    'steps': [{'id':'done','label':'verified','state':'completed','weight':1},
                              {'id':'pending','label':'remaining','state':'pending','weight':1}],
                    'reason':'Verified one of two equal steps',
                    'estimated_remaining_seconds':None, 'estimate_conditions':'',
                })
                return {'route':'respond','answer':'Progress is unknown.','reason':'old snapshot'}
            progress = kwargs['runtime_snapshot']['worker_question_inventory']['workers'][0]['progress']
            assert progress['overall_percent'] == 50
            return {'route':'respond','answer':'Verification is half complete.','reason':'new progress'}
        monkeypatch.setattr(runtime, 'generate_orchestrator_interaction', generate)
        assert service.orchestrator_message('How far along?')['answer'] == 'Verification is half complete.'
        assert len(calls) == 2
    finally:
        service.close()


def test_snapshot_ignores_bookkeeping_without_task_fact_changes(make_config, monkeypatch):
    config = make_config(model__context_limit=32000)
    runtime = AgentRuntime(config, model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    calls = []
    try:
        worker = service.workers.create('Stable queued work')
        def generate(**kwargs):
            calls.append(kwargs)
            state = runtime.history.rebuild_from_history(worker.session_id)
            runtime.history.record_event(state, 'state_rebuilt', {'session_id':state.session_id,'event_count':state.event_count})
            return {'route':'respond','answer':'The work is queued.','reason':'snapshot'}
        monkeypatch.setattr(runtime, 'generate_orchestrator_interaction', generate)
        assert service.orchestrator_message('What is running?')['answer'] == 'The work is queued.'
        assert len(calls) == 1
    finally:
        service.close()


def test_snapshot_includes_orchestrator_work_and_unprocessed_user_controls(make_config):
    runtime = AgentRuntime(make_config(), model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        state = service._load_orchestrator_state()
        original = 'Verify the exact archive Prüfbericht — Öl.zip before uploading.'
        correction = 'Do not upload anything until I review the verification.'
        runtime.history.set_active_run(state.session_id, run_id='run-current', user_text=original)
        receipt = runtime.history.enqueue_control_message(state.session_id, correction, source='orchestrator_user')
        snapshot = service._orchestrator_runtime_snapshot()
        assert snapshot['active_workers'] == []
        assert snapshot['orchestrator']['active_turn']['user_text'] == original
        assert snapshot['orchestrator']['pending_user_messages'][0]['message'] == correction
        assert snapshot['orchestrator']['pending_user_messages'][0]['control_id'] == receipt['control_id']
        before = service._orchestrator_task_snapshot_fingerprint(snapshot)
        runtime.history.update_active_run(state.session_id, run_id='run-current')
        assert service._orchestrator_task_snapshot_fingerprint(service._orchestrator_runtime_snapshot()) == before
        runtime.history.update_active_run(
            state.session_id, run_id='run-current', phase='inference',
            substate='awaiting_result', detail='waiting for action model stream',
            operation_kind='action',
        )
        during_stream = service._orchestrator_runtime_snapshot()
        assert during_stream['orchestrator']['active_turn']['substate'] == 'awaiting_result'
        assert service._orchestrator_task_snapshot_fingerprint(during_stream) == before
        runtime.history.clear_active_run(state.session_id, run_id='run-current')
        assert service._orchestrator_task_snapshot_fingerprint(service._orchestrator_runtime_snapshot()) != before
    finally:
        service.close()
