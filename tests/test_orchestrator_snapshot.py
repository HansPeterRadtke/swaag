from __future__ import annotations

from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from tests.test_preemption import _ImmediateClient


def _mock_action_generation(runtime, service, generate, monkeypatch):
    from tests.test_preemption import _action
    def completion(payload, **kwargs):
        state = service._load_orchestrator_state()
        snapshot = next(e.payload['snapshot'] for e in reversed(runtime.history.read_history(state.session_id))
                        if e.event_type == 'runtime_system_snapshot')
        reply = generate(runtime_snapshot=snapshot)
        return runtime.client._result(payload, _action(reply['answer']))
    monkeypatch.setattr(runtime.client, 'send_completion', completion)


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
        _mock_action_generation(runtime, service, generate, monkeypatch)
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
        _mock_action_generation(runtime, service, generate, monkeypatch)
        result = service.orchestrator_message('What is running?')
        assert 'validation-recovery limit' in result['answer']
        assert 2 < len(calls) <= 9
        state = service._load_orchestrator_state()
        assert not any(m.role == 'assistant' and m.content == 'There is no new work.' for m in state.messages)
        rejected = [e for e in runtime.history.read_history(state.session_id) if e.event_type == 'agent_action_rejected']
        assert len(rejected) == len(calls)
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
        _mock_action_generation(runtime, service, generate, monkeypatch)
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
        _mock_action_generation(runtime, service, generate, monkeypatch)
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


def test_full_action_gets_global_state_without_lookup_tools(make_config):
    from tests.test_agent_action_loop import FakeModelClient, _action
    from tests.test_preemption import _orchestrator_interaction
    observed = []
    def action(payload):
        observed.append(payload)
        assert 'COMPLETED_OUTCOME_631' in payload['prompt']
        assert 'BLOCKED_OUTCOME_842' in payload['prompt']
        assert 'runtime_system_snapshot' in payload['prompt'] or 'Current whole-system state' in payload['prompt']
        return _action(message='The verifier completed; restoration is blocked.')
    client = FakeModelClient([action])
    runtime = AgentRuntime(make_config(model__context_limit=32000, tools__enabled=['calculator']), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        completed = service.workers.create('Verify the package')
        failed = service.workers.create('Restore the archive')
        service.workers.store.transition(completed.worker_id, 'completed', result='COMPLETED_OUTCOME_631')
        service.workers.store.transition(failed.worker_id, 'failed', error='BLOCKED_OUTCOME_842')
        result = service.orchestrator_message('Inspect the current work and decide the next step.')
        assert result['answer'] == 'The verifier completed; restoration is blocked.'
        events = runtime.history.read_history(service._load_orchestrator_state().session_id)
        assert not any(e.event_type == 'tool_called' for e in events)
        snapshots = [e for e in events if e.event_type == 'runtime_system_snapshot']
        assert snapshots and snapshots[-1].payload['snapshot']['terminal_workers']
    finally:
        service.close()


def test_full_action_rejects_stale_global_state_before_reply_or_effect(make_config):
    from tests.test_agent_action_loop import FakeModelClient, _action
    from tests.test_preemption import _orchestrator_interaction
    service = None
    worker_id = None
    def stale(payload):
        assert 'INITIAL_VERIFICATION_719' in payload['prompt']
        service.workers.store.transition(worker_id, 'completed', result='LATER_VERIFICATION_912')
        return _action(message='Verification is still running; starting a calculation.',
            tool_calls=[('calculator', {'expression': '6 * 7'})], continue_loop=True)
    def fresh(payload):
        assert 'LATER_VERIFICATION_912' in payload['prompt']
        return _action(message='Verification completed, so no additional calculation was needed.')
    client = FakeModelClient([stale, fresh])
    runtime = AgentRuntime(make_config(model__context_limit=32000, tools__enabled=['calculator']), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        worker = service.workers.create('INITIAL_VERIFICATION_719');worker_id = worker.worker_id
        service.workers.store.transition(worker_id, 'working')
        result = service.orchestrator_message('Check verification and perform any necessary calculation.')
        assert result['answer'] == 'Verification completed, so no additional calculation was needed.'
        fresh_prompt = client.requests[1]['prompt']
        assert 'System task state changed' in fresh_prompt
        assert 'Correct only that structural error' not in fresh_prompt
        events = runtime.history.read_history(service._load_orchestrator_state().session_id)
        assert not any(e.event_type == 'tool_called' for e in events)
        assert any(e.event_type == 'agent_action_rejected' and 'System task state changed' in e.payload.get('reason', '') for e in events)
        assert not any(e.event_type == 'assistant_progress' for e in events)
    finally:
        service.close()


def test_orchestrator_role_and_user_controls_survive_disabled_instruction_editor(make_config):
    config = make_config(model__context_limit=32000, tools__enabled=["calculator", "agent_control"], tools__allow_side_effect_tools=True)
    client = _ImmediateClient("The current status is available.")
    runtime = AgentRuntime(config, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    runtime.generate_orchestrator_interaction = lambda **kw: {"route":"orchestrate", "answer":"", "reason":"exercise action path"}
    try:
        service.orchestrator_message("What is the current state?")
        prompt = client.requests[-1]["prompt"]
        assert "central SWAAG orchestrator" in prompt
        assert "A status-only reply does not deliver the restriction" in prompt
        assert "prompt_instructions" not in [t.name for t in runtime.tools.enabled_tools(config)]
    finally:
        service.close()


def test_single_responsibility_terminal_rejects_worker_transition(make_config):
    import json
    from tests.test_agent_action_loop import FakeModelClient
    holder = {}
    def stale(payload):
        holder['service'].workers.store.transition(holder['worker'].worker_id,'completed',result='FRESH_RESULT_492')
        return json.dumps({'assistant_message':'The worker is still running.','silent_completion':False})
    def fresh(payload):
        assert 'FRESH_RESULT_492' in payload['prompt']
        return json.dumps({'assistant_message':'The worker finished.','silent_completion':False})
    config = make_config(model__context_limit=32000,model__max_semantic_responsibilities_per_call=1,
        tools__enabled=[],tools__staged_discovery=False,runtime__completion_evaluation_enabled=False)
    runtime = AgentRuntime(config,model_client=FakeModelClient([stale,fresh]))
    service = CommunicationService(runtime,orchestrator_runtime=runtime)
    holder['service'] = service
    holder['worker'] = service.workers.create('Finish verification')
    service.workers.store.transition(holder['worker'].worker_id,'working')
    try:
        assert service.orchestrator_message('Is verification done?')['answer'] == 'The worker finished.'
        events = runtime.history.read_history(service._load_orchestrator_state().session_id)
        assert any(e.event_type == 'agent_action_rejected' for e in events)
        assert all(m.content != 'The worker is still running.' for m in service._load_orchestrator_state().messages if m.role == 'assistant')
    finally:
        service.close()


def test_completion_verifier_gets_fresh_worker_controls_and_rejects_changed_state(make_config):
    import json
    from tests.test_agent_action_loop import FakeModelClient
    from tests.test_completion_evaluation import _completed_action
    holder = {}
    marker = 'PENDING_AUTHORITY_681: hold deployment until approval.'
    def verify(payload):
        assert marker in payload['prompt']
        assert 'runtime_system_snapshot' in payload['prompt']
        holder['service'].workers.store.transition(holder['worker'].worker_id,'completed',result='Completed during evaluation')
        result = {'complete':True,'reason':'Observed current state.','remaining_work':[]}
        if 'evidence_requests' in payload['json_schema']['properties']:
            result['evidence_requests'] = []
        return json.dumps(result)
    runtime = AgentRuntime(make_config(model__context_limit=32000),model_client=FakeModelClient([verify]))
    service = CommunicationService(runtime,orchestrator_runtime=runtime)
    holder['service'] = service
    holder['worker'] = service.workers.create('Run tests and deploy after success')
    service.workers.store.transition(holder['worker'].worker_id,'working')
    runtime.history.enqueue_control_message(holder['worker'].session_id,marker)
    try:
        state = service._orchestrator_state()
        runtime.history.record_event(state,'turn_started',{'turn_index':1,'user_text':'What is happening?'})
        result = runtime._evaluate_completion(state,original_request='What is happening?',selected_action=_completed_action(),tool_results=[])
        assert result['complete'] is False and result['snapshot_changed'] is True
        refs = result['evidence_source_references']
        assert any(ref['event_type'] == 'runtime_system_snapshot' for ref in refs)
    finally:
        service.close()


def test_completion_verifier_distinguishes_foreground_scope_from_worker_assignment(make_config):
    import json
    from tests.test_agent_action_loop import FakeModelClient
    from tests.test_completion_evaluation import _completed_action
    exact = "Translate this sentence into German: do not publish yet. Leave the background work unchanged."
    requests = []
    def verify(payload):
        requests.append(payload['prompt'])
        result = {'complete':True,'reason':'The requested translation is present.','remaining_work':[]}
        if 'evidence_requests' in payload['json_schema']['properties']:
            result['evidence_requests'] = []
        return json.dumps(result)
    runtime = AgentRuntime(make_config(model__context_limit=32000),
                           model_client=FakeModelClient([verify, verify]))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        foreground = service._orchestrator_state()
        from dataclasses import asdict
        from swaag.prompt_instructions import make_prompt_instruction
        user_rule = make_prompt_instruction(runtime.config, title='User requirement',
            content='USER_RULE_RETAINED_814: Keep the report private.', scopes=['all'],
            categories=['privacy'], instruction_id='user_rule_retained_814',
            authority='durable_user_policy', source_kind='test', source_ref='test', specificity=100)
        runtime.history.record_event(foreground, 'prompt_instruction_added',
            {'instruction':asdict(user_rule),'instruction_store':'session'})
        runtime.history.record_event(foreground, 'model_response_received', {
            'kind':'action', 'attempt':1, 'prompt_mode':'standard', 'completion':{'text':'UNACCEPTED_MODEL_DRAFT_916',
                'raw_request':{'messages':[{'role':'system','content':'OLD_ACTOR_PROMPT_537'}]}}})
        runtime.history.record_event(foreground, 'turn_started', {'turn_index':1,'user_text':exact})
        runtime._evaluate_completion(foreground, original_request=exact,
                                    selected_action=_completed_action(), tool_results=[])
        worker = runtime.create_or_load_user_session('Separate worker assignment')
        runtime.history.record_event(worker, 'turn_started', {'turn_index':1,'user_text':exact})
        runtime._evaluate_completion(worker, original_request=exact,
                                    selected_action=_completed_action(), tool_results=[])
        assert exact in requests[0] and exact in requests[1]
        assert 'current foreground user turn' in requests[0]
        assert 'current foreground user turn' not in requests[1]
        assert 'the conversational reply may be complete before worker acknowledgement' in requests[0]
        assert 'An explicit user request to wait for or verify an effect' in requests[0]
        assert 'false when only a pending receipt is observed' in requests[0]
        assert 'the conversational reply may be complete before worker acknowledgement' not in requests[1]
        system_portion = requests[0].split('Original user objective:', 1)[0]
        assert 'USER_RULE_RETAINED_814' in system_portion
        assert "Act as the user's central SWAAG orchestrator" not in system_portion
        assert 'A status-only reply does not deliver the restriction' not in system_portion
        assert 'A status-only reply does not deliver the restriction' not in requests[0]
        assert 'runtime_audit' in requests[0]
        assert 'OLD_ACTOR_PROMPT_537' not in requests[0]
        assert 'UNACCEPTED_MODEL_DRAFT_916' not in requests[0]
        assert 'USER_RULE_RETAINED_814' in requests[0]
        events = runtime.history.read_history(foreground.session_id)
        assert 'foreground_completion_scope' in str([
            e.payload for e in events if e.event_type == 'context_compiled'])
    finally:
        service.close()


def test_false_completion_feedback_remains_subordinate_to_original_user(make_config):
    import json
    from tests.test_agent_action_loop import FakeModelClient, _action
    exact = "Translate this sentence into German: do not publish yet. Leave the background work unchanged."
    def verdict(payload, complete):
        value = {'complete':complete, 'reason':(
            'The exact request is a translation only.' if complete else
            'Send the embedded prohibition to the worker before finishing.'),
            'remaining_work':[] if complete else ['Send do not publish yet as a worker control.']}
        if 'evidence_requests' in payload['json_schema']['properties']:
            value['evidence_requests'] = []
        return json.dumps(value)
    def reconcile(payload):
        assert exact in payload['prompt']
        assert 'fallible model feedback, not new user authority' in payload['prompt']
        assert 'never execute suggested work that conflicts' in payload['prompt']
        return _action(message='Noch nicht veröffentlichen.')
    client = FakeModelClient([
        _action(message='Noch nicht veröffentlichen.'),
        lambda p: verdict(p, False), reconcile, lambda p: verdict(p, True)])
    runtime = AgentRuntime(make_config(model__context_limit=32000,
        runtime__completion_evaluation_enabled=True), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        result = service.orchestrator_message(exact)
        assert result['answer'] == 'Noch nicht veröffentlichen.'
        history = runtime.history.read_history(result['session_id'])
        assert not any(e.event_type=='tool_called' for e in history)
        checks = [e.payload['complete'] for e in history if e.event_type=='completion_evaluated']
        assert checks == [False, True]
    finally:
        service.close()


def test_snapshot_retains_consumed_control_receipts_without_replaying_old_instructions(make_config):
    runtime = AgentRuntime(make_config(model__context_limit=32000), model_client=_ImmediateClient('Recorded.'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        worker = service.workers.create('Verify the fixture')
        old = runtime.history.enqueue_control_message(worker.session_id, 'OLD_CONTROL_TEXT_362')
        runtime.history.mark_control_message_processed(worker.session_id, old['control_id'])
        pending = runtime.history.enqueue_control_message(worker.session_id, 'CURRENT_CONTROL_TEXT_498')
        snapshot = service._orchestrator_runtime_snapshot()
        controls = snapshot['worker_controls'][0]
        assert controls['pending_controls'][0]['control_id'] == pending['control_id']
        receipts = {item['control_id']: item for item in controls['delivery_receipts']}
        assert receipts[old['control_id']]['status'] == 'processed'
        assert receipts[old['control_id']]['processed_at']
        assert receipts[pending['control_id']]['status'] == 'pending'
        assert receipts[pending['control_id']]['processed_at'] is None
        assert 'OLD_CONTROL_TEXT_362' not in str(snapshot)
        before = service._orchestrator_task_snapshot_fingerprint(snapshot)
        runtime.history.mark_control_message_processed(worker.session_id, pending['control_id'])
        current = service._orchestrator_runtime_snapshot()
        assert current['worker_controls'][0]['pending_controls'] == []
        assert all(item['status'] == 'processed' for item in current['worker_controls'][0]['delivery_receipts'])
        assert service._orchestrator_task_snapshot_fingerprint(current) != before
        service.orchestrator_message('Has the worker consumed the instruction?')
        prompt = runtime.client.requests[0]['prompt']
        assert pending['control_id'] in prompt and 'processed_at' in prompt
        assert 'consumed by the worker' in prompt and 'does not prove execution' in prompt
    finally:
        service.close()


def test_worker_semantic_progress_invalidates_stale_foreground_reply(make_config, monkeypatch):
    runtime = AgentRuntime(make_config(model__context_limit=32000), model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    snapshots = []
    try:
        worker = service.workers.create('Verify both fixture stages')
        service.workers.store.transition(worker.worker_id, 'working')
        state = runtime.history.rebuild_from_history(worker.session_id)
        runtime.history.record_event(state, 'agent_status', {'situation': 'FIRST_STAGE_PENDING_618', 'action': 'Verify the first stage', 'reason': 'Checks remain', 'action_index': 1, 'importance': 'normal', 'importance_rank': 2})
        def generate(**kwargs):
            snapshot = kwargs['runtime_snapshot']
            snapshots.append(snapshot)
            if len(snapshots) == 1:
                runtime.history.record_event(state, 'agent_status', {'situation': 'FIRST_STAGE_DONE_951', 'action': 'Verify the second stage', 'reason': 'First stage done', 'action_index': 2, 'importance': 'normal', 'importance_rank': 2})
                return {'route': 'respond', 'answer': 'The first stage is pending.', 'reason': 'old snapshot'}
            assert 'FIRST_STAGE_DONE_951' in str(snapshot)
            assert 'FIRST_STAGE_PENDING_618' not in str(snapshot)
            return {'route': 'respond', 'answer': 'The worker reports the first stage is done.', 'reason': 'fresh snapshot'}
        _mock_action_generation(runtime, service, generate, monkeypatch)
        result = service.orchestrator_message('How far is the worker?')
        assert result['answer'] == 'The worker reports the first stage is done.'
        assert len(snapshots) == 2
        status = snapshots[-1]['worker_question_inventory']['workers'][0]['latest_model_status']
        assert status['status_kind'] == 'worker_action_status'
        assert status['status_event_sequence'] and status['status_event_hash']
        assert all(m.content != 'The first stage is pending.' for m in service._load_orchestrator_state().messages if m.role == 'assistant')
    finally:
        service.close()


def test_action_refreshes_snapshot_after_context_reduction(make_config, monkeypatch):
    from dataclasses import replace
    from swaag.grammar import agent_action_contract
    from swaag.runtime import RuntimeContextProjection
    from swaag.utils import sha256_text, stable_json_dumps
    runtime = AgentRuntime(make_config(model__context_limit=32000, tools__enabled=[],
        runtime__lean_on_overflow=False), model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        worker = service.workers.create('Complete while context is reduced')
        service.workers.store.transition(worker.worker_id, 'working')
        state = service._orchestrator_state()
        compile_original = runtime._compile_context
        compilations = []
        def compile_once_overflow(*args, **kwargs):
            value = compile_original(*args, **kwargs)
            compilations.append(value)
            if len(compilations) == 1:
                value.report = replace(value.report, context_limit=value.report.required_tokens - 1, fits=False)
            return value
        def project(_state, **kwargs):
            old_snapshot = kwargs['context_state']['runtime_system_snapshot']
            service.workers.store.transition(worker.worker_id, 'completed', result='FRESH_AFTER_REDUCTION_347')
            return 'runtime_system_snapshot', RuntimeContextProjection(
                sha256_text(stable_json_dumps(old_snapshot, indent=2)), 'STALE_PROJECTED_REPORT_917')
        monkeypatch.setattr(runtime, '_compile_context', compile_once_overflow)
        monkeypatch.setattr(runtime, '_project_largest_tool_result_for_overflow', lambda *a, **k: None)
        monkeypatch.setattr(runtime, '_project_runtime_context_for_overflow', project)
        prepared = runtime._prepare_action_call(state, original_request='Has the worker finished?',
            pending_messages=[], tool_specs=[], contract=agent_action_contract([]), validation_feedback='')
        assert 'FRESH_AFTER_REDUCTION_347' in prepared.assembly.prompt_text
        assert 'STALE_PROJECTED_REPORT_917' not in prepared.assembly.prompt_text
        assert prepared.runtime_snapshot['terminal_workers'][0]['status'] == 'completed'
        assert not runtime.client.requests
    finally:
        service.close()


def test_snapshot_changed_while_waiting_for_inference_is_not_sent(make_config, monkeypatch):
    from tests.test_agent_action_loop import FakeModelClient, _action
    def reply(payload):
        return _action(message=('The worker finished.' if 'FRESH_AT_ADMISSION_463' in payload['prompt']
                                else 'The worker is still working.'))
    client = FakeModelClient([reply, reply])
    runtime = AgentRuntime(make_config(model__context_limit=32000, tools__enabled=[]), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        worker = service.workers.create('Complete while inference is occupied')
        service.workers.store.transition(worker.worker_id, 'working')
        acquire = runtime.inference.acquire
        changed = []
        def finish_at_admission(*args, **kwargs):
            admitted = acquire(*args, **kwargs)
            if not changed:
                changed.append(True)
                service.workers.store.transition(worker.worker_id, 'completed', result='FRESH_AT_ADMISSION_463')
            return admitted
        monkeypatch.setattr(runtime.inference, 'acquire', finish_at_admission)
        result = service.orchestrator_message('Has the worker finished?')
        assert result['answer'] == 'The worker finished.'
        assert len(client.requests) == 1
        assert 'FRESH_AT_ADMISSION_463' in client.requests[0]['prompt']
        state = service._load_orchestrator_state()
        requests = runtime.inference.list(session_id=state.session_id)
        assert {item.status for item in requests} == {'superseded', 'completed'}
        assert runtime.preemption.active_call(state.session_id) is None
        assert all(m.content != 'The worker is still working.' for m in state.messages if m.role == 'assistant')
    finally:
        service.close()
