import tempfile
import unittest
import threading
import asyncio
import pytest
from contextlib import nullcontext
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from swaag.orchestrator_operations import OrchestratorOperations, OperationCanceled
from swaag.communication import CommunicationService

class DurableVoiceOperationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.store=OrchestratorOperations(self.root)
    def tearDown(self):self.tmp.cleanup()
    def test_concurrent_identical_requests_have_one_owner(self):
        with ThreadPoolExecutor(max_workers=8) as workers:
            results=list(workers.map(lambda _:self.store.claim('voice-one','exact words'),range(16)))
        self.assertEqual(1,sum(owner for owner,_ in results))
    def test_reopen_after_uncertain_effect_never_reexecutes(self):
        self.assertTrue(self.store.claim('voice-one','Create B')[0])
        self.store.activate('voice-one','session')
        reopened=OrchestratorOperations(self.root)
        owner,row=reopened.claim('voice-one','Create B')
        self.assertFalse(owner);self.assertEqual('running',row['status'])
    def test_completed_duplicate_returns_exact_canonical_result(self):
        self.store.claim('voice-one','  Exact\ntext  ')
        result={'answer':'Canonical\nanswer','session_id':'session'}
        self.store.finish('voice-one','completed',result)
        owner,row=OrchestratorOperations(self.root).claim('voice-one','  Exact\ntext  ')
        self.assertFalse(owner);self.assertEqual(result,row['result'])
        with self.assertRaises(ValueError):self.store.claim('voice-one','Exact text')
    def test_cancel_can_arrive_before_handoff(self):
        self.store.cancel('voice-one')
        owner,row=self.store.claim('voice-one','Create A')
        self.assertFalse(owner);self.assertEqual('canceled',row['status'])
    def test_cancel_during_inference_is_checked_before_tools(self):
        self.store.claim('voice-one','Create A');self.store.activate('voice-one','session')
        self.store.cancel('voice-one')
        with self.assertRaises(OperationCanceled):self.store.check_cancellation('voice-one')
    def test_cancel_completed_does_not_claim_to_undo_effects(self):
        self.store.claim('voice-one','Create B');self.store.finish('voice-one','completed',{'answer':'Created'})
        self.assertEqual('completed',self.store.cancel('voice-one')['status'])

    def test_activation_cannot_resurrect_canceled_operation(self):
        self.store.claim('voice-one', 'Create A')
        self.store.cancel('voice-one')
        self.store.finish('voice-one', 'canceled')
        with self.assertRaises(OperationCanceled):
            self.store.activate('voice-one', 'session')
        self.assertEqual('canceled', self.store.get('voice-one')['status'])

    def service(self):
        service=CommunicationService.__new__(CommunicationService)
        service.orchestrator_operations=self.store
        service._active_orchestrator_operation=''
        service._orchestrator_operation_lock=threading.Lock()
        service.orchestrator_runtime=Mock()
        return service

    def test_service_returns_completed_result_without_running_tools_again(self):
        service=self.service()
        service._orchestrator_message_once=Mock(return_value={'answer':'Created B','session_id':'s'})
        first=service.orchestrator_message('Create B','voice-one')
        second=service.orchestrator_message('Create B','voice-one')
        self.assertEqual(first['answer'],second['answer']);self.assertTrue(second['duplicate'])
        service._orchestrator_message_once.assert_called_once_with('Create B','voice-one')

    def test_service_never_repeats_uncertain_external_effect(self):
        service=self.service();effects=[]
        def execute(*args):
            effects.append('written')
            raise OSError('reply lost after write')
        service._orchestrator_message_once=execute
        with self.assertRaises(OSError):service.orchestrator_message('Create B','voice-one')
        with self.assertRaises(RuntimeError):service.orchestrator_message('Create B','voice-one')
        self.assertEqual(['written'],effects)
        self.assertEqual('outcome_unknown',self.store.get('voice-one')['status'])

    def test_stale_cancellation_does_not_cancel_a_different_operation(self):
        service=self.service()
        self.store.claim('voice-old','First');self.store.activate('voice-old','s')
        service._active_orchestrator_operation='voice-new'
        service.cancel_orchestrator_operation('voice-old','Stop old reply')
        service.orchestrator_runtime.preemption.request_run_cancellation.assert_not_called()

    def test_active_cancellation_targets_exact_durable_run(self):
        service=self.service()
        self.store.claim('voice-one','Create B');self.store.activate('voice-one','s')
        service._active_orchestrator_operation='voice-one'
        service.orchestrator_runtime.history.read_active_run.return_value={'run_id':'exact-run'}
        result=service.cancel_orchestrator_operation('voice-one','User interrupted')
        service.orchestrator_runtime.preemption.request_run_cancellation.assert_called_once_with(
            's','exact-run',reason='User interrupted')
        self.assertTrue(result['cancel_requested']);self.assertFalse(result['cancellation_acknowledged'])

    def test_interrupt_rpc_bypasses_saturated_normal_request_slots(self):
        async def run():
            service=self.service();service.runtime=Mock()
            service.runtime.telemetry.protocol_server_request.side_effect=lambda **kwargs:nullcontext()
            service._semaphore=asyncio.Semaphore(1)
            await service._semaphore.acquire()
            result=await asyncio.wait_for(service._dispatch_json_line_request(
                {'op':'orchestrator.interrupt','params':{'operation_id':'voice-one','reason':'Stop'}}),1)
            self.assertEqual('canceled',result['status'])
            status=await service._dispatch_json_line_request(
                {'op':'orchestrator.operation_status','params':{'operation_id':'voice-one'}})
            self.assertEqual('canceled',status['status'])
        asyncio.run(run())

def test_cancel_before_run_identity_cannot_execute_or_poison_next_turn(make_config, monkeypatch):
    from swaag.runtime import AgentRuntime
    from swaag.preemption import RunCancellationRequested
    from tests.test_agent_action_loop import FakeModelClient, _action
    client = FakeModelClient([_action(message='The new request is answered.')])
    runtime = AgentRuntime(make_config(model__context_limit=32000), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    entered, release = threading.Event(), threading.Event()
    original = runtime.run_turn_in_session
    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, 'run_turn_in_session', delayed)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(service.orchestrator_message, 'Create the canceled file.', 'voice-gap')
            assert entered.wait(5)
            service.cancel_orchestrator_operation('voice-gap', 'Withdraw this request')
            release.set()
            with pytest.raises(RunCancellationRequested):
                future.result(timeout=8)
        assert not client.requests
        assert service.orchestrator_operations.get('voice-gap')['status'] == 'canceled'
        state = service._load_orchestrator_state()
        assert runtime.history.list_pending_control_messages(state.session_id) == []
        assert runtime.history.list_authoritative_control_messages(state.session_id) == []
        monkeypatch.setattr(runtime, 'run_turn_in_session', original)
        assert service.orchestrator_message('Answer the new request.')['answer'] == 'The new request is answered.'
        assert len(client.requests) == 1
    finally:
        release.set()
        service.close()


if __name__=='__main__':unittest.main()


def test_voice_correction_reaches_running_action_before_execution_lock(make_config):
    from swaag.runtime import AgentRuntime
    from tests.test_agent_action_loop import FakeModelClient, _action
    entered, release = threading.Event(), threading.Event()
    def stale(payload):
        entered.set()
        assert release.wait(5)
        return _action(message='I am writing it.', tool_calls=[('echo', {'text':'STALE_EFFECT'})], continue_loop=True)
    def corrected(payload):
        assert 'Do not execute that action. Just acknowledge the correction.' in payload['prompt']
        return _action(message='The correction is acknowledged.')
    client = FakeModelClient([stale, corrected])
    runtime = AgentRuntime(make_config(model__context_limit=32000), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        first, old = service._start_voice_operation_stream('Execute the action.', 'original_voice')
        assert entered.wait(5)
        correction, new = service._start_voice_operation_stream(
            'Do not execute that action. Just acknowledge the correction.', 'corrected_voice')
        assert any(p['control_id'] == correction['control_id'] for p in
                   runtime.history.list_pending_control_messages(first['session_id']))
        assert not old.done()
        release.set()
        assert old.result(timeout=8)['answer'] == 'The correction is acknowledged.'
        assert new.result(timeout=8)['answer'] == 'The correction is acknowledged.'
        history = runtime.history.read_history(first['session_id'])
        assert not any(e.event_type == 'tool_called' for e in history)
        assert any(e.event_type == 'agent_action_rejected' for e in history)
        assert len(client.requests) == 2
        assert service.orchestrator_operations.get('corrected_voice')['status'] == 'completed'
    finally:
        release.set()
        service.close()


def test_queued_voice_withdrawal_needs_no_execution_slot_and_cannot_poison_next_turn(make_config):
    from swaag.runtime import AgentRuntime
    from swaag.session_lock import session_execution_lock
    from tests.test_agent_action_loop import FakeModelClient, _action
    client = FakeModelClient([_action(message='The new request is answered.')])
    runtime = AgentRuntime(make_config(model__context_limit=32000), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        state = service._load_orchestrator_state()
        with session_execution_lock(runtime.history, state.session_id):
            receipt, future = service._start_voice_operation_stream('Execute WITHDRAWN_ACTION.', 'queued_voice')
            assert not future.done()
            result = service.cancel_orchestrator_operation('queued_voice', 'Withdraw before execution')
            assert result['cancellation_acknowledged'] is True
            assert runtime.history.list_pending_control_messages(state.session_id) == []
        with pytest.raises(OperationCanceled):
            future.result(timeout=5)
        assert not client.requests
        assert service.orchestrator_operations.get('queued_voice')['status'] == 'canceled'
        assert service.orchestrator_message('Answer the new request.')['answer'] == 'The new request is answered.'
        assert 'WITHDRAWN_ACTION' not in client.requests[0]['prompt']
    finally:
        service.close()


def test_withdrawal_between_snapshot_and_action_commit_rejects_stale_authority(make_config):
    from swaag.runtime import AgentRuntime
    import json
    from swaag.action import action_from_payload
    from tests.test_agent_action_loop import _action
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    runtime.history.enqueue_control_message(state.session_id, 'WITHDRAWN_ACTION', source='orchestrator_user', control_id='withdrawn')
    pending = runtime.history.list_pending_control_messages(state.session_id)
    assert runtime.history.cancel_pending_control_message(state.session_id, 'withdrawn') is True
    action = action_from_payload(json.loads(_action(message='Unaccepted candidate.')), enabled_tool_names=[])
    assert runtime._consume_pending_control_messages(state, pending_payloads=pending, selected_action=action) is False
    assert not any(e.event_type == 'control_message_processed' for e in runtime.history.read_history(state.session_id))
    assert not state.messages


def test_cancel_after_receipt_consumption_recovers_committed_answer_without_replay(make_config, monkeypatch):
    from swaag.runtime import AgentRuntime
    from tests.test_agent_action_loop import FakeModelClient, _action
    entered, release = threading.Event(), threading.Event()
    cancellation = []
    def first(payload):
        entered.set()
        assert release.wait(5)
        return _action(message='Superseded draft.')
    client = FakeModelClient([first, _action(message='Both requests were reconciled.')])
    runtime = AgentRuntime(make_config(model__context_limit=32000), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    consume = runtime._consume_pending_control_messages
    def after_acceptance(state, **kwargs):
        accepted = consume(state, **kwargs)
        if accepted and any(p['control_id'] == service._voice_control_id('late_cancel')
                            for p in kwargs['pending_payloads']):
            cancellation.append(service.cancel_orchestrator_operation('late_cancel', 'Stop if still pending'))
        return accepted
    monkeypatch.setattr(runtime, '_consume_pending_control_messages', after_acceptance)
    try:
        _, old = service._start_voice_operation_stream('Answer the first request.', 'prior_turn')
        assert entered.wait(5)
        _, newer = service._start_voice_operation_stream('Also acknowledge this.', 'late_cancel')
        release.set()
        assert old.result(timeout=8)['answer'] == 'Both requests were reconciled.'
        assert newer.result(timeout=8)['answer'] == 'Both requests were reconciled.'
        assert len(cancellation) == 1 and cancellation[0]['cancellation_acknowledged'] is False
        assert service.orchestrator_operations.get('late_cancel')['status'] == 'completed'
        assert len(client.requests) == 2
    finally:
        release.set()
        service.close()
