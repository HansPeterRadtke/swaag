import tempfile
import unittest
import threading
import asyncio
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

if __name__=='__main__':unittest.main()
