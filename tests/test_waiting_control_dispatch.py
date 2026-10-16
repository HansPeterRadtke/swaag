"""A durable answer must reach a waiting worker, with no implicit task revival."""
import pytest
from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from swaag.types import Message
from tests.test_preemption import _ImmediateClient


def setup(make_config, monkeypatch):
    runtime = AgentRuntime(make_config(model__context_limit=32000,
        tools__enabled=['agent_control'], tools__allow_side_effect_tools=True),
        model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    submitted = []
    monkeypatch.setattr(service.workers, '_submit', submitted.append)
    source = service._orchestrator_state()
    original = 'The publication hold is lifted. Continue the same task.'
    runtime._record_message(source, Message(role='user', content=original, created_at='now'))
    runtime.history.record_event(source, 'turn_started', {'user_text':original, 'turn_index':1})
    return runtime, service, source, original, submitted


def send(runtime, source, ref):
    run = runtime.execute_tool_once('agent_control',
        {'session_ref':ref, 'message':'Apply the answer to the existing task.'},
        session_id=source.session_id)
    assert run.error is None
    return run.tool_result.output


@pytest.mark.parametrize('target_kind', ['worker', 'session'])
def test_exact_answer_dispatches_waiting_worker_once(make_config, monkeypatch, target_kind):
    runtime, service, source, original, submitted = setup(make_config, monkeypatch)
    try:
        worker = service.workers.create('Check and publish after approval.')
        service.workers.store.transition(worker.worker_id, 'input_required', expected={'created'})
        ref = worker.worker_id if target_kind == 'worker' else worker.session_id
        first = send(runtime, source, ref)
        retry = send(runtime, source, ref)
        assert service.workers.store.get(worker.worker_id).status == 'queued'
        assert submitted == [worker.worker_id]
        assert first['control_id'] == retry['control_id']
        pending = runtime.history.list_pending_control_messages(worker.session_id)
        assert len(pending) == 1
        assert pending[0]['authority'][0]['message'] == original
        # Consumed retry must not revive a subsequent wait.
        runtime.history.mark_control_message_processed(worker.session_id, first['control_id'])
        service.workers.store.transition(worker.worker_id, 'input_required', expected={'queued'})
        assert send(runtime, source, ref)['status'] == 'processed'
        assert service.workers.store.get(worker.worker_id).status == 'input_required'
        assert submitted == [worker.worker_id]
    finally:service.close()


@pytest.mark.parametrize('status', ['created','completed','failed','canceled','cancellation_requested'])
def test_control_does_not_implicitly_start_or_revive_other_states(make_config, monkeypatch, status):
    runtime, service, source, original, submitted = setup(make_config, monkeypatch)
    try:
        worker = service.workers.create('Existing task.')
        if status != 'created':service.workers.store.transition(worker.worker_id, status, expected={'created'})
        send(runtime, source, worker.worker_id)
        assert service.workers.store.get(worker.worker_id).status == status
        assert submitted == []
        assert len(runtime.history.list_pending_control_messages(worker.session_id)) == 1
    finally:service.close()


def test_answer_does_not_bypass_pending_delegated_tool(make_config, monkeypatch):
    runtime, service, source, original, submitted = setup(make_config, monkeypatch)
    try:
        worker = service.workers.create('Existing task.')
        service.workers.store.transition(worker.worker_id, 'input_required', expected={'created'})
        monkeypatch.setattr(runtime.delegated_tools, 'pending_call', lambda _session:object())
        send(runtime, source, worker.worker_id)
        assert service.workers.store.get(worker.worker_id).status == 'input_required'
        assert submitted == []
        assert runtime.history.list_pending_control_messages(worker.session_id)
    finally:service.close()


def test_service_recovers_enqueue_before_dispatch_crash_window(make_config, monkeypatch):
    runtime, service, source, original, submitted = setup(make_config, monkeypatch)
    try:
        worker = service.workers.create('Check and publish.')
        service.workers.store.transition(worker.worker_id, 'input_required', expected={'created'})
        receipt = runtime.history.enqueue_control_message(worker.session_id, 'Approved.',
            source='agent_control', authority=[{'message':original,'source_session_id':source.session_id}])
        # No tool callback survived; the ordinary service tick must recover it.
        service._advance_orchestration()
        service._advance_orchestration()
        assert service.workers.store.get(worker.worker_id).status == 'queued'
        assert submitted == [worker.worker_id]
        assert runtime.history.list_pending_control_messages(worker.session_id)[0]['control_id'] == receipt['control_id']
    finally:service.close()


def test_recovery_does_not_execute_question_revision_as_user_answer(make_config, monkeypatch):
    runtime, service, source, original, submitted = setup(make_config, monkeypatch)
    try:
        worker = service.workers.create('Existing task.')
        service.workers.store.transition(worker.worker_id, 'input_required', expected={'created'})
        runtime.history.enqueue_control_message(worker.session_id, '{}', source='question_revision')
        service._advance_orchestration()
        assert service.workers.store.get(worker.worker_id).status == 'input_required'
        assert submitted == []
    finally:service.close()


def test_delivery_racing_wait_commit_is_not_stranded(make_config, monkeypatch):
    import threading
    runtime, service, source, original, submitted = setup(make_config, monkeypatch)
    entered = threading.Event(); replies = []
    try:
        worker = service.workers.create('Existing task.')
        service.workers.store.transition(worker.worker_id, 'working', expected={'created'})
        find = service.workers.store.find_by_session
        def observed(session_id):
            result = find(session_id)
            if session_id == worker.session_id:entered.set()
            return result
        monkeypatch.setattr(service.workers.store, 'find_by_session', observed)
        thread = threading.Thread(target=lambda:replies.append(send(runtime, source, worker.worker_id)))
        with service.workers._control_transition_lock:
            thread.start()
            assert entered.wait(5), 'Target delivery must enter the owner even before its wait commits'
            service.workers.store.transition(worker.worker_id, 'input_required', expected={'working'})
        thread.join(5)
        assert not thread.is_alive()
        assert replies and service.workers.store.get(worker.worker_id).status == 'queued'
        assert submitted == [worker.worker_id]
    finally:service.close()


def test_delivery_uses_separate_model_owner(make_config, monkeypatch):
    from copy import deepcopy
    runtime = AgentRuntime(make_config(model__context_limit=32000,
        tools__enabled=['agent_control'], tools__allow_side_effect_tools=True),
        model_client=_ImmediateClient('unused'))
    route_config = deepcopy(runtime.config)
    route_config.sessions.root = runtime.config.sessions.root / 'separate-route'
    routed = AgentRuntime(route_config, model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime, worker_model_runtimes={'other':routed})
    submitted=[];default=[]
    monkeypatch.setattr(service.workers, '_submit', default.append)
    owner=service.worker_model_managers['other']
    monkeypatch.setattr(owner, '_submit', submitted.append)
    try:
        worker=owner.create('Existing task.')
        owner.store.transition(worker.worker_id,'input_required',expected={'created'})
        source=service._orchestrator_state()
        runtime._record_message(source,Message(role='user',content='Use the approved target.',created_at='now'))
        runtime.history.record_event(source,'turn_started',{'user_text':'Use the approved target.','turn_index':1})
        receipt=send(runtime,source,worker.worker_id)
        assert receipt['session_id']==worker.session_id
        assert owner.store.get(worker.worker_id).status=='queued'
        assert submitted==[worker.worker_id] and default==[]
        assert routed.history.list_pending_control_messages(worker.session_id)
    finally:service.close()


def test_archived_wait_is_not_reactivated(make_config, monkeypatch):
    runtime,service,source,original,submitted=setup(make_config,monkeypatch)
    try:
        worker=service.workers.create('Existing task.')
        service.workers.store.transition(worker.worker_id,'input_required',expected={'created'})
        service.workers.store.mark_archived(worker.worker_id,archive={'reason':'test archival marker'})
        send(runtime,source,worker.worker_id)
        service._advance_orchestration()
        assert submitted==[]
        assert service.workers.store.get(worker.worker_id).status=='input_required'
    finally:service.close()


def test_capacity_failure_retains_answer_for_service_recovery(make_config, monkeypatch):
    runtime,service,source,original,submitted=setup(make_config,monkeypatch)
    try:
        service.workers.store.max_active_workers=1
        busy=service.workers.create('Active task.')
        service.workers.store.transition(busy.worker_id,'queued',expected={'created'})
        waiting=service.workers.create('Waiting task.')
        service.workers.store.transition(waiting.worker_id,'input_required',expected={'created'})
        run=runtime.execute_tool_once('agent_control',{'session_ref':waiting.worker_id,'message':'Approved.'},session_id=source.session_id)
        assert run.error is not None
        pending=runtime.history.list_pending_control_messages(waiting.session_id)
        assert len(pending)==1 and pending[0]['authority'][0]['message']==original
        assert submitted==[]
        service.workers.store.transition(busy.worker_id,'completed',expected={'queued'})
        service._advance_orchestration()
        assert service.workers.store.get(waiting.worker_id).status=='queued'
        assert submitted==[waiting.worker_id]
    finally:service.close()
