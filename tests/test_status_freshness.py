"""Status answers are observations, not permission to publish known-stale facts."""
import pytest
from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from tests.test_preemption import _ImmediateClient


def test_inventory_batches_status_scan_and_refreshes_external_rejection(make_config, monkeypatch):
    from swaag.history import HistoryStore
    runtime = AgentRuntime(make_config(), model_client=_ImmediateClient('unused'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        workers = [service.workers.create('Inspect evidence') for _ in range(4)]
        operations = []
        for index, worker in enumerate(workers):
            operation = runtime.create_or_load_session(session_id=f'operation_status_{index}')
            runtime.history.record_event(operation, 'communication_status_generated', {
                'target_session_id': worker.session_id, 'question': 'Status?',
                'status': status_result(f'Observed worker {index}'),
                'mechanical_status': {}, 'source_event_references': [], 'evidence_projected': False})
            operations.append(operation)
        calls = []
        original = runtime.history.list_session_entries
        def counted(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)
        monkeypatch.setattr(runtime.history, 'list_session_entries', counted)
        def inventory():
            return service.orchestration_api.execute('questions.list')['inventory']
        first = inventory()
        assert len(calls) == 1
        assert {w['latest_model_status']['answer'] for w in first['workers']} == {
            f'Observed worker {index}' for index in range(4)}
        # A different writer invalidates one observation. No batch cache may
        # survive into the next inventory or erase other workers' evidence.
        external = HistoryStore(runtime.config.sessions.root)
        external.record_event(operations[0], 'communication_status_reply_rejected', {
            'target_session_id': workers[0].session_id, 'question': 'Status?',
            'attempt': 1, 'reason': 'changed', 'observed_fingerprint': 'before',
            'current_fingerprint': 'after', 'next_action': 'refresh'})
        second = inventory()
        assert len(calls) == 2
        statuses = {w['worker_id']: w['latest_model_status'] for w in second['workers']}
        assert statuses[workers[0].worker_id] is None
        assert all(statuses[w.worker_id]['answer'] == f'Observed worker {i}'
                   for i, w in enumerate(workers) if i)
    finally:
        service.close()

def status_result(answer):
    return {'answer':answer,'situation':'Observed','action':'Report','reason':'Evidence','importance':'normal','importance_rank':1,'evidence_sequences':[],'uncertainty':'Observed snapshot','escalate_to_stronger_model':False,'escalation_reason':'','evidence_projected':False,'source_event_references':[]}

def test_independent_status_refreshes_after_execution_changes(make_config,monkeypatch):
    main=AgentRuntime(make_config(model__context_limit=32000),model_client=_ImmediateClient('unused'))
    assistant=AgentRuntime(make_config(model__context_limit=32000),model_client=_ImmediateClient('unused'))
    target=main.create_or_load_session()
    service=CommunicationService(main,assistant_runtime=assistant)
    calls=[]
    def generate(operation_state,**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            main.history.record_event(target,'tool_result',{'tool_name':'calculator','raw_input':{'expression':'6*7'},'validated_input':{'expression':'6*7'},'output':{'result':42}})
            return status_result('No result has been recorded.')
        assert any(e.event_type=='tool_result' for e in kwargs['source_events'])
        return status_result('The calculation returned forty-two.')
    monkeypatch.setattr(assistant,'_generate_communication_status',generate)
    try:
        assert service.answer_status_question(target.session_id,'Has the calculation run?')=='The calculation returned forty-two.'
        assert len(calls)==2
        latest=main.history.latest_communication_status(target.session_id)
        assert latest.payload['status']['answer']=='The calculation returned forty-two.'
        rejections=[e for ent in assistant.history.list_session_entries(include_internal=True) for e in assistant.history.read_history(ent['session_id']) if e.event_type=='communication_status_reply_rejected']
        assert len(rejections)==1
    finally:service.close()

def test_continuous_task_changes_withhold_status_and_latest_projection(make_config,monkeypatch):
    main=AgentRuntime(make_config(model__context_limit=32000),model_client=_ImmediateClient('unused'))
    assistant=AgentRuntime(make_config(model__context_limit=32000),model_client=_ImmediateClient('unused'))
    target=main.create_or_load_session();service=CommunicationService(main,assistant_runtime=assistant)
    calls=[]
    def generate(operation_state,**kwargs):
        calls.append(kwargs)
        main.history.record_event(target,'tool_called',{'call_id':str(len(calls)),'tool_name':'calculator','tool_input':{'expression':'6*7'}})
        return status_result('No tool has started.')
    monkeypatch.setattr(assistant,'_generate_communication_status',generate)
    try:
        with pytest.raises(RuntimeError,match='Task state changed during both status'):
            service.answer_status_question(target.session_id,'What is running?')
        assert len(calls)==2
        assert main.history.latest_communication_status(target.session_id) is None
    finally:service.close()

def test_status_ignores_accounting_and_heartbeat_but_detects_queued_control(make_config,monkeypatch):
    main=AgentRuntime(make_config(model__context_limit=32000),model_client=_ImmediateClient('unused'))
    assistant=AgentRuntime(make_config(model__context_limit=32000),model_client=_ImmediateClient('unused'))
    target=main.create_or_load_session();service=CommunicationService(main,assistant_runtime=assistant)
    main.history.set_active_run(target.session_id,run_id='same-run',user_text='Keep working')
    main.history.update_active_run(target.session_id,run_id='same-run',phase='inference',substate='streaming',detail='initial heartbeat')
    calls=[]
    def generate(operation_state,**kwargs):
        calls.append(kwargs)
        main.history.record_event(target,'model_tokenize_result',{'text_hash':'fixture','tokens':40,'exact':True})
        main.history.update_active_run(target.session_id,run_id='same-run',phase='inference',substate='streaming',detail='new heartbeat')
        return status_result('The task is active.')
    monkeypatch.setattr(assistant,'_generate_communication_status',generate)
    try:
        assert service.answer_status_question(target.session_id,'Status?')=='The task is active.'
        assert len(calls)==1
        def changed(operation_state,**kwargs):
            calls.append(kwargs)
            if len(calls)==2:
                main.history.enqueue_control_message(target.session_id,'Stop after this step.',source='user')
            if len(calls)>2:
                assert kwargs['mechanical_status']['pending_user_message_receipts'][0]['message']=='Stop after this step.'
            return status_result('A control is pending.' if len(calls)>2 else 'No control is pending.')
        monkeypatch.setattr(assistant,'_generate_communication_status',changed)
        assert service.answer_status_question(target.session_id,'Anything pending?')=='A control is pending.'
        assert len(calls)==3
    finally:service.close()



def test_escalation_recaptures_after_preemption_wait(make_config,monkeypatch):
    main=AgentRuntime(make_config(model__context_limit=32000),model_client=_ImmediateClient('unused'))
    assistant=AgentRuntime(make_config(model__context_limit=32000),model_client=_ImmediateClient('unused'))
    target=main.create_or_load_session();service=CommunicationService(main,assistant_runtime=assistant)
    def weak(operation_state,**kwargs):
        return {**status_result('Uncertain draft.'),'escalate_to_stronger_model':True,'escalation_reason':'Need stronger interpretation'}
    def preempt(*args):
        main.history.record_event(target,'tool_result',{'tool_name':'calculator','raw_input':{},'validated_input':{},'output':{'result':'NEW_DURING_WAIT'}})
        return None
    calls=[]
    def strong(operation_state,**kwargs):
        calls.append(kwargs)
        assert any(e.payload.get('output',{}).get('result')=='NEW_DURING_WAIT' for e in kwargs['source_events'])
        return status_result('The new result is available.')
    monkeypatch.setattr(assistant,'_generate_communication_status',weak)
    monkeypatch.setattr(main,'_generate_communication_status',strong)
    monkeypatch.setattr(service,'_preempt_active_main_call',preempt)
    try:
        assert service.answer_status_question(target.session_id,'Latest result?')=='The new result is available.'
        assert len(calls)==1
        latest=main.history.latest_communication_status(target.session_id)
        assert latest.payload['status']['answer']=='The new result is available.'
    finally:service.close()

def test_failed_escalation_does_not_publish_small_model_draft(make_config,monkeypatch):
    main=AgentRuntime(make_config(model__context_limit=32000),model_client=_ImmediateClient('unused'))
    assistant=AgentRuntime(make_config(model__context_limit=32000),model_client=_ImmediateClient('unused'))
    target=main.create_or_load_session();service=CommunicationService(main,assistant_runtime=assistant)
    def weak(operation_state,**kwargs):
        return {**status_result('Unsupported provisional answer.'),'escalate_to_stronger_model':True,'escalation_reason':'Need evidence'}
    def strong(*args,**kwargs):raise RuntimeError('injected stronger backend failure')
    monkeypatch.setattr(assistant,'_generate_communication_status',weak)
    monkeypatch.setattr(main,'_generate_communication_status',strong)
    try:
        with pytest.raises(RuntimeError,match='injected stronger'):
            service.answer_status_question(target.session_id,'Status?')
        assert main.history.latest_communication_status(target.session_id) is None
    finally:service.close()


def test_independent_status_rejects_queued_snapshot_after_inference_starts(make_config, monkeypatch):
    main = AgentRuntime(make_config(model__context_limit=32000), model_client=_ImmediateClient('unused'))
    assistant = AgentRuntime(make_config(model__context_limit=32000), model_client=_ImmediateClient('unused'))
    target = main.create_or_load_session()
    main.history.set_active_run(target.session_id, run_id='live-main', user_text='Keep working')
    main.history.update_active_run(target.session_id, run_id='live-main', phase='queued_inference',
        substate='awaiting_capacity', active_kind='model', active_id='same-call', operation_kind='action')
    service = CommunicationService(main, assistant_runtime=assistant)
    calls = []
    def generate(operation_state, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            assert kwargs['mechanical_status']['active_run']['phase'] == 'queued_inference'
            main.history.update_active_run(target.session_id, run_id='live-main', phase='inference',
                substate='streaming', active_kind='model', active_id='same-call', operation_kind='action')
            return status_result('The main agent is waiting for model capacity.')
        assert kwargs['mechanical_status']['active_run']['phase'] == 'inference'
        return status_result('The main agent is generating its response.')
    monkeypatch.setattr(assistant, '_generate_communication_status', generate)
    try:
        assert service.answer_status_question(target.session_id, 'What is happening now?') == 'The main agent is generating its response.'
        assert len(calls) == 2
        latest = main.history.latest_communication_status(target.session_id)
        assert latest.payload['status']['answer'] == 'The main agent is generating its response.'
    finally:
        service.close()
