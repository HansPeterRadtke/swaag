"""Mechanical audit bulk must remain recoverable without crowding out task state."""
import json
from dataclasses import asdict
from swaag.runtime import AgentRuntime
from swaag.evidence_sources import RuntimeTelemetryEvidenceProvider, RuntimeAuditEvidenceProvider, runtime_telemetry_reference
from tests.test_completion_evaluation import _completed_action
from tests.test_question_lifecycle import ask


def test_turn_completion_keeps_answer_exact_but_recovers_budget_audit_separately(make_config):
    runtime=AgentRuntime(make_config(),model_client=object());state=runtime.create_or_load_session()
    event=runtime.history.record_event(state,'turn_finished',{'turn_index':1,'assistant_text':'The checks passed; approval is still required.','tool_steps':2,
        'budget_reports':[{'measurement':'EXACT_BUDGET_DETAIL_983','breakdown':['large']*2000}]})
    view=runtime._completion_historical_evidence_row(event)
    assert view['payload']['assistant_text']==event.payload['assistant_text']
    assert view['payload']['tool_steps']==2
    assert 'EXACT_BUDGET_DETAIL_983' not in json.dumps(view)
    rows=runtime._completion_evidence_source_inventory(state,evidence_rows=[],historical_events=[event])
    source=next(x for x in rows if x['source_kind']=='runtime_audit')
    exact=RuntimeAuditEvidenceProvider().reexpand(config=runtime.config,state=state,source=source)
    assert json.loads(exact['text'])==asdict(event)
    assert runtime.history.read_history(state.session_id)[-1].payload==event.payload


def test_model_envelopes_share_one_exact_inventory_range_without_dropping_task_results(make_config):
    runtime=AgentRuntime(make_config(),model_client=object());state=runtime.create_or_load_session()
    events=[]
    for i in range(12):
        events.append(runtime.history.record_event(state,'model_response_received',{
            'kind':'action','prompt_mode':'standard','attempt':1,'completion':{'text':f'RAW_UNACCEPTED_DRAFT_{i}'}}))
    task=runtime.history.record_event(state,'message_added',{'message':{
        'role':'user','content':'Preserve this exact actual instruction.','created_at':'now','metadata':{}}})
    source_events=[*events,task]
    rows=runtime._completion_historical_evidence_rows(source_events)
    assert runtime._communication_evidence_row(task) in rows
    assert 'RAW_UNACCEPTED_DRAFT_' not in json.dumps(rows)
    inventory=runtime._completion_evidence_source_inventory(state,evidence_rows=[],historical_events=source_events)
    assert len(inventory)==1
    assert inventory[0]['source_kind']=='runtime_telemetry' and inventory[0]['event_count']==12
    exact=RuntimeTelemetryEvidenceProvider().reexpand(config=runtime.config,state=state,source=inventory[0])
    assert json.loads(exact['text'])==[asdict(e) for e in events]


def test_older_telemetry_range_keeps_its_original_event_type_selection(make_config):
    runtime=AgentRuntime(make_config(),model_client=object());state=runtime.create_or_load_session()
    first=runtime.history.record_event(state,'budget_checked',{'kind':'action','prompt_mode':'standard','budget_report':{'one':1},'cap_error':''})
    runtime.history.record_event(state,'model_response_received',{'kind':'action','prompt_mode':'standard','attempt':1,'completion':{'text':'MODEL_ENVELOPE_EXCLUDED_BY_OLD_RANGE'}})
    last=runtime.history.record_event(state,'budget_checked',{'kind':'action','prompt_mode':'standard','budget_report':{'two':2},'cap_error':''})
    source=runtime_telemetry_reference([first,last])
    exact=RuntimeTelemetryEvidenceProvider().reexpand(config=runtime.config,state=state,source=source)
    assert json.loads(exact['text'])==[asdict(first),asdict(last)]


def test_completion_receives_current_open_questions_as_explicit_state(make_config):
    from tests.test_agent_action_loop import FakeModelClient
    def inspect(payload):
        result={'complete':False,'reason':'Record the answered question resolution.','remaining_work':['Resolve the existing question using its user evidence.']}
        if 'evidence_requests' in payload['json_schema']['properties']:result['evidence_requests']=[]
        return json.dumps(result)
    runtime=AgentRuntime(make_config(model__context_limit=32000),model_client=FakeModelClient([inspect]))
    state=runtime.create_or_load_session();ask(runtime,state)
    runtime.history.record_event(state,'turn_started',{'turn_index':1,'user_text':'Finish the authorized task and settle the question.'})
    result=runtime._evaluate_completion(state,original_request='Finish the authorized task and settle the question.',
        selected_action=_completed_action(),tool_results=[])
    assert result['complete'] is False
    requests=[r for r in runtime.client.requests if r['contract'].startswith('completion')]
    assert requests
    assert 'Current unresolved questions for this session' in requests[-1]['prompt']
    assert 'Which destination?' in requests[-1]['prompt']


def test_telemetry_type_metadata_cannot_drop_or_add_source_events(make_config):
    import pytest
    from swaag.history import HistoryInvariantError
    runtime=AgentRuntime(make_config(),model_client=object());state=runtime.create_or_load_session()
    event=runtime.history.record_event(state,'model_response_received',{
        'kind':'action','prompt_mode':'standard','completion':{'text':'EXACT_RAW_RESPONSE'}})
    source=runtime_telemetry_reference([event])
    provider=RuntimeTelemetryEvidenceProvider()
    for types in [None, {}, {'message_added':1}, {'model_response_received':True},
                  {'model_response_received':2}]:
        with pytest.raises(HistoryInvariantError):
            provider.reexpand(config=runtime.config,state=state,source={**source,'event_types':types})


def test_legacy_individual_model_audit_reference_remains_recoverable(make_config):
    from swaag.utils import stable_json_dumps,sha256_text
    runtime=AgentRuntime(make_config(),model_client=object());state=runtime.create_or_load_session()
    event=runtime.history.record_event(state,'model_response_received',{
        'kind':'action','prompt_mode':'standard','completion':{'text':'OLD_EXACT_RESPONSE'}})
    text=stable_json_dumps(asdict(event),indent=None)
    source={'source_kind':'runtime_audit','source_id':f'{state.session_id}:{event.sequence}',
        'sha256':sha256_text(text),'size_chars':len(text),'source_event_references':[{
            'session_id':state.session_id,'sequence':event.sequence,'hash':event.hash,
            'event_type':event.event_type}]}
    exact=RuntimeAuditEvidenceProvider().reexpand(config=runtime.config,state=state,source=source)
    assert json.loads(exact['text'])==asdict(event)
