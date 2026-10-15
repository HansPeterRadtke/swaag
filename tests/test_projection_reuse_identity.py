"""Objective-dependent reductions cannot answer a later different objective."""
from types import SimpleNamespace
import pytest
from swaag.runtime import AgentRuntime, BudgetExceededError
from swaag.types import Message
from swaag.grammar import agent_action_contract
from swaag.utils import sha256_text


@pytest.mark.parametrize('stored_objective', [None, 'Report Alpha.'])
def test_changed_or_missing_projection_objective_does_not_reuse_old_selection(make_config, monkeypatch, stored_objective):
    runtime=AgentRuntime(make_config(),model_client=object())
    state=runtime.create_or_load_session()
    payload={'source_event_sequence':12,'source_event_hash':'exact-source','tool_name':'read_file',
        'target_tokens':100,'original_tokens':200,'projected_tokens':40,'overflow_tokens':100,
        'projection':'Alpha is green; Beta was omitted for the prior question.'}
    if stored_objective is not None:payload['objective_sha256']=sha256_text(stored_objective)
    runtime.history.record_event(state,'tool_result_projected',payload)
    message=Message(role='tool',name='read_file',content='Alpha is green. Beta is red.',created_at='now',
        metadata={'source_event_sequence':12,'source_event_hash':'exact-source'})
    calls=[]
    def project(_state,**kwargs):
        calls.append(kwargs['original_request']);return 'Beta is red.'
    monkeypatch.setattr(runtime,'_create_tool_result_projection',project)
    result=runtime._project_largest_tool_result_for_overflow(state,original_request='Report Beta.',
        assembly=SimpleNamespace(components=[SimpleNamespace(name='history_tool_event_12')]),
        compilation=SimpleNamespace(overflow_tokens=40,report=SimpleNamespace(breakdown=[SimpleNamespace(name='history_tool_event_12',tokens=200)])),
        existing_projections={},messages=[message])
    assert result==(12,'Beta is red.')
    assert calls==['Report Beta.']


@pytest.mark.parametrize('pending', [[], ['Later instruction: report Beta, not Alpha.']])
def test_failed_identical_tool_reduction_is_not_repeated_during_one_preparation(make_config, monkeypatch, pending):
    runtime=AgentRuntime(make_config(model__context_limit=1500,context__max_compaction_rounds=3),
        model_client=object())
    state=runtime.create_or_load_session()
    message=Message(role='tool',name='read_file',content='RAW ' * 8000,created_at='now',
        metadata={'source_event_sequence':12,'source_event_hash':'exact-source'})
    state.messages=[message]
    monkeypatch.setattr(runtime.history,'read_authoritative_messages',lambda _sid:[message])
    monkeypatch.setattr(runtime,'_runtime_context_components',lambda *args,**kwargs:[])
    monkeypatch.setattr(runtime,'_project_runtime_context_for_overflow',lambda *args,**kwargs:('fixture',object()))
    calls=[]
    def decline(_state,**kwargs):
        assert all(text in kwargs['original_request'] for text in pending)
        calls.append((kwargs['source_event_sequence'] if 'source_event_sequence' in kwargs else kwargs['message'].metadata['source_event_sequence'],kwargs['target_tokens']))
        return ''
    monkeypatch.setattr(runtime,'_create_tool_result_projection',decline)
    with pytest.raises(BudgetExceededError):
        runtime._prepare_action_call(state,original_request='Report the result.',pending_messages=pending,
            tool_specs=[],contract=agent_action_contract([]),validation_feedback='')
    assert calls==[(12,64)], 'Unchanged source/objective/target must not pay the same failed semantic selection each round'


def test_archived_projection_reuse_requires_the_same_objective(make_config, monkeypatch, tmp_path):
    from swaag.history_archive import HistoryArchiveStore
    runtime=AgentRuntime(make_config(),model_client=object());state=runtime.create_or_load_session()
    event=runtime.history.record_event(state,'tool_result_projected',{
        'source_event_sequence':12,'source_event_hash':'source','tool_name':'read_file',
        'target_tokens':80,'original_tokens':200,'projected_tokens':40,'overflow_tokens':100,
        'objective_sha256':sha256_text('Alpha'),'projection':'Alpha is green.'})
    monkeypatch.setattr(runtime.history,'history_path',lambda _sid:tmp_path/'absent-history.jsonl')
    monkeypatch.setattr(HistoryArchiveStore,'read_events',lambda self,_sid:[event])
    for objective,expected in [('Alpha',True),('Beta',False)]:
        found=runtime._stored_tool_result_projection(state,original_request=objective,
            source_event_sequence=12,source_event_hash='source',target_tokens=80)
        assert (found is not None)==expected


def test_failed_selection_retries_when_target_or_objective_changes(make_config, monkeypatch):
    runtime=AgentRuntime(make_config(),model_client=object());state=runtime.create_or_load_session()
    message=Message(role='tool',name='read_file',content='Alpha green; Beta red.',created_at='now',
        metadata={'source_event_sequence':12,'source_event_hash':'source'})
    calls=[];failed=set()
    def decline(_state,**kwargs):
        calls.append((kwargs['original_request'],kwargs['target_tokens']));return ''
    monkeypatch.setattr(runtime,'_create_tool_result_projection',decline)
    for objective,overflow in [('Alpha',140),('Alpha',140),('Alpha',20),('Beta',140)]:
        runtime._project_largest_tool_result_for_overflow(state,original_request=objective,
            assembly=SimpleNamespace(components=[SimpleNamespace(name='history_tool_event_12')]),
            compilation=SimpleNamespace(overflow_tokens=overflow,report=SimpleNamespace(breakdown=[SimpleNamespace(name='history_tool_event_12',tokens=200)])),
            existing_projections={},messages=[message],failed_attempts=failed)
    assert calls==[('Alpha',64),('Alpha',148),('Beta',64)]
