import pytest
from swaag.runtime import AgentRuntime
from swaag.tokens import CountResult
from swaag.types import Message,PromptComponent
from tests.test_completion_evaluation import _CompletionClient,_completed_action
from tests.test_question_lifecycle import ask


class ReductionCaptured(Exception):
    pass


@pytest.mark.parametrize('single_verdict',[False,True])
def test_history_projection_uses_measured_room_and_identical_required_context(make_config,monkeypatch,single_verdict):
    runtime=AgentRuntime(make_config(model__context_limit=4000),model_client=_CompletionClient([]))
    state=runtime.create_or_load_user_session('Earlier objective.')
    runtime._record_message(state,Message(role='assistant',content='OLDER_FACT_643 '*6000,created_at='now'))
    runtime.history.record_event(state,'model_response_received',{'kind':'action','prompt_mode':'standard',
        'attempt':1,'completion':{'text':'Old transport record.'}})
    ask(runtime,state)
    runtime._record_message(state,Message(role='user',content='Finish now.',created_at='now'))
    monkeypatch.setattr(runtime,'_single_responsibility_mode',lambda:single_verdict)
    monkeypatch.setattr(runtime,'_current_authority_components',lambda *a:[PromptComponent(
        name='required_verbatim_authority',category='current_user',text='NEVER_DROP_AUTHORITY_914. '+('Required constraint. '*40))])
    counter=runtime._counter;changed=[]
    class UnderestimatedStandaloneHistory:
        def __init__(self,inner):self.inner=inner
        def count_text(self,text):
            actual=self.inner.count_text(text)
            if not changed and text.startswith('[') and 'OLDER_FACT_643' in text:
                changed.append(True)
                return CountResult(8,False,'standalone estimate differs from serialized request')
            return actual
    monkeypatch.setattr(runtime,'_counter',lambda *a,**kw:UnderestimatedStandaloneHistory(counter(*a,**kw)))
    compile_context=runtime._compile_context;captured={}
    def compile_probe(state,assembly,*a,**kw):
        result=compile_context(state,assembly,*a,**kw)
        if any('[purpose-specific historical evidence projection]' in c.text for c in assembly.components):
            captured['placeholder']=assembly
            captured['capacity']=result.available_input_tokens-result.report.input_tokens
        return result
    monkeypatch.setattr(runtime,'_compile_context',compile_probe)
    def capture_reduction(*a,**kw):
        captured['target']=kw['target_tokens'];raise ReductionCaptured()
    monkeypatch.setattr(runtime,'_reduce_text_hierarchically',capture_reduction)
    with pytest.raises(ReductionCaptured):
        runtime._evaluate_completion(state,original_request='Finish now.',selected_action=_completed_action(),tool_results=[])
    assert changed and captured['capacity']>128
    assert captured['target']>128, 'Do not subtract whole-request overflow from an unrelated standalone estimate'
    assert captured['target']<=captured['capacity']
    components={c.name:c.text for c in captured['placeholder'].components}
    assert 'NEVER_DROP_AUTHORITY_914' in components.get('required_verbatim_authority','')
    assert 'Which destination?' in components.get('completion_open_questions','')
    if single_verdict:assert 'completion_evidence_source_inventory' not in components
