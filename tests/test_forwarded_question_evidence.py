"""Original-user authority remains valid evidence after a foreground handoff."""
import pytest
from swaag.runtime import AgentRuntime
from swaag.tools.registry import ToolRegistry
from swaag.tools.base import ToolValidationError
from swaag.types import Message
from tests.test_question_lifecycle import ask, resolve_input


def fixture(make_config, *, authority=True, source='agent_control', receipt=True):
    runtime=AgentRuntime(make_config(),model_client=object())
    state=runtime.create_or_load_session()
    question=ask(runtime,state)
    origin=runtime.create_or_load_session()
    runtime._record_message(origin,Message(role='user',content='Use staging.',created_at='now'))
    origin_event=next(e for e in runtime.history.iter_history_reverse(origin.session_id) if e.event_type=='message_added')
    originals=[{'message':'Use staging.','source_session_id':origin.session_id,'source_event_sequence':origin_event.sequence}] if authority else []
    if receipt:
        control=runtime.history.enqueue_control_message(state.session_id,'Routing note is not authority.',
            source=source,authority=originals)
        control_id=control['control_id']
    else:control_id='invented'
    processed=runtime.history.record_event(state,'control_message_processed',{
        'control_id':control_id,'session_id':state.session_id,'message':'Routing note is not authority.','decision':{}})
    if receipt:runtime.history.mark_control_message_processed(state.session_id,control_id)
    return runtime,state,question,processed


def test_original_forwarded_user_answer_is_accepted_as_resolution_evidence(make_config):
    runtime,state,question,processed=fixture(make_config)
    _,result=ToolRegistry().dispatch('questions',resolve_input(question.id,processed.sequence),runtime.config,state)
    assert result.output['resolved'] is True
    event=result.generated_events[0]
    runtime.history.record_event(state,event.event_type,event.payload)
    assert not runtime.history.rebuild_from_history(state.session_id,prefer_checkpoint=False).open_questions
    assert runtime.history.list_authoritative_control_messages(state.session_id)[0]['authority'][0]['message']=='Use staging.'


@pytest.mark.parametrize('changes',[{'authority':False},{'source':'scheduler'},{'receipt':False}])
def test_non_user_or_unbacked_controls_are_not_user_answer_evidence(make_config,changes):
    runtime,state,question,processed=fixture(make_config,**changes)
    with pytest.raises(ToolValidationError,match='recorded user'):
        ToolRegistry().dispatch('questions',resolve_input(question.id,processed.sequence),runtime.config,state)
    assert runtime.history.rebuild_from_history(state.session_id).open_questions


def test_forwarded_authority_exposes_local_citable_sequence_without_routing_note(make_config):
    runtime,state,question,processed=fixture(make_config)
    text='\n'.join(runtime._authoritative_intervention_texts(state))
    assert f'local history event sequence {processed.sequence}' in text
    assert 'Use staging.' in text
    assert 'Routing note is not authority.' not in text
