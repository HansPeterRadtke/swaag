import json
from types import SimpleNamespace
from swaag.runtime import AgentRuntime
from swaag.workers import WorkerManager
from swaag.types import Message
from tests.test_agent_action_loop import FakeModelClient
from tests.test_workers import _action


def blocking(runtime,state):
    return runtime.history.record_event(state,'agent_question',{'action_index':1,
        'question':'May I publish?','criticality':'blocking','reason':'Await explicit approval.',
        'assumption_if_unanswered':''})


def test_worker_retains_a_blocking_question_from_an_earlier_turn(make_config,monkeypatch):
    runtime=AgentRuntime(make_config(),model_client=object());manager=WorkerManager(runtime)
    worker=manager.create('Publish only after approval.');state=runtime.history.rebuild_from_history(worker.session_id)
    runtime._record_message(state,Message(role='user',content='Publish only after approval.',created_at='now'))
    question=blocking(runtime,state)
    monkeypatch.setattr(runtime,'resume_turn_in_session',lambda *a:SimpleNamespace(assistant_text='Done'))
    try:
        manager.start(worker.worker_id);result=manager.wait(worker.worker_id,timeout_seconds=10)
        assert result.status=='input_required'
        assert runtime.history.rebuild_from_history(worker.session_id).open_questions[0]['question_id']==question.id
    finally:manager.shutdown()


def test_worker_does_not_wait_on_a_question_resolved_during_this_turn(make_config,monkeypatch):
    runtime=AgentRuntime(make_config(),model_client=object());manager=WorkerManager(runtime)
    worker=manager.create('Complete the authorized task.')
    def turn(state,*a):
        question=blocking(runtime,state)
        runtime._record_message(state,Message(role='user',content='Yes, publish.',created_at='now'))
        evidence=state.event_count
        runtime.history.record_event(state,'agent_question_resolved',{'question_id':question.id,
            'resolution':'answered_by_user','answer':'Approved.','reason':'Explicit user answer.',
            'evidence_sequences':[evidence]})
        return SimpleNamespace(assistant_text='Done')
    monkeypatch.setattr(runtime,'run_turn_in_session',turn)
    try:
        manager.start(worker.worker_id);result=manager.wait(worker.worker_id,timeout_seconds=10)
        assert result.status=='completed'
        assert not runtime.history.rebuild_from_history(worker.session_id).open_questions
    finally:manager.shutdown()


def test_worker_terminal_action_reconciles_durable_question_before_review(make_config):
    holder={}
    def resolve(payload):
        assert 'durable-state invariant' in payload['prompt']
        assert holder['question_id'] in payload['prompt']
        action=json.loads(_action('The approval is recorded.'));action['continue_loop']=True
        action['tool_calls']=[{'tool_name':'questions','arguments':{'operation':'resolve',
            'question_id':holder['question_id'],'resolution':'answered_by_user','answer':'Approved.',
            'reason':'The user explicitly approved publication.','evidence_sequences':[holder['evidence']]}}]
        return json.dumps(action)
    client=FakeModelClient([_action('Done'),resolve,_action('Done after reconciliation')])
    runtime=AgentRuntime(make_config(model__context_limit=32000,tools__enabled=['questions']),model_client=client)
    state=runtime.create_or_load_user_session('Publish after approval.');question=blocking(runtime,state)
    runtime._record_message(state,Message(role='user',content='Yes, publish.',created_at='now'))
    holder.update(question_id=question.id,evidence=state.event_count)
    runtime.bind_tool_runtime_capability(state.session_id,'worker_id','worker_fixture')
    result=runtime.resume_turn_in_session(state,'Publish after approval.')
    assert result.assistant_text=='Done after reconciliation'
    assert not runtime.history.rebuild_from_history(state.session_id).open_questions
    assert len(client.requests)==3


def test_foreground_status_can_finish_with_a_local_pending_question(make_config):
    client=FakeModelClient([_action('Publication is still waiting for approval.')])
    runtime=AgentRuntime(make_config(model__context_limit=32000),model_client=client)
    state=runtime.create_or_load_user_session('Report current status.');blocking(runtime,state)
    result=runtime.resume_turn_in_session(state,'Report current status.')
    assert result.assistant_text=='Publication is still waiting for approval.'
    assert len(runtime.history.rebuild_from_history(state.session_id).open_questions)==1
