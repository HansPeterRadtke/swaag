import copy
import json
from dataclasses import replace

import pytest

from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from swaag.snapshot_view import snapshot_prompt_view, snapshot_value
from swaag.tools.base import ToolContext, ToolValidationError
from swaag.tools.control import RuntimeSnapshotLookupTool
from swaag.types import Message
from swaag.foreground_history import foreground_action_history
from swaag.foreground_history import fold_identical_prior_user_messages
from tests.test_agent_action_loop import FakeModelClient
from tests.test_completion_evaluation import _completed_action
from tests.test_preemption import _ImmediateClient


@pytest.mark.parametrize('foreground,configured,expected', [
    (True, 0, 0), (True, 400, 400), (False, 0, 256)])
def test_foreground_initial_reservation_is_separate_from_worker_budget(
    make_config, foreground, configured, expected,
):
    from tests.test_agent_action_loop import _action
    config = make_config(model__context_limit=12000,
        context__reserved_response_tokens=256,
        communication__initial_foreground_output_tokens=configured,
        tools__enabled=['runtime_snapshot_lookup'])
    runtime = AgentRuntime(config, model_client=FakeModelClient([_action(message='Hello.')]))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        if foreground:
            result = service.orchestrator_message('Say hello.')
            session_id = result['session_id']
        else:
            session_id = runtime.run_turn('Say hello.').session_id
        event = next(e for e in runtime.history.iter_history_reverse(session_id,
            event_types=('context_compiled',)) if e.payload['kind'] == 'action')
        accounting = event.payload['accounting']
        assert accounting['minimum_output_tokens'] == expected
        assert accounting['reserved_response_tokens'] >= accounting['structured_output_floor_tokens']
    finally:
        service.close()


def test_foreground_output_exhaustion_retries_before_any_partial_effect(make_config):
    from tests.test_agent_action_loop import OutputLimitedFakeModelClient, _action
    class Client(OutputLimitedFakeModelClient):
        def send_completion(self, payload, *, timeout_seconds=None, progress_callback=None):
            result = super().send_completion(payload,
                timeout_seconds=timeout_seconds, progress_callback=progress_callback)
            if result.finish_reason == 'length':
                # Even a complete-looking tool call is unaccepted when the
                # provider reports truncation. No partial intent may execute.
                return replace(result, text=_action(message='Writing it.',
                    tool_calls=[('echo', {'text': 'UNACCEPTED_EFFECT'})], continue_loop=True))
            return result
    client = Client([_action(message='The complete answer.')])
    runtime = AgentRuntime(make_config(model__context_limit=12000,
        tools__enabled=['runtime_snapshot_lookup', 'echo']), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        result = service.orchestrator_message('Give the complete answer.')
        assert result['answer'] == 'The complete answer.'
        assert len(client.requests) == 2
        assert client.requests[1]['n_predict'] > client.requests[0]['n_predict']
        events = runtime.history.read_history(result['session_id'])
        assert not any(e.event_type == 'tool_called' for e in events)
        assert any(e.event_type == 'model_output_budget_exhausted' for e in events)
    finally:
        service.close()


def test_negative_foreground_reservation_is_invalid(tmp_path):
    from swaag.config import load_config
    with pytest.raises(ValueError, match='initial_foreground_output_tokens'):
        load_config(env={
            'SWAAG__SESSIONS__ROOT': str(tmp_path / 'sessions'),
            'SWAAG__COMMUNICATION__INITIAL_FOREGROUND_OUTPUT_TOKENS': '-1',
        })


def _snapshot():
    return {
        'captured_at': 'now',
        'active_workers': [{'worker_id': 'active', 'status': 'working',
                            'objective': 'Do not publish before verification.'}],
        'terminal_workers': [{'worker_id': 'finished', 'session_id': 'old-session',
                              'status': 'failed', 'objective': 'Verify backup',
                              'error': 'EXACT_BACKUP_FAILURE', 'result': None}],
        'terminal_plans': [], 'active_plans': [],
        'worker_controls': [{'pending_controls': [{'message': 'Keep the backup private.'}]}],
        'worker_question_inventory': {
            'complete': True,
            'questions': [{'worker_id': 'finished', 'question': 'Which backup?', 'blocking': True}],
            'pending_revisions': [{'question': 'Use the encrypted backup?'}],
            'workers': [{'worker_id': 'finished', 'status': 'failed', 'progress': {'reason': 'EXACT_BACKUP_FAILURE'}},
                        {'worker_id': 'active', 'status': 'working', 'progress': {'reason': 'Still verifying'}}],
        },
        'unknown_future_field': {'constraint': 'PRESERVE_UNRECOGNIZED_FIELDS'},
    }


def test_view_preserves_live_authority_all_questions_and_recoverable_terminal_details():
    snapshot = _snapshot()
    original = copy.deepcopy(snapshot)
    view = snapshot_prompt_view(snapshot, 42)
    for field in ('active_workers', 'worker_controls', 'unknown_future_field'):
        assert view[field] == snapshot[field]
    for field in ('questions', 'pending_revisions', 'complete'):
        assert view['worker_question_inventory'][field] == snapshot['worker_question_inventory'][field]
    assert view['worker_question_inventory']['workers'][0]['progress']['reason'] == 'Still verifying'
    assert 'EXACT_BACKUP_FAILURE' not in json.dumps(view)
    assert snapshot_value(snapshot, '/terminal_workers/0/error') == 'EXACT_BACKUP_FAILURE'
    assert snapshot_value(snapshot, '') == original == snapshot
    assert view['detail_access']['snapshot_sequence'] == 42


def test_snapshot_references_only_exact_duplicate_authoritative_utterance():
    text = '  Keep every word.\nDo not publish.  '
    snapshot = _snapshot()
    snapshot['orchestrator'] = {'active_turn': {'user_text': text}, 'pending_user_messages': [
        {'control_id': 'same', 'message': text, 'authority': ['verified'], 'priority': 1},
        {'control_id': 'correction', 'message': 'Actually, cancel it.', 'priority': 2},
        {'control_id': 'different_whitespace', 'message': text.strip(), 'priority': 3}]}
    original = copy.deepcopy(snapshot)
    view = snapshot_prompt_view(snapshot, 42, current_user_text=text, current_user_sequence=37)
    assert view['orchestrator']['active_turn']['user_text'] == {'exact_user_text_at_sequence': 37}
    controls = view['orchestrator']['pending_user_messages']
    assert controls[0] == {**original['orchestrator']['pending_user_messages'][0],
                          'message': {'exact_user_text_at_sequence': 37}}
    assert controls[1:] == original['orchestrator']['pending_user_messages'][1:]
    assert snapshot_value(snapshot, '/orchestrator') == original['orchestrator']
    assert snapshot == original
    assert snapshot_prompt_view(snapshot, 42)['orchestrator'] == original['orchestrator']


@pytest.mark.parametrize('pointer', ['terminal_workers', '/terminal_workers/-1', '/terminal_workers/01',
                                    '/terminal_workers/0/error/x', '/~2invalid', '/terminal_workers/99'])
def test_invalid_pointers_cannot_silently_select_different_data(pointer):
    with pytest.raises((ValueError, KeyError, IndexError)):
        snapshot_value(_snapshot(), pointer)


def test_lookup_uses_exact_record_from_current_session(make_config):
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    event = runtime.history.record_event(state, 'runtime_system_snapshot', {'snapshot': _snapshot()})
    context = ToolContext(runtime.config, state, None, runtime_capabilities={'system_snapshot_provider': lambda: {}})
    tool = RuntimeSnapshotLookupTool()
    result = tool.execute(tool.validate({'snapshot_sequence': event.sequence, 'pointer': '/terminal_workers/0/error'}), context)
    assert result.output['value'] == 'EXACT_BACKUP_FAILURE'
    assert result.output['snapshot_hash'] == event.hash
    other = runtime.create_or_load_session()
    other_context = ToolContext(runtime.config, other, None, runtime_capabilities=context.runtime_capabilities)
    with pytest.raises(ToolValidationError):
        tool.execute({'snapshot_sequence': event.sequence, 'pointer': ''}, other_context)
    with pytest.raises(ToolValidationError):
        tool.validate({'snapshot_sequence': True, 'pointer': ''})


def test_terminal_pending_controls_are_counted_and_exactly_recoverable():
    snapshot = _snapshot()
    instruction = 'Do not publish the backup if this task is resumed.'
    snapshot['worker_controls'].append({'worker_id':'finished', 'session_id':'old-session',
        'pending_controls':[{'message':instruction}], 'delivery_receipts':[{'status':'pending'}]})
    view = snapshot_prompt_view(snapshot, 42)
    assert view['terminal_control_details']['pending_controls_count'] == 1
    assert view['worker_controls'] == [snapshot['worker_controls'][0]]
    assert snapshot_value(snapshot, '/worker_controls/1/pending_controls/0/message') == instruction
    assert 'before historical claims or resuming work' in view['detail_access']['notice']


def test_large_terminal_history_does_not_enter_greeting_prompt(make_config):
    config = make_config(model__context_limit=16000, tools__enabled=['runtime_snapshot_lookup', 'orchestration_control'],
                         tools__staged_discovery=True)
    client = _ImmediateClient('Hello!')
    runtime = AgentRuntime(config, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        for i in range(40):
            worker = service.workers.create(('OLD_TASK_DETAIL_' + str(i) + ' ') * 200)
            service.workers.store.transition(worker.worker_id, 'completed', result='Exact result ' + str(i))
        assert service.orchestrator_message('Please say hello.')['answer'] == 'Hello!'
        assert len(client.requests) == 1
        prompt = client.requests[0]['prompt']
        assert 'OLD_TASK_DETAIL_' not in prompt
        assert 'runtime_snapshot_lookup' in prompt
        assert 'plan_spec' not in prompt
        state = service._load_orchestrator_state()
        snapshot = next(e.payload['snapshot'] for e in reversed(runtime.history.read_history(state.session_id))
                        if e.event_type == 'runtime_system_snapshot')
        assert len(snapshot['terminal_workers']) == 40
        assert 'OLD_TASK_DETAIL_0' in snapshot['terminal_workers'][0]['objective']
    finally:
        service.close()


def test_verifier_can_fetch_current_terminal_failure_before_rejecting_claim(make_config):
    holder = {}
    def inspect(payload):
        assert 'EXACT_BACKUP_FAILURE' not in payload['prompt']
        state = holder['state']
        event = next(e for e in reversed(holder['runtime'].history.read_history(state.session_id))
                     if e.event_type == 'runtime_system_snapshot')
        return json.dumps({'complete': False, 'reason': 'Read the exact backup result.', 'remaining_work': [],
            'evidence_requests': [{'source_kind': 'foreground_history',
                'source_id': f'{state.session_id}:{event.sequence}-{event.sequence}',
                'purpose': 'Verify the claimed success.', 'literal_query': ''}]})
    def reject(payload):
        assert 'EXACT_BACKUP_FAILURE' in payload['prompt']
        return json.dumps({'complete': False, 'reason': 'The backup failed.',
                           'remaining_work': ['Correct the success claim.'], 'evidence_requests': []})
    client = FakeModelClient([inspect, reject])
    runtime = AgentRuntime(make_config(model__context_limit=32000,
        tools__enabled=['runtime_snapshot_lookup']), model_client=client)
    state = runtime.create_or_load_session()
    holder.update(runtime=runtime, state=state)
    runtime._record_message(state, Message(role='user', content='Did the backup pass?', created_at='now'))
    runtime.history.record_event(state, 'turn_started', {'turn_index': 1, 'user_text': 'Did the backup pass?'})
    runtime.bind_tool_runtime_capability(state.session_id, 'system_snapshot_provider', _snapshot)
    action = replace(_completed_action(), assistant_message='The backup passed.')
    result = runtime._evaluate_completion(state, original_request='Did the backup pass?',
                                          selected_action=action, tool_results=[])
    assert result['complete'] is False and len(client.requests) == 2


def test_action_history_defers_only_prior_machine_actions_and_preserves_delivered_progress(make_config):
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    authority = '  Do not publish.\nKeep the backup private.  '
    runtime._record_message(state, Message(role='user', content=authority, created_at='old',
                                         metadata={'internal_action': True}))
    for i in range(40):
        runtime._record_message(state, Message(role='assistant', content='DRAFT_' + str(i),
            created_at='old', metadata={'internal_action': True}))
    progress = runtime.history.record_event(state, 'assistant_progress',
        {'action_index': 1, 'assistant_text': 'The check started; the result is not known yet.'})
    runtime._record_message(state, Message(role='tool', content='EXACT_TOOL_FAILURE', created_at='old'))
    runtime._record_message(state, Message(role='assistant', content='Visible final answer.', created_at='old'))
    runtime._record_message(state, Message(role='user', content='Check again, without publishing.', created_at='now'))
    turn = runtime.history.record_event(state, 'turn_started', {'turn_index': 2, 'user_text': 'Check again, without publishing.'})
    runtime._record_message(state, Message(role='assistant', content='CURRENT_ACTION', created_at='now',
                                         metadata={'internal_action': True}))
    original = runtime.history.read_authoritative_messages(state.session_id)
    view, notice = foreground_action_history(original, [progress], turn.sequence)
    assert [m.content for m in view] == [authority, progress.payload['assistant_text'],
        'EXACT_TOOL_FAILURE', 'Visible final answer.', 'Check again, without publishing.', 'CURRENT_ACTION']
    assert '40 prior-turn' in notice and 'history_window' in notice
    assert runtime.history.read_authoritative_messages(state.session_id) == original


def test_prior_machine_actions_do_not_force_greeting_compaction(make_config):
    config = make_config(model__context_limit=16000, tools__enabled=['runtime_snapshot_lookup',
        'history_window', 'orchestration_control'], tools__staged_discovery=True)
    client = _ImmediateClient('Hello!')
    runtime = AgentRuntime(config, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        state = service._load_orchestrator_state()
        runtime._record_message(state, Message(role='user', content='Keep my files private.', created_at='old'))
        for i in range(40):
            runtime._record_message(state, Message(role='assistant', content=('OLD_DRAFT ' * 200),
                                                  created_at='old', metadata={'internal_action': True}))
        runtime.history.record_event(state, 'assistant_progress', {'action_index': 1, 'assistant_text': 'I have started checking.'})
        assert service.orchestrator_message('Please say hello.')['answer'] == 'Hello!'
        assert len(client.requests) == 1
        prompt = client.requests[0]['prompt']
        assert 'Keep my files private.' in prompt and 'I have started checking.' in prompt
        assert 'OLD_DRAFT' not in prompt and 'prior-turn internal assistant action' in prompt
    finally:
        service.close()


def test_identical_user_repetitions_preserve_exact_text_order_count_and_current_turn():
    text = '  Do not publish.\nKeep this exact.  '
    messages = [Message(role='user', content=text, created_at='now',
        metadata={'source_message_event_sequence': n}) for n in (2, 5, 9, 13)]
    view = fold_identical_prior_user_messages(messages, 14)
    assert len(view) == 2
    assert view[0].content == view[1].content == text
    assert view[0].metadata['verbatim_repetition_sequences'] == [2, 5, 9]
    assert view[1] is messages[-1]
    assert [x['sequence'] for x in view[0].metadata['source_event_references']] == [2, 5, 9]
