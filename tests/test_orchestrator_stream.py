from __future__ import annotations
import asyncio
import json
import threading
import time
import pytest
from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from tests.test_agent_action_loop import FakeModelClient, _action
from tests.test_preemption import _ImmediateClient


def _service(make_config, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def finish(payload):
        entered.set()
        assert release.wait(8), 'test did not release final generation'
        return _action(message='The value was verified.')
    client = FakeModelClient([
        _action(message='I am checking the value.', tool_calls=[('echo', {'text':'verified'})], continue_loop=True),
        finish,
    ])
    runtime = AgentRuntime(make_config(model__context_limit=32000, communication__poll_seconds=0.01), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    return service, client, entered, release


def test_slow_speech_does_not_hold_tool_or_final_answer_and_reconnect_does_not_reexecute(make_config, monkeypatch):
    service, client, entered, release = _service(make_config, monkeypatch)
    try:
        stream = service.iter_orchestrator_message('Check the value.', control_id='stream_repeat')
        assert next(stream)['event'] == 'accepted'
        progress = next(stream)
        assert progress['event'] == 'assistant_progress'
        assert progress['assistant_text'] == 'I am checking the value.'
        assert entered.wait(2)
        history = service.orchestrator_runtime.history.read_history(progress['session_id'])
        receipt = next(e for e in history if e.event_type == 'tool_result')
        assert receipt.sequence < progress['sequence']
        # Hold the consumer at speech delivery; the real tool/model thread proceeds.
        release.set()
        deadline = time.monotonic()+8
        while time.monotonic()<deadline:
            state = service._load_orchestrator_state()
            if state.messages and state.messages[-1].content == 'The value was verified.':
                break
            time.sleep(0.01)
        else:
            pytest.fail('speech consumer blocked execution')
        stream.close()
        replay = list(service.iter_orchestrator_message('Check the value.', control_id='stream_repeat', after_sequence=progress['sequence']))
        assert [item['event'] for item in replay] == ['accepted','completed']
        assert replay[-1]['result']['answer'] == 'The value was verified.'
        assert len(client.requests) == 2
        with pytest.raises(ValueError, match='different finalized message'):
            list(service.iter_orchestrator_message('Execute another command.', control_id='stream_repeat'))
    finally:
        release.set(); service.close()


def test_jsonl_disconnect_keeps_accepted_work_and_replays_progress(make_config, monkeypatch):
    service, client, entered, release = _service(make_config, monkeypatch)
    async def exercise():
        server = await asyncio.start_server(service.handle_client, '127.0.0.1', 0)
        port = server.sockets[0].getsockname()[1]
        request = {'op':'orchestrator.message','message':'Check the value.','stream':True,'control_id':'disconnect_test'}
        try:
            reader, writer = await asyncio.open_connection('127.0.0.1', port)
            writer.write((json.dumps(request)+'\n').encode()); await writer.drain()
            accepted = json.loads(await asyncio.wait_for(reader.readline(), 5))
            assert accepted['event'] == 'accepted'
            progress = json.loads(await asyncio.wait_for(reader.readline(), 5))
            assert progress['event'] == 'assistant_progress'
            writer.close(); await writer.wait_closed()
            release.set()
            reader2, writer2 = await asyncio.open_connection('127.0.0.1', port)
            writer2.write((json.dumps({**request,'after_sequence':progress['sequence']})+'\n').encode()); await writer2.drain()
            assert json.loads(await asyncio.wait_for(reader2.readline(), 5))['event'] == 'accepted'
            final = json.loads(await asyncio.wait_for(reader2.readline(), 8))
            assert final['event'] == 'completed'
            assert final['result']['answer'] == 'The value was verified.'
            writer2.close(); await writer2.wait_closed()
            assert len(client.requests) == 2
        finally:
            release.set(); server.close(); await server.wait_closed()
    try:
        asyncio.run(exercise())
    finally:
        release.set(); service.close()


def test_stream_skips_prior_turns_and_validates_before_receipt(make_config):
    runtime = AgentRuntime(make_config(model__context_limit=32000, communication__poll_seconds=0.01), model_client=_ImmediateClient('Ready.'))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        service.orchestrator_message('Hello.')
        events = list(service.iter_orchestrator_message('Are you there?', control_id='second'))
        assert [e['event'] for e in events] == ['accepted','completed']
        before = len(runtime.history.read_history(events[-1]['session_id']))
        for kw in ({'control_id':'../escape'}, {'after_sequence':True}, {'after_sequence':-1}):
            with pytest.raises(ValueError):
                list(service.iter_orchestrator_message('Invalid', **kw))
        assert len(runtime.history.read_history(events[-1]['session_id'])) == before
    finally:
        service.close()


def test_failed_dispatch_never_publishes_success_draft(make_config):
    client = FakeModelClient([
        _action(message='I have sent the instruction.', tool_calls=[('agent_control', {'session_ref':'missing_target','message':'Hold.'})], continue_loop=True),
        _action(message='The target was not found, so the instruction was not queued.'),
    ])
    runtime = AgentRuntime(make_config(model__context_limit=32000,
        tools__enabled=['agent_control'],tools__allow_side_effect_tools=True,
        runtime__completion_evaluation_enabled=False,communication__poll_seconds=0.01),model_client=client)
    service = CommunicationService(runtime,orchestrator_runtime=runtime)
    try:
        events = list(service.iter_orchestrator_message('Tell the missing target to hold.',control_id='failed_dispatch'))
        assert [event['event'] for event in events] == ['accepted','completed']
        assert events[-1]['result']['answer'].startswith('The target was not found')
        history = runtime.history.read_history(events[-1]['session_id'])
        assert any(event.event_type == 'tool_error' for event in history)
        assert not any(event.event_type == 'assistant_progress' for event in history)
    finally:
        service.close()


def test_initial_empty_foreground_action_is_repaired_before_tool_execution(make_config):
    client = FakeModelClient([
        _action(message='', tool_calls=[('echo', {'text':'unaccepted draft'})], continue_loop=True),
        _action(message='I am checking that now.', tool_calls=[('echo', {'text':'verified'})], continue_loop=True),
        _action(message='The value was verified.'),
    ])
    runtime = AgentRuntime(make_config(model__context_limit=32000,
        runtime__completion_evaluation_enabled=False, communication__poll_seconds=0.01),
        model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        events = list(service.iter_orchestrator_message('Check the value.', control_id='required_ack'))
        progress = [e for e in events if e['event']=='assistant_progress']
        assert len(progress)==1 and progress[0]['assistant_text']=='I am checking that now.'
        history = runtime.history.read_history(events[-1]['session_id'])
        calls = [e for e in history if e.event_type=='tool_called']
        assert len(calls)==1 and calls[0].payload['tool_input']=={'text':'verified'}
        assert any(e.event_type=='agent_action_rejected' and
                   'initial foreground action' in e.payload.get('reason','') for e in history)
        receipt = next(e for e in history if e.event_type=='tool_result')
        assert receipt.sequence < progress[0]['sequence']
        assert len(client.requests)==3
        assert 'initial foreground action' in client.requests[0]['prompt']
        assert 'initial_foreground_acknowledgement' in str([
            e.payload for e in history if e.event_type=='context_compiled'])
    finally:
        service.close()


def test_processed_receipt_resumes_after_failure_without_repeating_effect(make_config):
    config = make_config(model__context_limit=32000, model__max_retries=0,
                         communication__poll_seconds=0.01)
    def fail(payload):
        raise RuntimeError('lost after committed effect')
    runtime = AgentRuntime(config, model_client=FakeModelClient([
        _action(message='I am checking the value.', tool_calls=[('echo', {'text':'verified'})], continue_loop=True),
        fail,
    ]))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        with pytest.raises(RuntimeError, match='lost after committed effect'):
            list(service.iter_orchestrator_message('Check the value.', control_id='resume_effect'))
        sid = service._load_orchestrator_state().session_id
        assert not runtime.history.list_pending_control_messages(sid)
    finally:
        service.close()
    def finish(payload):
        assert 'verified' in payload['prompt']
        return _action(message='The value was verified.')
    client = FakeModelClient([finish])
    runtime = AgentRuntime(config, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        events = list(service.iter_orchestrator_message('Check the value.', control_id='resume_effect'))
        assert events[-1]['result']['answer'] == 'The value was verified.'
        assert events[-1]['sequence'] is not None
        replay = list(service.iter_orchestrator_message('Check the value.', control_id='resume_effect'))
        assert replay[-1]['result'] == events[-1]['result']
        assert len(client.requests) == 1
        history = runtime.history.read_history(sid)
        assert len([e for e in history if e.event_type == 'tool_called']) == 1
        assert len([e for e in history if e.event_type == 'control_message_processed']) == 1
        assert [m.content for m in runtime.history.read_authoritative_messages(sid) if m.role == 'user'] == ['Check the value.']
    finally:
        service.close()


@pytest.mark.parametrize('ending', ['canceled', 'superseded'])
def test_processed_retry_never_revives_canceled_or_superseded_turn(make_config, ending):
    from swaag.preemption import RunCancellationRequested
    def fail(payload):
        if ending == 'canceled':
            raise RunCancellationRequested('explicit stop')
        raise RuntimeError('lost after committed effect')
    client = FakeModelClient([
        _action(message='I am checking the value.', tool_calls=[('echo', {'text':'verified'})], continue_loop=True),
        fail,
        _action(message='The new request is answered.'),
    ])
    runtime = AgentRuntime(make_config(model__context_limit=32000, model__max_retries=0,
        communication__poll_seconds=0.01), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        with pytest.raises(RuntimeError):
            list(service.iter_orchestrator_message('Check the value.', control_id='stopped_receipt'))
        if ending == 'superseded':
            service.orchestrator_message('Forget that request. Just answer this one.')
        before = len(client.requests)
        with pytest.raises(RuntimeError, match=ending):
            list(service.iter_orchestrator_message('Check the value.', control_id='stopped_receipt'))
        assert len(client.requests) == before
        events = runtime.history.read_history(service._load_orchestrator_state().session_id)
        assert len([e for e in events if e.event_type == 'tool_called']) == 1
    finally:
        service.close()


def test_resumed_receipt_reconciles_newer_pending_correction(make_config):
    def fail(payload):
        raise RuntimeError('lost after committed effect')
    def finish(payload):
        assert 'Do not check anything else; report only the saved result.' in payload['prompt']
        return _action(message='The saved result is verified; no further checks ran.')
    client = FakeModelClient([
        _action(message='I am checking the value.', tool_calls=[('echo', {'text':'verified'})], continue_loop=True),
        fail, finish,
    ])
    runtime = AgentRuntime(make_config(model__context_limit=32000, model__max_retries=0,
        communication__poll_seconds=0.01), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    try:
        with pytest.raises(RuntimeError, match='lost after committed effect'):
            list(service.iter_orchestrator_message('Check the value.', control_id='interrupted'))
        correction = service._receive_orchestrator_message(
            'Do not check anything else; report only the saved result.', control_id='new_correction')
        resumed = list(service.iter_orchestrator_message('Check the value.', control_id='interrupted'))
        assert resumed[-1]['result']['answer'] == 'The saved result is verified; no further checks ran.'
        assert service._process_orchestrator_receipt(correction) == resumed[-1]['result']
        history = runtime.history.read_history(resumed[-1]['session_id'])
        assert len([e for e in history if e.event_type == 'tool_called']) == 1
        assert len(client.requests) == 3
    finally:
        service.close()
