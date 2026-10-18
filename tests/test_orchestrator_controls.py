"""Finalized ordinary messages must reach an active planner before its next effect."""
import threading
import time
import pytest
from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from tests.test_agent_action_loop import FakeModelClient, _action


def test_ordinary_correction_invalidates_inflight_planner_action(make_config):
    entered, release = threading.Event(), threading.Event()
    exact = '\n  Cancel the calculation; report that it did not execute.  \n'
    prompts = []
    def stale(payload):
        entered.set()
        assert release.wait(8)
        return _action(message='I am checking that calculation.', tool_calls=[('calculator', {'expression':'6 * 7'})], continue_loop=True)
    def reconciled(payload):
        prompts.append(payload['prompt'])
        assert exact in payload['prompt']
        return _action(message='The calculation was canceled before execution.')
    client = FakeModelClient([
        stale, reconciled,
    ])
    c = make_config(model__context_limit=32000, tools__enabled=['calculator'], runtime__completion_evaluation_enabled=False)
    runtime = AgentRuntime(c, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    results, errors = {}, []
    def call(key, text):
        try: results[key] = service.orchestrator_message(text)
        except Exception as exc: errors.append(exc)
    a=threading.Thread(target=call,args=('first','Calculate six times seven.'))
    b=threading.Thread(target=call,args=('correction',exact))
    try:
        a.start(); assert entered.wait(5); b.start()
        state=service._load_orchestrator_state()
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            controls=runtime.history.list_pending_control_messages(state.session_id)
            if any(x["message"] == exact for x in controls):break
            time.sleep(.01)
        assert exact in [x['message'] for x in controls], 'Final message waited outside the active planner control inbox'
        release.set();a.join(8);b.join(8)
        assert not a.is_alive() and not b.is_alive()
        assert not errors
        assert results['correction']['answer']=='The calculation was canceled before execution.'
        events=runtime.history.read_history(state.session_id)
        assert not [e for e in events if e.event_type=='tool_called']
        assert any(e.event_type=='agent_action_rejected' and 'New user control arrived' in e.payload.get('reason','') for e in events)
        assert len(client.requests)==2, 'The correction must not be executed as a second new task'
        state=service._load_orchestrator_state()
        assert [m.content for m in state.messages if m.role=='user']==['Calculate six times seven.',exact]
    finally:
        release.set();a.join(10)
        if b.ident:b.join(10)
        service.close()


def test_received_controls_keep_order_and_later_turn_can_supersede_them(make_config):
    first = 'Hold all calculations.'
    second = 'Actually only hold multiplication; division may continue.'
    third = 'The hold is lifted. Calculate six times seven now.'
    entered, release = threading.Event(), threading.Event()
    def stale(payload):
        entered.set(); assert release.wait(8)
        return _action(message='I am checking that calculation.', tool_calls=[('calculator', {'expression':'6 * 7'})], continue_loop=True)
    def reconcile(payload):
        prompt=payload['prompt']
        assert prompt.index(first) < prompt.index(second)
        return _action(message='Multiplication is on hold; no calculation ran.')
    def newer(payload):
        prompt=payload['prompt']
        assert first in prompt and second in prompt and third in prompt
        # Older controls are now earlier conversation, not falsely labelled as
        # interventions later than the new explicit user request.
        assert '[USER INTERVENTION' not in prompt
        return _action(message='I am checking that calculation.', tool_calls=[('calculator', {'expression':'6 * 7'})], continue_loop=True)
    client=FakeModelClient([
         stale, reconcile,
         newer, _action(message='The result is forty-two.'),
    ])
    runtime=AgentRuntime(make_config(model__context_limit=32000,tools__enabled=['calculator'],runtime__completion_evaluation_enabled=False),model_client=client)
    service=CommunicationService(runtime,orchestrator_runtime=runtime)
    results,errors=[],[]
    def call(text):
        try:results.append(service.orchestrator_message(text))
        except Exception as exc:errors.append(exc)
    threads=[threading.Thread(target=call,args=(x,)) for x in ('Calculate six times seven.',first,second)]
    try:
        threads[0].start();assert entered.wait(5)
        session=service._load_orchestrator_state().session_id
        for t,text in zip(threads[1:],(first,second)):
            t.start();deadline=time.monotonic()+2
            while time.monotonic()<deadline:
                if any(x['message']==text for x in runtime.history.list_pending_control_messages(session)):break
                time.sleep(.01)
            else:pytest.fail('Correction not delivered while planner was active')
        release.set()
        for t in threads:t.join(10)
        assert not errors and not any(t.is_alive() for t in threads)
        assert len(client.requests)==2
        state=service._load_orchestrator_state()
        assert [x.content for x in state.messages if x.role=='user']==['Calculate six times seven.',first,second]
        assert service.orchestrator_message(third)['answer']=='The result is forty-two.'
        events=runtime.history.read_history(session)
        assert len([e for e in events if e.event_type=='tool_called'])==1
        assert len(client.requests)==4
    finally:
        release.set()
        for t in threads:
            if t.ident:t.join(10)
        service.close()


def test_final_message_survives_action_failure_and_reconciles_after_service_recreation(make_config):
    c=make_config(model__context_limit=32000,tools__enabled=['calculator'],runtime__completion_evaluation_enabled=False)
    exact='\n  Calculate six times seven.  \n'
    correction='Cancel that calculation; do not run it.'
    def fail(payload):raise RuntimeError('injected connection loss')
    runtime=AgentRuntime(c,model_client=FakeModelClient([fail] * 20))
    service=CommunicationService(runtime,orchestrator_runtime=runtime)
    try:
        with pytest.raises(RuntimeError,match='injected connection loss'):
            service.orchestrator_message(exact)
        session=service._load_orchestrator_state().session_id
        assert [x['message'] for x in service.runtime.history.list_pending_control_messages(session)]==[exact]
    finally:service.close()
    def reconcile(payload):
        assert exact in payload['prompt'] and correction in payload['prompt']
        return _action(message='The pending calculation was canceled without running.')
    client=FakeModelClient([reconcile])
    runtime=AgentRuntime(c,model_client=client)
    service=CommunicationService(runtime,orchestrator_runtime=runtime)
    try:
        assert service.orchestrator_message(correction)['answer']=='The pending calculation was canceled without running.'
        assert runtime.history.list_pending_control_messages(session)==[]
        events=runtime.history.read_history(session)
        assert not any(e.event_type=='tool_called' for e in events)
        users=[m.content for m in service._load_orchestrator_state().messages if m.role=='user']
        assert users==[exact,correction]
    finally:service.close()


def test_final_message_is_not_duplicated_after_action_connection_failure(make_config):
    c=make_config(model__context_limit=32000,model__max_retries=0,tools__enabled=['calculator'],runtime__completion_evaluation_enabled=False)
    exact='Calculate six times seven with the calculator.'
    correction='Cancel that calculation without running it.'
    def fail(payload):raise RuntimeError('injected action connection loss')
    runtime=AgentRuntime(c,model_client=FakeModelClient([fail]))
    service=CommunicationService(runtime,orchestrator_runtime=runtime)
    try:
        with pytest.raises(RuntimeError,match='injected action connection loss'):
            service.orchestrator_message(exact)
    finally:service.close()
    def reconcile(payload):
        assert exact in payload['prompt'] and correction in payload['prompt']
        return _action(message='Canceled before calculation.')
    runtime=AgentRuntime(c,model_client=FakeModelClient([reconcile]))
    service=CommunicationService(runtime,orchestrator_runtime=runtime)
    try:
        assert service.orchestrator_message(correction)['answer']=='Canceled before calculation.'
        state=service._load_orchestrator_state()
        assert [m.content for m in state.messages if m.role=='user']==[exact,correction]
        assert runtime.history.list_pending_control_messages(state.session_id)==[]
    finally:service.close()


def test_word_count_repair_receives_pending_authoritative_correction(make_config):
    import json
    from swaag.types import Message
    exact='Do not enforce that word count; preserve the exact reply as written.'
    def repair(payload):
        assert exact in payload['prompt']
        assert 'supersede conflicting earlier instructions' in payload['prompt']
        return json.dumps({'reason':'The later correction removed the word-count requirement.','words':None})
    runtime=AgentRuntime(make_config(model__context_limit=32000),model_client=FakeModelClient([repair]))
    state=runtime.create_or_load_session()
    runtime._record_message(state,Message(role='user',content='Reply in exactly twenty-five words.',created_at='2026-10-13T00:00:00Z'))
    runtime.history.enqueue_control_message(state.session_id,exact,source='orchestrator_user')
    assert runtime._repair_exact_word_count(state,original_request=state.messages[-1].content,draft='Exact current response.',target_word_count=25)==('Exact current response.',None)


def test_response_word_count_repair_keeps_authoritative_user_messages_without_bulky_tool_history(make_config):
    """A hallucinated word count must not replay enormous unrelated tool output."""
    import json
    from swaag.types import Message
    original = "Can you hear me? Please say hello."
    draft = "Yes, I can hear you. Hello."
    earlier_requirement = "Answer the earlier calculation in exactly 25 words."
    now_removed = "That was for the earlier calculation only; return to normal answers."
    tool_payload = "UNRELATED_PRIVATE_TOOL_OUTPUT_NOT_RESPONSE_AUTHORITY:" + ("Q" * 110_000)

    def verify(payload):
        prompt = payload["prompt"]
        assert original in prompt
        assert earlier_requirement in prompt
        assert now_removed in prompt
        assert "Yes, I can hear you. Hello." in prompt
        assert tool_payload not in prompt
        assert "UNRELATED_PRIVATE_TOOL_OUTPUT_NOT_RESPONSE_AUTHORITY" not in prompt
        assert len(prompt) < 25_000, "The scoped checker must not grow with historical tool payloads"
        return json.dumps({
            "reason": "No active instruction requires exactly thirteen words for this new greeting.",
            "words": None,
        })

    client = FakeModelClient([verify])
    runtime = AgentRuntime(make_config(model__context_limit=262144), model_client=client)
    state = runtime.create_or_load_session()
    for text in (earlier_requirement, now_removed):
        runtime._record_message(
            state, Message(role="user", content=text, created_at="2026-10-13T00:00:00Z")
        )
    for index in range(3):
        runtime._record_message(
            state, Message(
                role="tool", name="some_old_tool", content=tool_payload + str(index),
                created_at="2026-10-13T00:00:00Z"
            ),
        )
    runtime._record_message(
        state, Message(
            role="assistant", content="The earlier calculation is complete.",
            created_at="2026-10-13T00:00:00Z"
        ),
    )
    runtime._record_message(
        state, Message(role="user", content=original, created_at="2026-10-18T12:16:29Z")
    )
    proposed = {
        "assistant_message": draft,
        "response_constraints": {"exact_word_count": 13},
    }
    result = runtime._repair_declared_response_constraints(
        state, original_request=original, payload=proposed,
    )
    assert result["assistant_message"] == draft
    assert result["response_constraints"] == {"exact_word_count": None}
    assert len(client.requests) == 1
    assert client.requests[0]["contract"] == "response_word_count_repair_13"
    assert sum(1 for message in state.messages if message.role == "tool") == 3
    assert all(tool_payload in message.content for message in state.messages if message.role == "tool")


def test_response_word_count_repair_preserves_a_real_current_count(make_config):
    """Scoping out tool results must not disable an exact numeric user instruction."""
    import json
    from swaag.types import Message
    original = "Reply in exactly four words."
    def verify(payload):
        assert original in payload["prompt"]
        return json.dumps({
            "reason": "The current user explicitly requested four words.",
            "words": {f"word_{i:03d}": word for i, word in
                      enumerate(("One", "two", "three", "four"), start=1)},
        })

    runtime = AgentRuntime(make_config(model__context_limit=32000),
                           model_client=FakeModelClient([verify]))
    state = runtime.create_or_load_session()
    runtime._record_message(
        state, Message(role="user", content=original, created_at="2026-10-18T12:16:29Z")
    )
    result = runtime._repair_declared_response_constraints(
        state, original_request=original,
        payload={"assistant_message": "One two.", "response_constraints": {"exact_word_count": 4}},
    )
    assert result["assistant_message"] == "One two three four"
    assert result["response_constraints"] == {"exact_word_count": 4}


@pytest.mark.parametrize('responsibilities',[1,6])
def test_completion_checker_receives_latest_user_correction(make_config,responsibilities):
    import json
    from swaag.types import Message
    from tests.test_completion_evaluation import _completed_action
    original='Calculate six times seven.'
    correction='Cancel that calculation; report that no tool ran.'
    def evaluate(payload):
        assert correction in payload['prompt']
        assert 'Latest user message, verbatim' in payload['prompt']
        result={'complete':True}
        if responsibilities!=1:result.update(reason='The corrected objective is satisfied.',remaining_work=[])
        return json.dumps(result)
    runtime=AgentRuntime(make_config(model__context_limit=32000,model__max_semantic_responsibilities_per_call=responsibilities),model_client=FakeModelClient([evaluate]))
    state=runtime.create_or_load_session()
    runtime._record_message(state,Message(role='user',content=original,created_at='2026-10-13T00:00:00Z'))
    runtime._record_message(state,Message(role='user',content=correction,created_at='2026-10-13T00:00:01Z'))
    result=runtime._evaluate_completion(state,original_request=original,selected_action=_completed_action(),tool_results=[])
    assert result['complete'] is True


def test_completion_partition_stays_at_run_start_when_correction_becomes_user_message(make_config):
    import json
    from swaag.types import Message
    from tests.test_completion_evaluation import _completed_action
    original='Finish the authorized task.'
    first='Preserve the exact filename Prüfbericht — Öl.txt.'
    second='Cancel the pending calculation.'
    excluded='INFERENCE_BOOKKEEPING_IS_NOT_PRIOR_TURN_EVIDENCE_419'
    def evaluate(payload):
        assert first in payload['prompt'] and second in payload['prompt']
        assert excluded not in payload['prompt']
        result={'complete':True,'reason':'The corrected objective is complete.','remaining_work':[]}
        if 'evidence_requests' in payload['json_schema']['properties']:result['evidence_requests']=[]
        return json.dumps(result)
    runtime=AgentRuntime(make_config(model__context_limit=32000),model_client=FakeModelClient([evaluate]))
    state=runtime.create_or_load_session()
    runtime._record_message(state,Message(role='user',content=original,created_at='2026-10-13T00:00:00Z'))
    runtime.history.record_event(state,'turn_started',{'turn_index':1,'user_text':original,'execution_loop':'model_tool_loop'})
    runtime.history.record_event(state,'state_rebuilt',{'session_id':state.session_id,'event_count':state.event_count,'diagnostic':excluded})
    runtime._record_message(state,Message(role='user',content=first,created_at='2026-10-13T00:00:01Z'))
    runtime._record_message(state,Message(role='user',content=second,created_at='2026-10-13T00:00:02Z'))
    assert runtime._evaluate_completion(state,original_request=original,selected_action=_completed_action(),tool_results=[])['complete'] is True


@pytest.mark.parametrize("executor_workers", [None, 1])
def test_protocol_correction_is_received_before_request_semaphore(make_config, executor_workers):
    import asyncio
    entered, release = threading.Event(), threading.Event()
    correction = 'Cancel the calculation before any tool runs.'
    def stale(payload):
        entered.set()
        assert release.wait(8)
        return _action(message='I am checking that calculation.', tool_calls=[('calculator', {'expression': '6 * 7'})], continue_loop=True)
    def reconciled(payload):
        assert correction in payload['prompt']
        return _action(message='The calculation was canceled before execution.')
    client = FakeModelClient([
         stale, reconciled,
    ])
    runtime = AgentRuntime(make_config(model__context_limit=32000,
        tools__enabled=['calculator'], runtime__completion_evaluation_enabled=False), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime, max_concurrency=1)
    async def exercise():
        if executor_workers is not None:
            from concurrent.futures import ThreadPoolExecutor
            asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=executor_workers))
        first = asyncio.create_task(service._dispatch_json_line_request({
            'op': 'orchestrator.message', 'message': 'Calculate six times seven.'}))
        second = None
        try:
            started_deadline = time.monotonic() + 5
            while not entered.is_set() and time.monotonic() < started_deadline:
                await asyncio.sleep(.01)
            assert entered.is_set()
            second = asyncio.create_task(service._dispatch_json_line_request({
                'op': 'orchestrator.message', 'message': correction}))
            state = service._load_orchestrator_state()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                pending = runtime.history.list_pending_control_messages(state.session_id)
                if any(item['message'] == correction for item in pending):
                    break
                await asyncio.sleep(.01)
            assert any(item['message'] == correction for item in pending), (
                'The protocol semaphore blocked finalized control receipt')
            release.set()
            results = await asyncio.wait_for(asyncio.gather(first, second), 8)
            assert results[0]['answer'] == results[1]['answer'] == 'The calculation was canceled before execution.'
            events = runtime.history.read_history(state.session_id)
            assert not any(event.event_type == 'tool_called' for event in events)
            assert any(event.event_type == 'agent_action_rejected' for event in events)
            assert len(client.requests) == 2
        finally:
            release.set()
            await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)
    try:
        asyncio.run(exercise())
    finally:
        service.close()


def test_canceled_transport_keeps_bounded_ingress_until_receipt_finishes(make_config, monkeypatch):
    import asyncio
    from swaag.capacity import QueueCapacityError
    runtime = AgentRuntime(make_config(runtime__max_pending_controls=1), model_client=FakeModelClient([]))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    entered, release = threading.Event(), threading.Event()
    finish_receipt = threading.Event()
    receive = service._receive_orchestrator_message
    exact = 'Do not publish until I have reviewed the result.'
    def blocked(message):
        entered.set()
        assert release.wait(8)
        receipt = receive(message)
        assert finish_receipt.wait(8)
        return receipt
    monkeypatch.setattr(service, '_receive_orchestrator_message', blocked)
    async def exercise():
        request = asyncio.create_task(service._receive_protocol_orchestrator_message(exact))
        try:
            deadline = time.monotonic() + 3
            while not entered.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(.01)
            assert entered.is_set()
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            with pytest.raises(QueueCapacityError, match='ingress capacity'):
                await service._receive_protocol_orchestrator_message('Another message')
            release.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                state = service._load_orchestrator_state()
                pending = runtime.history.list_pending_control_messages(state.session_id)
                if pending:
                    break
                await asyncio.sleep(.01)
            assert [item['message'] for item in pending] == [exact]
            # Storage visibility precedes completion of the receipt-producing
            # job. Capacity must stay occupied until that job actually returns.
            assert not service._orchestrator_ingress_slots.acquire(blocking=False)
            finish_receipt.set()
            deadline = time.monotonic() + 3
            while not service._orchestrator_ingress_slots.acquire(blocking=False):
                assert time.monotonic() < deadline, 'Receipt job did not release ingress capacity'
                await asyncio.sleep(.01)
            service._orchestrator_ingress_slots.release()
        finally:
            release.set(); finish_receipt.set()
    try:
        asyncio.run(exercise())
    finally:
        service.close()
