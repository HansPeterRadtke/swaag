"""Fast reply is visibly committed before the real orchestrator executes a tool."""
import json
import threading

import pytest
from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from tests.test_agent_action_loop import FakeModelClient, _action


def fast(mode, reply):
    return json.dumps({"mode": mode, "reply": reply})


def setup(make_config, responses):
    cfg = make_config(model__context_limit=32000,
                      runtime__completion_evaluation_enabled=False,
                      communication__poll_seconds=0.01,
                      communication__fast_initial_reply=True,
                      communication__early_tool_acknowledgement=True)
    client = FakeModelClient(responses)
    runtime = AgentRuntime(cfg, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    return service, runtime, client


def test_simple_direct_answer_uses_only_one_llm_and_commits_one_turn(make_config):
    service, runtime, client = setup(make_config, [fast("answer", "I am SWAAG.")])
    try:
        stream = list(service.iter_orchestrator_message("Who are you?", control_id="fast_simple"))
        assert [row["event"] for row in stream] == ["accepted", "completed"]
        assert stream[-1]["result"]["answer"] == "I am SWAAG."
        sid = stream[0]["session_id"]
        events = runtime.history.read_history(sid)
        assert len(client.requests) == 1
        assert [r["contract"] for r in client.requests] == ["fast_foreground_response"]
        assert sum(e.event_type == "turn_started" for e in events) == 1
        assert sum(e.event_type == "turn_finished" for e in events) == 1
        assert sum(e.event_type == "control_message_processed" for e in events) == 1
        assert not any(e.event_type == "tool_called" for e in events)
        assert not runtime.history.list_pending_control_messages(sid)
        again = list(service.iter_orchestrator_message("Who are you?", control_id="fast_simple"))
        assert again[-1]["result"]["answer"] == "I am SWAAG."
        assert len(client.requests) == 1
    finally:
        service.close()


def test_delegate_speaks_before_deep_model_or_tool_and_keeps_original(make_config):
    deep_entered, allow_deep = threading.Event(), threading.Event()
    def deep_action(request):
        deep_entered.set()
        assert allow_deep.wait(9), 'caller did not allow deeper work'
        assert "Use calculator to compute six times seven." in request["prompt"]
        return _action(message="Checked.", tool_calls=[("echo", {"text": "verified"})], continue_loop=True)
    service, runtime, client = setup(make_config, [fast("delegate", "I will check that using the tool."),
                                                    deep_action, _action(message="Verified result.")])
    try:
        iterator = service.iter_orchestrator_message("Use calculator to compute six times seven.", control_id="fast_deep")
        assert next(iterator)["event"] == "accepted"
        progress = next(iterator)
        assert progress["event"] == "assistant_progress"
        assert progress["assistant_text"] == "I will check that using the tool."
        assert deep_entered.wait(3)
        sid = progress["session_id"]
        events = runtime.history.read_history(sid)
        assert not any(e.event_type == "tool_called" for e in events)
        assert not any(e.event_type == "control_message_processed" for e in events)
        assert [c["contract"] for c in client.requests[:2]] == ["fast_foreground_response", "agent_action"]
        allow_deep.set()
        tail = list(iterator)
        assert [e["event"] for e in tail] == ["completed"]
        assert tail[0]["result"]["answer"] == "Verified result."
        events = runtime.history.read_history(sid)
        assert sum(e.event_type == "tool_called" for e in events) == 1
        assert sum(e.event_type == "assistant_progress" for e in events) == 1
        assert len(client.requests) == 3
        reconnect = list(service.iter_orchestrator_message(
            "Use calculator to compute six times seven.", control_id="fast_deep",
            after_sequence=progress["sequence"]))
        assert [i["event"] for i in reconnect] == ["accepted", "completed"]
        assert len(client.requests) == 3
    finally:
        allow_deep.set()
        service.close()


def test_user_correction_while_fast_model_is_running_forces_full_authority(make_config):
    entered, release = threading.Event(), threading.Event()
    holder = {}
    def stale(_request):
        entered.set()
        assert release.wait(8)
        return fast("answer", "Old answer must never be committed")
    def full(request):
        assert "Stop; do not do that" in request["prompt"]
        return _action(message="I will not perform the prior action.")
    service, runtime, client = setup(make_config, [stale, full])
    reply, failures = [], []
    def act():
        try: reply.append(service.orchestrator_message("Read the file."))
        except Exception as exc: failures.append(exc)
    t=threading.Thread(target=act)
    try:
        t.start()
        assert entered.wait(4)
        received = service._receive_orchestrator_message("Stop; do not do that")
        release.set();t.join(10)
        assert not t.is_alive() and not failures
        assert reply[0]["answer"] == "I will not perform the prior action."
        assert not any(e.event_type=="turn_finished" and e.payload.get("assistant_text") == "Old answer must never be committed"
                       for e in runtime.history.read_history(received["session_id"]))
    finally:
        release.set();t.join(10);service.close()


def test_fast_unavailable_uses_complete_original_controller(make_config):
    # A malformed first response must not be mistaken for a direct answer.
    service, runtime, client = setup(make_config, ['{}', _action(message="Full controller result.")])
    try:
        reply = service.orchestrator_message('Please respond safely.')
        assert reply['answer'] == 'Full controller result.'
        assert any(e.event_type=='orchestrator_interaction_unavailable'
                   for e in runtime.history.read_history(reply['session_id']))
        assert len(client.requests) == 2
    finally:service.close()


def test_short_fast_context_does_not_discard_persistent_user_policy(make_config):
    from swaag.prompt_instructions import make_prompt_instruction
    from dataclasses import asdict
    service, runtime, client = setup(make_config, [_action(message="Full policy-respecting answer.")])
    try:
        state = service._orchestrator_state()
        rule = make_prompt_instruction(runtime.config, title="User rule", content="Reply as instructed.",
                                       scopes=["action"], categories=["style"],
                                       source_kind="user", source_ref="conversation",authority="durable_user_policy")
        runtime.history.record_event(state,'prompt_instruction_added',
                                     {'instruction':asdict(rule),'instruction_store':'session'})
        reply = service.orchestrator_message('Hello.')
        assert reply['answer'] == "Full policy-respecting answer."
        assert len(client.requests) == 1
        assert client.requests[0]['contract'] == 'agent_action'
    finally:service.close()


def test_promise_is_not_direct_completion_and_still_dispatches_full_controller(make_config):
    service, runtime, client = setup(make_config, [
        fast("answer", "I will check the current worker status for you."),
        _action(message="Current state is unknown until the actual lookup returns."),
    ])
    try:
        events = list(service.iter_orchestrator_message(
            'Check the workers and do not stop anything.', control_id='safety_promise'))
        assert [e['event'] for e in events] == ['accepted', 'assistant_progress', 'completed']
        assert events[-1]['result']['answer'] == 'Current state is unknown until the actual lookup returns.'
        assert len(client.requests) == 2
        assert client.requests[0]['contract'] == 'fast_foreground_response'
        assert client.requests[1]['contract'] == 'agent_action'
    finally:service.close()


def test_user_wide_policy_is_not_omitted_from_short_fast_context(make_config):
    service, runtime, client = setup(make_config, [_action(message='Full instruction-preserving answer.')])
    try:
        store = runtime.system_prompt_contributors[0].store
        store.add_trusted(title='Retain format', content='Respond in German.',
                          scopes=['action'], categories=['style'],
                          authority='durable_user_policy', source_kind='user',
                          source_ref='confirmed request', origin_session_id='global-policy-check')
        assert store.has_any_events()
        reply=service.orchestrator_message('Hello.')
        assert reply['answer']=='Full instruction-preserving answer.'
        assert [r['contract'] for r in client.requests] == ['agent_action']
    finally:service.close()


def test_long_conversation_does_not_force_full_prompt_for_independent_question(make_config):
    from swaag.types import Message
    from swaag.foreground_fast import generate_fast_foreground_response
    service, runtime, client = setup(make_config, [fast('answer','I am SWAAG.')])
    try:
        state=service._orchestrator_state()
        for i in range(300):
            # Older messages remain available in the authoritative store, but
            # loading them into a quick question would reintroduce huge latency.
            state.messages.append(Message('user', f'Old unrelated conversation {i}.', created_at='earlier'))
        response=generate_fast_foreground_response(runtime,state,'Who are you?')
        assert response and response.mode=='answer'
        assert len(client.requests)==1
        assert 'older_conversation_messages_available_only_to_full_orchestrator' in client.requests[0]['prompt']
        assert '294' in client.requests[0]['prompt']
    finally:service.close()


def test_fast_acknowledgement_is_reused_after_restart_without_duplicate_llm(make_config):
    service, runtime, client = setup(make_config, [fast('delegate','I will check the result.')])
    try:
        state=service._orchestrator_state()
        receipt=service._receive_orchestrator_message('Use calculator to compute a result.',control_id='repeat_ack')
        one=service._try_fast_foreground_reply(state,receipt)
        two=service._try_fast_foreground_reply(state,receipt)
        assert one['mode']==two['mode']=='delegate'
        assert two['reused'] is True
        assert len(client.requests)==1
        events=runtime.history.read_history(state.session_id)
        assert sum(e.event_type=='assistant_progress' for e in events)==1
    finally:service.close()


def test_live_sized_recent_history_fits_without_replaying_the_whole_session(make_config):
    from swaag.types import Message
    from swaag.foreground_fast import generate_fast_foreground_response
    service, runtime, client = setup(make_config, [fast('answer', 'I am SWAAG.')])
    try:
        state = service._orchestrator_state()
        for index, size in enumerate((143, 12, 103, 1708, 2439, 295)):
            state.messages.append(Message(
                role='assistant' if index % 2 else 'user',
                content='x' * size, created_at='old',
            ))
        response = generate_fast_foreground_response(runtime, state, 'Who are you?')
        assert response and response.mode == 'answer'
        assert len(client.requests) == 1
        prompt = client.requests[0]['prompt']
        assert 'older_conversation_messages_available_only_to_full_orchestrator' in prompt
        assert '2439' not in prompt  # The length isn't exposed as invented metadata.
        assert len(prompt) < 8000
    finally:service.close()


def test_quick_delegation_does_not_skip_deep_completion_evaluation(make_config):
    """Fast speech cannot turn a tool-requiring request into an unverified terminal reply."""
    from tests.test_compact_foreground_actions import action, verdict
    cfg = make_config(model__context_limit=32000,
                      runtime__completion_evaluation_enabled=True,
                      communication__fast_initial_reply=True,
                      communication__compact_foreground_actions=True,
                      communication__single_pass_direct_replies=True,
                      communication__direct_reply_verification=False,
                      tools__enabled=['runtime_snapshot_lookup', 'calculator'])
    client=FakeModelClient([
        fast('delegate','I will inspect that requirement.'),
        action('It appears complete without using a tool.'),
        verdict(False),
        action('Now calculating.', [{'tool_name':'calculator','arguments':{'expression':'6 * 7'}}]),
        action('The calculator returned forty-two.'),
        verdict(True),
    ])
    runtime=AgentRuntime(cfg,model_client=client)
    service=CommunicationService(runtime,orchestrator_runtime=runtime)
    try:
        stream=list(service.iter_orchestrator_message('Use the calculator to compute six times seven.',control_id='deep_check'))
        assert [x['event'] for x in stream]==['accepted','assistant_progress','completed']
        assert stream[-1]['result']['answer']=='The calculator returned forty-two.'
        assert [r['contract'] for r in client.requests]==[
            'fast_foreground_response','agent_action','completion_evaluation',
            'agent_action','agent_action','completion_evaluation']
        events=runtime.history.read_history(stream[0]['session_id'])
        assert any(e.event_type=='completion_rejected' for e in events)
        assert sum(e.event_type=='tool_called' for e in events)==1
    finally:service.close()
