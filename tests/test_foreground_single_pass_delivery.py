"""One foreground LLM turn and immediate acknowledgement while tool work proceeds."""
from __future__ import annotations

import threading

from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from tests.test_agent_action_loop import FakeModelClient, _action
from tests.test_compact_foreground_actions import action, runtime_for, verdict


def test_one_model_call_for_simple_direct_reply_even_with_verifier_configured(make_config):
    runtime, state, client = runtime_for(
        make_config, [action("Hello. You're speaking with SWAAG.")],
        communication__single_pass_direct_replies=True,
        communication__direct_reply_verification=True,
    )
    result = runtime.run_turn_in_session(state, "Who are you?")
    assert result.assistant_text == "Hello. You're speaking with SWAAG."
    assert [request["contract"] for request in client.requests] == ["agent_action"]
    events = runtime.history.read_history(state.session_id)
    completion = [e for e in events if e.event_type == "completion_evaluated"]
    assert len(completion) == 1
    assert completion[0].payload["verification_scope"] == "foreground_single_pass_unreviewed"
    assert completion[0].sequence < next(e.sequence for e in events if e.event_type == "turn_finished")
    assert not any(e.event_type == "assistant_progress" for e in events)


def test_deeper_tool_work_retains_normal_independent_completion(make_config):
    runtime, state, client = runtime_for(make_config, [
        action("I'll calculate it.", [{"tool_name": "calculator", "arguments": {"expression": "6 * 7"}}]),
        action("The verified result is forty-two."), verdict(),
    ], communication__single_pass_direct_replies=True)
    result = runtime.run_turn_in_session(state, "Use the calculator to compute six times seven.")
    assert result.assistant_text == "The verified result is forty-two."
    assert [request["contract"] for request in client.requests] == [
        "agent_action", "agent_action", "completion_evaluation"]
    assert not any(e.payload.get("verification_scope") == "foreground_single_pass_unreviewed"
                   for e in runtime.history.read_history(state.session_id)
                   if e.event_type == "completion_evaluated")


def test_stream_delivers_first_acknowledgement_before_tool_executes(make_config):
    entered_tool, release_tool = threading.Event(), threading.Event()
    client = FakeModelClient([
        _action(message="I will check the requested value.",
                tool_calls=[("echo", {"text": "verified"})], continue_loop=True),
        _action(message="The check has finished."),
    ])
    runtime = AgentRuntime(make_config(model__context_limit=32000,
        runtime__completion_evaluation_enabled=False,
        communication__poll_seconds=0.01,
        communication__early_tool_acknowledgement=True), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    original = runtime._execute_tool

    def blocked_tool(*args, **kwargs):
        entered_tool.set()
        assert release_tool.wait(8), "Test failed to unblock selected tool"
        return original(*args, **kwargs)

    runtime._execute_tool = blocked_tool
    stream = None
    try:
        stream = service.iter_orchestrator_message("Check the requested value.", control_id="early_delivery_test")
        assert next(stream)["event"] == "accepted"
        early = next(stream)
        assert early["event"] == "assistant_progress"
        assert early["assistant_text"] == "I will check the requested value."
        assert entered_tool.wait(2)
        events = runtime.history.read_history(early["session_id"])
        assert not any(e.event_type == "tool_result" for e in events), "Speech waited for tool completion"
        assert next(e for e in events if e.sequence == early["sequence"]).payload["delivery_phase"] == "after_action_acceptance_before_tools"
        release_tool.set()
        tail = list(stream)
        assert [e["event"] for e in tail] == ["completed"]
        assert tail[0]["result"]["answer"] == "The check has finished."
        assert len(client.requests) == 2
    finally:
        release_tool.set()
        if stream is not None:
            stream.close()
        service.close()


def test_no_early_tool_claim_when_user_control_appears_before_action_acceptance(make_config):
    holder = {}
    corrected = "Cancel the pending tool action; do not execute it."
    def pending_action(_request):
        runtime, state = holder["runtime"], holder["state"]
        runtime.history.enqueue_control_message(state.session_id, corrected, source="user")
        return _action(message="I'll execute the original tool.",
                       tool_calls=[("echo", {"text": "should-not-happen"})], continue_loop=True)
    client = FakeModelClient([pending_action, _action(message="No tool was executed.")])
    runtime = AgentRuntime(make_config(model__context_limit=32000,
        runtime__completion_evaluation_enabled=False,
        communication__early_tool_acknowledgement=True), model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    holder.update(runtime=runtime, state=service._load_orchestrator_state())
    try:
        reply = service.orchestrator_message("Run the original tool.")
        assert reply["answer"] == "No tool was executed."
        events = runtime.history.read_history(reply["session_id"])
        assert not any(e.event_type in ("assistant_progress", "tool_called") for e in events)
    finally:
        service.close()
