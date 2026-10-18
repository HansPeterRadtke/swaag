import json
import threading

import pytest

from swaag.action import ActionValidationError, action_from_payload, expand_compact_foreground_action
from swaag.communication import CommunicationService
from swaag.grammar import agent_action_contract
from swaag.runtime import AgentRuntime
from tests.test_agent_action_loop import FakeModelClient


def action(text, calls=(), questions=(), count=None):
    return json.dumps({"assistant_message": text, "tool_calls": list(calls),
                      "questions": list(questions), "response_constraints": {"exact_word_count": count}})


def verdict(complete=True):
    return json.dumps({"complete": complete, "reason": "Supported answer." if complete else "No phone playback evidence.",
                      "remaining_work": [] if complete else ["Remove the unsupported playback claim."],
                      "evidence_requests": []})


def runtime_for(make_config, responses, **overrides):
    overrides.setdefault("runtime__completion_evaluation_enabled", True)
    config = make_config(model__context_limit=32000,
        communication__compact_foreground_actions=True,
        tools__enabled=["runtime_snapshot_lookup", "calculator"], **overrides)
    client = FakeModelClient(responses)
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    runtime.bind_tool_runtime_capability(state.session_id, "system_snapshot_provider", lambda: {
        "captured_at": "now", "active_workers": [], "active_plans": [], "terminal_workers": [],
        "worker_question_inventory": {"questions": [], "workers": [], "complete": True,
                                      "scope": "all workers", "pending_revisions": []},
    })
    return runtime, state, client


def test_compact_schema_preserves_semantic_fields_and_exact_tool_schema():
    tools = [("calculator", "Compute", {"type": "object", "properties": {"expression": {"type": "string"}},
                                      "required": ["expression"], "additionalProperties": False}, "")]
    full = agent_action_contract(tools).json_schema
    compact = agent_action_contract(tools, compact_foreground=True).json_schema
    assert set(compact["required"]) == {"assistant_message", "tool_calls", "response_constraints", "questions"}
    assert compact["additionalProperties"] is False
    assert compact["properties"] == {k: v for k, v in full["properties"].items() if k in compact["required"]}
    assert "status" in agent_action_contract(tools).json_schema["required"]
    with pytest.raises(ValueError):
        agent_action_contract(tools, compact_foreground=True, allow_silent_completion=True)


@pytest.mark.parametrize("mutation", [
    {"tool_calls": "not an array"}, {"silent_completion": True}, {"continue_loop": False},
    {"status": {"situation": "invented success"}},
])
def test_invalid_compact_envelopes_fail_closed(mutation):
    payload = {**json.loads(action("Hello.")), **mutation}
    with pytest.raises(ActionValidationError):
        expand_compact_foreground_action(payload)


def test_compact_tools_still_require_permission_and_no_blocking_question():
    payload = json.loads(action("Checking.", [{"tool_name": "calculator", "arguments": {"expression": "6 * 7"}}]))
    expanded = expand_compact_foreground_action(payload)
    assert expanded["continue_loop"] is True and expanded["silent_completion"] is False
    with pytest.raises(ActionValidationError):
        action_from_payload(expanded, enabled_tool_names=[])
    payload["questions"] = [{"question": "May I calculate?", "criticality": "blocking", "importance": "normal",
                             "reason": "Permission is required.", "assumption_if_unanswered": ""}]
    with pytest.raises(ActionValidationError):
        action_from_payload(expand_compact_foreground_action(payload), enabled_tool_names=["calculator"])


def test_compact_reply_keeps_completion_verification_and_does_not_invent_status(make_config):
    def answer(request):
        assert set(request["json_schema"]["required"]) == {"assistant_message", "tool_calls", "questions", "response_constraints"}
        assert "Do not generate a separate status report" in request["prompt"]
        return action("Hello.")
    runtime, state, client = runtime_for(make_config, [answer, verdict()])
    result = runtime.run_turn_in_session(state, "Please say hello.")
    assert result.assistant_text == "Hello."
    assert [request["contract"] for request in client.requests] == ["agent_action", "completion_evaluation"]
    events = runtime.history.read_history(state.session_id)
    assert not any(event.event_type == "agent_status" for event in events)
    assert not any(event.event_type == "tool_called" for event in events)
    assert any(event.event_type == "completion_evaluated" and event.payload["complete"] for event in events)


def test_compact_false_playback_claim_is_rejected_before_reply(make_config):
    runtime, state, client = runtime_for(make_config, [
        action("Your phone successfully played the reply."), verdict(False),
        action("I received your message. I cannot confirm phone playback."), verdict(),
    ])
    result = runtime.run_turn_in_session(state, "Can you hear me? Did the phone play your reply?")
    assert result.assistant_text == "I received your message. I cannot confirm phone playback."
    events = runtime.history.read_history(state.session_id)
    assert any(event.event_type == "completion_rejected" for event in events)
    assert len(client.requests) == 4
    assert not any(event.event_type == "assistant_progress" for event in events)


def test_compact_flag_does_not_change_worker_contract(make_config):
    from tests.test_agent_action_loop import _action
    ordinary_verdict = json.loads(verdict())
    ordinary_verdict.pop("evidence_requests")
    runtime, state, client = runtime_for(make_config, [_action(message="Hello."), json.dumps(ordinary_verdict)])
    runtime.bind_tool_runtime_capability(state.session_id, "system_snapshot_provider", None)
    assert runtime.run_turn_in_session(state, "Please say hello.").assistant_text == "Hello."
    assert "status" in client.requests[0]["json_schema"]["required"]


def test_compact_tool_result_is_observed_before_the_verified_terminal_reply(make_config):
    runtime, state, client = runtime_for(make_config, [
        action("Checking.", [{"tool_name": "calculator", "arguments": {"expression": "6 * 7"}}]),
        action("The result is forty-two."), verdict(),
    ])
    assert runtime.run_turn_in_session(state, "Calculate six times seven.").assistant_text == "The result is forty-two."
    events = runtime.history.read_history(state.session_id)
    assert len([event for event in events if event.event_type == "tool_called"]) == 1
    assert len([event for event in events if event.event_type == "tool_result"]) == 1
    assert "42" in client.requests[1]["prompt"]
    assert client.requests[-1]["contract"] == "completion_evaluation"


def test_compact_inflight_correction_prevents_stale_tool_effect(make_config):
    entered, release = threading.Event(), threading.Event()
    exact = "  Cancel that calculation.\nDo not run it.  "
    def stale(request):
        entered.set()
        assert release.wait(8)
        return action("Checking.", [{"tool_name": "calculator", "arguments": {"expression": "6 * 7"}}])
    def corrected(request):
        assert exact in request["prompt"]
        return action("The calculation was canceled before execution.")
    runtime, _, client = runtime_for(make_config, [stale, corrected], runtime__completion_evaluation_enabled=False)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    result, errors = [], []
    def run():
        try:
            result.append(service.orchestrator_message("Calculate six times seven."))
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    try:
        thread.start()
        assert entered.wait(5)
        receipt = service._receive_orchestrator_message(exact)
        release.set()
        thread.join(10)
        assert not thread.is_alive() and not errors
        assert result[0]["answer"] == "The calculation was canceled before execution."
        events = runtime.history.read_history(receipt["session_id"])
        assert not any(event.event_type == "tool_called" for event in events)
        assert any(event.event_type == "agent_action_rejected" and "New user control arrived" in event.payload.get("reason", "") for event in events)
        assert len(client.requests) == 2
    finally:
        release.set()
        thread.join(10)
        service.close()
