import json

import pytest

from swaag.types import Message
from tests.test_compact_foreground_actions import action, runtime_for, verdict


def direct(approved=True):
    return json.dumps({"direct_reply_sufficient": approved})


def test_direct_reply_uses_independent_check_with_prior_authority_and_fresh_state(make_config):
    def check(request):
        assert request["contract"] == "direct_conversational_reply_check"
        assert "Call the project Iris, never disclose its password." in request["prompt"]
        assert "Fresh runtime observation" in request["prompt"]
        return direct()
    runtime, state, client = runtime_for(make_config, [action("Hello."), check],
        communication__direct_reply_verification=True)
    runtime._record_message(state, Message(role="user",
        content="Call the project Iris, never disclose its password.", created_at="earlier"))
    result = runtime.run_turn_in_session(state, "Please say hello.")
    assert result.assistant_text == "Hello."
    events = runtime.history.read_history(state.session_id)
    checks = [e for e in events if e.event_type == "completion_evaluated"]
    assert len(checks) == 1 and checks[0].payload["verification_scope"] == "direct_conversation_only"
    assert checks[0].sequence < next(e.sequence for e in events if e.event_type == "turn_finished")
    assert len(client.requests) == 2 and not result.tool_results


@pytest.mark.parametrize("response", [direct(False), '{"direct_reply_sufficient":"true"}',
    '{"direct_reply_sufficient":true,"extra":1}', 'not JSON'])
def test_nonapproval_and_invalid_output_use_full_checker(make_config, response):
    runtime, state, client = runtime_for(make_config, [action("Hello."), response, verdict()],
        communication__direct_reply_verification=True)
    assert runtime.run_turn_in_session(state, "Please say hello.").assistant_text == "Hello."
    assert [r["contract"] for r in client.requests] == [
        "agent_action", "direct_conversational_reply_check", "completion_evaluation"]


def test_missing_playback_evidence_reaches_full_checker_and_never_delivers_false_claim(make_config):
    runtime, state, client = runtime_for(make_config, [
        action("Your phone played my reply."), direct(False), verdict(False),
        action("Your message reached me. Phone playback remains unverified."), verdict()],
        communication__direct_reply_verification=True)
    result = runtime.run_turn_in_session(state, "Did my phone play the reply?")
    assert result.assistant_text == "Your message reached me. Phone playback remains unverified."
    assert len(client.requests) == 5
    delivered = [e.payload for e in runtime.history.read_history(state.session_id)
                 if e.event_type in {"assistant_progress", "turn_finished"}]
    assert all("Your phone played my reply." not in json.dumps(item) for item in delivered)


def test_tool_effects_keep_full_completion_evidence(make_config):
    runtime, state, client = runtime_for(make_config, [
        action("Checking.", [{"tool_name": "calculator", "arguments": {"expression": "6 * 7"}}]),
        action("Forty-two."), verdict()], communication__direct_reply_verification=True)
    assert runtime.run_turn_in_session(state, "Calculate six times seven.").assistant_text == "Forty-two."
    assert [r["contract"] for r in client.requests] == ["agent_action", "agent_action", "completion_evaluation"]
    assert "42" in client.requests[-1]["prompt"]


def test_correction_during_direct_verification_prevents_stale_delivery(make_config):
    holder = {}
    def intervened(request):
        runtime, state = holder["runtime"], holder["state"]
        runtime.history.enqueue_control_message(state.session_id,
            "Instead reply with the exact word goodbye.", source="user")
        return direct()
    runtime, state, client = runtime_for(make_config, [action("Hello."), intervened,
        action("goodbye"), verdict()], communication__direct_reply_verification=True)
    holder.update(runtime=runtime, state=state)
    result = runtime.run_turn_in_session(state, "Please say hello.")
    assert result.assistant_text == "goodbye"
    events = runtime.history.read_history(state.session_id)
    assert not any(e.event_type == "turn_finished" and e.payload["assistant_text"] == "Hello." for e in events)
    assert "Instead reply with the exact word goodbye." in client.requests[-1]["prompt"]


def test_state_change_during_verification_falls_back_to_fresh_full_evidence(make_config):
    snapshot = {"captured_at": "now", "active_workers": [], "active_plans": [],
        "terminal_workers": [], "revision": 1,
        "worker_question_inventory": {"complete": True, "scope": "all", "questions": [],
                                     "workers": [], "pending_revisions": []}}
    def changed(request):
        snapshot["revision"] = 2
        return direct()
    runtime, state, client = runtime_for(make_config, [action("Hello."), changed, verdict()],
        communication__direct_reply_verification=True)
    runtime.bind_tool_runtime_capability(state.session_id, "system_snapshot_provider", lambda: dict(snapshot))
    runtime.bind_tool_runtime_capability(state.session_id, "system_snapshot_fingerprint", lambda s: s["revision"])
    assert runtime.run_turn_in_session(state, "Please say hello.").assistant_text == "Hello."
    assert client.requests[-1]["contract"] == "completion_evaluation"
    events = runtime.history.read_history(state.session_id)
    assert not any(e.event_type == "completion_evaluated" and e.payload.get("verification_scope")
                   == "direct_conversation_only" for e in events)


def test_complete_looking_but_truncated_verdict_never_grants_approval(make_config):
    from dataclasses import replace
    runtime, state, client = runtime_for(make_config, [action("Hello."), direct(), verdict()],
        communication__direct_reply_verification=True)
    send = client.send_completion
    def truncated(payload, *, timeout_seconds=None, progress_callback=None):
        result = send(payload, timeout_seconds=timeout_seconds, progress_callback=progress_callback)
        return replace(result, finish_reason="length") if payload["contract"] == "direct_conversational_reply_check" else result
    client.send_completion = truncated
    assert runtime.run_turn_in_session(state, "Please say hello.").assistant_text == "Hello."
    assert client.requests[-1]["contract"] == "completion_evaluation"
    events = runtime.history.read_history(state.session_id)
    assert any(e.event_type == "model_output_budget_exhausted" for e in events)
    assert not any(e.event_type == "completion_evaluated" and e.payload.get("verification_scope")
                   == "direct_conversation_only" for e in events)
