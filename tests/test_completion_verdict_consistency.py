"""A contradictory model completion verdict cannot authorize a final result.

This contract applies to EVERY current task, not source-audit filenames or
particular phrases. Model text remains exact and retrievable in history.
"""
from __future__ import annotations

import json

import pytest

from swaag.runtime import AgentRuntime
from tests.test_agent_action_loop import FakeModelClient, _action
from tests.test_completion_evaluation import _completed_action


@pytest.mark.parametrize("remaining_work", [
    ["No remaining work"],
    [""],
    ["Need exact source verification"],
    ["No remaining work", "Check one more fact"],
])
def test_nonempty_completion_remaining_work_is_never_success(
    make_config, tmp_path, remaining_work,
):
    config = make_config(model__context_limit=32000)
    config.sessions.root = tmp_path / "sessions"
    model = FakeModelClient([json.dumps({
        "complete": True,
        "reason": "My own answer says every requested task was finished.",
        "remaining_work": remaining_work,
    })])
    runtime = AgentRuntime(config, model_client=model)
    state = runtime.create_or_load_session()
    result = runtime._evaluate_completion(
        state, original_request="Verify the actual tool results before completion.",
        selected_action=_completed_action(), tool_results=[],
    )
    assert result["complete"] is False
    assert result["completion_claim_inconsistent"] is True
    assert result["model_claimed_complete"] is True
    assert result["model_reported_remaining_work"] == remaining_work
    assert len(result["remaining_work"]) == 1
    assert "Reassess" in result["remaining_work"][0]
    assert "Do not merely" in result["remaining_work"][0]
    assert result["model_reported_reason"] == "My own answer says every requested task was finished."
    assert "nonempty remaining_work" in result["reason"]
    recorded = runtime.history.read_history(state.session_id)
    evaluations = [r.payload for r in recorded if r.event_type == "completion_evaluated"]
    assert evaluations[-1]["complete"] is False
    model_responses = [r.payload for r in recorded if r.event_type == "model_response_received"]
    assert len(model_responses) == 1
    original = json.loads(model_responses[0]["completion"]["text"])
    assert original["remaining_work"] == remaining_work
    assert original["complete"] is True


@pytest.mark.parametrize("complete,remaining_work,expected", [
    (True, [], True),
    (False, [], False),
    (False, ["Do not claim verification."], False),
])
def test_consistent_completion_verdict_preserves_original_semantic_answer(
    make_config, tmp_path, complete, remaining_work, expected,
):
    config = make_config(model__context_limit=32000)
    config.sessions.root = tmp_path / "sessions"
    model = FakeModelClient([json.dumps({
        "complete": complete, "reason": "Evidence inspected.",
        "remaining_work": remaining_work,
    })])
    runtime = AgentRuntime(config, model_client=model)
    state = runtime.create_or_load_session()
    result = runtime._evaluate_completion(
        state, original_request="Summarize the evidence.",
        selected_action=_completed_action(), tool_results=[],
    )
    assert result["complete"] is expected
    assert result["completion_claim_inconsistent"] is False
    assert result["reason"] == "Evidence inspected."
    assert result["remaining_work"] == remaining_work


def test_one_responsibility_verdict_without_remaining_field_is_unchanged(make_config, tmp_path):
    config = make_config(model__max_semantic_responsibilities_per_call=1,
                         model__context_limit=32000)
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config, model_client=FakeModelClient(['{"complete":true}']))
    state = runtime.create_or_load_session()
    result = runtime._evaluate_completion(
        state, original_request="Confirm the source.",
        selected_action=_completed_action(), tool_results=[],
    )
    assert result["complete"] is True
    assert result["completion_claim_inconsistent"] is False
    assert result["model_reported_remaining_work"] == []


def test_inconsistent_verdict_is_bounded_by_existing_action_retry_limit(make_config, tmp_path):
    """Same invalid reviewer verdict twice must end as incomplete, never success."""
    config = make_config(model__context_limit=32000,
                         runtime__completion_evaluation_enabled=True,
                         runtime__max_completion_rejections=2)
    config.sessions.root = tmp_path / "sessions"
    unsupported = json.dumps({
        "complete": True,
        "reason": "Everything was done, trust me.",
        "remaining_work": ["No remaining work"],
    })
    responses = [
        _action(message="I examined the source."), unsupported,
        _action(message="I examined the source again."), unsupported,
    ]
    model = FakeModelClient(responses)
    runtime = AgentRuntime(config, model_client=model)
    outcome = runtime.run_turn("Inspect the actual file and confirm its contents.")
    assert "task remains incomplete" in outcome.assistant_text.lower()
    assert len(model.requests) == 4
    events = runtime.history.read_history(outcome.session_id)
    rejected = [event.payload for event in events if event.event_type == "completion_rejected"]
    assert [row["attempt"] for row in rejected] == [1, 2]
    assert rejected[-1]["retry_budget_exhausted"] is True
    evaluated = [event.payload for event in events if event.event_type == "completion_evaluated"]
    assert len(evaluated) == 2
    assert all(row["completion_claim_inconsistent"] and not row["complete"] for row in evaluated)
    assert sum(event.event_type == "turn_finished" for event in events) == 1


def test_current_completion_prompt_explicitly_rejects_status_text_as_remaining_work(make_config):
    from swaag.prompts import PromptBuilder
    builder=PromptBuilder(make_config())
    assembly=builder.build_completion_evaluation_prompt(
        original_request="Count actual declarations and add them.",
        assistant_message="The count is verified.",
        status_json='{"importance":"normal"}',
    )
    text=assembly.prompt_text
    assert 'remaining_work' in text and '["No remaining work"]' in text
    assert "correct arithmetic result from wrong inputs" in text
    assert "Verify claimed numeric counts" in text
    assert "Count actual declarations and add them." in text
    assert assembly.kind == "completion_evaluation"
