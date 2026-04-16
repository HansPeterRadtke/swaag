from __future__ import annotations

import pytest

from swaag.expander import ExpansionValidationError, expand_task, expanded_task_from_payload
from swaag.types import DecisionOutcome, PromptAnalysis


def test_expander_turns_vague_prompt_into_bounded_goal() -> None:
    prompt = "make a game"
    # Use an explicit "vague" analysis (the fallback returns "unstructured"
    # for everything, which is not what this test is exercising).
    analysis = PromptAnalysis(
        task_type="vague",
        completeness="incomplete",
        requires_expansion=True,
        requires_decomposition=False,
        confidence=0.6,
        detected_entities=[],
        detected_goals=["make"],
    )
    decision = DecisionOutcome(
        split_task=False,
        expand_task=True,
        ask_user=False,
        assume_missing=True,
        generate_ideas=True,
        confidence=analysis.confidence,
        reason="test",
    )

    expanded = expand_task(prompt, analysis, decision)

    assert expanded.original_goal == prompt
    assert "playable loop" in expanded.expanded_goal
    assert expanded.scope
    assert expanded.constraints
    assert expanded.expected_outputs


def test_expander_is_deterministic() -> None:
    prompt = "build a local agent"
    analysis = PromptAnalysis(
        task_type="unstructured",
        completeness="partial",
        requires_expansion=True,
        requires_decomposition=True,
        confidence=0.5,
        detected_entities=["agent"],
        detected_goals=["build"],
    )
    decision = DecisionOutcome(
        split_task=True,
        expand_task=True,
        ask_user=False,
        assume_missing=False,
        generate_ideas=False,
        confidence=0.5,
        reason="test",
    )

    assert expand_task(prompt, analysis, decision) == expand_task(prompt, analysis, decision)


def test_expanded_task_from_payload_rejects_empty_goal() -> None:
    with pytest.raises(ExpansionValidationError):
        expanded_task_from_payload(
            {
                "original_goal": "",
                "expanded_goal": "",
                "scope": [],
                "constraints": [],
                "expected_outputs": [],
                "assumptions": [],
            },
            original_goal="make a game",
        )
