from __future__ import annotations

import pytest

from swaag.decision import (
    DecisionValidationError,
    decide_from_analysis_emergency_fallback,
    decision_from_payload,
    validate_decision,
)
from swaag.types import DecisionOutcome, PromptAnalysis


def test_vague_prompt_requires_expansion() -> None:
    analysis = PromptAnalysis(
        task_type="vague",
        completeness="incomplete",
        requires_expansion=True,
        requires_decomposition=False,
        confidence=0.65,
    )

    decision = decide_from_analysis_emergency_fallback(analysis)

    assert decision.expand_task is True
    assert decision.assume_missing is True
    assert decision.generate_ideas is True
    assert decision.direct_response is False


def test_structured_prompt_does_not_force_clarification() -> None:
    analysis = PromptAnalysis(
        task_type="structured",
        completeness="complete",
        requires_expansion=False,
        requires_decomposition=True,
        confidence=0.95,
    )

    decision = decide_from_analysis_emergency_fallback(analysis)

    assert decision.ask_user is False
    assert decision.expand_task is False
    assert decision.direct_response is False


def test_incorrect_decision_is_rejected() -> None:
    analysis = PromptAnalysis(
        task_type="vague",
        completeness="incomplete",
        requires_expansion=True,
        requires_decomposition=False,
        confidence=0.65,
    )
    bad = DecisionOutcome(
        split_task=False,
        expand_task=False,
        ask_user=True,
        assume_missing=False,
        generate_ideas=False,
        confidence=0.1,
        reason="bad",
        direct_response=True,
    )

    with pytest.raises(DecisionValidationError):
        validate_decision(analysis, bad)


def test_decision_from_payload_requires_reason() -> None:
    analysis = PromptAnalysis(
        task_type="structured",
        completeness="complete",
        requires_expansion=False,
        requires_decomposition=True,
        confidence=0.95,
    )

    with pytest.raises(DecisionValidationError):
        decision_from_payload(
            {
                "split_task": True,
                "expand_task": False,
                "ask_user": False,
                "assume_missing": False,
                "generate_ideas": False,
                "confidence": 0.9,
                "reason": "",
            },
            analysis,
        )


def test_direct_response_cannot_request_expansion_or_clarification() -> None:
    analysis = PromptAnalysis(
        task_type="structured",
        completeness="complete",
        requires_expansion=False,
        requires_decomposition=False,
        confidence=0.95,
    )

    with pytest.raises(DecisionValidationError):
        decision_from_payload(
            {
                "split_task": False,
                "expand_task": True,
                "ask_user": False,
                "assume_missing": False,
                "generate_ideas": False,
                "direct_response": True,
                "confidence": 0.9,
                "reason": "contradictory",
            },
            analysis,
        )


def test_decision_from_payload_parses_direct_response() -> None:
    analysis = PromptAnalysis(
        task_type="structured",
        completeness="complete",
        requires_expansion=False,
        requires_decomposition=False,
        confidence=0.95,
    )

    decision = decision_from_payload(
        {
            "split_task": False,
            "expand_task": False,
            "ask_user": False,
            "assume_missing": False,
            "generate_ideas": False,
            "direct_response": True,
            "confidence": 0.9,
            "reason": "single direct answer is enough",
        },
        analysis,
    )

    assert decision.direct_response is True
