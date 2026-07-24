from __future__ import annotations

import pytest

from swaag.decision import (
    DecisionValidationError,
    decision_from_payload,
    validate_decision,
)
from swaag.types import DecisionOutcome


def test_incorrect_decision_is_rejected() -> None:
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
        validate_decision(bad)


def test_decision_from_payload_rejects_missing_reason() -> None:
    with pytest.raises(DecisionValidationError):
        decision_from_payload(
            {
                "split_task": True,
                "expand_task": False,
                "ask_user": False,
                "assume_missing": False,
                "generate_ideas": False,
                "direct_response": False,
                "execution_mode": "full_plan",
                "preferred_tool_name": "",
                "confidence": 0.9,
                "reason": "",
            },
        )


def test_direct_response_with_expansion_is_rejected() -> None:
    with pytest.raises(DecisionValidationError):
        decision_from_payload(
            {
                "split_task": False,
                "expand_task": True,
                "ask_user": False,
                "assume_missing": False,
                "generate_ideas": False,
                "direct_response": True,
                "execution_mode": "direct_response",
                "preferred_tool_name": "",
                "confidence": 0.9,
                "reason": "contradictory",
            },
        )


def test_decision_from_payload_parses_direct_response() -> None:
    decision = decision_from_payload(
        {
            "split_task": False,
            "expand_task": False,
            "ask_user": False,
            "assume_missing": False,
            "generate_ideas": False,
            "direct_response": True,
            "execution_mode": "direct_response",
            "preferred_tool_name": "",
            "confidence": 0.9,
            "reason": "single direct answer is enough",
        },
    )

    assert decision.direct_response is True


def test_clarification_mode_requires_ask_user() -> None:
    with pytest.raises(DecisionValidationError, match="requires ask_user=true"):
        decision_from_payload({
            "split_task": False, "expand_task": False, "ask_user": False,
            "assume_missing": False, "generate_ideas": False, "direct_response": False,
            "execution_mode": "clarification", "preferred_tool_name": "",
            "confidence": 0.9, "reason": "ask now",
        })


def test_clarification_mode_parses_immediate_question() -> None:
    decision = decision_from_payload({
        "split_task": False, "expand_task": False, "ask_user": True,
        "assume_missing": False, "generate_ideas": False, "direct_response": False,
        "execution_mode": "clarification", "preferred_tool_name": "",
        "confidence": 0.9, "reason": "missing information",
    })
    assert decision.ask_user is True
    assert decision.execution_mode == "clarification"


def test_evidence_required_rejects_immediate_clarification() -> None:
    with pytest.raises(DecisionValidationError, match="requires execution_mode='full_plan' or 'single_tool'"):
        decision_from_payload({
            "split_task": False, "expand_task": False, "ask_user": True,
            "assume_missing": False, "generate_ideas": False, "direct_response": False,
            "execution_mode": "clarification", "preferred_tool_name": "",
            "evidence_required_before_response": True,
            "evidence_call_count": 1,
            "confidence": 0.9, "reason": "must read files first",
        })


def test_evidence_required_accepts_tool_backed_plan() -> None:
    decision = decision_from_payload({
        "split_task": False, "expand_task": False, "ask_user": True,
        "assume_missing": False, "generate_ideas": False, "direct_response": False,
        "execution_mode": "full_plan", "preferred_tool_name": "",
        "evidence_required_before_response": True,
        "evidence_call_count": 2,
        "confidence": 0.9, "reason": "must read files first",
    })
    assert decision.evidence_required_before_response is True


def test_multiple_evidence_calls_reject_single_tool_mode() -> None:
    with pytest.raises(DecisionValidationError, match="evidence_call_count>1 requires execution_mode='full_plan'"):
        decision_from_payload({
            "split_task": False, "expand_task": False, "ask_user": True,
            "assume_missing": False, "generate_ideas": False, "direct_response": False,
            "execution_mode": "single_tool", "preferred_tool_name": "read_file",
            "evidence_required_before_response": True, "evidence_call_count": 2,
            "confidence": 0.9, "reason": "two files must be read",
        })
