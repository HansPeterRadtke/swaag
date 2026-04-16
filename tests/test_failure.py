"""Tests for the LLM-driven failure classification system.

These tests cover three layers:

1. The strict ``classify_failure_from_payload`` parser/validator for LLM
   responses.
2. The deterministic ``FAILURE_KIND_DEFAULTS`` policy table that maps a
   semantic kind to mechanical retry/replan/wait behavior.
3. The emergency-only ``classify_failure_emergency_fallback`` which must
   never inspect error strings or reason text for keywords.
"""

from __future__ import annotations

import pytest

from swaag.failure import (
    FAILURE_KIND_DEFAULTS,
    FailureValidationError,
    classify_failure_emergency_fallback,
    classify_failure_from_payload,
    policy_for_kind,
)
from swaag.types import FailureKind, PlanStep
from typing import get_args


def _tool_step() -> PlanStep:
    return PlanStep(
        step_id="step_tool",
        title="Use calculator",
        goal="Compute",
        kind="tool",
        expected_tool="calculator",
        input_text="2 + 2",
        expected_output="number",
        done_condition="tool_result:calculator",
        success_criteria="calculator returns a value",
    )


# ----- payload parser / validator ----------------------------------------------------


def test_payload_parser_accepts_well_formed_response() -> None:
    payload = {
        "kind": "transient_external_wait",
        "retryable": True,
        "requires_replan": False,
        "suggested_strategy_mode": "conservative",
        "wait_seconds": 5.0,
        "reason": "rate-limited by upstream API",
    }
    classification = classify_failure_from_payload(payload)
    assert classification.kind == "transient_external_wait"
    assert classification.retryable is True
    assert classification.wait_seconds == 5.0
    assert classification.source == "llm"


def test_payload_parser_rejects_unknown_kind() -> None:
    payload = {
        "kind": "definitely_not_a_kind",
        "retryable": False,
        "requires_replan": True,
        "suggested_strategy_mode": "conservative",
        "wait_seconds": 0,
        "reason": "x",
    }
    with pytest.raises(FailureValidationError):
        classify_failure_from_payload(payload)


def test_payload_parser_rejects_unknown_strategy_mode() -> None:
    payload = {
        "kind": "tool_failure",
        "retryable": False,
        "requires_replan": True,
        "suggested_strategy_mode": "yolo",
        "wait_seconds": 0,
        "reason": "x",
    }
    with pytest.raises(FailureValidationError):
        classify_failure_from_payload(payload)


def test_payload_parser_rejects_blank_reason() -> None:
    payload = {
        "kind": "tool_failure",
        "retryable": False,
        "requires_replan": True,
        "suggested_strategy_mode": "conservative",
        "wait_seconds": 0,
        "reason": "   ",
    }
    with pytest.raises(FailureValidationError):
        classify_failure_from_payload(payload)


def test_payload_parser_rejects_negative_wait() -> None:
    payload = {
        "kind": "tool_failure",
        "retryable": False,
        "requires_replan": True,
        "suggested_strategy_mode": "conservative",
        "wait_seconds": -1.0,
        "reason": "x",
    }
    with pytest.raises(FailureValidationError):
        classify_failure_from_payload(payload)


# ----- deterministic kind→policy table -----------------------------------------------


def test_every_failure_kind_has_a_policy() -> None:
    declared_kinds = set(get_args(FailureKind))
    policy_kinds = set(FAILURE_KIND_DEFAULTS.keys())
    assert declared_kinds == policy_kinds, (
        f"FailureKind and FAILURE_KIND_DEFAULTS are out of sync: "
        f"missing={declared_kinds - policy_kinds} extra={policy_kinds - declared_kinds}"
    )


def test_policy_for_known_kind() -> None:
    policy = policy_for_kind("transient_external_wait")
    assert policy.retryable is True
    assert policy.wait_seconds > 0


def test_policy_for_unknown_kind_raises() -> None:
    with pytest.raises(FailureValidationError):
        policy_for_kind("nope")


# ----- emergency-only deterministic fallback -----------------------------------------


def test_emergency_fallback_is_safe_neutral() -> None:
    classification = classify_failure_emergency_fallback(
        step=_tool_step(), error_type="BudgetExceededError", reason="budget exceeded"
    )
    assert classification.source == "deterministic_fallback_emergency_only"
    assert classification.kind == "reasoning_failure"
    assert classification.retryable is False
    assert classification.requires_replan is True
    assert classification.suggested_strategy_mode == "conservative"


def test_emergency_fallback_does_not_inspect_strings() -> None:
    """The fallback must produce the same result regardless of the keywords
    in the reason or error_type. If it ever switches based on substring
    content, this test will fail."""

    a = classify_failure_emergency_fallback(
        step=None, error_type="BudgetExceededError", reason="permission denied"
    )
    b = classify_failure_emergency_fallback(
        step=None, error_type="TimeoutError", reason="verification mismatch"
    )
    c = classify_failure_emergency_fallback(
        step=None, error_type="ValueError", reason="anything at all"
    )
    for classification in (a, b, c):
        assert classification.kind == "reasoning_failure"
        assert classification.retryable is False
        assert classification.requires_replan is True
        assert classification.source == "deterministic_fallback_emergency_only"
