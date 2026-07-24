"""Tests for the LLM-driven failure classification contract parser."""

from __future__ import annotations

import pytest

from swaag.failure import FailureValidationError, classify_failure_from_payload


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
