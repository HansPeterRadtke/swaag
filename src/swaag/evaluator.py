from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swaag.types import PlanStep, SessionMetrics, ToolExecutionResult
from swaag.verification import VerificationOutcome


@dataclass(slots=True)
class EvaluationOutcome:
    passed: bool
    confidence: float
    reason: str
    requires_retry: bool
    requires_replan: bool


@dataclass(slots=True)
class SessionEvaluation:
    success_rate: float
    average_cost_per_action: float
    retries: int
    failures: int
    total_actions: int
    stop_reason_counts: dict[str, int]


def evaluate_step(step: PlanStep, *, tool_result: ToolExecutionResult | None = None, assistant_text: str = "") -> EvaluationOutcome:
    if step.done_condition == "assistant_response_nonempty":
        passed = bool(assistant_text.strip())
        return EvaluationOutcome(passed=passed, confidence=1.0 if passed else 0.0, reason="assistant_response_nonempty", requires_retry=not passed, requires_replan=not passed)
    if step.done_condition == "reasoning_result_nonempty":
        passed = bool(assistant_text.strip())
        return EvaluationOutcome(passed=passed, confidence=0.9 if passed else 0.0, reason="reasoning_result_nonempty", requires_retry=not passed, requires_replan=not passed)
    if step.done_condition.startswith("tool_result:"):
        expected_tool = step.done_condition.split(":", 1)[1]
        passed = tool_result is not None and tool_result.tool_name == expected_tool and bool(tool_result.output)
        confidence = 0.95 if passed else 0.2
        return EvaluationOutcome(
            passed=passed,
            confidence=confidence,
            reason=f"tool_result:{expected_tool}",
            requires_retry=not passed,
            requires_replan=not passed,
        )
    return EvaluationOutcome(passed=False, confidence=0.0, reason=f"unknown_done_condition:{step.done_condition}", requires_retry=False, requires_replan=True)


def evaluate_verification(step: PlanStep, verification: VerificationOutcome) -> EvaluationOutcome:
    if verification.verification_passed and not verification.evidence:
        return EvaluationOutcome(
            passed=False,
            confidence=0.0,
            reason="missing_verification_evidence",
            requires_retry=False,
            requires_replan=True,
        )
    if verification.verification_passed and verification.conditions_failed:
        return EvaluationOutcome(
            passed=False,
            confidence=0.0,
            reason="inconsistent_verification_state",
            requires_retry=False,
            requires_replan=True,
        )
    if verification.verification_passed and not verification.conditions_met:
        return EvaluationOutcome(
            passed=False,
            confidence=0.0,
            reason="incomplete_verification_result",
            requires_retry=False,
            requires_replan=True,
        )
    if verification.verification_passed:
        return EvaluationOutcome(
            passed=True,
            confidence=verification.confidence,
            reason=verification.reason,
            requires_retry=False,
            requires_replan=False,
        )
    if verification.verification_type_used != "llm_fallback":
        return EvaluationOutcome(
            passed=False,
            confidence=verification.confidence,
            reason=verification.reason,
            requires_retry=verification.requires_retry,
            requires_replan=verification.requires_replan,
        )
    return EvaluationOutcome(
        passed=False,
        confidence=verification.confidence,
        reason=verification.reason,
        requires_retry=verification.requires_retry,
        requires_replan=verification.requires_replan,
    )


def evaluate_session_metrics(metrics: SessionMetrics) -> SessionEvaluation:
    completed_turns = metrics.successful_turns + metrics.failed_turns
    success_rate = 0.0 if completed_turns == 0 else metrics.successful_turns / completed_turns
    average_cost = 0.0 if metrics.action_count == 0 else metrics.total_cost_units / metrics.action_count
    failures = metrics.tool_failures + metrics.verification_failures + metrics.steps_failed
    return SessionEvaluation(
        success_rate=success_rate,
        average_cost_per_action=average_cost,
        retries=metrics.retries,
        failures=failures,
        total_actions=metrics.action_count,
        stop_reason_counts=dict(metrics.stop_reason_counts),
    )
