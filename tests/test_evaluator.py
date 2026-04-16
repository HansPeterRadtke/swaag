from __future__ import annotations

from swaag.evaluator import evaluate_session_metrics, evaluate_step, evaluate_verification
from swaag.types import PlanStep, SessionMetrics, ToolExecutionResult
from swaag.verification import VerificationOutcome


def _tool_step() -> PlanStep:
    return PlanStep(
        step_id="step_calc",
        title="Compute the value",
        goal="Compute the value",
        kind="tool",
        expected_tool="calculator",
        input_text="compute",
        expected_output="Calculated value",
        done_condition="tool_result:calculator",
        success_criteria="The calculator returns the value.",
    )


def test_evaluator_detects_incorrect_tool_result() -> None:
    evaluation = evaluate_step(_tool_step(), tool_result=None)

    assert evaluation.passed is False
    assert evaluation.requires_retry is True
    assert evaluation.requires_replan is True


def test_evaluator_accepts_correct_tool_result() -> None:
    result = ToolExecutionResult(tool_name="calculator", output={"expression": "2 + 2", "result": 4}, display_text="4")
    evaluation = evaluate_step(_tool_step(), tool_result=result)

    assert evaluation.passed is True
    assert evaluation.confidence == 0.95


def test_session_evaluation_summarizes_cost_and_success_rate() -> None:
    summary = evaluate_session_metrics(
        SessionMetrics(
            action_count=4,
            total_cost_units=6.0,
            retries=1,
            tool_failures=1,
            verification_failures=1,
            steps_failed=1,
            successful_turns=3,
            failed_turns=1,
            stop_reason_counts={"answered": 3, "tool_call_budget_reached": 1},
        )
    )

    assert summary.success_rate == 0.75
    assert summary.average_cost_per_action == 1.5
    assert summary.failures == 3
    assert summary.stop_reason_counts["answered"] == 3


def test_evaluator_cannot_override_failed_deterministic_verification() -> None:
    verification = VerificationOutcome(
        verification_passed=False,
        verification_type_used="composite",
        conditions_met=[],
        conditions_failed=["wrong_result"],
        evidence={"wrong_result": {"actual": 4, "expected": 5}},
        confidence=0.0,
        reason="wrong_result",
        requires_retry=True,
        requires_replan=False,
    )
    evaluation = evaluate_verification(_tool_step(), verification)

    assert evaluation.passed is False
    assert evaluation.requires_retry is True


def test_evaluator_rejects_pass_without_evidence() -> None:
    verification = VerificationOutcome(
        verification_passed=True,
        verification_type_used="composite",
        conditions_met=["result_ok"],
        conditions_failed=[],
        evidence={},
        confidence=1.0,
        reason="verified",
        requires_retry=False,
        requires_replan=False,
    )

    evaluation = evaluate_verification(_tool_step(), verification)

    assert evaluation.passed is False
    assert evaluation.reason == "missing_verification_evidence"


def test_evaluator_rejects_incomplete_verification_state() -> None:
    verification = VerificationOutcome(
        verification_passed=True,
        verification_type_used="composite",
        conditions_met=[],
        conditions_failed=[],
        evidence={"result_ok": {"actual": 4, "expected": 4}},
        confidence=1.0,
        reason="verified",
        requires_retry=False,
        requires_replan=False,
    )

    evaluation = evaluate_verification(_tool_step(), verification)

    assert evaluation.passed is False
    assert evaluation.reason == "incomplete_verification_result"
