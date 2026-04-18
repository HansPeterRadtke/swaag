from __future__ import annotations

from typing import Any


TASK_SCORE_COMPONENT_WEIGHTS: dict[str, float] = {
    "final_outcome": 50.0,
    "verification_contract": 30.0,
    "quality_and_planning": 20.0,
}


def _fraction_from_checks(summary: dict[str, Any], *, passed_key: str = "passed") -> float:
    checks = summary.get("checks", {})
    if isinstance(checks, dict) and checks:
        total = len(checks)
        passed = sum(1 for value in checks.values() if bool(value))
        return passed / total
    if passed_key in summary:
        return 1.0 if bool(summary.get(passed_key)) else 0.0
    return 1.0


def build_task_rubric(
    *,
    success: bool,
    verification_summary: dict[str, Any],
    quality_summary: dict[str, Any],
) -> tuple[float, dict[str, dict[str, Any]]]:
    verification_fraction = _fraction_from_checks(verification_summary)
    quality_fraction = _fraction_from_checks(quality_summary)
    outcome_fraction = 1.0 if success else 0.0
    components = {
        "final_outcome": {
            "weight": TASK_SCORE_COMPONENT_WEIGHTS["final_outcome"],
            "fraction": outcome_fraction,
            "earned": TASK_SCORE_COMPONENT_WEIGHTS["final_outcome"] * outcome_fraction,
            "reason": "final verifier accepted the task" if success else "final verifier did not accept the task",
        },
        "verification_contract": {
            "weight": TASK_SCORE_COMPONENT_WEIGHTS["verification_contract"],
            "fraction": verification_fraction,
            "earned": TASK_SCORE_COMPONENT_WEIGHTS["verification_contract"] * verification_fraction,
            "reason": verification_summary.get("reason", ""),
        },
        "quality_and_planning": {
            "weight": TASK_SCORE_COMPONENT_WEIGHTS["quality_and_planning"],
            "fraction": quality_fraction,
            "earned": TASK_SCORE_COMPONENT_WEIGHTS["quality_and_planning"] * quality_fraction,
            "reason": "quality oracle satisfied" if quality_summary.get("passed", True) else "quality oracle mismatch",
        },
    }
    total = round(sum(float(item["earned"]) for item in components.values()), 2)
    for item in components.values():
        weight = float(item["weight"])
        earned = float(item["earned"])
        item["percent"] = 0.0 if weight <= 0 else round((earned / weight) * 100.0, 2)
    return total, components
