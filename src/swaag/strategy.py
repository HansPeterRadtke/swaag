"""Strategy-selection contract parsing.

The model owns strategy choice. This module materialises model-returned
strategy labels and numeric controls without mapping those labels to fixed
semantic workflows, file roles, or tool sets.
"""

from __future__ import annotations

from dataclasses import replace

from swaag.failure import FailureClassification
from swaag.types import Plan, SessionMetrics, StrategySelection


class StrategyValidationError(ValueError):
    pass


_PROFILE_CATALOG: dict[str, dict] = {
    "coding": {
        "strategy_name": "exploratory",
        "mode": "exploratory",
        "tool_chain_depth": 2,
        "verification_intensity": 0.95,
        "retry_same_action_limit": 1,
        "replan_after_failures": 2,
        "confidence_floor": 0.45,
        "explore_before_commit": True,
    },
    "file_edit": {
        "strategy_name": "conservative",
        "mode": "conservative",
        "tool_chain_depth": 1,
        "verification_intensity": 0.95,
        "retry_same_action_limit": 1,
        "replan_after_failures": 1,
        "confidence_floor": 0.55,
        "explore_before_commit": False,
    },
    "reading": {
        "strategy_name": "conservative",
        "mode": "conservative",
        "tool_chain_depth": 1,
        "verification_intensity": 0.9,
        "retry_same_action_limit": 1,
        "replan_after_failures": 1,
        "confidence_floor": 0.6,
        "explore_before_commit": False,
    },
    "multi_step": {
        "strategy_name": "exploratory",
        "mode": "exploratory",
        "tool_chain_depth": 2,
        "verification_intensity": 1.0,
        "retry_same_action_limit": 1,
        "replan_after_failures": 2,
        "confidence_floor": 0.5,
        "explore_before_commit": True,
    },
    "generic": {
        "strategy_name": "conservative",
        "mode": "conservative",
        "tool_chain_depth": 1,
        "verification_intensity": 0.8,
        "retry_same_action_limit": 1,
        "replan_after_failures": 1,
        "confidence_floor": 0.6,
        "explore_before_commit": False,
    },
}

def available_profiles() -> list[str]:
    return list(_PROFILE_CATALOG.keys())


def build_strategy_from_profile(profile_name: str, *, reason: str) -> StrategySelection:
    if profile_name not in _PROFILE_CATALOG:
        raise StrategyValidationError(f"Unknown task_profile: {profile_name}")
    profile = _PROFILE_CATALOG[profile_name]
    return StrategySelection(
        strategy_name=profile["strategy_name"],
        mode=profile["mode"],
        explore_before_commit=profile["explore_before_commit"],
        validate_assumptions=True,
        simplify_if_stuck=True,
        switch_on_failure=True,
        reason=reason,
        tool_chain_depth=profile["tool_chain_depth"],
        verification_intensity=profile["verification_intensity"],
        retry_same_action_limit=profile["retry_same_action_limit"],
        replan_after_failures=profile["replan_after_failures"],
        confidence_floor=profile["confidence_floor"],
        task_profile=profile_name,
        required_step_kinds=[],
        expected_flow=[],
    )


def strategy_from_payload(payload: dict) -> StrategySelection:
    """Parse and validate an LLM strategy_selection response."""

    profile_name = str(payload.get("task_profile", "")).strip() or "generic"
    if profile_name not in _PROFILE_CATALOG:
        raise StrategyValidationError(f"Unknown task_profile: {profile_name}")
    base = build_strategy_from_profile(profile_name, reason=str(payload.get("reason", "")).strip() or f"profile={profile_name}")
    strategy_name = str(payload.get("strategy_name", "")).strip() or base.strategy_name
    if strategy_name not in {"conservative", "exploratory", "recovery", "verification_heavy"}:
        raise StrategyValidationError(f"Unknown strategy_name: {strategy_name}")
    explore = bool(payload.get("explore_before_commit", base.explore_before_commit))
    try:
        tool_chain_depth = int(payload.get("tool_chain_depth", base.tool_chain_depth))
    except (TypeError, ValueError) as exc:
        raise StrategyValidationError(f"tool_chain_depth must be int: {exc}") from exc
    if not 1 <= tool_chain_depth <= 3:
        raise StrategyValidationError("tool_chain_depth must be between 1 and 3")
    try:
        verification_intensity = float(payload.get("verification_intensity", base.verification_intensity))
    except (TypeError, ValueError) as exc:
        raise StrategyValidationError(f"verification_intensity must be number: {exc}") from exc
    return replace(
        base,
        strategy_name=strategy_name,
        mode=strategy_name,
        explore_before_commit=explore,
        tool_chain_depth=tool_chain_depth,
        verification_intensity=verification_intensity,
    )


def adapt_strategy(
    current: StrategySelection,
    *,
    failure: FailureClassification | None,
    metrics: SessionMetrics,
    verification_failed: bool,
) -> StrategySelection:
    del metrics, verification_failed
    if failure is None:
        return current
    suggested = failure.suggested_strategy_mode
    if suggested not in {"conservative", "recovery", "verification_heavy"}:
        return current
    if current.strategy_name == suggested and current.mode == suggested:
        return current
    return replace(
        current,
        strategy_name=suggested,
        mode=suggested,
        reason=f"failure_classifier_suggested:{failure.kind};{failure.reason}",
    )

def validate_plan_against_strategy(
    plan: Plan,
    strategy: StrategySelection,
    *,
    completed_step_kinds=(),
) -> None:
    del plan, strategy, completed_step_kinds
