"""Strategy selection.

The semantic part of strategy selection — *which* execution profile fits a
given goal — is performed by the LLM via the
:func:`swaag.grammar.strategy_selection_contract` schema. This module
provides:

* a small static catalogue of profiles (each is just a name plus its
  *deterministic structural constraints*: allowed tools, required step kinds,
  expected flow). This is configuration data, not classification logic.
* :func:`build_strategy_from_profile` / :func:`strategy_from_payload` to
  materialise a :class:`StrategySelection` from a profile name (chosen by the
  LLM) or a parsed LLM payload.
* :func:`select_strategy_emergency_default` for emergency-only callers that have no
  LLM context yet; it returns the safe ``generic`` profile and is not used
  on the normal runtime path.
* :func:`adapt_strategy` and :func:`validate_plan_against_strategy`, which are
  deterministic structural operations and are kept as-is.

The previous trigger-word classifier (``infer_task_profile`` /
``_contains_token``) has been removed: profiles are no longer chosen by
matching hard-coded vocabulary against the prompt text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from swaag.failure import FailureClassification
from swaag.types import Plan, SessionMetrics, StrategySelection


class StrategyValidationError(ValueError):
    pass


_PROFILE_CATALOG: dict[str, dict] = {
    "coding": {
        "allowed_tools": [
            "read_text", "read_file", "edit_text", "write_file", "run_tests",
            "shell_command", "browser_search", "browser_browse", "notes", "echo", "calculator",
        ],
        "required_step_kinds": ["read", "write", "respond"],
        "expected_flow": ["read", "write", "respond"],
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
        "allowed_tools": [
            "read_text", "read_file", "edit_text", "write_file", "list_files", "notes", "echo",
        ],
        "required_step_kinds": ["write", "respond"],
        "expected_flow": ["write", "respond"],
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
        "allowed_tools": [
            "read_text", "read_file", "list_files", "browser_search", "browser_browse",
            "notes", "echo", "calculator",
        ],
        "required_step_kinds": ["read", "respond"],
        "expected_flow": ["read", "respond"],
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
        "allowed_tools": [
            "read_text", "read_file", "edit_text", "write_file", "list_files", "run_tests",
            "shell_command", "browser_search", "browser_browse", "notes", "echo", "calculator",
        ],
        "required_step_kinds": ["read", "write", "respond"],
        "expected_flow": ["read", "write", "respond"],
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
        "allowed_tools": [
            "echo", "time_now", "calculator", "read_text", "read_file", "list_files",
            "browser_search", "browser_browse", "notes", "edit_text", "write_file", "run_tests",
            "shell_command",
        ],
        "required_step_kinds": ["respond"],
        "expected_flow": ["respond"],
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

_BROAD_RECONCILIATION_PROFILES: frozenset[str] = frozenset({"generic", "multi_step"})


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
        allowed_tools=list(profile["allowed_tools"]),
        required_step_kinds=list(profile["required_step_kinds"]),
        expected_flow=list(profile["expected_flow"]),
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
        mode=base.mode if strategy_name in {"conservative", "exploratory"} else strategy_name,
        explore_before_commit=explore,
        tool_chain_depth=tool_chain_depth,
        verification_intensity=verification_intensity,
    )


def select_strategy_emergency_default(reason: str = "no_llm_context") -> StrategySelection:
    """Safe default returned when no LLM-derived selection is available yet."""

    profile = _PROFILE_CATALOG["generic"]
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
        task_profile="generic",
        allowed_tools=list(profile["allowed_tools"]),
        required_step_kinds=list(profile["required_step_kinds"]),
        expected_flow=list(profile["expected_flow"]),
    )

def adapt_strategy(
    current: StrategySelection,
    *,
    failure: FailureClassification | None,
    metrics: SessionMetrics,
    verification_failed: bool,
) -> StrategySelection:
    if failure is not None:
        if failure.suggested_strategy_mode == "recovery":
            return replace(
                current,
                strategy_name="recovery",
                mode="recovery",
                explore_before_commit=False,
                verification_intensity=1.0,
                retry_same_action_limit=0,
                replan_after_failures=1,
                confidence_floor=max(current.confidence_floor, 0.7),
                reason=f"failure={failure.kind}",
            )
        if failure.suggested_strategy_mode == "verification_heavy":
            return replace(
                current,
                strategy_name="verification_heavy",
                mode="verification_heavy",
                explore_before_commit=False,
                verification_intensity=1.2,
                retry_same_action_limit=1,
                replan_after_failures=1,
                confidence_floor=max(current.confidence_floor, 0.8),
                reason=f"failure={failure.kind}",
            )
    if verification_failed or metrics.verification_failures > metrics.verification_passes:
        return replace(
            current,
            strategy_name="verification_heavy",
            mode="verification_heavy",
            explore_before_commit=False,
            verification_intensity=1.2,
            retry_same_action_limit=1,
            replan_after_failures=1,
            confidence_floor=max(current.confidence_floor, 0.75),
            reason="verification_pressure",
        )
    return current


def _semantic_step_kinds(kind: str, expected_tool: str | None) -> list[str]:
    semantic = [kind]
    if expected_tool in {"read_text", "read_file", "list_files", "browser_search", "browser_browse"}:
        semantic.append("read")
    if expected_tool in {"edit_text", "write_file"}:
        semantic.append("write")
    if kind == "tool" and expected_tool == "calculator":
        semantic.append("read")
    return semantic


def _effective_step_kinds(plan: Plan, *, completed_step_kinds: Sequence[str] = ()) -> list[str]:
    step_kinds: list[str] = list(completed_step_kinds)
    for step in plan.steps:
        step_kinds.extend(_semantic_step_kinds(step.kind, step.expected_tool))
    return step_kinds


def _flow_matches(effective_step_kinds: Sequence[str], expected_flow: Sequence[str]) -> bool:
    if not expected_flow:
        return True
    current_index = 0
    for kind in effective_step_kinds:
        if current_index < len(expected_flow) and kind == expected_flow[current_index]:
            current_index += 1
    return current_index >= len(expected_flow)


def _structural_profile_rank(
    profile_name: str,
    effective_step_kinds: Sequence[str],
    *,
    current_profile: str,
) -> tuple[int, int, int, int, int, int, str]:
    profile = _PROFILE_CATALOG[profile_name]
    structural_kinds = [kind for kind in effective_step_kinds if kind in {"read", "write", "respond"}]
    ordered_unique: list[str] = []
    for kind in structural_kinds:
        if not ordered_unique or ordered_unique[-1] != kind:
            ordered_unique.append(kind)
    exact_required = int(set(profile["required_step_kinds"]) == set(ordered_unique))
    exact_flow = int(list(profile["expected_flow"]) == ordered_unique)
    required_count = len(profile["required_step_kinds"])
    expected_flow_count = len(profile["expected_flow"])
    allowed_tool_count = len(profile["allowed_tools"])
    current_bonus = 1 if profile_name == current_profile else 0
    return (
        exact_flow,
        exact_required,
        required_count,
        expected_flow_count,
        -allowed_tool_count,
        current_bonus,
        profile_name,
    )


def _preserves_semantic_commitment(current_profile: str, candidate_profile: str) -> bool:
    if current_profile in _BROAD_RECONCILIATION_PROFILES:
        return True
    current_required = set(_PROFILE_CATALOG[current_profile]["required_step_kinds"])
    candidate_required = set(_PROFILE_CATALOG[candidate_profile]["required_step_kinds"])
    return current_required.issubset(candidate_required)


def reconcile_strategy_to_plan(
    strategy: StrategySelection,
    plan: Plan,
    *,
    completed_step_kinds: Sequence[str] = (),
) -> StrategySelection:
    effective_step_kinds = _effective_step_kinds(plan, completed_step_kinds=completed_step_kinds)
    plan_tools = {
        step.expected_tool
        for step in plan.steps
        if step.expected_tool not in {None, ""}
    }
    compatible_profiles: list[str] = []
    for profile_name, profile in _PROFILE_CATALOG.items():
        if any(kind not in effective_step_kinds for kind in profile["required_step_kinds"]):
            continue
        if not _flow_matches(effective_step_kinds, profile["expected_flow"]):
            continue
        if any(tool not in profile["allowed_tools"] for tool in plan_tools):
            continue
        if not _preserves_semantic_commitment(strategy.task_profile, profile_name):
            continue
        compatible_profiles.append(profile_name)
    if not compatible_profiles:
        raise StrategyValidationError(
            "No compatible strategy profile matches plan structure while preserving "
            f"the semantic commitment of profile {strategy.task_profile}"
        )
    best_profile = sorted(
        compatible_profiles,
        key=lambda name: _structural_profile_rank(
            name,
            effective_step_kinds,
            current_profile=strategy.task_profile,
        ),
        reverse=True,
    )[0]
    if best_profile == strategy.task_profile:
        return strategy
    base = build_strategy_from_profile(
        best_profile,
        reason=f"plan_reconciled:{strategy.task_profile}->{best_profile};{strategy.reason}",
    )
    strategy_name = strategy.strategy_name
    return replace(
        base,
        strategy_name=strategy_name,
        mode=base.mode if strategy_name in {"conservative", "exploratory"} else strategy_name,
        explore_before_commit=strategy.explore_before_commit,
        tool_chain_depth=min(strategy.tool_chain_depth, base.tool_chain_depth),
        verification_intensity=max(strategy.verification_intensity, base.verification_intensity),
        retry_same_action_limit=min(strategy.retry_same_action_limit, base.retry_same_action_limit),
        replan_after_failures=min(strategy.replan_after_failures, base.replan_after_failures),
        confidence_floor=max(strategy.confidence_floor, base.confidence_floor),
        reason=f"plan_reconciled:{strategy.task_profile}->{best_profile};{strategy.reason}",
    )


def validate_plan_against_strategy(
    plan: Plan,
    strategy: StrategySelection,
    *,
    completed_step_kinds: Sequence[str] = (),
) -> None:
    if not strategy.required_step_kinds:
        return
    effective_step_kinds = _effective_step_kinds(plan, completed_step_kinds=completed_step_kinds)
    missing_required = [kind for kind in strategy.required_step_kinds if kind not in effective_step_kinds]
    if missing_required:
        raise StrategyValidationError(
            f"Plan is missing required step kinds for profile {strategy.task_profile}: {', '.join(missing_required)}"
        )
    if strategy.allowed_tools:
        disallowed_tools = sorted(
            {
                step.expected_tool
                for step in plan.steps
                if step.expected_tool not in {None, ""} and step.expected_tool not in strategy.allowed_tools
            }
        )
        if disallowed_tools:
            raise StrategyValidationError(
                f"Plan uses tools outside the allowed set for profile {strategy.task_profile}: {', '.join(disallowed_tools)}"
            )
    expected_flow = list(strategy.expected_flow)
    if not _flow_matches(effective_step_kinds, expected_flow):
        raise StrategyValidationError(
            f"Plan does not follow expected flow for profile {strategy.task_profile}: expected subsequence {expected_flow}"
        )
