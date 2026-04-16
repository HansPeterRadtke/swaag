"""Failure classification.

Semantic failure classification (what kind of error this is, whether to
retry, whether to replan, how long to wait) is the LLM's job. The LLM is
asked, via the ``failure_classification`` JSON-schema contract, to label a
failure with one of the kinds defined in :data:`swaag.types.FailureKind`
and to set the mechanical policy fields.

This module provides:

* :func:`classify_failure_from_payload` — strict parser/validator for the
  LLM response. The LLM is the only allowed source of semantic
  classification.
* :func:`classify_failure_emergency_fallback` — an emergency-only
  deterministic fallback used when the LLM call itself failed. It does NOT
  inspect error strings or reason text for keywords; it returns a single
  neutral ``reasoning_failure`` classification that is non-retryable and
  forces a replan, so the loop fails safely instead of silently retrying.
* :data:`FAILURE_KIND_DEFAULTS` — a deterministic policy lookup that maps a
  semantic kind to the structural retry/replan/wait defaults. Mechanical
  decisions like "transient_external_wait => retryable=True" are not
  semantic; they are mandated by the kind itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

from swaag.types import FailureKind, PlanStep


class FailureValidationError(ValueError):
    pass


@dataclass(slots=True)
class FailureClassification:
    kind: FailureKind
    retryable: bool
    requires_replan: bool
    suggested_strategy_mode: str
    reason: str
    wait_seconds: float = 0.0
    source: str = "llm"


_ALLOWED_KINDS = set(get_args(FailureKind))
_ALLOWED_MODES = {"conservative", "recovery", "verification_heavy"}


@dataclass(frozen=True, slots=True)
class _KindPolicy:
    retryable: bool
    requires_replan: bool
    suggested_strategy_mode: str
    wait_seconds: float


# Deterministic, non-semantic policy table. The LLM picks the *kind*; the
# mechanical consequences of that kind are spelled out here so they cannot
# drift between callers. These are not heuristics — they are the definition
# of what each kind means in policy terms.
FAILURE_KIND_DEFAULTS: dict[str, _KindPolicy] = {
    "tool_failure": _KindPolicy(False, True, "conservative", 0.0),
    "reasoning_failure": _KindPolicy(True, False, "recovery", 0.0),
    "planning_failure": _KindPolicy(False, True, "recovery", 0.0),
    "missing_information": _KindPolicy(False, True, "conservative", 0.0),
    "verification_failure": _KindPolicy(True, False, "verification_heavy", 0.0),
    "budget_failure": _KindPolicy(False, True, "conservative", 0.0),
    "state_inconsistency": _KindPolicy(False, True, "recovery", 0.0),
    "transient_external_wait": _KindPolicy(True, False, "conservative", 5.0),
    "retry_now": _KindPolicy(True, False, "recovery", 0.0),
    "retry_later_backoff": _KindPolicy(True, False, "conservative", 15.0),
    "deterministic_permanent": _KindPolicy(False, True, "conservative", 0.0),
    "side_effect_unsafe": _KindPolicy(False, True, "conservative", 0.0),
    "needs_replan": _KindPolicy(False, True, "recovery", 0.0),
    "needs_clarification": _KindPolicy(False, True, "conservative", 0.0),
    "blocked_external": _KindPolicy(False, True, "conservative", 0.0),
    "continue_other": _KindPolicy(False, False, "recovery", 0.0),
}


def policy_for_kind(kind: str) -> _KindPolicy:
    if kind not in FAILURE_KIND_DEFAULTS:
        raise FailureValidationError(f"Unknown failure kind: {kind}")
    return FAILURE_KIND_DEFAULTS[kind]


def classify_failure_from_payload(payload: dict) -> FailureClassification:
    """Parse and validate an LLM ``failure_classification`` response.

    The LLM may override the policy defaults (for example, marking a
    ``tool_failure`` as ``retryable=True`` because it has identified a
    transient cause), but the kind must be valid and the strategy mode must
    be one of the allowed values.
    """

    kind = str(payload.get("kind", "")).strip()
    if kind not in _ALLOWED_KINDS:
        raise FailureValidationError(f"Unknown failure kind: {kind}")
    mode = str(payload.get("suggested_strategy_mode", "")).strip()
    if mode not in _ALLOWED_MODES:
        raise FailureValidationError(f"Unknown suggested_strategy_mode: {mode}")
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise FailureValidationError("Failure classification reason must not be empty")
    try:
        wait_seconds = float(payload.get("wait_seconds", 0.0))
    except (TypeError, ValueError) as exc:
        raise FailureValidationError("wait_seconds must be a number") from exc
    if wait_seconds < 0.0:
        raise FailureValidationError("wait_seconds must be non-negative")
    return FailureClassification(
        kind=kind,  # type: ignore[arg-type]
        retryable=bool(payload.get("retryable", False)),
        requires_replan=bool(payload.get("requires_replan", False)),
        suggested_strategy_mode=mode,
        reason=reason,
        wait_seconds=wait_seconds,
        source="llm",
    )


def classify_failure_emergency_fallback(
    *,
    step: PlanStep | None,
    error: Exception | None = None,
    error_type: str | None = None,
    reason: str = "",
) -> FailureClassification:
    """Emergency deterministic fallback. EMERGENCY USE ONLY.

    This function is invoked only when the LLM-driven failure classifier
    itself fails (network error, contract validation error, etc.). It does
    NOT inspect error strings, reason text, exception class names, or any
    other content for semantic content. It returns a single neutral
    classification that is safe under all circumstances:

    * non-retryable (so we do not silently spin on a failure of unknown
      cause),
    * requires replan (so the planner sees the failure and reconsiders),
    * conservative strategy mode (so the next attempt is cautious),
    * source="deterministic_fallback_emergency_only" so traces and tests
      can detect when the LLM path has been bypassed.
    """

    del step, error, error_type
    return FailureClassification(
        kind="reasoning_failure",
        retryable=False,
        requires_replan=True,
        suggested_strategy_mode="conservative",
        reason=reason or "llm_failure_classifier_unavailable",
        wait_seconds=0.0,
        source="deterministic_fallback_emergency_only",
    )

