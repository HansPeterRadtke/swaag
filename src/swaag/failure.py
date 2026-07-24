"""Failure classification.

Semantic failure classification (what kind of error this is, whether to
retry, whether to replan, how long to wait) is the LLM's job. The LLM is
asked, via the ``failure_classification`` JSON-schema contract, to label a
failure with one of the kinds defined in :data:`swaag.types.FailureKind`
and to set the mechanical policy fields.

This module provides :func:`classify_failure_from_payload`, a strict parser
and validator for the LLM response. The LLM is the only allowed source of
semantic failure classification and recovery policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

from swaag.types import FailureKind


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


__all__ = [
    "FailureClassification",
    "FailureValidationError",
    "classify_failure_from_payload",
]
