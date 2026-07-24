"""Prompt-analysis contract parsing.

The model owns semantic prompt analysis. This module only validates and
materialises model-returned ``prompt_analysis`` JSON payloads.
"""

from __future__ import annotations

from swaag.types import PromptAnalysis

_ALLOWED_TASK_TYPES = {"structured", "unstructured", "vague", "incomplete", "already_decomposed"}
_ALLOWED_COMPLETENESS = {"complete", "partial", "incomplete"}


class PromptAnalysisValidationError(ValueError):
    pass


def validate_analysis(analysis: PromptAnalysis) -> None:
    if analysis.task_type not in _ALLOWED_TASK_TYPES:
        raise PromptAnalysisValidationError(f"Unknown task_type: {analysis.task_type}")
    if analysis.completeness not in _ALLOWED_COMPLETENESS:
        raise PromptAnalysisValidationError(f"Unknown completeness: {analysis.completeness}")
    if not (0.0 <= float(analysis.confidence) <= 1.0):
        raise PromptAnalysisValidationError("confidence must be between 0 and 1")


def analysis_from_payload(payload: dict) -> PromptAnalysis:
    task_type = str(payload.get("task_type", "")).strip()
    completeness = str(payload.get("completeness", "")).strip()
    requires_expansion = bool(payload.get("requires_expansion"))
    requires_decomposition = bool(payload.get("requires_decomposition"))
    analysis = PromptAnalysis(
        task_type=task_type,  # type: ignore[arg-type]
        completeness=completeness,  # type: ignore[arg-type]
        requires_expansion=requires_expansion,
        requires_decomposition=requires_decomposition,
        confidence=float(payload.get("confidence", 0.0)),
        missing_required_information=bool(payload.get("missing_required_information", False)),
        detected_entities=[str(item).strip()[:96] for item in payload.get("detected_entities", []) if str(item).strip()][:8],
        detected_goals=[str(item).strip()[:96] for item in payload.get("detected_goals", []) if str(item).strip()][:4],
    )
    validate_analysis(analysis)
    return analysis
