"""Prompt analysis.

The semantic classification of a prompt (vague vs structured vs incomplete,
whether it requires expansion / decomposition, etc.) is performed by the LLM
through the :func:`swaag.grammar.prompt_analysis_contract` schema. This
module exists to:

* parse and validate LLM-returned analyses (:func:`analysis_from_payload`),
* provide a tiny *deterministic* emergency fallback
  (:func:`analyze_prompt_emergency_fallback`)
  used only when the LLM call fails outright. The fallback intentionally
  refuses to make a confident classification: it produces a neutral
  ``unstructured``/``partial`` analysis with extracted entities so the rest
  of the pipeline can keep running.

The heuristic word-lists (``_VAGUE_MARKERS``, ``_GOAL_VERBS``,
``_OPEN_ENDED_NOUNS``) and the trigger-word ``analyze_prompt`` classifier
that previously lived here have been removed; classification is no longer
allowed to be derived from hard-coded vocabulary lists.
"""

from __future__ import annotations

import re

from swaag.types import PromptAnalysis

_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
_PATH_HINT_RE = re.compile(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+")

_ALLOWED_TASK_TYPES = {"structured", "unstructured", "vague", "incomplete", "already_decomposed"}
_ALLOWED_COMPLETENESS = {"complete", "partial", "incomplete"}


class PromptAnalysisValidationError(ValueError):
    pass


def _detect_entities(text: str) -> list[str]:
    detected: list[str] = []
    for token in _WORD_RE.findall(text):
        stripped = token.rstrip(".,!?;:")
        if not stripped:
            continue
        if "/" in stripped or _PATH_HINT_RE.search(stripped) is not None:
            if stripped not in detected:
                detected.append(stripped)
        if len(detected) >= 8:
            break
    return detected


def analyze_prompt_emergency_fallback(text: str) -> PromptAnalysis:
    """Tiny deterministic fallback used only if the LLM call fails.

    No semantic word lists. No trigger-term classification. Returns a
    neutral analysis that lets the rest of the pipeline keep going.
    """

    stripped = text.strip()
    if not stripped:
        return PromptAnalysis(
            task_type="incomplete",
            completeness="incomplete",
            requires_expansion=True,
            requires_decomposition=False,
            confidence=0.5,
            detected_entities=[],
            detected_goals=[],
        )
    detected_entities = _detect_entities(stripped)
    return PromptAnalysis(
        task_type="unstructured",
        completeness="partial",
        requires_expansion=True,
        requires_decomposition=True,
        confidence=0.5,
        detected_entities=detected_entities,
        detected_goals=[],
    )


# NOTE: there is intentionally no `analyze_prompt = ...` alias. Callers
# must explicitly opt into the deterministic emergency fallback by
# importing :func:`analyze_prompt_emergency_fallback`. The main
# (LLM-driven) path lives in
# :meth:`swaag.runtime.AgentRuntime._analyze_prompt_frontend`.


def validate_analysis(analysis: PromptAnalysis) -> None:
    if analysis.task_type not in _ALLOWED_TASK_TYPES:
        raise PromptAnalysisValidationError(f"Unknown task_type: {analysis.task_type}")
    if analysis.completeness not in _ALLOWED_COMPLETENESS:
        raise PromptAnalysisValidationError(f"Unknown completeness: {analysis.completeness}")
    if not (0.0 <= float(analysis.confidence) <= 1.0):
        raise PromptAnalysisValidationError("confidence must be between 0 and 1")
    if analysis.task_type == "vague" and not analysis.requires_expansion:
        raise PromptAnalysisValidationError("vague prompts must require expansion")
    if analysis.task_type == "already_decomposed" and not analysis.requires_decomposition:
        raise PromptAnalysisValidationError("already_decomposed prompts must require decomposition")
    if analysis.completeness == "complete" and analysis.task_type == "incomplete":
        raise PromptAnalysisValidationError("incomplete task_type cannot have completeness=complete")


def analysis_from_payload(payload: dict) -> PromptAnalysis:
    analysis = PromptAnalysis(
        task_type=str(payload.get("task_type", "")).strip(),  # type: ignore[arg-type]
        completeness=str(payload.get("completeness", "")).strip(),  # type: ignore[arg-type]
        requires_expansion=bool(payload.get("requires_expansion")),
        requires_decomposition=bool(payload.get("requires_decomposition")),
        confidence=float(payload.get("confidence", 0.0)),
        detected_entities=[str(item).strip() for item in payload.get("detected_entities", []) if str(item).strip()],
        detected_goals=[str(item).strip() for item in payload.get("detected_goals", []) if str(item).strip()],
    )
    validate_analysis(analysis)
    return analysis
