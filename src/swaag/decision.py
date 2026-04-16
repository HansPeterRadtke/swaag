"""Task decision.

The semantic decision (whether a prompt should be split, expanded, or
clarified with the user) is made by the LLM through the
``task_decision`` JSON-schema contract. This module exists only to:

* parse and validate LLM-returned decisions (:func:`decision_from_payload`),
* provide a tiny deterministic emergency fallback
  (:func:`decide_from_analysis_emergency_fallback`)
  used when the LLM call fails outright. The fallback simply mirrors what the
  fallback :class:`~swaag.types.PromptAnalysis` already says — it does
  *not* re-implement semantic classification.
"""

from __future__ import annotations

from swaag.types import DecisionOutcome, PromptAnalysis


class DecisionValidationError(ValueError):
    pass


def decide_from_analysis_emergency_fallback(analysis: PromptAnalysis) -> DecisionOutcome:
    """Tiny deterministic fallback used only when the LLM call fails."""

    decision = DecisionOutcome(
        split_task=analysis.requires_decomposition,
        expand_task=analysis.requires_expansion,
        ask_user=analysis.completeness == "incomplete" and analysis.task_type != "vague",
        assume_missing=analysis.task_type == "vague",
        generate_ideas=analysis.task_type in {"vague", "unstructured"},
        confidence=analysis.confidence,
        reason=f"fallback;task_type={analysis.task_type};completeness={analysis.completeness}",
        direct_response=False,
        execution_mode="full_plan",
        preferred_tool_name="",
    )
    validate_decision(analysis, decision)
    return decision


# NOTE: there is intentionally no `decide_from_analysis = ...` alias.
# Callers must explicitly import the emergency fallback. The main
# (LLM-driven) decision path lives in
# :meth:`swaag.runtime.AgentRuntime._decide_prompt_frontend`.


def validate_decision(analysis: PromptAnalysis, decision: DecisionOutcome) -> None:
    if decision.ask_user and decision.assume_missing:
        raise DecisionValidationError("Decision cannot both ask the user and assume missing details")
    if decision.direct_response and (decision.split_task or decision.expand_task or decision.ask_user):
        raise DecisionValidationError("Direct responses cannot also request planning, expansion, or clarification")
    if decision.direct_response and decision.execution_mode != "direct_response":
        raise DecisionValidationError("Direct responses must use execution_mode='direct_response'")
    if decision.execution_mode == "direct_response" and not decision.direct_response:
        raise DecisionValidationError("execution_mode='direct_response' requires direct_response=true")
    if decision.execution_mode == "single_tool":
        if decision.direct_response:
            raise DecisionValidationError("single_tool execution cannot also be a direct response")
        if not decision.preferred_tool_name:
            raise DecisionValidationError("single_tool execution requires a preferred tool name")
    elif decision.preferred_tool_name:
        raise DecisionValidationError("preferred_tool_name must be empty unless execution_mode='single_tool'")
    if analysis.task_type == "vague" and not decision.expand_task:
        raise DecisionValidationError("Vague prompts must request expansion")
    if analysis.task_type == "structured" and decision.ask_user:
        raise DecisionValidationError("Structured prompts must not force clarification")


def decision_from_payload(payload: dict, analysis: PromptAnalysis) -> DecisionOutcome:
    decision = DecisionOutcome(
        split_task=bool(payload.get("split_task")),
        expand_task=bool(payload.get("expand_task")),
        ask_user=bool(payload.get("ask_user")),
        assume_missing=bool(payload.get("assume_missing")),
        generate_ideas=bool(payload.get("generate_ideas")),
        confidence=float(payload.get("confidence", 0.0)),
        reason=str(payload.get("reason", "")).strip(),
        direct_response=bool(payload.get("direct_response", False)),
        execution_mode=str(payload.get("execution_mode", "direct_response" if payload.get("direct_response", False) else "full_plan")),
        preferred_tool_name=str(payload.get("preferred_tool_name", "")).strip(),
    )
    if not decision.reason:
        raise DecisionValidationError("Decision reason must not be empty")
    if not (0.0 <= float(decision.confidence) <= 1.0):
        raise DecisionValidationError("Decision confidence must be between 0 and 1")
    validate_decision(analysis, decision)
    return decision
