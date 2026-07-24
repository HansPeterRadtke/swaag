"""Task-decision contract parsing.

The model owns whether a user request should be answered directly, planned,
expanded, clarified, or handled through a single tool. This module only
validates and materialises model-returned ``task_decision`` JSON payloads.
"""

from __future__ import annotations

from swaag.types import DecisionOutcome


class DecisionValidationError(ValueError):
    pass


def validate_decision(decision: DecisionOutcome) -> None:
    if decision.direct_response and (decision.split_task or decision.expand_task or decision.ask_user):
        raise DecisionValidationError("Direct responses cannot also request planning, expansion, or clarification")
    if decision.direct_response and decision.execution_mode != "direct_response":
        raise DecisionValidationError("Direct responses must use execution_mode='direct_response'")
    if decision.execution_mode == "direct_response" and not decision.direct_response:
        raise DecisionValidationError("execution_mode='direct_response' requires direct_response=true")
    if decision.evidence_call_count < 0:
        raise DecisionValidationError("evidence_call_count must be a non-negative integer")
    if decision.evidence_required_before_response:
        if decision.evidence_call_count < 1:
            raise DecisionValidationError(
                "evidence_required_before_response=true requires evidence_call_count>=1"
            )
        if decision.execution_mode in {"direct_response", "clarification"}:
            raise DecisionValidationError(
                "evidence_required_before_response=true requires execution_mode='full_plan' or 'single_tool'"
            )
        if decision.evidence_call_count > 1 and decision.execution_mode != "full_plan":
            raise DecisionValidationError(
                "evidence_call_count>1 requires execution_mode='full_plan'"
            )
    elif decision.evidence_call_count != 0:
        raise DecisionValidationError(
            "evidence_required_before_response=false requires evidence_call_count=0"
        )
    if decision.execution_mode == "single_tool":
        if decision.direct_response:
            raise DecisionValidationError("single_tool execution cannot also be a direct response")
        if not decision.preferred_tool_name:
            raise DecisionValidationError("single_tool execution requires a preferred tool name")
    elif decision.execution_mode == "clarification":
        if not decision.ask_user:
            raise DecisionValidationError("execution_mode='clarification' requires ask_user=true")
        if decision.direct_response:
            raise DecisionValidationError("clarification execution cannot also be a direct response")
        if decision.preferred_tool_name:
            raise DecisionValidationError("clarification execution cannot declare a preferred tool")
    elif decision.preferred_tool_name:
        raise DecisionValidationError("preferred_tool_name must be empty unless execution_mode='single_tool'")
    if not decision.reason.strip():
        raise DecisionValidationError("Decision reason must not be empty")


def decision_from_payload(payload: dict) -> DecisionOutcome:
    split_task = bool(payload.get("split_task"))
    expand_task = bool(payload.get("expand_task"))
    ask_user = bool(payload.get("ask_user"))
    direct_response = bool(payload.get("direct_response", False))
    execution_mode = str(payload.get("execution_mode", "direct_response" if direct_response else "full_plan"))
    preferred_tool_name = str(payload.get("preferred_tool_name", "")).strip()
    decision = DecisionOutcome(
        split_task=split_task,
        expand_task=expand_task,
        ask_user=ask_user,
        assume_missing=bool(payload.get("assume_missing")),
        generate_ideas=bool(payload.get("generate_ideas")),
        confidence=float(payload.get("confidence", 0.0)),
        reason=str(payload.get("reason", "")).strip(),
        direct_response=direct_response,
        execution_mode=execution_mode,
        preferred_tool_name=preferred_tool_name,
        evidence_required_before_response=bool(payload.get("evidence_required_before_response", False)),
        evidence_call_count=int(payload.get("evidence_call_count", 0)),
    )
    if not (0.0 <= float(decision.confidence) <= 1.0):
        raise DecisionValidationError("Decision confidence must be between 0 and 1")
    validate_decision(decision)
    return decision
