from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from swaag.schema_portability import assert_portable_json_schema
from swaag.types import ContractSpec


def _string() -> dict[str, Any]:
    return {"type": "string"}


def _number() -> dict[str, Any]:
    return {"type": "number"}


def _integer() -> dict[str, Any]:
    return {"type": "integer"}


def _boolean() -> dict[str, Any]:
    return {"type": "boolean"}


def _array(item_schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": item_schema}


def _closed_object(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def _contract(name: str, schema: dict[str, Any]) -> ContractSpec:
    assert_portable_json_schema(schema, schema_name=name)
    return ContractSpec(name=name, mode="json_schema", json_schema=schema)


def text_response_contract(name: str = "text_response") -> ContractSpec:
    return _contract(name, _closed_object({"text": _string()}))


def yes_no_contract() -> ContractSpec:
    return _contract("yes_no", _closed_object({"answer": {"type": "string", "enum": ["yes", "no"]}}))


def tool_decision_contract(tool_names: Iterable[str]) -> ContractSpec:
    names = sorted(tool_names)
    schema = _closed_object(
        {
            "action": {"type": "string", "enum": ["respond", "call_tool"]},
            "response": _string(),
            "tool_name": {"type": "string", "enum": ["none", *names]},
            "tool_input": _closed_object({}),
        }
    )
    return _contract("tool_decision", schema)


def tool_input_contract(tool_name: str, input_schema: dict[str, Any]) -> ContractSpec:
    schema = deepcopy(input_schema)
    return _contract(f"tool_input:{tool_name}", schema)


def prompt_analysis_contract() -> ContractSpec:
    schema = _closed_object(
        {
            "task_type": {"type": "string", "enum": ["structured", "unstructured", "vague", "incomplete", "already_decomposed"]},
            "completeness": {"type": "string", "enum": ["complete", "partial", "incomplete"]},
            "requires_expansion": _boolean(),
            "requires_decomposition": _boolean(),
            "missing_required_information": _boolean(),
            "confidence": _number(),
            "detected_entities": _array(_string()),
            "detected_goals": _array(_string()),
        }
    )
    return _contract("prompt_analysis", schema)


def task_decision_contract(tool_names: Iterable[str] = ()) -> ContractSpec:
    names = sorted(set(tool_names))
    schema = _closed_object(
        {
            "split_task": _boolean(),
            "expand_task": _boolean(),
            "ask_user": _boolean(),
            "assume_missing": _boolean(),
            "generate_ideas": _boolean(),
            "direct_response": _boolean(),
            "execution_mode": {"type": "string", "enum": ["full_plan", "single_tool", "direct_response", "clarification"]},
            "preferred_tool_name": {"type": "string", "enum": ["", *names]},
            "evidence_required_before_response": _boolean(),
            "evidence_call_count": {"type": "integer"},
            "confidence": _number(),
            "reason": _string(),
        }
    )
    return _contract("task_decision", schema)


def task_decision_semantic_review_contract() -> ContractSpec:
    return _contract(
        "task_decision_semantic_review",
        _closed_object(
            {
                "decision_matches_request": _boolean(),
                "decision_is_internally_consistent": _boolean(),
                "required_evidence_sources": _array(_string()),
                "minimum_evidence_call_count": _integer(),
                "selected_mode_and_tool_can_cover_declared_count": _boolean(),
                "feedback": _string(),
            }
        ),
    )


def task_expansion_contract() -> ContractSpec:
    schema = _closed_object(
        {
            "original_goal": _string(),
            "expanded_goal": _string(),
            "scope": _array(_string()),
            "constraints": _array(_string()),
            "expected_outputs": _array(_string()),
            "assumptions": _array(_string()),
        }
    )
    return _contract("task_expansion", schema)


def active_session_control_contract() -> ContractSpec:
    schema = _closed_object(
        {
            "action": {
                "type": "string",
                "enum": [
                    "status",
                    "session_summary",
                    "continue_with_note",
                    "cancel",
                    "stop",
                    "replace_task",
                    "queue_after_current",
                    "clarify_conflict",
                ],
            },
            "reason": _string(),
            "response_text": _string(),
            "added_context": _string(),
            "replacement_goal": _string(),
            "queued_task": _string(),
            "clarification_question": _string(),
        }
    )
    return _contract("active_session_control", schema)


def summary_contract() -> ContractSpec:
    return _contract("summary", _closed_object({"summary": _string()}))


def plan_contract(
    tool_names: Iterable[str],
    *,
    context_limit: int = 2048,
    max_steps: int | None = None,
) -> ContractSpec:
    del context_limit, max_steps
    names = sorted(tool_names)
    verification_types = ["composite"]
    check_types = [
        "dependencies_completed",
        "artifact_present",
        "tool_name_equals",
        "tool_output_nonempty",
        "tool_output_schema_valid",
        "tool_files_changed",
        "file_exists",
        "file_contains",
        "json_schema_valid",
        "function_exists",
        "symbol_exists",
        "command_success",
        "string_nonempty",
        "exact_match",
        "numeric_tolerance",
        "string_match",
        "criterion",
    ]

    def check_properties(check_type_schema: dict[str, Any], *, include_condition: bool) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "name": _string(),
            "check_type": check_type_schema,
            "artifact": _string(),
            "actual_source": _string(),
            "expected": _string(),
            "expected_json": _string(),
            "schema_json": _string(),
            "criterion": _string(),
            "path": _string(),
            "pattern": _string(),
            "function_name": _string(),
            "symbol": _string(),
            "command": _array(_string()),
            "cwd": _string(),
            "tolerance": _number(),
            "regex": _boolean(),
            "mode": _string(),
        }
        if include_condition:
            properties["condition"] = {"type": "string", "enum": ["required", "optional"]}
        return properties

    check_schema = _closed_object(
        check_properties({"type": "string", "enum": check_types}, include_condition=True)
    )
    step_schema = _closed_object(
        {
            "step_id": _string(),
            "title": _string(),
            "goal": _string(),
            "kind": {"type": "string", "enum": ["tool", "read", "write", "reasoning", "note", "respond"]},
            "expected_tool": {"type": "string", "enum": ["", *names]},
            "input_text": _string(),
            "expected_output": _string(),
            "expected_outputs": _array(_string()),
            "success_criteria": _string(),
            "verification_type": {"type": "string", "enum": verification_types},
            "verification_checks": _array(check_schema),
            "input_refs": _array(_string()),
            "output_refs": _array(_string()),
            "fallback_strategy": _string(),
            "depends_on": _array(_string()),
        }
    )
    schema = _closed_object(
        {
            "goal": _string(),
            "success_criteria": _string(),
            "fallback_strategy": _string(),
            "steps": _array(step_schema),
        }
    )
    return _contract("task_plan", schema)

def strategy_selection_contract() -> ContractSpec:
    schema = _closed_object(
        {
            "task_profile": {
                "type": "string",
                "enum": ["coding", "file_edit", "reading", "multi_step", "generic"],
            },
            "strategy_name": {"type": "string", "enum": ["conservative", "exploratory"]},
            "explore_before_commit": _boolean(),
            "tool_chain_depth": _integer(),
            "verification_intensity": _number(),
            "reason": _string(),
        }
    )
    return _contract("strategy_selection", schema)


def failure_classification_contract() -> ContractSpec:
    schema = _closed_object(
        {
            "kind": {
                "type": "string",
                "enum": [
                    "tool_failure",
                    "reasoning_failure",
                    "planning_failure",
                    "missing_information",
                    "verification_failure",
                    "budget_failure",
                    "state_inconsistency",
                    "transient_external_wait",
                    "retry_now",
                    "retry_later_backoff",
                    "deterministic_permanent",
                    "side_effect_unsafe",
                    "needs_replan",
                    "needs_clarification",
                    "blocked_external",
                    "continue_other",
                ],
            },
            "retryable": _boolean(),
            "requires_replan": _boolean(),
            "suggested_strategy_mode": {
                "type": "string",
                "enum": ["conservative", "recovery", "verification_heavy"],
            },
            "wait_seconds": _number(),
            "reason": _string(),
        }
    )
    return _contract("failure_classification", schema)


def action_selection_contract() -> ContractSpec:
    schema = _closed_object(
        {
            "action": {
                "type": "string",
                "enum": ["execute_step", "retry_step", "replan", "stop", "answer_directly"],
            },
            "reason": _string(),
        }
    )
    return _contract("action_selection", schema)


def subagent_selection_contract(candidate_types: Iterable[str]) -> ContractSpec:
    ordered = ["none", *[item for item in dict.fromkeys(str(item).strip() for item in candidate_types if str(item).strip()) if item != "none"]]
    schema = _closed_object(
        {
            "spawn": _boolean(),
            "subagent_type": {
                "type": "string",
                "enum": ordered,
            },
            "reason": _string(),
            "focus": _string(),
        }
    )
    return _contract("subagent_selection", schema)


def relevance_scoring_contract(item_count: int) -> ContractSpec:
    if item_count <= 0:
        raise ValueError("relevance scoring requires at least one candidate")
    schema = _closed_object({f"score_{index}": _number() for index in range(item_count)})
    return _contract("relevance_scoring", schema)


def verification_contract(
    criteria_names: Iterable[str],
    *,
    name: str = "verification",
    candidate_excerpt_ids: Iterable[str] | None = None,
) -> ContractSpec:
    ordered = list(dict.fromkeys(str(item).strip() for item in criteria_names if str(item).strip()))
    excerpt_ids = list(
        dict.fromkeys(str(item).strip() for item in (candidate_excerpt_ids or []) if isinstance(item, str) and item.strip())
    )
    excerpt_id_schema = {"type": "string", "enum": ["", *excerpt_ids]}
    criterion_schema = _closed_object(
        {
            "name": {"type": "string", "enum": ordered},
            "passed": _boolean(),
            "evidence": _string(),
            "candidate_excerpt_id_1": excerpt_id_schema,
            "candidate_excerpt_id_2": excerpt_id_schema,
            "candidate_excerpt_id_3": excerpt_id_schema,
        }
    )
    schema = _closed_object({"criteria": _array(criterion_schema)})
    return _contract(name, schema)


NULLABLE_STRING_SCHEMA = _nullable(_string())
NULLABLE_INTEGER_SCHEMA = _nullable(_integer())
NULLABLE_STRING_ARRAY_SCHEMA = _nullable(_array(_string()))
