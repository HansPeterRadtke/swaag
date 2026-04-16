from __future__ import annotations

from typing import Any, Iterable

from swaag.types import ContractSpec


def _bounded_string(*, max_length: int, min_length: int = 0) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "maxLength": max_length}
    if min_length:
        schema["minLength"] = min_length
    return schema


def _bounded_string_array(
    *,
    max_items: int,
    max_length: int,
    min_items: int = 0,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "array",
        "maxItems": max_items,
        "items": _bounded_string(max_length=max_length, min_length=1),
    }
    if min_items:
        schema["minItems"] = min_items
    return schema


def plain_text_contract() -> ContractSpec:
    return ContractSpec(name="plain_text", mode="plain")


def yes_no_contract() -> ContractSpec:
    return ContractSpec(
        name="yes_no",
        mode="gbnf",
        grammar='root ::= ("yes" | "no")',
    )


def tool_decision_contract(tool_names: Iterable[str]) -> ContractSpec:
    names = sorted(tool_names)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["respond", "call_tool"]},
            "response": _bounded_string(max_length=512),
            "tool_name": {"type": "string", "enum": ["none", *names]},
            "tool_input": {"type": "object", "additionalProperties": True},
        },
        "required": ["action", "response", "tool_name", "tool_input"],
        "additionalProperties": False,
    }
    return ContractSpec(name="tool_decision", mode="json_schema", json_schema=schema)


def tool_input_contract(tool_name: str, input_schema: dict[str, Any]) -> ContractSpec:
    return ContractSpec(name=f"tool_input:{tool_name}", mode="json_schema", json_schema=input_schema)


def prompt_analysis_contract() -> ContractSpec:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task_type": {"type": "string", "enum": ["structured", "unstructured", "vague", "incomplete", "already_decomposed"]},
            "completeness": {"type": "string", "enum": ["complete", "partial", "incomplete"]},
            "requires_expansion": {"type": "boolean"},
            "requires_decomposition": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "detected_entities": _bounded_string_array(max_items=8, max_length=96),
            "detected_goals": _bounded_string_array(max_items=4, max_length=96),
        },
        "required": [
            "task_type",
            "completeness",
            "requires_expansion",
            "requires_decomposition",
            "confidence",
            "detected_entities",
            "detected_goals",
        ],
        "additionalProperties": False,
    }
    return ContractSpec(name="prompt_analysis", mode="json_schema", json_schema=schema)


def task_decision_contract(tool_names: Iterable[str] = ()) -> ContractSpec:
    names = sorted(set(tool_names))
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "split_task": {"type": "boolean"},
            "expand_task": {"type": "boolean"},
            "ask_user": {"type": "boolean"},
            "assume_missing": {"type": "boolean"},
            "generate_ideas": {"type": "boolean"},
            "direct_response": {"type": "boolean"},
            "execution_mode": {"type": "string", "enum": ["full_plan", "single_tool", "direct_response"]},
            "preferred_tool_name": {"type": "string", "enum": ["", *names]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": _bounded_string(max_length=96, min_length=1),
        },
        "required": [
            "split_task",
            "expand_task",
            "ask_user",
            "assume_missing",
            "generate_ideas",
            "direct_response",
            "execution_mode",
            "preferred_tool_name",
            "confidence",
            "reason",
        ],
        "additionalProperties": False,
    }
    return ContractSpec(name="task_decision", mode="json_schema", json_schema=schema)


def task_expansion_contract() -> ContractSpec:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "original_goal": _bounded_string(max_length=320, min_length=1),
            "expanded_goal": _bounded_string(max_length=512, min_length=1),
            "scope": _bounded_string_array(max_items=8, max_length=160),
            "constraints": _bounded_string_array(max_items=8, max_length=160),
            "expected_outputs": _bounded_string_array(max_items=8, max_length=160),
            "assumptions": _bounded_string_array(max_items=8, max_length=160),
        },
        "required": [
            "original_goal",
            "expanded_goal",
            "scope",
            "constraints",
            "expected_outputs",
            "assumptions",
        ],
        "additionalProperties": False,
    }
    return ContractSpec(name="task_expansion", mode="json_schema", json_schema=schema)


def active_session_control_contract() -> ContractSpec:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
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
            "reason": _bounded_string(max_length=160, min_length=1),
            "response_text": _bounded_string(max_length=320),
            "added_context": _bounded_string(max_length=320),
            "replacement_goal": _bounded_string(max_length=320),
            "queued_task": _bounded_string(max_length=320),
            "clarification_question": _bounded_string(max_length=320),
        },
        "required": [
            "action",
            "reason",
            "response_text",
            "added_context",
            "replacement_goal",
            "queued_task",
            "clarification_question",
        ],
        "additionalProperties": False,
    }
    return ContractSpec(name="active_session_control", mode="json_schema", json_schema=schema)


def summary_contract() -> ContractSpec:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "summary": _bounded_string(max_length=1200, min_length=1),
        },
        "required": ["summary"],
        "additionalProperties": False,
    }
    return ContractSpec(name="summary", mode="json_schema", json_schema=schema)


def _scaled_limit(context_limit: int, *, minimum: int, medium: int, maximum: int) -> int:
    if context_limit <= 2048:
        return max(minimum, min(medium, int(round(context_limit / 2048 * medium))))
    scaled = int(round(medium + ((context_limit - 2048) / 6144) * (maximum - medium)))
    return max(minimum, min(maximum, scaled))


def plan_contract(
    tool_names: Iterable[str],
    *,
    context_limit: int = 2048,
    max_steps: int | None = None,
) -> ContractSpec:
    names = sorted(tool_names)
    verification_types = ["execution", "structural", "value", "composite", "llm_fallback"]
    bounded_max_steps = max_steps if max_steps is not None else _scaled_limit(context_limit, minimum=3, medium=4, maximum=6)
    bounded_max_steps = max(2, min(8, int(bounded_max_steps)))
    goal_length = _scaled_limit(context_limit, minimum=160, medium=224, maximum=320)
    success_length = _scaled_limit(context_limit, minimum=120, medium=160, maximum=240)
    fallback_length = _scaled_limit(context_limit, minimum=96, medium=144, maximum=220)
    title_length = _scaled_limit(context_limit, minimum=48, medium=72, maximum=96)
    step_goal_length = _scaled_limit(context_limit, minimum=120, medium=180, maximum=256)
    input_text_length = _scaled_limit(context_limit, minimum=120, medium=180, maximum=256)
    expected_output_length = _scaled_limit(context_limit, minimum=48, medium=72, maximum=120)
    condition_length = _scaled_limit(context_limit, minimum=48, medium=72, maximum=96)
    ref_length = _scaled_limit(context_limit, minimum=32, medium=48, maximum=80)
    verification_items = max(2, min(6, bounded_max_steps))
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "goal": _bounded_string(max_length=goal_length, min_length=1),
            "success_criteria": _bounded_string(max_length=success_length, min_length=1),
            "fallback_strategy": _bounded_string(max_length=fallback_length, min_length=1),
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": bounded_max_steps,
                "items": {
                    "type": "object",
                    "properties": {
                        "step_id": _bounded_string(max_length=48, min_length=1),
                        "title": _bounded_string(max_length=title_length, min_length=1),
                        "goal": _bounded_string(max_length=step_goal_length, min_length=1),
                        "kind": {"type": "string", "enum": ["tool", "read", "write", "reasoning", "note", "respond"]},
                        "expected_tool": {"type": "string", "enum": ["", *names]},
                        "input_text": _bounded_string(max_length=input_text_length, min_length=1),
                        "expected_output": _bounded_string(max_length=expected_output_length, min_length=1),
                        "expected_outputs": _bounded_string_array(max_items=max(2, min(4, bounded_max_steps)), max_length=expected_output_length, min_items=1),
                        "done_condition": _bounded_string(max_length=condition_length, min_length=1),
                        "success_criteria": _bounded_string(max_length=success_length, min_length=1),
                        "verification_type": {"type": "string", "enum": verification_types},
                        "verification_checks": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": verification_items,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": _bounded_string(max_length=ref_length, min_length=1),
                                    "check_type": _bounded_string(max_length=48, min_length=1),
                                    "artifact": _bounded_string(max_length=ref_length),
                                    "actual_source": _bounded_string(max_length=ref_length),
                                    "expected": {},
                                    "criterion": _bounded_string(max_length=success_length),
                                    "path": _bounded_string(max_length=input_text_length),
                                    "pattern": _bounded_string(max_length=expected_output_length),
                                    "command": {
                                        "type": "array",
                                        "maxItems": max(4, bounded_max_steps + 1),
                                        "items": _bounded_string(max_length=96, min_length=1),
                                    },
                                    "cwd": _bounded_string(max_length=input_text_length),
                                    "tolerance": {"type": "number"},
                                    "mode": _bounded_string(max_length=32),
                                },
                                "required": ["name", "check_type"],
                                "additionalProperties": True,
                            },
                        },
                        "required_conditions": _bounded_string_array(max_items=verification_items, max_length=ref_length, min_items=1),
                        "optional_conditions": _bounded_string_array(max_items=verification_items, max_length=ref_length),
                        "input_refs": _bounded_string_array(max_items=max(2, bounded_max_steps), max_length=ref_length),
                        "output_refs": _bounded_string_array(max_items=max(2, bounded_max_steps), max_length=ref_length),
                        "fallback_strategy": _bounded_string(max_length=fallback_length, min_length=1),
                        "depends_on": _bounded_string_array(max_items=max(2, bounded_max_steps), max_length=ref_length),
                    },
                    "required": [
                        "step_id",
                        "title",
                        "goal",
                        "kind",
                        "expected_tool",
                        "input_text",
                        "expected_output",
                        "done_condition",
                        "success_criteria",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["goal", "steps"],
        "additionalProperties": False,
    }
    return ContractSpec(name="task_plan", mode="json_schema", json_schema=schema)


def strategy_selection_contract() -> ContractSpec:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task_profile": {
                "type": "string",
                "enum": ["coding", "file_edit", "reading", "multi_step", "generic"],
            },
            "strategy_name": {"type": "string", "enum": ["conservative", "exploratory"]},
            "explore_before_commit": {"type": "boolean"},
            "tool_chain_depth": {"type": "integer", "minimum": 1, "maximum": 3},
            "verification_intensity": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "reason": _bounded_string(max_length=160, min_length=1),
        },
        "required": [
            "task_profile",
            "strategy_name",
            "explore_before_commit",
            "tool_chain_depth",
            "verification_intensity",
            "reason",
        ],
        "additionalProperties": False,
    }
    return ContractSpec(name="strategy_selection", mode="json_schema", json_schema=schema)


def failure_classification_contract() -> ContractSpec:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
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
            "retryable": {"type": "boolean"},
            "requires_replan": {"type": "boolean"},
            "suggested_strategy_mode": {
                "type": "string",
                "enum": ["conservative", "recovery", "verification_heavy"],
            },
            "wait_seconds": {"type": "number", "minimum": 0},
            "reason": _bounded_string(max_length=200, min_length=1),
        },
        "required": [
            "kind",
            "retryable",
            "requires_replan",
            "suggested_strategy_mode",
            "wait_seconds",
            "reason",
        ],
        "additionalProperties": False,
    }
    return ContractSpec(name="failure_classification", mode="json_schema", json_schema=schema)


def action_selection_contract() -> ContractSpec:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["execute_step", "retry_step", "replan", "stop", "answer_directly"],
            },
            "reason": _bounded_string(max_length=160, min_length=1),
        },
        "required": ["action", "reason"],
        "additionalProperties": False,
    }
    return ContractSpec(name="action_selection", mode="json_schema", json_schema=schema)


def subagent_selection_contract(candidate_types: Iterable[str]) -> ContractSpec:
    ordered = ["none", *[item for item in dict.fromkeys(str(item).strip() for item in candidate_types if str(item).strip()) if item != "none"]]
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "spawn": {"type": "boolean"},
            "subagent_type": {
                "type": "string",
                "enum": ordered,
            },
            "reason": _bounded_string(max_length=160, min_length=1),
            "focus": _bounded_string(max_length=160),
        },
        "required": ["spawn", "subagent_type", "reason", "focus"],
        "additionalProperties": False,
    }
    return ContractSpec(name="subagent_selection", mode="json_schema", json_schema=schema)


def generation_decomposition_contract() -> ContractSpec:
    unit_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "unit_id": _bounded_string(max_length=64, min_length=1),
            "title": _bounded_string(max_length=120, min_length=1),
            "instruction": _bounded_string(max_length=240, min_length=1),
        },
        "required": ["unit_id", "title", "instruction"],
        "additionalProperties": False,
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "output_class": {
                "type": "string",
                "enum": ["bounded_structured", "schema_bounded", "open_ended"],
            },
            "reason": _bounded_string(max_length=160, min_length=1),
            "units": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": unit_schema,
            },
        },
        "required": ["output_class", "reason", "units"],
        "additionalProperties": False,
    }
    return ContractSpec(name="generation_decomposition", mode="json_schema", json_schema=schema)


def overflow_recovery_contract() -> ContractSpec:
    unit_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "unit_id": _bounded_string(max_length=64, min_length=1),
            "title": _bounded_string(max_length=120, min_length=1),
            "instruction": _bounded_string(max_length=240, min_length=1),
        },
        "required": ["unit_id", "title", "instruction"],
        "additionalProperties": False,
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "keep_partial": {"type": "boolean"},
            "reason": _bounded_string(max_length=160, min_length=1),
            "next_units": {
                "type": "array",
                "minItems": 0,
                "maxItems": 8,
                "items": unit_schema,
            },
        },
        "required": ["keep_partial", "reason", "next_units"],
        "additionalProperties": False,
    }
    return ContractSpec(name="overflow_recovery", mode="json_schema", json_schema=schema)


def relevance_scoring_contract(item_count: int) -> ContractSpec:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": item_count,
                "maxItems": item_count,
            },
        },
        "required": ["scores"],
        "additionalProperties": False,
    }
    return ContractSpec(name="relevance_scoring", mode="json_schema", json_schema=schema)


def verification_contract(criteria_names: Iterable[str]) -> ContractSpec:
    ordered = list(dict.fromkeys(str(item).strip() for item in criteria_names if str(item).strip()))
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": ordered},
                        "passed": {"type": "boolean"},
                        "evidence": _bounded_string(max_length=200),
                    },
                    "required": ["name", "passed", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["criteria"],
        "additionalProperties": False,
    }
    return ContractSpec(name="verification", mode="json_schema", json_schema=schema)
