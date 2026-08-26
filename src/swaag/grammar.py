from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from swaag.schema_portability import assert_portable_json_schema
from swaag.types import ContractSpec


def _string() -> dict[str, Any]:
    return {"type": "string"}


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


def _contract(name: str, schema: dict[str, Any]) -> ContractSpec:
    assert_portable_json_schema(schema, schema_name=name)
    return ContractSpec(name=name, mode="json_schema", json_schema=schema)


def yes_no_contract() -> ContractSpec:
    return _contract(
        "yes_no",
        _closed_object({"answer": {"type": "string", "enum": ["yes", "no"]}}),
    )


def summary_contract() -> ContractSpec:
    return _contract(
        "summary",
        _closed_object({"summary": _string(), "preserve_recent_messages": {"type": "integer"}}),
    )


def tool_result_projection_contract() -> ContractSpec:
    return _contract(
        "tool_result_projection",
        _closed_object({"projection": _string()}),
    )


def completion_evaluation_contract() -> ContractSpec:
    return _contract(
        "completion_evaluation",
        _closed_object({"complete": _boolean(), "reason": _string(), "remaining_work": _array(_string())}),
    )


def agent_action_contract(tool_specs: Iterable[tuple], *, allow_silent_completion: bool = False) -> ContractSpec:
    tool_call_variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sorted(tool_specs, key=lambda value: str(value[0])):
        name = str(item[0])
        if name in seen:
            continue
        seen.add(name)
        input_schema = item[2]
        if not isinstance(input_schema, dict):
            raise TypeError(f"Tool {name!r} has no JSON object input schema")
        tool_call_variants.append(
            _closed_object(
                {
                    "tool_name": {"type": "string", "enum": [name]},
                    "arguments": deepcopy(input_schema),
                }
            )
        )
    if tool_call_variants:
        tool_call_schema: dict[str, Any] = {"anyOf": tool_call_variants}
    else:
        tool_call_schema = _closed_object(
            {
                "tool_name": {"type": "string", "enum": []},
                "arguments": _closed_object({}),
            }
        )
    return _contract(
        "agent_action",
        _closed_object(
            {
                "assistant_message": _string(),
                "tool_calls": _array(tool_call_schema),
                "continue_loop": _boolean(),
                "silent_completion": _boolean() if allow_silent_completion else {"type": "boolean", "enum": [False]},
                "status": _closed_object(
                    {
                        "situation": _string(),
                        "action": _string(),
                        "reason": _string(),
                        "importance": {"type": "string", "enum": ["minor", "normal", "major", "critical"]},
                    }
                ),
                "questions": _array(
                    _closed_object(
                        {
                            "question": _string(),
                            "criticality": {"type": "string", "enum": ["optional", "blocking"]},
                            "reason": _string(),
                            "assumption_if_unanswered": _string(),
                        }
                    )
                ),
            }
        ),
    )


def history_analysis_contract() -> ContractSpec:
    return _contract(
        "history_analysis",
        _closed_object(
            {
                "goal_constraints": _array(_string()),
                "failure_evidence": _array(_string()),
                "candidate_root_causes": _array(_string()),
                "source_sequences": _array({"type": "integer"}),
                "wrong_strategy": _string(),
                "recommended_strategy": _string(),
                "uncertainties": _array(_string()),
            }
        ),
    )
