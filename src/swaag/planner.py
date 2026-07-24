from __future__ import annotations

from dataclasses import asdict
import json
from typing import Iterable

from swaag.types import Plan, PlanStep, PlanStepKind, PlanStepStatus
from swaag.utils import new_id, utc_now_iso


class PlanValidationError(ValueError):
    pass


_ALLOWED_STEP_KINDS: set[str] = {"tool", "read", "write", "reasoning", "note", "respond"}
_ALLOWED_VERIFICATION_TYPES: set[str] = {"execution", "structural", "value", "composite", "llm_fallback"}
_PLANNED_VERIFICATION_TYPES: set[str] = {"composite"}
_TOOL_REQUIRED_KINDS: set[str] = {"tool", "read", "write", "note"}
_ALLOWED_TRANSITIONS: dict[PlanStepStatus, set[PlanStepStatus]] = {
    "pending": {"running", "skipped"},
    "running": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
    "skipped": set(),
}


def _normalize_check_list(raw_checks: object) -> list[dict[str, object]]:
    if not isinstance(raw_checks, list):
        raise PlanValidationError("verification_checks must be a list")
    normalized: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for index, raw_check in enumerate(raw_checks, start=1):
        if not isinstance(raw_check, dict):
            raise PlanValidationError(f"verification check {index} must be an object")
        name = str(raw_check.get("name", "")).strip()
        check_type = str(raw_check.get("check_type", "")).strip()
        if not name:
            raise PlanValidationError(f"verification check {index} is missing name")
        if not check_type:
            raise PlanValidationError(f"verification check {name} is missing check_type")
        if name in seen_names:
            raise PlanValidationError(f"duplicate verification check name {name}")
        normalized_check = dict(raw_check)
        normalized_check["name"] = name
        normalized_check["check_type"] = check_type
        if "expected" not in normalized_check:
            expected_json = str(normalized_check.get("expected_json", "")).strip()
            if expected_json:
                try:
                    normalized_check["expected"] = json.loads(expected_json)
                except json.JSONDecodeError:
                    normalized_check["expected"] = expected_json
        normalized.append(normalized_check)
        seen_names.add(name)
    return normalized


def _normalize_condition_values(raw_conditions: object) -> list[str]:
    if not isinstance(raw_conditions, list):
        raise PlanValidationError("condition list must be a list")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, raw_condition in enumerate(raw_conditions, start=1):
        value = str(raw_condition).strip()
        if not value:
            raise PlanValidationError(f"condition {index} is empty")
        if value in seen:
            raise PlanValidationError(f"duplicate condition {value}")
        normalized.append(value)
        seen.add(value)
    return normalized


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _validate_file_contains_check_shape(check: dict[str, object], *, step_id: str, check_name: str) -> None:
    expected_json = str(check.get("expected_json", "")).strip()
    if expected_json:
        try:
            parsed = json.loads(expected_json)
        except json.JSONDecodeError as exc:
            raise PlanValidationError(
                f"Plan step {step_id} file_contains check {check_name} expected_json must be JSON text; "
                'for a text target use a JSON string such as "\\"status: ready\\"" or leave expected_json empty when pattern is set'
            ) from exc
        if not isinstance(parsed, str) or not parsed.strip():
            raise PlanValidationError(
                f"Plan step {step_id} file_contains check {check_name} expected_json must decode to a non-empty string"
            )
        return

    pattern = check.get("pattern")
    if isinstance(pattern, str) and pattern.strip():
        return

    expected = check.get("expected")
    if isinstance(expected, str) and expected.strip():
        return
    if expected is not None:
        raise PlanValidationError(
            f"Plan step {step_id} file_contains check {check_name} expected value must be a non-empty string"
        )

    raise PlanValidationError(
        f"Plan step {step_id} file_contains check {check_name} must declare a non-empty pattern or textual expected_json"
    )


def _validate_tool_name_equals_check_shape(
    check: dict[str, object],
    *,
    step_id: str,
    check_name: str,
    expected_tool: str | None,
) -> None:
    expected = check.get("expected")
    if not isinstance(expected, str) or not expected.strip():
        raise PlanValidationError(
            f"Plan step {step_id} tool_name_equals check {check_name} must declare a non-empty expected tool name"
        )
    if expected_tool and expected.strip() != expected_tool:
        raise PlanValidationError(
            f"Plan step {step_id} tool_name_equals check {check_name} expected {expected.strip()!r} "
            f"but the step declares expected_tool={expected_tool!r}"
        )


def _validate_command_success_check_shape(check: dict[str, object], *, step_id: str, check_name: str) -> None:
    command = check.get("command")
    if not isinstance(command, list) or not command:
        raise PlanValidationError(f"Plan step {step_id} command_success check {check_name} must declare a command list")
    if not all(isinstance(part, str) and part.strip() for part in command):
        raise PlanValidationError(
            f"Plan step {step_id} command_success check {check_name} command entries must be non-empty strings"
        )


def _require_nonempty_string(check: dict[str, object], field: str, *, step_id: str, check_name: str) -> str:
    value = check.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError(
            f"Plan step {step_id} check {check_name} must declare a non-empty {field}"
        )
    return value.strip()


def _validate_json_schema_check_shape(check: dict[str, object], *, step_id: str, check_name: str) -> None:
    _require_nonempty_string(check, "actual_source", step_id=step_id, check_name=check_name)
    legacy = check.get("schema")
    if isinstance(legacy, dict):
        return
    schema_json = _require_nonempty_string(check, "schema_json", step_id=step_id, check_name=check_name)
    try:
        parsed = json.loads(schema_json)
    except json.JSONDecodeError as exc:
        raise PlanValidationError(
            f"Plan step {step_id} json_schema_valid check {check_name} schema_json must be valid JSON object text"
        ) from exc
    if not isinstance(parsed, dict):
        raise PlanValidationError(
            f"Plan step {step_id} json_schema_valid check {check_name} schema_json must decode to an object"
        )


def _validate_symbol_check_shape(
    check: dict[str, object],
    *,
    step_id: str,
    check_name: str,
    field: str,
) -> None:
    _require_nonempty_string(check, "path", step_id=step_id, check_name=check_name)
    _require_nonempty_string(check, field, step_id=step_id, check_name=check_name)


def _validate_value_check_shape(
    check: dict[str, object],
    *,
    step_id: str,
    check_name: str,
    numeric: bool = False,
    string_only: bool = False,
) -> None:
    _require_nonempty_string(check, "actual_source", step_id=step_id, check_name=check_name)
    expected = check.get("expected")
    if string_only:
        expected = _require_nonempty_string(check, "expected", step_id=step_id, check_name=check_name)
    elif expected is None or isinstance(expected, (dict, list)) or (isinstance(expected, str) and not expected.strip()):
        raise PlanValidationError(
            f"Plan step {step_id} check {check_name} must declare a non-empty expected"
        )
    if numeric:
        if isinstance(expected, bool):
            raise PlanValidationError(
                f"Plan step {step_id} numeric_tolerance check {check_name} expected must be numeric text"
            )
        try:
            float(expected)
        except (TypeError, ValueError) as exc:
            raise PlanValidationError(
                f"Plan step {step_id} numeric_tolerance check {check_name} expected must be numeric text"
            ) from exc
        tolerance = check.get("tolerance")
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or float(tolerance) < 0:
            raise PlanValidationError(
                f"Plan step {step_id} numeric_tolerance check {check_name} tolerance must be a non-negative number"
            )


def _validate_check_payload_shape(
    check: dict[str, object],
    *,
    step_id: str,
    check_name: str,
    expected_tool: str | None,
) -> None:
    check_type = str(check.get("check_type", "")).strip()
    if check_type == "file_contains":
        _validate_file_contains_check_shape(check, step_id=step_id, check_name=check_name)
    elif check_type == "command_success":
        _validate_command_success_check_shape(check, step_id=step_id, check_name=check_name)
    elif check_type == "tool_name_equals":
        _validate_tool_name_equals_check_shape(
            check,
            step_id=step_id,
            check_name=check_name,
            expected_tool=expected_tool,
        )
    elif check_type == "artifact_present":
        _require_nonempty_string(check, "artifact", step_id=step_id, check_name=check_name)
    elif check_type == "json_schema_valid":
        _validate_json_schema_check_shape(check, step_id=step_id, check_name=check_name)
    elif check_type == "function_exists":
        _validate_symbol_check_shape(check, step_id=step_id, check_name=check_name, field="function_name")
    elif check_type == "symbol_exists":
        _validate_symbol_check_shape(check, step_id=step_id, check_name=check_name, field="symbol")
    elif check_type == "string_nonempty":
        _require_nonempty_string(check, "actual_source", step_id=step_id, check_name=check_name)
    elif check_type == "exact_match":
        _validate_value_check_shape(check, step_id=step_id, check_name=check_name)
    elif check_type == "string_match":
        _validate_value_check_shape(check, step_id=step_id, check_name=check_name, string_only=True)
    elif check_type == "numeric_tolerance":
        _validate_value_check_shape(check, step_id=step_id, check_name=check_name, numeric=True)
    elif check_type == "criterion":
        criterion = check.get("criterion")
        if not isinstance(criterion, str) or not criterion.strip():
            raise PlanValidationError(f"Plan step {step_id} check {check_name} is missing criterion text")


def _ensure_dependencies_check(verification_checks: list[dict[str, object]]) -> None:
    if any(str(check.get("name", "")).strip() == "dependencies_completed" for check in verification_checks):
        return
    verification_checks.insert(0, {"name": "dependencies_completed", "check_type": "dependencies_completed"})


def _conditions_from_local_check_status(
    verification_checks: list[dict[str, object]],
) -> tuple[list[str], list[str]] | None:
    statuses = [check.get("condition") for check in verification_checks]
    if not any(status is not None for status in statuses):
        return None
    if not all(status in {"required", "optional"} for status in statuses):
        raise PlanValidationError(
            "every verification check must declare condition as required or optional when local condition status is used"
        )
    required_conditions: list[str] = []
    optional_conditions: list[str] = []
    for check in verification_checks:
        name = str(check["name"])
        status = str(check.pop("condition"))
        if status == "required":
            required_conditions.append(name)
        else:
            optional_conditions.append(name)
    return required_conditions, optional_conditions


def _normalize_conditions(
    raw_required: object,
    raw_optional: object,
    *,
    verification_checks: list[dict[str, object]],
    input_refs: list[str],
    depends_on: list[str],
) -> tuple[list[str], list[str]]:
    valid_names = {str(item["name"]).strip() for item in verification_checks if str(item.get("name", "")).strip()}
    dependency_markers = set(input_refs) | set(depends_on)

    optional_conditions: list[str] = []
    for value in _normalize_condition_values(raw_optional):
        if value in valid_names:
            _append_unique(optional_conditions, value)
        elif value == "dependencies_completed":
            _ensure_dependencies_check(verification_checks)
            valid_names.add("dependencies_completed")
            _append_unique(optional_conditions, "dependencies_completed")
        elif value in dependency_markers:
            _ensure_dependencies_check(verification_checks)
            valid_names.add("dependencies_completed")
            _append_unique(optional_conditions, "dependencies_completed")
        else:
            raise PlanValidationError(f"condition {value} does not reference a declared verification check")

    required_conditions: list[str] = []
    for value in _normalize_condition_values(raw_required):
        if value in valid_names:
            _append_unique(required_conditions, value)
        elif value == "dependencies_completed":
            _ensure_dependencies_check(verification_checks)
            valid_names.add("dependencies_completed")
            _append_unique(required_conditions, "dependencies_completed")
        elif value in dependency_markers:
            _ensure_dependencies_check(verification_checks)
            valid_names.add("dependencies_completed")
            _append_unique(required_conditions, "dependencies_completed")
        else:
            raise PlanValidationError(f"condition {value} does not reference a declared verification check")

    return required_conditions, optional_conditions


def _normalize_ref_list(raw_refs: object) -> list[str]:
    if not isinstance(raw_refs, list):
        raise PlanValidationError("reference list must be a list")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, raw_ref in enumerate(raw_refs, start=1):
        value = str(raw_ref).strip()
        if not value:
            raise PlanValidationError(f"reference {index} is empty")
        if value in seen:
            raise PlanValidationError(f"duplicate reference {value}")
        normalized.append(value)
        seen.add(value)
    return normalized


def _normalize_step_kind(kind: str, expected_tool: str | None) -> PlanStepKind:
    del expected_tool
    if kind not in _ALLOWED_STEP_KINDS:
        raise PlanValidationError(f"Unknown plan step kind: {kind}")
    return kind  # type: ignore[return-value]


def _normalize_step_payload(
    raw_step: dict[str, object],
    *,
    kind: PlanStepKind,
    expected_tool: str | None,
) -> dict[str, object]:
    step_label = str(raw_step.get("step_id", "")).strip() or str(raw_step.get("title", "")).strip() or "unknown"
    title = str(raw_step.get("title", "")).strip()
    goal = str(raw_step.get("goal", "")).strip()
    input_text = str(raw_step.get("input_text", "")).strip()
    expected_output = str(raw_step.get("expected_output", "")).strip()
    done_condition = str(raw_step.get("done_condition", "")).strip()
    success_criteria = str(raw_step.get("success_criteria", "")).strip()
    fallback_strategy = str(raw_step.get("fallback_strategy", "")).strip()
    missing = [
        field
        for field, value in {
            "title": title,
            "goal": goal,
            "input_text": input_text,
            "expected_output": expected_output,
            "done_condition": done_condition,
            "success_criteria": success_criteria,
            "fallback_strategy": fallback_strategy,
        }.items()
        if not value
    ]
    if missing:
        raise PlanValidationError(f"Plan step {step_label} is missing required model fields: {', '.join(missing)}")
    expected_outputs = _normalize_ref_list(raw_step.get("expected_outputs", []))
    if not expected_outputs:
        raise PlanValidationError(f"Plan step {step_label} must declare expected_outputs")
    verification_checks = _normalize_check_list(raw_step.get("verification_checks", []))
    input_refs = _normalize_ref_list(raw_step.get("input_refs", []))
    output_refs = _normalize_ref_list(raw_step.get("output_refs", []))
    depends_on = _normalize_ref_list(raw_step.get("depends_on", []))
    local_conditions = _conditions_from_local_check_status(verification_checks)
    if local_conditions is None:
        required_conditions, optional_conditions = _normalize_conditions(
            raw_step.get("required_conditions", []),
            raw_step.get("optional_conditions", []),
            verification_checks=verification_checks,
            input_refs=input_refs,
            depends_on=depends_on,
        )
    else:
        required_conditions, optional_conditions = local_conditions
    required_set = set(required_conditions)
    overlap = sorted(name for name in optional_conditions if name in required_set)
    if overlap:
        raise PlanValidationError(f"Plan step {step_label} marks conditions as both required and optional: {', '.join(overlap)}")
    verification_type = str(raw_step.get("verification_type", "")).strip()
    if verification_type not in _ALLOWED_VERIFICATION_TYPES:
        raise PlanValidationError(f"Plan step {step_label} uses invalid verification_type {verification_type!r}")
    if verification_type not in _PLANNED_VERIFICATION_TYPES:
        raise PlanValidationError(f"Plan step {step_label} must use composite verification")
    if kind in _TOOL_REQUIRED_KINDS and verification_type != "composite":
        raise PlanValidationError(f"Plan step {step_label} must use composite verification")
    return {
        "step_id": str(raw_step.get("step_id", "")).strip() or new_id("step"),
        "title": title,
        "goal": goal,
        "kind": kind,
        "expected_tool": expected_tool,
        "input_text": input_text,
        "expected_output": expected_output,
        "done_condition": done_condition,
        "success_criteria": success_criteria,
        "expected_outputs": expected_outputs,
        "verification_type": verification_type,
        "verification_checks": verification_checks,
        "required_conditions": required_conditions,
        "optional_conditions": optional_conditions,
        "input_refs": input_refs,
        "output_refs": output_refs,
        "fallback_strategy": fallback_strategy,
        "depends_on": depends_on,
    }



def _validate_dependencies(steps: list[PlanStep]) -> None:
    step_ids = {step.step_id for step in steps}
    if len(step_ids) != len(steps):
        raise PlanValidationError("Plan contains duplicate step ids")
    for step in steps:
        for dependency in step.depends_on:
            if dependency not in step_ids:
                raise PlanValidationError(f"Plan step {step.step_id} depends on unknown step {dependency}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def _walk(step_id: str) -> None:
        if step_id in visited:
            return
        if step_id in visiting:
            raise PlanValidationError(f"Circular dependency detected at {step_id}")
        visiting.add(step_id)
        step = next(item for item in steps if item.step_id == step_id)
        for dependency in step.depends_on:
            _walk(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step in steps:
        _walk(step.step_id)


def _topological_sort(steps: list[PlanStep]) -> list[PlanStep]:
    by_id = {step.step_id: step for step in steps}
    incoming = {step.step_id: set(step.depends_on) for step in steps}
    original_order = {step.step_id: index for index, step in enumerate(steps)}
    completed: set[str] = set()
    ordered: list[PlanStep] = []
    while len(ordered) < len(steps):
        ready = sorted(
            [step_id for step_id, deps in incoming.items() if step_id not in completed and not deps],
            key=lambda step_id: (original_order[step_id], step_id),
        )
        if not ready:
            raise PlanValidationError("Plan graph could not be topologically sorted")
        for step_id in ready:
            ordered.append(by_id[step_id])
            completed.add(step_id)
            for deps in incoming.values():
                deps.discard(step_id)
    return ordered



def _validate_step(step: PlanStep, available_tools: set[str]) -> None:
    if not step.title.strip():
        raise PlanValidationError(f"Plan step {step.step_id} has an empty title")
    if not step.goal.strip():
        raise PlanValidationError(f"Plan step {step.step_id} has an empty goal")
    if not step.input_text.strip():
        raise PlanValidationError(f"Plan step {step.step_id} has an empty input_text")
    if not step.expected_output.strip():
        raise PlanValidationError(f"Plan step {step.step_id} has an empty expected_output")
    if not step.done_condition.strip():
        raise PlanValidationError(f"Plan step {step.step_id} has an empty done_condition")
    if not step.success_criteria.strip():
        raise PlanValidationError(f"Plan step {step.step_id} has an empty success_criteria")
    if not step.expected_outputs or not all(item.strip() for item in step.expected_outputs):
        raise PlanValidationError(f"Plan step {step.step_id} must declare at least one expected output")
    if step.verification_type not in _ALLOWED_VERIFICATION_TYPES:
        raise PlanValidationError(
            f"Plan step {step.step_id} uses invalid verification_type {step.verification_type!r}"
        )
    if step.verification_type not in _PLANNED_VERIFICATION_TYPES:
        raise PlanValidationError(f"Plan step {step.step_id} must use composite verification")
    if not step.verification_checks:
        raise PlanValidationError(f"Plan step {step.step_id} must declare verification_checks")
    if not step.required_conditions:
        raise PlanValidationError(f"Plan step {step.step_id} must declare required_conditions")
    check_names = set()
    for check in step.verification_checks:
        if not isinstance(check, dict):
            raise PlanValidationError(f"Plan step {step.step_id} has invalid verification check")
        name = str(check.get("name", "")).strip()
        if not name:
            raise PlanValidationError(f"Plan step {step.step_id} has a verification check without a name")
        if name in check_names:
            raise PlanValidationError(f"Plan step {step.step_id} has duplicate verification check name {name}")
        check_type = str(check.get("check_type", "")).strip()
        if not check_type:
            raise PlanValidationError(f"Plan step {step.step_id} check {name} is missing check_type")
        actual_source = str(check.get("actual_source", "")).strip()
        if actual_source.startswith("assistant") and actual_source != "assistant_text":
            raise PlanValidationError(
                f"Plan step {step.step_id} check {name} must use actual_source='assistant_text' for assistant output verification"
            )
        check_names.add(name)
        _validate_check_payload_shape(
            check,
            step_id=step.step_id,
            check_name=name,
            expected_tool=step.expected_tool,
        )
        if check_type == "criterion":
            criterion = str(check.get("criterion", "")).strip()
            if not criterion:
                raise PlanValidationError(f"Plan step {step.step_id} check {name} is missing criterion text")
    unknown_required = set(step.required_conditions) - check_names
    if unknown_required:
        raise PlanValidationError(
            f"Plan step {step.step_id} references unknown required conditions: {', '.join(sorted(unknown_required))}"
        )
    unknown_optional = set(step.optional_conditions) - check_names
    if unknown_optional:
        raise PlanValidationError(
            f"Plan step {step.step_id} references unknown optional conditions: {', '.join(sorted(unknown_optional))}"
        )
    if step.kind in _TOOL_REQUIRED_KINDS:
        if not step.expected_tool:
            raise PlanValidationError(f"Plan step {step.step_id} requires a tool")
        if step.expected_tool not in available_tools:
            raise PlanValidationError(f"Plan step {step.step_id} references unknown tool {step.expected_tool}")
        expected_done_condition = f"tool_result:{step.expected_tool}"
        if step.done_condition != expected_done_condition:
            raise PlanValidationError(
                f"Plan step {step.step_id} must use done_condition={expected_done_condition!r}"
            )
    elif step.kind == "respond":
        if step.expected_tool not in {None, ""}:
            raise PlanValidationError(f"Respond step {step.step_id} must not declare a tool")
        if step.done_condition != "assistant_response_nonempty":
            raise PlanValidationError("Respond steps must use done_condition='assistant_response_nonempty'")
        if step.verification_type != "composite":
            raise PlanValidationError("Respond steps must use verification_type='composite'")
    elif step.kind == "reasoning":
        if step.done_condition != "reasoning_result_nonempty":
            raise PlanValidationError("Reasoning steps must use done_condition='reasoning_result_nonempty'")
        if step.verification_type != "composite":
            raise PlanValidationError("Reasoning steps must use verification_type='composite'")
    elif step.expected_tool not in {None, ""} and step.expected_tool not in available_tools:
        raise PlanValidationError(f"Plan step {step.step_id} references unknown tool {step.expected_tool}")





def plan_from_payload(payload: dict, *, available_tools: Iterable[str], plan_id: str | None = None) -> Plan:
    available_tool_set = set(available_tools)
    goal = str(payload.get("goal", "")).strip()
    if not goal:
        raise PlanValidationError("Plan goal must not be empty")
    success_criteria = str(payload.get("success_criteria", "")).strip()
    if not success_criteria:
        raise PlanValidationError("Plan success_criteria must not be empty")
    fallback_strategy = str(payload.get("fallback_strategy", "")).strip()
    if not fallback_strategy:
        raise PlanValidationError("Plan fallback_strategy must not be empty")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanValidationError("Plan must contain at least one step")
    now = utc_now_iso()
    steps: list[PlanStep] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise PlanValidationError(f"Plan step {index} is not an object")
        expected_tool_raw = str(raw_step.get("expected_tool", "")).strip() or None
        kind = _normalize_step_kind(str(raw_step.get("kind", "")).strip(), expected_tool_raw)
        normalized_step = _normalize_step_payload(raw_step, kind=kind, expected_tool=expected_tool_raw)
        step = PlanStep(
            step_id=str(normalized_step["step_id"]),
            title=str(normalized_step["title"]),
            goal=str(normalized_step["goal"]),
            kind=kind,
            expected_tool=expected_tool_raw,
            input_text=str(normalized_step["input_text"]),
            expected_output=str(normalized_step["expected_output"]),
            done_condition=str(normalized_step["done_condition"]),
            success_criteria=str(normalized_step["success_criteria"]),
            expected_outputs=list(normalized_step["expected_outputs"]),  # type: ignore[arg-type]
            verification_type=normalized_step["verification_type"],  # type: ignore[arg-type]
            verification_checks=list(normalized_step["verification_checks"]),  # type: ignore[arg-type]
            required_conditions=list(normalized_step["required_conditions"]),  # type: ignore[arg-type]
            optional_conditions=list(normalized_step["optional_conditions"]),  # type: ignore[arg-type]
            input_refs=list(normalized_step["input_refs"]),  # type: ignore[arg-type]
            output_refs=list(normalized_step["output_refs"]),  # type: ignore[arg-type]
            fallback_strategy=str(normalized_step["fallback_strategy"]),
            depends_on=list(normalized_step["depends_on"]),  # type: ignore[arg-type]
            status="pending",
            last_updated=now,
        )
        _validate_step(step, available_tool_set)
        steps.append(step)
    _validate_dependencies(steps)
    steps = _topological_sort(steps)
    if not steps or steps[-1].kind != "respond":
        raise PlanValidationError("Model plan must end with a respond step")
    plan = Plan(
        plan_id=plan_id or str(payload.get("plan_id", "")).strip() or new_id("plan"),
        goal=goal,
        steps=steps,
        success_criteria=success_criteria,
        fallback_strategy=fallback_strategy,
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=steps[0].step_id,
    )
    return plan






































def current_step(plan: Plan | None) -> PlanStep | None:
    return next_executable_step(plan)



def ready_steps(plan: Plan | None) -> list[PlanStep]:
    if plan is None:
        return []
    ready: list[PlanStep] = []
    for step in plan.steps:
        if step.status != "pending":
            continue
        if all(_step_status(plan, dependency) == "completed" for dependency in step.depends_on):
            ready.append(step)
    return ready



def next_executable_step(plan: Plan | None) -> PlanStep | None:
    ready = ready_steps(plan)
    return ready[0] if ready else None



def _step_status(plan: Plan, step_id: str) -> PlanStepStatus:
    for step in plan.steps:
        if step.step_id == step_id:
            return step.status
    raise PlanValidationError(f"Unknown step id: {step_id}")



def _get_step(plan: Plan, step_id: str) -> PlanStep:
    for step in plan.steps:
        if step.step_id == step_id:
            return step
    raise PlanValidationError(f"Unknown step id: {step_id}")



def transition_step(plan: Plan, step_id: str, new_status: PlanStepStatus) -> Plan:
    step = _get_step(plan, step_id)
    if new_status not in _ALLOWED_TRANSITIONS[step.status]:
        raise PlanValidationError(f"Invalid transition for {step_id}: {step.status} -> {new_status}")
    if new_status == "running":
        expected = next_executable_step(plan)
        if expected is None or expected.step_id != step_id:
            raise PlanValidationError(f"Cannot start step {step_id} before its dependencies are completed")
    if new_status in {"completed", "failed"} and step.status != "running":
        raise PlanValidationError(f"Step {step_id} must be running before it can become {new_status}")
    step.status = new_status
    step.last_updated = utc_now_iso()
    if new_status == "running":
        plan.current_step_id = step_id
    else:
        next_step = next_executable_step(plan)
        plan.current_step_id = next_step.step_id if next_step is not None else None
    if all(item.status in {"completed", "skipped"} for item in plan.steps):
        plan.status = "completed"
    elif any(item.status == "failed" for item in plan.steps):
        plan.status = "failed"
    else:
        plan.status = "active"
    plan.updated_at = utc_now_iso()
    return plan



def mark_step_in_progress(plan: Plan, step_id: str) -> Plan:
    return transition_step(plan, step_id, "running")



def mark_step_completed(plan: Plan, step_id: str) -> Plan:
    return transition_step(plan, step_id, "completed")



def mark_step_failed(plan: Plan, step_id: str) -> Plan:
    return transition_step(plan, step_id, "failed")



def procedural_memory_from_plan(plan: Plan) -> str:
    titles = " -> ".join(step.title for step in plan.steps)
    return f"Goal pattern: {plan.goal[:120]} | Strategy: {titles} | Fallback: {plan.fallback_strategy[:120]}"



def plan_as_payload(plan: Plan) -> dict:
    return asdict(plan)
