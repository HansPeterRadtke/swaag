from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from swaag.config import AgentConfig
from swaag.types import Plan, PlanStep, PlanStepKind, PlanStepStatus, SessionState, VerificationType
from swaag.utils import new_id, utc_now_iso


class PlanValidationError(ValueError):
    pass


_ALLOWED_STEP_KINDS: set[str] = {"tool", "read", "write", "reasoning", "note", "respond"}
_ALLOWED_VERIFICATION_TYPES: set[str] = {"execution", "structural", "value", "composite", "llm_fallback"}
_TOOL_REQUIRED_KINDS: set[str] = {"tool", "read", "write", "note"}
_ALLOWED_TRANSITIONS: dict[PlanStepStatus, set[PlanStepStatus]] = {
    "pending": {"running", "skipped"},
    "running": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
    "skipped": set(),
}


def _default_done_condition(kind: PlanStepKind, expected_tool: str | None) -> str:
    if kind in _TOOL_REQUIRED_KINDS:
        return f"tool_result:{expected_tool or ''}"
    if kind == "respond":
        return "assistant_response_nonempty"
    return "reasoning_result_nonempty"


def _normalize_check_list(raw_checks: object) -> list[dict[str, object]]:
    if not isinstance(raw_checks, list):
        return []
    normalized: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for raw_check in raw_checks:
        if not isinstance(raw_check, dict):
            continue
        name = str(raw_check.get("name", "")).strip()
        check_type = str(raw_check.get("check_type", "")).strip()
        if not name or not check_type or name in seen_names:
            continue
        normalized_check = dict(raw_check)
        normalized_check["name"] = name
        normalized_check["check_type"] = check_type
        normalized.append(normalized_check)
        seen_names.add(name)
    return normalized


def _normalize_condition_list(raw_conditions: object, valid_names: set[str]) -> list[str]:
    if not isinstance(raw_conditions, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_condition in raw_conditions:
        value = str(raw_condition).strip()
        if not value or value in seen or value not in valid_names:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _normalize_ref_list(raw_refs: object) -> list[str]:
    if not isinstance(raw_refs, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_ref in raw_refs:
        value = str(raw_ref).strip()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _normalize_step_kind(kind: str, expected_tool: str | None) -> PlanStepKind:
    if kind not in _ALLOWED_STEP_KINDS:
        raise PlanValidationError(f"Unknown plan step kind: {kind}")
    if kind == "tool":
        if expected_tool == "read_text":
            return "read"
        if expected_tool == "read_file":
            return "read"
        if expected_tool == "edit_text":
            return "write"
        if expected_tool == "write_file":
            return "write"
        if expected_tool == "notes":
            return "note"
        return "tool"
    return kind  # type: ignore[return-value]


def default_verification_contract(
    *,
    kind: PlanStepKind,
    expected_tool: str | None,
    expected_output: str,
    done_condition: str,
    success_criteria: str,
) -> tuple[list[str], VerificationType, list[dict[str, object]], list[str], list[str]]:
    if kind in {"tool", "read", "write", "note"}:
        checks: list[dict[str, object]] = [
            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
            {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
            {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": expected_tool or ""},
            {"name": "output_nonempty", "check_type": "tool_output_nonempty"},
            {"name": "output_schema_valid", "check_type": "tool_output_schema_valid"},
        ]
        return (
            [expected_output],
            "composite",
            checks,
            [str(item["name"]) for item in checks],
            [],
        )
    deterministic_checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {
            "name": "assistant_text_nonempty" if kind == "respond" else "reasoning_text_nonempty",
            "check_type": "string_nonempty",
            "actual_source": "assistant_text",
        },
        {
            "name": "meets_success_criteria",
            "check_type": "criterion",
            "criterion": success_criteria,
        },
        {
            "name": "satisfies_done_condition",
            "check_type": "criterion",
            "criterion": done_condition,
        },
    ]
    return (
        [expected_output],
        "llm_fallback",
        deterministic_checks,
        [str(item["name"]) for item in deterministic_checks],
        [],
    )


def _normalize_step_payload(
    raw_step: dict[str, object],
    *,
    kind: PlanStepKind,
    expected_tool: str | None,
) -> dict[str, object]:
    title = str(raw_step.get("title", "")).strip() or "Untitled step"
    goal = str(raw_step.get("goal", "")).strip() or title
    expected_outputs = _normalize_ref_list(raw_step.get("expected_outputs", []))
    expected_output = str(raw_step.get("expected_output", "")).strip()
    if not expected_output:
        expected_output = expected_outputs[0] if expected_outputs else goal
    if not expected_outputs:
        expected_outputs = [expected_output]
    success_criteria = str(raw_step.get("success_criteria", "")).strip() or expected_output
    done_condition = str(raw_step.get("done_condition", "")).strip() or _default_done_condition(kind, expected_tool)
    input_text = str(raw_step.get("input_text", "")).strip() or goal or title or "Use the available context."
    (
        default_expected_outputs,
        default_verification_type,
        default_checks,
        default_required,
        default_optional,
    ) = default_verification_contract(
        kind=kind,
        expected_tool=expected_tool,
        expected_output=expected_output,
        done_condition=done_condition,
        success_criteria=success_criteria,
    )
    verification_checks = _normalize_check_list(raw_step.get("verification_checks", []))
    if not verification_checks:
        verification_checks = [dict(item) for item in default_checks]
    valid_names = {str(item["name"]).strip() for item in verification_checks if str(item.get("name", "")).strip()}
    required_conditions = _normalize_condition_list(raw_step.get("required_conditions", []), valid_names)
    if not required_conditions:
        required_conditions = [name for name in default_required if name in valid_names] or sorted(valid_names)
    optional_conditions = _normalize_condition_list(raw_step.get("optional_conditions", []), valid_names)
    required_set = set(required_conditions)
    optional_conditions = [name for name in optional_conditions if name not in required_set]
    if not optional_conditions:
        optional_conditions = [name for name in default_optional if name in valid_names and name not in required_set]
    verification_type = str(raw_step.get("verification_type", "")).strip()
    if verification_type not in _ALLOWED_VERIFICATION_TYPES:
        verification_type = default_verification_type
    if kind in _TOOL_REQUIRED_KINDS and verification_type != default_verification_type:
        verification_type = default_verification_type
        verification_checks = [dict(item) for item in default_checks]
        valid_names = {str(item["name"]).strip() for item in verification_checks if str(item.get("name", "")).strip()}
        required_conditions = [name for name in default_required if name in valid_names] or sorted(valid_names)
        required_set = set(required_conditions)
        optional_conditions = [name for name in default_optional if name in valid_names and name not in required_set]
    if verification_type == "llm_fallback":
        criterion_names = [
            str(check.get("name", "")).strip()
            for check in verification_checks
            if str(check.get("check_type", "")).strip() == "criterion" and str(check.get("criterion", "")).strip()
        ]
        if not criterion_names:
            verification_type = default_verification_type
            verification_checks = [dict(item) for item in default_checks]
            valid_names = {str(item["name"]).strip() for item in verification_checks if str(item.get("name", "")).strip()}
            required_conditions = [name for name in default_required if name in valid_names] or sorted(valid_names)
            required_set = set(required_conditions)
            optional_conditions = [name for name in default_optional if name in valid_names and name not in required_set]
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
        "expected_outputs": expected_outputs or list(default_expected_outputs),
        "verification_type": verification_type,
        "verification_checks": verification_checks,
        "required_conditions": required_conditions,
        "optional_conditions": optional_conditions,
        "input_refs": _normalize_ref_list(raw_step.get("input_refs", [])),
        "output_refs": _normalize_ref_list(raw_step.get("output_refs", [])),
        "fallback_strategy": (
            str(raw_step.get("fallback_strategy", "")).strip()
            or "If this step fails, replan from the latest valid state."
        ),
        "depends_on": _normalize_ref_list(raw_step.get("depends_on", [])),
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
    if not step.verification_checks:
        raise PlanValidationError(f"Plan step {step.step_id} must declare verification_checks")
    if not step.required_conditions:
        raise PlanValidationError(f"Plan step {step.step_id} must declare required_conditions")
    check_names = set()
    criterion_names: set[str] = set()
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
        check_names.add(name)
        if check_type == "criterion":
            criterion = str(check.get("criterion", "")).strip()
            if not criterion:
                raise PlanValidationError(f"Plan step {step.step_id} check {name} is missing criterion text")
            criterion_names.add(name)
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
    if step.verification_type == "llm_fallback" and not criterion_names:
        raise PlanValidationError(
            f"Plan step {step.step_id} requires at least one criterion check for llm_fallback verification"
        )
    if step.kind in _TOOL_REQUIRED_KINDS:
        if not step.expected_tool:
            raise PlanValidationError(f"Plan step {step.step_id} requires a tool")
        if step.expected_tool not in available_tools:
            raise PlanValidationError(f"Plan step {step.step_id} references unknown tool {step.expected_tool}")
        if step.verification_type == "llm_fallback":
            raise PlanValidationError(f"Plan step {step.step_id} must use deterministic verification")
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
        if step.verification_type not in {"composite", "llm_fallback"}:
            raise PlanValidationError("Respond steps must use verification_type='composite' or 'llm_fallback'")
    elif step.kind == "reasoning":
        if step.done_condition != "reasoning_result_nonempty":
            raise PlanValidationError("Reasoning steps must use done_condition='reasoning_result_nonempty'")
        if step.verification_type not in {"composite", "llm_fallback"}:
            raise PlanValidationError("Reasoning steps must use verification_type='composite' or 'llm_fallback'")
    elif step.expected_tool not in {None, ""} and step.expected_tool not in available_tools:
        raise PlanValidationError(f"Plan step {step.step_id} references unknown tool {step.expected_tool}")



def plan_from_payload(payload: dict, *, available_tools: Iterable[str], plan_id: str | None = None) -> Plan:
    available_tool_set = set(available_tools)
    goal = str(payload.get("goal", "")).strip()
    if not goal:
        raise PlanValidationError("Plan goal must not be empty")
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
    if steps[-1].kind != "respond":
        raise PlanValidationError("The final plan step must be a respond step")
    plan = Plan(
        plan_id=plan_id or str(payload.get("plan_id", "")).strip() or new_id("plan"),
        goal=goal,
        steps=steps,
        success_criteria=str(payload.get("success_criteria", "")).strip() or "Complete the task correctly and safely.",
        fallback_strategy=str(payload.get("fallback_strategy", "")).strip() or "If a step fails, replan from the latest valid state.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=steps[0].step_id,
    )
    return plan



def create_direct_tool_plan(goal: str, tool_name: str, *, expected_output: str = "Tool result") -> Plan:
    now = utc_now_iso()
    tool_step = PlanStep(
        step_id=new_id("step"),
        title=f"Execute tool {tool_name}",
        goal=f"Execute {tool_name}",
        kind=_normalize_step_kind("tool", tool_name),
        expected_tool=tool_name,
        input_text=goal,
        expected_output=expected_output,
        done_condition=f"tool_result:{tool_name}",
        success_criteria=f"The tool {tool_name} finishes successfully.",
        expected_outputs=[expected_output],
        verification_type="composite",
        verification_checks=[],
        required_conditions=[],
        optional_conditions=[],
        output_refs=[tool_name],
        fallback_strategy="If the tool fails, stop and report the failure.",
        status="pending",
        last_updated=now,
    )
    (
        tool_step.expected_outputs,
        tool_step.verification_type,
        tool_step.verification_checks,
        tool_step.required_conditions,
        tool_step.optional_conditions,
    ) = default_verification_contract(
        kind=tool_step.kind,
        expected_tool=tool_step.expected_tool,
        expected_output=tool_step.expected_output,
        done_condition=tool_step.done_condition,
        success_criteria=tool_step.success_criteria,
    )
    answer_step = PlanStep(
        step_id=new_id("step"),
        title="Answer the user",
        goal="Produce the final response",
        kind="respond",
        expected_tool=None,
        input_text=goal,
        expected_output="Final assistant response",
        done_condition="assistant_response_nonempty",
        success_criteria=goal.strip() or "The user receives the final answer.",
        expected_outputs=["Final assistant response"],
        verification_type="llm_fallback",
        verification_checks=[],
        required_conditions=[],
        optional_conditions=[],
        input_refs=[tool_name],
        fallback_strategy="If the answer cannot be produced, report the failure clearly.",
        depends_on=[tool_step.step_id],
        status="pending",
        last_updated=now,
    )
    (
        answer_step.expected_outputs,
        answer_step.verification_type,
        answer_step.verification_checks,
        answer_step.required_conditions,
        answer_step.optional_conditions,
    ) = default_verification_contract(
        kind=answer_step.kind,
        expected_tool=answer_step.expected_tool,
        expected_output=answer_step.expected_output,
        done_condition=answer_step.done_condition,
        success_criteria=answer_step.success_criteria,
    )
    return Plan(
        plan_id=new_id("plan"),
        goal=goal,
        steps=[tool_step, answer_step],
        success_criteria="Complete the direct tool request correctly.",
        fallback_strategy="Stop after a tool failure and report it.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=tool_step.step_id,
    )


def create_direct_response_plan(goal: str, *, expected_output: str = "Final assistant response") -> Plan:
    now = utc_now_iso()
    answer_step = PlanStep(
        step_id=new_id("step"),
        title="Answer the user directly",
        goal="Produce the final response",
        kind="respond",
        expected_tool=None,
        input_text=goal,
        expected_output=expected_output,
        done_condition="assistant_response_nonempty",
        success_criteria=goal.strip() or "The user receives a complete direct answer.",
        expected_outputs=[expected_output],
        verification_type="llm_fallback",
        verification_checks=[],
        required_conditions=[],
        optional_conditions=[],
        fallback_strategy="If the answer cannot be produced directly, report the failure clearly.",
        status="pending",
        last_updated=now,
    )
    (
        answer_step.expected_outputs,
        answer_step.verification_type,
        answer_step.verification_checks,
        answer_step.required_conditions,
        answer_step.optional_conditions,
    ) = default_verification_contract(
        kind=answer_step.kind,
        expected_tool=answer_step.expected_tool,
        expected_output=answer_step.expected_output,
        done_condition=answer_step.done_condition,
        success_criteria=answer_step.success_criteria,
    )
    return Plan(
        plan_id=new_id("plan"),
        goal=goal,
        steps=[answer_step],
        success_criteria=goal.strip() or "Answer the user directly and correctly.",
        fallback_strategy="If direct answering fails, stop and report the failure.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=answer_step.step_id,
    )


def _shell_command_checks(*, stdout_label: str) -> tuple[list[dict[str, object]], list[str], list[str]]:
    checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
        {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "shell_command"},
        {"name": "command_exit_zero", "check_type": "exact_match", "actual_source": "tool_output.exit_code", "expected": 0},
        {"name": stdout_label, "check_type": "string_nonempty", "actual_source": "tool_output.stdout"},
        {"name": "output_schema_valid", "check_type": "tool_output_schema_valid"},
    ]
    required = [str(item["name"]) for item in checks]
    return checks, required, []


def _edit_text_checks() -> tuple[list[dict[str, object]], list[str], list[str]]:
    checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
        {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
        {"name": "tool_output_nonempty", "check_type": "tool_output_nonempty"},
        {"name": "tool_output_schema_valid", "check_type": "tool_output_schema_valid"},
        {"name": "tool_files_changed", "check_type": "tool_files_changed"},
    ]
    required = [str(item["name"]) for item in checks]
    return checks, required, []


def _run_tests_checks() -> tuple[list[dict[str, object]], list[str], list[str]]:
    checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
        {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "run_tests"},
        {"name": "command_exit_zero", "check_type": "exact_match", "actual_source": "tool_output.exit_code", "expected": 0},
        {"name": "tool_output_schema_valid", "check_type": "tool_output_schema_valid"},
    ]
    required = [str(item["name"]) for item in checks]
    return checks, required, []


def create_shell_recovery_plan(goal: str) -> Plan:
    """Deterministic recovery plan for coding-style tasks when plan JSON fails.

    The runtime still executes through the ordinary agent loop, tool decision,
    history, and verification paths. This only supplies a structurally valid
    inspect -> edit -> verify -> respond skeleton so the agent can continue
    instead of dying before any real tool work begins.
    """

    now = utc_now_iso()
    inspect_checks, inspect_required, inspect_optional = _shell_command_checks(stdout_label="inspection_stdout_nonempty")
    inspect_step = PlanStep(
        step_id=new_id("step"),
        title="Inspect failing area",
        goal="Locate the failing test or symbol and inspect the most relevant implementation before editing.",
        kind="read",
        expected_tool="shell_command",
        input_text=(
            "Use repo-local shell commands to search for the exact failing test name first when one is provided. "
            "Do not broaden the search to generic issue words before you have located that exact test or named symbol. "
            "Once located, inspect only the most relevant nearby source and print concise evidence for the likely fix."
        ),
        expected_output="Inspection evidence",
        done_condition="tool_result:shell_command",
        success_criteria="Relevant failing-test or source evidence is printed.",
        expected_outputs=["Inspection evidence"],
        verification_type="composite",
        verification_checks=inspect_checks,
        required_conditions=inspect_required,
        optional_conditions=inspect_optional,
        output_refs=["inspection"],
        fallback_strategy="If inspection fails, stop and report the blocker.",
        status="pending",
        last_updated=now,
    )
    patch_checks, patch_required, patch_optional = _edit_text_checks()
    patch_step = PlanStep(
        step_id=new_id("step"),
        title="Patch source",
        goal="Apply the smallest code fix in the relevant implementation file.",
        kind="write",
        expected_tool="edit_text",
        input_text=(
            "Edit the relevant implementation file identified during inspection. "
            "Apply one minimal code change in the source file, not in docs or unrelated tests. "
            "Prefer replace_pattern_once or replace_range over rewriting whole files."
        ),
        expected_output="Patched source file",
        done_condition="tool_result:edit_text",
        success_criteria="The minimal source fix is applied to the right file.",
        expected_outputs=["Patched source file"],
        verification_type="composite",
        verification_checks=patch_checks,
        required_conditions=patch_required,
        optional_conditions=patch_optional,
        input_refs=["inspection"],
        output_refs=["patched_source"],
        fallback_strategy="If patching fails, stop and report the exact failure.",
        depends_on=[inspect_step.step_id],
        status="pending",
        last_updated=now,
    )
    verify_checks, verify_required, verify_optional = _run_tests_checks()
    verify_step = PlanStep(
        step_id=new_id("step"),
        title="Verify targeted test",
        goal="Run the narrowest relevant verification command for the patched area.",
        kind="tool",
        expected_tool="run_tests",
        input_text=(
            "Run the narrowest relevant test command for the failing test or touched file. "
            "Prefer the exact failing test when available; otherwise run the narrowest related pytest target."
        ),
        expected_output="Verification result",
        done_condition="tool_result:run_tests",
        success_criteria="The targeted verification command exits successfully.",
        expected_outputs=["Verification result"],
        verification_type="composite",
        verification_checks=verify_checks,
        required_conditions=verify_required,
        optional_conditions=verify_optional,
        input_refs=["inspection", "patched_source"],
        output_refs=["verification"],
        fallback_strategy="If verification fails, stop and report the exact failure.",
        depends_on=[patch_step.step_id],
        status="pending",
        last_updated=now,
    )
    answer_step = PlanStep(
        step_id=new_id("step"),
        title="Report result",
        goal="Summarize the code change and verification result.",
        kind="respond",
        expected_tool=None,
        input_text=goal,
        expected_output="Final assistant response",
        done_condition="assistant_response_nonempty",
        success_criteria=goal.strip() or "The user receives the final answer.",
        expected_outputs=["Final assistant response"],
        verification_type="llm_fallback",
        verification_checks=[],
        required_conditions=[],
        optional_conditions=[],
        input_refs=["verification"],
        fallback_strategy="If the final response cannot be produced, report the failure clearly.",
        depends_on=[verify_step.step_id],
        status="pending",
        last_updated=now,
    )
    (
        answer_step.expected_outputs,
        answer_step.verification_type,
        answer_step.verification_checks,
        answer_step.required_conditions,
        answer_step.optional_conditions,
    ) = default_verification_contract(
        kind=answer_step.kind,
        expected_tool=answer_step.expected_tool,
        expected_output=answer_step.expected_output,
        done_condition=answer_step.done_condition,
        success_criteria=answer_step.success_criteria,
    )
    return Plan(
        plan_id=new_id("plan"),
        goal=goal,
        steps=[inspect_step, patch_step, verify_step, answer_step],
        success_criteria="Inspect the failing area, apply the minimal fix, verify it, and report the outcome.",
        fallback_strategy="If a recovery step fails, stop and report the exact blocker.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=inspect_step.step_id,
    )



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
