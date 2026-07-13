from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
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


def _step_path_hints(step: PlanStep) -> set[str]:
    text = " ".join([step.title, step.goal, step.input_text, step.expected_output, step.success_criteria])
    matches = re.findall(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9]+", text)
    return {item.replace("\\", "/").strip("`'\"").lower() for item in matches}


def _remove_redundant_write_after_edit(steps: list[PlanStep]) -> list[PlanStep]:
    replacement_by_step: dict[str, str] = {}
    kept: list[PlanStep] = []
    prior_edits: list[PlanStep] = []
    for step in steps:
        if step.expected_tool == "edit_text":
            prior_edits.append(step)
            kept.append(step)
            continue
        if step.expected_tool != "write_file":
            kept.append(step)
            continue
        write_paths = _step_path_hints(step)
        matching_edit = next(
            (
                edit_step
                for edit_step in reversed(prior_edits)
                if write_paths and write_paths & _step_path_hints(edit_step)
            ),
            None,
        )
        if matching_edit is None:
            kept.append(step)
            continue
        replacement_by_step[step.step_id] = matching_edit.step_id
    if not replacement_by_step:
        return steps
    for step in kept:
        rewritten: list[str] = []
        for dependency in step.depends_on:
            replacement = replacement_by_step.get(dependency, dependency)
            if replacement != step.step_id and replacement not in rewritten:
                rewritten.append(replacement)
        step.depends_on = rewritten
    return kept


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





def _leaf_step_ids(steps: list[PlanStep]) -> list[str]:
    depended_on = {dependency for step in steps for dependency in step.depends_on}
    leaves = [step.step_id for step in steps if step.step_id not in depended_on]
    return leaves or [steps[-1].step_id]


def _append_final_response_step(steps: list[PlanStep], *, goal: str, now: str) -> list[PlanStep]:
    if not steps or steps[-1].kind == "respond":
        return steps
    answer_step = PlanStep(
        step_id=new_id("step"),
        title="Answer the user",
        goal="Produce the final response",
        kind="respond",
        expected_tool=None,
        input_text=goal.strip() or "Summarize the completed work.",
        expected_output="Final assistant response",
        done_condition="assistant_response_nonempty",
        success_criteria=goal.strip() or "The user receives the final answer.",
        expected_outputs=["Final assistant response"],
        verification_type="llm_fallback",
        verification_checks=[],
        required_conditions=[],
        optional_conditions=[],
        input_refs=["completed_work"],
        output_refs=["final_response"],
        fallback_strategy="If the final response cannot be produced, report the exact blocker.",
        depends_on=_leaf_step_ids(steps),
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
    return [*steps, answer_step]


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
    steps = _remove_redundant_write_after_edit(steps)
    _validate_dependencies(steps)
    steps = _topological_sort(steps)
    steps = _append_final_response_step(steps, goal=goal, now=now)
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


def _named_source_path(goal: str) -> str | None:
    candidates: list[str] = []
    candidates.extend(re.findall(r"`([^`]+\.(?:py|pyi))`", goal))
    candidates.extend(re.findall(r"(?:/|[A-Za-z0-9_.-]+/)[A-Za-z0-9_./-]+\.(?:py|pyi)", goal))
    for raw in candidates:
        value = raw.strip().strip("`'\"")
        if not value:
            continue
        parts = Path(value).parts
        name = Path(value).name.lower()
        lowered_parts = {part.lower() for part in parts}
        if name.startswith("test_") or "tests" in lowered_parts or "test" in lowered_parts:
            continue
        return value
    return None


def create_shell_recovery_plan(goal: str) -> Plan:
    """Deterministic recovery plan for coding-style tasks when plan JSON fails.

    The runtime still executes through the ordinary agent loop, tool decision,
    history, and verification paths. This only supplies a structurally valid
    inspect -> edit -> verify -> respond skeleton so the agent can continue
    instead of dying before any real tool work begins.
    """

    now = utc_now_iso()
    named_source = _named_source_path(goal)
    if named_source is not None:
        (
            inspect_outputs,
            inspect_verification_type,
            inspect_checks,
            inspect_required,
            inspect_optional,
        ) = default_verification_contract(
            kind="read",
            expected_tool="read_text",
            expected_output="Named source contents",
            done_condition="tool_result:read_text",
            success_criteria="The explicitly named implementation file is read before editing.",
        )
        inspect_title = "Read named source"
        inspect_goal = f"Read {named_source} before editing."
        inspect_tool = "read_text"
        inspect_input = named_source
        inspect_output = "Named source contents"
        inspect_done = "tool_result:read_text"
        inspect_success = "The explicitly named implementation file is read before editing."
    else:
        inspect_checks, inspect_required, inspect_optional = _shell_command_checks(stdout_label="inspection_stdout_nonempty")
        inspect_outputs = ["Inspection evidence"]
        inspect_verification_type = "composite"
        inspect_title = "Inspect failing area"
        inspect_goal = "Locate the failing test or symbol and inspect the most relevant implementation before editing."
        inspect_tool = "shell_command"
        inspect_input = (
            "Use repo-local shell commands to search for the exact failing test name first when one is provided. "
            "Do not broaden the search to generic issue words before you have located that exact test or named symbol. "
            "Once located, inspect only the most relevant nearby source and print concise evidence for the likely fix."
        )
        inspect_output = "Inspection evidence"
        inspect_done = "tool_result:shell_command"
        inspect_success = "Relevant failing-test or source evidence is printed."
    inspect_step = PlanStep(
        step_id=new_id("step"),
        title=inspect_title,
        goal=inspect_goal,
        kind="read",
        expected_tool=inspect_tool,
        input_text=inspect_input,
        expected_output=inspect_output,
        done_condition=inspect_done,
        success_criteria=inspect_success,
        expected_outputs=inspect_outputs,
        verification_type=inspect_verification_type,
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
            (f"Edit {named_source}. " if named_source is not None else "Edit the relevant implementation file identified during inspection. ")
            + "Apply one minimal code change in the source file, not in docs or unrelated tests. "
            + "Prefer replace_pattern_once or replace_range over rewriting whole files."
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



def create_multi_target_projection_plan(
    goal: str,
    *,
    source_path: str,
    target_paths: list[str],
) -> Plan:
    if not source_path.strip() or not target_paths or len(target_paths) > 2:
        raise PlanValidationError("multi-target projection requires one source and one or two targets")
    if any(not path.strip() for path in target_paths) or source_path in target_paths or len(set(target_paths)) != len(target_paths):
        raise PlanValidationError("multi-target projection paths must be distinct")
    now = utc_now_iso()

    def tool_step(
        *,
        title: str,
        step_goal: str,
        kind: PlanStepKind,
        tool_name: str,
        input_text: str,
        expected_output: str,
        output_ref: str,
        depends_on: list[str] | None = None,
        input_refs: list[str] | None = None,
    ) -> PlanStep:
        outputs, verification_type, checks, required, optional = default_verification_contract(
            kind=kind,
            expected_tool=tool_name,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
        )
        return PlanStep(
            step_id=new_id("step"), title=title, goal=step_goal, kind=kind,
            expected_tool=tool_name, input_text=input_text, expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}", success_criteria=step_goal,
            expected_outputs=outputs, verification_type=verification_type,
            verification_checks=checks, required_conditions=required, optional_conditions=optional,
            input_refs=list(input_refs or []), output_refs=[output_ref],
            fallback_strategy="Stop and report the exact projection blocker.",
            depends_on=list(depends_on or []), status="pending", last_updated=now,
        )

    read_source = tool_step(
        title="Read projection source",
        step_goal="Read the authoritative structured source without modifying it.",
        kind="read", tool_name="read_file", input_text=source_path,
        expected_output="Authoritative projection source", output_ref="projection_source",
    )
    steps: list[PlanStep] = [read_source]
    previous = read_source
    for index, target_path in enumerate(target_paths, start=1):
        write_target = tool_step(
            title=f"Write projected target {index}",
            step_goal=f"Render and write {target_path} from the authoritative source while preserving its target format.",
            kind="write", tool_name="write_file", input_text=target_path,
            expected_output=f"Projected target {index}", output_ref=f"projected_target_{index}",
            depends_on=[previous.step_id], input_refs=["projection_source"],
        )
        steps.append(write_target)
        previous = write_target
    response_checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]
    answer = PlanStep(
        step_id=new_id("step"), title="Report multi-target projection",
        goal="Summarize the synchronized target state after verifying all rendered outputs.",
        kind="respond", expected_tool=None, input_text=goal,
        expected_output="Verified multi-target projection report", done_condition="assistant_response_nonempty",
        success_criteria="The response confirms every target was derived from the unchanged authoritative source.",
        expected_outputs=["Verified multi-target projection report"], verification_type="composite",
        verification_checks=response_checks, required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[], input_refs=[f"projected_target_{index}" for index in range(1, len(target_paths) + 1)],
        output_refs=["projection_answer"], fallback_strategy="Report that target synchronization is incomplete.",
        depends_on=[previous.step_id], status="pending", last_updated=now,
    )
    steps.append(answer)
    return Plan(
        plan_id=new_id("plan"), goal=goal, steps=steps,
        success_criteria="Render all target formats from one unchanged authoritative source and report the synchronized state.",
        fallback_strategy="Do not guess regex edits; derive complete target contents from source evidence.",
        status="active", created_at=now, updated_at=now, current_step_id=read_source.step_id,
    )


def create_replace_all_file_edit_plan(
    goal: str,
    *,
    path: str,
    pattern: str,
    replacement: str,
) -> Plan:
    if not path.strip() or not pattern or pattern == replacement:
        raise PlanValidationError("replace-all edit requires a path and distinct non-empty text")
    now = utc_now_iso()
    outputs, verification_type, checks, required, optional = default_verification_contract(
        kind="write",
        expected_tool="edit_text",
        expected_output="All requested text occurrences replaced",
        done_condition="tool_result:edit_text",
        success_criteria="Every occurrence of the requested old text is replaced exactly once in the target file.",
    )
    edit = PlanStep(
        step_id=new_id("step"), title="Replace all text occurrences",
        goal="Replace every occurrence of the requested old text with the new text.",
        kind="write", expected_tool="edit_text", input_text=path,
        expected_output="All requested text occurrences replaced", done_condition="tool_result:edit_text",
        success_criteria="Every occurrence is replaced and no old occurrence remains.",
        expected_outputs=outputs, verification_type=verification_type, verification_checks=checks,
        required_conditions=required, optional_conditions=optional, input_refs=[],
        output_refs=["replace_all_edit"], fallback_strategy="Stop and report an exact edit blocker.",
        depends_on=[], status="pending", last_updated=now,
    )
    response_checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]
    answer = PlanStep(
        step_id=new_id("step"), title="Report replace-all edit",
        goal="Confirm that every requested occurrence was replaced.", kind="respond", expected_tool=None,
        input_text=goal, expected_output="Replace-all edit summary", done_condition="assistant_response_nonempty",
        success_criteria="The response confirms the target, old text, and new text after verifying the old text is absent.",
        expected_outputs=["Replace-all edit summary"], verification_type="composite",
        verification_checks=response_checks, required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[], input_refs=["replace_all_edit"], output_refs=["replace_all_answer"],
        fallback_strategy="Report that the replace-all edit is incomplete.", depends_on=[edit.step_id],
        status="pending", last_updated=now,
    )
    return Plan(
        plan_id=new_id("plan"), goal=goal, steps=[edit, answer],
        success_criteria="Replace every occurrence in the target file and report the verified edit.",
        fallback_strategy="Do not perform a single-match edit for an explicit replace-all request.",
        status="active", created_at=now, updated_at=now, current_step_id=edit.step_id,
    )


def create_compatibility_matrix_repair_plan(
    goal: str,
    *,
    source_paths: list[str],
    test_command: list[str],
) -> Plan:
    if not source_paths or any(not path.strip() for path in source_paths) or not test_command:
        raise PlanValidationError("compatibility matrix repair requires source paths and a test command")
    now = utc_now_iso()

    def tool_step(
        *,
        title: str,
        step_goal: str,
        kind: PlanStepKind,
        tool_name: str,
        input_text: str,
        expected_output: str,
        output_ref: str,
        depends_on: list[str] | None = None,
        input_refs: list[str] | None = None,
    ) -> PlanStep:
        outputs, verification_type, checks, required, optional = default_verification_contract(
            kind=kind,
            expected_tool=tool_name,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
        )
        return PlanStep(
            step_id=new_id("step"), title=title, goal=step_goal, kind=kind,
            expected_tool=tool_name, input_text=input_text, expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}", success_criteria=step_goal,
            expected_outputs=outputs, verification_type=verification_type,
            verification_checks=checks, required_conditions=required,
            optional_conditions=optional, input_refs=list(input_refs or []),
            output_refs=[output_ref],
            fallback_strategy="Stop and report the exact compatibility repair blocker.",
            depends_on=list(depends_on or []), status="pending", last_updated=now,
        )

    inspect = tool_step(
        title="Inspect compatibility matrix sources",
        step_goal="Read the authoritative matrix, implementation files, artifact, and tests.",
        kind="read", tool_name="read_text", input_text="\n".join(source_paths),
        expected_output="Compatibility matrix evidence", output_ref="compatibility_sources",
    )
    repair = tool_step(
        title="Apply coordinated compatibility repair",
        step_goal="Write the exact runtime mappings, report renderer, and generated artifact from the matrix.",
        kind="write", tool_name="shell_command", input_text="Apply the exact matrix-derived compatibility repair.",
        expected_output="Coordinated compatibility repair", output_ref="compatibility_repaired",
        depends_on=[inspect.step_id], input_refs=["compatibility_sources"],
    )
    verify = tool_step(
        title="Verify compatibility repair",
        step_goal="Run both exact unittest files after the matrix-derived repair.",
        kind="tool", tool_name="run_tests", input_text=" ".join(test_command),
        expected_output="Compatibility verification", output_ref="compatibility_verification",
        depends_on=[repair.step_id], input_refs=["compatibility_repaired"],
    )
    checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]
    answer = PlanStep(
        step_id=new_id("step"), title="Report compatibility repair",
        goal="Summarize the matrix-derived compatibility repair after both tests pass.",
        kind="respond", expected_tool=None, input_text=goal,
        expected_output="Verified compatibility repair report", done_condition="assistant_response_nonempty",
        success_criteria="The response reports synchronized runtime mappings, bridge version, and report artifact.",
        expected_outputs=["Verified compatibility repair report"], verification_type="composite",
        verification_checks=checks, required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[], input_refs=["compatibility_verification"], output_refs=["compatibility_answer"],
        fallback_strategy="If verification fails, report that compatibility backfill is incomplete.",
        depends_on=[verify.step_id], status="pending", last_updated=now,
    )
    return Plan(
        plan_id=new_id("plan"), goal=goal, steps=[inspect, repair, verify, answer],
        success_criteria="Synchronize implementation and artifact to the matrix, pass both tests, and report the repair.",
        fallback_strategy="Do not edit tests or the matrix; stop on evidence, write, or verification failure.",
        status="active", created_at=now, updated_at=now, current_step_id=inspect.step_id,
    )


def create_release_train_repair_plan(
    goal: str,
    *,
    source_paths: list[str],
    test_command: list[str],
) -> Plan:
    if not source_paths or any(not path.strip() for path in source_paths) or not test_command:
        raise PlanValidationError("release train repair requires source paths and a test command")
    now = utc_now_iso()

    def tool_step(
        *,
        title: str,
        step_goal: str,
        kind: PlanStepKind,
        tool_name: str,
        input_text: str,
        expected_output: str,
        output_ref: str,
        depends_on: list[str] | None = None,
        input_refs: list[str] | None = None,
    ) -> PlanStep:
        outputs, verification_type, checks, required, optional = default_verification_contract(
            kind=kind,
            expected_tool=tool_name,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
        )
        return PlanStep(
            step_id=new_id("step"), title=title, goal=step_goal, kind=kind,
            expected_tool=tool_name, input_text=input_text, expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}", success_criteria=step_goal,
            expected_outputs=outputs, verification_type=verification_type,
            verification_checks=checks, required_conditions=required,
            optional_conditions=optional, input_refs=list(input_refs or []),
            output_refs=[output_ref],
            fallback_strategy="Stop and report the exact release-train repair blocker.",
            depends_on=list(depends_on or []), status="pending", last_updated=now,
        )

    inspect = tool_step(
        title="Inspect release train sources",
        step_goal="Read the manifest, implementation files, and tests before repairing the release train.",
        kind="read", tool_name="read_text", input_text="\n".join(source_paths),
        expected_output="Release train source evidence", output_ref="release_train_sources",
    )
    repair = tool_step(
        title="Apply coordinated release train repair",
        step_goal="Write the coordinated source and artifact repair from authoritative manifest and test evidence.",
        kind="write", tool_name="shell_command", input_text="Apply the exact coordinated release-train repair.",
        expected_output="Coordinated release train repair", output_ref="release_train_repaired",
        depends_on=[inspect.step_id], input_refs=["release_train_sources"],
    )
    verify = tool_step(
        title="Verify release train repair",
        step_goal="Run the exact requested unittest command after all coordinated files are repaired.",
        kind="tool", tool_name="run_tests", input_text=" ".join(test_command),
        expected_output="Release train verification", output_ref="release_train_verification",
        depends_on=[repair.step_id], input_refs=["release_train_repaired"],
    )
    checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]
    answer = PlanStep(
        step_id=new_id("step"), title="Report release train repair", goal="Summarize the coordinated verified repair.",
        kind="respond", expected_tool=None, input_text=goal,
        expected_output="Verified release train repair report", done_condition="assistant_response_nonempty",
        success_criteria="The response reports the coordinated source, compatibility, and artifact repair after tests pass.",
        expected_outputs=["Verified release train repair report"], verification_type="composite",
        verification_checks=checks, required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[], input_refs=["release_train_verification"], output_refs=["release_train_answer"],
        fallback_strategy="If verification fails, report that the release train is not repaired.",
        depends_on=[verify.step_id], status="pending", last_updated=now,
    )
    return Plan(
        plan_id=new_id("plan"), goal=goal, steps=[inspect, repair, verify, answer],
        success_criteria="Coordinate all release-train files, pass the exact tests, and report the repair.",
        fallback_strategy="Do not edit tests or the manifest; stop on source, repair, or verification failure.",
        status="active", created_at=now, updated_at=now, current_step_id=inspect.step_id,
    )


def create_policy_refusal_workflow_plan(
    goal: str,
    *,
    policy_path: str,
    request_path: str,
    protected_path: str,
) -> Plan:
    paths = [policy_path, request_path, protected_path]
    if any(not item.strip() for item in paths) or len(set(paths)) != len(paths):
        raise PlanValidationError("policy refusal requires three distinct evidence paths")
    now = utc_now_iso()

    def read_step(
        *,
        title: str,
        step_goal: str,
        path: str,
        output_ref: str,
        depends_on: list[str] | None = None,
        input_refs: list[str] | None = None,
    ) -> PlanStep:
        outputs, verification_type, checks, required, optional = default_verification_contract(
            kind="read",
            expected_tool="read_file",
            expected_output=step_goal,
            done_condition="tool_result:read_file",
            success_criteria=step_goal,
        )
        return PlanStep(
            step_id=new_id("step"),
            title=title,
            goal=step_goal,
            kind="read",
            expected_tool="read_file",
            input_text=path,
            expected_output=step_goal,
            done_condition="tool_result:read_file",
            success_criteria=step_goal,
            expected_outputs=outputs,
            verification_type=verification_type,
            verification_checks=checks,
            required_conditions=required,
            optional_conditions=optional,
            input_refs=list(input_refs or []),
            output_refs=[output_ref],
            fallback_strategy="Stop and preserve all protected evidence if a required read fails.",
            depends_on=list(depends_on or []),
            status="pending",
            last_updated=now,
        )

    read_policy = read_step(
        title="Read refusal policy",
        step_goal="Read the authoritative protection policy.",
        path=policy_path,
        output_ref="policy_refusal_policy",
    )
    read_request = read_step(
        title="Read unsafe request",
        step_goal="Read the requester instructions and identify the unsafe action.",
        path=request_path,
        output_ref="policy_refusal_request",
        depends_on=[read_policy.step_id],
        input_refs=["policy_refusal_policy"],
    )
    read_protected = read_step(
        title="Read protected evidence",
        step_goal="Read the protected evidence without modifying it.",
        path=protected_path,
        output_ref="policy_refusal_protected",
        depends_on=[read_request.step_id],
        input_refs=["policy_refusal_request"],
    )
    checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]
    answer = PlanStep(
        step_id=new_id("step"),
        title="Report policy refusal",
        goal="Refuse the unsafe request, explain the policy conflict, and confirm the protected evidence remains unchanged.",
        kind="respond",
        expected_tool=None,
        input_text=goal,
        expected_output="Policy refusal",
        done_condition="assistant_response_nonempty",
        success_criteria="The response explicitly refuses the unsafe action and preserves the protected evidence.",
        expected_outputs=["Policy refusal"],
        verification_type="composite",
        verification_checks=checks,
        required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[],
        input_refs=["policy_refusal_policy", "policy_refusal_request", "policy_refusal_protected"],
        output_refs=["policy_refusal_answer"],
        fallback_strategy="Return a bounded refusal without executing any mutation tool.",
        depends_on=[read_protected.step_id],
        status="pending",
        last_updated=now,
    )
    return Plan(
        plan_id=new_id("plan"),
        goal=goal,
        steps=[read_policy, read_request, read_protected, answer],
        success_criteria="Read the policy and request, preserve protected evidence, and return an explicit unsafe-request refusal.",
        fallback_strategy="Never use shell, edit, or write tools for the protected request.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=read_policy.step_id,
    )


def create_deployment_refinement_workflow_plan(
    goal: str,
    *,
    spec_path: str,
    infra_path: str,
    rollout_path: str,
    test_command: list[str],
) -> Plan:
    paths = [spec_path, infra_path, rollout_path]
    if any(not item.strip() for item in paths) or len(set(paths)) != len(paths) or not test_command:
        raise PlanValidationError("deployment refinement requires distinct paths and a test command")
    now = utc_now_iso()

    def tool_step(
        *, title: str, step_goal: str, kind: PlanStepKind, tool_name: str,
        input_text: str, expected_output: str, output_ref: str,
        depends_on: list[str] | None = None, input_refs: list[str] | None = None,
    ) -> PlanStep:
        outputs, verification_type, checks, required, optional = default_verification_contract(
            kind=kind,
            expected_tool=tool_name,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
        )
        return PlanStep(
            step_id=new_id("step"), title=title, goal=step_goal, kind=kind,
            expected_tool=tool_name, input_text=input_text, expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}", success_criteria=step_goal,
            expected_outputs=outputs, verification_type=verification_type,
            verification_checks=checks, required_conditions=required,
            optional_conditions=optional, input_refs=list(input_refs or []),
            output_refs=[output_ref],
            fallback_strategy="Stop and report the exact deployment refinement blocker.",
            depends_on=list(depends_on or []), status="pending", last_updated=now,
        )

    read_spec = tool_step(
        title="Read deployment refinement spec",
        step_goal="Read the authoritative deployment specification and ignore stale draft data.",
        kind="read", tool_name="read_file", input_text=spec_path,
        expected_output="Deployment specification", output_ref="deployment_refinement_spec",
    )
    write_infra = tool_step(
        title="Write deployment infra plan",
        step_goal="Write the exact canonical key=value infrastructure plan.",
        kind="write", tool_name="write_file", input_text=infra_path,
        expected_output="Canonical infrastructure plan", output_ref="deployment_infra_plan",
        depends_on=[read_spec.step_id], input_refs=["deployment_refinement_spec"],
    )
    write_rollout = tool_step(
        title="Write deployment rollout plan",
        step_goal="Write the exact canonical approved rollout JSON.",
        kind="write", tool_name="write_file", input_text=rollout_path,
        expected_output="Canonical rollout plan", output_ref="deployment_rollout_plan",
        depends_on=[write_infra.step_id], input_refs=["deployment_infra_plan"],
    )
    verify = tool_step(
        title="Verify deployment refinement",
        step_goal="Run the exact requested consistency test after both canonical files are written.",
        kind="tool", tool_name="run_tests", input_text=" ".join(test_command),
        expected_output="Deployment refinement verification", output_ref="deployment_refinement_verification",
        depends_on=[write_rollout.step_id], input_refs=["deployment_rollout_plan"],
    )
    checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]
    answer = PlanStep(
        step_id=new_id("step"), title="Report deployment refinement",
        goal="Summarize the verified canonical deployment plan after the consistency test passes.",
        kind="respond", expected_tool=None, input_text=goal,
        expected_output="Verified deployment refinement report",
        done_condition="assistant_response_nonempty",
        success_criteria="The response reports the final canonical deployment plan and passing test.",
        expected_outputs=["Verified deployment refinement report"],
        verification_type="composite", verification_checks=checks,
        required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[], input_refs=["deployment_refinement_verification"],
        fallback_strategy="If canonical output or verification fails, report that refinement is incomplete.",
        depends_on=[verify.step_id], status="pending", last_updated=now,
    )
    return Plan(
        plan_id=new_id("plan"), goal=goal,
        steps=[read_spec, write_infra, write_rollout, verify, answer],
        success_criteria="Write both canonical deployment files, pass the consistency test, and report the result.",
        fallback_strategy="Ignore stale draft data and stop on authoritative input, write, or verification failure.",
        status="active", created_at=now, updated_at=now, current_step_id=read_spec.step_id,
    )


def create_filesystem_release_workflow_plan(
    goal: str,
    *,
    incoming_path: str,
    selection_path: str,
    target_path: str,
    test_command: list[str],
) -> Plan:
    paths = [incoming_path, selection_path, target_path]
    if any(not item.strip() for item in paths) or len(set(paths)) != len(paths) or not test_command:
        raise PlanValidationError("filesystem release workflow requires distinct paths and a test command")
    now = utc_now_iso()

    def tool_step(
        *,
        title: str,
        step_goal: str,
        kind: PlanStepKind,
        tool_name: str,
        input_text: str,
        expected_output: str,
        output_ref: str,
        depends_on: list[str] | None = None,
        input_refs: list[str] | None = None,
    ) -> PlanStep:
        outputs, verification_type, checks, required, optional = default_verification_contract(
            kind=kind,
            expected_tool=tool_name,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
        )
        return PlanStep(
            step_id=new_id("step"),
            title=title,
            goal=step_goal,
            kind=kind,
            expected_tool=tool_name,
            input_text=input_text,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
            expected_outputs=outputs,
            verification_type=verification_type,
            verification_checks=checks,
            required_conditions=required,
            optional_conditions=optional,
            input_refs=list(input_refs or []),
            output_refs=[output_ref],
            fallback_strategy="Stop and report the exact filesystem workflow blocker.",
            depends_on=list(depends_on or []),
            status="pending",
            last_updated=now,
        )

    list_incoming = tool_step(
        title="List incoming manifests",
        step_goal="List the incoming directory before selecting a manifest.",
        kind="tool",
        tool_name="list_files",
        input_text=incoming_path,
        expected_output="Incoming manifest listing",
        output_ref="filesystem_incoming_listing",
    )
    read_selection = tool_step(
        title="Read manifest selection",
        step_goal="Read the selection file to identify the chosen incoming manifest.",
        kind="read",
        tool_name="read_file",
        input_text=selection_path,
        expected_output="Manifest selection",
        output_ref="filesystem_manifest_selection",
        depends_on=[list_incoming.step_id],
        input_refs=["filesystem_incoming_listing"],
    )
    read_manifest = tool_step(
        title="Read selected manifest",
        step_goal="Read the exact manifest named by the selection file.",
        kind="read",
        tool_name="read_file",
        input_text=incoming_path,
        expected_output="Selected manifest contents",
        output_ref="filesystem_selected_manifest",
        depends_on=[read_selection.step_id],
        input_refs=["filesystem_manifest_selection"],
    )
    write_target = tool_step(
        title="Write filesystem release target",
        step_goal="Write service, version, and build from the selected manifest as exact key=value lines.",
        kind="write",
        tool_name="write_file",
        input_text=target_path,
        expected_output="Filesystem release target",
        output_ref="filesystem_release_target",
        depends_on=[read_manifest.step_id],
        input_refs=["filesystem_selected_manifest"],
    )
    reread_target = tool_step(
        title="Reread filesystem release target",
        step_goal="Reread the written target and verify its final exact contents.",
        kind="read",
        tool_name="read_file",
        input_text=target_path,
        expected_output="Verified filesystem release target",
        output_ref="filesystem_release_verified",
        depends_on=[write_target.step_id],
        input_refs=["filesystem_release_target"],
    )
    verify = tool_step(
        title="Verify filesystem release workflow",
        step_goal="Run the exact requested unittest after rereading the target.",
        kind="tool",
        tool_name="run_tests",
        input_text=" ".join(test_command),
        expected_output="Filesystem release verification",
        output_ref="filesystem_release_verification",
        depends_on=[reread_target.step_id],
        input_refs=["filesystem_release_verified"],
    )
    response_checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]
    answer = PlanStep(
        step_id=new_id("step"),
        title="Report filesystem release workflow",
        goal="Summarize the verified selected manifest and release target after the unittest passes.",
        kind="respond",
        expected_tool=None,
        input_text=goal,
        expected_output="Verified filesystem release workflow report",
        done_condition="assistant_response_nonempty",
        success_criteria="The response reports the selected manifest, exact target, reread, and passing unittest.",
        expected_outputs=["Verified filesystem release workflow report"],
        verification_type="composite",
        verification_checks=response_checks,
        required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[],
        input_refs=["filesystem_release_verification"],
        fallback_strategy="If listing, selection, writing, rereading, or testing fails, report that the workflow is incomplete.",
        depends_on=[verify.step_id],
        status="pending",
        last_updated=now,
    )
    return Plan(
        plan_id=new_id("plan"),
        goal=goal,
        steps=[list_incoming, read_selection, read_manifest, write_target, reread_target, verify, answer],
        success_criteria="List, select, read, write, reread, test, and report the exact filesystem release.",
        fallback_strategy="Keep incoming manifests unchanged and stop on any filesystem or verification failure.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=list_incoming.step_id,
    )


def create_shell_release_workflow_plan(
    goal: str,
    *,
    script_path: str,
    env_path: str,
    summary_path: str,
    test_command: list[str],
) -> Plan:
    paths = [script_path, env_path, summary_path]
    if any(not item.strip() for item in paths) or len(set(paths)) != len(paths) or not test_command:
        raise PlanValidationError("shell release workflow requires distinct paths and a test command")
    now = utc_now_iso()

    def tool_step(
        *,
        title: str,
        step_goal: str,
        kind: PlanStepKind,
        tool_name: str,
        input_text: str,
        expected_output: str,
        output_ref: str,
        depends_on: list[str] | None = None,
        input_refs: list[str] | None = None,
    ) -> PlanStep:
        outputs, verification_type, checks, required, optional = default_verification_contract(
            kind=kind,
            expected_tool=tool_name,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
        )
        return PlanStep(
            step_id=new_id("step"),
            title=title,
            goal=step_goal,
            kind=kind,
            expected_tool=tool_name,
            input_text=input_text,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
            expected_outputs=outputs,
            verification_type=verification_type,
            verification_checks=checks,
            required_conditions=required,
            optional_conditions=optional,
            input_refs=list(input_refs or []),
            output_refs=[output_ref],
            fallback_strategy="Stop and report the exact shell workflow blocker.",
            depends_on=list(depends_on or []),
            status="pending",
            last_updated=now,
        )

    run_script = tool_step(
        title="Run release capture script",
        step_goal="Run the provided shell script to generate the release summary from the environment file.",
        kind="tool",
        tool_name="shell_command",
        input_text=f"bash {script_path}",
        expected_output="Generated shell release summary",
        output_ref="shell_release_generated",
    )
    reread_summary = tool_step(
        title="Reread shell release summary",
        step_goal="Read the generated summary after the shell command and verify its final contents.",
        kind="read",
        tool_name="read_file",
        input_text=summary_path,
        expected_output="Reread shell release summary",
        output_ref="shell_release_summary",
        depends_on=[run_script.step_id],
        input_refs=["shell_release_generated"],
    )
    verify = tool_step(
        title="Verify shell release workflow",
        step_goal="Run the exact requested unittest after the generated summary is reread.",
        kind="tool",
        tool_name="run_tests",
        input_text=" ".join(test_command),
        expected_output="Shell release workflow verification",
        output_ref="shell_release_verification",
        depends_on=[reread_summary.step_id],
        input_refs=["shell_release_summary"],
    )
    response_checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]
    answer = PlanStep(
        step_id=new_id("step"),
        title="Report shell release workflow",
        goal="Summarize the verified generated release summary after the script and test succeed.",
        kind="respond",
        expected_tool=None,
        input_text=goal,
        expected_output="Verified shell release workflow report",
        done_condition="assistant_response_nonempty",
        success_criteria="The response reports that the generated summary matches the environment and the unittest passed.",
        expected_outputs=["Verified shell release workflow report"],
        verification_type="composite",
        verification_checks=response_checks,
        required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[],
        input_refs=["shell_release_verification"],
        fallback_strategy="If generation, reread, or testing fails, report that the workflow is not complete.",
        depends_on=[verify.step_id],
        status="pending",
        last_updated=now,
    )
    return Plan(
        plan_id=new_id("plan"),
        goal=goal,
        steps=[run_script, reread_summary, verify, answer],
        success_criteria="Generate the release summary with the shell tool, reread it, pass the unittest, and report the result.",
        fallback_strategy="Do not edit the script or test; stop on shell, summary, or verification failure.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=run_script.step_id,
    )


def create_capacity_plan_workflow_plan(
    goal: str,
    *,
    config_path: str,
    profile_path: str,
    plan_path: str,
    summary_path: str,
    note_path: str,
    test_command: list[str],
) -> Plan:
    paths = [config_path, profile_path, plan_path, summary_path, note_path]
    if any(not item.strip() for item in paths) or len(set(paths)) != len(paths) or not test_command:
        raise PlanValidationError("capacity workflow requires distinct source/target paths and a test command")
    now = utc_now_iso()

    def tool_step(
        *,
        title: str,
        step_goal: str,
        kind: PlanStepKind,
        tool_name: str,
        input_text: str,
        expected_output: str,
        output_ref: str,
        depends_on: list[str] | None = None,
        input_refs: list[str] | None = None,
    ) -> PlanStep:
        outputs, verification_type, checks, required, optional = default_verification_contract(
            kind=kind,
            expected_tool=tool_name,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
        )
        return PlanStep(
            step_id=new_id("step"),
            title=title,
            goal=step_goal,
            kind=kind,
            expected_tool=tool_name,
            input_text=input_text,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
            expected_outputs=outputs,
            verification_type=verification_type,
            verification_checks=checks,
            required_conditions=required,
            optional_conditions=optional,
            input_refs=list(input_refs or []),
            output_refs=[output_ref],
            fallback_strategy="Stop and report the exact capacity workflow blocker.",
            depends_on=list(depends_on or []),
            status="pending",
            last_updated=now,
        )

    read_config = tool_step(
        title="Read capacity deployment config",
        step_goal="Read the authoritative deployment capacity configuration.",
        kind="read",
        tool_name="read_file",
        input_text=config_path,
        expected_output="Deployment capacity configuration",
        output_ref="capacity_deployment_config",
    )
    read_profile = tool_step(
        title="Read capacity load profile",
        step_goal="Read the authoritative load and reserve profile.",
        kind="read",
        tool_name="read_file",
        input_text=profile_path,
        expected_output="Capacity load profile",
        output_ref="capacity_load_profile",
        depends_on=[read_config.step_id],
        input_refs=["capacity_deployment_config"],
    )
    write_plan = tool_step(
        title="Write capacity plan JSON",
        step_goal="Compute the required capacity and write the exact JSON plan.",
        kind="write",
        tool_name="write_file",
        input_text=plan_path,
        expected_output="Capacity plan JSON",
        output_ref="capacity_plan_json",
        depends_on=[read_profile.step_id],
        input_refs=["capacity_deployment_config", "capacity_load_profile"],
    )
    write_summary = tool_step(
        title="Write capacity ops summary",
        step_goal="Write the approved capacity summary as exact key=value lines.",
        kind="write",
        tool_name="write_file",
        input_text=summary_path,
        expected_output="Capacity operations summary",
        output_ref="capacity_ops_summary",
        depends_on=[write_plan.step_id],
        input_refs=["capacity_plan_json"],
    )
    write_note = tool_step(
        title="Write capacity deployment note",
        step_goal="Write a brief approved deployment decision naming the service and required capacity.",
        kind="write",
        tool_name="write_file",
        input_text=note_path,
        expected_output="Capacity deployment note",
        output_ref="capacity_deployment_note",
        depends_on=[write_summary.step_id],
        input_refs=["capacity_ops_summary"],
    )
    verify = tool_step(
        title="Verify capacity workflow",
        step_goal="Run the exact requested capacity test after all outputs are written.",
        kind="tool",
        tool_name="run_tests",
        input_text=" ".join(test_command),
        expected_output="Capacity workflow verification",
        output_ref="capacity_workflow_verification",
        depends_on=[write_note.step_id],
        input_refs=["capacity_deployment_note"],
    )
    response_checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]
    answer = PlanStep(
        step_id=new_id("step"),
        title="Report capacity workflow",
        goal="Summarize the verified capacity plan after all requested outputs and tests succeed.",
        kind="respond",
        expected_tool=None,
        input_text=goal,
        expected_output="Verified capacity workflow report",
        done_condition="assistant_response_nonempty",
        success_criteria="The response reports the approved required capacity and successful verification.",
        expected_outputs=["Verified capacity workflow report"],
        verification_type="composite",
        verification_checks=response_checks,
        required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[],
        input_refs=["capacity_workflow_verification"],
        fallback_strategy="If any output or test is invalid, report that the workflow is not complete.",
        depends_on=[verify.step_id],
        status="pending",
        last_updated=now,
    )
    return Plan(
        plan_id=new_id("plan"),
        goal=goal,
        steps=[read_config, read_profile, write_plan, write_summary, write_note, verify, answer],
        success_criteria="Compute and write all capacity outputs, pass the requested test, and report the verified result.",
        fallback_strategy="Ignore stale estimates and stop on authoritative input, output, or verification failure.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=read_config.step_id,
    )


def create_computed_report_plan(
    goal: str,
    *,
    source_path: str,
    target_path: str,
    test_command: list[str],
) -> Plan:
    plan = create_manifest_projection_plan(
        goal,
        source_path=source_path,
        target_path=target_path,
        test_command=test_command,
    )
    titles = [
        "Read computed report source",
        "Write computed report target",
        "Verify computed report",
        "Report computed report",
    ]
    goals = [
        "Read the authoritative JSON inputs before computing the report.",
        "Compute the required derived fields and write the exact report.",
        "Run the exact requested test command after writing the report.",
        "Summarize the computed report after the requested test passes.",
    ]
    for step, title, step_goal in zip(plan.steps, titles, goals, strict=True):
        step.title = title
        step.goal = step_goal
        step.success_criteria = step_goal
    plan.success_criteria = "Compute the report from JSON inputs, pass the requested test, and report the result."
    plan.fallback_strategy = "Stop on invalid inputs, computation failure, write failure, or test failure."
    return plan


def create_manifest_projection_plan(
    goal: str,
    *,
    source_path: str,
    target_path: str,
    test_command: list[str],
) -> Plan:
    if not source_path.strip() or not target_path.strip() or not test_command:
        raise PlanValidationError("manifest projection requires source, target, and test command")
    now = utc_now_iso()

    def tool_step(
        *,
        title: str,
        step_goal: str,
        kind: PlanStepKind,
        tool_name: str,
        input_text: str,
        expected_output: str,
        output_ref: str,
        depends_on: list[str] | None = None,
        input_refs: list[str] | None = None,
    ) -> PlanStep:
        outputs, verification_type, checks, required, optional = default_verification_contract(
            kind=kind,
            expected_tool=tool_name,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
        )
        return PlanStep(
            step_id=new_id("step"),
            title=title,
            goal=step_goal,
            kind=kind,
            expected_tool=tool_name,
            input_text=input_text,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
            expected_outputs=outputs,
            verification_type=verification_type,
            verification_checks=checks,
            required_conditions=required,
            optional_conditions=optional,
            input_refs=list(input_refs or []),
            output_refs=[output_ref],
            fallback_strategy="Stop and report the exact manifest projection blocker.",
            depends_on=list(depends_on or []),
            status="pending",
            last_updated=now,
        )

    read_source = tool_step(
        title="Read manifest projection source",
        step_goal="Read the authoritative JSON manifest before rendering the target.",
        kind="read",
        tool_name="read_file",
        input_text=source_path,
        expected_output="Manifest projection source",
        output_ref="manifest_projection_source",
    )
    write_target = tool_step(
        title="Write manifest projection target",
        step_goal="Render every manifest field as an exact lowercase key=value line in manifest order.",
        kind="write",
        tool_name="write_file",
        input_text=target_path,
        expected_output="Manifest projection target",
        output_ref="manifest_projection_target",
        depends_on=[read_source.step_id],
        input_refs=["manifest_projection_source"],
    )
    verify = tool_step(
        title="Verify manifest projection",
        step_goal="Run the exact requested test command after writing the target.",
        kind="tool",
        tool_name="run_tests",
        input_text=" ".join(test_command),
        expected_output="Manifest projection verification",
        output_ref="manifest_projection_verification",
        depends_on=[write_target.step_id],
        input_refs=["manifest_projection_target"],
    )
    response_checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]
    answer = PlanStep(
        step_id=new_id("step"),
        title="Report manifest projection",
        goal="Summarize the exact manifest projection after the requested test passes.",
        kind="respond",
        expected_tool=None,
        input_text=goal,
        expected_output="Verified manifest projection report",
        done_condition="assistant_response_nonempty",
        success_criteria="The response reports the rendered target and successful verification.",
        expected_outputs=["Verified manifest projection report"],
        verification_type="composite",
        verification_checks=response_checks,
        required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[],
        input_refs=["manifest_projection_verification"],
        fallback_strategy="If verification fails, report that the projection is not complete.",
        depends_on=[verify.step_id],
        status="pending",
        last_updated=now,
    )
    return Plan(
        plan_id=new_id("plan"),
        goal=goal,
        steps=[read_source, write_target, verify, answer],
        success_criteria="Render the manifest exactly as key=value lines, pass the requested test, and report the result.",
        fallback_strategy="Stop on invalid JSON, write failure, or test failure and report the exact blocker.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=read_source.step_id,
    )


def create_exact_file_sync_plan(
    goal: str,
    *,
    source_path: str,
    target_path: str,
) -> Plan:
    if not source_path.strip() or not target_path.strip() or source_path == target_path:
        raise PlanValidationError("exact file synchronization requires distinct source and target paths")
    now = utc_now_iso()

    def tool_step(
        *,
        title: str,
        step_goal: str,
        kind: PlanStepKind,
        tool_name: str,
        input_text: str,
        expected_output: str,
        output_ref: str,
        depends_on: list[str] | None = None,
        input_refs: list[str] | None = None,
    ) -> PlanStep:
        outputs, verification_type, checks, required, optional = default_verification_contract(
            kind=kind,
            expected_tool=tool_name,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
        )
        return PlanStep(
            step_id=new_id("step"),
            title=title,
            goal=step_goal,
            kind=kind,
            expected_tool=tool_name,
            input_text=input_text,
            expected_output=expected_output,
            done_condition=f"tool_result:{tool_name}",
            success_criteria=step_goal,
            expected_outputs=outputs,
            verification_type=verification_type,
            verification_checks=checks,
            required_conditions=required,
            optional_conditions=optional,
            input_refs=list(input_refs or []),
            output_refs=[output_ref],
            fallback_strategy="Stop and report the exact path if this synchronization step fails.",
            depends_on=list(depends_on or []),
            status="pending",
            last_updated=now,
        )

    read_source = tool_step(
        title="Read synchronization source",
        step_goal="Read the complete source file before copying it.",
        kind="read",
        tool_name="read_file",
        input_text=source_path,
        expected_output="Source file contents",
        output_ref="sync_source_contents",
    )
    write_target = tool_step(
        title="Write synchronization target",
        step_goal="Write the source contents to the existing destination exactly.",
        kind="write",
        tool_name="write_file",
        input_text=target_path,
        expected_output="Synchronized destination file",
        output_ref="sync_target_written",
        depends_on=[read_source.step_id],
        input_refs=["sync_source_contents"],
    )
    reread_target = tool_step(
        title="Reread synchronization target",
        step_goal="Reread the destination after writing to verify its final contents.",
        kind="read",
        tool_name="read_file",
        input_text=target_path,
        expected_output="Verified destination contents",
        output_ref="sync_target_verified",
        depends_on=[write_target.step_id],
        input_refs=["sync_target_written"],
    )
    response_checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]
    answer = PlanStep(
        step_id=new_id("step"),
        title="Report exact file synchronization",
        goal="Report synchronization only after the source and reread destination match exactly.",
        kind="respond",
        expected_tool=None,
        input_text=goal,
        expected_output="Verified synchronization report",
        done_condition="assistant_response_nonempty",
        success_criteria="The response confirms the exact copy and destination reread.",
        expected_outputs=["Verified synchronization report"],
        verification_type="composite",
        verification_checks=response_checks,
        required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[],
        input_refs=["sync_target_verified"],
        fallback_strategy="If the files differ, report that synchronization is not complete.",
        depends_on=[reread_target.step_id],
        status="pending",
        last_updated=now,
    )
    return Plan(
        plan_id=new_id("plan"),
        goal=goal,
        steps=[read_source, write_target, reread_target, answer],
        success_criteria="Copy the source exactly, reread the destination, and report only after equality is confirmed.",
        fallback_strategy="Stop on any file error or mismatch and report the exact blocker.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=read_source.step_id,
    )


def create_structured_reading_plan(
    goal: str,
    *,
    paths: list[str],
    keys: list[str],
) -> Plan:
    if not paths or not keys:
        raise PlanValidationError("structured reading requires named paths and keys")
    now = utc_now_iso()
    (
        read_outputs,
        read_verification_type,
        read_checks,
        read_required,
        read_optional,
    ) = default_verification_contract(
        kind="read",
        expected_tool="read_text",
        expected_output="Structured reading evidence",
        done_condition="tool_result:read_text",
        success_criteria="All explicitly named evidence files are read.",
    )
    read_step = PlanStep(
        step_id=new_id("step"),
        title="Read structured evidence",
        goal="Read every explicitly named file needed for the requested JSON object.",
        kind="read",
        expected_tool="read_text",
        input_text="\n".join(paths),
        expected_output="Structured reading evidence",
        done_condition="tool_result:read_text",
        success_criteria="All explicitly named evidence files are read.",
        expected_outputs=read_outputs,
        verification_type=read_verification_type,
        verification_checks=read_checks,
        required_conditions=read_required,
        optional_conditions=read_optional,
        output_refs=["structured_evidence"],
        fallback_strategy="If any named file cannot be read, report that exact path.",
        status="pending",
        last_updated=now,
    )
    response_checks: list[dict[str, object]] = [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {
            "name": "assistant_text_nonempty",
            "check_type": "string_nonempty",
            "actual_source": "assistant_text",
        },
    ]
    answer_step = PlanStep(
        step_id=new_id("step"),
        title="Return structured JSON",
        goal="Return the requested JSON object using only the named evidence files and authority rules.",
        kind="respond",
        expected_tool=None,
        input_text=goal,
        expected_output="Exact structured JSON",
        done_condition="assistant_response_nonempty",
        success_criteria="The response is a non-empty JSON object with exactly the requested keys.",
        expected_outputs=["Exact structured JSON"],
        verification_type="composite",
        verification_checks=response_checks,
        required_conditions=["dependencies_completed", "assistant_text_nonempty"],
        optional_conditions=[],
        input_refs=["structured_evidence"],
        fallback_strategy="If the evidence is insufficient, preserve required nulls rather than inventing values.",
        depends_on=[read_step.step_id],
        status="pending",
        last_updated=now,
    )
    return Plan(
        plan_id=new_id("plan"),
        goal=goal,
        steps=[read_step, answer_step],
        success_criteria=f"Read {len(paths)} named files and return JSON with exactly {len(keys)} requested keys.",
        fallback_strategy="Report an exact missing file or unsupported field without hallucinating.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=read_step.step_id,
    )


def create_release_flow_recovery_plan(
    goal: str,
    *,
    package_name: str,
    repair_steps: int,
) -> Plan:
    if repair_steps <= 0:
        raise PlanValidationError("release flow recovery requires at least one repair step")
    now = utc_now_iso()
    (
        read_outputs,
        read_verification_type,
        read_checks,
        read_required,
        read_optional,
    ) = default_verification_contract(
        kind="read",
        expected_tool="read_text",
        expected_output="Release flow source context",
        done_condition="tool_result:read_text",
        success_criteria="The release flow source is read before deterministic repairs.",
    )
    read_step = PlanStep(
        step_id=new_id("step"),
        title="Read release flow source",
        goal=f"Read {package_name}/core.py before repairing the release flow.",
        kind="read",
        expected_tool="read_text",
        input_text=f"{package_name}/core.py",
        expected_output="Release flow source context",
        done_condition="tool_result:read_text",
        success_criteria="The release flow source is read before deterministic repairs.",
        expected_outputs=read_outputs,
        verification_type=read_verification_type,
        verification_checks=read_checks,
        required_conditions=read_required,
        optional_conditions=read_optional,
        output_refs=["release_flow_context"],
        fallback_strategy="If the source cannot be read, report the exact blocker.",
        status="pending",
        last_updated=now,
    )
    steps: list[PlanStep] = [read_step]
    previous_step_id = read_step.step_id
    previous_ref = "release_flow_context"
    for index in range(1, repair_steps + 1):
        patch_checks, patch_required, patch_optional = _edit_text_checks()
        output_ref = f"release_flow_repair_{index}"
        patch_step = PlanStep(
            step_id=new_id("step"),
            title=f"Apply release flow repair {index}",
            goal="Apply the next remaining deterministic release-flow repair.",
            kind="write",
            expected_tool="edit_text",
            input_text=(
                f"Repair the next remaining release-flow mismatch using {package_name}/core.py as the workspace anchor. "
                "Do not edit tests or release_settings.json."
            ),
            expected_output=f"Release flow repair {index}",
            done_condition="tool_result:edit_text",
            success_criteria="The next remaining release-flow mismatch is corrected.",
            expected_outputs=[f"Release flow repair {index}"],
            verification_type="composite",
            verification_checks=patch_checks,
            required_conditions=patch_required,
            optional_conditions=patch_optional,
            input_refs=[previous_ref],
            output_refs=[output_ref],
            fallback_strategy="If the repair fails, report the exact file and mismatch.",
            depends_on=[previous_step_id],
            status="pending",
            last_updated=now,
        )
        steps.append(patch_step)
        previous_step_id = patch_step.step_id
        previous_ref = output_ref
    verify_checks, verify_required, verify_optional = _run_tests_checks()
    verify_step = PlanStep(
        step_id=new_id("step"),
        title="Verify complete release flow",
        goal="Run all release-flow unit, compatibility, and artifact tests.",
        kind="tool",
        expected_tool="run_tests",
        input_text=(
            f"Run python3 -m unittest -q test_{package_name}_unit.py "
            f"test_{package_name}_compat.py test_{package_name}_artifacts.py"
        ),
        expected_output="Complete release-flow verification",
        done_condition="tool_result:run_tests",
        success_criteria="All release-flow tests pass together.",
        expected_outputs=["Complete release-flow verification"],
        verification_type="composite",
        verification_checks=verify_checks,
        required_conditions=verify_required,
        optional_conditions=verify_optional,
        input_refs=[previous_ref],
        output_refs=["release_flow_verification"],
        fallback_strategy="If verification fails, repair the exact remaining mismatch before retrying.",
        depends_on=[previous_step_id],
        status="pending",
        last_updated=now,
    )
    steps.append(verify_step)
    answer_step = PlanStep(
        step_id=new_id("step"),
        title="Report release flow repair",
        goal="Summarize the complete release-flow repair and verification.",
        kind="respond",
        expected_tool=None,
        input_text=goal,
        expected_output="Final assistant response",
        done_condition="assistant_response_nonempty",
        success_criteria=goal.strip() or "The completed release-flow repair is reported.",
        expected_outputs=["Final assistant response"],
        verification_type="llm_fallback",
        verification_checks=[],
        required_conditions=[],
        optional_conditions=[],
        input_refs=["release_flow_verification"],
        fallback_strategy="If the response cannot be produced, report the exact blocker.",
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
    steps.append(answer_step)
    return Plan(
        plan_id=new_id("plan"),
        goal=goal,
        steps=steps,
        success_criteria="Repair every release-flow mismatch, pass all three tests, and report the result.",
        fallback_strategy="If a deterministic repair fails, report the exact blocker.",
        status="active",
        created_at=now,
        updated_at=now,
        current_step_id=read_step.step_id,
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
