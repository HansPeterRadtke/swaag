from __future__ import annotations

from swaag.types import ExpandedTask


class ExpansionValidationError(ValueError):
    pass


def validate_expanded_task(expanded_task: ExpandedTask) -> None:
    if not expanded_task.original_goal.strip():
        raise ExpansionValidationError("ExpandedTask.original_goal must not be empty")
    if not expanded_task.expanded_goal.strip():
        raise ExpansionValidationError("ExpandedTask.expanded_goal must not be empty")
    if not expanded_task.scope:
        raise ExpansionValidationError("ExpandedTask.scope must not be empty")
    if not expanded_task.expected_outputs:
        raise ExpansionValidationError("ExpandedTask.expected_outputs must not be empty")


def expanded_task_from_payload(payload: dict, *, original_goal: str) -> ExpandedTask:
    expanded = ExpandedTask(
        original_goal=str(payload.get("original_goal", "")).strip() or original_goal.strip(),
        expanded_goal=str(payload.get("expanded_goal", "")).strip(),
        scope=[str(item).strip() for item in payload.get("scope", []) if str(item).strip()],
        constraints=[str(item).strip() for item in payload.get("constraints", []) if str(item).strip()],
        expected_outputs=[str(item).strip() for item in payload.get("expected_outputs", []) if str(item).strip()],
        assumptions=[str(item).strip() for item in payload.get("assumptions", []) if str(item).strip()],
    )
    validate_expanded_task(expanded)
    return expanded
