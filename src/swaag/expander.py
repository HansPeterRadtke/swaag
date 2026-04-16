from __future__ import annotations

from swaag.types import DecisionOutcome, ExpandedTask, PromptAnalysis


class ExpansionValidationError(ValueError):
    pass


def expand_task(prompt: str, analysis: PromptAnalysis, decision: DecisionOutcome) -> ExpandedTask:
    stripped = prompt.strip()
    lower = stripped.lower()
    if "game" in lower:
        scope = ["single playable loop", "one core mechanic", "small asset footprint"]
        constraints = ["deterministic prototype", "no unnecessary dependencies"]
        expected_outputs = ["game design summary", "implementation plan", "playable prototype"]
        assumptions = ["genre=arcade prototype", "camera=single-view", "assets=placeholder"]
        expanded_goal = stripped + " Build a small arcade prototype with one core mechanic and a playable loop."
    else:
        scope = ["bounded first implementation", "explicit success criteria"]
        constraints = ["stay within current repository", "prefer deterministic steps"]
        expected_outputs = ["working implementation", "tests", "verifiable result"]
        assumptions = [] if not decision.assume_missing else ["prefer minimal safe assumptions over broad scope"]
        expanded_goal = stripped
        if decision.expand_task and analysis.task_type in {"vague", "unstructured", "incomplete"}:
            expanded_goal = stripped + " Produce a concrete, bounded implementation with explicit scope and expected outputs."
    return ExpandedTask(
        original_goal=stripped,
        expanded_goal=expanded_goal,
        scope=scope,
        constraints=constraints,
        expected_outputs=expected_outputs,
        assumptions=assumptions,
    )


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
