from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from swaag.types import Plan, PlanStep

_ARTIFACT_PLACEHOLDER_RE = re.compile(r"^\{\{\s*(?P<name>[^{}\r\n]+?)\s*\}\}$")


def artifact_labels_from_steps(steps: Iterable[PlanStep]) -> set[str]:
    labels: set[str] = set()
    for step in steps:
        for value in [*step.output_refs, *step.expected_outputs]:
            text = str(value).strip()
            if text:
                labels.add(text)
    return labels


def artifact_labels_from_plan(plan: Plan | None) -> set[str]:
    if plan is None:
        return set()
    return artifact_labels_from_steps(plan.steps)


def parse_json_input_text(input_text: str) -> Any | None:
    text = input_text.strip()
    if not text or text[0] not in "[{\"":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def unresolved_artifact_placeholders(value: Any, artifact_labels: set[str]) -> list[str]:
    if not artifact_labels:
        return []
    found: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            match = _ARTIFACT_PLACEHOLDER_RE.match(item.strip())
            if match and match.group("name").strip() in artifact_labels:
                found.append(match.group(0))
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return found
