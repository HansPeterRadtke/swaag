"""Durably append verified worker-state changes at foreground reply delivery.

The model's completed answer and recorded tool effects remain authoritative.
This supplement states only current SWAAG worker lifecycle facts obtained from
successful orchestration receipts from the same turn. It does not infer the
user's meaning from text or treat a worker result as proven external effects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

_TERMINAL_LABELS = {
    "completed": "completed",
    "failed": "failed",
    "canceled": "canceled",
    "input_required": "waiting for input",
}


def current_started_worker_status(
    recent_events: Iterable[Any], get_worker: Callable[[str], Any],
) -> list[dict[str, str]]:
    """Return terminal statuses for workers started in this exact turn.

    Input history must be newest-first and end at `turn_started`. Action drafts,
    errors, assistant prose and previous turns never authorize a status claim.
    """
    started: set[str] = set()
    bounded = False
    for event in recent_events:
        if event.event_type == "turn_started":
            bounded = True
            break
        if event.event_type != "tool_result":
            continue
        payload = event.payload
        if payload.get("tool_name") != "orchestration_control":
            continue
        result = payload.get("output")
        if not isinstance(result, dict):
            continue
        ids = result.get("started_worker_ids")
        if not isinstance(ids, list):
            continue
        started.update(item for item in ids
                       if isinstance(item, str) and item.startswith("worker_"))
    if not bounded:
        return []
    observations: list[dict[str, str]] = []
    for worker_id in sorted(started):
        try:
            record = get_worker(worker_id)
        except (KeyError, ValueError, FileNotFoundError):
            continue
        if record.worker_id != worker_id or record.status not in _TERMINAL_LABELS:
            continue
        observations.append({"worker_id": worker_id, "status": record.status})
    return observations


def status_suffix(observations: list[dict[str, str]]) -> str:
    """Explain exact durable lifecycle, not independent task-effect success."""
    if not observations:
        return ""
    if len(observations) == 1:
        status = observations[0]["status"]
        tense = {
            "completed": "has completed",
            "failed": "has failed",
            "canceled": "was canceled",
            "input_required": "is waiting for input",
        }[status]
        return ("\n\nUpdated status from SWAAG's worker records: "
                f"the delegated worker {tense}.")
    summary = ", ".join(
        f"{sum(row['status'] == status for row in observations)} {label}"
        for status, label in _TERMINAL_LABELS.items()
        if any(row["status"] == status for row in observations)
    )
    return f"\n\nUpdated status from SWAAG's worker records: {summary}."
