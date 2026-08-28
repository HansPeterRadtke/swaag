from __future__ import annotations

import os
import socket
from typing import Any

from swaag.utils import utc_now_iso

WORKER_PHASES = frozenset({
    "starting", "context_compilation", "queued_inference", "inference", "tool_execution",
    "completion_evaluation", "structured_output", "response_presentation", "waiting_for_user", "verification", "completed", "cancelled", "failed",
    "semantic_status",
})

WORKER_SUBSTATES = {
    "starting": frozenset({"initializing", "resuming", "processing_controls"}),
    "context_compilation": frozenset({
        "collecting_inputs",
        "resolving_instructions",
        "serializing_prompt",
        "measuring_context",
        "context_fit",
        "context_overflow",
    }),
    "queued_inference": frozenset({"awaiting_capacity", "retrying"}),
    "inference": frozenset({"dispatching", "awaiting_result", "streaming", "retrying"}),
    "tool_execution": frozenset({"preparing", "running", "verifying"}),
    "completion_evaluation": frozenset({
        "collecting_evidence",
        "requesting_evidence",
        "reducing_evidence",
        "evaluating",
    }),
    "structured_output": frozenset({"preparing", "generating", "validating", "repairing"}),
    "response_presentation": frozenset({"selecting", "rendering", "evaluating", "repairing"}),
    "semantic_status": frozenset({"collecting_evidence", "evaluating", "repairing"}),
    "waiting_for_user": frozenset({"blocked"}),
    "verification": frozenset({"validating_model_output", "validating_tool_effect"}),
    "completed": frozenset({"terminal"}),
    "cancelled": frozenset({"terminal"}),
    "failed": frozenset({"terminal"}),
}

DEFAULT_WORKER_SUBSTATES = {
    "starting": "initializing",
    "context_compilation": "collecting_inputs",
    "queued_inference": "awaiting_capacity",
    "inference": "awaiting_result",
    "tool_execution": "running",
    "completion_evaluation": "evaluating",
    "structured_output": "generating",
    "response_presentation": "rendering",
    "semantic_status": "evaluating",
    "waiting_for_user": "blocked",
    "verification": "validating_model_output",
    "completed": "terminal",
    "cancelled": "terminal",
    "failed": "terminal",
}


def validate_worker_phase(phase: str) -> str:
    value = str(phase).strip()
    if value not in WORKER_PHASES:
        raise ValueError(f"unknown worker phase: {value}")
    return value


def validate_worker_substate(phase: str, substate: str = "") -> str:
    validated_phase = validate_worker_phase(phase)
    value = str(substate).strip() or DEFAULT_WORKER_SUBSTATES[validated_phase]
    if value not in WORKER_SUBSTATES[validated_phase]:
        raise ValueError(
            f"unknown worker substate for {validated_phase}: {value}"
        )
    return value


def heartbeat_payload(
    *,
    phase: str,
    substate: str = "",
    detail: str = "",
    active_kind: str = "",
    active_id: str = "",
    operation_kind: str = "",
) -> dict[str, Any]:
    validated_phase = validate_worker_phase(phase)
    return {
        "phase": validated_phase,
        "substate": validate_worker_substate(validated_phase, substate),
        "detail": str(detail),
        "active_kind": str(active_kind),
        "active_id": str(active_id),
        "operation_kind": str(operation_kind),
        "heartbeat_at": utc_now_iso(),
    }


def systemd_notify(*fields: str) -> bool:
    """Best-effort sd_notify without a hard dependency on python-systemd."""
    address = os.environ.get("NOTIFY_SOCKET", "")
    if not address:
        return False
    if address.startswith("@"):  # Linux abstract namespace
        address = "\0" + address[1:]
    message = "\n".join(str(item) for item in fields if str(item))
    if not message:
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.connect(address)
        sock.sendall(message.encode("utf-8"))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def watchdog_interval_seconds(*, default_seconds: float = 10.0) -> float:
    raw = os.environ.get("WATCHDOG_USEC", "").strip()
    if not raw:
        return max(0.5, float(default_seconds))
    try:
        watchdog_seconds = int(raw) / 1_000_000.0
    except ValueError:
        return max(0.5, float(default_seconds))
    # Ping at half the watchdog interval, bounded away from a busy loop.
    return max(0.5, watchdog_seconds / 2.0)
