from __future__ import annotations

import os
import socket
from typing import Any

from swaag.utils import utc_now_iso

WORKER_PHASES = frozenset({
    "starting", "context_compilation", "queued_inference", "inference", "tool_execution",
    "completion_evaluation", "waiting_for_user", "verification", "completed", "cancelled", "failed",
})


def validate_worker_phase(phase: str) -> str:
    value = str(phase).strip()
    if value not in WORKER_PHASES:
        raise ValueError(f"unknown worker phase: {value}")
    return value


def heartbeat_payload(*, phase: str, detail: str = "", active_kind: str = "", active_id: str = "") -> dict[str, Any]:
    return {
        "phase": validate_worker_phase(phase),
        "detail": str(detail),
        "active_kind": str(active_kind),
        "active_id": str(active_id),
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
