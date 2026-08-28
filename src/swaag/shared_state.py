from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True, frozen=True)
class SharedStateSnapshot:
    protocol: str
    external_context_id: str
    revision: int
    source_id: str
    source_kind: str
    state: Any
    state_sha256: str
    client_supplied: bool
    created_at: str
    source_session_id: str | None = None
    source_call_id: str | None = None
    base_revision: int | None = None
    base_state_sha256: str | None = None
    patch: tuple[dict[str, Any], ...] | None = None
    patch_sha256: str | None = None
    history_event_sequence: int | None = None
    history_event_hash: str | None = None

    def tool_payload(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "external_context_id": self.external_context_id,
            "revision": self.revision,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "state": self.state,
            "state_sha256": self.state_sha256,
            "client_supplied": self.client_supplied,
            "created_at": self.created_at,
            "source_session_id": self.source_session_id,
            "source_call_id": self.source_call_id,
            "base_revision": self.base_revision,
            "base_state_sha256": self.base_state_sha256,
            "patch": None if self.patch is None else list(self.patch),
            "patch_sha256": self.patch_sha256,
            "history_event_sequence": self.history_event_sequence,
            "history_event_hash": self.history_event_hash,
        }


def shared_state_event_payload(snapshot: SharedStateSnapshot) -> dict[str, Any]:
    """Build the canonical history payload for one durable agent state patch."""
    return {
        "source_call_id": snapshot.source_call_id,
        "protocol": snapshot.protocol,
        "external_context_id": snapshot.external_context_id,
        "base_revision": snapshot.base_revision,
        "base_state_sha256": snapshot.base_state_sha256,
        "revision": snapshot.revision,
        "state_sha256": snapshot.state_sha256,
        "delta": list(snapshot.patch or ()),
        "patch_sha256": snapshot.patch_sha256,
    }


class SharedStateConflictError(RuntimeError):
    def __init__(self, message: str, current: SharedStateSnapshot):
        super().__init__(message)
        self.current = current


class SharedStateChannel(Protocol):
    def snapshot(self) -> SharedStateSnapshot:
        ...

    def apply_patch(
        self,
        *,
        source_call_id: str,
        base_revision: int,
        base_state_sha256: str,
        patch: list[dict[str, Any]],
    ) -> SharedStateSnapshot:
        ...

    def link_history(
        self,
        *,
        source_call_id: str,
        sequence: int,
        event_hash: str,
    ) -> SharedStateSnapshot:
        ...
