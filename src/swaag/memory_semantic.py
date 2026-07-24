from __future__ import annotations

from dataclasses import asdict

from swaag.config import AgentConfig
from swaag.security import provenance_for_event, semantic_item_is_trusted, should_promote_to_semantic, trust_level_for_event
from swaag.types import HistoryEvent, SemanticMemoryItem, SessionState, TrustLevel
from swaag.utils import new_id, stable_json_dumps


_MEMORY_EVENT_TYPES = frozenset({"tool_result", "step_completed"})


def _snapshot_item(
    *,
    event: HistoryEvent,
    trust_level: TrustLevel,
) -> SemanticMemoryItem:
    snapshot = {
        "event_type": event.event_type,
        "payload": event.payload,
        "metadata": event.metadata,
    }
    tool_name = str(event.payload.get("tool_name", "")).strip()
    tags = [event.event_type]
    if tool_name:
        tags.append(f"tool:{tool_name}")
    return SemanticMemoryItem(
        memory_id=new_id("mem"),
        memory_kind="event_snapshot",
        content=stable_json_dumps(snapshot),
        source_event_id=event.id,
        trust_level=trust_level,
        tags=tags,
        created_at=event.timestamp,
        metadata={
            "source_event_type": event.event_type,
            "source_sequence": event.sequence,
            "provenance": provenance_for_event(event.event_type, event.payload, event.id),
            "raw_event": snapshot,
        },
    )


def extract_from_event(config: AgentConfig, event: HistoryEvent) -> tuple[list[SemanticMemoryItem], str | None]:
    trust_level = event.metadata.get("trust_level") if isinstance(event.metadata, dict) else None
    if trust_level not in {"trusted", "untrusted", "derived"}:
        trust_level = trust_level_for_event(event.event_type, event.payload, event.metadata)
    if event.event_type not in _MEMORY_EVENT_TYPES:
        return [], f"event type {event.event_type} does not produce memory snapshot"
    if not should_promote_to_semantic(config, trust_level=trust_level):
        return [], f"trust level {trust_level} is not promotable"
    return [_snapshot_item(event=event, trust_level=trust_level)], None


def retrieve_memory(config: AgentConfig, state: SessionState, query: str, *, limit: int) -> list[SemanticMemoryItem]:
    del query
    eligible = [item for item in state.semantic_memory if semantic_item_is_trusted(config, item)]
    eligible.sort(
        key=lambda item: (
            int(item.metadata.get("source_sequence", 0)) if isinstance(item.metadata, dict) else 0,
            item.created_at,
            item.memory_id,
        ),
        reverse=True,
    )
    return eligible[:limit]


def state_semantic_snapshot(state: SessionState) -> list[dict]:
    return [asdict(item) for item in state.semantic_memory]
