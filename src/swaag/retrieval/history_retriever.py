from __future__ import annotations

from collections.abc import Iterable
import re

from swaag.config import AgentConfig
from swaag.retrieval.ranker import RetrievalCandidate
from swaag.types import HistoryEvent, Message, SessionState


_GENERATED_ID_RE = re.compile(r"\b(?:mem|note|session|plan)_[a-z0-9]+\b")
_ISO_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00\b")


def _sanitize_event_preview(event_type: str, payload) -> str:
    text = str(payload)
    text = _GENERATED_ID_RE.sub("<generated_id>", text)
    text = _ISO_TIMESTAMP_RE.sub("<timestamp>", text)
    return text


def _sanitize_tool_message(message: Message) -> str:
    metadata = message.metadata or {}
    output = metadata.get("output") if isinstance(metadata, dict) else None
    details: list[str] = []
    if isinstance(output, dict):
        path = str(output.get("source_ref") or output.get("path") or output.get("target_path") or "").strip()
        if path:
            details.append(f"path={path}")
        if "finished" in output:
            details.append(f"finished={bool(output.get('finished'))}")
        if "changed" in output:
            details.append(f"changed={bool(output.get('changed'))}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"tool={message.name or 'unknown'} untrusted_output_omitted{suffix}"


def history_candidates(
    state: SessionState,
    *,
    config: AgentConfig,
    counter,
    history_events: Iterable[HistoryEvent] | None = None,
    for_planning: bool = False,
) -> list[RetrievalCandidate]:
    pol = config.selection_policy
    candidates: list[RetrievalCandidate] = []
    total = max(len(state.messages), 1)
    for index, message in enumerate(state.messages):
        sanitize_tool = for_planning and message.role == "tool"
        full_text = _sanitize_tool_message(message) if sanitize_tool else message.content
        ranking_text = full_text[: pol.retrieval_history_ranking_chars]
        payload = message
        if sanitize_tool:
            payload = Message(
                role=message.role,
                content=full_text,
                name=message.name,
                created_at=message.created_at,
                metadata=message.metadata,
            )
        if message.role == "tool":
            structural = pol.retrieval_structural_tool_message
        elif message.role == "user":
            structural = pol.retrieval_structural_user_message
        else:
            structural = 0.0
        candidates.append(
            RetrievalCandidate(
                item_type="history_message",
                item_id=f"message:{index}",
                source="messages",
                text=ranking_text,
                token_cost=counter.count_text(full_text).tokens,
                payload=payload,
                metadata={"recency": (index + 1) / total, "structural": structural},
            )
        )
    if history_events is None:
        return candidates
    events = list(history_events)
    event_total = max(len(events), 1)
    for index, event in enumerate(events):
        payload_text = _sanitize_event_preview(event.event_type, event.payload)
        preview = payload_text[: pol.retrieval_event_ranking_chars]
        if for_planning and event.event_type in {"tool_result", "tool_called"}:
            preview = "tool_event untrusted_output_omitted"
        structural = 0.0
        if event.event_type in {"verification_failed", "error", "step_failed"}:
            structural = pol.retrieval_structural_failed_event
        elif event.event_type in {"plan_created", "plan_updated", "subagent_reported", "summary_created", "history_compressed"}:
            structural = pol.retrieval_structural_summary_event
        candidates.append(
            RetrievalCandidate(
                item_type="history_event",
                item_id=f"event:{event.sequence}",
                source="history",
                text=f"{event.event_type} {preview}",
                token_cost=counter.count_text(preview).tokens,
                payload=event,
                metadata={"recency": (index + 1) / event_total, "structural": structural},
            )
        )
    return candidates
