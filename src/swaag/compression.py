from __future__ import annotations

from typing import Any, Iterable

from swaag.types import Message


def _event_reference(
    *,
    sequence: Any,
    event_hash: Any,
    event_type: Any,
    session_id: Any,
    relationship: str,
) -> dict[str, Any] | None:
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
        return None
    return {
        "sequence": int(sequence),
        "hash": str(event_hash or ""),
        "event_type": str(event_type or ""),
        "session_id": str(session_id or ""),
        "relationship": relationship,
    }


def source_event_ranges(references: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_session: dict[str, set[int]] = {}
    for item in references:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sequence"), int)
            or isinstance(item.get("sequence"), bool)
            or int(item["sequence"]) <= 0
        ):
            continue
        by_session.setdefault(str(item.get("session_id", "")), set()).add(
            int(item["sequence"])
        )
    ranges: list[dict[str, Any]] = []
    for session_id, sequence_set in sorted(by_session.items()):
        sequences = sorted(sequence_set)
        start = previous = sequences[0]
        for sequence in sequences[1:]:
            if sequence == previous + 1:
                previous = sequence
                continue
            ranges.append(
                {
                    "session_id": session_id,
                    "start_sequence": start,
                    "end_sequence": previous,
                }
            )
            start = previous = sequence
        ranges.append(
            {
                "session_id": session_id,
                "start_sequence": start,
                "end_sequence": previous,
            }
        )
    return ranges


def message_source_event_references(messages: Iterable[Message]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()

    def add(reference: dict[str, Any] | None) -> None:
        if reference is None:
            return
        key = (
            str(reference.get("session_id", "")),
            int(reference["sequence"]),
            str(reference.get("event_type", "")),
            str(reference.get("relationship", "")),
        )
        if key in seen:
            return
        seen.add(key)
        references.append(reference)

    for message in messages:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        local_session_id = (
            metadata.get("source_message_session_id")
            or metadata.get("projection_session_id")
            or metadata.get("source_event_session_id")
            or ""
        )
        nested = metadata.get("source_event_references", [])
        if isinstance(nested, list):
            for item in nested:
                if not isinstance(item, dict):
                    continue
                add(
                    _event_reference(
                        sequence=item.get("sequence"),
                        event_hash=item.get("hash"),
                        event_type=item.get("event_type"),
                        session_id=item.get("session_id") or local_session_id,
                        relationship=str(item.get("relationship") or "summary_source"),
                    )
                )
        add(
            _event_reference(
                sequence=metadata.get("source_message_event_sequence"),
                event_hash=metadata.get("source_message_event_hash"),
                event_type=metadata.get("source_message_event_type", "message_added"),
                session_id=metadata.get("source_message_session_id"),
                relationship="message",
            )
        )
        add(
            _event_reference(
                sequence=metadata.get("source_event_sequence"),
                event_hash=metadata.get("source_event_hash"),
                event_type=metadata.get("source_event_type"),
                session_id=metadata.get("source_event_session_id") or local_session_id,
                relationship="authoritative_payload",
            )
        )
        add(
            _event_reference(
                sequence=metadata.get("projection_event_sequence"),
                event_hash=metadata.get("projection_event_hash"),
                event_type=metadata.get("projection_event_type"),
                session_id=metadata.get("projection_session_id"),
                relationship="derived_projection",
            )
        )
    return sorted(
        references,
        key=lambda item: (item["session_id"], item["sequence"], item["relationship"]),
    )


def summary_provenance_text(message: Message) -> str:
    if message.role != "summary" or not isinstance(message.metadata, dict):
        return ""
    ranges = message.metadata.get("source_event_ranges")
    if not isinstance(ranges, list) or not ranges:
        return ""
    rendered = ", ".join(
        (f"{item.get('session_id')}:" if item.get("session_id") else "")
        + (
            str(item.get("start_sequence"))
            if item.get("start_sequence") == item.get("end_sequence")
            else f"{item.get('start_sequence')}-{item.get('end_sequence')}"
        )
        for item in ranges
        if isinstance(item, dict)
    )
    projection_sequence = message.metadata.get("projection_event_sequence", "unknown")
    return (
        "[DERIVED HISTORY SUMMARY; raw events remain authoritative; "
        f"projection_event={projection_sequence}; source_event_ranges={rendered}; "
        "use history_window to re-expand exact events]\n"
    )


def summary_message_payload(
    summary_text: str,
    *,
    source_message_count: int,
    created_at: str,
    source_message_start: int = 0,
    source_event_references: Iterable[dict[str, Any]] = (),
) -> dict:
    references = [dict(item) for item in source_event_references]
    return {
        "role": "summary",
        "content": summary_text,
        "created_at": created_at,
        "metadata": {
            "projection_kind": "history_summary",
            "source_message_start": int(source_message_start),
            "source_message_count": int(source_message_count),
            "source_event_references": references,
            "source_event_ranges": source_event_ranges(references),
        },
    }
