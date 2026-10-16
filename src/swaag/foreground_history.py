"""Exact conversational view with prior machine actions available on demand."""
from dataclasses import replace
from swaag.types import Message
from swaag.compression import message_source_event_references
from swaag.utils import stable_json_dumps


def render_foreground_completion_history(rows):
    """Losslessly encode ordinary conversation rows with one explicit column map.

    Extended messages and evidence objects retain every field. Mixed entries
    stay in their original order, and JSON escaping preserves message boundaries.
    """
    columns = ('sequence', 'role', 'content')
    return stable_json_dumps({
        'conversation_array_columns': list(columns),
        'ordered_history': [
            [row[key] for key in columns] if set(row) == set(columns) else row
            for row in rows
        ],
    })


def fold_identical_prior_user_messages(messages, turn_sequence):
    """Lossless run-length encoding; no utterance or repetition is discarded."""
    current_user_sequence = max((m.metadata.get('source_message_event_sequence', 0)
        for m in messages if m.role == 'user'
        and m.metadata.get('source_message_event_sequence', 0) < turn_sequence), default=0)
    result = []
    for message in messages:
        sequence = message.metadata.get('source_message_event_sequence')
        previous = result[-1] if result else None
        if (message.role == 'user' and previous is not None and previous.role == 'user'
                and message.content == previous.content and message.name == previous.name
                and isinstance(sequence, int) and sequence < current_user_sequence):
            prior_sequences = previous.metadata.get('verbatim_repetition_sequences',
                [previous.metadata.get('source_message_event_sequence')])
            result[-1] = replace(previous, metadata={**previous.metadata,
                'verbatim_repetition_sequences': [*prior_sequences, sequence],
                'source_event_references': message_source_event_references([previous, message])})
        else:
            result.append(message)
    return result


def foreground_action_history(messages, events, turn_sequence):
    """Defer only typed, earlier-turn assistant actions, never user authority.

    Effect receipts and every visible utterance remain exact. Older read-only
    snapshot payloads remain retrievable by their exact tool-event identity.
    Accepted progress is
    reconstructed from its delivery event, not from an unverified draft inside
    the action envelope. The durable events themselves are never changed.
    """
    deferred = []
    snapshot_reads = []
    kept = []
    for message in messages:
        sequence = message.metadata.get("source_message_event_sequence")
        output = message.metadata.get("output")
        source_sequence = message.metadata.get("source_event_sequence")
        if (message.role == "tool" and message.name == "runtime_snapshot_lookup"
                and isinstance(sequence, int) and not isinstance(sequence, bool)
                and sequence < turn_sequence and isinstance(source_sequence, int)
                and message.metadata.get("source_event_type") == "tool_result"
                and not message.metadata.get("context_updates")
                and isinstance(output, dict)
                and set(output) == {"observed_at", "pointer", "snapshot_hash", "snapshot_sequence", "value"}):
            reference = {key: value for key, value in output.items() if key != "value"}
            reference["value"] = {"exact_tool_result_event": source_sequence,
                "notice": "Earlier read-only snapshot; retrieve this event before historical claims."}
            kept.append(replace(message, content=stable_json_dumps(reference),
                metadata={**message.metadata, "output": reference}))
            snapshot_reads.append(source_sequence)
            continue
        if (message.role == "assistant" and message.metadata.get("internal_action") is True
                and isinstance(sequence, int) and not isinstance(sequence, bool)
                and sequence < turn_sequence):
            deferred.append(sequence)
        else:
            kept.append(message)
    if not deferred and not snapshot_reads:
        return messages, ""
    for event in events:
        if event.event_type != "assistant_progress" or event.sequence >= turn_sequence:
            continue
        text = event.payload.get("assistant_text")
        if isinstance(text, str) and text.strip():
            kept.append(Message(role="assistant", content=text, created_at=event.timestamp,
                metadata={"source_message_event_sequence": event.sequence,
                          "source_message_event_hash": event.hash,
                          "source_message_event_type": event.event_type,
                          "source_message_session_id": event.session_id,
                          "delivered_progress": True}))
    # Authoritative messages have sequence provenance. Unknown records stay in
    # their existing relative order; this view never drops them by inference.
    kept.sort(key=lambda message: message.metadata.get("source_message_event_sequence", 0))
    notice = (
        f"{len(deferred)} prior-turn internal assistant action envelopes are deferred. "
        f"{len(snapshot_reads)} earlier read-only snapshot payloads are referenced by exact tool-event sequence. "
        "Every user message, visible assistant reply, accepted progress utterance, "
        "effect receipt and current-turn action is retained exactly. Internal drafts "
        "are not evidence of execution or user authorization. If prior machine "
        "actions or referenced snapshot payloads are needed, discover history_window/history_search and read their "
        "exact durable events. No source history has been deleted.\n\n"
    )
    return fold_identical_prior_user_messages(kept, turn_sequence), notice
