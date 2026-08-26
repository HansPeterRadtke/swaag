from __future__ import annotations


def summary_message_payload(summary_text: str, *, source_message_count: int, created_at: str) -> dict:
    return {
        "role": "summary",
        "content": summary_text,
        "created_at": created_at,
        "source_message_count": int(source_message_count),
    }
