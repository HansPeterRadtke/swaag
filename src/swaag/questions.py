"""Question lifecycle derived from canonical history, never a second authority."""
from __future__ import annotations

from typing import Any

RESOLUTIONS = ("answered_by_user", "answered_by_research", "superseded", "irrelevant", "expired")


def validate_question_capacity(state, questions, config) -> None:
    """Reject overflow before committing an action or executing its tools."""
    if not questions:
        return
    if len(state.open_questions) + len(questions) > config.runtime.max_open_questions:
        raise ValueError("Open-question capacity reached; resolve existing questions before adding more")
    existing = sum(len(str(item.get(key, ""))) for item in state.open_questions
                   for key in ("question", "reason", "assumption_if_unanswered"))
    added = sum(len(getattr(item, key)) for item in questions
                for key in ("question", "reason", "assumption_if_unanswered"))
    if existing + added > config.runtime.max_open_question_chars:
        raise ValueError("Open-question character budget reached; resolve existing questions first")


def question_record(event) -> dict[str, Any]:
    return {"question_id": event.id, "source_event_sequence": event.sequence,
            "created_at": event.timestamp,
            **{key: event.payload[key] for key in
               ("question", "criticality", "reason", "assumption_if_unanswered")}}
