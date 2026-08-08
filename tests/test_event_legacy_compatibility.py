from __future__ import annotations

import pytest

from swaag.events import ALLOWED_EVENT_TYPES, LEGACY_EVENT_TYPES, EventSchemaError, create_event, verify_event_integrity


def test_planner_era_events_are_not_creatable_by_current_runtime() -> None:
    assert "plan_created" in LEGACY_EVENT_TYPES
    assert "plan_created" not in ALLOWED_EVENT_TYPES
    with pytest.raises(EventSchemaError, match="Unknown event type"):
        create_event(session_id="s", sequence=1, event_type="plan_created", payload={"goal": "g", "plan": {}}, prev_hash=None)


def test_legacy_v1_event_still_verifies_for_history_compatibility() -> None:
    # Build with the historical body/hash shape, then verify through the legacy-readable path.
    from swaag.events import HistoryEvent, compute_event_hash
    event = HistoryEvent(
        id="event_legacy",
        sequence=1,
        session_id="s",
        timestamp="2026-01-01T00:00:00Z",
        type="plan_created",
        version=1,
        payload={"goal": "g", "plan": {}},
        metadata={},
        prev_hash=None,
        hash="",
    )
    event.hash = compute_event_hash(event)
    verify_event_integrity(event, None)
