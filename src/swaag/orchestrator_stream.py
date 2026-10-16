"""Replay committed foreground speech without holding up execution or TTS consumers."""
from __future__ import annotations

from typing import Any


class OrchestratorEventCursor:
    """Follow only the turn that consumed this durable finalized-message receipt.

    Replaying linkage from the start allows reconnects after process loss. Reads
    are paged and retained memory is bounded; the history index serves followups.
    Only complete, authority-fenced assistant progress is exposed, never reasoning.
    """
    def __init__(self, history, receipt: dict[str, Any], *, after_sequence: int = 0):
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
            raise ValueError("after_sequence must be a non-negative integer")
        self.history = history
        self.session_id = str(receipt["session_id"])
        self.control_id = str(receipt["control_id"])
        self.after_sequence = after_sequence
        self.next_sequence = 1
        self.active = False
        self.finished = False
        self.final_sequence = None

    def poll(self):
        if self.finished:
            return []
        visible = []
        events = self.history.read_history_window(
            self.session_id, start_sequence=self.next_sequence, limit=128,
        )
        for event in events:
            self.next_sequence = event.sequence + 1
            if (event.event_type == "control_message_processed"
                    and event.payload.get("control_id") == self.control_id):
                self.active = True
            elif self.active and event.event_type == "turn_started":
                # Explicit recovery links survive process loss; unrelated later
                # turns must never be attributed to this receipt.
                if self.control_id not in event.payload.get("resumed_control_ids", []):
                    self.finished = True
                    break
            elif self.active and event.event_type == "turn_finished":
                self.finished = True
                self.final_sequence = event.sequence
                break
            elif (self.active and event.event_type == "assistant_progress"
                    and event.sequence > self.after_sequence):
                visible.append({
                    "event": "assistant_progress", "session_id": self.session_id,
                    "control_id": self.control_id, "sequence": event.sequence,
                    "assistant_text": str(event.payload["assistant_text"]),
                })
        return visible

    def caught_up(self):
        return self.finished or not self.history.read_history_window(
            self.session_id, start_sequence=self.next_sequence, limit=1,
        )
