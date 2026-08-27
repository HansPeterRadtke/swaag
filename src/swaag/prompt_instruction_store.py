from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from swaag.config import AgentConfig
from swaag.prompt_instructions import (
    PromptInstructionError,
    enforce_prompt_instruction_limits,
    make_prompt_instruction,
)
from swaag.sqlite_schema import apply_sqlite_migrations
from swaag.types import PromptInstruction
from swaag.utils import new_id, sha256_text, stable_json_dumps, utc_now_iso


class PromptInstructionStoreError(RuntimeError):
    pass


_MIGRATIONS = (
    (
        """
        CREATE TABLE IF NOT EXISTS prompt_instruction_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            instruction_id TEXT NOT NULL,
            instruction_json TEXT,
            origin_session_id TEXT NOT NULL,
            previous_hash TEXT,
            event_hash TEXT NOT NULL UNIQUE
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS prompt_instruction_events_instruction
        ON prompt_instruction_events(instruction_id, sequence)
        """,
    ),
)


@dataclass(slots=True, frozen=True)
class PromptInstructionStoreEvent:
    sequence: int
    event_id: str
    timestamp: str
    action: str
    instruction_id: str
    instruction: PromptInstruction | None
    origin_session_id: str
    previous_hash: str | None
    event_hash: str


@dataclass(slots=True, frozen=True)
class PromptInstructionStoreMutation:
    instruction: PromptInstruction | None
    event: PromptInstructionStoreEvent


def _event_material(
    *,
    sequence: int,
    event_id: str,
    timestamp: str,
    action: str,
    instruction_id: str,
    instruction_payload: dict[str, Any] | None,
    origin_session_id: str,
    previous_hash: str | None,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "event_id": event_id,
        "timestamp": timestamp,
        "action": action,
        "instruction_id": instruction_id,
        "instruction": instruction_payload,
        "origin_session_id": origin_session_id,
        "previous_hash": previous_hash,
    }


class PromptInstructionStore:
    """Append-only instructions shared by all sessions for one local user."""

    def __init__(self, root: Path, config: AgentConfig):
        self.path = Path(root).expanduser() / "user_prompt_instructions.sqlite3"
        self.config = config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            apply_sqlite_migrations(
                connection,
                store_name="user prompt instruction store",
                migrations=_MIGRATIONS,
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @staticmethod
    def _decode_instruction(raw: str | None) -> tuple[dict[str, Any] | None, PromptInstruction | None]:
        if raw is None:
            return None, None
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise PromptInstructionStoreError(
                "prompt instruction event payload must be an object"
            )
        return payload, PromptInstruction(**payload)

    def _read_events(
        self, connection: sqlite3.Connection
    ) -> list[PromptInstructionStoreEvent]:
        rows = connection.execute(
            """
            SELECT sequence, event_id, timestamp, action, instruction_id,
                   instruction_json, origin_session_id, previous_hash, event_hash
            FROM prompt_instruction_events
            ORDER BY sequence
            """
        ).fetchall()
        events: list[PromptInstructionStoreEvent] = []
        expected_previous: str | None = None
        for row in rows:
            payload, instruction = self._decode_instruction(row["instruction_json"])
            material = _event_material(
                sequence=int(row["sequence"]),
                event_id=str(row["event_id"]),
                timestamp=str(row["timestamp"]),
                action=str(row["action"]),
                instruction_id=str(row["instruction_id"]),
                instruction_payload=payload,
                origin_session_id=str(row["origin_session_id"]),
                previous_hash=row["previous_hash"],
            )
            calculated = sha256_text(stable_json_dumps(material, indent=None))
            if row["previous_hash"] != expected_previous:
                raise PromptInstructionStoreError(
                    "user prompt instruction event chain is discontinuous"
                )
            if str(row["event_hash"]) != calculated:
                raise PromptInstructionStoreError(
                    "user prompt instruction event hash verification failed"
                )
            event = PromptInstructionStoreEvent(
                sequence=int(row["sequence"]),
                event_id=str(row["event_id"]),
                timestamp=str(row["timestamp"]),
                action=str(row["action"]),
                instruction_id=str(row["instruction_id"]),
                instruction=instruction,
                origin_session_id=str(row["origin_session_id"]),
                previous_hash=row["previous_hash"],
                event_hash=str(row["event_hash"]),
            )
            events.append(event)
            expected_previous = event.event_hash
        return events

    @staticmethod
    def _rebuild(
        events: list[PromptInstructionStoreEvent],
    ) -> list[PromptInstruction]:
        current: dict[str, PromptInstruction] = {}
        order: list[str] = []
        for event in events:
            if event.action in {"add", "replace"}:
                if event.instruction is None:
                    raise PromptInstructionStoreError(
                        f"{event.action} event is missing its instruction"
                    )
                instruction = PromptInstruction(**asdict(event.instruction))
                instruction.metadata = dict(instruction.metadata)
                instruction.metadata.update(
                    {
                        "instruction_store": "user",
                        "store_event_sequence": event.sequence,
                        "store_event_id": event.event_id,
                        "store_event_hash": event.event_hash,
                        "store_origin_session_id": event.origin_session_id,
                    }
                )
                if event.instruction_id not in order:
                    order.append(event.instruction_id)
                current[event.instruction_id] = instruction
            elif event.action == "remove":
                current.pop(event.instruction_id, None)
                order = [item for item in order if item != event.instruction_id]
            else:
                raise PromptInstructionStoreError(
                    f"unknown user prompt instruction event action: {event.action}"
                )
        return [current[instruction_id] for instruction_id in order]

    def list(self) -> list[PromptInstruction]:
        with self._connect() as connection:
            return self._rebuild(self._read_events(connection))

    def events(self) -> list[PromptInstructionStoreEvent]:
        with self._connect() as connection:
            return self._read_events(connection)

    def _append(
        self,
        *,
        action: str,
        instruction_id: str,
        instruction: PromptInstruction | None,
        origin_session_id: str,
    ) -> PromptInstructionStoreMutation:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                events = self._read_events(connection)
                current = self._rebuild(events)
                existing = next(
                    (
                        item
                        for item in current
                        if item.instruction_id == instruction_id
                    ),
                    None,
                )
                if action == "add" and existing is not None:
                    raise PromptInstructionError(
                        f"user prompt instruction already exists: {instruction_id}"
                    )
                if action in {"replace", "remove"} and existing is None:
                    raise PromptInstructionError(
                        f"unknown user prompt instruction: {instruction_id}"
                    )
                if action in {"add", "replace"} and instruction is None:
                    raise PromptInstructionError(
                        f"{action} requires a prompt instruction"
                    )
                if action == "replace":
                    assert existing is not None
                    assert instruction is not None
                    instruction.created_at = existing.created_at
                candidate = [
                    item
                    for item in current
                    if item.instruction_id != instruction_id
                ]
                if instruction is not None:
                    candidate.append(instruction)
                enforce_prompt_instruction_limits(self.config, candidate)

                sequence = (events[-1].sequence + 1) if events else 1
                event_id = new_id("user_instruction_event")
                timestamp = utc_now_iso()
                previous_hash = events[-1].event_hash if events else None
                payload = asdict(instruction) if instruction is not None else None
                material = _event_material(
                    sequence=sequence,
                    event_id=event_id,
                    timestamp=timestamp,
                    action=action,
                    instruction_id=instruction_id,
                    instruction_payload=payload,
                    origin_session_id=origin_session_id,
                    previous_hash=previous_hash,
                )
                event_hash = sha256_text(stable_json_dumps(material, indent=None))
                connection.execute(
                    """
                    INSERT INTO prompt_instruction_events (
                        sequence, event_id, timestamp, action, instruction_id,
                        instruction_json, origin_session_id, previous_hash, event_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sequence,
                        event_id,
                        timestamp,
                        action,
                        instruction_id,
                        None
                        if payload is None
                        else stable_json_dumps(payload, indent=None),
                        origin_session_id,
                        previous_hash,
                        event_hash,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        event = PromptInstructionStoreEvent(
            sequence=sequence,
            event_id=event_id,
            timestamp=timestamp,
            action=action,
            instruction_id=instruction_id,
            instruction=instruction,
            origin_session_id=origin_session_id,
            previous_hash=previous_hash,
            event_hash=event_hash,
        )
        return PromptInstructionStoreMutation(instruction=instruction, event=event)

    def add(
        self,
        *,
        title: str,
        content: str,
        scopes: list[str],
        categories: list[str] | None = None,
        origin_session_id: str,
    ) -> PromptInstructionStoreMutation:
        instruction = make_prompt_instruction(
            self.config,
            title=title,
            content=content,
            scopes=scopes,
            categories=categories,
        )
        return self._append(
            action="add",
            instruction_id=instruction.instruction_id,
            instruction=instruction,
            origin_session_id=origin_session_id,
        )

    def replace(
        self,
        *,
        instruction_id: str,
        title: str,
        content: str,
        scopes: list[str],
        categories: list[str] | None = None,
        origin_session_id: str,
    ) -> PromptInstructionStoreMutation:
        instruction = make_prompt_instruction(
            self.config,
            title=title,
            content=content,
            scopes=scopes,
            categories=categories,
            instruction_id=instruction_id,
        )
        return self._append(
            action="replace",
            instruction_id=instruction_id,
            instruction=instruction,
            origin_session_id=origin_session_id,
        )

    def remove(
        self,
        *,
        instruction_id: str,
        origin_session_id: str,
    ) -> PromptInstructionStoreMutation:
        return self._append(
            action="remove",
            instruction_id=instruction_id,
            instruction=None,
            origin_session_id=origin_session_id,
        )
