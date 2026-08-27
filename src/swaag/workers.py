from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from swaag.preemption import ModelCallStateChanged, RunCancellationRequested
from swaag.runtime import AgentRuntime
from swaag.sqlite_schema import apply_sqlite_migrations
from swaag.structured_output import (
    CallerOutputSpec,
    merge_caller_output,
    prepare_caller_output_spec,
)
from swaag.types import AttachmentReference, HistoryEvent
from swaag.utils import new_id, stable_json_dumps, utc_now_iso


WORKER_TERMINAL_STATES = frozenset({"completed", "failed", "canceled"})
WORKER_RESUMABLE_STATES = frozenset({"failed", "canceled", "input_required", "completed"})
WORKER_ACTIVE_STATES = frozenset({"queued", "working", "cancellation_requested"})
WORKER_COMPLETION_MODES = frozenset({"natural", "continuous"})
WORKER_PRESENTATION_MODES = frozenset({"visual", "audio"})
_CONTINUOUS_WORKER_CONTROL = (
    "This worker has explicit continuous completion mode. Treat the previous cycle's "
    "result as provisional, reassess the original objective and authoritative evidence, "
    "and choose the next materially useful improvement, verification, research step, or "
    "experiment. Do not merely restate the provisional result. The worker remains active "
    "until an explicit cancellation or a genuinely blocking question."
)
WORKER_STREAM_EVENT_TYPES = frozenset(
    {
        "agent_question",
        "agent_status",
        "assistant_progress",
        "tool_called",
        "tool_error",
        "tool_result",
    }
)
_WORKER_STORE_MIGRATIONS = (
    (
        """
        CREATE TABLE IF NOT EXISTS workers (
            worker_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL UNIQUE,
            objective TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            archived_at TEXT,
            result TEXT,
            error TEXT,
            run_count INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS workers_status_updated
        ON workers(status, updated_at, worker_id)
        """,
        """
        CREATE TABLE IF NOT EXISTS worker_events (
            worker_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            event_id TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (worker_id, sequence),
            FOREIGN KEY (worker_id) REFERENCES workers(worker_id)
        )
        """,
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS worker_history_cursors (
            worker_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            through_sequence INTEGER NOT NULL,
            FOREIGN KEY (worker_id) REFERENCES workers(worker_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS worker_history_links (
            worker_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            history_sequence INTEGER NOT NULL,
            history_event_id TEXT NOT NULL,
            history_event_hash TEXT NOT NULL,
            worker_event_id TEXT NOT NULL UNIQUE,
            PRIMARY KEY (worker_id, session_id, history_sequence),
            FOREIGN KEY (worker_id) REFERENCES workers(worker_id)
        )
        """,
    ),
    (
        """
        ALTER TABLE workers
        ADD COLUMN completion_mode TEXT NOT NULL DEFAULT 'natural'
        """,
    ),
    (
        """
        ALTER TABLE workers
        ADD COLUMN presentation_modes_json TEXT NOT NULL DEFAULT '[]'
        """,
    ),
)


@dataclass(slots=True, frozen=True)
class WorkerRecord:
    worker_id: str
    session_id: str
    objective: str
    status: str
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    archived_at: str | None
    result: str | None
    error: str | None
    run_count: int
    completion_mode: str
    presentation_modes: list[str]


@dataclass(slots=True, frozen=True)
class WorkerEvent:
    event_id: str
    worker_id: str
    sequence: int
    timestamp: str
    event_type: str
    payload: dict[str, Any]


class WorkerStore:
    """Durable mechanical worker/task state, separate from semantic session history."""

    def __init__(self, root: Path):
        self.path = Path(root).expanduser() / "workers.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            apply_sqlite_migrations(
                connection,
                store_name="worker store",
                migrations=_WORKER_STORE_MIGRATIONS,
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _record(row: sqlite3.Row) -> WorkerRecord:
        payload = dict(row)
        raw_modes = json.loads(str(payload.pop("presentation_modes_json", "[]")))
        if not isinstance(raw_modes, list) or not all(
            isinstance(item, str) and item in WORKER_PRESENTATION_MODES
            for item in raw_modes
        ):
            raise RuntimeError("worker store contains invalid presentation modes")
        payload["presentation_modes"] = list(raw_modes)
        return WorkerRecord(**payload)

    @staticmethod
    def _event(row: sqlite3.Row) -> WorkerEvent:
        payload = dict(row)
        payload["payload"] = json.loads(str(payload.pop("payload_json")))
        return WorkerEvent(**payload)

    def create(
        self,
        session_id: str,
        objective: str,
        *,
        output_spec: CallerOutputSpec | None = None,
        completion_mode: str = "natural",
        presentation_modes: list[str] | None = None,
    ) -> WorkerRecord:
        text = objective.strip()
        if not text:
            raise ValueError("worker objective must not be empty")
        mode = str(completion_mode).strip()
        if mode not in WORKER_COMPLETION_MODES:
            raise ValueError(
                "worker completion_mode must be one of "
                f"{sorted(WORKER_COMPLETION_MODES)}"
            )
        requested_presentations = sorted(set(presentation_modes or []))
        unknown_presentations = (
            set(requested_presentations) - WORKER_PRESENTATION_MODES
        )
        if unknown_presentations:
            raise ValueError(
                "worker presentation_modes must contain only visual and/or audio"
            )
        worker_id = new_id("worker")
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO workers(
                    worker_id, session_id, objective, status, created_at, updated_at,
                    completion_mode, presentation_modes_json
                ) VALUES (?, ?, ?, 'created', ?, ?, ?, ?)
                """,
                (
                    worker_id,
                    session_id,
                    text,
                    now,
                    now,
                    mode,
                    stable_json_dumps(requested_presentations, indent=None),
                ),
            )
            self._append_event(
                connection,
                worker_id,
                "worker_created",
                {
                    "session_id": session_id,
                    "objective": text,
                    "status": "created",
                    "completion_mode": mode,
                    "presentation_modes": requested_presentations,
                    "caller_output_spec": (
                        output_spec.payload() if output_spec is not None else None
                    ),
                },
                timestamp=now,
            )
            connection.commit()
        return self.get(worker_id)

    def get(self, worker_id: str) -> WorkerRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workers WHERE worker_id=?", (worker_id,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Unknown worker: {worker_id}")
        return self._record(row)

    def snapshot_with_event_cursor(self, worker_id: str) -> tuple[WorkerRecord, int]:
        """Read task state and its event cursor from one SQLite snapshot."""
        with self._connect() as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT * FROM workers WHERE worker_id=?", (worker_id,)
            ).fetchone()
            if row is None:
                raise FileNotFoundError(f"Unknown worker: {worker_id}")
            cursor = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM worker_events WHERE worker_id=?",
                (worker_id,),
            ).fetchone()
        return self._record(row), int(cursor[0])

    def list(
        self,
        *,
        statuses: Iterable[str] | None = None,
        include_archived: bool = False,
    ) -> list[WorkerRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        status_values = sorted({str(item) for item in statuses or () if str(item)})
        if status_values:
            clauses.append("status IN (" + ",".join("?" for _ in status_values) + ")")
            params.extend(status_values)
        if not include_archived:
            clauses.append("archived_at IS NULL")
        sql = "SELECT * FROM workers"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, worker_id"
        with self._connect() as connection:
            return [self._record(row) for row in connection.execute(sql, params)]

    def events(self, worker_id: str, *, after_sequence: int = 0) -> list[WorkerEvent]:
        self.get(worker_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, worker_id, sequence, timestamp, event_type, payload_json
                FROM worker_events
                WHERE worker_id=? AND sequence>?
                ORDER BY sequence
                """,
                (worker_id, max(0, int(after_sequence))),
            ).fetchall()
        return [self._event(row) for row in rows]

    def append_event(
        self, worker_id: str, event_type: str, payload: dict[str, Any]
    ) -> WorkerEvent:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM workers WHERE worker_id=?", (worker_id,)
            ).fetchone() is None:
                raise FileNotFoundError(f"Unknown worker: {worker_id}")
            event = self._append_event(connection, worker_id, event_type, payload)
            connection.commit()
        return event

    def history_cursor(self, worker_id: str) -> int:
        self.get(worker_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT through_sequence FROM worker_history_cursors WHERE worker_id=?",
                (worker_id,),
            ).fetchone()
        return 0 if row is None else int(row[0])

    def sync_history_references(
        self,
        worker_id: str,
        session_id: str,
        events: Iterable[HistoryEvent],
    ) -> list[WorkerEvent]:
        """Atomically advance a history cursor and link stream-worthy source events."""
        source_events = list(events)
        if not source_events:
            return []
        appended: list[WorkerEvent] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT session_id FROM workers WHERE worker_id=?", (worker_id,)
            ).fetchone()
            if row is None:
                raise FileNotFoundError(f"Unknown worker: {worker_id}")
            if str(row[0]) != session_id:
                raise ValueError(f"Worker {worker_id} does not own session {session_id}")
            cursor_row = connection.execute(
                "SELECT through_sequence FROM worker_history_cursors WHERE worker_id=?",
                (worker_id,),
            ).fetchone()
            through_sequence = 0 if cursor_row is None else int(cursor_row[0])
            for source in source_events:
                source_sequence = int(source.sequence)
                if source_sequence <= through_sequence:
                    continue
                if source.event_type in WORKER_STREAM_EVENT_TYPES:
                    existing = connection.execute(
                        """
                        SELECT 1 FROM worker_history_links
                        WHERE worker_id=? AND session_id=? AND history_sequence=?
                        """,
                        (worker_id, session_id, source_sequence),
                    ).fetchone()
                    if existing is None:
                        linked = self._append_event(
                            connection,
                            worker_id,
                            "worker_history_event",
                            {
                                "session_id": session_id,
                                "history_sequence": source_sequence,
                                "history_event_id": str(source.id),
                                "history_event_hash": str(source.hash),
                                "history_event_type": str(source.event_type),
                            },
                            timestamp=str(source.timestamp),
                        )
                        connection.execute(
                            """
                            INSERT INTO worker_history_links(
                                worker_id, session_id, history_sequence,
                                history_event_id, history_event_hash, worker_event_id
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                worker_id,
                                session_id,
                                source_sequence,
                                str(source.id),
                                str(source.hash),
                                linked.event_id,
                            ),
                        )
                        appended.append(linked)
                through_sequence = max(through_sequence, source_sequence)
            connection.execute(
                """
                INSERT INTO worker_history_cursors(worker_id, session_id, through_sequence)
                VALUES (?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    through_sequence=MAX(
                        worker_history_cursors.through_sequence,
                        excluded.through_sequence
                    )
                """,
                (worker_id, session_id, through_sequence),
            )
            connection.commit()
        return appended

    def transition(
        self,
        worker_id: str,
        status: str,
        *,
        expected: Iterable[str] | None = None,
        result: str | None = None,
        error: str | None = None,
        increment_run_count: bool = False,
        event_type: str = "worker_status_changed",
        event_payload: dict[str, Any] | None = None,
    ) -> WorkerRecord:
        now = utc_now_iso()
        expected_states = set(expected or ())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM workers WHERE worker_id=?", (worker_id,)
            ).fetchone()
            if row is None:
                raise FileNotFoundError(f"Unknown worker: {worker_id}")
            current = self._record(row)
            if expected_states and current.status not in expected_states:
                raise ValueError(
                    f"Worker {worker_id} is {current.status}; expected one of {sorted(expected_states)}"
                )
            started_at = current.started_at
            if status == "working" and started_at is None:
                started_at = now
            completed_at = now if status in {"completed", "failed", "canceled"} else None
            run_count = current.run_count + (1 if increment_run_count else 0)
            connection.execute(
                """
                UPDATE workers SET
                    status=?, updated_at=?, started_at=?, completed_at=?, archived_at=?,
                    result=?, error=?, run_count=?
                WHERE worker_id=?
                """,
                (
                    status,
                    now,
                    started_at,
                    completed_at,
                    current.archived_at,
                    result,
                    error,
                    run_count,
                    worker_id,
                ),
            )
            payload = {
                "from_status": current.status,
                "to_status": status,
                "run_count": run_count,
                **dict(event_payload or {}),
            }
            if result is not None:
                payload["result"] = result
            if error is not None:
                payload["error"] = error
            self._append_event(connection, worker_id, event_type, payload, timestamp=now)
            connection.commit()
        return self.get(worker_id)

    def mark_archived(
        self, worker_id: str, *, archive: dict[str, Any]
    ) -> WorkerRecord:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM workers WHERE worker_id=?", (worker_id,)
            ).fetchone()
            if row is None:
                raise FileNotFoundError(f"Unknown worker: {worker_id}")
            current = self._record(row)
            if current.archived_at is not None:
                return current
            connection.execute(
                "UPDATE workers SET archived_at=?, updated_at=? WHERE worker_id=?",
                (now, now, worker_id),
            )
            self._append_event(
                connection,
                worker_id,
                "worker_archived",
                {"status": current.status, "archive": archive},
                timestamp=now,
            )
            connection.commit()
        return self.get(worker_id)

    def _append_event(
        self,
        connection: sqlite3.Connection,
        worker_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        timestamp: str | None = None,
    ) -> WorkerEvent:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM worker_events WHERE worker_id=?",
            (worker_id,),
        ).fetchone()
        sequence = int(row[0])
        event = WorkerEvent(
            event_id=new_id("worker_event"),
            worker_id=worker_id,
            sequence=sequence,
            timestamp=timestamp or utc_now_iso(),
            event_type=event_type,
            payload=dict(payload),
        )
        connection.execute(
            """
            INSERT INTO worker_events(
                worker_id, sequence, event_id, timestamp, event_type, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.worker_id,
                event.sequence,
                event.event_id,
                event.timestamp,
                event.event_type,
                stable_json_dumps(event.payload, indent=None),
            ),
        )
        return event


class WorkerManager:
    """Runs simple sequential agents as independently addressable durable workers."""

    def __init__(self, runtime: AgentRuntime, *, max_workers: int = 4):
        self.runtime = runtime
        self.store = WorkerStore(runtime.config.sessions.root)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix="swaag-worker"
        )
        self._futures: dict[str, Future[None]] = {}
        self._control_transition_lock = threading.RLock()

    def create(
        self,
        objective: str,
        *,
        name: str | None = None,
        output_schema: dict[str, Any] | None = None,
        mechanical_fields: dict[str, str] | None = None,
        completion_mode: str = "natural",
        presentation_modes: list[str] | None = None,
    ) -> WorkerRecord:
        output_spec = prepare_caller_output_spec(output_schema, mechanical_fields)
        mode = str(completion_mode).strip()
        if mode not in WORKER_COMPLETION_MODES:
            raise ValueError(
                "completion_mode must be one of "
                f"{sorted(WORKER_COMPLETION_MODES)}"
            )
        if mode == "continuous" and output_spec is not None:
            raise ValueError(
                "continuous workers cannot have a terminal output_schema"
            )
        requested_presentations = sorted(set(presentation_modes or []))
        if set(requested_presentations) - WORKER_PRESENTATION_MODES:
            raise ValueError(
                "presentation_modes must contain only visual and/or audio"
            )
        if mode == "continuous" and requested_presentations:
            raise ValueError(
                "continuous workers cannot have terminal response presentations"
            )
        state = self.runtime.create_or_load_session()
        if name and name.strip():
            state = self.runtime.history.rename_session(state.session_id, name.strip())
        return self.store.create(
            state.session_id,
            objective,
            output_spec=output_spec,
            completion_mode=mode,
            presentation_modes=requested_presentations,
        )

    def start(self, worker_id: str) -> WorkerRecord:
        current = self.store.get(worker_id)
        if current.archived_at is not None:
            raise ValueError(f"Worker {worker_id} is archived")
        queued = self.store.transition(
            worker_id,
            "queued",
            expected={"created"},
            event_type="worker_queued",
        )
        self._submit(worker_id)
        return queued

    def resume(self, worker_id: str, *, message: str | None = None) -> WorkerRecord:
        with self._control_transition_lock:
            current = self.store.get(worker_id)
            if current.archived_at is not None:
                raise ValueError(f"Worker {worker_id} is archived")
            if current.status not in WORKER_RESUMABLE_STATES:
                raise ValueError(f"Worker {worker_id} cannot resume from {current.status}")
            if message is not None and message.strip():
                self._queue_message(current, message.strip(), source="worker_resume")
            queued = self.store.transition(
                worker_id,
                "queued",
                expected=WORKER_RESUMABLE_STATES,
                event_type="worker_resumed",
            )
        self._submit(worker_id)
        return queued

    def message(
        self,
        worker_id: str,
        message: str,
        *,
        source: str = "worker_control",
        resume_if_idle: bool = True,
    ) -> WorkerRecord:
        with self._control_transition_lock:
            current = self.store.get(worker_id)
            if current.archived_at is not None:
                raise ValueError(f"Worker {worker_id} is archived")
            if current.status == "cancellation_requested":
                raise ValueError(f"Worker {worker_id} is being canceled")
            self._queue_message(current, message, source=source)
            status_at_delivery = current.status
        if status_at_delivery == "working":
            preemption = self.runtime.preemption.request_preemption(
                current.session_id, message, source=source
            )
            if preemption is not None:
                timeout = max(
                    1.0,
                    float(self.runtime.config.model.timeout_seconds),
                    float(self.runtime.config.model.structured_timeout_seconds),
                )
                interrupted = self.runtime.preemption.wait_for_status(
                    preemption.preemption_id,
                    {"interrupted", "failed"},
                    timeout_seconds=timeout,
                )
                if interrupted.status == "failed":
                    raise RuntimeError(interrupted.reply or "worker redirect preemption failed")
                self.runtime.preemption.complete(
                    preemption.preemption_id,
                    target_changed=True,
                    reply="worker control queued",
                )
        elif resume_if_idle and status_at_delivery in WORKER_RESUMABLE_STATES:
            return self.resume(worker_id)
        latest = self.store.get(worker_id)
        if resume_if_idle and latest.status in WORKER_RESUMABLE_STATES:
            return self.resume(worker_id)
        return latest

    def cancel(self, worker_id: str, *, reason: str = "user requested cancellation") -> WorkerRecord:
        current = self.store.get(worker_id)
        if current.status == "canceled":
            return current
        if current.status in {"completed", "failed"} or current.archived_at is not None:
            raise ValueError(f"Worker {worker_id} cannot be canceled from {current.status}")
        requested = self.store.transition(
            worker_id,
            "cancellation_requested",
            expected={"created", "queued", "working", "input_required"},
            result=current.result,
            event_type="worker_cancellation_requested",
            event_payload={"reason": reason},
        )
        active = self.runtime.history.read_active_run(current.session_id)
        if active is None:
            return self.store.transition(
                worker_id,
                "canceled",
                expected={"cancellation_requested"},
                event_type="worker_canceled",
                event_payload={"reason": reason, "active_run": False},
            )
        run_id = str(active.get("run_id", ""))
        self.runtime.preemption.request_run_cancellation(
            current.session_id, run_id, reason=reason
        )
        return requested

    def archive(self, worker_id: str) -> WorkerRecord:
        current = self.store.get(worker_id)
        if current.status in WORKER_ACTIVE_STATES:
            raise ValueError(f"Worker {worker_id} cannot be archived while {current.status}")
        if current.archived_at is not None:
            return current
        archived = self.runtime.history.archive_session(current.session_id, remove_active=True)
        return self.store.mark_archived(worker_id, archive=archived)

    def add_attachment(
        self,
        worker_id: str,
        data: bytes,
        *,
        original_name: str,
        media_type: str = "",
        source: str = "task_api",
    ) -> AttachmentReference:
        current = self.store.get(worker_id)
        if current.archived_at is not None:
            raise ValueError(f"Worker {worker_id} is archived")
        if current.status in WORKER_ACTIVE_STATES:
            raise ValueError(
                f"Worker {worker_id} is {current.status}; attach raw inputs before starting or after it becomes idle"
            )
        reference = self.runtime.add_attachment(
            data,
            original_name=original_name,
            media_type=media_type,
            source=source,
            session_id=current.session_id,
        )
        self.store.append_event(
            worker_id,
            "worker_attachment_added",
            {
                "attachment_id": reference.attachment_id,
                "original_name": reference.original_name,
                "media_type": reference.media_type,
                "size_bytes": reference.size_bytes,
                "sha256": reference.sha256,
                "source": reference.source,
            },
        )
        return reference

    def attachments(self, worker_id: str) -> list[AttachmentReference]:
        record = self.store.get(worker_id)
        state = self.runtime.history.rebuild_from_history(
            record.session_id, write_projections=False
        )
        return list(state.attachments)

    def inspect(self, worker_id: str) -> dict[str, Any]:
        record = self.store.get(worker_id)
        self._sync_history_events(record)
        events = self.store.events(worker_id)
        output_spec = self._output_spec(record, events=events)
        active_run = self.runtime.history.read_active_run(record.session_id)
        execution_diagnostics = self._execution_diagnostics(
            record,
            events=events,
            active_run=active_run,
        )
        inference_requests = self.runtime.inference.list(
            session_id=record.session_id
        )
        state = self.runtime.history.rebuild_from_history(
            record.session_id, write_projections=False
        )
        return {
            **asdict(record),
            "active_run": active_run,
            "execution_diagnostics": execution_diagnostics,
            "inference_requests": [
                asdict(item) for item in inference_requests[-10:]
            ],
            "mechanical_status": self.runtime.session_status_payload(state),
            "semantic_status": self.runtime.latest_semantic_status_payload(state),
            "attachments": [
                {
                    key: value
                    for key, value in asdict(item).items()
                    if key != "storage_ref"
                }
                for item in state.attachments
            ],
            "caller_output_spec": (
                output_spec.payload() if output_spec is not None else None
            ),
            "structured_output": self._structured_output_from_events(events),
            "presentations": self._presentations_from_events(events),
            "latest_event_sequence": events[-1].sequence,
        }

    def _execution_diagnostics(
        self,
        record: WorkerRecord,
        *,
        events: list[WorkerEvent],
        active_run: dict[str, Any] | None,
    ) -> dict[str, Any]:
        transition = next(
            (
                event
                for event in reversed(events)
                if "to_status" in event.payload or event.event_type == "worker_created"
            ),
            None,
        )
        future = self._futures.get(record.worker_id)
        if future is None:
            local_run_state = "not_registered"
        elif future.cancelled():
            local_run_state = "cancelled"
        elif future.running():
            local_run_state = "running"
        elif future.done():
            local_run_state = "finished"
        else:
            local_run_state = "queued"
        active_operation = None
        if active_run is not None:
            active_operation = {
                "run_id": str(active_run.get("run_id", "")),
                "phase": str(active_run.get("phase", "unknown")),
                "detail": str(active_run.get("detail", "")),
                "active_kind": str(active_run.get("active_kind", "")),
                "active_id": str(active_run.get("active_id", "")),
                "started_at": str(active_run.get("started_at", "")),
                "updated_at": str(active_run.get("updated_at", "")),
                "heartbeat_at": str(active_run.get("heartbeat_at", "")),
                "heartbeat_age_seconds": _timestamp_age_seconds(
                    active_run.get("heartbeat_at")
                ),
                "pid": active_run.get("pid"),
                "pid_alive": _pid_is_alive(active_run.get("pid")),
            }
        return {
            "observed_at": utc_now_iso(),
            "last_transition": (
                None
                if transition is None
                else {
                    "sequence": transition.sequence,
                    "timestamp": transition.timestamp,
                    "event_type": transition.event_type,
                    "from_status": transition.payload.get("from_status"),
                    "to_status": transition.payload.get(
                        "to_status", transition.payload.get("status")
                    ),
                }
            ),
            "active_operation": active_operation,
            "local_supervisor": {
                "manager_pid": os.getpid(),
                "manager_process_alive": _pid_is_alive(os.getpid()),
                "run_state": local_run_state,
            },
        }

    def list(self, *, include_archived: bool = False) -> list[WorkerRecord]:
        return self.store.list(include_archived=include_archived)

    def events(self, worker_id: str, *, after_sequence: int = 0) -> list[WorkerEvent]:
        record = self.store.get(worker_id)
        self._sync_history_events(record)
        events = self.store.events(worker_id, after_sequence=after_sequence)
        return self._hydrate_history_events(record, events)

    def stream_snapshot(self, worker_id: str) -> tuple[WorkerRecord, int]:
        record = self.store.get(worker_id)
        self._sync_history_events(record)
        return self.store.snapshot_with_event_cursor(worker_id)

    def _sync_history_events(self, record: WorkerRecord) -> None:
        cursor = self.store.history_cursor(record.worker_id)
        source_events = self.runtime.history.iter_history(
            record.session_id,
            start_sequence=cursor + 1,
        )
        self.store.sync_history_references(
            record.worker_id,
            record.session_id,
            source_events,
        )

    def _hydrate_history_events(
        self,
        record: WorkerRecord,
        events: list[WorkerEvent],
    ) -> list[WorkerEvent]:
        references = {
            int(event.payload["history_sequence"]): event
            for event in events
            if event.event_type == "worker_history_event"
            and isinstance(event.payload.get("history_sequence"), int)
        }
        if not references:
            return events
        source_by_sequence = {
            event.sequence: event
            for event in self.runtime.history.iter_history(
                record.session_id,
                start_sequence=min(references),
                end_sequence=max(references),
            )
        }
        output: list[WorkerEvent] = []
        for event in events:
            source_sequence = event.payload.get("history_sequence")
            if event.event_type != "worker_history_event" or not isinstance(
                source_sequence, int
            ):
                output.append(event)
                continue
            source = source_by_sequence.get(source_sequence)
            payload = dict(event.payload)
            if source is None:
                payload["canonical_event_unavailable"] = True
            else:
                expected = (
                    payload.get("history_event_id"),
                    payload.get("history_event_hash"),
                    payload.get("history_event_type"),
                )
                actual = (source.id, source.hash, source.event_type)
                if actual != expected:
                    raise RuntimeError(
                        f"Worker {record.worker_id} history reference mismatch at "
                        f"sequence {source_sequence}"
                    )
                payload["canonical_event"] = asdict(source)
            output.append(
                WorkerEvent(
                    event_id=event.event_id,
                    worker_id=event.worker_id,
                    sequence=event.sequence,
                    timestamp=event.timestamp,
                    event_type=event.event_type,
                    payload=payload,
                )
            )
        return output

    @staticmethod
    def event_from_payload(payload: dict[str, Any]) -> WorkerEvent:
        return WorkerEvent(**payload)

    def structured_output(self, worker_id: str) -> dict[str, Any] | None:
        return self._structured_output_from_events(self.store.events(worker_id))

    def presentations(self, worker_id: str) -> dict[str, Any] | None:
        return self._presentations_from_events(self.store.events(worker_id))

    @staticmethod
    def _structured_output_from_events(
        events: list[WorkerEvent],
    ) -> dict[str, Any] | None:
        for event in reversed(events):
            output = event.payload.get("structured_output")
            if isinstance(output, dict):
                return dict(output)
        return None

    @staticmethod
    def _presentations_from_events(
        events: list[WorkerEvent],
    ) -> dict[str, Any] | None:
        for event in reversed(events):
            presentations = event.payload.get("presentations")
            if isinstance(presentations, dict):
                return dict(presentations)
        return None

    def wait(
        self,
        worker_id: str,
        *,
        timeout_seconds: float | None = 30.0,
    ) -> WorkerRecord:
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + max(0.0, float(timeout_seconds))
        )
        while True:
            item = self.store.get(worker_id)
            if item.status in WORKER_TERMINAL_STATES or item.status == "input_required":
                return item
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for worker {worker_id}; status={item.status}")
            time.sleep(0.02)

    def reconcile_orphans(self) -> list[WorkerRecord]:
        reconciled: list[WorkerRecord] = []
        for item in self.store.list(statuses=WORKER_ACTIVE_STATES, include_archived=True):
            active = self.runtime.history.read_active_run(item.session_id)
            if active is not None and _pid_is_alive(active.get("pid")):
                continue
            if item.status == "cancellation_requested":
                status, event_type, error = "canceled", "worker_canceled", None
            else:
                status, event_type = "failed", "worker_orphaned"
                error = "Worker process ended before its durable run reached a terminal state"
            reconciled.append(
                self.store.transition(
                    item.worker_id,
                    status,
                    expected={item.status},
                    error=error,
                    event_type=event_type,
                    event_payload={"recovered_orphan": True},
                )
            )
        return reconciled

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _submit(self, worker_id: str) -> None:
        previous = self._futures.get(worker_id)
        if previous is not None and not previous.done():
            raise RuntimeError(f"Worker {worker_id} already has a local run")
        self._futures[worker_id] = self._executor.submit(self._run_worker, worker_id)

    def _continue_for_pending_controls(
        self,
        worker_id: str,
        working: WorkerRecord,
        *,
        phase: str,
        provisional_result: str,
    ) -> WorkerRecord | None:
        """Atomically keep a local worker active when a delivered control is pending."""
        with self._control_transition_lock:
            pending = self.runtime.history.list_pending_control_messages(
                working.session_id
            )
            if not pending:
                return None
            self._sync_history_events(working)
            return self.store.transition(
                worker_id,
                "working",
                expected={"working"},
                result=provisional_result,
                increment_run_count=True,
                event_type="worker_control_continuation_started",
                event_payload={
                    "phase": phase,
                    "provisional": True,
                    "pending_control_ids": [
                        str(item.get("control_id", "")) for item in pending
                    ],
                },
            )

    def _run_worker(self, worker_id: str) -> None:
        queued = self.store.get(worker_id)
        if queued.status == "cancellation_requested":
            self.store.transition(
                worker_id,
                "canceled",
                expected={"cancellation_requested"},
                event_type="worker_canceled",
            )
            return
        if queued.status != "queued":
            return
        working = self.store.transition(
            worker_id,
            "working",
            expected={"queued"},
            increment_run_count=True,
            event_type="worker_started",
        )
        state = self.runtime.history.rebuild_from_history(
            working.session_id, write_projections=False
        )
        output_spec = self._output_spec(working)
        try:
            while True:
                latest = self.store.get(worker_id)
                if latest.status == "cancellation_requested":
                    self.store.transition(
                        worker_id,
                        "canceled",
                        expected={"cancellation_requested"},
                        result=latest.result,
                        event_type="worker_canceled",
                        event_payload={"between_continuous_cycles": True},
                    )
                    return
                if latest.status != "working":
                    return
                working = latest
                first_sequence = state.event_count + 1
                if working.run_count == 1 and not state.messages:
                    result = self.runtime.run_turn_in_session(state, working.objective)
                else:
                    result = self.runtime.resume_turn_in_session(
                        state, working.objective
                    )
                latest = self.store.get(worker_id)
                if latest.status == "cancellation_requested":
                    self._sync_history_events(working)
                    self.store.transition(
                        worker_id,
                        "canceled",
                        expected={"cancellation_requested"},
                        result=result.assistant_text,
                        event_type="worker_canceled",
                        event_payload={"completion_raced_cancellation": True},
                    )
                    return
                if latest.status != "working":
                    return
                continued = self._continue_for_pending_controls(
                    worker_id,
                    working,
                    phase="worker_result",
                    provisional_result=result.assistant_text,
                )
                if continued is not None:
                    working = continued
                    continue
                blocking = any(
                    event.event_type == "agent_question"
                    and event.payload.get("criticality") == "blocking"
                    for event in self.runtime.history.iter_history(
                        working.session_id, start_sequence=first_sequence
                    )
                )
                if blocking:
                    self._sync_history_events(working)
                    self.store.transition(
                        worker_id,
                        "input_required",
                        expected={"working"},
                        result=result.assistant_text,
                        event_type="worker_input_required",
                    )
                    return

                if working.completion_mode == "continuous":
                    self._sync_history_events(working)
                    continued = self.store.transition(
                        worker_id,
                        "working",
                        expected={"working"},
                        result=result.assistant_text,
                        event_type="worker_iteration_completed",
                        event_payload={
                            "completion_mode": "continuous",
                            "provisional": True,
                        },
                    )
                    self._queue_message(
                        continued,
                        _CONTINUOUS_WORKER_CONTROL,
                        source="worker_continuous",
                    )
                    latest = self.store.get(worker_id)
                    if latest.status != "working":
                        continue
                    working = self.store.transition(
                        worker_id,
                        "working",
                        expected={"working"},
                        result=continued.result,
                        increment_run_count=True,
                        event_type="worker_continuation_started",
                        event_payload={"completion_mode": "continuous"},
                    )
                    continue

                presentations = None
                if working.presentation_modes:
                    try:
                        presentations = self.runtime.generate_response_presentations(
                            state,
                            original_request=working.objective,
                            assistant_message=result.assistant_text,
                            modes=working.presentation_modes,
                        )
                    except ModelCallStateChanged:
                        continued = self._continue_for_pending_controls(
                            worker_id,
                            working,
                            phase="response_presentation",
                            provisional_result=result.assistant_text,
                        )
                        if continued is None:
                            raise
                        working = continued
                        continue
                    latest = self.store.get(worker_id)
                    if latest.status == "cancellation_requested":
                        self._sync_history_events(working)
                        self.store.transition(
                            worker_id,
                            "canceled",
                            expected={"cancellation_requested"},
                            result=result.assistant_text,
                            event_type="worker_canceled",
                            event_payload={
                                "presentation_raced_cancellation": True
                            },
                        )
                        return
                    continued = self._continue_for_pending_controls(
                        worker_id,
                        working,
                        phase="response_presentation",
                        provisional_result=result.assistant_text,
                    )
                    if continued is not None:
                        working = continued
                        continue

                structured_output = None
                if output_spec is not None:
                    try:
                        semantic_output = self.runtime.generate_caller_structured_output(
                            state,
                            original_request=working.objective,
                            assistant_message=result.assistant_text,
                            tool_results=result.tool_results,
                            semantic_schema=output_spec.semantic_schema,
                        )
                    except ModelCallStateChanged:
                        continued = self._continue_for_pending_controls(
                            worker_id,
                            working,
                            phase="caller_structured_output",
                            provisional_result=result.assistant_text,
                        )
                        if continued is None:
                            raise
                        working = continued
                        continue
                    structured_output = merge_caller_output(
                        output_spec,
                        semantic_output,
                        {
                            "worker_id": working.worker_id,
                            "session_id": working.session_id,
                            "objective": working.objective,
                            "status": "completed",
                            "created_at": working.created_at,
                            "started_at": working.started_at or "",
                            "run_count": working.run_count,
                        },
                    )
                    latest = self.store.get(worker_id)
                    if latest.status == "cancellation_requested":
                        self._sync_history_events(working)
                        self.store.transition(
                            worker_id,
                            "canceled",
                            expected={"cancellation_requested"},
                            result=result.assistant_text,
                            event_type="worker_canceled",
                            event_payload={
                                "structured_output_raced_cancellation": True
                            },
                        )
                        return
                    continued = self._continue_for_pending_controls(
                        worker_id,
                        working,
                        phase="caller_structured_output",
                        provisional_result=result.assistant_text,
                    )
                    if continued is not None:
                        working = continued
                        continue
                with self._control_transition_lock:
                    continued = self._continue_for_pending_controls(
                        worker_id,
                        working,
                        phase="terminal_commit",
                        provisional_result=result.assistant_text,
                    )
                    if continued is None:
                        self._sync_history_events(working)
                        terminal_payload = {}
                        if structured_output is not None:
                            terminal_payload["structured_output"] = structured_output
                        if presentations is not None:
                            terminal_payload["presentations"] = presentations
                        self.store.transition(
                            worker_id,
                            "completed",
                            expected={"working"},
                            result=result.assistant_text,
                            event_type="worker_completed",
                            event_payload=terminal_payload or None,
                        )
                        return
                working = continued
                continue
        except RunCancellationRequested as exc:
            error = str(exc)
            try:
                self._sync_history_events(working)
            except Exception as sync_exc:
                error += f"; history event sync failed: {type(sync_exc).__name__}: {sync_exc}"
            latest = self.store.get(worker_id)
            if latest.status == "canceled":
                return
            self.store.transition(
                worker_id,
                "canceled",
                expected={"working", "cancellation_requested"},
                result=latest.result,
                error=error,
                event_type="worker_canceled",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                self._sync_history_events(working)
            except Exception as sync_exc:
                error += f"; history event sync failed: {type(sync_exc).__name__}: {sync_exc}"
            latest = self.store.get(worker_id)
            if latest.status == "canceled":
                return
            if latest.status == "cancellation_requested":
                self.store.transition(
                    worker_id,
                    "canceled",
                    expected={"cancellation_requested"},
                    result=latest.result,
                    error=error,
                    event_type="worker_canceled",
                    event_payload={"failure_raced_cancellation": True},
                )
                return
            self.store.transition(
                worker_id,
                "failed",
                expected={"working"},
                result=latest.result,
                error=error,
                event_type="worker_failed",
            )

    def _queue_message(self, current: WorkerRecord, message: str, *, source: str) -> None:
        text = message.strip()
        if not text:
            raise ValueError("worker message must not be empty")
        control = self.runtime.history.enqueue_control_message(
            current.session_id, text, source=source
        )
        self.store.append_event(
            current.worker_id,
            "worker_message_queued",
            {
                "control_id": control["control_id"],
                "message": text,
                "source": source,
            },
        )

    def _output_spec(
        self,
        record: WorkerRecord,
        *,
        events: list[WorkerEvent] | None = None,
    ) -> CallerOutputSpec | None:
        events = self.store.events(record.worker_id) if events is None else events
        if not events:
            return None
        payload = events[0].payload.get("caller_output_spec")
        if not isinstance(payload, dict):
            return None
        schema = payload.get("schema")
        mechanical_fields = payload.get("mechanical_fields")
        if not isinstance(schema, dict) or not isinstance(mechanical_fields, dict):
            raise RuntimeError(f"Worker {record.worker_id} has an invalid caller output spec")
        return prepare_caller_output_spec(schema, mechanical_fields)


def _pid_is_alive(value: Any) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _timestamp_age_seconds(value: Any) -> float | None:
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return round(
        max(0.0, (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds()),
        3,
    )
