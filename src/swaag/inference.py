from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from swaag.preemption import ModelCallPreempted
from swaag.sqlite_schema import apply_sqlite_migrations
from swaag.utils import new_id, utc_now_iso


INFERENCE_TERMINAL_STATES = frozenset(
    {"completed", "failed", "cancelled", "superseded"}
)
_INFERENCE_STORE_MIGRATIONS = (
    (
        """
        CREATE TABLE IF NOT EXISTS inference_requests (
            request_id TEXT PRIMARY KEY,
            backend_key TEXT NOT NULL,
            session_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            call_id TEXT NOT NULL UNIQUE,
            call_kind TEXT NOT NULL,
            source TEXT NOT NULL,
            priority INTEGER NOT NULL,
            status TEXT NOT NULL,
            owner_pid INTEGER NOT NULL,
            queued_at TEXT NOT NULL,
            queued_epoch REAL NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            backend_capacity INTEGER,
            capacity_source TEXT,
            queue_wait_seconds REAL,
            cancellation_requested_at TEXT,
            error TEXT
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS inference_requests_backend_status
        ON inference_requests(backend_key, status, priority, queued_epoch)
        """,
        """
        CREATE INDEX IF NOT EXISTS inference_requests_session
        ON inference_requests(session_id, queued_epoch, request_id)
        """,
    ),
)


@dataclass(slots=True, frozen=True)
class InferenceRequest:
    request_id: str
    backend_key: str
    session_id: str
    run_id: str
    call_id: str
    call_kind: str
    source: str
    priority: int
    status: str
    owner_pid: int
    queued_at: str
    queued_epoch: float
    started_at: str | None
    completed_at: str | None
    updated_at: str
    attempt_count: int
    backend_capacity: int | None
    capacity_source: str | None
    queue_wait_seconds: float | None
    cancellation_requested_at: str | None
    error: str | None


class InferenceRequestCoordinator:
    """Durable, backend-neutral admission and lifecycle for model requests."""

    def __init__(
        self,
        root: Path,
        *,
        backend_key: str,
        capacity_resolver: Callable[[], tuple[int, str]],
        poll_seconds: float = 0.02,
        aging_seconds_per_priority: float = 1.0,
    ):
        self.path = Path(root).expanduser() / "inference_requests.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend_key = str(backend_key)
        self.capacity_resolver = capacity_resolver
        self.poll_seconds = max(0.005, float(poll_seconds))
        self.aging_seconds_per_priority = max(
            0.001, float(aging_seconds_per_priority)
        )
        self._capacity: tuple[int, str] | None = None
        with self._connect() as connection:
            apply_sqlite_migrations(
                connection,
                store_name="inference request store",
                migrations=_INFERENCE_STORE_MIGRATIONS,
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _record(row: sqlite3.Row | None) -> InferenceRequest | None:
        return None if row is None else InferenceRequest(**dict(row))

    def enqueue(
        self,
        *,
        session_id: str,
        run_id: str,
        call_id: str,
        call_kind: str,
        priority: int,
        source: str,
    ) -> InferenceRequest:
        now = utc_now_iso()
        queued_epoch = time.time()
        request_id = new_id("inference_request")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inference_requests(
                    request_id, backend_key, session_id, run_id, call_id,
                    call_kind, source, priority, status, owner_pid,
                    queued_at, queued_epoch, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    request_id,
                    self.backend_key,
                    session_id,
                    run_id,
                    call_id,
                    call_kind,
                    source,
                    int(priority),
                    os.getpid(),
                    now,
                    queued_epoch,
                    now,
                ),
            )
        item = self.get(request_id)
        if item is None:
            raise RuntimeError("failed to persist inference request")
        return item

    def get(self, request_id: str) -> InferenceRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM inference_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
        return self._record(row)

    def by_call_id(self, call_id: str) -> InferenceRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM inference_requests WHERE call_id=?", (call_id,)
            ).fetchone()
        return self._record(row)

    def list(
        self,
        *,
        statuses: Iterable[str] | None = None,
        session_id: str | None = None,
    ) -> list[InferenceRequest]:
        clauses: list[str] = []
        params: list[Any] = []
        values = sorted({str(item) for item in statuses or () if str(item)})
        if values:
            clauses.append("status IN (" + ",".join("?" for _ in values) + ")")
            params.extend(values)
        if session_id is not None:
            clauses.append("session_id=?")
            params.append(session_id)
        sql = "SELECT * FROM inference_requests"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY queued_epoch, request_id"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [item for row in rows if (item := self._record(row)) is not None]

    def acquire(
        self,
        request_id: str,
        *,
        cancel_check: Callable[[], bool] | None = None,
        timeout_seconds: float | None = None,
    ) -> InferenceRequest:
        capacity, capacity_source = self._resolved_capacity()
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + max(0.0, float(timeout_seconds))
        )
        while True:
            if cancel_check is not None and cancel_check():
                raise ModelCallPreempted("model call preempted while queued")
            self.reconcile_orphans()
            now_epoch = time.time()
            now = utc_now_iso()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM inference_requests WHERE request_id=?",
                    (request_id,),
                ).fetchone()
                item = self._record(row)
                if item is None:
                    raise FileNotFoundError(f"Unknown inference request: {request_id}")
                if item.status == "running":
                    connection.commit()
                    return item
                if item.status != "queued":
                    raise RuntimeError(
                        f"Inference request {request_id} is {item.status}; expected queued"
                    )
                active_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM inference_requests
                        WHERE backend_key=? AND status='running'
                        """,
                        (self.backend_key,),
                    ).fetchone()[0]
                )
                candidate = connection.execute(
                    """
                    SELECT request_id FROM inference_requests
                    WHERE backend_key=? AND status='queued'
                    ORDER BY
                        priority + CAST((? - queued_epoch) / ? AS INTEGER) DESC,
                        queued_epoch,
                        request_id
                    LIMIT 1
                    """,
                    (
                        self.backend_key,
                        now_epoch,
                        self.aging_seconds_per_priority,
                    ),
                ).fetchone()
                if active_count < capacity and candidate is not None and candidate[0] == request_id:
                    queue_wait = max(0.0, now_epoch - float(item.queued_epoch))
                    connection.execute(
                        """
                        UPDATE inference_requests SET
                            status='running', started_at=COALESCE(started_at, ?),
                            updated_at=?, attempt_count=attempt_count+1,
                            backend_capacity=?, capacity_source=?, queue_wait_seconds=?
                        WHERE request_id=? AND status='queued'
                        """,
                        (
                            now,
                            now,
                            capacity,
                            capacity_source,
                            queue_wait,
                            request_id,
                        ),
                    )
                    connection.commit()
                    acquired = self.get(request_id)
                    if acquired is None:
                        raise RuntimeError("acquired inference request disappeared")
                    return acquired
                connection.commit()
            if deadline is not None and time.monotonic() >= deadline:
                self.fail(request_id, error="inference queue admission timed out")
                raise TimeoutError(
                    f"Timed out waiting for inference request {request_id} admission"
                )
            time.sleep(self.poll_seconds)

    def requeue(self, request_id: str, *, reason: str) -> InferenceRequest:
        return self._transition_from_running(
            request_id,
            "queued",
            error=reason,
            reset_queue=True,
        )

    def complete(self, request_id: str) -> InferenceRequest:
        return self._transition_from_running(request_id, "completed")

    def fail(self, request_id: str, *, error: str) -> InferenceRequest:
        return self._transition(
            request_id,
            "failed",
            expected={"queued", "running"},
            error=error,
        )

    def cancel(
        self,
        request_id: str,
        *,
        reason: str,
        requested_at: str | None = None,
    ) -> InferenceRequest:
        return self._transition(
            request_id,
            "cancelled",
            expected={"queued", "running"},
            error=reason,
            cancellation_requested_at=requested_at or utc_now_iso(),
        )

    def supersede(self, request_id: str, *, reason: str) -> InferenceRequest:
        return self._transition(
            request_id,
            "superseded",
            expected={"queued", "running"},
            error=reason,
        )

    def queue_depth(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM inference_requests
                    WHERE backend_key=? AND status='queued'
                    """,
                    (self.backend_key,),
                ).fetchone()[0]
            )

    def active_count(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM inference_requests
                    WHERE backend_key=? AND status='running'
                    """,
                    (self.backend_key,),
                ).fetchone()[0]
            )

    def reconcile_orphans(self) -> list[InferenceRequest]:
        reconciled: list[InferenceRequest] = []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM inference_requests
                WHERE backend_key=? AND status IN ('queued', 'running')
                """,
                (self.backend_key,),
            ).fetchall()
        for row in rows:
            item = self._record(row)
            if item is None or _pid_is_alive(item.owner_pid):
                continue
            try:
                reconciled.append(
                    self.fail(
                        item.request_id,
                        error="inference owner process ended before terminal state",
                    )
                )
            except RuntimeError:
                continue
        return reconciled

    def _resolved_capacity(self) -> tuple[int, str]:
        if self._capacity is None:
            value, source = self.capacity_resolver()
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"invalid inference backend capacity: {value!r}")
            self._capacity = int(value), str(source)
        return self._capacity

    def _transition_from_running(
        self,
        request_id: str,
        status: str,
        *,
        error: str | None = None,
        reset_queue: bool = False,
    ) -> InferenceRequest:
        return self._transition(
            request_id,
            status,
            expected={"running"},
            error=error,
            reset_queue=reset_queue,
        )

    def _transition(
        self,
        request_id: str,
        status: str,
        *,
        expected: set[str],
        error: str | None = None,
        reset_queue: bool = False,
        cancellation_requested_at: str | None = None,
    ) -> InferenceRequest:
        now = utc_now_iso()
        queued_epoch = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM inference_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise FileNotFoundError(f"Unknown inference request: {request_id}")
            current = str(row["status"])
            if current == status or current in INFERENCE_TERMINAL_STATES:
                connection.commit()
                item = self.get(request_id)
                if item is None:
                    raise RuntimeError("inference request disappeared")
                return item
            if current not in expected:
                raise RuntimeError(
                    f"Inference request {request_id} is {current}; expected {sorted(expected)}"
                )
            completed_at = now if status in INFERENCE_TERMINAL_STATES else None
            if reset_queue:
                connection.execute(
                    """
                    UPDATE inference_requests SET
                        status=?, queued_at=?, queued_epoch=?, updated_at=?,
                        completed_at=NULL, error=?
                    WHERE request_id=?
                    """,
                    (status, now, queued_epoch, now, error, request_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE inference_requests SET
                        status=?, updated_at=?, completed_at=?, error=?,
                        cancellation_requested_at=COALESCE(?, cancellation_requested_at)
                    WHERE request_id=?
                    """,
                    (
                        status,
                        now,
                        completed_at,
                        error,
                        cancellation_requested_at,
                        request_id,
                    ),
                )
            connection.commit()
        item = self.get(request_id)
        if item is None:
            raise RuntimeError("inference request disappeared")
        return item


def _pid_is_alive(value: Any) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
