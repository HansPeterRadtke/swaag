from __future__ import annotations

import os
import math
import sqlite3
import time
from datetime import datetime, timezone
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
    (
        "ALTER TABLE inference_requests ADD COLUMN fair_weight REAL NOT NULL DEFAULT 1.0",
        """
        CREATE TABLE IF NOT EXISTS inference_fair_state (
            backend_key TEXT NOT NULL,
            source TEXT NOT NULL,
            virtual_service REAL NOT NULL DEFAULT 0.0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(backend_key, source)
        )
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
    fair_weight: float = 1.0


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
        max_running_seconds: float | None = None,
    ):
        self.path = Path(root).expanduser() / "inference_requests.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend_key = str(backend_key)
        self.capacity_resolver = capacity_resolver
        self.poll_seconds = max(0.005, float(poll_seconds))
        self.aging_seconds_per_priority = max(
            0.001, float(aging_seconds_per_priority)
        )
        self.max_running_seconds = (
            None
            if max_running_seconds is None
            else max(1.0, float(max_running_seconds))
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
        fair_weight: float = 1.0,
    ) -> InferenceRequest:
        if (
            isinstance(fair_weight, bool)
            or not isinstance(fair_weight, (int, float))
            or not math.isfinite(fair_weight)
            or fair_weight <= 0
        ):
            raise ValueError("inference fair_weight must be finite and positive")
        fair_weight = float(fair_weight)
        now = utc_now_iso()
        queued_epoch = time.time()
        request_id = new_id("inference_request")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inference_requests(
                    request_id, backend_key, session_id, run_id, call_id,
                    call_kind, source, priority, status, owner_pid,
                    queued_at, queued_epoch, updated_at, fair_weight
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
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
                    fair_weight,
                ),
            )
            existing_fair = connection.execute(
                "SELECT 1 FROM inference_fair_state WHERE backend_key=? AND source=?",
                (self.backend_key, str(source)),
            ).fetchone()
            if existing_fair is None:
                minimum = connection.execute(
                    "SELECT MIN(virtual_service) FROM inference_fair_state WHERE backend_key=?",
                    (self.backend_key,),
                ).fetchone()[0]
                connection.execute(
                    "INSERT INTO inference_fair_state(backend_key,source,virtual_service,updated_at) VALUES(?,?,?,?)",
                    (
                        self.backend_key,
                        str(source),
                        0.0 if minimum is None else float(minimum),
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

    def _fair_candidate(
        self, connection: sqlite3.Connection, *, now_epoch: float
    ) -> str | None:
        rows = connection.execute(
            """
            SELECT r.request_id, r.source, r.priority, r.queued_epoch,
                   r.fair_weight, COALESCE(f.virtual_service, 0.0) AS virtual_service
            FROM inference_requests AS r
            LEFT JOIN inference_fair_state AS f
              ON f.backend_key=r.backend_key AND f.source=r.source
            WHERE r.backend_key=? AND r.status='queued'
            """,
            (self.backend_key,),
        ).fetchall()
        if not rows:
            return None
        effective = [
            (
                int(row["priority"])
                + int(
                    max(0.0, now_epoch - float(row["queued_epoch"]))
                    / self.aging_seconds_per_priority
                ),
                row,
            )
            for row in rows
        ]
        highest = max(score for score, _row in effective)
        eligible = [row for score, row in effective if score == highest]
        chosen = min(
            eligible,
            key=lambda row: (
                float(row["virtual_service"]),
                float(row["queued_epoch"]),
                str(row["request_id"]),
            ),
        )
        return str(chosen["request_id"])

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
                candidate_id = self._fair_candidate(
                    connection, now_epoch=now_epoch
                )
                if active_count < capacity and candidate_id == request_id:
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
                    connection.execute(
                        """
                        UPDATE inference_fair_state
                        SET virtual_service=virtual_service + ?, updated_at=?
                        WHERE backend_key=? AND source=?
                        """,
                        (
                            1.0 / max(0.000001, float(item.fair_weight)),
                            now,
                            self.backend_key,
                            item.source,
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

    def suspend(self, request_id: str, *, reason: str) -> InferenceRequest:
        """Release backend capacity while a preempted call is not yet eligible to replay."""
        return self._transition_from_running(
            request_id,
            "suspended",
            error=reason,
        )

    def resume(self, request_id: str, *, reason: str | None = None) -> InferenceRequest:
        """Make a suspended preempted call runnable again once replay is actually ready."""
        return self._transition(
            request_id,
            "queued",
            expected={"suspended"},
            error=reason,
            reset_queue=True,
        )

    def complete(self, request_id: str) -> InferenceRequest:
        return self._transition_from_running(request_id, "completed")

    def fail(self, request_id: str, *, error: str) -> InferenceRequest:
        return self._transition(
            request_id,
            "failed",
            expected={"queued", "running", "suspended"},
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
            expected={"queued", "running", "suspended"},
            error=reason,
            cancellation_requested_at=requested_at or utc_now_iso(),
        )

    def supersede(self, request_id: str, *, reason: str) -> InferenceRequest:
        return self._transition(
            request_id,
            "superseded",
            expected={"queued", "running", "suspended"},
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

    def touch_running(self, request_id: str) -> InferenceRequest | None:
        """Refresh durable liveness for a running request without changing its state."""
        now = utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE inference_requests
                SET updated_at=?
                WHERE request_id=? AND status='running'
                """,
                (now, request_id),
            )
        if cursor.rowcount != 1:
            return self.get(request_id)
        return self.get(request_id)

    def reconcile_orphans(self) -> list[InferenceRequest]:
        reconciled: list[InferenceRequest] = []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM inference_requests
                WHERE backend_key=? AND status IN ('queued', 'running', 'suspended')
                """,
                (self.backend_key,),
            ).fetchall()
        now_epoch = time.time()
        for row in rows:
            item = self._record(row)
            if item is None:
                continue
            owner_alive = _pid_is_alive(item.owner_pid)
            heartbeat_epoch = _iso_epoch(item.updated_at)
            stale_running = (
                item.status == "running"
                and self.max_running_seconds is not None
                and heartbeat_epoch is not None
                and now_epoch - float(heartbeat_epoch) > self.max_running_seconds
            )
            if owner_alive and not stale_running:
                continue
            error = (
                f"inference liveness heartbeat stale for more than {self.max_running_seconds:.1f}s"
                if stale_running
                else "inference owner process ended before terminal state"
            )
            try:
                reconciled.append(self.fail(item.request_id, error=error))
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
                        started_at=NULL, completed_at=NULL, backend_capacity=NULL,
                        capacity_source=NULL, queue_wait_seconds=NULL, error=?
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


def _iso_epoch(value: str) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None
