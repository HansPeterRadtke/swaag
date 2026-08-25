from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swaag.utils import new_id, sha256_text, stable_json_dumps, utc_now_iso


class ModelCallPreempted(RuntimeError):
    """Raised when a live model request is intentionally cancelled for communication."""


class ModelCallStateChanged(RuntimeError):
    """Raised when communication changed target state, so a stale request must not replay."""


@dataclass(slots=True, frozen=True)
class ActiveModelCall:
    session_id: str
    call_id: str
    kind: str
    request_json: str
    request_sha256: str
    started_at: str

    @property
    def request(self) -> dict[str, Any]:
        value = json.loads(self.request_json)
        if not isinstance(value, dict):
            raise ValueError("active model request must be a JSON object")
        return value


@dataclass(slots=True, frozen=True)
class PreemptionRequest:
    preemption_id: str
    session_id: str
    call_id: str
    message: str
    source: str
    status: str
    target_changed: bool
    reply: str | None
    created_at: str
    updated_at: str


class ModelPreemptionCoordinator:
    """Durable cross-process coordinator for main-call preemption and exact replay."""

    def __init__(self, root: Path):
        self.path = Path(root).expanduser() / "model_preemption.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS active_calls (
                    session_id TEXT PRIMARY KEY,
                    call_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    started_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS preemptions (
                    preemption_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    target_changed INTEGER NOT NULL DEFAULT 0,
                    reply TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS preemptions_call_status
                    ON preemptions(call_id, status, created_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _request_json(request: dict[str, Any]) -> str:
        return stable_json_dumps(request, indent=None)

    def register_active(self, session_id: str, call_id: str, kind: str, request: dict[str, Any]) -> ActiveModelCall:
        request_json = self._request_json(request)
        row = ActiveModelCall(
            session_id=session_id,
            call_id=call_id,
            kind=kind,
            request_json=request_json,
            request_sha256=sha256_text(request_json),
            started_at=utc_now_iso(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO active_calls(session_id,call_id,kind,request_json,request_sha256,started_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    call_id=excluded.call_id,
                    kind=excluded.kind,
                    request_json=excluded.request_json,
                    request_sha256=excluded.request_sha256,
                    started_at=excluded.started_at
                """,
                (row.session_id, row.call_id, row.kind, row.request_json, row.request_sha256, row.started_at),
            )
        return row

    def clear_active(self, session_id: str, call_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM active_calls WHERE session_id=? AND call_id=?", (session_id, call_id))

    def active_call(self, session_id: str) -> ActiveModelCall | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM active_calls WHERE session_id=?", (session_id,)).fetchone()
        return ActiveModelCall(**dict(row)) if row else None

    def request_preemption(self, session_id: str, message: str, *, source: str = "communication") -> PreemptionRequest | None:
        active = self.active_call(session_id)
        if active is None:
            return None
        now = utc_now_iso()
        preemption_id = new_id("preemption")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO preemptions(
                    preemption_id,session_id,call_id,message,source,status,target_changed,reply,created_at,updated_at
                ) VALUES(?,?,?,?,?,'requested',0,NULL,?,?)
                """,
                (preemption_id, session_id, active.call_id, message, source, now, now),
            )
        return self.get(preemption_id)

    def pending_for_call(self, session_id: str, call_id: str) -> PreemptionRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM preemptions
                WHERE session_id=? AND call_id=? AND status='requested'
                ORDER BY created_at, preemption_id LIMIT 1
                """,
                (session_id, call_id),
            ).fetchone()
        return self._row(row)

    def get(self, preemption_id: str) -> PreemptionRequest | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM preemptions WHERE preemption_id=?", (preemption_id,)).fetchone()
        return self._row(row)

    @staticmethod
    def _row(row: sqlite3.Row | None) -> PreemptionRequest | None:
        if row is None:
            return None
        payload = dict(row)
        payload["target_changed"] = bool(payload["target_changed"])
        return PreemptionRequest(**payload)

    def _set(self, preemption_id: str, status: str, *, target_changed: bool | None = None, reply: str | None = None) -> None:
        with self._connect() as connection:
            current = connection.execute("SELECT target_changed,reply FROM preemptions WHERE preemption_id=?", (preemption_id,)).fetchone()
            if current is None:
                raise FileNotFoundError(f"Unknown preemption id: {preemption_id}")
            changed = bool(current["target_changed"]) if target_changed is None else bool(target_changed)
            effective_reply = current["reply"] if reply is None else reply
            connection.execute(
                "UPDATE preemptions SET status=?,target_changed=?,reply=?,updated_at=? WHERE preemption_id=?",
                (status, int(changed), effective_reply, utc_now_iso(), preemption_id),
            )

    def mark_interrupted(self, preemption_id: str) -> None:
        self._set(preemption_id, "interrupted")

    def mark_assistant_running(self, preemption_id: str) -> None:
        self._set(preemption_id, "assistant_running")

    def complete(self, preemption_id: str, *, target_changed: bool, reply: str | None = None) -> None:
        self._set(preemption_id, "completed", target_changed=target_changed, reply=reply)

    def fail(self, preemption_id: str, reply: str) -> None:
        self._set(preemption_id, "failed", reply=reply)

    def wait_for_status(self, preemption_id: str, statuses: set[str], *, timeout_seconds: float, poll_seconds: float = 0.02) -> PreemptionRequest:
        deadline = time.monotonic() + max(0.01, timeout_seconds)
        while True:
            item = self.get(preemption_id)
            if item is None:
                raise FileNotFoundError(f"Unknown preemption id: {preemption_id}")
            if item.status in statuses:
                return item
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for preemption {preemption_id} status in {sorted(statuses)}; current={item.status}")
            time.sleep(max(0.005, poll_seconds))
