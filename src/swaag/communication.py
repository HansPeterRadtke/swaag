from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import hashlib
import ipaddress
import json
import signal
import sqlite3
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import jsonpatch

from swaag.config import AgentConfig
from swaag.delegated_tools import DelegatedToolCall, DelegatedToolResultInput
from swaag.heartbeat import systemd_notify, watchdog_interval_seconds
from swaag.mcp import McpAdapter, McpHttpResponse, McpHttpSubscription
from swaag.protocol_adapters import (
    A2AContentTypeNotSupportedError,
    A2AProtocolError,
    A2AProjectionAdapter,
    A2ATaskNotCancelableError,
    A2AUnsupportedOperationError,
    AgUiProjectionAdapter,
    AgUiRunInput,
    OpenWebUiProjectionAdapter,
)
from swaag.runtime import AgentRuntime
from swaag.shared_state import (
    SharedStateChannel,
    SharedStateConflictError,
    SharedStateSnapshot,
    shared_state_event_payload,
)
from swaag.sqlite_schema import apply_sqlite_migrations
from swaag.task_api import TaskApi
from swaag.telemetry import record_http_response_status, record_protocol_correlation
from swaag.utils import new_id, stable_json_dumps, utc_now_iso
from swaag.workers import WORKER_TERMINAL_STATES, WorkerManager, WorkerRecord


_COMMUNICATION_STORE_MIGRATIONS = (
    (
        """
        CREATE TABLE IF NOT EXISTS requests (
            correlation_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            message TEXT NOT NULL,
            source TEXT NOT NULL,
            priority INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            reply TEXT
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS requests_pending
        ON requests(status, priority DESC, created_at, correlation_id)
        """,
    ),
    (
        """
        CREATE TABLE protocol_contexts (
            protocol TEXT NOT NULL,
            external_context_id TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (protocol, external_context_id)
        )
        """,
        """
        CREATE TABLE protocol_messages (
            protocol TEXT NOT NULL,
            external_message_id TEXT NOT NULL,
            external_context_id TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (protocol, external_message_id)
        )
        """,
        """
        CREATE INDEX protocol_messages_context
        ON protocol_messages(protocol, external_context_id, created_at)
        """,
    ),
    (
        """
        ALTER TABLE protocol_messages
        ADD COLUMN start_sequence INTEGER NOT NULL DEFAULT 0
        """,
        """
        ALTER TABLE protocol_messages
        ADD COLUMN end_sequence INTEGER
        """,
        """
        CREATE INDEX protocol_messages_stream_bounds
        ON protocol_messages(
            protocol, external_context_id, worker_id,
            start_sequence, end_sequence
        )
        """,
    ),
    (
        """
        CREATE TABLE protocol_state_snapshots (
            protocol TEXT NOT NULL,
            external_context_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            external_message_id TEXT NOT NULL,
            state_json TEXT NOT NULL,
            state_sha256 TEXT NOT NULL,
            client_supplied INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (protocol, external_context_id, revision),
            UNIQUE (protocol, external_message_id)
        )
        """,
        """
        CREATE INDEX protocol_state_snapshots_latest
        ON protocol_state_snapshots(protocol, external_context_id, revision DESC)
        """,
    ),
    (
        """
        ALTER TABLE protocol_state_snapshots
        ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'client_snapshot'
        """,
        """
        ALTER TABLE protocol_state_snapshots
        ADD COLUMN source_session_id TEXT
        """,
        """
        ALTER TABLE protocol_state_snapshots
        ADD COLUMN source_call_id TEXT
        """,
        """
        ALTER TABLE protocol_state_snapshots
        ADD COLUMN base_revision INTEGER
        """,
        """
        ALTER TABLE protocol_state_snapshots
        ADD COLUMN base_state_sha256 TEXT
        """,
        """
        ALTER TABLE protocol_state_snapshots
        ADD COLUMN patch_json TEXT
        """,
        """
        ALTER TABLE protocol_state_snapshots
        ADD COLUMN patch_sha256 TEXT
        """,
        """
        ALTER TABLE protocol_state_snapshots
        ADD COLUMN history_event_sequence INTEGER
        """,
        """
        ALTER TABLE protocol_state_snapshots
        ADD COLUMN history_event_hash TEXT
        """,
        """
        CREATE UNIQUE INDEX protocol_state_agent_calls
        ON protocol_state_snapshots(protocol, source_call_id)
        WHERE source_call_id IS NOT NULL
        """,
    ),
)


def require_loopback_bind_host(host: str) -> str:
    candidate = str(host).strip()
    if candidate.casefold() == "localhost":
        return candidate
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError as exc:
        raise ValueError(
            "The unauthenticated communication service may bind only to an explicit loopback address"
        ) from exc
    if not address.is_loopback:
        raise ValueError(
            "The unauthenticated communication service may bind only to an explicit loopback address"
        )
    return candidate


@dataclass(slots=True, frozen=True)
class CommunicationRequest:
    correlation_id: str
    session_id: str
    message: str
    source: str
    priority: int
    status: str
    created_at: str
    completed_at: str | None = None
    reply: str | None = None


ProtocolStateSnapshot = SharedStateSnapshot


class CommunicationStore:
    def __init__(self, root: Path):
        self.path = Path(root).expanduser() / "communication.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            apply_sqlite_migrations(
                connection,
                store_name="communication store",
                migrations=_COMMUNICATION_STORE_MIGRATIONS,
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def create(self, session_id: str, message: str, *, source: str = "communication") -> CommunicationRequest:
        text = message.strip()
        if not text:
            raise ValueError("communication message must not be empty")
        priority = 0
        request = CommunicationRequest(new_id("correlation"), session_id, text, source, priority, "queued", utc_now_iso())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO requests(correlation_id,session_id,message,source,priority,status,created_at) VALUES(?,?,?,?,?,?,?)",
                (request.correlation_id, request.session_id, request.message, request.source, request.priority, request.status, request.created_at),
            )
        return request

    def get(self, correlation_id: str) -> CommunicationRequest | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM requests WHERE correlation_id=?", (correlation_id,)).fetchone()
        return CommunicationRequest(**dict(row)) if row else None

    def next_pending(self, session_id: str | None = None) -> CommunicationRequest | None:
        sql = "SELECT * FROM requests WHERE status='queued'"
        params: tuple[object, ...] = ()
        if session_id is not None:
            sql += " AND session_id=?"
            params = (session_id,)
        sql += " ORDER BY created_at, correlation_id LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return CommunicationRequest(**dict(row)) if row else None

    def set_status(self, correlation_id: str, status: str, *, reply: str | None = None) -> None:
        completed = utc_now_iso() if status in {"completed", "failed"} else None
        with self._connect() as connection:
            connection.execute(
                "UPDATE requests SET status=?,completed_at=?,reply=? WHERE correlation_id=?",
                (status, completed, reply, correlation_id),
            )

    def protocol_worker(self, protocol: str, external_context_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT worker_id FROM protocol_contexts
                WHERE protocol=? AND external_context_id=?
                """,
                (protocol, external_context_id),
            ).fetchone()
        return None if row is None else str(row[0])

    def set_protocol_worker(
        self,
        protocol: str,
        external_context_id: str,
        worker_id: str,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO protocol_contexts(
                    protocol, external_context_id, worker_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(protocol, external_context_id) DO UPDATE SET
                    worker_id=excluded.worker_id,
                    updated_at=excluded.updated_at
                """,
                (protocol, external_context_id, worker_id, now, now),
            )

    def protocol_context_bindings(self, protocol: str) -> list[tuple[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT external_context_id, worker_id
                FROM protocol_contexts
                WHERE protocol=?
                ORDER BY external_context_id
                """,
                (protocol,),
            ).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

    @staticmethod
    def _canonical_state_json(value: Any) -> str:
        try:
            return json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"protocol state must be strict JSON: {exc}") from exc

    @classmethod
    def _protocol_state(cls, row: sqlite3.Row) -> ProtocolStateSnapshot:
        state_json = str(row["state_json"])
        state_sha256 = hashlib.sha256(state_json.encode("utf-8")).hexdigest()
        if state_sha256 != str(row["state_sha256"]):
            raise RuntimeError("protocol state snapshot hash verification failed")
        state = json.loads(state_json)
        if cls._canonical_state_json(state) != state_json:
            raise RuntimeError("protocol state snapshot is not canonical strict JSON")
        source_kind = str(row["source_kind"])
        if source_kind not in {"client_snapshot", "agent_patch"}:
            raise RuntimeError(f"protocol state source kind is invalid: {source_kind}")
        patch_json = row["patch_json"]
        patch: tuple[dict[str, Any], ...] | None = None
        patch_sha256 = None
        if (patch_json is None) != (row["patch_sha256"] is None):
            raise RuntimeError("protocol state patch lineage is incomplete")
        if patch_json is not None:
            patch_text = str(patch_json)
            patch_sha256 = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
            if patch_sha256 != str(row["patch_sha256"]):
                raise RuntimeError("protocol state patch hash verification failed")
            raw_patch = json.loads(patch_text)
            if not isinstance(raw_patch, list) or any(
                not isinstance(item, dict) for item in raw_patch
            ):
                raise RuntimeError("protocol state patch is not an operation array")
            if cls._canonical_state_json(raw_patch) != patch_text:
                raise RuntimeError("protocol state patch is not canonical strict JSON")
            patch = tuple(dict(item) for item in raw_patch)
        if source_kind == "client_snapshot" and (
            patch is not None
            or row["source_session_id"] is not None
            or row["source_call_id"] is not None
            or row["history_event_sequence"] is not None
        ):
            raise RuntimeError("client protocol state snapshot has agent lineage")
        if source_kind == "agent_patch" and (
            patch is None
            or row["source_session_id"] is None
            or row["source_call_id"] is None
            or row["base_revision"] is None
            or row["base_state_sha256"] is None
            or bool(row["client_supplied"])
        ):
            raise RuntimeError("agent protocol state update has incomplete lineage")
        if (row["history_event_sequence"] is None) != (
            row["history_event_hash"] is None
        ):
            raise RuntimeError("protocol state history lineage is incomplete")
        return ProtocolStateSnapshot(
            protocol=str(row["protocol"]),
            external_context_id=str(row["external_context_id"]),
            revision=int(row["revision"]),
            source_id=str(row["external_message_id"]),
            source_kind=source_kind,
            state=state,
            state_sha256=state_sha256,
            client_supplied=bool(row["client_supplied"]),
            created_at=str(row["created_at"]),
            source_session_id=(
                None
                if row["source_session_id"] is None
                else str(row["source_session_id"])
            ),
            source_call_id=(
                None if row["source_call_id"] is None else str(row["source_call_id"])
            ),
            base_revision=(
                None if row["base_revision"] is None else int(row["base_revision"])
            ),
            base_state_sha256=(
                None
                if row["base_state_sha256"] is None
                else str(row["base_state_sha256"])
            ),
            patch=patch,
            patch_sha256=patch_sha256,
            history_event_sequence=(
                None
                if row["history_event_sequence"] is None
                else int(row["history_event_sequence"])
            ),
            history_event_hash=(
                None
                if row["history_event_hash"] is None
                else str(row["history_event_hash"])
            ),
        )

    @staticmethod
    def _latest_protocol_state_row(
        connection: sqlite3.Connection,
        protocol: str,
        external_context_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT snapshots.*
            FROM protocol_state_snapshots AS snapshots
            LEFT JOIN protocol_messages AS messages
              ON messages.protocol=snapshots.protocol
             AND messages.external_message_id=snapshots.external_message_id
            WHERE snapshots.protocol=? AND snapshots.external_context_id=?
              AND (
                snapshots.source_kind='agent_patch'
                OR messages.external_message_id IS NOT NULL
              )
            ORDER BY snapshots.revision DESC LIMIT 1
            """,
            (protocol, external_context_id),
        ).fetchone()

    def bind_protocol_state(
        self,
        protocol: str,
        external_context_id: str,
        external_message_id: str,
        *,
        state: Any,
        client_supplied: bool,
    ) -> ProtocolStateSnapshot:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM protocol_state_snapshots
                WHERE protocol=? AND external_message_id=?
                """,
                (protocol, external_message_id),
            ).fetchone()
            if existing is not None:
                if str(existing["external_context_id"]) != external_context_id:
                    raise ValueError(
                        "protocol message is already bound to another context"
                    )
                if str(existing["source_kind"]) != "client_snapshot":
                    raise ValueError(
                        "protocol message id collides with an agent state update"
                    )
                return self._protocol_state(existing)
            latest = self._latest_protocol_state_row(
                connection, protocol, external_context_id
            )
            latest_snapshot = (
                None if latest is None else self._protocol_state(latest)
            )
            effective_state = (
                state
                if client_supplied
                else latest_snapshot.state
                if latest_snapshot is not None
                else {}
            )
            state_json = self._canonical_state_json(effective_state)
            max_row = connection.execute(
                """
                SELECT COALESCE(MAX(revision), 0)
                FROM protocol_state_snapshots
                WHERE protocol=? AND external_context_id=?
                """,
                (protocol, external_context_id),
            ).fetchone()
            revision = int(max_row[0]) + 1
            created_at = utc_now_iso()
            state_sha256 = hashlib.sha256(state_json.encode("utf-8")).hexdigest()
            connection.execute(
                """
                INSERT INTO protocol_state_snapshots(
                    protocol, external_context_id, revision, external_message_id,
                    state_json, state_sha256, client_supplied, created_at,
                    source_kind, base_revision, base_state_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'client_snapshot', ?, ?)
                """,
                (
                    protocol,
                    external_context_id,
                    revision,
                    external_message_id,
                    state_json,
                    state_sha256,
                    int(client_supplied),
                    created_at,
                    (
                        None
                        if latest_snapshot is None
                        else latest_snapshot.revision
                    ),
                    (
                        None
                        if latest_snapshot is None
                        else latest_snapshot.state_sha256
                    ),
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM protocol_state_snapshots
                WHERE protocol=? AND external_message_id=?
                """,
                (protocol, external_message_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("protocol state snapshot was not stored")
        return self._protocol_state(row)

    def latest_protocol_state(
        self, protocol: str, external_context_id: str
    ) -> ProtocolStateSnapshot | None:
        with self._connect() as connection:
            row = self._latest_protocol_state_row(
                connection, protocol, external_context_id
            )
        return None if row is None else self._protocol_state(row)

    def protocol_state_for_agent_call(
        self, protocol: str, source_call_id: str
    ) -> ProtocolStateSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM protocol_state_snapshots
                WHERE protocol=? AND source_call_id=?
                """,
                (protocol, source_call_id),
            ).fetchone()
        return None if row is None else self._protocol_state(row)

    def protocol_state_updates(
        self, protocol: str
    ) -> list[ProtocolStateSnapshot]:
        """Return every durable agent patch in deterministic recovery order."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM protocol_state_snapshots
                WHERE protocol=? AND source_kind='agent_patch'
                ORDER BY created_at, external_context_id, revision
                """,
                (protocol,),
            ).fetchall()
        return [self._protocol_state(row) for row in rows]

    def apply_protocol_state_patch(
        self,
        protocol: str,
        external_context_id: str,
        *,
        source_session_id: str,
        source_call_id: str,
        base_revision: int,
        base_state_sha256: str,
        patch: list[dict[str, Any]],
    ) -> ProtocolStateSnapshot:
        if not source_session_id or not source_call_id:
            raise ValueError("agent state update source ids must be non-empty")
        if not isinstance(patch, list) or not patch:
            raise ValueError("agent state update patch must be a non-empty array")
        patch_json = self._canonical_state_json(patch)
        normalized_patch = json.loads(patch_json)
        patch_sha256 = hashlib.sha256(patch_json.encode("utf-8")).hexdigest()
        source_id = f"agent-state:{source_call_id}"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM protocol_state_snapshots
                WHERE protocol=? AND source_call_id=?
                """,
                (protocol, source_call_id),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["external_context_id"]),
                    str(existing["source_session_id"]),
                    int(existing["base_revision"]),
                    str(existing["base_state_sha256"]),
                    str(existing["patch_json"]),
                ) != (
                    external_context_id,
                    source_session_id,
                    int(base_revision),
                    base_state_sha256,
                    patch_json,
                ):
                    raise ValueError(
                        "agent state call is already bound to a different exact update"
                    )
                return self._protocol_state(existing)
            latest_row = self._latest_protocol_state_row(
                connection, protocol, external_context_id
            )
            if latest_row is None:
                raise ValueError("shared state has no accepted baseline snapshot")
            latest = self._protocol_state(latest_row)
            if (latest.revision, latest.state_sha256) != (
                int(base_revision),
                base_state_sha256,
            ):
                raise SharedStateConflictError(
                    "shared state changed after the supplied base revision",
                    latest,
                )
            try:
                updated_state = jsonpatch.apply_patch(
                    latest.state, normalized_patch, in_place=False
                )
            except (jsonpatch.JsonPatchException, jsonpatch.JsonPointerException) as exc:
                raise ValueError(f"invalid RFC 6902 shared-state patch: {exc}") from exc
            state_json = self._canonical_state_json(updated_state)
            if state_json == self._canonical_state_json(latest.state):
                raise ValueError("shared-state patch made no state change")
            state_sha256 = hashlib.sha256(state_json.encode("utf-8")).hexdigest()
            max_row = connection.execute(
                """
                SELECT COALESCE(MAX(revision), 0)
                FROM protocol_state_snapshots
                WHERE protocol=? AND external_context_id=?
                """,
                (protocol, external_context_id),
            ).fetchone()
            revision = int(max_row[0]) + 1
            created_at = utc_now_iso()
            connection.execute(
                """
                INSERT INTO protocol_state_snapshots(
                    protocol, external_context_id, revision, external_message_id,
                    state_json, state_sha256, client_supplied, created_at,
                    source_kind, source_session_id, source_call_id, base_revision,
                    base_state_sha256, patch_json, patch_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, 'agent_patch', ?, ?, ?, ?, ?, ?)
                """,
                (
                    protocol,
                    external_context_id,
                    revision,
                    source_id,
                    state_json,
                    state_sha256,
                    created_at,
                    source_session_id,
                    source_call_id,
                    latest.revision,
                    latest.state_sha256,
                    patch_json,
                    patch_sha256,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM protocol_state_snapshots
                WHERE protocol=? AND source_call_id=?
                """,
                (protocol, source_call_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("agent protocol state update was not stored")
        return self._protocol_state(row)

    def link_protocol_state_history(
        self,
        protocol: str,
        source_call_id: str,
        *,
        source_session_id: str,
        sequence: int,
        event_hash: str,
    ) -> ProtocolStateSnapshot:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM protocol_state_snapshots
                WHERE protocol=? AND source_call_id=?
                """,
                (protocol, source_call_id),
            ).fetchone()
            if row is None or str(row["source_session_id"]) != source_session_id:
                raise ValueError("agent shared-state update is unknown for this session")
            current = self._protocol_state(row)
            existing = (current.history_event_sequence, current.history_event_hash)
            requested = (int(sequence), event_hash)
            if current.history_event_sequence is not None and existing != requested:
                raise ValueError(
                    "agent shared-state update is linked to different history"
                )
            connection.execute(
                """
                UPDATE protocol_state_snapshots
                SET history_event_sequence=?, history_event_hash=?
                WHERE protocol=? AND source_call_id=?
                """,
                (int(sequence), event_hash, protocol, source_call_id),
            )
            stored = connection.execute(
                """
                SELECT * FROM protocol_state_snapshots
                WHERE protocol=? AND source_call_id=?
                """,
                (protocol, source_call_id),
            ).fetchone()
        if stored is None:
            raise RuntimeError("agent shared-state history link was not stored")
        return self._protocol_state(stored)

    def protocol_state_for_message(
        self, protocol: str, external_message_id: str
    ) -> ProtocolStateSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM protocol_state_snapshots
                WHERE protocol=? AND external_message_id=?
                """,
                (protocol, external_message_id),
            ).fetchone()
        return None if row is None else self._protocol_state(row)

    def protocol_message(
        self,
        protocol: str,
        external_message_id: str,
    ) -> tuple[str, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT external_context_id, worker_id FROM protocol_messages
                WHERE protocol=? AND external_message_id=?
                """,
                (protocol, external_message_id),
            ).fetchone()
        return None if row is None else (str(row[0]), str(row[1]))

    def protocol_message_bounds(
        self,
        protocol: str,
        external_message_id: str,
    ) -> tuple[str, str, int, int | None] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT external_context_id, worker_id, start_sequence, end_sequence
                FROM protocol_messages
                WHERE protocol=? AND external_message_id=?
                """,
                (protocol, external_message_id),
            ).fetchone()
        if row is None:
            return None
        return (
            str(row[0]),
            str(row[1]),
            int(row[2]),
            None if row[3] is None else int(row[3]),
        )

    def record_protocol_message(
        self,
        protocol: str,
        external_message_id: str,
        external_context_id: str,
        worker_id: str,
        *,
        start_sequence: int = 0,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO protocol_messages(
                    protocol, external_message_id, external_context_id,
                    worker_id, created_at, start_sequence
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    protocol,
                    external_message_id,
                    external_context_id,
                    worker_id,
                    utc_now_iso(),
                    start_sequence,
                ),
            )

    def finish_protocol_message(
        self,
        protocol: str,
        external_message_id: str,
        *,
        end_sequence: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE protocol_messages SET end_sequence=?
                WHERE protocol=? AND external_message_id=?
                  AND (end_sequence IS NULL OR end_sequence>?)
                """,
                (
                    end_sequence,
                    protocol,
                    external_message_id,
                    end_sequence,
                ),
            )

    def close_protocol_streams(
        self,
        protocol: str,
        external_context_id: str,
        worker_id: str,
        *,
        end_sequence: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE protocol_messages SET end_sequence=?
                WHERE protocol=? AND external_context_id=? AND worker_id=?
                  AND end_sequence IS NULL
                """,
                (end_sequence, protocol, external_context_id, worker_id),
            )


class _AgUiSharedStateChannel:
    def __init__(
        self,
        service: "CommunicationService",
        *,
        external_context_id: str,
        worker_id: str,
        session_id: str,
    ):
        self._service = service
        self._external_context_id = external_context_id
        self._worker_id = worker_id
        self._session_id = session_id

    def _verify_binding(self) -> None:
        worker_id = self._service.store.protocol_worker(
            "ag_ui", self._external_context_id
        )
        if worker_id != self._worker_id:
            raise RuntimeError(
                "AG-UI shared-state channel is no longer bound to this worker"
            )
        record = self._service.workers.store.get(self._worker_id)
        if record.session_id != self._session_id:
            raise RuntimeError(
                "AG-UI shared-state worker no longer owns this session"
            )

    def _verify_history_link(
        self,
        snapshot: SharedStateSnapshot,
        *,
        sequence: int,
        event_hash: str,
    ) -> None:
        event = next(
            self._service.runtime.history.iter_history(
                self._session_id,
                start_sequence=sequence,
                end_sequence=sequence,
            ),
            None,
        )
        expected_payload = shared_state_event_payload(snapshot)
        if event is None or (
            event.session_id,
            event.event_type,
            event.hash,
            event.payload,
        ) != (
            self._session_id,
            "shared_state_updated",
            event_hash,
            expected_payload,
        ):
            raise RuntimeError(
                "AG-UI shared-state history link failed exact integrity validation"
            )

    def snapshot(self) -> SharedStateSnapshot:
        with self._service._protocol_send_lock:
            self._verify_binding()
            snapshot = self._service.store.latest_protocol_state(
                "ag_ui", self._external_context_id
            )
            if snapshot is None:
                raise RuntimeError("AG-UI shared state has no accepted baseline")
            if snapshot.source_kind == "agent_patch":
                if (
                    snapshot.history_event_sequence is None
                    or snapshot.history_event_hash is None
                ):
                    raise RuntimeError(
                        "AG-UI shared-state update lacks canonical history lineage"
                    )
                self._verify_history_link(
                    snapshot,
                    sequence=snapshot.history_event_sequence,
                    event_hash=snapshot.history_event_hash,
                )
            return snapshot

    def apply_patch(
        self,
        *,
        source_call_id: str,
        base_revision: int,
        base_state_sha256: str,
        patch: list[dict[str, Any]],
    ) -> SharedStateSnapshot:
        with self._service._protocol_send_lock:
            self._verify_binding()
            return self._service.store.apply_protocol_state_patch(
                "ag_ui",
                self._external_context_id,
                source_session_id=self._session_id,
                source_call_id=source_call_id,
                base_revision=base_revision,
                base_state_sha256=base_state_sha256,
                patch=patch,
            )

    def link_history(
        self,
        *,
        source_call_id: str,
        sequence: int,
        event_hash: str,
    ) -> SharedStateSnapshot:
        with self._service._protocol_send_lock:
            snapshot = self._service.store.protocol_state_for_agent_call(
                "ag_ui", source_call_id
            )
            if snapshot is None or snapshot.source_session_id != self._session_id:
                raise ValueError(
                    "agent shared-state update is unknown for this session"
                )
            self._verify_history_link(
                snapshot,
                sequence=sequence,
                event_hash=event_hash,
            )
            return self._service.store.link_protocol_state_history(
                "ag_ui",
                source_call_id,
                source_session_id=self._session_id,
                sequence=sequence,
                event_hash=event_hash,
            )


class CommunicationService:
    """Separate correlated communication/control service using the canonical AgentRuntime."""

    def __init__(self, runtime: AgentRuntime, *, assistant_runtime: AgentRuntime | None = None, max_concurrency: int = 4):
        self.runtime = runtime
        self.assistant_runtime = assistant_runtime
        self.store = CommunicationStore(runtime.config.sessions.root)
        self._protocol_send_lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self.workers = WorkerManager(runtime, max_workers=max_concurrency)
        self.task_api = TaskApi(self.workers)
        self.mcp = McpAdapter(runtime)
        self._advertised_host = str(runtime.config.communication.host).strip()
        self._advertised_port = int(runtime.config.communication.port)
        self._reconcile_ag_ui_shared_state_history()
        self._restore_ag_ui_shared_state_channels()

    def _reconcile_ag_ui_shared_state_history(self) -> None:
        """Repair the cross-store crash window without inventing state."""
        with self._protocol_send_lock:
            for snapshot in self.store.protocol_state_updates("ag_ui"):
                session_id = snapshot.source_session_id
                source_call_id = snapshot.source_call_id
                if not session_id or not source_call_id:
                    raise RuntimeError(
                        "AG-UI shared-state update has incomplete source lineage"
                    )
                expected_payload = shared_state_event_payload(snapshot)
                if snapshot.history_event_sequence is not None:
                    if snapshot.history_event_hash is None:
                        raise RuntimeError(
                            "AG-UI shared-state update has incomplete history lineage"
                        )
                    event = next(
                        self.runtime.history.iter_history(
                            session_id,
                            start_sequence=snapshot.history_event_sequence,
                            end_sequence=snapshot.history_event_sequence,
                        ),
                        None,
                    )
                    if event is None or (
                        event.event_type,
                        event.hash,
                        event.payload,
                    ) != (
                        "shared_state_updated",
                        snapshot.history_event_hash,
                        expected_payload,
                    ):
                        raise RuntimeError(
                            "AG-UI shared-state history link failed exact integrity validation"
                        )
                    continue

                matches = [
                    event
                    for event in self.runtime.history.iter_history_reverse(
                        session_id, event_types=("shared_state_updated",)
                    )
                    if event.payload.get("source_call_id") == source_call_id
                ]
                if len(matches) > 1:
                    raise RuntimeError(
                        "AG-UI shared-state update has duplicate canonical history"
                    )
                if matches:
                    event = matches[0]
                    if event.payload != expected_payload:
                        raise RuntimeError(
                            "AG-UI shared-state history differs from durable state"
                        )
                else:
                    state = self.runtime.history.rebuild_from_history(
                        session_id, write_projections=False
                    )
                    event = self.runtime.history.record_event(
                        state, "shared_state_updated", expected_payload
                    )
                self.store.link_protocol_state_history(
                    "ag_ui",
                    source_call_id,
                    source_session_id=session_id,
                    sequence=event.sequence,
                    event_hash=event.hash,
                )

    def _bind_ag_ui_shared_state(
        self, record: WorkerRecord, external_context_id: str
    ) -> None:
        channel: SharedStateChannel = _AgUiSharedStateChannel(
            self,
            external_context_id=external_context_id,
            worker_id=record.worker_id,
            session_id=record.session_id,
        )
        self.runtime.bind_tool_runtime_capability(
            record.session_id, "shared_state", channel
        )

    def _restore_ag_ui_shared_state_channels(self) -> None:
        for external_context_id, worker_id in self.store.protocol_context_bindings(
            "ag_ui"
        ):
            try:
                record = self.workers.store.get(worker_id)
            except FileNotFoundError:
                continue
            self._bind_ag_ui_shared_state(record, external_context_id)

    @classmethod
    def from_config(cls, config: AgentConfig) -> "CommunicationService":
        return cls.from_runtime(AgentRuntime(config))

    @classmethod
    def from_runtime(cls, main: AgentRuntime) -> "CommunicationService":
        config = main.config
        assistant = None
        if getattr(config, "communication", None) and config.communication.enabled:
            if config.communication.model_base_url:
                assistant_config = copy.deepcopy(config)
                assistant_config.model.base_url = config.communication.model_base_url
                assistant_config.tools.enabled = list(config.communication.enabled_tools)
                assistant_config.tools.allow_side_effect_tools = False
                assistant = AgentRuntime(assistant_config)
            return cls(main, assistant_runtime=assistant, max_concurrency=config.communication.max_concurrent_requests)
        return cls(main)

    def submit(self, session_ref: str | None, message: str, *, source: str = "communication") -> CommunicationRequest:
        session_id = self.runtime.resolve_session_ref(session_ref, latest_if_none=True)
        if session_id is None:
            raise FileNotFoundError("No target session available")
        return self.store.create(session_id, message, source=source)

    def status(self, correlation_id: str) -> CommunicationRequest:
        request = self.store.get(correlation_id)
        if request is None:
            raise FileNotFoundError(f"Unknown correlation id: {correlation_id}")
        return request

    def _preempt_active_main_call(self, session_id: str, message: str):
        request = self.runtime.preemption.request_preemption(session_id, message, source="communication")
        if request is None:
            return None
        timeout = max(
            1.0,
            float(self.runtime.config.model.structured_timeout_seconds),
            float(self.runtime.config.model.timeout_seconds),
        )
        interrupted = self.runtime.preemption.wait_for_status(
            request.preemption_id,
            {"interrupted", "failed"},
            timeout_seconds=timeout,
            poll_seconds=0.02,
        )
        if interrupted.status == "failed":
            raise RuntimeError(interrupted.reply or "main model preemption failed")
        self.runtime.preemption.mark_assistant_running(request.preemption_id)
        return request

    def _complete_preemption(self, request, *, target_changed: bool, reply: str | None = None) -> None:
        if request is not None:
            self.runtime.preemption.complete(
                request.preemption_id,
                target_changed=target_changed,
                reply=reply,
            )

    def _fail_preemption(self, request, exc: Exception) -> None:
        if request is not None:
            self.runtime.preemption.fail(request.preemption_id, f"{type(exc).__name__}: {exc}")

    def process_once(self, *, session_id: str | None = None) -> CommunicationRequest | None:
        request = self.store.next_pending(session_id)
        if request is None:
            return None
        self.store.set_status(request.correlation_id, "processing")
        preemption = None
        try:
            preemption = self._preempt_active_main_call(request.session_id, request.message)
            self.runtime.history.enqueue_control_message(
                request.session_id,
                request.message,
                source=f"communication:{request.correlation_id}",
                control_id=request.correlation_id,
            )
            state = self.runtime.history.rebuild_from_history(request.session_id, write_projections=False)
            with self.runtime.inference_priority(
                100, source="communication_control"
            ):
                result = self.runtime.run_pending_controls_in_session(state)
            reply = result.assistant_text if result is not None else ""
            self.store.set_status(request.correlation_id, "completed", reply=reply)
            self._complete_preemption(preemption, target_changed=True, reply=reply)
        except Exception as exc:
            self._fail_preemption(preemption, exc)
            self.store.set_status(request.correlation_id, "failed", reply=f"{type(exc).__name__}: {exc}")
        return self.status(request.correlation_id)

    async def process_once_async(self, *, session_id: str | None = None) -> CommunicationRequest | None:
        async with self._semaphore:
            return await asyncio.to_thread(self.process_once, session_id=session_id)

    def answer_status_question(self, session_ref: str | None, question: str) -> str:
        session_id = self.runtime.resolve_session_ref(session_ref, latest_if_none=True)
        if session_id is None:
            raise FileNotFoundError("No target session available")
        preemption = None
        try:
            if self.assistant_runtime is None:
                preemption = self._preempt_active_main_call(session_id, question)
            state = self.runtime.history.rebuild_from_history(
                session_id, write_projections=False
            )
            mechanical_status = self.runtime.session_status_payload(state)
            source_events = self.runtime.history.read_history(session_id)
            semantic_runtime = self.assistant_runtime or self.runtime
            assistant_failure: Exception | None = None
            try:
                with semantic_runtime.inference_priority(
                    100, source="communication_status"
                ):
                    semantic_status = semantic_runtime.generate_communication_status(
                        target_session_id=session_id,
                        question=question,
                        mechanical_status=mechanical_status,
                        source_events=source_events,
                    )
            except Exception as exc:
                if self.assistant_runtime is None:
                    raise
                assistant_failure = exc
                operation_session_id = str(
                    getattr(exc, "swaag_operation_session_id", "")
                )
                if not operation_session_id:
                    raise
                operation_state = self.assistant_runtime.history.rebuild_from_history(
                    operation_session_id,
                    write_projections=False,
                )
                unavailable_event = next(
                    (
                        event
                        for event in reversed(
                            self.assistant_runtime.history.read_history(
                                operation_session_id
                            )
                        )
                        if event.event_type == "communication_status_unavailable"
                    ),
                    None,
                )
                if unavailable_event is None:
                    raise
                semantic_status = {
                    "operation_session_id": operation_session_id,
                    "escalate_to_stronger_model": True,
                    "escalation_reason": (
                        "The separate communication status operation failed "
                        f"mechanically: {type(exc).__name__}: {exc}"
                    ),
                    "source_event_references": unavailable_event.payload[
                        "source_event_references"
                    ],
                }
            if (
                self.assistant_runtime is not None
                and bool(semantic_status["escalate_to_stronger_model"])
            ):
                operation_session_id = str(
                    semantic_status["operation_session_id"]
                )
                operation_state = self.assistant_runtime.history.rebuild_from_history(
                    operation_session_id,
                    write_projections=False,
                )
                escalation_event = self.assistant_runtime.history.record_event(
                    operation_state,
                    "communication_status_escalation_requested",
                    {
                        "target_session_id": session_id,
                        "question": question,
                        "reason": semantic_status["escalation_reason"],
                        "trigger": (
                            "assistant_failure"
                            if assistant_failure is not None
                            else "semantic_request"
                        ),
                        "status_operation_session_id": operation_session_id,
                        "source_event_references": semantic_status[
                            "source_event_references"
                        ],
                    },
                )
                preemption = self._preempt_active_main_call(session_id, question)
                try:
                    with self.runtime.inference_priority(
                        100, source="communication_status_escalation"
                    ):
                        stronger_status = self.runtime.generate_communication_status(
                            target_session_id=session_id,
                            question=question,
                            mechanical_status=mechanical_status,
                            source_events=source_events,
                        )
                except Exception as exc:
                    self.assistant_runtime.history.record_event(
                        operation_state,
                        "communication_status_escalation_failed",
                        {
                            "target_session_id": session_id,
                            "question": question,
                            "request_event_sequence": escalation_event.sequence,
                            "request_event_hash": escalation_event.hash,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise
                answer = str(stronger_status["answer"])
                self.assistant_runtime.history.record_event(
                    operation_state,
                    "communication_status_escalation_resolved",
                    {
                        "target_session_id": session_id,
                        "question": question,
                        "request_event_sequence": escalation_event.sequence,
                        "request_event_hash": escalation_event.hash,
                        "stronger_operation_session_id": stronger_status[
                            "operation_session_id"
                        ],
                        "answer_sha256": hashlib.sha256(
                            answer.encode("utf-8")
                        ).hexdigest(),
                        "stronger_model_requested_further_escalation": bool(
                            stronger_status["escalate_to_stronger_model"]
                        ),
                    },
                )
            else:
                answer = str(semantic_status["answer"])
            self._complete_preemption(preemption, target_changed=False, reply=answer)
            return answer
        except Exception as exc:
            self._fail_preemption(preemption, exc)
            raise

    def protocol_projection(
        self,
        protocol: str,
        operation: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if protocol == "open_webui" and operation == "send":
            return self._open_webui_send(params)
        if protocol == "a2a" and operation == "send":
            return self._a2a_send(params, wait_for_completion=True)
        if protocol == "a2a" and operation == "list":
            return self._a2a_list(params)
        worker_id = str(
            params.get("worker_id")
            or (params.get("id") if protocol == "a2a" else "")
        ).strip()
        if not worker_id:
            raise ValueError(
                ("id" if protocol == "a2a" else "worker_id")
                + " must be a non-empty string"
            )
        record = self.workers.store.get(worker_id)
        if protocol == "a2a" and operation == "get":
            return {
                "protocol": A2AProjectionAdapter.protocol_version,
                "task": self._a2a_task(
                    record,
                    history_length=_a2a_history_length(params),
                ),
            }
        if protocol == "a2a" and operation == "cancel":
            if record.status in {"completed", "failed"} or record.archived_at is not None:
                raise A2ATaskNotCancelableError(
                    f"A2A task {worker_id} cannot be canceled from {record.status}"
                )
            canceled = self.workers.cancel(
                worker_id,
                reason=str(params.get("reason") or "A2A cancellation"),
            )
            return {
                "protocol": A2AProjectionAdapter.protocol_version,
                "task": A2AProjectionAdapter().task(canceled),
            }
        if protocol == "a2a" and operation == "subscribe":
            if record.status in {"completed", "failed", "canceled"}:
                raise ValueError("A2A cannot subscribe to a terminal task")
            page = self.task_api.execute(
                "events.wait",
                {**params, "worker_id": worker_id},
            )
            current = self.workers.store.get(worker_id)
            return self._a2a_subscription_response(record, current, page)
        if protocol == "open_webui" and operation == "get":
            page = self.task_api.execute(
                "events",
                {**params, "worker_id": worker_id},
            )
            return {
                **OpenWebUiProjectionAdapter().response(
                    record,
                    [
                        self.workers.event_from_payload(item)
                        for item in page["events"]
                    ],
                ),
                "next_sequence": page["next_sequence"],
                "has_more": page["has_more"],
            }
        if protocol == "ag_ui" and operation in {"events", "subscribe"}:
            page = self.task_api.execute(
                "events.wait" if operation == "subscribe" else "events",
                params,
            )
            projected = AgUiProjectionAdapter().events(
                record,
                [
                    self.workers.event_from_payload(item)
                    for item in page["events"]
                ],
            )
            return {
                "protocol": "ag-ui",
                "worker_id": worker_id,
                "events": projected,
                "next_sequence": page["next_sequence"],
                "has_more": page["has_more"],
                **(
                    {
                        "terminal": page["terminal"],
                        "timed_out": page["timed_out"],
                    }
                    if operation == "subscribe"
                    else {}
                ),
            }
        raise ValueError(f"unsupported protocol operation: {protocol}.{operation}")

    def _open_webui_send(self, params: dict[str, Any]) -> dict[str, Any]:
        conversation_id = _required_protocol_text(
            params, "conversation_id", protocol="Open WebUI"
        )
        request_id = _required_protocol_text(
            params, "request_id", protocol="Open WebUI"
        )
        record_protocol_correlation(
            protocol="open_webui",
            request_id=request_id,
            context_id=conversation_id,
        )
        message = _required_protocol_text(params, "message", protocol="Open WebUI")
        attachments = params.get("attachments", [])
        if not isinstance(attachments, list) or any(
            not isinstance(item, dict) for item in attachments
        ):
            raise ValueError("Open WebUI attachments must be an array of objects")

        with self._protocol_send_lock:
            duplicate = self.store.protocol_message_bounds("open_webui", request_id)
            if duplicate is not None:
                duplicate_context_id, worker_id, start_sequence, _end_sequence = duplicate
                if duplicate_context_id != conversation_id:
                    raise ValueError(
                        "Open WebUI request_id is already bound to another conversation"
                    )
                record = self.workers.store.get(worker_id)
                record_protocol_correlation(
                    protocol="open_webui",
                    request_id=request_id,
                    context_id=conversation_id,
                    worker_id=record.worker_id,
                    session_id=record.session_id,
                )
                return {
                    **OpenWebUiProjectionAdapter().response(record),
                    "conversation_id": conversation_id,
                    "next_sequence": start_sequence,
                    "duplicate": True,
                }

            worker_id = self.store.protocol_worker("open_webui", conversation_id)
            record = None
            if worker_id is not None:
                try:
                    record = self.workers.store.get(worker_id)
                except FileNotFoundError:
                    worker_id = None
            if record is not None and record.archived_at is not None:
                worker_id = None
                record = None

            if worker_id is None:
                created = self.task_api.execute(
                    "create",
                    {
                        "objective": message,
                        "attachments": attachments,
                        "attachment_source": "open_webui",
                        "start": False,
                    },
                )
                worker_id = str(created["worker"]["worker_id"])
                self.store.set_protocol_worker(
                    "open_webui", conversation_id, worker_id
                )
                _created, start_sequence = self.workers.stream_snapshot(worker_id)
                record = self.workers.start(worker_id)
            else:
                _current, start_sequence = self.workers.stream_snapshot(worker_id)
                for attachment in attachments:
                    self.task_api.execute(
                        "attachment.add",
                        {
                            **attachment,
                            "worker_id": worker_id,
                            "source": "open_webui",
                        },
                    )
                record = self.workers.message(
                    worker_id,
                    message,
                    source=f"open_webui:{request_id}",
                )

            self.store.record_protocol_message(
                "open_webui",
                request_id,
                conversation_id,
                worker_id,
                start_sequence=start_sequence,
            )
            if record is None:
                raise RuntimeError("Open WebUI worker did not start")
            record_protocol_correlation(
                protocol="open_webui",
                request_id=request_id,
                context_id=conversation_id,
                worker_id=record.worker_id,
                session_id=record.session_id,
            )
            return {
                **OpenWebUiProjectionAdapter().response(record),
                "conversation_id": conversation_id,
                "next_sequence": start_sequence,
                "duplicate": False,
            }

    def _ag_ui_begin(
        self,
        run: AgUiRunInput,
    ) -> tuple[WorkerRecord, int, int | None, bool, ProtocolStateSnapshot]:
        record_protocol_correlation(
            protocol="ag_ui",
            request_id=run.run_id,
            context_id=run.thread_id,
        )
        with self._protocol_send_lock:
            duplicate = self.store.protocol_message_bounds("ag_ui", run.run_id)
            if duplicate is not None:
                context_id, worker_id, start_sequence, end_sequence = duplicate
                if context_id != run.thread_id:
                    raise ValueError(
                        "AG-UI runId is already bound to another thread"
                    )
                record = self.workers.store.get(worker_id)
                state_snapshot = self.store.protocol_state_for_message(
                    "ag_ui", run.run_id
                )
                if state_snapshot is None:
                    state_snapshot = self.store.bind_protocol_state(
                        "ag_ui",
                        run.thread_id,
                        run.run_id,
                        state={},
                        client_supplied=False,
                    )
                self._bind_ag_ui_shared_state(record, run.thread_id)
                record_protocol_correlation(
                    protocol="ag_ui",
                    request_id=run.run_id,
                    context_id=run.thread_id,
                    worker_id=record.worker_id,
                    session_id=record.session_id,
                )
                return (
                    record,
                    start_sequence,
                    end_sequence,
                    True,
                    state_snapshot,
                )

            collisions = sorted(
                {tool.name for tool in run.client_tools}
                & self.runtime.tools.registered_names()
            )
            if collisions:
                raise ValueError(
                    "AG-UI client tool names collide with server capabilities: "
                    + ", ".join(collisions)
                )

            worker_id = self.store.protocol_worker("ag_ui", run.thread_id)
            record = None
            if worker_id is not None:
                try:
                    record = self.workers.store.get(worker_id)
                except FileNotFoundError:
                    worker_id = None
            if record is not None and record.archived_at is not None:
                if run.resume:
                    raise ValueError("AG-UI cannot resume an archived thread")
                worker_id = None
                record = None

            delegated_call: DelegatedToolCall | None = None
            delegated_result: DelegatedToolResultInput | None = None
            if worker_id is None:
                if run.resume:
                    raise ValueError("AG-UI cannot resume an unknown thread")
                if run.client_tool_results:
                    raise ValueError(
                        "AG-UI tool messages cannot address an unknown thread"
                    )
            else:
                if record is None:
                    raise RuntimeError("AG-UI worker lookup did not return a record")
                delegated_call = self._ag_ui_delegated_call(record)
                if delegated_call is not None:
                    if run.resume:
                        raise ValueError(
                            "AG-UI delegated tool results use tool messages, not resume entries"
                        )
                    matches = [
                        result
                        for result in run.client_tool_results
                        if result.call_id == delegated_call.call_id
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "AG-UI run must provide exactly one tool message for "
                            f"delegated call {delegated_call.call_id}"
                        )
                    delegated_result = matches[0]
                    self._verify_ag_ui_historical_tool_results(
                        record,
                        run.client_tool_results,
                        exclude_call_id=delegated_call.call_id,
                    )
                    if delegated_call.status in {"resolved", "failed"}:
                        verified = (
                            self.runtime.delegated_tools.verify_result_message(
                                record.session_id, delegated_result
                            )
                        )
                        if (
                            verified.result_source,
                            verified.result_external_request_id,
                        ) != ("ag_ui", run.run_id):
                            raise ValueError(
                                "delegated tool call already has a different exact result"
                            )
                else:
                    self._verify_ag_ui_historical_tool_results(
                        record, run.client_tool_results
                    )
                    if record.status == "input_required" and not run.resume:
                        raise ValueError(
                            "AG-UI human input must resolve the open interrupt"
                        )

            state_snapshot = self.store.bind_protocol_state(
                "ag_ui",
                run.thread_id,
                run.run_id,
                state=run.state,
                client_supplied=run.state_supplied,
            )
            state_context = self._ag_ui_state_context(state_snapshot)

            if worker_id is None:
                created = self.task_api.execute(
                    "create",
                    {
                        "objective": run.initial_text + state_context,
                        "attachments": list(run.initial_attachments),
                        "attachment_source": "ag_ui",
                        "start": False,
                    },
                )
                worker_id = str(created["worker"]["worker_id"])
                self.store.set_protocol_worker("ag_ui", run.thread_id, worker_id)
                created_record = self.workers.store.get(worker_id)
                self._bind_ag_ui_shared_state(created_record, run.thread_id)
                self.runtime.delegated_tools.bind_catalog(
                    created_record.session_id,
                    source="ag_ui",
                    external_context_id=run.thread_id,
                    external_request_id=run.run_id,
                    tools=run.client_tools,
                )
                _created_record, start_sequence = self.workers.stream_snapshot(
                    worker_id
                )
                record = self.workers.start(worker_id)
            else:
                self._bind_ag_ui_shared_state(record, run.thread_id)
                if delegated_call is not None:
                    if delegated_result is None:
                        raise RuntimeError(
                            "validated AG-UI delegated result is unavailable"
                        )
                    self.runtime.accept_delegated_tool_result(
                        record.session_id,
                        delegated_call.call_id,
                        source="ag_ui",
                        external_request_id=run.run_id,
                        result=delegated_result,
                    )
                    self.runtime.delegated_tools.bind_catalog(
                        record.session_id,
                        source="ag_ui",
                        external_context_id=run.thread_id,
                        external_request_id=run.run_id,
                        tools=run.client_tools,
                    )
                    _current, start_sequence = self.workers.stream_snapshot(
                        worker_id
                    )
                    self.store.close_protocol_streams(
                        "ag_ui",
                        run.thread_id,
                        worker_id,
                        end_sequence=start_sequence,
                    )
                    record = self.workers.message(
                        worker_id,
                        (
                            "The connected client returned the exact result for "
                            f"delegated tool call {delegated_call.call_id}."
                            + run.context_text
                            + state_context
                        ),
                        source=f"ag_ui:{run.run_id}",
                    )
                elif run.resume:
                    self.runtime.delegated_tools.bind_catalog(
                        record.session_id,
                        source="ag_ui",
                        external_context_id=run.thread_id,
                        external_request_id=run.run_id,
                        tools=run.client_tools,
                    )
                    _current, start_sequence = self.workers.stream_snapshot(worker_id)
                    self.store.close_protocol_streams(
                        "ag_ui",
                        run.thread_id,
                        worker_id,
                        end_sequence=start_sequence,
                    )
                    record = self._ag_ui_resume(
                        worker_id, run, state_context=state_context
                    )
                else:
                    for attachment in run.attachments:
                        self.task_api.execute(
                            "attachment.add",
                            {
                                **attachment,
                                "worker_id": worker_id,
                                "source": "ag_ui",
                            },
                        )
                    self.runtime.delegated_tools.bind_catalog(
                        record.session_id,
                        source="ag_ui",
                        external_context_id=run.thread_id,
                        external_request_id=run.run_id,
                        tools=run.client_tools,
                    )
                    _current, start_sequence = self.workers.stream_snapshot(worker_id)
                    self.store.close_protocol_streams(
                        "ag_ui",
                        run.thread_id,
                        worker_id,
                        end_sequence=start_sequence,
                    )
                    record = self.workers.message(
                        worker_id,
                        run.text + state_context,
                        source=f"ag_ui:{run.run_id}",
                    )

            self.store.record_protocol_message(
                "ag_ui",
                run.run_id,
                run.thread_id,
                worker_id,
                start_sequence=start_sequence,
            )
            record_protocol_correlation(
                protocol="ag_ui",
                request_id=run.run_id,
                context_id=run.thread_id,
                worker_id=record.worker_id,
                session_id=record.session_id,
            )
            return record, start_sequence, None, False, state_snapshot

    def _verify_ag_ui_historical_tool_results(
        self,
        record: WorkerRecord,
        results: tuple[DelegatedToolResultInput, ...],
        *,
        exclude_call_id: str | None = None,
    ) -> None:
        for result in results:
            if result.call_id == exclude_call_id:
                continue
            self.runtime.delegated_tools.verify_result_message(
                record.session_id, result
            )

    def _ag_ui_delegated_call(
        self, record: WorkerRecord
    ) -> DelegatedToolCall | None:
        if record.status != "input_required":
            return None
        input_event = next(
            (
                event
                for event in reversed(self.workers.store.events(record.worker_id))
                if event.event_type
                in {
                    "worker_delegated_tool_input_required",
                    "worker_input_required",
                }
            ),
            None,
        )
        if (
            input_event is None
            or input_event.event_type != "worker_delegated_tool_input_required"
        ):
            return None
        call_id = input_event.payload.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise RuntimeError("delegated tool wait event has no call id")
        call = self.runtime.delegated_tools.call(call_id)
        if call is None or call.session_id != record.session_id:
            raise RuntimeError("delegated tool wait event references an unknown call")
        if call.status not in {"pending", "resolved", "failed"}:
            raise ValueError(
                f"delegated tool call cannot resume from {call.status}"
            )
        return call

    @staticmethod
    def _ag_ui_state_context(snapshot: ProtocolStateSnapshot) -> str:
        return (
            "\n\nCurrent AG-UI shared state (exact JSON; this snapshot supersedes "
            "earlier state for the thread):\n"
            + stable_json_dumps(
                {
                    "revision": snapshot.revision,
                    "sha256": snapshot.state_sha256,
                    "state": snapshot.state,
                },
                indent=None,
            )
        )

    def _ag_ui_resume(
        self,
        worker_id: str,
        run: AgUiRunInput,
        *,
        state_context: str = "",
    ) -> WorkerRecord:
        current = self.workers.store.get(worker_id)
        if current.status != "input_required":
            raise ValueError(
                f"AG-UI thread is {current.status}, not awaiting an interrupt response"
            )
        if len(run.resume) != 1:
            raise ValueError("AG-UI requires exactly one response to the open interrupt")
        event = next(
            (
                item
                for item in reversed(self.workers.store.events(worker_id))
                if item.event_type == "worker_input_required"
            ),
            None,
        )
        if event is None:
            raise RuntimeError("AG-UI input-required worker has no durable interrupt event")
        expected_interrupt_id = f"{worker_id}-input-{event.sequence}"
        response = run.resume[0]
        interrupt_id = response.get("interruptId")
        if interrupt_id != expected_interrupt_id:
            raise ValueError("AG-UI resume does not address the open interrupt")
        status = response.get("status")
        if status == "cancelled":
            return self.workers.cancel(
                worker_id,
                reason="AG-UI client canceled the open interrupt",
            )
        if status != "resolved":
            raise ValueError("AG-UI resume status must be resolved or cancelled")
        if "payload" not in response or response["payload"] is None:
            raise ValueError("AG-UI resolved interrupt requires a payload")
        payload = response["payload"]
        answer = payload.strip() if isinstance(payload, str) else stable_json_dumps(
            payload, indent=None
        )
        if not answer:
            raise ValueError("AG-UI resolved interrupt payload must not be empty")
        message = (
            "AG-UI interrupt response:\n"
            + answer
            + run.context_text
            + state_context
        )
        return self.workers.message(
            worker_id,
            message,
            source=f"ag_ui:{run.run_id}",
        )

    def _a2a_subscription_response(
        self,
        initial_record: WorkerRecord,
        current_record: WorkerRecord,
        page: dict[str, Any],
    ) -> dict[str, Any]:
        adapter = A2AProjectionAdapter()
        updates = adapter.updates(
            current_record,
            [
                self.workers.event_from_payload(item)
                for item in page["events"]
            ],
        )
        return {
            "protocol": adapter.protocol_version,
            "worker_id": current_record.worker_id,
            "stream": [{"task": adapter.task(initial_record)}, *updates],
            "next_sequence": page["next_sequence"],
            "has_more": page["has_more"],
            "terminal": page["terminal"],
            "timed_out": page["timed_out"],
        }

    def _a2a_list(self, params: dict[str, Any]) -> dict[str, Any]:
        page_size = params.get("pageSize", params.get("page_size", 50))
        if (
            not isinstance(page_size, int)
            or isinstance(page_size, bool)
            or not 1 <= page_size <= 100
        ):
            raise ValueError("A2A pageSize must be an integer between 1 and 100")
        context_id = params.get("contextId", params.get("context_id"))
        if context_id is not None and (
            not isinstance(context_id, str) or not context_id.strip()
        ):
            raise ValueError("A2A contextId must be a non-empty string when provided")
        state_filter = params.get("status")
        if state_filter is not None and (
            not isinstance(state_filter, str) or not state_filter.strip()
        ):
            raise ValueError("A2A status must be a non-empty string when provided")
        history_length = _a2a_history_length(params, default=0)
        include_artifacts = params.get(
            "includeArtifacts", params.get("include_artifacts", False)
        )
        if not isinstance(include_artifacts, bool):
            raise ValueError("A2A includeArtifacts must be a boolean")
        timestamp_after = params.get(
            "statusTimestampAfter", params.get("status_timestamp_after")
        )
        parsed_timestamp_after = (
            None
            if timestamp_after is None
            else _parse_a2a_timestamp(timestamp_after, "statusTimestampAfter")
        )
        adapter = A2AProjectionAdapter()
        records = self.workers.list()
        if context_id is not None:
            records = [item for item in records if item.session_id == context_id.strip()]
        if state_filter is not None:
            records = [
                item
                for item in records
                if adapter.task(item)["status"]["state"] == state_filter.strip()
            ]
        if parsed_timestamp_after is not None:
            records = [
                item
                for item in records
                if _parse_a2a_timestamp(item.updated_at, "worker updated_at")
                >= parsed_timestamp_after
            ]
        records.sort(key=lambda item: (item.updated_at, item.worker_id), reverse=True)
        total_size = len(records)
        page_token = params.get("pageToken", params.get("page_token", ""))
        if not isinstance(page_token, str):
            raise ValueError("A2A pageToken must be a string")
        if page_token:
            cursor = _decode_a2a_page_token(page_token)
            records = [
                item
                for item in records
                if (item.updated_at, item.worker_id) < cursor
            ]
        page = records[:page_size]
        next_page_token = (
            _encode_a2a_page_token(page[-1])
            if len(records) > len(page) and page
            else ""
        )
        return {
            "protocol": adapter.protocol_version,
            "tasks": [
                self._a2a_task(
                    item,
                    history_length=history_length,
                    include_artifacts=include_artifacts,
                )
                for item in page
            ],
            "totalSize": total_size,
            "pageSize": page_size,
            "nextPageToken": next_page_token,
        }

    def _a2a_send(
        self,
        params: dict[str, Any],
        *,
        wait_for_completion: bool,
    ) -> dict[str, Any]:
        adapter = A2AProjectionAdapter()
        message = adapter.user_message(params)
        record_protocol_correlation(
            protocol="a2a",
            request_id=message.message_id,
            context_id=message.context_id or "",
            worker_id=message.task_id or "",
        )
        if message.task_id is None:
            if message.context_id is not None:
                raise ValueError(
                    "A2A client-provided contextId is not accepted for a new task"
                )
            created = self.task_api.execute(
                "create",
                {
                    "objective": message.text,
                    "attachments": list(message.attachments),
                    "attachment_source": "a2a",
                    "start": True,
                },
            )
            worker_id = str(created["worker"]["worker_id"])
            record = self.workers.store.get(worker_id)
        else:
            worker_id = message.task_id
            record = self.workers.store.get(worker_id)
            if record.status in WORKER_TERMINAL_STATES:
                raise A2AUnsupportedOperationError(
                    f"A2A task {worker_id} cannot accept messages from {record.status}"
                )
            if (
                message.context_id is not None
                and message.context_id != record.session_id
            ):
                raise ValueError("A2A taskId and contextId refer to different contexts")
            for attachment in message.attachments:
                self.task_api.execute(
                    "attachment.add",
                    {**attachment, "worker_id": worker_id, "source": "a2a"},
                )
            record = self.workers.message(
                worker_id,
                message.text,
                source=f"a2a:{message.message_id}",
            )
        if wait_for_completion and not message.return_immediately:
            record = self.workers.wait(worker_id, timeout_seconds=None)
        record_protocol_correlation(
            protocol="a2a",
            request_id=message.message_id,
            context_id=record.session_id,
            worker_id=record.worker_id,
            session_id=record.session_id,
        )
        return {
            "protocol": adapter.protocol_version,
            "task": self._a2a_task(
                record,
                history_length=message.history_length,
            ),
        }

    def _a2a_task(
        self,
        record: WorkerRecord,
        *,
        history_length: int | None = 0,
        include_artifacts: bool = True,
    ) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        if history_length != 0:
            for event in self.runtime.history.read_history(record.session_id):
                if event.event_type != "message_added":
                    continue
                message = event.payload.get("message")
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                content = message.get("content")
                if role not in {"user", "assistant"} or not isinstance(content, str):
                    continue
                history.append(
                    {
                        "messageId": event.id,
                        "contextId": record.session_id,
                        "taskId": record.worker_id,
                        "role": "ROLE_USER" if role == "user" else "ROLE_AGENT",
                        "parts": [{"text": content}],
                        "metadata": {
                            "swaagHistorySequence": event.sequence,
                            "swaagHistoryHash": event.hash,
                        },
                    }
                )
            if history_length is not None:
                history = history[-history_length:]
        return A2AProjectionAdapter().task(
            record,
            history=history,
            include_artifacts=include_artifacts,
        )

    async def _wait_task_events(
        self,
        params: dict[str, Any],
        *,
        input_required_is_terminal: bool = True,
    ) -> dict[str, Any]:
        timeout = params.get("timeout_seconds", 30.0)
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not 0 <= float(timeout) <= 60
        ):
            raise ValueError("timeout_seconds must be between 0 and 60")
        deadline = asyncio.get_running_loop().time() + float(timeout)
        probe = {
            key: value for key, value in params.items() if key != "timeout_seconds"
        }
        while True:
            page = await asyncio.to_thread(
                self.task_api.execute,
                "events",
                probe,
            )
            record = self.workers.store.get(str(params.get("worker_id", "")).strip())
            terminal = record.status in WORKER_TERMINAL_STATES or (
                input_required_is_terminal and record.status == "input_required"
            )
            if terminal:
                # State and its transition event commit together. The first page
                # may have observed only an earlier event immediately before that
                # commit, so always re-read after observing terminal state.
                page = await asyncio.to_thread(
                    self.task_api.execute,
                    "events",
                    probe,
                )
            if page["events"] or terminal:
                return {**page, "terminal": terminal, "timed_out": False}
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return {**page, "terminal": False, "timed_out": True}
            await asyncio.sleep(min(0.05, remaining))

    async def _protocol_projection_async(
        self,
        protocol: str,
        operation: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if protocol == "a2a" and operation == "send":
            message = A2AProjectionAdapter().user_message(params)
            response = await asyncio.to_thread(
                self._a2a_send,
                params,
                wait_for_completion=False,
            )
            if message.return_immediately:
                return response
            worker_id = str(response["task"]["id"])
            while True:
                record = self.workers.store.get(worker_id)
                if (
                    record.status in WORKER_TERMINAL_STATES
                    or record.status == "input_required"
                ):
                    return {
                        "protocol": A2AProjectionAdapter.protocol_version,
                        "task": self._a2a_task(
                            record,
                            history_length=message.history_length,
                        ),
                    }
                await asyncio.sleep(0.05)
        if protocol == "ag_ui" and operation == "subscribe":
            page = await self._wait_task_events(params)
            record = self.workers.store.get(str(params.get("worker_id", "")).strip())
            return {
                "protocol": "ag-ui",
                "worker_id": record.worker_id,
                "events": AgUiProjectionAdapter().events(
                    record,
                    [
                        self.workers.event_from_payload(item)
                        for item in page["events"]
                    ],
                ),
                "next_sequence": page["next_sequence"],
                "has_more": page["has_more"],
                "terminal": page["terminal"],
                "timed_out": page["timed_out"],
            }
        if protocol == "a2a" and operation == "subscribe":
            worker_id = str(params.get("worker_id") or params.get("id") or "").strip()
            initial = self.workers.store.get(worker_id)
            if initial.status in {"completed", "failed", "canceled"}:
                raise ValueError("A2A cannot subscribe to a terminal task")
            page = await self._wait_task_events({**params, "worker_id": worker_id})
            current = self.workers.store.get(worker_id)
            return self._a2a_subscription_response(initial, current, page)
        async with self._semaphore:
            return await asyncio.to_thread(
                self.protocol_projection,
                protocol,
                operation,
                params,
            )

    def _a2a_agent_card(self) -> dict[str, Any]:
        host = self._advertised_host
        if host in {"", "0.0.0.0", "::"}:
            host = "127.0.0.1"
        try:
            if ipaddress.ip_address(host).version == 6:
                host = f"[{host}]"
        except ValueError:
            pass
        port = self._advertised_port
        return {
            "name": "Swaag",
            "description": (
                "Durable autonomous workers with resumable tasks, exact history, "
                "attachments, semantic completion evaluation, and cancellation."
            ),
            "supportedInterfaces": [
                {
                    "url": f"http://{host}:{port}/a2a/v1",
                    "protocolBinding": "JSONRPC",
                    "protocolVersion": A2AProjectionAdapter.protocol_version,
                },
                {
                    "url": f"http://{host}:{port}/a2a/rest",
                    "protocolBinding": "HTTP+JSON",
                    "protocolVersion": A2AProjectionAdapter.protocol_version,
                },
            ],
            "version": "0.1.0",
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
            },
            "defaultInputModes": ["text/plain", "application/octet-stream"],
            "defaultOutputModes": ["text/plain", "application/json"],
            "skills": [
                {
                    "id": "durable-autonomous-work",
                    "name": "Durable autonomous work",
                    "description": (
                        "Start, inspect, redirect, cancel, and stream independently "
                        "addressable long-running workers."
                    ),
                    "tags": ["agent", "durable", "long-running", "tools"],
                }
            ],
        }

    def _ag_ui_capabilities(self) -> dict[str, Any]:
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in self.runtime.tools.enabled_domain_tools(self.runtime.config)
        ]
        state_deltas = any(
            tool.name == "shared_state"
            for tool in self.runtime.tools.enabled_domain_tools(
                self.runtime.config,
                runtime_capabilities={"shared_state": object()},
            )
        )
        return {
            "identity": {
                "name": "Swaag",
                "type": "swaag",
                "description": (
                    "Durable autonomous workers with resumable tasks, exact history, "
                    "attachments, semantic completion evaluation, and cancellation."
                ),
                "version": "0.1.0",
                "provider": "Swaag",
                "documentationUrl": "https://github.com/HansPeterRadtke/swaag",
            },
            "transport": {
                "streaming": True,
                "websocket": False,
                "httpBinary": False,
                "pushNotifications": False,
                "resumable": False,
            },
            "tools": {
                "supported": bool(tools),
                "items": tools,
                "parallelCalls": False,
                "clientProvided": True,
            },
            "output": {
                "structuredOutput": False,
                "supportedMimeTypes": ["text/plain"],
            },
            "state": {
                "snapshots": True,
                "deltas": state_deltas,
                "memory": False,
                "persistentState": True,
            },
            "multimodal": {
                "input": {
                    "image": True,
                    "audio": True,
                    "video": True,
                    "pdf": True,
                    "file": True,
                }
            },
            "execution": {
                "maxIterations": int(self.runtime.config.runtime.max_total_actions)
            },
            "humanInTheLoop": {
                "supported": True,
                "interventions": True,
                "feedback": True,
                "interrupts": True,
            },
        }

    @staticmethod
    async def _write_http_response(
        writer: asyncio.StreamWriter,
        *,
        status: int,
        reason: str,
        body: bytes = b"",
        content_type: str = "application/json",
        headers: dict[str, str] | None = None,
    ) -> None:
        record_http_response_status(status)
        response_headers = {
            "Content-Length": str(len(body)),
            "Content-Type": content_type,
            "Connection": "close",
            "X-Content-Type-Options": "nosniff",
            **(headers or {}),
        }
        head = [f"HTTP/1.1 {status} {reason}"]
        head.extend(f"{key}: {value}" for key, value in response_headers.items())
        writer.write(("\r\n".join(head) + "\r\n\r\n").encode("ascii") + body)
        await writer.drain()

    @classmethod
    async def _write_mcp_http_response(
        cls,
        writer: asyncio.StreamWriter,
        response: McpHttpResponse,
    ) -> None:
        body = (
            b""
            if response.payload is None
            else json.dumps(response.payload, sort_keys=True).encode()
        )
        await cls._write_http_response(
            writer,
            status=response.status,
            reason=HTTPStatus(response.status).phrase,
            body=body,
            headers=dict(response.headers or {}),
        )

    async def _handle_mcp_http(
        self,
        *,
        headers: dict[str, str],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        preflight = self.mcp.http_preflight(headers)
        if preflight is not None:
            await self._write_mcp_http_response(writer, preflight)
            return
        raw_length = headers.get("content-length", "")
        if not raw_length.isdigit() or not 0 < int(raw_length) <= 1_048_576:
            await self._write_mcp_http_response(
                writer,
                McpHttpResponse(
                    400,
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32600,
                            "message": (
                                "Content-Length must be between 1 and 1048576"
                            ),
                        },
                    },
                ),
            )
            return
        try:
            body = await reader.readexactly(int(raw_length))
        except asyncio.IncompleteReadError:
            await self._write_mcp_http_response(
                writer,
                McpHttpResponse(
                    400,
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Incomplete JSON request body",
                        },
                    },
                ),
            )
            return
        try:
            request = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            await self._write_mcp_http_response(
                writer,
                McpHttpResponse(
                    400,
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Invalid JSON payload"},
                    },
                ),
            )
            return
        if not isinstance(request, dict):
            await self._write_mcp_http_response(
                writer,
                McpHttpResponse(
                    400,
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32600,
                            "message": "MCP request must be one JSON object",
                        },
                    },
                ),
            )
            return
        request_id = request.get("id")
        record_protocol_correlation(
            protocol="mcp",
            request_id=(
                str(request_id)
                if isinstance(request_id, (str, int))
                and not isinstance(request_id, bool)
                else ""
            ),
        )
        if request.get("method") == "subscriptions/listen":
            prepared = self.mcp.prepare_http_subscription(request, headers)
            if isinstance(prepared, McpHttpResponse):
                await self._write_mcp_http_response(writer, prepared)
                return
            await self._write_mcp_subscription(writer, prepared)
            return
        try:
            async with self._semaphore:
                response = await asyncio.to_thread(
                    self.mcp.handle_http,
                    request,
                    headers,
                )
        except Exception:
            response = McpHttpResponse(
                500,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": "Internal error"},
                },
            )
        await self._write_mcp_http_response(writer, response)

    async def _write_mcp_subscription(
        self,
        writer: asyncio.StreamWriter,
        subscription: McpHttpSubscription,
    ) -> None:
        record_http_response_status(200)
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: close\r\n"
            b"X-Accel-Buffering: no\r\n"
            b"X-Content-Type-Options: nosniff\r\n\r\n"
        )

        async def emit(message: dict[str, Any]) -> None:
            writer.write(
                ("data: " + json.dumps(message, sort_keys=True) + "\n\n").encode()
            )
            await writer.drain()

        subscription_meta = {
            "io.modelcontextprotocol/subscriptionId": subscription.request_id
        }
        await emit(
            {
                "jsonrpc": "2.0",
                "method": "notifications/subscriptions/acknowledged",
                "params": {
                    "notifications": subscription.honored_filter,
                    "_meta": subscription_meta,
                },
            }
        )
        catalog_sha256 = subscription.initial_tool_catalog_sha256
        keepalive_at = time.monotonic() + 15.0
        server_stopping = False
        try:
            while not writer.is_closing() and not subscription.cancelled.is_set():
                await asyncio.sleep(0.25)
                current_sha256 = await asyncio.to_thread(
                    self.mcp.tool_catalog_sha256
                )
                if (
                    subscription.honored_filter.get("toolsListChanged") is True
                    and current_sha256 != catalog_sha256
                ):
                    catalog_sha256 = current_sha256
                    await emit(
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/tools/list_changed",
                            "params": {"_meta": subscription_meta},
                        }
                    )
                if time.monotonic() >= keepalive_at:
                    writer.write(b": keepalive\n\n")
                    await writer.drain()
                    keepalive_at = time.monotonic() + 15.0
        except asyncio.CancelledError:
            server_stopping = True
            raise
        finally:
            self.mcp.finish_http_subscription(subscription.request_id)
            if (
                server_stopping
                and not writer.is_closing()
                and not subscription.cancelled.is_set()
            ):
                try:
                    await emit(
                        {
                            "jsonrpc": "2.0",
                            "id": subscription.request_id,
                            "result": {
                                "resultType": "complete",
                                "_meta": {
                                    **subscription_meta,
                                    "io.modelcontextprotocol/serverInfo": {
                                        "name": "swaag",
                                        "version": "0.1",
                                    },
                                },
                            },
                        }
                    )
                except (ConnectionError, BrokenPipeError):
                    pass

    @staticmethod
    def _a2a_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    @classmethod
    def _a2a_exception_payload(
        cls,
        request_id: Any,
        exc: Exception,
    ) -> dict[str, Any]:
        if isinstance(exc, A2AProtocolError):
            return cls._a2a_error(request_id, exc.jsonrpc_code, str(exc))
        if isinstance(exc, FileNotFoundError):
            return cls._a2a_error(request_id, -32001, str(exc))
        if isinstance(exc, (TypeError, ValueError)):
            return cls._a2a_error(request_id, -32602, str(exc))
        return cls._a2a_error(request_id, -32603, "Internal error")

    @staticmethod
    def _a2a_rest_error(
        *,
        status: int,
        status_name: str,
        message: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        details: list[dict[str, Any]] = []
        if reason is not None:
            details.append(
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": reason,
                    "domain": "a2a-protocol.org",
                }
            )
        elif status == 400:
            details.append(
                {
                    "@type": "type.googleapis.com/google.rpc.BadRequest",
                    "fieldViolations": [
                        {"field": "request", "description": message}
                    ],
                }
            )
        return {
            "error": {
                "code": status,
                "status": status_name,
                "message": message,
                "details": details,
            }
        }

    @classmethod
    def _a2a_rest_exception(
        cls, exc: Exception
    ) -> tuple[int, str, dict[str, Any]]:
        if isinstance(exc, FileNotFoundError):
            status, status_name, reason = 404, "NOT_FOUND", "TASK_NOT_FOUND"
        elif isinstance(exc, A2ATaskNotCancelableError):
            status, status_name, reason = (
                400,
                "FAILED_PRECONDITION",
                "TASK_NOT_CANCELABLE",
            )
        elif isinstance(exc, A2AUnsupportedOperationError):
            status, status_name, reason = (
                400,
                "FAILED_PRECONDITION",
                "UNSUPPORTED_OPERATION",
            )
        elif isinstance(exc, A2AContentTypeNotSupportedError):
            status, status_name, reason = (
                400,
                "INVALID_ARGUMENT",
                "CONTENT_TYPE_NOT_SUPPORTED",
            )
        elif isinstance(exc, (TypeError, ValueError)):
            status, status_name, reason = 400, "INVALID_ARGUMENT", None
        else:
            status, status_name, reason = 500, "INTERNAL", None
        message = str(exc) if status < 500 else "Internal error"
        return (
            status,
            HTTPStatus(status).phrase,
            cls._a2a_rest_error(
                status=status,
                status_name=status_name,
                message=message,
                reason=reason,
            ),
        )

    async def _read_http_headers(
        self,
        reader: asyncio.StreamReader,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        total = 0
        while True:
            line = await reader.readline()
            total += len(line)
            if total > 65_536:
                raise ValueError("HTTP headers exceed 65536 bytes")
            if line in {b"\r\n", b"\n", b""}:
                break
            try:
                name, value = line.decode("ascii").split(":", 1)
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError("malformed HTTP header") from exc
            normalized = name.strip().casefold()
            if not normalized or normalized in headers:
                raise ValueError("duplicate or empty HTTP header")
            headers[normalized] = value.strip()
        return headers

    async def _write_a2a_sse(
        self,
        writer: asyncio.StreamWriter,
        *,
        request_id: Any,
        initial: WorkerRecord,
        after_sequence: int,
        history_length: int | None = 0,
        jsonrpc_envelope: bool = True,
    ) -> None:
        record_http_response_status(200)
        headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "X-Content-Type-Options: nosniff\r\n\r\n"
        )
        writer.write(headers.encode("ascii"))

        async def emit(result: dict[str, Any]) -> None:
            payload = (
                {"jsonrpc": "2.0", "id": request_id, "result": result}
                if jsonrpc_envelope
                else result
            )
            writer.write(("data: " + json.dumps(payload, sort_keys=True) + "\n\n").encode())
            await writer.drain()

        adapter = A2AProjectionAdapter()
        await emit(
            {
                "task": self._a2a_task(
                    initial,
                    history_length=history_length,
                )
            }
        )
        cursor = after_sequence
        while not writer.is_closing():
            page = await self._wait_task_events(
                {
                    "worker_id": initial.worker_id,
                    "after_sequence": cursor,
                    "limit": 100,
                    "timeout_seconds": 30,
                },
                input_required_is_terminal=False,
            )
            current = self.workers.store.get(initial.worker_id)
            updates = adapter.updates(
                current,
                [
                    self.workers.event_from_payload(item)
                    for item in page["events"]
                ],
            )
            for update in updates:
                await emit(update)
            cursor = int(page["next_sequence"])
            if page["terminal"] and not page["has_more"]:
                break
            if not updates:
                writer.write(b": keepalive\n\n")
                await writer.drain()

    async def _write_ag_ui_sse(
        self,
        writer: asyncio.StreamWriter,
        *,
        run: AgUiRunInput,
        record: WorkerRecord,
        start_sequence: int,
        end_sequence: int | None,
        duplicate: bool,
        state_snapshot: ProtocolStateSnapshot,
    ) -> None:
        record_http_response_status(200)
        headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n"
            "X-Accel-Buffering: no\r\n"
            "X-Content-Type-Options: nosniff\r\n\r\n"
        )
        writer.write(headers.encode("ascii"))

        async def emit(event: dict[str, Any]) -> None:
            writer.write(
                ("data: " + json.dumps(event, sort_keys=True) + "\n\n").encode()
            )
            await writer.drain()

        started: dict[str, Any] = {
            "type": "RUN_STARTED",
            "threadId": run.thread_id,
            "runId": run.run_id,
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "metadata": {
                "swaagWorkerId": record.worker_id,
                "swaagDuplicateRun": duplicate,
            },
        }
        if run.parent_run_id is not None:
            started["parentRunId"] = run.parent_run_id
        await emit(started)
        await emit(
            {
                "type": "STATE_SNAPSHOT",
                "snapshot": state_snapshot.state,
                "timestamp": int(
                    datetime.fromisoformat(state_snapshot.created_at).timestamp() * 1000
                ),
                "metadata": {
                    "swaagStateRevision": state_snapshot.revision,
                    "swaagStateSha256": state_snapshot.state_sha256,
                    "swaagClientSupplied": state_snapshot.client_supplied,
                },
            }
        )

        adapter = AgUiProjectionAdapter()
        cursor = start_sequence
        terminal_emitted = False
        while not writer.is_closing():
            if end_sequence is None:
                bounds = await asyncio.to_thread(
                    self.store.protocol_message_bounds,
                    "ag_ui",
                    run.run_id,
                )
                if bounds is not None:
                    end_sequence = bounds[3]
            if end_sequence is not None and cursor >= end_sequence:
                break
            page = await self._wait_task_events(
                {
                    "worker_id": record.worker_id,
                    "after_sequence": cursor,
                    "limit": 100,
                    "timeout_seconds": 30,
                }
            )
            if end_sequence is None:
                bounds = await asyncio.to_thread(
                    self.store.protocol_message_bounds,
                    "ag_ui",
                    run.run_id,
                )
                if bounds is not None:
                    end_sequence = bounds[3]
            raw_events = page["events"]
            if end_sequence is not None:
                raw_events = [
                    item
                    for item in raw_events
                    if int(item["sequence"]) <= end_sequence
                ]
            current = self.workers.store.get(record.worker_id)
            projected = adapter.events(
                current,
                [self.workers.event_from_payload(item) for item in raw_events],
                thread_id=run.thread_id,
                run_id=run.run_id,
                state_baseline_revision=state_snapshot.revision,
            )
            projected = [
                item for item in projected if item.get("type") != "RUN_STARTED"
            ]
            for event in projected:
                await emit(event)
                if event.get("type") in {"RUN_FINISHED", "RUN_ERROR"}:
                    terminal_emitted = True
            if raw_events:
                cursor = int(raw_events[-1]["sequence"])
            elif end_sequence is None:
                cursor = int(page["next_sequence"])

            reached_bound = end_sequence is not None and (
                cursor >= end_sequence
                or int(page["next_sequence"]) >= end_sequence
            )
            if reached_bound:
                cursor = end_sequence
                break
            if page["terminal"] and not page["has_more"]:
                self.store.finish_protocol_message(
                    "ag_ui",
                    run.run_id,
                    end_sequence=cursor,
                )
                if not terminal_emitted:
                    await emit(
                        {
                            "type": "RUN_ERROR",
                            "message": (
                                "Swaag reached a durable terminal state without "
                                "a projectable terminal event"
                            ),
                            "code": "SWAAG_TERMINAL_EVENT_MISSING",
                        }
                    )
                return
            if not projected:
                writer.write(b": keepalive\n\n")
                await writer.drain()

        if end_sequence is not None and not terminal_emitted:
            await emit(
                {
                    "type": "RUN_ERROR",
                    "message": "This AG-UI run was superseded by a newer thread run",
                    "code": "SWAAG_RUN_SUPERSEDED",
                }
            )

    async def _handle_ag_ui_http(
        self,
        *,
        request: dict[str, Any],
        headers: dict[str, str],
        writer: asyncio.StreamWriter,
    ) -> None:
        accept = headers.get("accept", "*/*")
        if "text/event-stream" not in accept and "*/*" not in accept:
            await self._write_http_response(
                writer,
                status=406,
                reason="Not Acceptable",
                body=b'{"error":"Accept must allow text/event-stream"}',
            )
            return
        try:
            run = AgUiProjectionAdapter().user_run(request)
            record, start_sequence, end_sequence, duplicate, state_snapshot = (
                await asyncio.to_thread(self._ag_ui_begin, run)
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            await self._write_http_response(
                writer,
                status=400,
                reason="Bad Request",
                body=json.dumps({"error": str(exc)}, sort_keys=True).encode(),
            )
            return
        except Exception:
            await self._write_http_response(
                writer,
                status=500,
                reason="Internal Server Error",
                body=b'{"error":"AG-UI run initialization failed"}',
            )
            return
        try:
            await self._write_ag_ui_sse(
                writer,
                run=run,
                record=record,
                start_sequence=start_sequence,
                end_sequence=end_sequence,
                duplicate=duplicate,
                state_snapshot=state_snapshot,
            )
        except (ConnectionError, BrokenPipeError):
            pass
        except asyncio.CancelledError:
            if not writer.is_closing():
                try:
                    writer.write(
                        b'data: {"code":"SWAAG_SERVICE_STOPPING",'
                        b'"message":"Swaag communication service is stopping",'
                        b'"type":"RUN_ERROR"}\n\n'
                    )
                    await writer.drain()
                except (ConnectionError, BrokenPipeError):
                    pass
            raise
        except Exception as exc:
            if not writer.is_closing():
                payload = {
                    "type": "RUN_ERROR",
                    "message": f"{type(exc).__name__}: {exc}",
                    "code": "SWAAG_AG_UI_TRANSPORT_ERROR",
                }
                try:
                    writer.write(
                        (
                            "data: "
                            + json.dumps(payload, sort_keys=True)
                            + "\n\n"
                        ).encode()
                    )
                    await writer.drain()
                except (ConnectionError, BrokenPipeError):
                    pass

    async def _handle_a2a_http(
        self,
        *,
        request: dict[str, Any],
        headers: dict[str, str],
        writer: asyncio.StreamWriter,
    ) -> None:
        request_id = request.get("id")
        record_protocol_correlation(
            protocol="a2a",
            request_id=str(request_id) if request_id is not None else "",
        )
        if headers.get("a2a-version") != A2AProjectionAdapter.protocol_version:
            payload = self._a2a_error(
                request_id,
                -32009,
                "Version not supported; send A2A-Version: 1.0",
            )
            await self._write_http_response(
                writer,
                status=200,
                reason="OK",
                body=json.dumps(payload, sort_keys=True).encode(),
            )
            return
        if request.get("jsonrpc") != "2.0" or request_id is None:
            payload = self._a2a_error(request_id, -32600, "Invalid request")
        else:
            method = request.get("method")
            params = request.get("params") or {}
            if not isinstance(method, str) or not isinstance(params, dict):
                payload = self._a2a_error(request_id, -32600, "Invalid request")
            else:
                operation_by_method = {
                    "SendMessage": "send",
                    "GetTask": "get",
                    "ListTasks": "list",
                    "CancelTask": "cancel",
                }
                if method in {"SendStreamingMessage", "SubscribeToTask"}:
                    try:
                        if method == "SendStreamingMessage":
                            parsed_message = A2AProjectionAdapter().user_message(params)
                            response = await asyncio.to_thread(
                                self._a2a_send,
                                params,
                                wait_for_completion=False,
                            )
                            initial, cursor = self.workers.stream_snapshot(
                                str(response["task"]["id"])
                            )
                            history_length = parsed_message.history_length
                        else:
                            worker_id = str(params.get("id", "")).strip()
                            initial, cursor = self.workers.stream_snapshot(worker_id)
                            if initial.status in WORKER_TERMINAL_STATES:
                                raise A2AUnsupportedOperationError(
                                    "A2A cannot subscribe to a terminal task"
                                )
                            history_length = 0
                    except Exception as exc:
                        error = self._a2a_exception_payload(request_id, exc)
                        await self._write_http_response(
                            writer,
                            status=200,
                            reason="OK",
                            body=json.dumps(error, sort_keys=True).encode(),
                        )
                        return
                    try:
                        await self._write_a2a_sse(
                            writer,
                            request_id=request_id,
                            initial=initial,
                            after_sequence=cursor,
                            history_length=history_length,
                        )
                    except (ConnectionError, BrokenPipeError):
                        pass
                    return
                operation = operation_by_method.get(method)
                if operation is None:
                    payload = self._a2a_error(request_id, -32601, "Method not found")
                else:
                    try:
                        response = await self._protocol_projection_async(
                            "a2a", operation, params
                        )
                        if operation == "send":
                            result: Any = {"task": response["task"]}
                        elif operation in {"get", "cancel"}:
                            result = response["task"]
                        else:
                            result = {
                                key: value
                                for key, value in response.items()
                                if key != "protocol"
                            }
                        payload = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": result,
                        }
                    except Exception as exc:
                        payload = self._a2a_exception_payload(request_id, exc)
        await self._write_http_response(
            writer,
            status=200,
            reason="OK",
            body=json.dumps(payload, sort_keys=True).encode(),
        )

    async def _handle_a2a_rest_http(
        self,
        *,
        method: str,
        path: str,
        query: str,
        request: dict[str, Any] | None,
        headers: dict[str, str],
        writer: asyncio.StreamWriter,
    ) -> None:
        record_protocol_correlation(protocol="a2a", request_id="")
        if headers.get("a2a-version") != A2AProjectionAdapter.protocol_version:
            payload = self._a2a_rest_error(
                status=400,
                status_name="FAILED_PRECONDITION",
                message="Version not supported; send A2A-Version: 1.0",
                reason="VERSION_NOT_SUPPORTED",
            )
            await self._write_http_response(
                writer,
                status=400,
                reason="Bad Request",
                body=json.dumps(payload, sort_keys=True).encode(),
                content_type="application/a2a+json",
            )
            return

        stream: tuple[WorkerRecord, int, int | None] | None = None
        response_body: dict[str, Any] | None = None
        try:
            if method == "POST" and path == "/message:send":
                if request is None:
                    raise ValueError("A2A SendMessage requires a JSON request object")
                response = await self._protocol_projection_async(
                    "a2a", "send", request
                )
                response_body = {"task": response["task"]}
            elif method == "POST" and path == "/message:stream":
                if request is None:
                    raise ValueError(
                        "A2A SendStreamingMessage requires a JSON request object"
                    )
                parsed_message = A2AProjectionAdapter().user_message(request)
                response = await asyncio.to_thread(
                    self._a2a_send,
                    request,
                    wait_for_completion=False,
                )
                initial, cursor = self.workers.stream_snapshot(
                    str(response["task"]["id"])
                )
                stream = (initial, cursor, parsed_message.history_length)
            elif method == "GET" and path == "/tasks":
                params = _a2a_rest_query_params(
                    query,
                    allowed={
                        "contextId",
                        "status",
                        "pageSize",
                        "pageToken",
                        "historyLength",
                        "statusTimestampAfter",
                        "includeArtifacts",
                    },
                )
                response = await self._protocol_projection_async(
                    "a2a", "list", params
                )
                response_body = {
                    key: value
                    for key, value in response.items()
                    if key != "protocol"
                }
            elif method == "GET" and path.startswith("/tasks/"):
                task_id, operation = _a2a_rest_task_path(path)
                if operation == "subscribe":
                    if query:
                        _a2a_rest_query_params(query, allowed=set())
                    initial, cursor = self.workers.stream_snapshot(task_id)
                    if initial.status in WORKER_TERMINAL_STATES:
                        raise A2AUnsupportedOperationError(
                            "A2A cannot subscribe to a terminal task"
                        )
                    stream = (initial, cursor, 0)
                elif operation is None:
                    params = _a2a_rest_query_params(
                        query, allowed={"historyLength"}
                    )
                    response = await self._protocol_projection_async(
                        "a2a", "get", {**params, "id": task_id}
                    )
                    response_body = response["task"]
                else:
                    raise ValueError("A2A REST task path does not support this method")
            elif method == "POST" and path.startswith("/tasks/"):
                task_id, operation = _a2a_rest_task_path(path)
                if operation == "subscribe":
                    if query:
                        _a2a_rest_query_params(query, allowed=set())
                    initial, cursor = self.workers.stream_snapshot(task_id)
                    if initial.status in WORKER_TERMINAL_STATES:
                        raise A2AUnsupportedOperationError(
                            "A2A cannot subscribe to a terminal task"
                        )
                    stream = (initial, cursor, 0)
                elif operation == "cancel":
                    if query:
                        _a2a_rest_query_params(query, allowed=set())
                    response = await self._protocol_projection_async(
                        "a2a", "cancel", {"id": task_id}
                    )
                    response_body = response["task"]
                else:
                    raise ValueError("A2A REST task path does not support this method")
            else:
                payload = self._a2a_rest_error(
                    status=404,
                    status_name="NOT_FOUND",
                    message="A2A HTTP+JSON route not found",
                )
                await self._write_http_response(
                    writer,
                    status=404,
                    reason="Not Found",
                    body=json.dumps(payload, sort_keys=True).encode(),
                    content_type="application/a2a+json",
                )
                return
        except Exception as exc:
            status, reason, payload = self._a2a_rest_exception(exc)
            await self._write_http_response(
                writer,
                status=status,
                reason=reason,
                body=json.dumps(payload, sort_keys=True).encode(),
                content_type="application/a2a+json",
            )
            return

        if stream is not None:
            initial, cursor, history_length = stream
            try:
                await self._write_a2a_sse(
                    writer,
                    request_id=None,
                    initial=initial,
                    after_sequence=cursor,
                    history_length=history_length,
                    jsonrpc_envelope=False,
                )
            except (ConnectionError, BrokenPipeError):
                pass
            return
        if response_body is None:
            raise RuntimeError("A2A HTTP+JSON route produced no response")
        await self._write_http_response(
            writer,
            status=200,
            reason="OK",
            body=json.dumps(response_body, sort_keys=True).encode(),
            content_type="application/a2a+json",
        )

    async def _handle_http_client(
        self,
        first_line: bytes,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            parts = first_line.decode("ascii").strip().split()
            if len(parts) != 3 or parts[2] not in {"HTTP/1.0", "HTTP/1.1"}:
                raise ValueError("malformed HTTP request line")
            method, request_target, _version = parts
            target = urlsplit(request_target)
            if (
                target.scheme
                or target.netloc
                or target.fragment
                or not target.path.startswith("/")
            ):
                raise ValueError("HTTP request target must be an origin-form path")
            path = target.path
            query = target.query
            headers = await self._read_http_headers(reader)
        except (asyncio.IncompleteReadError, ValueError) as exc:
            await self._write_http_response(
                writer,
                status=400,
                reason="Bad Request",
                body=json.dumps({"error": str(exc)}, sort_keys=True).encode(),
            )
            return

        with self.runtime.telemetry.http_server_request(
            method=method,
            path=request_target,
            headers=headers,
        ):
            is_a2a_rest = path == "/a2a/rest" or path.startswith("/a2a/rest/")
            try:
                if path == "/mcp":
                    http_enabled = self.runtime.config.mcp.enabled and (
                        self.runtime.config.mcp.transport
                        in {"streamable_http", "both"}
                    )
                    if not http_enabled:
                        await self._write_http_response(
                            writer,
                            status=404,
                            reason="Not Found",
                            body=b'{"error":"not found"}',
                        )
                    else:
                        origin_response = self.mcp.http_origin_preflight(headers)
                        if origin_response is not None:
                            await self._write_mcp_http_response(
                                writer, origin_response
                            )
                            return
                    if http_enabled and method != "POST":
                        await self._write_http_response(
                            writer,
                            status=405,
                            reason="Method Not Allowed",
                            body=b'{"error":"MCP Streamable HTTP accepts POST only"}',
                            headers={"Allow": "POST"},
                        )
                    elif http_enabled:
                        await self._handle_mcp_http(
                            headers=headers,
                            reader=reader,
                            writer=writer,
                        )
                    return
                if method == "GET" and path == "/.well-known/agent-card.json":
                    body = json.dumps(self._a2a_agent_card(), sort_keys=True).encode()
                    etag = '"' + hashlib.sha256(body).hexdigest() + '"'
                    if headers.get("if-none-match") == etag:
                        await self._write_http_response(
                            writer,
                            status=304,
                            reason="Not Modified",
                            headers={"Cache-Control": "public, max-age=300", "ETag": etag},
                        )
                    else:
                        await self._write_http_response(
                            writer,
                            status=200,
                            reason="OK",
                            body=body,
                            headers={"Cache-Control": "public, max-age=300", "ETag": etag},
                        )
                    return
                if path == "/ag-ui/capabilities":
                    if method != "GET":
                        await self._write_http_response(
                            writer,
                            status=405,
                            reason="Method Not Allowed",
                            body=b'{"error":"AG-UI capability discovery accepts GET only"}',
                            headers={"Allow": "GET"},
                        )
                    else:
                        await self._write_http_response(
                            writer,
                            status=200,
                            reason="OK",
                            body=json.dumps(
                                self._ag_ui_capabilities(), sort_keys=True
                            ).encode(),
                            headers={"Cache-Control": "no-store"},
                        )
                    return
                is_jsonrpc = method == "POST" and path == "/a2a/v1"
                is_ag_ui = method == "POST" and path == "/ag-ui"
                if not (is_jsonrpc or is_ag_ui or is_a2a_rest):
                    await self._write_http_response(
                        writer,
                        status=404,
                        reason="Not Found",
                        body=b'{"error":"not found"}',
                    )
                    return
                rest_path = path.removeprefix("/a2a/rest") if is_a2a_rest else ""
                requires_body = is_jsonrpc or is_ag_ui or (
                    method == "POST"
                    and rest_path in {"/message:send", "/message:stream"}
                )
                request: dict[str, Any] | None = None
                raw_length = headers.get("content-length", "")
                if requires_body:
                    allowed_content_types = (
                        {"application/json", "application/a2a+json"}
                        if is_a2a_rest
                        else {"application/json"}
                    )
                    content_type = (
                        headers.get("content-type", "").split(";", 1)[0].strip()
                    )
                    if content_type not in allowed_content_types:
                        expected = " or ".join(sorted(allowed_content_types))
                        error = f"Content-Type must be {expected}"
                        if is_a2a_rest:
                            raise A2AContentTypeNotSupportedError(error)
                        raise ValueError(error)
                    if (
                        not raw_length.isdigit()
                        or not 0 < int(raw_length) <= 1_048_576
                    ):
                        raise ValueError(
                            "Content-Length must be between 1 and 1048576"
                        )
                    body = await reader.readexactly(int(raw_length))
                    try:
                        decoded = json.loads(body)
                    except json.JSONDecodeError:
                        if is_jsonrpc:
                            payload = self._a2a_error(
                                None, -32700, "Invalid JSON payload"
                            )
                            status, reason, response_type = 200, "OK", "application/json"
                        elif is_a2a_rest:
                            payload = self._a2a_rest_error(
                                status=400,
                                status_name="INVALID_ARGUMENT",
                                message="Invalid JSON payload",
                            )
                            status, reason, response_type = (
                                400,
                                "Bad Request",
                                "application/a2a+json",
                            )
                        else:
                            payload = {"error": "Invalid JSON payload"}
                            status, reason, response_type = (
                                400,
                                "Bad Request",
                                "application/json",
                            )
                        await self._write_http_response(
                            writer,
                            status=status,
                            reason=reason,
                            body=json.dumps(payload, sort_keys=True).encode(),
                            content_type=response_type,
                        )
                        return
                    if not isinstance(decoded, dict):
                        if is_jsonrpc:
                            payload = self._a2a_error(None, -32600, "Invalid request")
                            status, reason, response_type = 200, "OK", "application/json"
                        elif is_a2a_rest:
                            payload = self._a2a_rest_error(
                                status=400,
                                status_name="INVALID_ARGUMENT",
                                message="A2A request must be an object",
                            )
                            status, reason, response_type = (
                                400,
                                "Bad Request",
                                "application/a2a+json",
                            )
                        else:
                            payload = {"error": "AG-UI request must be an object"}
                            status, reason, response_type = (
                                400,
                                "Bad Request",
                                "application/json",
                            )
                        await self._write_http_response(
                            writer,
                            status=status,
                            reason=reason,
                            body=json.dumps(payload, sort_keys=True).encode(),
                            content_type=response_type,
                        )
                        return
                    request = decoded
                elif raw_length not in {"", "0"}:
                    raise ValueError("This A2A HTTP+JSON operation does not accept a body")

                if is_ag_ui:
                    if request is None:
                        raise RuntimeError("AG-UI request body was not decoded")
                    await self._handle_ag_ui_http(
                        request=request,
                        headers=headers,
                        writer=writer,
                    )
                    return
                if is_jsonrpc:
                    if request is None:
                        raise RuntimeError("A2A JSON-RPC request body was not decoded")
                    await self._handle_a2a_http(
                        request=request,
                        headers=headers,
                        writer=writer,
                    )
                    return
                await self._handle_a2a_rest_http(
                    method=method,
                    path=rest_path,
                    query=query,
                    request=request,
                    headers=headers,
                    writer=writer,
                )
            except (asyncio.IncompleteReadError, ValueError) as exc:
                if is_a2a_rest:
                    status, reason, payload = self._a2a_rest_exception(exc)
                    await self._write_http_response(
                        writer,
                        status=status,
                        reason=reason,
                        body=json.dumps(payload, sort_keys=True).encode(),
                        content_type="application/a2a+json",
                    )
                    return
                await self._write_http_response(
                    writer,
                    status=400,
                    reason="Bad Request",
                    body=json.dumps({"error": str(exc)}, sort_keys=True).encode(),
                )

    async def _dispatch_json_line_request(self, request: dict[str, Any]) -> Any:
        op = str(request.get("op", ""))
        with self.runtime.telemetry.protocol_server_request(
            protocol="swaag.jsonl",
            operation=op,
            carrier=request.get("trace_context"),
        ):
            if op == "submit":
                item = self.submit(
                    request.get("session"),
                    str(request.get("message", "")),
                    source=str(request.get("source", "communication")),
                )
                return asdict(item)
            if op == "status":
                return asdict(
                    self.status(str(request.get("correlation_id", "")))
                )
            if op == "process":
                item = await self.process_once_async(
                    session_id=request.get("session_id")
                )
                return None if item is None else asdict(item)
            if op == "ask_status":
                async with self._semaphore:
                    answer = await asyncio.to_thread(
                        self.answer_status_question,
                        request.get("session"),
                        str(
                            request.get(
                                "question", "What is the current status?"
                            )
                        ),
                    )
                return {"answer": answer}
            if op.startswith("task."):
                params = request.get("params") or {}
                if not isinstance(params, dict):
                    raise ValueError("task operation params must be an object")
                task_operation = op.removeprefix("task.")
                if task_operation == "events.wait":
                    return await self._wait_task_events(params)
                async with self._semaphore:
                    return await asyncio.to_thread(
                        self.task_api.execute,
                        task_operation,
                        params,
                    )
            if op.startswith(("ag_ui.", "a2a.", "open_webui.")):
                params = request.get("params") or {}
                if not isinstance(params, dict):
                    raise ValueError("protocol operation params must be an object")
                protocol, operation = op.split(".", 1)
                return await self._protocol_projection_async(
                    protocol,
                    operation,
                    params,
                )
            raise ValueError(f"unknown communication op: {op}")

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            request_parts = line.rstrip(b"\r\n").split(b" ")
            if (
                len(request_parts) == 3
                and request_parts[2] in {b"HTTP/1.0", b"HTTP/1.1"}
            ):
                await self._handle_http_client(line, reader, writer)
                return
            while line and not reader.at_eof():
                if not line:
                    break
                try:
                    request = json.loads(line.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("request must be an object")
                    response = await self._dispatch_json_line_request(request)
                    payload = {"ok": True, "result": response}
                except Exception as exc:
                    payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                writer.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
                await writer.drain()
                line = await reader.readline()
        finally:
            writer.close()
            await writer.wait_closed()

    async def _watchdog_loop(self) -> None:
        interval = watchdog_interval_seconds(default_seconds=10.0)
        while True:
            systemd_notify("WATCHDOG=1", "STATUS=swaag communication service healthy")
            await asyncio.sleep(interval)

    async def _wakeup_loop(self) -> None:
        from swaag.wakeup_dispatcher import dispatch_once

        while True:
            try:
                await asyncio.to_thread(
                    dispatch_once,
                    self.runtime.config,
                    runtime=self.runtime,
                    workers=self.workers,
                )
            except Exception as exc:
                print(
                    f"swaag wakeup dispatch failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            await asyncio.sleep(1.0)

    async def serve_tcp(self, host: str, port: int) -> None:
        host = require_loopback_bind_host(host)
        loop = asyncio.get_running_loop()
        serving_task = asyncio.current_task()
        stop_requested = False
        registered_signals: list[signal.Signals] = []
        server: asyncio.Server | None = None
        background_tasks: list[asyncio.Task[None]] = []

        def request_stop() -> None:
            nonlocal stop_requested
            stop_requested = True
            if serving_task is not None:
                serving_task.cancel()

        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(signum, request_stop)
            except (NotImplementedError, RuntimeError):
                continue
            registered_signals.append(signum)
        try:
            self.workers.reconcile_orphans()
            server = await asyncio.start_server(self.handle_client, host, port)
            bound_port = (
                int(server.sockets[0].getsockname()[1])
                if server.sockets
                else int(port)
            )
            self._advertised_host = host
            self._advertised_port = bound_port
            systemd_notify(
                "READY=1",
                f"STATUS=swaag communication listening on {host}:{bound_port}",
            )
            background_tasks = [
                asyncio.create_task(
                    self._watchdog_loop(), name="swaag-systemd-watchdog"
                ),
                asyncio.create_task(
                    self._wakeup_loop(), name="swaag-wakeup-dispatcher"
                ),
            ]
            async with server:
                await server.serve_forever()
        except asyncio.CancelledError:
            if not stop_requested:
                raise
        finally:
            for signum in registered_signals:
                loop.remove_signal_handler(signum)
            if server is not None:
                server.close()
                await server.wait_closed()
            for task in background_tasks:
                task.cancel()
            for task in background_tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self.workers.shutdown(wait=False)
            systemd_notify("STOPPING=1", "STATUS=swaag communication stopping")


def _encode_a2a_page_token(record: WorkerRecord) -> str:
    payload = json.dumps(
        [record.updated_at, record.worker_id],
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _a2a_history_length(
    payload: dict[str, Any],
    *,
    default: int | None = None,
) -> int | None:
    value = payload.get("historyLength", payload.get("history_length", default))
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError("A2A historyLength must be a non-negative integer")
    return value


def _a2a_rest_query_params(
    query: str,
    *,
    allowed: set[str],
) -> dict[str, Any]:
    parsed = parse_qs(
        query,
        keep_blank_values=True,
        strict_parsing=True,
        max_num_fields=32,
    )
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(
            "A2A HTTP+JSON query contains unsupported fields: "
            + ", ".join(unknown)
        )
    duplicated = sorted(name for name, values in parsed.items() if len(values) != 1)
    if duplicated:
        raise ValueError(
            "A2A HTTP+JSON query fields must be singular: "
            + ", ".join(duplicated)
        )
    result: dict[str, Any] = {}
    for name, values in parsed.items():
        value = values[0]
        if name in {"pageSize", "historyLength"}:
            if not value.isdigit():
                raise ValueError(f"A2A {name} must be a non-negative integer")
            result[name] = int(value)
        elif name == "includeArtifacts":
            if value not in {"true", "false"}:
                raise ValueError("A2A includeArtifacts must be true or false")
            result[name] = value == "true"
        else:
            result[name] = value
    return result


def _a2a_rest_task_path(path: str) -> tuple[str, str | None]:
    encoded = path.removeprefix("/tasks/")
    operation: str | None = None
    for suffix, candidate in ((":cancel", "cancel"), (":subscribe", "subscribe")):
        if encoded.endswith(suffix):
            encoded = encoded[: -len(suffix)]
            operation = candidate
            break
    try:
        task_id = unquote(encoded, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("A2A task id is not valid UTF-8") from exc
    if not task_id or any(character in task_id for character in {"/", "\\", "\x00"}):
        raise ValueError("A2A task id must be one non-empty path segment")
    return task_id, operation


def _required_protocol_text(
    payload: dict[str, Any],
    key: str,
    *,
    protocol: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{protocol} {key} must be a non-empty string")
    return value.strip()


def _parse_a2a_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"A2A {field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"A2A {field} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"A2A {field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _decode_a2a_page_token(value: str) -> tuple[str, str]:
    try:
        padding = "=" * (-len(value) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode(value + padding).decode("utf-8")
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error) as exc:
        raise ValueError("A2A pageToken is invalid") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or any(not isinstance(item, str) or not item for item in payload)
    ):
        raise ValueError("A2A pageToken is invalid")
    return payload[0], payload[1]
