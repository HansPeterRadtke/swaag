from __future__ import annotations

import json
import os
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from swaag.sqlite_schema import apply_sqlite_migrations
from swaag.types import HistoryEvent


_ARCHIVE_CATALOG_MIGRATIONS = (
    (
        """
        CREATE TABLE IF NOT EXISTS archived_sessions (
            session_id TEXT PRIMARY KEY,
            session_name TEXT NOT NULL,
            shard_path TEXT NOT NULL,
            event_count INTEGER NOT NULL,
            archived_at TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS archived_sessions_name
        ON archived_sessions(session_name)
        """,
    ),
)


@dataclass(slots=True, frozen=True)
class ArchiveEntry:
    session_id: str
    session_name: str
    shard_path: str
    event_count: int
    archived_at: str


class HistoryArchiveStore:
    """Immutable per-session SQLite shards. Exact source events remain authoritative."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        self.archive_root = self.root / "archives"
        self.catalog_path = self.archive_root / "catalog.sqlite3"
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self._init_catalog()

    def _catalog(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.catalog_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_catalog(self) -> None:
        with self._catalog() as connection:
            apply_sqlite_migrations(
                connection,
                store_name="history archive catalog",
                migrations=_ARCHIVE_CATALOG_MIGRATIONS,
            )

    def archive_events(self, session_id: str, session_name: str, events: Iterable[HistoryEvent]) -> ArchiveEntry:
        events = list(events)
        if not events:
            raise ValueError("Cannot archive a session with no events")
        now = datetime.now(timezone.utc)
        month_dir = self.archive_root / now.strftime("%Y-%m")
        month_dir.mkdir(parents=True, exist_ok=True)
        shard = month_dir / f"{session_id}.sqlite3"
        tmp = shard.with_suffix(".tmp.sqlite3")
        tmp.unlink(missing_ok=True)
        connection = sqlite3.connect(tmp)
        try:
            connection.executescript(
                """
                CREATE TABLE events (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    prev_hash TEXT,
                    event_hash TEXT NOT NULL,
                    PRIMARY KEY(session_id, sequence)
                );
                CREATE VIRTUAL TABLE events_fts USING fts5(sequence UNINDEXED, event_type, content, tokenize='unicode61');
                """
            )
            for event in events:
                payload = json.dumps(event.payload, sort_keys=True, separators=(",", ":"))
                metadata = json.dumps(event.metadata, sort_keys=True, separators=(",", ":"))
                connection.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (event.session_id, event.sequence, event.id, event.timestamp, event.event_type, payload, metadata, event.prev_hash, event.hash),
                )
                connection.execute(
                    "INSERT INTO events_fts(sequence, event_type, content) VALUES (?, ?, ?)",
                    (event.sequence, event.event_type, json.dumps({"type": event.event_type, "payload": event.payload}, sort_keys=True)),
                )
            connection.execute("PRAGMA user_version=1")
            connection.commit()
        finally:
            connection.close()
        os.replace(tmp, shard)
        os.chmod(shard, 0o444)
        archived_at = now.isoformat()
        entry = ArchiveEntry(session_id, session_name, str(shard), len(events), archived_at)
        with self._catalog() as catalog:
            catalog.execute(
                """INSERT INTO archived_sessions VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET session_name=excluded.session_name,
                    shard_path=excluded.shard_path,event_count=excluded.event_count,archived_at=excluded.archived_at""",
                (entry.session_id, entry.session_name, entry.shard_path, entry.event_count, entry.archived_at),
            )
        return entry

    def list_entries(self) -> list[ArchiveEntry]:
        with self._catalog() as connection:
            rows = connection.execute("SELECT * FROM archived_sessions ORDER BY archived_at DESC").fetchall()
        return [ArchiveEntry(**dict(row)) for row in rows]

    def resolve(self, ref: str) -> ArchiveEntry | None:
        with self._catalog() as connection:
            row = connection.execute("SELECT * FROM archived_sessions WHERE session_id=?", (ref,)).fetchone()
            if row is not None:
                return ArchiveEntry(**dict(row))
            rows = connection.execute("SELECT * FROM archived_sessions WHERE lower(session_name)=lower(?)", (ref,)).fetchall()
        if len(rows) > 1:
            raise ValueError(f"Archived session name is ambiguous: {ref}")
        return ArchiveEntry(**dict(rows[0])) if rows else None

    def read_events(self, ref: str, *, start_sequence: int = 1, end_sequence: int | None = None) -> list[HistoryEvent]:
        entry = self.resolve(ref)
        if entry is None:
            raise FileNotFoundError(f"Unknown archived session: {ref}")
        uri = f"file:{entry.shard_path}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        sql = "SELECT * FROM events WHERE sequence>=?"
        params: list[object] = [start_sequence]
        if end_sequence is not None:
            sql += " AND sequence<=?"
            params.append(end_sequence)
        sql += " ORDER BY sequence"
        rows = connection.execute(sql, params).fetchall()
        connection.close()
        return [
            HistoryEvent(
                id=str(row["event_id"]), sequence=int(row["sequence"]), session_id=str(row["session_id"]),
                timestamp=str(row["timestamp"]), type=str(row["event_type"]), version=1,
                payload=json.loads(str(row["payload_json"])), metadata=json.loads(str(row["metadata_json"])),
                prev_hash=row["prev_hash"], hash=str(row["event_hash"]),
            )
            for row in rows
        ]

    def search(self, ref: str, query: str, *, limit: int = 8) -> list[dict[str, object]]:
        entry = self.resolve(ref)
        if entry is None:
            raise FileNotFoundError(f"Unknown archived session: {ref}")
        terms = [token for token in query.replace('"', ' ').split() if token]
        if not terms:
            return []
        match = " OR ".join(f'"{term}"' for term in terms)
        uri = f"file:{entry.shard_path}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT sequence,event_type,content FROM events_fts WHERE events_fts MATCH ? ORDER BY bm25(events_fts) LIMIT ?",
            (match, limit),
        ).fetchall()
        connection.close()
        return [dict(row) for row in rows]
