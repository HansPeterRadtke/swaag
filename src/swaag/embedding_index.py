from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
from concurrent.futures import ThreadPoolExecutor, Future

import requests

from swaag.sqlite_schema import apply_sqlite_migrations


_EMBEDDING_INDEX_MIGRATIONS = (
    (
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            session_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            field TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            PRIMARY KEY(session_id, sequence, field)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS embeddings_session_sequence
        ON embeddings(session_id, sequence)
        """,
        """
        CREATE TABLE IF NOT EXISTS indexed_sessions (
            session_id TEXT PRIMARY KEY,
            complete_through INTEGER NOT NULL
        )
        """,
    ),
)


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class OpenAICompatibleEmbeddingProvider:
    base_url: str
    endpoint: str
    model: str
    timeout_seconds: float = 30.0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = requests.post(
            self.base_url.rstrip("/") + "/" + self.endpoint.lstrip("/"),
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        rows = sorted(payload.get("data", []), key=lambda item: int(item.get("index", 0)))
        vectors = [list(map(float, item["embedding"])) for item in rows]
        if len(vectors) != len(texts):
            raise RuntimeError(f"Embedding endpoint returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors


@dataclass(slots=True, frozen=True)
class EmbeddingMatch:
    session_id: str
    sequence: int
    field: str
    score: float


class DerivedEmbeddingIndex:
    """Rebuildable, non-authoritative semantic index over exact history events."""

    def __init__(self, root: Path, provider: EmbeddingProvider):
        self.root = Path(root).expanduser()
        self.path = self.root / "derived_embeddings.sqlite3"
        self.provider = provider
        self.root.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _init(self) -> None:
        with self._connect() as connection:
            apply_sqlite_migrations(
                connection,
                store_name="derived embedding index",
                migrations=_EMBEDDING_INDEX_MIGRATIONS,
            )

    @staticmethod
    def _hash_text(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


    def complete_through(self, session_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT complete_through FROM indexed_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def mark_complete_through(self, session_id: str, sequence: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO indexed_sessions(session_id,complete_through) VALUES(?,?)
                ON CONFLICT(session_id) DO UPDATE SET complete_through=excluded.complete_through""",
                (session_id, int(sequence)),
            )

    def upsert(self, session_id: str, sequence: int, field: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        vector = self.provider.embed([text])[0]
        if not vector:
            raise ValueError("Embedding vector must not be empty")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO embeddings(session_id, sequence, field, text_hash, vector_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id, sequence, field) DO UPDATE SET
                    text_hash=excluded.text_hash, vector_json=excluded.vector_json
                """,
                (session_id, int(sequence), field, self._hash_text(text), json.dumps(vector, separators=(",", ":"))),
            )

    def rebuild_session(
        self,
        session_id: str,
        records: Sequence[tuple[int, str, str]],
    ) -> int:
        """records are (sequence, field, text); canonical history remains elsewhere."""
        with self._connect() as connection:
            connection.execute("DELETE FROM embeddings WHERE session_id=?", (session_id,))
            connection.execute("DELETE FROM indexed_sessions WHERE session_id=?", (session_id,))
        count = 0
        highest = 0
        for sequence, field, text in records:
            highest = max(highest, int(sequence))
            if text.strip():
                self.upsert(session_id, sequence, field, text)
                count += 1
        if highest:
            self.mark_complete_through(session_id, highest)
        return count

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b) or not a:
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else -1.0

    def search(self, query: str, *, session_id: str | None = None, limit: int = 8) -> list[EmbeddingMatch]:
        if limit <= 0:
            return []
        query_vector = self.provider.embed([query])[0]
        sql = "SELECT session_id, sequence, field, vector_json FROM embeddings"
        params: tuple[object, ...] = ()
        if session_id is not None:
            sql += " WHERE session_id=?"
            params = (session_id,)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        matches = [
            EmbeddingMatch(
                session_id=str(row["session_id"]),
                sequence=int(row["sequence"]),
                field=str(row["field"]),
                score=self._cosine(query_vector, json.loads(str(row["vector_json"]))),
            )
            for row in rows
        ]
        matches.sort(key=lambda item: (item.score, item.sequence), reverse=True)
        return matches[:limit]


class AsyncEmbeddingIndexer:
    """Best-effort background derived indexer; canonical event persistence never waits on it."""

    def __init__(self, index: DerivedEmbeddingIndex, fields: Sequence[str]):
        self.index = index
        self.fields = tuple(fields)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swaag-embeddings")
        self._last_future: Future[object] | None = None

    def submit(self, event) -> None:
        self._last_future = self._executor.submit(self._index_event, event)

    def _index_event(self, event) -> None:
        if event.event_type == "agent_status":
            for field in self.fields:
                value = event.payload.get(field)
                if isinstance(value, str) and value.strip():
                    self.index.upsert(event.session_id, event.sequence, field, value)
        self.index.mark_complete_through(event.session_id, event.sequence)

    def flush(self, timeout: float | None = None) -> None:
        if self._last_future is not None:
            self._last_future.result(timeout=timeout)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
