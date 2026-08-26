from __future__ import annotations

import sqlite3

import pytest

from swaag.communication import CommunicationStore
from swaag.embedding_index import DerivedEmbeddingIndex
from swaag.history import HistoryStore
from swaag.history_archive import HistoryArchiveStore
from swaag.inference import InferenceRequestCoordinator
from swaag.preemption import ModelPreemptionCoordinator
from swaag.sqlite_schema import (
    UnsupportedSchemaVersionError,
    apply_sqlite_migrations,
)
from swaag.workers import WorkerStore


class _Embeddings:
    def embed(self, texts):
        return [[1.0] for _text in texts]


def _version(path) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])


def test_migrations_adopt_unversioned_state_without_losing_rows(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE durable(value TEXT NOT NULL)")
        connection.execute("INSERT INTO durable VALUES ('preserved')")

    with sqlite3.connect(path) as connection:
        version = apply_sqlite_migrations(
            connection,
            store_name="test store",
            migrations=(
                ("CREATE TABLE IF NOT EXISTS durable(value TEXT NOT NULL)",),
                ("CREATE INDEX IF NOT EXISTS durable_value ON durable(value)",),
            ),
        )
        rows = connection.execute("SELECT value FROM durable").fetchall()

    assert version == 2
    assert _version(path) == 2
    assert rows == [("preserved",)]


def test_migration_failure_rolls_back_schema_and_version(tmp_path) -> None:
    path = tmp_path / "rollback.sqlite3"
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.OperationalError):
            apply_sqlite_migrations(
                connection,
                store_name="test store",
                migrations=(("CREATE TABLE first(value TEXT)", "INVALID SQL"),),
            )
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

    assert tables == []
    assert _version(path) == 0


def test_newer_durable_schema_is_rejected_without_mutation(tmp_path) -> None:
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=9")
        with pytest.raises(UnsupportedSchemaVersionError, match="newer than supported"):
            apply_sqlite_migrations(
                connection,
                store_name="test store",
                migrations=(("CREATE TABLE current(value TEXT)",),),
            )
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall() == []


def test_all_runtime_sqlite_stores_record_explicit_schema_versions(tmp_path) -> None:
    sessions = tmp_path / "sessions"
    communication = CommunicationStore(sessions)
    workers = WorkerStore(sessions)
    history = HistoryStore(sessions)
    archives = HistoryArchiveStore(sessions)
    inference = InferenceRequestCoordinator(
        sessions,
        backend_key="test",
        capacity_resolver=lambda: (1, "test"),
    )
    preemption = ModelPreemptionCoordinator(sessions)
    embeddings = DerivedEmbeddingIndex(sessions, _Embeddings())

    assert _version(communication.path) == 1
    assert _version(workers.path) == 3
    assert _version(history.sqlite_history_path()) == 1
    assert _version(archives.catalog_path) == 1
    assert _version(inference.path) == 1
    assert _version(preemption.path) == 1
    assert _version(embeddings.path) == 1


def test_worker_completion_mode_migration_preserves_existing_rows(tmp_path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    path = sessions / "workers.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE workers (
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
            """
        )
        connection.execute(
            """
            INSERT INTO workers(
                worker_id, session_id, objective, status, created_at, updated_at
            ) VALUES ('worker_old', 'session_old', 'preserved', 'created', 'now', 'now')
            """
        )
        connection.execute("PRAGMA user_version=2")

    store = WorkerStore(sessions)
    record = store.get("worker_old")

    assert _version(store.path) == 3
    assert record.objective == "preserved"
    assert record.completion_mode == "natural"
