from __future__ import annotations

import sqlite3

import pytest

from swaag.communication import CommunicationStore, _COMMUNICATION_STORE_MIGRATIONS
from swaag.delegated_tools import DelegatedToolStore
from swaag.embedding_index import DerivedEmbeddingIndex
from swaag.history import HistoryStore
from swaag.history_archive import HistoryArchiveStore
from swaag.inference import InferenceRequestCoordinator
from swaag.preemption import ModelPreemptionCoordinator
from swaag.prompt_instruction_store import PromptInstructionStore
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


def test_all_runtime_sqlite_stores_record_explicit_schema_versions(
    tmp_path, make_config
) -> None:
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
    prompt_instructions = PromptInstructionStore(sessions, make_config())
    delegated_tools = DelegatedToolStore(sessions)

    assert _version(communication.path) == 4
    assert _version(workers.path) == 4
    assert _version(history.sqlite_history_path()) == 1
    assert _version(archives.catalog_path) == 1
    assert _version(inference.path) == 1
    assert _version(preemption.path) == 1
    assert _version(embeddings.path) == 1
    assert _version(prompt_instructions.path) == 1
    assert _version(delegated_tools.path) == 1


def test_communication_stream_bounds_migration_preserves_protocol_mappings(
    tmp_path,
) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    path = sessions / "communication.sqlite3"
    with sqlite3.connect(path) as connection:
        apply_sqlite_migrations(
            connection,
            store_name="communication store",
            migrations=_COMMUNICATION_STORE_MIGRATIONS[:2],
        )
        connection.execute(
            """
            INSERT INTO protocol_messages(
                protocol, external_message_id, external_context_id,
                worker_id, created_at
            ) VALUES ('open_webui', 'message-1', 'chat-1', 'worker-1', 'now')
            """
        )

    store = CommunicationStore(sessions)

    assert _version(path) == 4
    assert store.protocol_message_bounds("open_webui", "message-1") == (
        "chat-1",
        "worker-1",
        0,
        None,
    )


def test_protocol_state_snapshots_are_exact_idempotent_and_inherited(tmp_path) -> None:
    store = CommunicationStore(tmp_path / "sessions")
    first = store.bind_protocol_state(
        "ag_ui",
        "thread-1",
        "run-1",
        state={"nested": {"value": 3}, "items": [1, 2]},
        client_supplied=True,
    )
    store.record_protocol_message(
        "ag_ui", "run-1", "thread-1", "worker-1"
    )
    duplicate = store.bind_protocol_state(
        "ag_ui",
        "thread-1",
        "run-1",
        state={"ignored": True},
        client_supplied=True,
    )
    inherited = store.bind_protocol_state(
        "ag_ui",
        "thread-1",
        "run-2",
        state=None,
        client_supplied=False,
    )

    assert duplicate == first
    assert inherited.revision == 2
    assert inherited.state == first.state
    assert inherited.state_sha256 == first.state_sha256
    assert inherited.client_supplied is False


def test_worker_lifecycle_option_migrations_preserve_existing_rows(tmp_path) -> None:
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

    assert _version(store.path) == 4
    assert record.objective == "preserved"
    assert record.completion_mode == "natural"
    assert record.presentation_modes == []
