"""Optional foreground SQLite state validation must not wait for writer locks."""
from __future__ import annotations

import sqlite3
import threading
import time

from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from tests.test_preemption import _ImmediateClient


def test_fast_revision_reader_fails_closed_without_waiting_for_busy_writer(make_config):
    runtime = AgentRuntime(
        make_config(model__context_limit=32000),
        model_client=_ImmediateClient("unused"),
    )
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    writer = sqlite3.connect(service.workers.store.path, timeout=1)
    try:
        assert writer.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        writer.execute("PRAGMA locking_mode=EXCLUSIVE")
        writer.execute("BEGIN EXCLUSIVE")
        results = []
        finished = threading.Event()

        def probe():
            start = time.monotonic()
            try:
                result = service._foreground_task_revision()
                results.append(("success", time.monotonic() - start, result))
            except sqlite3.Error as exc:
                results.append(("sqlite_busy", time.monotonic() - start, str(exc)))
            finally:
                finished.set()

        probe_thread = threading.Thread(target=probe, daemon=True)
        probe_thread.start()
        # This exclusive writer is deliberately held *longer* than the fast
        # state's read budget; the new reader must fail closed immediately.
        blocked = not finished.wait(0.25)
        writer.rollback()
        writer.close()
        probe_thread.join(timeout=3)
        assert finished.is_set(), "Revision read never finished after releasing the lock"
        assert not blocked, f"Foreground revision spent over 250 ms behind a SQLite writer: {results}"
        assert results[0][0] == "sqlite_busy"
        assert results[0][1] < 0.25
        # The fallback may retry only after the competing writer releases it.
        assert service._foreground_task_revision()
    finally:
        try:
            writer.rollback()
        except sqlite3.ProgrammingError:
            pass  # The successful path already closed the disposable writer.
        writer.close()
        service.close()


def test_fast_revision_connection_is_read_only_and_preserves_special_paths(tmp_path):
    from swaag.communication import CommunicationService

    directory = tmp_path / "path with ? and #"
    directory.mkdir()
    path = directory / "history test #?.sqlite3"
    with sqlite3.connect(path) as writable:
        writable.execute("CREATE TABLE probe (value TEXT)")
        writable.execute("INSERT INTO probe VALUES ('observed')")
    with CommunicationService._foreground_revision_connection(path) as connection:
        assert connection.execute("SELECT value FROM probe").fetchone()[0] == "observed"
        import pytest
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO probe VALUES ('unwanted')")
    with sqlite3.connect(path) as writable:
        assert writable.execute("SELECT count(*) FROM probe").fetchone()[0] == 1
    missing = directory / "nonexistent.sqlite3"
    assert not missing.exists()
    with pytest.raises(sqlite3.OperationalError):
        with CommunicationService._foreground_revision_connection(missing):
            pass
    assert not missing.exists()


def test_fast_plan_revision_does_not_wait_for_exclusive_plan_writer(make_config):
    runtime = AgentRuntime(make_config(model__context_limit=32000),
                           model_client=_ImmediateClient("unused"))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    path = service.orchestration.store.path
    writer = sqlite3.connect(path, timeout=1)
    try:
        assert writer.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        writer.execute("PRAGMA locking_mode=EXCLUSIVE")
        writer.execute("BEGIN EXCLUSIVE")
        done = threading.Event()
        result = []

        def query():
            start = time.monotonic()
            try:
                service._foreground_task_revision()
                result.append(("success", time.monotonic()-start))
            except sqlite3.Error:
                result.append(("busy", time.monotonic()-start))
            finally:
                done.set()

        thread = threading.Thread(target=query, daemon=True)
        thread.start()
        is_fast = done.wait(.25)
        writer.rollback()
        writer.close()  # EXCLUSIVE locking_mode persists even after rollback.
        thread.join(timeout=2)
        assert done.is_set()
        assert is_fast and result[0][0] == "busy" and result[0][1] < .25
        assert service._foreground_task_revision()
    finally:
        try:
            writer.rollback()
        except sqlite3.ProgrammingError:
            pass
        writer.close()
        service.close()


def test_fast_history_revision_does_not_wait_for_exclusive_history_writer(make_config):
    runtime = AgentRuntime(make_config(model__context_limit=32000),
                           model_client=_ImmediateClient("unused"))
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    service.workers.create("Evaluate workload")
    path = runtime.history.sqlite_history_path()
    writer = sqlite3.connect(path, timeout=1)
    try:
        assert writer.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        writer.execute("PRAGMA locking_mode=EXCLUSIVE")
        writer.execute("BEGIN EXCLUSIVE")
        done = threading.Event()
        result = []

        def query():
            start = time.monotonic()
            try:
                service._foreground_task_revision()
                result.append(("success", time.monotonic()-start))
            except sqlite3.Error:
                result.append(("busy", time.monotonic()-start))
            finally:
                done.set()

        thread = threading.Thread(target=query, daemon=True)
        thread.start()
        is_fast = done.wait(.25)
        writer.rollback()
        writer.close()  # EXCLUSIVE locking_mode persists even after rollback.
        thread.join(timeout=2)
        assert done.is_set()
        assert is_fast and result[0][0] == "busy" and result[0][1] < .25
        assert service._foreground_task_revision()
    finally:
        try:
            writer.rollback()
        except sqlite3.ProgrammingError:
            pass
        writer.close()
        service.close()
