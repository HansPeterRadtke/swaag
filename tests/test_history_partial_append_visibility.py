"""Keep streaming/readers out of a real in-progress JSONL append window.

Regression for the independently demonstrated partial-append read failure.
All events and SQLite sessions exist only inside tmp_path. No production data,
model requests, external tools or background scheduler actions are exercised.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from unittest.mock import patch

import pytest

from swaag.history import HistoryCorruptionError, HistoryStore
from swaag.orchestrator_stream import OrchestratorEventCursor


@pytest.mark.parametrize('read_path', ['whole_history', 'has_event', 'stream_cursor_caught_up'])
def test_reader_waits_for_writer_sqlite_commit_before_indexing_partial_append(tmp_path, read_path):
    writer = HistoryStore(tmp_path / 'sessions')
    state = writer.create(config_fingerprint='isolated-research', model_base_url='http://unused')
    reader = HistoryStore(writer.root, write_projections=False)
    assert len(reader.read_history(state.session_id)) == 1
    next_seq = state.event_count + 1
    started_append = threading.Event()
    release_append = threading.Event()
    reader_started = threading.Event()
    reader_done = threading.Event()
    writer_errors = []
    reader_errors = []
    reader_results = []

    def inject_split(path, content, *, encoding):
        data = content.encode(encoding)
        assert len(data) > 200
        with Path(path).open('ab') as handle:
            handle.write(data[:180])
            handle.flush()
            os.fsync(handle.fileno())
            started_append.set()
            if not release_append.wait(6):
                raise AssertionError('writer timeout while the reader was verifying')
            handle.write(data[180:])
            handle.flush()
            os.fsync(handle.fileno())

    def write_one():
        try:
            writer.record_event(state, 'error', {
                'operation': 'isolated_append_window',
                'error': 'full event must not be read before sqlite commit',
                'error_type': 'Test',
            })
        except Exception as exc:
            writer_errors.append(exc)

    def read_during_append():
        reader_started.set()
        try:
            if read_path == 'whole_history':
                reader_results.append(reader.read_history(state.session_id))
            elif read_path == 'has_event':
                reader_results.append(reader.has_history_at_or_after(state.session_id, next_seq))
            else:
                cursor = OrchestratorEventCursor(reader, {
                    'session_id': state.session_id,
                    'control_id': 'isolated-nonmatching-control',
                    'history_anchor_sequence': next_seq,
                })
                reader_results.append(cursor.caught_up())
        except Exception as exc:
            reader_errors.append(exc)
        finally:
            reader_done.set()

    # Split only the existing SQLite-protected append; do not patch readers or
    # omit the real history index / hash-chain validation code.
    with patch('swaag.history.append_text', side_effect=inject_split):
        write_thread = threading.Thread(target=write_one, name='isolated-SWAAG-journal-writer')
        read_thread = threading.Thread(target=read_during_append, name='isolated-SWAAG-stream-reader')
        write_thread.start()
        try:
            assert started_append.wait(5), 'writer did not open the partial append window'
            raw = writer.history_path(state.session_id).read_bytes()
            assert raw and not raw.endswith(b'\n'), 'the writer has only emitted an incomplete final line'
            read_thread.start()
            assert reader_started.wait(5)
            # The peer reader must not validate/read a journal that the writer
            # has not finished or committed. This fails against old source.
            assert not reader_done.wait(0.15), 'reader observed a partially appended event before commit'
        finally:
            release_append.set()
            write_thread.join(9)
            if read_thread.ident is not None:
                read_thread.join(9)
    assert not write_thread.is_alive()
    assert not read_thread.is_alive()
    assert not writer_errors, writer_errors
    assert not reader_errors, reader_errors
    assert len(reader_results) == 1
    if read_path == 'whole_history':
        assert reader_results[0][-1].sequence == next_seq
        assert reader_results[0][-1].payload['operation'] == 'isolated_append_window'
    elif read_path == 'has_event':
        assert reader_results[0] is True
    else:
        assert reader_results[0] is False, 'the stream cursor must still see the appended event'
    assert reader.has_history_at_or_after(state.session_id, next_seq)
    assert reader.rebuild_from_history(state.session_id).event_count == state.event_count


def test_unsanctioned_complete_external_corruption_is_still_rejected(tmp_path):
    owner = HistoryStore(tmp_path / 'sessions')
    state = owner.create(config_fingerprint='isolated-research', model_base_url='http://unused')
    owner.record_event(state, 'error', {
        'operation': 'source_tamper', 'error': 'original_content', 'error_type': 'Test',
    })
    second_reader = HistoryStore(owner.root, write_projections=False)
    assert second_reader.read_history(state.session_id)[-1].payload['error'] == 'original_content'
    path = owner.history_path(state.session_id)
    original = path.read_bytes()
    changed = original.replace(b'original_content', b'changed_content_', 1)
    assert changed != original and len(changed) == len(original)
    path.write_bytes(changed)
    with pytest.raises(HistoryCorruptionError):
        second_reader.read_history(state.session_id)


def test_slow_historical_scan_does_not_block_other_session_durable_writer(tmp_path):
    """A large journal rescan must not hold SQLite's global writer lock."""
    store = HistoryStore(tmp_path / 'sessions')
    first = store.create(config_fingerprint='isolated-research', model_base_url='http://unused')
    second = store.create(config_fingerprint='isolated-research', model_base_url='http://unused')
    reader = HistoryStore(store.root, write_projections=False)
    assert reader.read_history(first.session_id)
    assert reader.read_history(second.session_id)
    # Force a legitimate revalidation of only the first session's journal.
    with store.history_path(first.session_id).open('a', encoding='utf-8') as handle:
        handle.write('\n')
    scanning = threading.Event()
    finish_scan = threading.Event()
    writer_finished = threading.Event()
    exceptions = []
    original_scan = reader._iter_history_jsonl

    def paused_scan(*args, **kwargs):
        scanning.set()
        if not finish_scan.wait(6):
            raise AssertionError('timed out during isolated history scan')
        yield from original_scan(*args, **kwargs)

    def slow_reader():
        try:
            assert reader.read_history(first.session_id)
        except Exception as exc:
            exceptions.append(exc)

    def other_session_write():
        try:
            store.record_event(second, 'error', {
                'operation': 'other_session_not_blocked',
                'error': 'unrelated concurrent writer',
                'error_type': 'Test',
            })
        except Exception as exc:
            exceptions.append(exc)
        finally:
            writer_finished.set()

    with patch.object(reader, '_iter_history_jsonl', side_effect=paused_scan):
        scan_thread = threading.Thread(target=slow_reader)
        writer_thread = threading.Thread(target=other_session_write)
        scan_thread.start()
        try:
            assert scanning.wait(5), 'expected a forced valid journal rescan'
            writer_thread.start()
            assert writer_finished.wait(0.8), 'unrelated session writer blocked by a large history scan'
        finally:
            finish_scan.set()
            scan_thread.join(8)
            if writer_thread.ident is not None:
                writer_thread.join(8)
    assert not scan_thread.is_alive()
    assert not writer_thread.is_alive()
    assert not exceptions, exceptions
    assert store.read_history(second.session_id)[-1].payload['operation'] == 'other_session_not_blocked'


def _cross_process_writer(root, session_id, opened, release, result_queue):
    """Isolated child, separate Python process and independent SQLite client."""
    try:
        history = HistoryStore(Path(root), write_projections=False)
        state = history.rebuild_from_history(session_id)
        assert state.event_count >= 1

        def split(path, content, *, encoding):
            data = content.encode(encoding)
            with Path(path).open('ab') as stream:
                stream.write(data[:180]);stream.flush();os.fsync(stream.fileno())
                opened.set()
                if not release.wait(8):
                    raise AssertionError('cross-process writer release timed out')
                stream.write(data[180:]);stream.flush();os.fsync(stream.fileno())

        with patch('swaag.history.append_text', side_effect=split):
            history.record_event(state, 'error', {
                'operation': 'cross_process_partial_append',
                'error': 'reader must wait for this exact source event commit',
                'error_type': 'Test',
            })
        result_queue.put(('ok', state.event_count))
    except BaseException as exc:
        result_queue.put(('error', type(exc).__name__ + ': ' + str(exc)))


def test_cross_process_history_writer_and_stream_reader_cannot_observe_partial_event(tmp_path):
    """A second OS process (not merely another thread) respects the fence."""
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    writer = HistoryStore(tmp_path / 'sessions')
    state = writer.create(config_fingerprint='isolated-research', model_base_url='http://unused')
    reader = HistoryStore(writer.root, write_projections=False)
    assert reader.read_history(state.session_id)
    expected_sequence = state.event_count + 1
    opened, release = ctx.Event(), ctx.Event()
    outcomes = ctx.Queue()
    process = ctx.Process(target=_cross_process_writer,
                          args=(str(writer.root), state.session_id, opened, release, outcomes))
    process.start()
    read_thread = None
    reader_done = threading.Event()
    errors = []
    result = []

    def check_cursor():
        try:
            cursor = OrchestratorEventCursor(reader, {
                'session_id': state.session_id,
                'control_id': 'synthetic-no-match',
                'history_anchor_sequence': expected_sequence,
            })
            result.append(cursor.caught_up())
        except BaseException as exc:
            errors.append(exc)
        finally:
            reader_done.set()

    try:
        assert opened.wait(8), 'independent writer never opened partial append window'
        assert not writer.history_path(state.session_id).read_bytes().endswith(b'\n')
        read_thread = threading.Thread(target=check_cursor)
        read_thread.start()
        assert not reader_done.wait(0.15), 'separate-process incomplete event leaked to stream reader'
    finally:
        release.set()
        process.join(12)
        if read_thread is not None:
            read_thread.join(12)
        if process.is_alive():
            process.terminate();process.join(5)
    assert not process.is_alive(), 'aborted only the test child, not a real service'
    assert process.exitcode == 0
    assert outcomes.get(timeout=3) == ('ok', expected_sequence)
    assert not errors, errors
    assert not read_thread.is_alive()
    assert result == [False], 'new committed source event must remain visible to cursor'
    assert reader.read_history(state.session_id)[-1].payload['operation'] == 'cross_process_partial_append'


def test_incomplete_external_tail_without_a_writer_is_genuine_corruption(tmp_path):
    """Do not hide a permanently torn or forged JSONL source behind retry."""
    owner = HistoryStore(tmp_path / 'sessions')
    state = owner.create(config_fingerprint='isolated-research', model_base_url='http://unused')
    observer = HistoryStore(owner.root, write_projections=False)
    assert observer.read_history(state.session_id)
    with owner.history_path(state.session_id).open('ab') as out:
        out.write(b'{"incomplete_jsonl_record": "not_fully_written')
        out.flush();os.fsync(out.fileno())
    with pytest.raises(HistoryCorruptionError):
        observer.has_history_at_or_after(state.session_id, state.event_count+1)
