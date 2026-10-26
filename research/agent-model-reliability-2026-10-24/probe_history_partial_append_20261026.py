#!/usr/bin/env python3
"""Deterministic test-only emulation of a JSONL reader during a split append.

No production files or external model calls. Proves a reader/writer temporal
window if the writer's encoded event has been partially persisted in a file.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Event,Thread
from unittest.mock import patch

import swaag.history as history

SOURCE=Path(__file__).resolve().parents[2]/'src/swaag/history.py'
VERSION='swaag_history_partial_append_read_race_v1_20261026'


def probe():
    with tempfile.TemporaryDirectory(prefix='swaag-jsonl-read-window-') as root:
        p=Path(root)/'sessions'
        writer=history.HistoryStore(p)
        state=writer.create(config_fingerprint='research',model_base_url='http://fake-unconnected')
        reader=history.HistoryStore(p,write_projections=False)
        assert len(reader.read_history(state.session_id))>=1
        baseline=writer.history_path(state.session_id).read_bytes()
        opened=Event();unblock=Event();outcomes=[]
        def split(path,text,encoding):
            # Simulates a real preempted writer between UTF-8 append system
            # calls. Does not modify the production append implementation.
            with path.open('a',encoding=encoding) as f:
                f.write(text[:150]);f.flush();os.fsync(f.fileno())
                opened.set()
                assert unblock.wait(8),'Reader did not reach split append'
                f.write(text[150:]);f.flush();os.fsync(f.fileno())
        def write():
            try:writer.record_event(state,'error',{'operation':'isolated_test',
                              'error':'THIS_IS_COMPLETE_ONLY_AFTER_RELEASE','error_type':'Test'})
            except Exception as exc:outcomes.append((type(exc).__name__,str(exc)))
        with patch.object(history,'append_text',side_effect=split):
            thread=Thread(target=write,name='isolated-jsonl-writer')
            thread.start()
            try:
                assert opened.wait(5),'Writing did not begin'
                partial=writer.history_path(state.session_id).read_bytes()
                assert partial.startswith(baseline) and len(partial)>len(baseline)
                try:
                    reader.read_history(state.session_id)
                    first_error=None
                except history.HistoryCorruptionError as exc:
                    first_error=type(exc).__name__
            finally:
                unblock.set();thread.join(10)
        assert not thread.is_alive() and not outcomes
        final=writer.history_path(state.session_id).read_bytes()
        assert len(final)>len(partial)
        recovered=reader.read_history(state.session_id)
        assert recovered[-1].payload['error']=='THIS_IS_COMPLETE_ONLY_AFTER_RELEASE'
        assert first_error=='HistoryCorruptionError'
        return {'version':VERSION,
            'canonical_history_py_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'withheld_write_tail_deterministically':True,
            'reader_error_during_split_append':first_error,
            'reader_recovered_after_append_complete':len(recovered)>=2,
            'final_history_chain_verifies':writer.rebuild_from_history(state.session_id).event_count==state.event_count,
            'temporary_partial_history_size':len(partial),
            'complete_history_size':len(final),
            'no_production_history_files_touched':True,
            'no_model_calls_or_external_tool_effects':True,
            'interpretation':'For an in-progress append, a separate reader can see a truncated final JSONL record. This is a deterministic fault-injection demonstration, not measured spontaneous failure frequency or confirmed root cause of one previous failure.'}

if __name__=='__main__':print(json.dumps(probe(),indent=2))
