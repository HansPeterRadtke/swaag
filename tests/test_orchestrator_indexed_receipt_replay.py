"""Old/complete receipts must replay from indexed lifecycle events, not huge prompts."""
from __future__ import annotations

import pytest
from tests.test_fast_foreground_reply import setup, fast


def test_reconnect_replays_exact_answer_without_materializing_giant_journal(make_config,monkeypatch):
    service,runtime,client=setup(make_config,[fast('answer','I am SWAAG.')])
    try:
        receipt=service._receive_orchestrator_message('Who are you?',control_id='indexed_replay')
        original=service._process_orchestrator_receipt(receipt)
        assert original['answer']=='I am SWAAG.'
        monkeypatch.setattr(runtime.history,'read_history',lambda *a,**k: (_ for _ in ()).throw(
            AssertionError('All-event history materialization is prohibited for receipt replay')))
        result=service._process_orchestrator_receipt(receipt)
        assert result==original
        assert len(client.requests)==1
    finally:service.close()


def test_very_old_completed_receipt_replays_correct_reply_after_many_new_turns(make_config,monkeypatch):
    service,runtime,client=setup(make_config,[fast('answer',f'Answer {i}') for i in range(30)])
    try:
        first=service._receive_orchestrator_message('First question',control_id='very_old')
        service._process_orchestrator_receipt(first)
        for i in range(20):
            other=service._receive_orchestrator_message('Different question '+str(i),control_id=f'later_{i}')
            service._process_orchestrator_receipt(other)
        monkeypatch.setattr(runtime.history,'read_history',lambda *a,**k: (_ for _ in ()).throw(
            AssertionError('Do not materialize the full history to replay an older receipt')))
        assert service._process_orchestrator_receipt(first)['answer']=='Answer 0'
        assert len(client.requests)==21
    finally:service.close()
