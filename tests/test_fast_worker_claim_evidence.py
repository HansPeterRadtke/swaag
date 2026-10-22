"""A fast worker inventory answer needs an actual fresh observation.

These are output evidence checks, never an intent router or a replacement
for the full evidence-capable orchestrator.
"""
from __future__ import annotations

import pytest

from swaag.foreground_reply_evidence import normalize_fast_foreground_claim
from swaag.foreground_fast import generate_fast_foreground_response
from tests.test_fast_foreground_reply import fast, setup


@pytest.mark.parametrize("claimed", [
    "There are no active workers.",
    "There is one active worker.",
    "There are three active workers.",
    "Your warehouse worker is currently working.",
])
def test_without_current_world_never_certify_explicit_live_worker_claim(claimed):
    mode, reply = normalize_fast_foreground_claim(
        "answer", claimed, "How many workers are active right now?",
    )
    assert mode == "delegate"
    assert reply != claimed


@pytest.mark.parametrize("claimed", [
    "There are no active workers.",
    "There are three active workers.",
])
def test_incomplete_worker_snapshot_cannot_certify_live_count(claimed):
    mode, reply = normalize_fast_foreground_claim(
        "answer", claimed, "How many workers are active?",
        current_world={"snapshot_complete": False, "workers": []},
    )
    assert mode == "delegate"
    assert reply != claimed


@pytest.mark.parametrize("claimed", [
    "There are no active workers.",
    "There are three active workers.",
    "There is 2 active workers.",
])
def test_fresh_worker_count_mismatch_rejects_generated_reply(claimed):
    snapshot = {
        "snapshot_complete": True, "active_worker_count": 1,
        "workers": [{"id": "one", "status": "working"}],
    }
    mode, reply = normalize_fast_foreground_claim(
        "answer", claimed, "How many workers are active?",
        current_world=snapshot,
    )
    assert mode == "delegate"
    assert reply != claimed


def test_correct_fresh_count_stays_fast_and_historical_claim_can_be_quoted():
    snapshot = {
        "snapshot_complete": True, "active_worker_count": 1,
        "workers": [{"id": "one", "status": "working"}],
    }
    claim = "There is one active worker."
    assert normalize_fast_foreground_claim(
        "answer", claim, "How many workers are active?", current_world=snapshot
    ) == ("answer", claim)

    historical = "The warehouse worker has stopped."
    assert normalize_fast_foreground_claim(
        "answer", historical, "What did an earlier note claim?"
    ) == ("answer", historical)


def test_single_call_fast_worker_claim_without_evidence_falls_back(make_config):
    from tests.test_agent_action_loop import _action

    service, runtime, client = setup(make_config, [
        fast("answer", "There are no active workers."),
        _action(message="The current worker inventory must be checked."),
    ])
    try:
        answer = service.orchestrator_message("How many workers are active?")
        assert answer["answer"] == "The current worker inventory must be checked."
        assert [call["contract"] for call in client.requests] == [
            "fast_foreground_response", "agent_action",
        ]
    finally:
        service.close()


def test_current_fresh_zero_worker_inventory_can_answer_without_deep_model(make_config):
    service, runtime, client = setup(
        make_config, [fast("answer", "There are no active workers.")]
    )
    try:
        state = service._orchestrator_state()
        result = generate_fast_foreground_response(
            runtime, state, "How many active workers are there?",
            observed_world={
                "captured_at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "active_workers": [],
                "active_plans": [],
                "terminal_workers": [],
                "worker_question_inventory": {"questions": [], "complete": True},
            },
        )
        assert result and result.mode == "answer"
        assert len(client.requests) == 1
    finally:
        service.close()


@pytest.mark.parametrize("world", [
    {"snapshot_complete": True, "workers": "not-a-list", "active_worker_count": 0},
    {"snapshot_complete": True, "workers": [], "active_worker_count": False},
    {"snapshot_complete": True, "workers": [], "active_worker_count": None},
    {"snapshot_complete": True, "workers": [{"id": "w", "status": "working"}], "active_worker_count": 0},
    {"snapshot_complete": True, "workers": [None], "active_worker_count": 1},
])
def test_broken_claim_evidence_is_not_certified_even_if_snapshot_says_complete(world):
    mode, _ = normalize_fast_foreground_claim(
        "answer", "There are no active workers.",
        "How many active workers are there?", current_world=world,
    )
    assert mode == "delegate"


def test_full_but_valid_empty_world_still_supports_zero_workers():
    world = {"snapshot_complete": True, "active_worker_count": 0, "workers": []}
    assert normalize_fast_foreground_claim(
        "answer", "There are no active workers.",
        "How many active workers are there?", current_world=world,
    ) == ("answer", "There are no active workers.")
