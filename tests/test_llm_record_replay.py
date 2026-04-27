from __future__ import annotations

import json
from pathlib import Path

import pytest

from swaag.grammar import yes_no_contract
from swaag.testing.llm_record_replay import MissingReplayEntryError, RecordReplayModelClient
from tests.helpers import FakeModelClient


def test_record_replay_client_records_and_replays_by_full_request_payload(tmp_path: Path) -> None:
    cassette_path = tmp_path / "cassette.json"
    contract = yes_no_contract()
    recording_delegate = FakeModelClient(responses=["yes"])
    recording_client = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=recording_delegate,
        request_metadata={"model_name": "fixture-model", "model_version": "v1"},
    )

    request = recording_client.build_completion_request("Answer yes", max_tokens=4, contract=contract)
    recorded = recording_client.send_completion(request, timeout_seconds=7)

    assert recorded.text == "yes"
    assert cassette_path.exists()
    cassette_payload = json.loads(cassette_path.read_text(encoding="utf-8"))
    assert cassette_payload["request_metadata"]["model_name"] == "fixture-model"
    assert len(cassette_payload["entries"]) == 1

    replay_client = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="replay",
        delegate=FakeModelClient(responses=["no"]),
        request_metadata={"model_name": "fixture-model", "model_version": "v1"},
    )
    replayed = replay_client.send_completion(request, timeout_seconds=7)

    assert replayed.text == "yes"
    assert replay_client.delegate.requests == []


def test_record_replay_client_key_changes_when_request_metadata_changes(tmp_path: Path) -> None:
    cassette_path = tmp_path / "cassette.json"
    contract = yes_no_contract()
    recording_client = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=FakeModelClient(responses=["yes"]),
        request_metadata={"model_name": "fixture-model", "structured_output_mode": "server_schema"},
    )
    request = recording_client.build_completion_request("Answer yes", max_tokens=4, contract=contract)
    recording_client.send_completion(request, timeout_seconds=5)

    replay_client = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="replay",
        delegate=FakeModelClient(),
        request_metadata={"model_name": "fixture-model", "structured_output_mode": "post_validate"},
    )

    with pytest.raises(MissingReplayEntryError):
        replay_client.send_completion(request, timeout_seconds=5)

def test_record_replay_client_record_mode_replays_existing_entries_without_calling_delegate(tmp_path: Path) -> None:
    """In 'record' mode, existing cassette entries are replayed without calling the delegate."""
    cassette_path = tmp_path / "cassette.json"
    contract = yes_no_contract()
    recording_delegate = FakeModelClient(responses=["yes"])
    recording_client = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=recording_delegate,
        request_metadata={"model_name": "fixture-model", "model_version": "v1"},
    )
    request = recording_client.build_completion_request("Answer yes", max_tokens=4, contract=contract)
    recording_client.send_completion(request, timeout_seconds=5)

    assert recording_client.recorded_count == 1
    assert recording_client.replayed_count == 0

    # New client in "record" mode with same cassette: should replay without calling delegate
    replay_delegate = FakeModelClient(responses=["no"])
    replay_client = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=replay_delegate,
        request_metadata={"model_name": "fixture-model", "model_version": "v1"},
    )
    result = replay_client.send_completion(request, timeout_seconds=5)

    assert result.text == "yes"  # replayed, not re-recorded
    assert replay_delegate.requests == []  # delegate never called
    assert replay_client.replayed_count == 1
    assert replay_client.recorded_count == 0
