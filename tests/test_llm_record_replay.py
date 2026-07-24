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

    assert json.loads(recorded.text) == {"answer": "yes"}
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

    assert json.loads(replayed.text) == {"answer": "yes"}
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
        request_metadata={"model_name": "fixture-model", "structured_output_mode": "server_schema_v2"},
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

    assert json.loads(result.text) == {"answer": "yes"}  # replayed, not re-recorded
    assert replay_delegate.requests == []  # delegate never called
    assert replay_client.replayed_count == 1
    assert replay_client.recorded_count == 0


def test_record_replay_default_metadata_versions_runtime_contract(tmp_path: Path) -> None:
    cassette_path = tmp_path / "cassette.json"
    client = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=FakeModelClient(responses=["yes"]),
    )
    request = client.build_completion_request("Answer yes", max_tokens=4, contract=yes_no_contract())
    client.send_completion(request, timeout_seconds=5)
    payload = json.loads(cassette_path.read_text(encoding="utf-8"))
    metadata = payload["request_metadata"]
    assert metadata["replay_contract_version"] == "2026-07-24-exact-request-cache-v2"
    assert metadata["canonicalize_dynamic_values"] is False
    assert metadata["model_transport"] == "streaming_token_timeout"


def test_record_replay_exact_mode_does_not_collapse_dynamic_prompt_values(tmp_path: Path) -> None:
    cassette_path = tmp_path / "exact-cache.json"
    contract = yes_no_contract()
    recorder = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=FakeModelClient(responses=["yes"]),
        request_metadata={"model_name": "fixture-model", "model_version": "v1"},
    )
    first = recorder.build_completion_request(
        "Session session_aaaaaaaaaaaa at 2026-07-24T01:00:00+00:00",
        max_tokens=4,
        contract=contract,
    )
    recorder.send_completion(first)
    replayer = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="replay",
        delegate=FakeModelClient(),
        request_metadata={"model_name": "fixture-model", "model_version": "v1"},
    )
    second = replayer.build_completion_request(
        "Session session_bbbbbbbbbbbb at 2026-07-24T02:00:00+00:00",
        max_tokens=4,
        contract=contract,
    )
    with pytest.raises(MissingReplayEntryError):
        replayer.send_completion(second)


def test_record_replay_key_covers_generation_parameters(tmp_path: Path) -> None:
    cassette_path = tmp_path / "generation-key.json"
    contract = yes_no_contract()
    recorder = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=FakeModelClient(responses=["yes"]),
        request_metadata={"model_name": "fixture-model", "model_version": "v1"},
    )
    request = recorder.build_completion_request("Answer yes", max_tokens=4, contract=contract)
    request.update(
        {
            "model": "fixture-model-v1",
            "seed": 11,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": ["END"],
        }
    )
    recorder.send_completion(request)
    changed_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["no"]}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    for field, changed in (
        ("model", "fixture-model-v2"),
        ("seed", 37),
        ("temperature", 0.2),
        ("top_p", 0.9),
        ("stop", ["STOP"]),
        ("n_predict", 8),
        ("json_schema", changed_schema),
    ):
        replayer = RecordReplayModelClient(
            cassette_path=cassette_path,
            mode="replay",
            delegate=FakeModelClient(),
            request_metadata={"model_name": "fixture-model", "model_version": "v1"},
        )
        changed_request = dict(request)
        changed_request[field] = changed
        with pytest.raises(MissingReplayEntryError):
            replayer.send_completion(changed_request)


def test_record_mode_merges_entries_from_clients_with_stale_initial_views(tmp_path: Path) -> None:
    cassette_path = tmp_path / "shared-cache.json"
    contract = yes_no_contract()
    first = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=FakeModelClient(responses=["yes"]),
        request_metadata={"model_name": "fixture-model"},
    )
    second = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=FakeModelClient(responses=["no"]),
        request_metadata={"model_name": "fixture-model"},
    )
    first.send_completion(first.build_completion_request("first", max_tokens=4, contract=contract))
    second.send_completion(second.build_completion_request("second", max_tokens=4, contract=contract))

    payload = json.loads(cassette_path.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 2
    assert not list(tmp_path.glob(".*.tmp"))


def test_record_mode_forwards_progress_callback_on_cache_miss(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []

    class ProgressDelegate(FakeModelClient):
        def send_completion(self, payload, *, timeout_seconds=None, progress_callback=None):
            if progress_callback is not None:
                progress_callback({"completion_tokens": 1})
            return super().send_completion(
                payload,
                timeout_seconds=timeout_seconds,
                progress_callback=progress_callback,
            )

    client = RecordReplayModelClient(
        cassette_path=tmp_path / "progress.json",
        mode="record",
        delegate=ProgressDelegate(responses=["yes"]),
    )
    request = client.build_completion_request("progress", max_tokens=4, contract=yes_no_contract())
    client.send_completion(request, progress_callback=seen.append)
    assert seen == [{"completion_tokens": 1}]


def test_record_replay_transport_timeout_is_not_part_of_output_identity(tmp_path: Path) -> None:
    cassette_path = tmp_path / "timeout-independent.json"
    contract = yes_no_contract()
    recorder = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=FakeModelClient(responses=["yes"]),
        request_metadata={"model_name": "fixture-model"},
    )
    request = recorder.build_completion_request("same output request", max_tokens=4, contract=contract)
    recorder.send_completion(request, timeout_seconds=5)

    replayer = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="replay",
        delegate=FakeModelClient(),
        request_metadata={"model_name": "fixture-model"},
    )
    assert json.loads(replayer.send_completion(request, timeout_seconds=90).text) == {"answer": "yes"}


def test_offline_replay_reuses_recorded_model_fingerprint_only_for_matching_metadata(tmp_path: Path) -> None:
    cassette_path = tmp_path / "offline-model.json"
    contract = yes_no_contract()

    class IdentityModel(FakeModelClient):
        def __init__(self, *, identity, responses=None):
            super().__init__(responses=responses)
            self._identity = identity

        def cache_identity(self):
            return dict(self._identity)

    resolved_identity = {
        "status": "resolved",
        "configured_model_identity": "",
        "base_url": "http://127.0.0.1:14829",
        "completion_endpoint": "/completion",
        "profile_name": "small_fast",
        "server_properties_sha256": "abc123",
    }
    recorder = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=IdentityModel(identity=resolved_identity, responses=["yes"]),
        request_metadata={
            "model_base_url": "http://127.0.0.1:14829",
            "completion_endpoint": "/completion",
            "model_profile": "small_fast",
            "structured_output_mode": "server_schema",
            "configured_seed": 11,
            "cache_scope": "runtime",
        },
    )
    request = recorder.build_completion_request("offline replay", max_tokens=4, contract=contract)
    recorder.send_completion(request)

    unresolved_identity = {
        "status": "unresolved",
        "configured_model_identity": "",
        "base_url": "http://127.0.0.1:14829",
        "completion_endpoint": "/completion",
        "profile_name": "small_fast",
        "probe_error_type": "ConnectionError",
    }
    replayer = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="replay",
        delegate=IdentityModel(identity=unresolved_identity),
        request_metadata={
            "model_base_url": "http://127.0.0.1:14829",
            "completion_endpoint": "/completion",
            "model_profile": "small_fast",
            "structured_output_mode": "server_schema",
            "configured_seed": 11,
            "cache_scope": "runtime",
        },
    )
    assert json.loads(replayer.send_completion(request).text) == {"answer": "yes"}

    mismatched = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="replay",
        delegate=IdentityModel(identity=unresolved_identity),
        request_metadata={
            "model_base_url": "http://127.0.0.1:14829",
            "completion_endpoint": "/completion",
            "model_profile": "different_profile",
            "structured_output_mode": "server_schema",
            "configured_seed": 11,
            "cache_scope": "runtime",
        },
    )
    with pytest.raises(MissingReplayEntryError):
        mismatched.send_completion(request)
