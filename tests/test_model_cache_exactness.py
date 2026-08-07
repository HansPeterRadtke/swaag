from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from swaag.model_cache import RecordReplayModelClient
from swaag.types import CompletionResult


class Delegate:
    def __init__(self) -> None:
        self.calls = 0
        self.identity_version = "v1"
        self.config = SimpleNamespace(
            model=SimpleNamespace(
                base_url="http://model:14829",
                completion_endpoint="/completion",
                profile_name="model-a",
                structured_output_mode="server_schema",
                seed=42,
            )
        )

    def cache_identity(self):
        return {
            "status": "resolved",
            "configured_model_identity": "model-a",
            "server_properties_sha256": self.identity_version,
        }

    def send_completion(self, payload, *, timeout_seconds=None, progress_callback=None):
        self.calls += 1
        return CompletionResult(
            text=f"response-{self.calls}",
            raw_request=payload,
            raw_response={"content": f"response-{self.calls}"},
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
        )


def base_payload() -> dict:
    return {
        "prompt": "prompt-a",
        "n_predict": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stop": ["STOP"],
        "json_schema": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        "model": "model-a",
    }


def client(tmp_path: Path, delegate: Delegate) -> RecordReplayModelClient:
    return RecordReplayModelClient(
        cassette_path=tmp_path / "cassette.json",
        mode="record",
        delegate=delegate,
        request_metadata={"task": "cache-exactness"},
    )


def test_identical_request_replays_and_every_output_field_change_misses(tmp_path: Path) -> None:
    delegate = Delegate()
    cached = client(tmp_path, delegate)
    base = base_payload()
    assert cached.send_completion(copy.deepcopy(base)).text == "response-1"
    assert delegate.calls == 1
    assert cached.send_completion(copy.deepcopy(base)).text == "response-1"
    assert delegate.calls == 1

    mutations = [
        ("prompt", "prompt-b"),
        ("n_predict", 129),
        ("temperature", 0.1),
        ("top_p", 0.9),
        ("seed", 43),
        ("stop", ["OTHER"]),
        ("json_schema", {"type": "object", "properties": {"y": {"type": "integer"}}, "required": ["y"]}),
        ("model", "model-b"),
    ]
    for field, value in mutations:
        changed = copy.deepcopy(base)
        changed[field] = value
        before = delegate.calls
        cached.send_completion(changed)
        assert delegate.calls == before + 1, field
        cached.send_completion(copy.deepcopy(changed))
        assert delegate.calls == before + 1, field


def test_model_identity_refreshes_before_every_lookup(tmp_path: Path) -> None:
    delegate = Delegate()
    cached = client(tmp_path, delegate)
    payload = base_payload()
    assert cached.send_completion(payload).text == "response-1"
    assert delegate.calls == 1
    delegate.identity_version = "v2"
    assert cached.send_completion(payload).text == "response-2"
    assert delegate.calls == 2
    assert cached.send_completion(payload).text == "response-2"
    assert delegate.calls == 2


def test_cassette_is_reloaded_from_disk_before_every_lookup(tmp_path: Path) -> None:
    first_delegate = Delegate()
    first = client(tmp_path, first_delegate)
    payload = base_payload()
    assert first.send_completion(payload).text == "response-1"
    assert first_delegate.calls == 1

    second_delegate = Delegate()
    second = client(tmp_path, second_delegate)
    assert second.send_completion(payload).text == "response-1"
    assert second_delegate.calls == 0

    cassette = tmp_path / "cassette.json"
    data = json.loads(cassette.read_text())
    data["entries"] = []
    cassette.write_text(json.dumps(data))
    assert second.send_completion(payload).text == "response-1"
    assert second_delegate.calls == 1
