from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from swaag.model_cache import MissingReplayEntryError, RecordReplayModelClient
from swaag.tokens import ConservativeEstimator
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
                context_limit=2048,
            )
        )
        self.context_probes = 0
        self.cancel_checks = []
        self.render_calls = 0
        self.identity_calls = 0
        self.token_calls = 0

    def cache_identity(self):
        self.identity_calls += 1
        return {
            "status": "resolved",
            "configured_model_identity": "model-a",
            "server_properties_sha256": self.identity_version,
        }

    def context_limit_resolution(self):
        self.context_probes += 1
        return 22016, "server_props:n_ctx"

    def render_chat_prompt(self, messages):
        self.render_calls += 1
        return {
            "prompt": f"SYS:{messages[0]['content']} USER:{messages[1]['content']}",
            "chat_template_sha256": "a" * 64,
            "prompt_protocol_sha256": "b" * 64,
        }

    def tokenize(self, text):
        self.token_calls += 1
        return len(text)

    def send_completion(
        self,
        payload,
        *,
        timeout_seconds=None,
        progress_callback=None,
        cancel_check=None,
    ):
        self.calls += 1
        self.cancel_checks.append(cancel_check)
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


def test_chat_template_rendering_records_and_replays_without_network(
    tmp_path: Path,
) -> None:
    recorded_delegate = Delegate()
    recorded = client(tmp_path, recorded_delegate)
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "task"},
    ]
    first = recorded.render_chat_prompt(messages)
    assert recorded_delegate.render_calls == 1
    assert recorded.render_chat_prompt(messages) == first
    assert recorded_delegate.render_calls == 1

    replay_delegate = Delegate()
    replay = RecordReplayModelClient(
        cassette_path=tmp_path / "cassette.json",
        mode="replay",
        delegate=replay_delegate,
        request_metadata={"task": "cache-exactness"},
    )

    assert replay.render_chat_prompt(messages) == first
    assert replay_delegate.render_calls == 0
    assert replay_delegate.identity_calls == 0


def test_token_counts_record_and_replay_without_network(tmp_path: Path) -> None:
    recorded_delegate = Delegate()
    recorded = client(tmp_path, recorded_delegate)

    assert recorded.tokenize("exact text") == 10
    assert recorded.tokenize("exact text") == 10
    assert recorded_delegate.token_calls == 1

    replay_delegate = Delegate()
    replay = RecordReplayModelClient(
        cassette_path=tmp_path / "cassette.json",
        mode="replay",
        delegate=replay_delegate,
        request_metadata={"task": "cache-exactness"},
    )

    assert replay.tokenize("exact text") == 10
    assert replay_delegate.token_calls == 0


def test_estimated_token_counts_are_disclosed_and_not_recorded_as_exact(
    tmp_path: Path,
) -> None:
    class OpaqueDelegate(Delegate):
        def tokenize(self, text):
            self.token_calls += 1
            raise RuntimeError("provider serialization is opaque")

        def count_text(self, text):
            return ConservativeEstimator().count_text(text)

    recorded_delegate = OpaqueDelegate()
    recorded = client(tmp_path, recorded_delegate)

    estimate = recorded.count_text("opaque provider input")

    assert estimate.exact is False
    assert estimate.strategy == "chars_per_token"

    replay_delegate = OpaqueDelegate()
    replay = RecordReplayModelClient(
        cassette_path=tmp_path / "cassette.json",
        mode="replay",
        delegate=replay_delegate,
        request_metadata={"task": "cache-exactness"},
    )
    replay_estimate = replay.count_text("opaque provider input")

    assert replay_estimate.exact is False
    assert replay_delegate.token_calls == 0
    assert replay_delegate.identity_calls == 0
    with pytest.raises(MissingReplayEntryError, match="tokenize result"):
        replay.tokenize("not recorded")
    assert replay_delegate.token_calls == 0


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


def test_record_mode_uses_live_capacity_but_replay_uses_configured_fallback(tmp_path: Path) -> None:
    delegate = Delegate()
    record_client = client(tmp_path, delegate)
    assert record_client.context_limit_resolution() == (22016, "server_props:n_ctx")
    assert delegate.context_probes == 1

    replay_client = RecordReplayModelClient(
        cassette_path=tmp_path / "cassette.json",
        mode="replay",
        delegate=delegate,
    )
    assert replay_client.context_limit_resolution() == (2048, "configured:replay")
    assert delegate.context_probes == 1


def test_record_mode_propagates_cooperative_cancellation(tmp_path: Path) -> None:
    delegate = Delegate()
    cached = client(tmp_path, delegate)
    cancellation = lambda: False

    cached.send_completion(base_payload(), cancel_check=cancellation)

    assert delegate.cancel_checks == [cancellation]
