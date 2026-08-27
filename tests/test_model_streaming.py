from __future__ import annotations

import json
import threading

import pytest

from swaag.grammar import yes_no_contract
from swaag.model import LlamaCppClient
from swaag.preemption import ModelCallPreempted


class _FakeResponse:
    status_code = 200
    text = ""

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, *, decode_unicode: bool):
        assert decode_unicode is True
        yield ":"
        yield ": keepalive"
        yield 'data: ' + json.dumps({"content": "hel", "tokens_predicted": 1})
        yield ''
        yield 'data: ' + json.dumps({"content": "lo", "tokens_predicted": 2, "stop": True})
        yield 'data: [DONE]'


class _LimitedResponse(_FakeResponse):
    def iter_lines(self, *, decode_unicode: bool):
        assert decode_unicode is True
        yield 'data: ' + json.dumps({"content": "{", "tokens_predicted": 1})
        yield 'data: ' + json.dumps(
            {
                "content": "",
                "tokens_predicted": 1,
                "stop": True,
                "stop_type": "limit",
                "truncated": False,
            }
        )


class _CloseInterruptedResponse(_FakeResponse):
    def __init__(self) -> None:
        self.closed = threading.Event()

    def close(self) -> None:
        self.closed.set()

    def iter_lines(self, *, decode_unicode: bool):
        assert decode_unicode is True
        self.closed.wait(timeout=1)
        raise AttributeError("stream reader disappeared during close")
        yield ""  # pragma: no cover - keep this method an iterator


def test_streaming_client_ignores_sse_comments_and_keepalives(make_config, monkeypatch) -> None:
    config = make_config()
    client = LlamaCppClient(config)
    captured = {}

    def fake_post(url, *, json, timeout, stream):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        captured["stream"] = stream
        return _FakeResponse()

    monkeypatch.setattr("swaag.model.requests.post", fake_post)
    result = client.send_completion({"prompt": "x", "n_predict": 16})
    assert result.text == "hello"
    assert result.completion_tokens == 2
    assert captured["stream"] is True
    assert captured["payload"]["stream"] is True


def test_streaming_client_preserves_output_limit_finish_reason(make_config, monkeypatch) -> None:
    client = LlamaCppClient(make_config())
    monkeypatch.setattr(
        "swaag.model.requests.post",
        lambda *args, **kwargs: _LimitedResponse(),
    )

    result = client.send_completion({"prompt": "x", "n_predict": 1})

    assert result.finish_reason == "length"
    assert result.raw_response["stop_type"] == "limit"


def test_close_induced_transport_error_is_reported_as_preemption(
    make_config, monkeypatch
) -> None:
    response = _CloseInterruptedResponse()
    client = LlamaCppClient(make_config())
    monkeypatch.setattr(
        "swaag.model.requests.post",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(ModelCallPreempted):
        client.send_completion(
            {"prompt": "x", "n_predict": 16},
            cancel_check=lambda: True,
            cancel_poll_seconds=0.001,
        )

    assert response.closed.is_set()


def test_long_live_verification_uses_benchmark_timeout(make_config) -> None:
    config = make_config(
        model__timeout_seconds=10,
        model__verification_timeout_seconds=150,
        model__benchmark_timeout_seconds=360,
    )
    client = LlamaCppClient(config)

    policy = client.select_request_policy(
        contract=yes_no_contract(),
        kind="verification",
        prompt="x" * 1201,
        max_tokens=64,
        live_mode=True,
    )

    assert policy.effective_timeout_seconds == 360


def test_short_live_verification_keeps_verification_timeout(make_config) -> None:
    config = make_config(
        model__timeout_seconds=10,
        model__verification_timeout_seconds=150,
        model__benchmark_timeout_seconds=360,
    )
    client = LlamaCppClient(config)

    policy = client.select_request_policy(
        contract=yes_no_contract(),
        kind="verification",
        prompt="short",
        max_tokens=64,
        live_mode=True,
    )

    assert policy.effective_timeout_seconds == 150
