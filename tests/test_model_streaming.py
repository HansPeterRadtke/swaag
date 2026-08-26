from __future__ import annotations

import json

from swaag.model import LlamaCppClient


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
