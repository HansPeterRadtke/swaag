from __future__ import annotations

import json
import threading

import pytest
import requests

from swaag.environment.artifacts import TextArtifactStore
from swaag.grammar import yes_no_contract
from swaag.model import LlamaCppClient, ModelHTTPError, ModelClientError
from swaag.preemption import ModelCallPreempted
from swaag.runtime import AgentRuntime


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


class _ErrorResponse(_FakeResponse):
    status_code = 500

    def __init__(self, text: str) -> None:
        self.text = text
        self.closed = False
        self.request = object()

    def raise_for_status(self) -> None:
        raise requests.HTTPError(
            "500 Server Error",
            request=self.request,
            response=self,
        )

    def close(self) -> None:
        self.closed = True


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


def test_http_failure_keeps_exact_body_behind_bounded_exception(
    make_config, monkeypatch
) -> None:
    body = "provider-error-start\n" + ("complete-body-segment\n" * 200)
    response = _ErrorResponse(body)
    client = LlamaCppClient(make_config())
    monkeypatch.setattr(
        "swaag.model.requests.post",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(ModelHTTPError) as captured:
        client.send_completion({"prompt": "x", "n_predict": 16})

    assert captured.value.response_body == body
    assert len(str(captured.value)) < 550
    assert "exact_body_chars=" in str(captured.value)
    assert response.closed is True


def test_runtime_commits_exact_model_http_failure_artifact(
    make_config, tmp_path
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    response = _ErrorResponse("unused")
    body = "exact provider body\n" * 200
    error = ModelHTTPError(
        "bounded failure",
        response_body=body,
        request=response.request,
        response=response,
    )
    guard = runtime.history.guard(state, "test:model_failure")

    evidence = runtime._record_model_failure_evidence(state, guard, error)

    artifact = TextArtifactStore(config.sessions.root, state.session_id).read(
        evidence["artifact_id"], max_chars=len(body) + 1
    )
    assert artifact["text"] == body
    assert evidence["http_status"] == 500
    assert runtime.history.read_history(state.session_id)[-1].event_type == "artifact_created"


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


class _UsageResponse(_FakeResponse):
    def iter_lines(self, *, decode_unicode: bool):
        assert decode_unicode is True
        yield 'data: ' + json.dumps(
            {
                "content": "a",
                "tokens_evaluated": 12,
                "tokens_predicted": 1,
            }
        )
        yield 'data: ' + json.dumps(
            {
                "content": "b",
                "tokens_evaluated": 12,
                "tokens_predicted": 2,
                "stop": True,
            }
        )


def test_streaming_progress_exposes_backend_reported_usage_evidence(
    make_config, monkeypatch
) -> None:
    client = LlamaCppClient(make_config())
    monkeypatch.setattr(
        "swaag.model.requests.post",
        lambda *args, **kwargs: _UsageResponse(),
    )
    progress = []

    result = client.send_completion(
        {"prompt": "x", "n_predict": 4},
        progress_callback=progress.append,
    )

    assert result.text == "ab"
    assert progress
    assert progress[-1]["backend_prompt_tokens"] == 12
    assert progress[-1]["backend_completion_tokens"] == 2
    assert progress[-1]["prompt_tokens_source"] == "tokens_evaluated"
    assert progress[-1]["completion_tokens_source"] == "tokens_predicted"


@pytest.mark.parametrize("inferred_encoding", [None, "ISO-8859-1", "utf-8"])
def test_sse_preserves_utf8_with_split_bytes_and_inferred_encoding(
    make_config, monkeypatch, inferred_encoding
):
    import io

    class ByteChunks(io.BytesIO):
        def read(self, size=-1):
            return super().read(1 if size != 0 else 0)

    expected = "Grüße — 東京 🚀"
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "text/event-stream"
    response.encoding = inferred_encoding
    response.raw = ByteChunks((
        ": keepalive\n\ndata: "
        + json.dumps({"content": expected, "stop": True}, ensure_ascii=False)
        + "\n\ndata: [DONE]\n\n"
    ).encode("utf-8"))
    monkeypatch.setattr("swaag.model.requests.post", lambda *args, **kwargs: response)

    result = LlamaCppClient(make_config()).send_completion({"prompt": "x", "n_predict": 64})

    assert result.text == expected
    assert result.raw_response["content"] == expected
    assert response.raw.closed


class _ChatUsageResponse(_FakeResponse):
    def __init__(self, events):
        self.events = events

    def iter_lines(self, *, decode_unicode):
        for event in self.events:
            yield "data: " + json.dumps(event)
        yield "data: [DONE]"
        raise AssertionError("The client must stop at the SSE terminator")


def _chat_chunk(content="", *, finish=None):
    return {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]}


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_chat_stream_retains_trailing_usage_and_answer_finish_reason(
    make_config, monkeypatch, finish
):
    client = LlamaCppClient(make_config(model__base_url="https://inference.example/v1"))
    usage = {"prompt_tokens": 90, "completion_tokens": 11, "total_tokens": 101,
             "prompt_tokens_details": {"cached_tokens": 32}}
    response = _ChatUsageResponse([
        _chat_chunk("several tokens in one event"),
        _chat_chunk(finish=finish),
        {"choices": [], "usage": usage},
    ])
    payloads = []
    def post(url, **kwargs):
        payloads.append(kwargs["json"])
        return response
    monkeypatch.setattr("swaag.model.requests.post", post)
    progress = []
    result = client._send_completion({"messages": [{"role": "user", "content": "x"}]},
                                     progress_callback=progress.append)
    assert payloads[0]["stream_options"]["include_usage"] is True
    assert result.text == "several tokens in one event"
    assert result.prompt_tokens == 90
    assert result.completion_tokens == 11
    assert result.finish_reason == finish
    assert result.raw_response["usage"] == usage
    assert result.raw_response["completion_tokens_source"] == "usage.completion_tokens"
    assert progress[-1]["backend_prompt_tokens"] == 90
    assert progress[-1]["backend_completion_tokens"] == 11
    assert progress[-1]["prompt_tokens_source"] == "usage.prompt_tokens"
    assert progress[-1]["completion_tokens_source"] == "usage.completion_tokens"


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": True, "completion_tokens": False},
                                   {"prompt_tokens": -1, "completion_tokens": "11"}])
def test_chat_missing_or_invalid_usage_is_explicit_event_count_fallback(
    make_config, monkeypatch, usage
):
    client = LlamaCppClient(make_config(model__base_url="https://inference.example/v1"))
    response = _ChatUsageResponse([_chat_chunk("hello"), _chat_chunk(finish="stop"),
                                   {"choices": [], "usage": usage}])
    monkeypatch.setattr("swaag.model.requests.post", lambda *a, **kw: response)
    result = client._send_completion({"messages": [{"role": "user", "content": "x"}]})
    assert result.prompt_tokens is None
    assert result.completion_tokens == 1
    assert result.finish_reason == "stop"
    assert result.raw_response["completion_tokens_source"] == "stream_event_count"


def test_native_stream_does_not_request_chat_usage(make_config, monkeypatch):
    payloads = []
    def post(url, **kwargs):
        payloads.append(kwargs["json"])
        return _FakeResponse()
    monkeypatch.setattr("swaag.model.requests.post", post)
    result = LlamaCppClient(make_config())._send_completion({"prompt": "x", "n_predict": 4})
    assert "stream_options" not in payloads[0]
    assert result.completion_tokens == 2
    assert result.raw_response["completion_tokens_source"] == "tokens_predicted"



def test_chat_error_after_partial_content_is_not_a_success(make_config, monkeypatch):
    client = LlamaCppClient(make_config(model__base_url="https://inference.example/v1"))
    response = _ChatUsageResponse([_chat_chunk("partial"), {"error": {"message": "engine failed"}}])
    monkeypatch.setattr("swaag.model.requests.post", lambda *a, **kw: response)
    with pytest.raises(ModelClientError, match="engine failed"):
        client._send_completion({"messages": [{"role": "user", "content": "x"}]})


def test_chat_without_usage_trailer_or_done_can_finish_at_eof(make_config, monkeypatch):
    class EOFResponse(_FakeResponse):
        def iter_lines(self, *, decode_unicode):
            yield "data: " + json.dumps(_chat_chunk("answer", finish="stop"))
    client = LlamaCppClient(make_config(model__base_url="https://inference.example/v1"))
    monkeypatch.setattr("swaag.model.requests.post", lambda *a, **kw: EOFResponse())
    result = client._send_completion({"messages": [{"role": "user", "content": "x"}]})
    assert result.text == "answer"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens is None
    assert result.raw_response["completion_tokens_source"] == "stream_event_count"

def test_live_stream_has_absolute_deadline_even_with_frequent_tokens(make_config, monkeypatch):
    """A six-hour continuously progressing unclosed JSON response must stop."""
    class _HardDeadlineResponse(_CloseInterruptedResponse):
        def iter_lines(self, *, decode_unicode: bool):
            import time
            assert decode_unicode
            index = 0
            while not self.closed.is_set():
                time.sleep(.005)
                index += 1
                # Real model progress never stops; inactivity watchdog alone
                # would permit unbounded multi-hour output.
                yield 'data: ' + json.dumps({"content":"x", "tokens_predicted":index})
            raise AttributeError("stream reader disappeared after deadline")

    response = _HardDeadlineResponse()
    config = make_config(model__max_generation_wall_seconds=1.0)
    client = LlamaCppClient(config)
    monkeypatch.setattr("swaag.model.requests.post", lambda *args, **kwargs: response)
    with pytest.raises(requests.ReadTimeout, match="absolute 1.0s wall deadline"):
        client.send_completion(
            {"prompt": "x", "n_predict": 100000},
            cancel_poll_seconds=.005,
        )
    assert response.closed.is_set()


def test_generation_deadline_is_validated(make_config, monkeypatch):
    config = make_config(model__max_generation_wall_seconds=-2)
    client = LlamaCppClient(config)
    with pytest.raises(ValueError, match="max_generation_wall_seconds"):
        client._send_completion({"prompt": "x"}, cancel_poll_seconds=.005)
