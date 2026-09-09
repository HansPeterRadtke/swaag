from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from swaag.grammar import tool_decision_contract, yes_no_contract
from swaag.model import LlamaCppClient, ModelClientError


class _Handler(BaseHTTPRequestHandler):
    requests = []
    malformed = False
    forced_error_payload: dict | None = None

    def log_message(self, format: str, *args):  # noqa: A003
        return

    def _json_response(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _stream_response(self, chunks: list[dict]) -> None:
        raw_parts = [f"data: {json.dumps(chunk)}\n\n".encode("utf-8") for chunk in chunks]
        raw_parts.append(b"data: [DONE]\n\n")
        raw = b"".join(raw_parts)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json_response({"status": "ok"})
            return
        self._json_response({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        type(self).requests.append((self.path, body))
        if self.path == "/tokenize":
            self._json_response({"tokens": list(range(len(body["content"].split())))})
            return
        if self.path == "/completion":
            if type(self).forced_error_payload is not None:
                self._json_response(type(self).forced_error_payload, status=400)
                return
            if type(self).malformed:
                self._json_response({"unexpected": True})
                return
            if "grammar" in body:
                self._stream_response([{"content": "y", "tokens_evaluated": 3}, {"content": "es", "stop": True, "tokens_predicted": 1}])
                return
            payload = json.dumps({"action": "respond", "response": "ok", "tool_name": "none", "tool_input": {}})
            self._stream_response([{"content": payload[:10], "tokens_evaluated": 6}, {"content": payload[10:], "stop": True, "tokens_predicted": 8}])
            return
        self._json_response({"error": "not found"}, status=404)


def test_llama_cpp_client_request_construction(make_config) -> None:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = make_config(model__base_url=f"http://127.0.0.1:{server.server_port}")
        client = LlamaCppClient(config)
        assert client.health()["status"] == "ok"
        assert client.tokenize("one two three") == 3
        grammar_request = client.build_completion_request("prompt", max_tokens=4, contract=yes_no_contract())
        assert "grammar" in grammar_request
        assert grammar_request["temperature"] == config.model.temperature
        assert grammar_request["seed"] == config.model.seed
        grammar_result = client.send_completion(grammar_request)
        assert grammar_result.text == "yes"
        schema_request = client.build_completion_request("prompt", max_tokens=32, contract=tool_decision_contract(["echo"]))
        assert "json_schema" in schema_request
        schema_result = client.send_completion(schema_request)
        assert json.loads(schema_result.text)["response"] == "ok"
        completion_requests = [body for path, body in _Handler.requests if path == "/completion"]
        assert any("grammar" in item for item in completion_requests)
        assert any("json_schema" in item for item in completion_requests)
        assert all(item.get("stream") is True for item in completion_requests)
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_llama_cpp_client_rejects_malformed_response(make_config) -> None:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    _Handler.malformed = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = make_config(model__base_url=f"http://127.0.0.1:{server.server_port}")
        client = LlamaCppClient(config)
        with pytest.raises(ModelClientError):
            client.send_completion(client.build_completion_request("prompt", max_tokens=4, contract=yes_no_contract()))
    finally:
        _Handler.malformed = False
        server.shutdown()
        thread.join(timeout=5)


def test_llama_cpp_client_surfaces_http_error_details(make_config) -> None:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    _Handler.forced_error_payload = {
        "error": {
            "code": 400,
            "type": "exceed_context_size_error",
            "message": "the request exceeds the available context size, try increasing it",
        }
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = make_config(model__base_url=f"http://127.0.0.1:{server.server_port}")
        client = LlamaCppClient(config)
        with pytest.raises(requests.HTTPError, match="exceed_context_size_error"):
            client.send_completion(client.build_completion_request("prompt", max_tokens=4, contract=yes_no_contract()))
    finally:
        _Handler.forced_error_payload = None
        server.shutdown()
        thread.join(timeout=5)


def test_llama_cpp_client_surfaces_timeout(make_config, monkeypatch) -> None:
    config = make_config()
    client = LlamaCppClient(config)

    def _timeout(*args, **kwargs):
        raise requests.Timeout("boom")

    monkeypatch.setattr(requests, "post", _timeout)
    with pytest.raises(requests.Timeout):
        client.send_completion(client.build_completion_request("prompt", max_tokens=4, contract=yes_no_contract()))


def test_llama_cpp_client_streaming_timeout_uses_call_timeout_and_closes_response(make_config, monkeypatch) -> None:
    config = make_config(model__connect_timeout_seconds=3, model__progress_poll_seconds=1.0)
    client = LlamaCppClient(config)
    captured: dict[str, object] = {}

    class _Response:
        def __init__(self) -> None:
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.closed = True

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, decode_unicode: bool = True):
            del decode_unicode
            yield 'data: {"content":"ok","tokens_evaluated":1}'
            yield 'data: {"content":"","stop":true,"tokens_predicted":1}'
            yield "data: [DONE]"

    response = _Response()

    def _post(*args, **kwargs):
        del args
        captured["timeout"] = kwargs.get("timeout")
        return response

    monkeypatch.setattr(requests, "post", _post)

    result = client.send_completion(
        client.build_completion_request("prompt", max_tokens=4, contract=yes_no_contract()),
        timeout_seconds=7,
    )

    assert captured["timeout"] == (3, 7.0)
    assert response.closed is True
    assert result.text == "ok"


def test_request_policy_selects_timeout_by_contract_kind_and_profile(make_config) -> None:
    config = make_config(model__profile_name="small_fast", model__timeout_seconds=30, model__simple_timeout_seconds=20, model__structured_timeout_seconds=40, model__verification_timeout_seconds=50)
    client = LlamaCppClient(config)

    plain = client.select_request_policy(contract=yes_no_contract(), kind="answer", prompt="prompt", max_tokens=16)
    verify = client.select_request_policy(contract=tool_decision_contract(["echo"]), kind="verification", prompt="prompt", max_tokens=16)

    assert plain.effective_timeout_seconds == 40
    assert verify.effective_timeout_seconds == 50
    assert plain.profile_name == "small_fast"


def test_post_validate_mode_keeps_server_schema_enforcement(make_config) -> None:
    config = make_config(model__structured_output_mode="post_validate")
    client = LlamaCppClient(config)
    original = tool_decision_contract(["echo"])

    resolved, policy = client.resolve_contract(original, kind="decision", prompt="Return JSON only.", max_tokens=32)
    request = client.build_completion_request("Return JSON only.", max_tokens=32, contract=resolved)

    assert original.mode == "json_schema"
    assert resolved.mode == "json_schema"
    assert policy.effective_contract_mode == "json_schema"
    assert "json_schema" in request


def test_auto_mode_keeps_server_schema_on_mid_context_profile(make_config) -> None:
    config = make_config(model__profile_name="mid_context", model__structured_output_mode="auto")
    client = LlamaCppClient(config)
    original = tool_decision_contract(["echo"])

    resolved, policy = client.resolve_contract(original, kind="decision", prompt="Return JSON only.", max_tokens=32)
    request = client.build_completion_request("Return JSON only.", max_tokens=32, contract=resolved)

    assert resolved.mode == "json_schema"
    assert policy.effective_contract_mode == "json_schema"
    assert "json_schema" in request


def test_auto_mode_keeps_server_schema_on_small_fast_for_large_prompt(make_config) -> None:
    config = make_config(model__profile_name="small_fast", model__structured_output_mode="auto")
    client = LlamaCppClient(config)
    original = tool_decision_contract(["echo"])
    prompt = "Return JSON only. " + ("x" * 1400)

    resolved, policy = client.resolve_contract(original, kind="decision", prompt=prompt, max_tokens=128)
    request = client.build_completion_request(prompt, max_tokens=128, contract=resolved)

    assert resolved.mode == "json_schema"
    assert policy.effective_contract_mode == "json_schema"
    assert "json_schema" in request
