from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from swaag.grammar import tool_decision_contract, yes_no_contract
from swaag.model import LlamaCppClient, ModelClientError, completion_url
from swaag.types import ContractSpec


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
            schema = body.get("json_schema") or {}
            properties = set((schema.get("properties") or {}).keys())
            if properties == {"answer"}:
                self._json_response({"content": json.dumps({"answer": "yes"}), "stop": True, "tokens_evaluated": 3, "tokens_predicted": 4})
                return
            self._json_response({"content": json.dumps({"action": "respond", "response": "ok", "tool_name": "none", "tool_input": {}}), "stop": True, "tokens_evaluated": 6, "tokens_predicted": 8})
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
        yes_no_request = client.build_completion_request("prompt", max_tokens=4, contract=yes_no_contract())
        assert "json_schema" in yes_no_request
        assert yes_no_request["temperature"] == config.model.temperature
        assert yes_no_request["seed"] == config.model.seed
        yes_no_result = client.send_completion(yes_no_request)
        assert json.loads(yes_no_result.text)["answer"] == "yes"
        schema_request = client.build_completion_request("prompt", max_tokens=32, contract=tool_decision_contract(["echo"]))
        assert "json_schema" in schema_request
        schema_result = client.send_completion(schema_request)
        assert json.loads(schema_result.text)["response"] == "ok"
        completion_requests = [body for path, body in _Handler.requests if path == "/completion"]
        assert any((item.get("json_schema") or {}).get("properties", {}).keys() == {"answer"} for item in completion_requests)
        assert any("json_schema" in item for item in completion_requests)
        assert not any("grammar" in item for item in completion_requests)
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_openai_compatible_request_uses_chat_completions_structured_outputs(make_config) -> None:
    config = make_config(
        model__base_url="https://openrouter.ai/api/v1",
        model__profile_name="openai/gpt-4o-mini",
    )
    client = LlamaCppClient(config)
    contract = tool_decision_contract(["echo"])

    request = client.build_completion_request("prompt", max_tokens=64, contract=contract)

    assert completion_url(config.model.base_url, config.model.completion_endpoint) == "https://openrouter.ai/api/v1/chat/completions"
    assert request["model"] == "openai/gpt-4o-mini"
    assert request["messages"] == [{"role": "user", "content": "prompt"}]
    assert request["max_tokens"] == 64
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"] == contract.json_schema
    assert request["provider"] == {"require_parameters": True}
    assert "json_schema" not in request
    assert "grammar" not in request


def test_live_model_client_rejects_unconstrained_contracts(make_config) -> None:
    client = LlamaCppClient(make_config())
    contract = ContractSpec(name="bad", mode="plain")  # type: ignore[arg-type]

    with pytest.raises(ModelClientError, match="json_schema"):
        client.build_completion_request("prompt", max_tokens=4, contract=contract)


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


def test_llama_cpp_client_stream_timeout_uses_policy_timeout(make_config, monkeypatch) -> None:
    config = make_config(model__connect_timeout_seconds=3)
    client = LlamaCppClient(config)
    seen: dict[str, object] = {}

    def _timeout(*args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise requests.Timeout("boom")

    monkeypatch.delenv("SWAAG_MODEL_TOKEN_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(requests, "post", _timeout)
    with pytest.raises(requests.Timeout):
        client.send_completion(client.build_completion_request("prompt", max_tokens=4, contract=yes_no_contract()), timeout_seconds=180)

    assert seen["timeout"] == (3, 180.0)


def test_llama_cpp_client_stream_timeout_env_override_wins(make_config, monkeypatch) -> None:
    config = make_config(model__connect_timeout_seconds=3)
    client = LlamaCppClient(config)
    seen: dict[str, object] = {}

    def _timeout(*args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise requests.Timeout("boom")

    monkeypatch.setenv("SWAAG_MODEL_TOKEN_TIMEOUT_SECONDS", "7.5")
    monkeypatch.setattr(requests, "post", _timeout)
    with pytest.raises(requests.Timeout):
        client.send_completion(client.build_completion_request("prompt", max_tokens=4, contract=yes_no_contract()), timeout_seconds=180)

    assert seen["timeout"] == (3, 7.5)


def test_request_policy_selects_timeout_by_contract_kind_and_profile(make_config) -> None:
    config = make_config(model__profile_name="small_fast", model__timeout_seconds=30, model__simple_timeout_seconds=20, model__structured_timeout_seconds=40, model__verification_timeout_seconds=50)
    client = LlamaCppClient(config)

    plain = client.select_request_policy(contract=yes_no_contract(), kind="answer", prompt="prompt", max_tokens=16)
    verify = client.select_request_policy(contract=tool_decision_contract(["echo"]), kind="verification", prompt="prompt", max_tokens=16)

    assert plain.effective_timeout_seconds == 40
    assert verify.effective_timeout_seconds == 50
    assert plain.profile_name == "small_fast"


def test_server_schema_mode_uses_schema_enforcement(make_config) -> None:
    config = make_config(model__structured_output_mode="server_schema")
    client = LlamaCppClient(config)
    original = tool_decision_contract(["echo"])

    resolved, policy = client.resolve_contract(original, kind="decision", prompt="Return JSON only.", max_tokens=32)
    request = client.build_completion_request("Return JSON only.", max_tokens=32, contract=resolved)

    assert original.mode == "json_schema"
    assert resolved.mode == "json_schema"
    assert policy.effective_contract_mode == "json_schema"
    assert "json_schema" in request


def test_non_server_schema_mode_is_rejected(make_config) -> None:
    config = make_config(model__structured_output_mode="auto")
    client = LlamaCppClient(config)

    with pytest.raises(ModelClientError, match="server_schema"):
        client.resolve_contract(tool_decision_contract(["echo"]), kind="decision", prompt="Return JSON only.", max_tokens=32)
