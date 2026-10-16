from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest
import requests

from swaag.grammar import yes_no_contract
from swaag.model import LlamaCppClient
from swaag.model_cache import RecordReplayModelClient
from swaag.runtime import AgentRuntime
from swaag.types import PromptAssembly, PromptComponent


@pytest.fixture
def tokenizer_server():
    rows = []
    closed = set()
    condition = threading.Condition()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = body["content"]
            with condition:
                rows.append({"port": self.client_address[1], "body": body,
                             "authorization": self.headers.get("Authorization"),
                             "cookie": self.headers.get("Cookie")})
            payload = json.dumps({"tokens": list(range(len(text)))}).encode()
            self.send_response(400 if text == "fail-http" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Set-Cookie", "unwanted=retained; Path=/")
            self.end_headers()
            self.wfile.write(payload)

        def finish(self):
            try:
                super().finish()
            finally:
                with condition:
                    closed.add(self.client_address[1])
                    condition.notify_all()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def assert_closed(ports):
        with condition:
            assert condition.wait_for(lambda: set(ports) <= closed, timeout=2)

    yield f"http://127.0.0.1:{server.server_port}", rows, assert_closed
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_runtime_compilation_reuses_connection_without_changing_accounting(
    make_config, tokenizer_server, monkeypatch
):
    url, rows, assert_closed = tokenizer_server
    config = make_config(model__base_url=url, context__allow_estimate_fallback=False)
    runtime = AgentRuntime(config)
    # Exercise real context accounting, HTTP, and runtime boundary; prompt
    # serialization itself has separate exact-protocol integration coverage.
    monkeypatch.setattr(runtime, "_require_system_prompt", lambda assembly: None)
    monkeypatch.setattr(runtime, "_materialize_prompt_protocol", lambda assembly: None)
    assembly = PromptAssembly(
        kind="action", prompt_mode="standard", prompt_text="policy question evidence",
        components=[PromptComponent(name="system", text="policy "),
                    PromptComponent(name="user", text="question "),
                    PromptComponent(name="state", text="evidence")],
    )
    result = runtime._compile_context(
        None, assembly, yes_no_contract(), minimum_output_tokens=16,
        context_limit_resolution=(4096, "fixture"), include_prompt_instructions=False,
    )
    assert result.report.input_tokens == len(assembly.prompt_text)
    assert result.report.exact and result.report.fits
    assert len(rows) >= 4
    assert len({row["port"] for row in rows}) == 1
    assert_closed([row["port"] for row in rows])


def test_scopes_are_nested_bounded_and_do_not_retain_cookies(
    make_config, tokenizer_server, monkeypatch
):
    url, rows, assert_closed = tokenizer_server
    client = LlamaCppClient(make_config(model__base_url=url, model__api_key_env="TEST_TOKEN"))
    monkeypatch.setenv("TEST_TOKEN", "first")
    with client.tokenization_scope():
        assert client.tokenize("alpha") == 5
        monkeypatch.setenv("TEST_TOKEN", "second")
        with client.tokenization_scope():
            assert client.tokenize("beta") == 4
        assert client.tokenize("gamma") == 5
    first_port = rows[0]["port"]
    assert {row["port"] for row in rows} == {first_port}
    assert_closed([first_port])
    assert [row["authorization"] for row in rows] == ["Bearer first", "Bearer second", "Bearer second"]
    assert all(row["cookie"] is None for row in rows)
    with client.tokenization_scope():
        assert client.tokenize("delta") == 5
    assert rows[-1]["port"] != first_port
    assert_closed([rows[-1]["port"]])


def test_concurrent_scopes_on_one_client_have_distinct_connections(
    make_config, tokenizer_server
):
    url, rows, assert_closed = tokenizer_server
    client = LlamaCppClient(make_config(model__base_url=url))
    barrier = threading.Barrier(2)

    def worker(prefix):
        with client.tokenization_scope():
            barrier.wait(timeout=2)
            for i in range(3):
                assert client.tokenize(prefix+str(i)) == 2

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, prefix) for prefix in ("a", "b")]
        for future in futures:
            future.result(timeout=5)
    ports = [{row["port"] for row in rows if row["body"]["content"].startswith(prefix)}
             for prefix in ("a", "b")]
    assert len(ports[0]) == len(ports[1]) == 1
    assert ports[0].isdisjoint(ports[1])
    assert_closed(ports[0] | ports[1])


def test_http_failure_and_cancellation_close_owned_connections(
    make_config, tokenizer_server
):
    url, rows, assert_closed = tokenizer_server
    client = LlamaCppClient(make_config(model__base_url=url))
    with pytest.raises(requests.HTTPError):
        with client.tokenization_scope():
            client.tokenize("fail-http")
    assert len(rows) == 1  # No new retry policy.
    assert_closed([rows[-1]["port"]])
    with pytest.raises(KeyboardInterrupt):
        with client.tokenization_scope():
            client.tokenize("before-cancel")
            raise KeyboardInterrupt()
    assert_closed([rows[-1]["port"]])
    with client.tokenization_scope():
        assert client.tokenize("after-cancel") == 12
    assert len({row["port"] for row in rows}) == 3
    assert_closed([rows[-1]["port"]])


def test_record_replay_scope_keeps_identity_checks_and_offline_replay(
    make_config, tokenizer_server, monkeypatch, tmp_path
):
    url, rows, assert_closed = tokenizer_server
    client = LlamaCppClient(make_config(model__base_url=url))
    identities = []
    def identity(self):
        identities.append("probe")
        return {"status":"resolved", "model":"fixture"}
    monkeypatch.setattr(LlamaCppClient, "cache_identity", identity)
    cassette = tmp_path/"model.jsonl"
    wrapper = RecordReplayModelClient(cassette_path=cassette, mode="record", delegate=client)
    before = len(identities)
    with wrapper.tokenization_scope():
        assert wrapper.tokenize("alpha") == 5
        assert wrapper.tokenize("beta") == 4
    assert len(identities) >= before+2
    assert len({row["port"] for row in rows}) == 1
    assert_closed([row["port"] for row in rows])
    replay = RecordReplayModelClient(cassette_path=cassette, mode="replay", delegate=client)
    monkeypatch.setattr(LlamaCppClient, "tokenization_scope", lambda self: pytest.fail("offline delegate accessed"))
    with replay.tokenization_scope():
        assert replay.tokenize("alpha") == 5
        assert replay.tokenize("beta") == 4
    assert len(rows) == 2
