from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from swaag.config import ExternalMcpServerConfig, ExternalToolsConfig
from swaag.delegated_tools import prepare_delegated_tool_spec
from swaag.external_mcp import ExternalMcpClient, ExternalMcpError, ExternalMcpManager
from swaag.runtime import AgentRuntime


def _write_fake_mcp_server(path: Path) -> None:
    path.write_text(
        '''import json, sys\n'''
        '''for line in sys.stdin:\n'''
        '''    if not line.strip(): continue\n'''
        '''    req=json.loads(line); method=req["method"]; rid=req["id"]\n'''
        '''    if method=="tools/list": result={"resultType":"complete","tools":[{"name":"dummy_lookup","description":"Look up a dummy external value.","inputSchema":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"],"additionalProperties":False}}]}\n'''
        '''    elif method=="tools/call": result={"resultType":"complete","content":[{"type":"text","text":"VALUE:"+req["params"]["arguments"]["key"]}],"structuredContent":{"value":"VALUE:"+req["params"]["arguments"]["key"]},"isError":False}\n'''
        '''    else: result={"resultType":"complete"}\n'''
        '''    print(json.dumps({"jsonrpc":"2.0","id":rid,"result":result}), flush=True)\n'''
    )


def test_external_mcp_stdio_discovers_and_calls_schema_driven_tool(tmp_path: Path) -> None:
    server = tmp_path / "fake_mcp.py"
    _write_fake_mcp_server(server)
    config = ExternalMcpServerConfig(
        enabled=True,
        optional=False,
        transport="stdio",
        command=[sys.executable, str(server)],
        url="",
        header_env={},
        credential_command=[],
        credential_refresh_skew_seconds=30.0,
        timeout_seconds=5.0,
    )
    client = ExternalMcpClient("dummy", config)
    tools = client.list_tools()
    assert [tool.name for tool in tools] == ["dummy_lookup"]
    assert tools[0].metadata["external_execution_mode"] == "runtime"
    assert tools[0].metadata["external_provider_id"] == "mcp:dummy"
    assert tools[0].metadata["mcp_input_schema"]["required"] == ["key"]
    result = client.call_tool("dummy_lookup", {"key": "alpha"})
    assert result.is_error is False
    assert result.structured_content == {"value": "VALUE:alpha"}


def test_optional_external_mcp_server_disappears_from_catalog(tmp_path: Path) -> None:
    manager = ExternalMcpManager(
        ExternalToolsConfig(
            mcp_servers={
                "missing": ExternalMcpServerConfig(
                    enabled=True,
                    optional=True,
                    transport="stdio",
                    command=[str(tmp_path / "does-not-exist")],
                    url="",
                    header_env={},
                    credential_command=[],
                    credential_refresh_skew_seconds=30.0,
                    timeout_seconds=1.0,
                )
            }
        )
    )
    assert manager.specs() == ()
    assert "missing" in manager.discovery_errors


def test_required_external_mcp_server_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ExternalMcpError, match="unavailable"):
        ExternalMcpManager(
            ExternalToolsConfig(
                mcp_servers={
                    "missing": ExternalMcpServerConfig(
                        enabled=True,
                        optional=False,
                        transport="stdio",
                        command=[str(tmp_path / "does-not-exist")],
                        url="",
                        header_env={},
                        credential_command=[],
                        credential_refresh_skew_seconds=30.0,
                        timeout_seconds=1.0,
                    )
                }
            )
        )


def test_runtime_mcp_tool_is_external_not_system_and_records_normal_tool_result(
    make_config, tmp_path: Path
) -> None:
    server = tmp_path / "fake_mcp.py"
    _write_fake_mcp_server(server)
    config = make_config()
    config.external_tools.mcp_servers["dummy"] = ExternalMcpServerConfig(
        enabled=True,
        optional=False,
        transport="stdio",
        command=[sys.executable, str(server)],
        url="",
        header_env={},
        credential_command=[],
        credential_refresh_skew_seconds=30.0,
        timeout_seconds=5.0,
    )
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    specs = runtime.runtime_external_tools.specs()
    assert [spec.name for spec in specs] == ["dummy_lookup"]
    assert "dummy_lookup" not in runtime.tools.system_tool_names()
    index = runtime.tools.capability_index(config, specs)
    assert any(name == "dummy_lookup" for name, _description, _guidance in index)
    staged = runtime.tools.staged_prompt_tuples(config, ["dummy_lookup"], specs)
    dummy = next(item for item in staged if item[0] == "dummy_lookup")
    assert dummy[2]["required"] == ["key"]
    assert "external tool" in dummy[3].lower()
    assert "provider adapter" in dummy[3].lower()
    assert "MCP" not in dummy[3]

    result = runtime._execute_runtime_external_tool(
        state, spec=specs[0], arguments={"key": "beta"}
    )
    assert result is not None
    assert result.output["structured_content"] == {"value": "VALUE:beta"}
    tool_events = [
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "tool_result"
    ]
    assert tool_events[-1].payload["executor"] == "external_runtime"
    assert tool_events[-1].payload["tool_name"] == "dummy_lookup"


def test_mcp_schema_normalization_preserves_optional_semantics() -> None:
    from swaag.external_mcp import _mcp_execution_arguments, _portable_mcp_schema

    original = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            "engine": {"type": "string", "enum": ["auto", "bing"], "default": "auto"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    portable = _portable_mcp_schema(original, root=True)
    assert portable["required"] == ["query", "limit", "engine"]
    assert portable["properties"]["query"] == {"type": "string"}
    assert portable["properties"]["limit"] == {
        "anyOf": [{"type": "integer"}, {"type": "null"}]
    }
    assert portable["properties"]["engine"] == {
        "anyOf": [
            {"type": "string", "enum": ["auto", "bing"]},
            {"type": "null"},
        ]
    }
    assert _mcp_execution_arguments(
        {"query": "otters", "limit": None, "engine": None}, original
    ) == {"query": "otters"}


def test_mcp_schema_normalization_accepts_zero_argument_tool() -> None:
    from swaag.external_mcp import _portable_mcp_schema

    assert _portable_mcp_schema(
        {"type": "object", "properties": {}, "additionalProperties": False}, root=True
    ) == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_external_mcp_streamable_http_json_transport() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", "0"))
            request = json.loads(self.rfile.read(length))
            result = {
                "resultType": "complete",
                "tools": [
                    {
                        "name": "remote_ping",
                        "description": "Ping a remote external provider.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    }
                ],
            }
            body = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"], "result": result}
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ExternalMcpClient(
            "remote",
            ExternalMcpServerConfig(
                enabled=True,
                optional=False,
                transport="streamable_http",
                command=[],
                url=f"http://127.0.0.1:{server.server_port}/mcp",
                header_env={},
                credential_command=[],
                credential_refresh_skew_seconds=30.0,
                timeout_seconds=5.0,
            ),
        )
        tools = client.list_tools()
        assert [tool.name for tool in tools] == ["remote_ping"]
        assert tools[0].parameters["required"] == []
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_external_mcp_streamable_http_uses_header_environment(monkeypatch) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    observed = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            observed["authorization"] = self.headers.get("Authorization")
            length = int(self.headers.get("content-length", "0"))
            request = json.loads(self.rfile.read(length))
            body = json.dumps({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"resultType": "complete", "tools": []},
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SWAAG_TEST_MCP_AUTH", "Bearer external-secret")
    try:
        client = ExternalMcpClient(
            "remote-auth",
            ExternalMcpServerConfig(
                enabled=True,
                optional=False,
                transport="streamable_http",
                command=[],
                url=f"http://127.0.0.1:{server.server_port}/mcp",
                header_env={"Authorization": "SWAAG_TEST_MCP_AUTH"},
                credential_command=[],
                credential_refresh_skew_seconds=30.0,
                timeout_seconds=5.0,
            ),
        )
        assert client.list_tools() == ()
        assert observed["authorization"] == "Bearer external-secret"
        assert "external-secret" not in repr(client.config)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_external_mcp_missing_header_environment_fails_before_network(monkeypatch) -> None:
    monkeypatch.delenv("SWAAG_TEST_MISSING_MCP_AUTH", raising=False)
    client = ExternalMcpClient(
        "remote-auth",
        ExternalMcpServerConfig(
            enabled=True,
            optional=False,
            transport="streamable_http",
            command=[],
            url="http://127.0.0.1:1/mcp",
            header_env={"Authorization": "SWAAG_TEST_MISSING_MCP_AUTH"},
            credential_command=[],
            credential_refresh_skew_seconds=30.0,
            timeout_seconds=1.0,
        ),
    )
    with pytest.raises(ExternalMcpError, match="SWAAG_TEST_MISSING_MCP_AUTH"):
        client.list_tools()



def _write_credential_provider(path: Path, log_path: Path, *, expiring: bool = False) -> None:
    path.write_text(
        "import json,sys,time\n"
        f"log={str(log_path)!r}\n"
        "request=json.load(sys.stdin)\n"
        "with open(log,'a',encoding='utf-8') as f: f.write(request['reason']+'\\n')\n"
        "token='fresh-token' if request['reason']=='unauthorized' else 'stale-token'\n"
        + (
            "expires=time.time()+1\n"
            if expiring
            else "expires=time.time()+3600\n"
        )
        + "print(json.dumps({'headers':{'Authorization':'Bearer '+token},'expires_at_epoch':expires}))\n"
    )


def test_external_mcp_credential_provider_refreshes_after_unauthorized(tmp_path: Path) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    log_path = tmp_path / "credential.log"
    provider = tmp_path / "credential_provider.py"
    _write_credential_provider(provider, log_path)
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            authorization = self.headers.get("Authorization")
            observed.append(authorization)
            length = int(self.headers.get("content-length", "0"))
            request = json.loads(self.rfile.read(length))
            if authorization != "Bearer fresh-token":
                self.send_response(401)
                self.send_header("content-length", "0")
                self.end_headers()
                return
            body = json.dumps({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"resultType": "complete", "tools": []},
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ExternalMcpClient(
            "oauth-helper",
            ExternalMcpServerConfig(
                enabled=True,
                optional=False,
                transport="streamable_http",
                command=[],
                url=f"http://127.0.0.1:{server.server_port}/mcp",
                header_env={},
                credential_command=[sys.executable, str(provider)],
                credential_refresh_skew_seconds=30.0,
                timeout_seconds=5.0,
            ),
        )
        assert client.list_tools() == ()
        assert observed == ["Bearer stale-token", "Bearer fresh-token"]
        assert log_path.read_text().splitlines() == ["initial_or_expired", "unauthorized"]
        # Fresh credentials are cached; another request does not invoke the provider again.
        assert client.list_tools() == ()
        assert observed[-1] == "Bearer fresh-token"
        assert log_path.read_text().splitlines() == ["initial_or_expired", "unauthorized"]
        assert "fresh-token" not in repr(client.config)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_external_mcp_credential_provider_refreshes_before_expiry(tmp_path: Path) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    log_path = tmp_path / "credential.log"
    provider = tmp_path / "credential_provider.py"
    _write_credential_provider(provider, log_path, expiring=True)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", "0"))
            request = json.loads(self.rfile.read(length))
            body = json.dumps({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"resultType": "complete", "tools": []},
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ExternalMcpClient(
            "expiring-helper",
            ExternalMcpServerConfig(
                enabled=True,
                optional=False,
                transport="streamable_http",
                command=[],
                url=f"http://127.0.0.1:{server.server_port}/mcp",
                header_env={},
                credential_command=[sys.executable, str(provider)],
                credential_refresh_skew_seconds=30.0,
                timeout_seconds=5.0,
            ),
        )
        client.list_tools()
        client.list_tools()
        assert log_path.read_text().splitlines() == [
            "initial_or_expired",
            "initial_or_expired",
        ]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_external_mcp_credential_provider_conflicting_static_header_fails(
    tmp_path: Path, monkeypatch
) -> None:
    provider = tmp_path / "credential_provider.py"
    log_path = tmp_path / "credential.log"
    _write_credential_provider(provider, log_path)
    monkeypatch.setenv("SWAAG_STATIC_AUTH", "Bearer static")
    client = ExternalMcpClient(
        "conflict",
        ExternalMcpServerConfig(
            enabled=True,
            optional=False,
            transport="streamable_http",
            command=[],
            url="http://127.0.0.1:1/mcp",
            header_env={"Authorization": "SWAAG_STATIC_AUTH"},
            credential_command=[sys.executable, str(provider)],
            credential_refresh_skew_seconds=30.0,
            timeout_seconds=1.0,
        ),
    )
    with pytest.raises(ExternalMcpError, match="conflicts with configured header Authorization"):
        client.list_tools()


def test_oversized_external_mcp_result_uses_generic_projection_and_exact_history_recovery(
    make_config, tmp_path: Path
) -> None:
    from swaag.model import CompletionRequestPolicy
    from swaag.types import CompletionResult, ContractSpec

    marker = "EXTERNAL-MCP-REQUIRED-MARKER-91827"

    class ProjectionClient:
        is_deterministic_test_client = True

        def __init__(self):
            self.requests = []

        def tokenize(self, text: str) -> int:
            return len(text)

        def tokenize_selection(self, text: str) -> int:
            return len(text)

        def select_request_policy(self, *, contract: ContractSpec, **_kwargs):
            return CompletionRequestPolicy(
                "test", "server_schema", contract.mode, 30, 0.01
            )

        def resolve_contract(self, contract: ContractSpec, **kwargs):
            return contract, self.select_request_policy(contract=contract, **kwargs)

        def build_completion_request(
            self,
            prompt: str,
            *,
            max_tokens: int,
            contract: ContractSpec,
            temperature=None,
        ):
            return {
                "prompt": prompt,
                "n_predict": max_tokens,
                "contract": contract.name,
                "json_schema": contract.json_schema,
            }

        def send_completion(self, payload, **_kwargs):
            self.requests.append(payload)
            projection = marker if marker in payload["prompt"] else "irrelevant bulk"
            response = json.dumps({"projection": projection})
            return CompletionResult(
                text=response,
                raw_request=payload,
                raw_response={"content": response},
                prompt_tokens=None,
                completion_tokens=None,
                finish_reason="stop",
            )

    server = tmp_path / "large_mcp.py"
    server.write_text(
        "import json,sys\n"
        f"marker={marker!r}\n"
        "bulk='noise-'*600 + marker + '-tail'*600\n"
        "for line in sys.stdin:\n"
        "  if not line.strip(): continue\n"
        "  req=json.loads(line); rid=req['id']\n"
        "  if req['method']=='tools/list':\n"
        "    result={'resultType':'complete','tools':[{'name':'external_large','description':'Return large external evidence.','inputSchema':{'type':'object','properties':{},'additionalProperties':False}}]}\n"
        "  elif req['method']=='tools/call':\n"
        "    result={'resultType':'complete','content':[{'type':'text','text':bulk}],'structuredContent':{'evidence':bulk},'isError':False}\n"
        "  else: result={'resultType':'complete'}\n"
        "  print(json.dumps({'jsonrpc':'2.0','id':rid,'result':result}),flush=True)\n"
    )
    config = make_config(
        model__context_limit=2_000,
        context__max_compaction_rounds=4,
        context__safety_margin_tokens=32,
    )
    config.sessions.root = tmp_path / "sessions"
    config.external_tools.mcp_servers["large"] = ExternalMcpServerConfig(
        enabled=True,
        optional=False,
        transport="stdio",
        command=[sys.executable, str(server)],
        url="",
        header_env={},
        credential_command=[],
        credential_refresh_skew_seconds=30.0,
        timeout_seconds=5.0,
    )
    client = ProjectionClient()
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    spec = runtime.runtime_external_tools.specs()[0]

    result = runtime._execute_runtime_external_tool(state, spec=spec, arguments={})
    assert result is not None
    assert marker in result.display_text

    events_before_projection = runtime.history.read_history(state.session_id)
    source = next(
        event
        for event in reversed(events_before_projection)
        if event.event_type == "tool_result" and event.payload.get("tool_name") == "external_large"
    )
    exact_evidence = source.payload["output"]["structured_content"]["evidence"]
    assert marker in exact_evidence
    assert len(exact_evidence) > 5_000
    assert source.payload["executor"] == "external_runtime"

    tool_message = state.messages[-1]
    assert tool_message.metadata["source_event_sequence"] == source.sequence
    assert tool_message.metadata["source_event_hash"] == source.hash
    projection = runtime._create_tool_result_projection(
        state,
        original_request="Recover only the required external marker.",
        message=tool_message,
        target_tokens=128,
        original_tokens=len(tool_message.content),
        overflow_tokens=10_000,
    )
    assert projection == marker
    assert client.requests

    projected = next(
        event
        for event in reversed(runtime.history.read_history(state.session_id))
        if event.event_type == "tool_result_projected"
    )
    assert projected.payload["source_event_sequence"] == source.sequence
    assert projected.payload["source_event_hash"] == source.hash
    assert projected.payload["tool_name"] == "external_large"
    assert projected.payload["projected_tokens"] < projected.payload["original_tokens"]

    # Projection is disposable prompt material. The full MCP evidence is still
    # authoritative and boundedly re-readable from normal history.
    recovered = runtime.history.read_history_window(
        state.session_id, start_sequence=source.sequence, limit=1
    )[0]
    assert recovered.hash == source.hash
    assert recovered.payload["output"]["structured_content"]["evidence"] == exact_evidence


def test_external_mcp_http_error_preserves_exact_body_as_artifact(
    make_config, tmp_path: Path
) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from swaag.environment.artifacts import TextArtifactStore

    marker = "EXTERNAL-MCP-ERROR-MARKER-77219"
    body_text = "failure-prefix-" + ("x" * 3500) + marker + ("y" * 1200)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            _ = self.rfile.read(int(self.headers.get("content-length", "0")))
            body = body_text.encode()
            self.send_response(502)
            self.send_header("content-type", "text/plain; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = make_config()
        config.sessions.root = tmp_path / "sessions"
        config.external_tools.mcp_servers["failing"] = ExternalMcpServerConfig(
            enabled=True,
            optional=True,
            transport="streamable_http",
            command=[],
            url=f"http://127.0.0.1:{server.server_port}/mcp",
            header_env={},
            credential_command=[],
            credential_refresh_skew_seconds=30.0,
            timeout_seconds=5.0,
        )
        runtime = AgentRuntime(config, model_client=object())
        state = runtime.create_or_load_session()
        # Avoid construction-time discovery hitting the failing server: inject the
        # external tool spec after runtime construction and retain the same manager
        # client for execution.
        client = ExternalMcpClient("failing", config.external_tools.mcp_servers["failing"])
        spec = prepare_delegated_tool_spec(
            {
                "name": "failing_external",
                "description": "Return a failing external response.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "metadata": {
                    "external_execution_mode": "runtime",
                    "external_provider_id": "mcp:failing",
                    "mcp_server": "failing",
                    "mcp_input_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        )
        mcp_adapter = runtime.runtime_external_tools._adapters[0]
        mcp_adapter._clients["failing"] = client
        from swaag.external_mcp import ExternalMcpTool

        mcp_adapter._tools_by_name[spec.name] = ExternalMcpTool(
            server_name="failing", spec=spec
        )
        runtime.runtime_external_tools.refresh()

        result = runtime._execute_runtime_external_tool(state, spec=spec, arguments={})
        assert result is None
        error_event = next(
            event
            for event in reversed(runtime.history.read_history(state.session_id))
            if event.event_type == "tool_error"
        )
        evidence = error_event.payload["evidence"]
        assert evidence["http_status"] == 502
        assert evidence["response_body_chars"] == len(body_text)
        assert len(evidence["response_body_preview"]) == 1000
        assert evidence["response_body_finished"] is False
        assert marker not in evidence["response_body_preview"]
        assert "response_body" not in evidence

        artifact = TextArtifactStore(config.sessions.root, state.session_id)
        recovered = artifact.read(
            evidence["response_body_artifact_id"], start_offset=0, max_chars=len(body_text) + 1
        )
        assert recovered["finished"] is True
        assert recovered["sha256"] == evidence["response_body_sha256"]
        assert recovered["text"] == body_text
        assert marker in recovered["text"]

        tool_message = state.messages[-1]
        assert marker not in tool_message.content
        assert evidence["response_body_artifact_id"] in tool_message.content
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
