from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from swaag.config import ExternalMcpServerConfig, ExternalToolsConfig
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
    assert tools[0].metadata["external_executor"] == "mcp"
    assert tools[0].metadata["mcp_server"] == "dummy"
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
    specs = runtime.external_mcp.specs()
    assert [spec.name for spec in specs] == ["dummy_lookup"]
    assert "dummy_lookup" not in runtime.tools.system_tool_names()
    index = runtime.tools.capability_index(config, specs)
    assert any(name == "dummy_lookup" for name, _description, _guidance in index)
    staged = runtime.tools.staged_prompt_tuples(config, ["dummy_lookup"], specs)
    dummy = next(item for item in staged if item[0] == "dummy_lookup")
    assert dummy[2]["required"] == ["key"]
    assert "MCP" in dummy[3]

    result = runtime._execute_external_mcp_tool(
        state, spec=specs[0], arguments={"key": "beta"}
    )
    assert result is not None
    assert result.output["structured_content"] == {"value": "VALUE:beta"}
    tool_events = [
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "tool_result"
    ]
    assert tool_events[-1].payload["executor"] == "mcp"
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
