from __future__ import annotations

import asyncio
import base64
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from swaag.communication import CommunicationService
from swaag.config import load_config
from swaag.mcp import McpAdapter, McpOAuthResourceServer
from swaag.runtime import AgentRuntime


class _NoInferenceClient:
    is_deterministic_test_client = True
    mode = ""

    def __init__(self) -> None:
        self.accesses: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.accesses.append(name)
        raise AssertionError(f"MCP test attempted model-client access: {name}")


def _metadata(version: str = "2026-07-28") -> dict[str, Any]:
    return {
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": version,
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {
                "name": "swaag-test",
                "version": "1",
            },
        }
    }


def _headers(method: str, *, name: str | None = None) -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
        "mcp-protocol-version": "2026-07-28",
        "mcp-method": method,
    }
    if name is not None:
        headers["mcp-name"] = name
    return headers


def test_mcp_http_routes_by_headers_and_returns_modern_protocol_errors(
    make_config,
) -> None:
    adapter = McpAdapter(AgentRuntime(make_config(), model_client=object()))
    discover = {
        "jsonrpc": "2.0",
        "id": "discover",
        "method": "server/discover",
        "params": _metadata(),
    }

    accepted = adapter.handle_http(discover, _headers("server/discover"))
    mismatched = adapter.handle_http(
        discover,
        {
            **_headers("server/discover"),
            "mcp-method": "tools/list",
        },
    )
    unsupported_request = {
        **discover,
        "id": "unsupported",
        "params": _metadata("2099-01-01"),
    }
    unsupported = adapter.handle_http(
        unsupported_request,
        {
            **_headers("server/discover"),
            "mcp-protocol-version": "2099-01-01",
        },
    )
    unknown = adapter.handle_http(
        {
            **discover,
            "id": "unknown",
            "method": "example/unknown",
        },
        _headers("example/unknown"),
    )

    assert accepted.status == 200
    assert accepted.payload["result"]["supportedVersions"] == ["2026-07-28"]
    assert mismatched.status == 400
    assert mismatched.payload["error"]["code"] == -32020
    assert unsupported.status == 400
    assert unsupported.payload["error"] == {
        "code": -32022,
        "message": "Unsupported protocol version: '2099-01-01'",
        "data": {
            "supported": ["2026-07-28"],
            "requested": "2099-01-01",
        },
    }
    assert unknown.status == 404
    assert unknown.payload["error"]["code"] == -32601


def test_mcp_rejects_missing_version_as_invalid_metadata(make_config) -> None:
    adapter = McpAdapter(AgentRuntime(make_config(), model_client=object()))
    response = adapter.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": {
                "_meta": {
                    "io.modelcontextprotocol/clientCapabilities": {},
                }
            },
        }
    )

    assert response["error"]["code"] == -32602
    assert "protocolVersion" in response["error"]["message"]


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "http://localhost",
        "https://localhost:8443",
        "http://127.0.0.1:3000",
        "http://[::1]:3000",
        "http://127.42.3.9",
    ],
)
def test_mcp_http_preflight_allows_only_local_or_absent_origins(
    make_config, origin: str | None
) -> None:
    adapter = McpAdapter(AgentRuntime(make_config(), model_client=object()))
    headers = _headers("server/discover")
    if origin is not None:
        headers["origin"] = origin

    assert adapter.http_preflight(headers) is None


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "https://example.com",
        "http://localhost.example.com",
        "http://127.0.0.1.example.com",
        "http://localhost/path",
        "http://user@localhost",
        "http://localhost:invalid",
        " http://localhost",
    ],
)
def test_mcp_http_preflight_rejects_nonlocal_or_malformed_origins(
    make_config, origin: str
) -> None:
    adapter = McpAdapter(AgentRuntime(make_config(), model_client=object()))
    response = adapter.http_preflight(
        {**_headers("server/discover"), "origin": origin}
    )

    assert response is not None
    assert response.status == 403


def test_mcp_http_preflight_requires_both_response_media_types(make_config) -> None:
    adapter = McpAdapter(AgentRuntime(make_config(), model_client=object()))

    missing_sse = adapter.http_preflight(
        {**_headers("server/discover"), "accept": "application/json"}
    )
    disabled_json = adapter.http_preflight(
        {
            **_headers("server/discover"),
            "accept": "application/json;q=0, text/event-stream",
        }
    )
    wrong_content_type = adapter.http_preflight(
        {**_headers("server/discover"), "content-type": "text/plain"}
    )

    assert missing_sse is not None and missing_sse.status == 406
    assert disabled_json is not None and disabled_json.status == 406
    assert wrong_content_type is not None and wrong_content_type.status == 415


def test_mcp_http_decodes_names_and_validates_annotated_tool_parameters(
    make_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AgentRuntime(make_config(), model_client=object())
    adapter = McpAdapter(runtime)
    calculator = runtime.tools.get("calculator")
    annotated_schema = copy.deepcopy(calculator.input_schema)
    annotated_schema["properties"]["expression"]["x-mcp-header"] = "Expression"
    monkeypatch.setattr(calculator, "input_schema", annotated_schema)
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            **_metadata(),
            "name": "calculator",
            "arguments": {"expression": "6 * 7"},
        },
    }

    missing = adapter.handle_http(
        request,
        _headers("tools/call", name="calculator"),
    )
    accepted = adapter.handle_http(
        request,
        {
            **_headers("tools/call", name="calculator"),
            "mcp-param-expression": "6 * 7",
        },
    )
    encoded_name = "=?base64?" + base64.b64encode("calculator".encode()).decode() + "?="
    encoded = adapter.handle_http(
        request,
        {
            **_headers("tools/call", name=encoded_name),
            "mcp-param-expression": (
                "=?base64?" + base64.b64encode(b"6 * 7").decode() + "?="
            ),
        },
    )

    assert missing.status == 400
    assert missing.payload["error"]["code"] == -32020
    assert accepted.status == 200
    assert accepted.payload["result"]["structuredContent"]["result"] == 42
    assert encoded.status == 200


def test_mcp_header_schema_scan_supports_nested_primitives_and_rejects_ambiguity(
    make_config,
) -> None:
    adapter = McpAdapter(AgentRuntime(make_config(), model_client=object()))
    schema = {
        "type": "object",
        "properties": {
            "routing": {
                "type": "object",
                "properties": {
                    "tenant": {"type": "string", "x-mcp-header": "Tenant"},
                    "shard": {"type": "integer", "x-mcp-header": "Shard"},
                    "dry_run": {"type": "boolean", "x-mcp-header": "Dry-Run"},
                },
            }
        },
    }

    mirrored = adapter._mirrored_parameters(schema)

    assert [(item.header_name, item.path, item.value_type) for item in mirrored] == [
        ("Tenant", ("routing", "tenant"), "string"),
        ("Shard", ("routing", "shard"), "integer"),
        ("Dry-Run", ("routing", "dry_run"), "boolean"),
    ]
    with pytest.raises(ValueError, match="case-insensitively unique"):
        adapter._mirrored_parameters(
            {
                "type": "object",
                "properties": {
                    "first": {"type": "string", "x-mcp-header": "Route"},
                    "second": {"type": "string", "x-mcp-header": "route"},
                },
            }
        )
    with pytest.raises(ValueError, match="non-properties"):
        adapter._mirrored_parameters(
            {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"type": "string", "x-mcp-header": "Value"},
                    }
                },
            }
        )
    with pytest.raises(ValueError, match="object properties"):
        adapter._mirrored_parameters(
            {
                "type": "object",
                "properties": {
                    "not_an_object": {
                        "type": "string",
                        "properties": {
                            "nested": {
                                "type": "string",
                                "x-mcp-header": "Nested",
                            }
                        },
                    }
                },
            }
        )


def test_mcp_streamable_http_listener_is_gated_and_never_uses_inference(
    make_config, tmp_path: Path
) -> None:
    async def exercise() -> None:
        config = make_config()
        config.sessions.root = tmp_path / "sessions"
        config.mcp.enabled = True
        config.mcp.transport = "streamable_http"
        no_inference = _NoInferenceClient()
        runtime = AgentRuntime(config, model_client=no_inference)
        runtime.create_or_load_session()
        service = CommunicationService(runtime)
        server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
        port = int(server.sockets[0].getsockname()[1])

        async def http(
            method: str,
            *,
            payload: dict[str, Any] | None = None,
            extra_headers: dict[str, str] | None = None,
        ) -> tuple[str, dict[str, str], bytes]:
            body = b"" if payload is None else json.dumps(payload).encode()
            request_headers = {
                "Host": "localhost",
                **(extra_headers or {}),
            }
            if payload is not None:
                request_headers["Content-Length"] = str(len(body))
            raw = f"{method} /mcp HTTP/1.1\r\n".encode()
            raw += b"".join(
                f"{name}: {value}\r\n".encode()
                for name, value in request_headers.items()
            )
            raw += b"\r\n" + body
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(raw)
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            head, response_body = response.split(b"\r\n\r\n", 1)
            lines = head.decode().split("\r\n")
            response_headers = {
                name.casefold(): value.strip()
                for name, value in (line.split(":", 1) for line in lines[1:])
            }
            return lines[0], response_headers, response_body

        discover = {
            "jsonrpc": "2.0",
            "id": "discover-http",
            "method": "server/discover",
            "params": _metadata(),
        }
        status, response_headers, body = await http(
            "POST",
            payload=discover,
            extra_headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2026-07-28",
                "Mcp-Method": "server/discover",
                "Origin": "http://localhost",
            },
        )
        get_status, get_headers, _get_body = await http("GET")
        bad_get_status, _bad_get_headers, _bad_get_body = await http(
            "GET", extra_headers={"Origin": "https://example.com"}
        )
        forbidden_status, _forbidden_headers, forbidden_body = await http(
            "POST",
            payload=discover,
            extra_headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2026-07-28",
                "Mcp-Method": "server/discover",
                "Origin": "https://example.com",
            },
        )

        server.close()
        await server.wait_closed()
        service.workers.shutdown()

        assert status == "HTTP/1.1 200 OK"
        assert response_headers["content-type"] == "application/json"
        assert json.loads(body)["result"]["supportedVersions"] == ["2026-07-28"]
        assert get_status == "HTTP/1.1 405 Method Not Allowed"
        assert get_headers["allow"] == "POST"
        assert bad_get_status == "HTTP/1.1 403 Forbidden"
        assert forbidden_status == "HTTP/1.1 403 Forbidden"
        assert json.loads(forbidden_body)["error"]["code"] == -32600
        assert no_inference.accesses == []

    asyncio.run(exercise())


def test_mcp_subscription_acknowledges_changes_and_cancels_without_inference(
    make_config, tmp_path: Path
) -> None:
    async def exercise() -> None:
        config = make_config()
        config.sessions.root = tmp_path / "sessions"
        config.mcp.enabled = True
        config.mcp.transport = "streamable_http"
        no_inference = _NoInferenceClient()
        runtime = AgentRuntime(config, model_client=no_inference)
        runtime.create_or_load_session()
        service = CommunicationService(runtime)
        server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
        port = int(server.sockets[0].getsockname()[1])

        async def post(payload: dict[str, Any], headers: dict[str, str]):
            body = json.dumps(payload).encode()
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            request = b"POST /mcp HTTP/1.1\r\nHost: localhost\r\n"
            request += b"".join(
                f"{name}: {value}\r\n".encode()
                for name, value in {
                    **headers,
                    "Content-Length": str(len(body)),
                }.items()
            )
            writer.write(request + b"\r\n" + body)
            await writer.drain()
            return reader, writer

        subscription = {
            "jsonrpc": "2.0",
            "id": "listen-test",
            "method": "subscriptions/listen",
            "params": {
                **_metadata(),
                "notifications": {"toolsListChanged": True},
            },
        }
        reader, writer = await post(
            subscription,
            _headers("subscriptions/listen"),
        )
        response_head = await asyncio.wait_for(
            reader.readuntil(b"\r\n\r\n"), timeout=2
        )
        ack_frame = await asyncio.wait_for(reader.readuntil(b"\n\n"), timeout=2)
        ack = json.loads(ack_frame.removeprefix(b"data: ").strip())

        assert response_head.startswith(b"HTTP/1.1 200 OK")
        assert b"Content-Type: text/event-stream" in response_head
        assert ack == {
            "jsonrpc": "2.0",
            "method": "notifications/subscriptions/acknowledged",
            "params": {
                "notifications": {"toolsListChanged": True},
                "_meta": {
                    "io.modelcontextprotocol/subscriptionId": "listen-test"
                },
            },
        }

        config.tools.enabled = ["calculator"]
        changed_frame = await asyncio.wait_for(
            reader.readuntil(b"\n\n"), timeout=2
        )
        changed = json.loads(changed_frame.removeprefix(b"data: ").strip())
        assert changed == {
            "jsonrpc": "2.0",
            "method": "notifications/tools/list_changed",
            "params": {
                "_meta": {
                    "io.modelcontextprotocol/subscriptionId": "listen-test"
                }
            },
        }

        cancel = {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {**_metadata(), "requestId": "listen-test"},
        }
        cancel_reader, cancel_writer = await post(
            cancel,
            _headers("notifications/cancelled"),
        )
        cancel_response = await asyncio.wait_for(cancel_reader.read(), timeout=2)
        cancel_writer.close()
        await cancel_writer.wait_closed()
        assert cancel_response.startswith(b"HTTP/1.1 202 Accepted")
        assert await asyncio.wait_for(reader.read(), timeout=2) == b""

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()
        service.workers.shutdown()
        assert no_inference.accesses == []

    asyncio.run(exercise())


def test_mcp_subscription_rejects_unimplemented_filters_without_claiming_them(
    make_config,
) -> None:
    adapter = McpAdapter(AgentRuntime(make_config(), model_client=object()))
    request = {
        "jsonrpc": "2.0",
        "id": "listen-filter",
        "method": "subscriptions/listen",
        "params": {
            **_metadata(),
            "notifications": {
                "toolsListChanged": True,
                "promptsListChanged": True,
                "resourcesListChanged": True,
                "resourceSubscriptions": ["file:///example"],
            },
        },
    }

    prepared = adapter.prepare_http_subscription(
        request, _headers("subscriptions/listen")
    )

    assert prepared.honored_filter == {"toolsListChanged": True}
    adapter.finish_http_subscription(prepared.request_id)


@pytest.mark.parametrize("transport", ["stdio", "streamable_http", "both"])
def test_mcp_transport_configuration_accepts_explicit_bindings(
    tmp_path: Path, transport: str
) -> None:
    config = load_config(
        env={
            "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
            "SWAAG__MCP__TRANSPORT": transport,
        }
    )

    assert config.mcp.transport == transport


def test_mcp_multi_round_trip_handler_requires_explicit_registration(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=object())
    adapter = McpAdapter(runtime)
    request = {
        "jsonrpc": "2.0",
        "id": "retry",
        "method": "tools/call",
        "params": {
            **_metadata(),
            "name": "calculator",
            "arguments": {"expression": "6 * 7"},
            "requestState": "opaque",
            "inputResponses": {"x": {"action": "accept", "content": {"value": 2}}},
        },
    }
    response = adapter.handle_http(request, _headers("tools/call", name="calculator"))
    assert response.status == 200
    assert response.payload["error"]["code"] == -32602
    assert "does not support MCP multi-round-trip" in response.payload["error"]["message"]


def test_mcp_multi_round_trip_handler_retries_into_canonical_tool_execution(make_config) -> None:
    from swaag.mcp import McpInputRequired

    runtime = AgentRuntime(make_config(), model_client=object())
    adapter = McpAdapter(runtime)

    def handler(arguments, context):
        if not context.input_responses:
            return McpInputRequired(
                {
                    "factor": {
                        "method": "elicitation/create",
                        "params": {
                            "mode": "form",
                            "message": "Factor?",
                            "requestedSchema": {
                                "type": "object",
                                "properties": {"factor": {"type": "number"}},
                                "required": ["factor"],
                            },
                        },
                    }
                },
                "state-v1",
            )
        assert context.request_state == "state-v1"
        factor = context.input_responses["factor"]["content"]["factor"]
        return {"expression": f"({arguments['expression']}) * ({factor})"}

    adapter.register_multi_round_trip_handler("calculator", handler)
    first = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {**_metadata(), "name": "calculator", "arguments": {"expression": "6 * 7"}},
    }
    initial = adapter.handle_http(first, _headers("tools/call", name="calculator"))
    assert initial.status == 200
    assert initial.payload["result"]["resultType"] == "input_required"
    assert initial.payload["result"]["requestState"] == "state-v1"
    retry = copy.deepcopy(first)
    retry["id"] = 2
    retry["params"]["requestState"] = "state-v1"
    retry["params"]["inputResponses"] = {
        "factor": {"action": "accept", "content": {"factor": 3}}
    }
    completed = adapter.handle_http(retry, _headers("tools/call", name="calculator"))
    assert completed.status == 200
    assert completed.payload["result"]["resultType"] == "complete"
    assert completed.payload["result"]["structuredContent"]["result"] == 126


def test_mcp_multi_round_trip_rejects_wrapped_input_response(make_config) -> None:
    from swaag.mcp import McpInputRequired

    runtime = AgentRuntime(make_config(), model_client=object())
    adapter = McpAdapter(runtime, multi_round_trip_handlers={
        "calculator": lambda arguments, context: McpInputRequired({}, "state")
    })
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            **_metadata(),
            "name": "calculator",
            "arguments": {"expression": "1"},
            "inputResponses": {"factor": {"method": "elicitation/create", "result": {}}},
        },
    }
    response = adapter.handle_http(request, _headers("tools/call", name="calculator"))
    assert response.status == 200
    assert response.payload["error"]["code"] == -32602
    assert "bare MCP input response objects" in response.payload["error"]["message"]




def test_mcp_oauth_config_fails_closed_when_incomplete(monkeypatch) -> None:
    env = {
        "SWAAG__MCP__AUTHORIZATION__ENABLED": "true",
        "SWAAG__MCP__AUTHORIZATION__RESOURCE_URI": "https://mcp.example.test/mcp",
    }
    with pytest.raises(ValueError, match="authorization_servers"):
        load_config(env=env)

class _FakeIntrospectionResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> dict[str, Any]:
        return dict(self._payload)


def _enable_mcp_oauth(config) -> None:
    config.mcp.enabled = True
    config.mcp.transport = "streamable_http"
    auth = config.mcp.authorization
    auth.enabled = True
    auth.resource_uri = "https://mcp.example.test/mcp"
    auth.authorization_servers = ["https://auth.example.test"]
    auth.allowed_origins = ["https://client.example.test"]
    auth.introspection_url = "https://auth.example.test/introspect"
    auth.introspection_client_id = "swaag-resource"
    auth.introspection_client_secret = "secret"
    auth.required_scopes = ["mcp:tools"]
    auth.timeout_seconds = 1.0


def test_mcp_oauth_resource_server_challenges_and_validates(monkeypatch, make_config) -> None:
    config = make_config()
    _enable_mcp_oauth(config)
    auth = McpOAuthResourceServer(config.mcp.authorization)

    missing = auth.authorize({})
    assert missing is not None and missing.status == 401
    assert missing.headers is not None
    assert 'resource_metadata="https://mcp.example.test/.well-known/oauth-protected-resource/mcp"' in missing.headers["WWW-Authenticate"]

    payloads = iter([
        {"active": False},
        {"active": True, "aud": "https://other.example/mcp", "scope": "mcp:tools"},
        {"active": True, "aud": "https://mcp.example.test/mcp", "scope": "other"},
        {"active": True, "aud": ["https://mcp.example.test/mcp"], "scope": "mcp:tools extra"},
    ])
    calls: list[dict[str, Any]] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return _FakeIntrospectionResponse(next(payloads))

    monkeypatch.setattr("swaag.mcp.requests.post", fake_post)
    invalid = auth.authorize({"authorization": "Bearer bad"})
    assert invalid is not None and invalid.status == 401
    wrong_aud = auth.authorize({"authorization": "Bearer wrong-aud"})
    assert wrong_aud is not None and wrong_aud.status == 401
    insufficient = auth.authorize({"authorization": "Bearer narrow"})
    assert insufficient is not None and insufficient.status == 403
    assert insufficient.headers is not None and 'scope="mcp:tools"' in insufficient.headers["WWW-Authenticate"]
    assert auth.authorize({"authorization": "Bearer good"}) is None
    assert len(calls) == 4
    assert calls[-1]["url"] == "https://auth.example.test/introspect"
    assert calls[-1]["auth"] == ("swaag-resource", "secret")
    assert calls[-1]["data"] == {"token": "good", "token_type_hint": "access_token"}


def test_dedicated_mcp_http_listener_serves_metadata_and_requires_oauth(monkeypatch, make_config, tmp_path: Path) -> None:
    async def exercise() -> None:
        config = make_config()
        config.sessions.root = tmp_path / "sessions-oauth"
        _enable_mcp_oauth(config)
        no_inference = _NoInferenceClient()
        runtime = AgentRuntime(config, model_client=no_inference)
        runtime.create_or_load_session()
        service = CommunicationService(runtime)
        monkeypatch.setattr(
            "swaag.mcp.requests.post",
            lambda *args, **kwargs: _FakeIntrospectionResponse({
                "active": True,
                "aud": "https://mcp.example.test/mcp",
                "scope": "mcp:tools",
            }),
        )
        server = await asyncio.start_server(service.handle_mcp_http_client, "127.0.0.1", 0)
        port = int(server.sockets[0].getsockname()[1])

        async def request(method: str, path: str, *, payload=None, headers=None):
            body = b"" if payload is None else json.dumps(payload).encode()
            merged = {"Host": "localhost", **(headers or {})}
            if payload is not None:
                merged["Content-Length"] = str(len(body))
            raw = f"{method} {path} HTTP/1.1\r\n".encode()
            raw += b"".join(f"{k}: {v}\r\n".encode() for k, v in merged.items())
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(raw + b"\r\n" + body)
            await writer.drain()
            response = await reader.read()
            writer.close(); await writer.wait_closed()
            head, response_body = response.split(b"\r\n\r\n", 1)
            lines = head.decode().split("\r\n")
            response_headers = {k.casefold(): v.strip() for k, v in (line.split(":", 1) for line in lines[1:])}
            return lines[0], response_headers, response_body

        status, _headers_out, body = await request("GET", "/.well-known/oauth-protected-resource/mcp")
        assert status == "HTTP/1.1 200 OK"
        metadata = json.loads(body)
        assert metadata["resource"] == "https://mcp.example.test/mcp"
        assert metadata["authorization_servers"] == ["https://auth.example.test"]
        assert metadata["scopes_supported"] == ["mcp:tools"]

        discover = {"jsonrpc": "2.0", "id": "auth-discover", "method": "server/discover", "params": _metadata()}
        base_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2026-07-28",
            "Mcp-Method": "server/discover",
            "Origin": "https://client.example.test",
        }
        unauthorized_status, unauthorized_headers, _ = await request("POST", "/mcp", payload=discover, headers=base_headers)
        assert unauthorized_status == "HTTP/1.1 401 Unauthorized"
        assert unauthorized_headers["www-authenticate"].startswith("Bearer ")
        ok_status, _ok_headers, ok_body = await request(
            "POST", "/mcp", payload=discover, headers={**base_headers, "Authorization": "Bearer good"}
        )
        assert ok_status == "HTTP/1.1 200 OK"
        assert json.loads(ok_body)["result"]["supportedVersions"] == ["2026-07-28"]

        server.close(); await server.wait_closed(); service.workers.shutdown()
        assert no_inference.accesses == []

    asyncio.run(exercise())
