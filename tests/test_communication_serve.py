from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from swaag.cli import _build_parser
from swaag.communication import CommunicationService, require_loopback_bind_host
from swaag.delegated_tools import (
    DelegatedToolInputRequired,
    DelegatedToolResultInput,
    prepare_delegated_tool_spec,
)
from swaag.heartbeat import watchdog_interval_seconds
from swaag.protocol_adapters import AgUiProjectionAdapter
from swaag.runtime import AgentRuntime
from swaag.telemetry import OperationalTelemetry


def _ag_ui_input(*, run_id: str = "run-1", resume=None) -> dict:
    payload = {
        "threadId": "thread-1",
        "runId": run_id,
        "state": {},
        "messages": [
            {"id": "user-1", "role": "user", "content": "Complete the run."}
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    if resume is not None:
        payload["resume"] = resume
    return payload


def test_mcp_http_cli_requires_explicit_bind() -> None:
    parser = _build_parser()
    args = parser.parse_args(["mcp-http", "--host", "127.0.0.1", "--port", "9443"])
    assert args.command == "mcp-http"
    assert args.host == "127.0.0.1"
    assert args.port == 9443


def test_communication_serve_cli_accepts_config_defaults():
    parser = _build_parser()
    args = parser.parse_args(["communication", "serve"] )
    assert args.communication_command == "serve"
    assert args.host is None
    assert args.port is None


@pytest.mark.parametrize("host", ["127.0.0.1", "127.10.20.30", "::1", "localhost"])
def test_communication_bind_accepts_only_loopback(host: str) -> None:
    assert require_loopback_bind_host(host) == host


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10", "example.test", ""])
def test_communication_bind_rejects_unauthenticated_non_loopback(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        require_loopback_bind_host(host)


def test_dedicated_mcp_http_server_rejects_non_loopback(make_config) -> None:
    async def exercise() -> None:
        config = make_config()
        config.mcp.enabled = True
        config.mcp.transport = "streamable_http"
        runtime = AgentRuntime(config, model_client=object())
        service = CommunicationService(runtime)
        with pytest.raises(ValueError, match="loopback"):
            await service.serve_mcp_http("0.0.0.0", 9443)
        service.workers.shutdown()

    asyncio.run(exercise())


def test_watchdog_interval_uses_half_systemd_window(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "20000000")
    assert watchdog_interval_seconds() == 10.0


def test_watchdog_interval_has_safe_default(monkeypatch):
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)
    assert watchdog_interval_seconds(default_seconds=7.0) == 7.0


@pytest.mark.skipif(os.name != "posix", reason="system service signal contract is POSIX-only")
def test_communication_serve_exits_cleanly_on_sigterm(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    env = os.environ.copy()
    env.update(
        {
            "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
            "SWAAG__COMMUNICATION__ENABLED": "true",
            "SWAAG__COMMUNICATION__HOST": "127.0.0.1",
            "SWAAG__COMMUNICATION__PORT": str(port),
            "SWAAG__MODEL__BASE_URL": "http://127.0.0.1:1",
            "OTEL_SDK_DISABLED": "true",
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "swaag",
            "communication",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    f"communication service exited before listening: {stdout} {stderr}"
                )
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise AssertionError("communication service did not start")
                time.sleep(0.05)
        process.terminate()
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.returncode == 0, (stdout, stderr)
    assert stderr == ""


def test_communication_transport_exposes_task_api(make_config):
    async def exercise() -> None:
        service = CommunicationService(AgentRuntime(make_config(), model_client=object()))
        server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            (
                json.dumps(
                    {
                        "op": "task.create",
                        "params": {"objective": "durable communication task"},
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        created = json.loads((await reader.readline()).decode())
        writer.write((json.dumps({"op": "task.list", "params": {}}) + "\n").encode())
        await writer.drain()
        listed = json.loads((await reader.readline()).decode())
        worker_id = created["result"]["worker"]["worker_id"]

        async def request(payload):
            writer.write((json.dumps(payload) + "\n").encode())
            await writer.drain()
            return json.loads((await reader.readline()).decode())

        ag_ui = await request(
            {
                "op": "ag_ui.events",
                "params": {
                    "worker_id": worker_id,
                    "after_sequence": 0,
                    "limit": 1,
                },
            }
        )
        subscribed = await request(
            {
                "op": "ag_ui.subscribe",
                "params": {
                    "worker_id": worker_id,
                    "after_sequence": ag_ui["result"]["next_sequence"],
                    "timeout_seconds": 0.01,
                },
            }
        )
        a2a = await request(
            {"op": "a2a.get", "params": {"id": worker_id}}
        )
        a2a_subscription = await request(
            {
                "op": "a2a.subscribe",
                "params": {
                    "id": worker_id,
                    "after_sequence": ag_ui["result"]["next_sequence"],
                    "timeout_seconds": 0.01,
                },
            }
        )
        a2a_canceled = await request(
            {"op": "a2a.cancel", "params": {"id": worker_id}}
        )
        open_webui = await request(
            {"op": "open_webui.get", "params": {"worker_id": worker_id}}
        )
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()
        service.workers.shutdown()

        assert created["ok"] is True
        assert listed["result"]["workers"][0]["worker_id"] == worker_id
        assert ag_ui["result"]["events"][0]["type"] == "ACTIVITY_SNAPSHOT"
        assert ag_ui["result"]["next_sequence"] == 1
        assert subscribed["result"]["events"] == []
        assert subscribed["result"]["timed_out"] is True
        assert subscribed["result"]["terminal"] is False
        assert a2a["result"]["task"]["id"] == worker_id
        assert a2a_subscription["result"]["stream"][0]["task"]["id"] == worker_id
        assert a2a_subscription["result"]["timed_out"] is True
        assert a2a_canceled["result"]["task"]["status"]["state"] == "TASK_STATE_CANCELED"
        assert open_webui["result"]["metadata"]["worker_id"] == worker_id

    asyncio.run(exercise())


def test_communication_transport_serves_a2a_agent_card_and_jsonrpc(make_config):
    async def exercise() -> None:
        config = make_config()
        service = CommunicationService(AgentRuntime(config, model_client=object()))
        server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        created = service.task_api.execute(
            "create",
            {"objective": "inspect durable HTTP binding", "start": False},
        )
        worker_id = created["worker"]["worker_id"]

        async def http(raw: bytes) -> tuple[str, dict[str, str], bytes]:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(raw)
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            head, body = response.split(b"\r\n\r\n", 1)
            lines = head.decode().split("\r\n")
            headers = {
                name.casefold(): value.strip()
                for name, value in (line.split(":", 1) for line in lines[1:])
            }
            return lines[0], headers, body

        status, headers, body = await http(
            b"GET /.well-known/agent-card.json HTTP/1.1\r\nHost: localhost\r\n\r\n"
        )
        card = json.loads(body)
        assert status == "HTTP/1.1 200 OK"
        assert headers["cache-control"] == "public, max-age=300"
        assert card["supportedInterfaces"][0]["protocolVersion"] == "1.0"
        assert [
            item["protocolBinding"] for item in card["supportedInterfaces"]
        ] == ["JSONRPC", "HTTP+JSON"]
        assert card["capabilities"]["streaming"] is True
        cached_status, cached_headers, cached_body = await http(
            b"GET /.well-known/agent-card.json HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            + f"If-None-Match: {headers['etag']}\r\n\r\n".encode()
        )
        assert cached_status == "HTTP/1.1 304 Not Modified"
        assert cached_headers["etag"] == headers["etag"]
        assert cached_body == b""

        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "get-1",
                "method": "GetTask",
                "params": {"id": worker_id},
            }
        ).encode()
        status, _headers, body = await http(
            b"POST /a2a/v1 HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"A2A-Version: 1.0\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(request)}\r\n\r\n".encode()
            + request
        )
        response = json.loads(body)
        assert status == "HTTP/1.1 200 OK"
        assert response["id"] == "get-1"
        assert response["result"]["id"] == worker_id

        stream_request = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "stream-1",
                "method": "SubscribeToTask",
                "params": {"id": worker_id},
            }
        ).encode()
        stream_reader, stream_writer = await asyncio.open_connection(
            "127.0.0.1", port
        )
        stream_writer.write(
            b"POST /a2a/v1 HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"A2A-Version: 1.0\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(stream_request)}\r\n\r\n".encode()
            + stream_request
        )
        await stream_writer.drain()
        stream_head = await stream_reader.readuntil(b"\r\n\r\n")
        initial_event = await stream_reader.readuntil(b"\n\n")
        service.workers.cancel(worker_id, reason="finish transport stream test")
        remaining_events = await asyncio.wait_for(stream_reader.read(), timeout=2)
        stream_writer.close()
        await stream_writer.wait_closed()

        initial_payload = json.loads(initial_event.removeprefix(b"data: "))
        update_payloads = [
            json.loads(block.removeprefix(b"data: "))
            for block in remaining_events.split(b"\n\n")
            if block.startswith(b"data: ")
        ]
        assert b"Content-Type: text/event-stream" in stream_head
        assert initial_payload["result"]["task"]["id"] == worker_id
        update_states = [
            item["result"]["statusUpdate"]["status"]["state"]
            for item in update_payloads
        ]
        assert update_states, remaining_events
        assert "TASK_STATE_SUBMITTED" not in update_states
        assert (
            update_states[-1] == "TASK_STATE_CANCELED"
        )

        terminal_subscription = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "terminal-stream",
                "method": "SubscribeToTask",
                "params": {"id": worker_id},
            }
        ).encode()
        _terminal_status, _terminal_headers, terminal_body = await http(
            b"POST /a2a/v1 HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"A2A-Version: 1.0\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(terminal_subscription)}\r\n\r\n".encode()
            + terminal_subscription
        )
        assert json.loads(terminal_body)["error"]["code"] == -32004

        malformed_status, _malformed_headers, malformed_body = await http(
            b"POST /a2a/v1 HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"A2A-Version: 1.0\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 1\r\n\r\n{"
        )
        assert malformed_status == "HTTP/1.1 200 OK"
        assert json.loads(malformed_body)["error"]["code"] == -32700

        missing_version = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "bad-version",
                "method": "GetTask",
                "params": {"id": worker_id},
            }
        ).encode()
        _status, _headers, body = await http(
            b"POST /a2a/v1 HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(missing_version)}\r\n\r\n".encode()
            + missing_version
        )
        assert json.loads(body)["error"]["code"] == -32009

        server.close()
        await server.wait_closed()
        service.workers.shutdown()

    asyncio.run(exercise())


def test_a2a_bearer_boundary_advertises_https_and_fails_closed(make_config):
    async def exercise() -> None:
        config = make_config()
        config.a2a_authorization.enabled = True
        config.a2a_authorization.public_base_url = "https://agents.example.test/swaag"
        config.a2a_authorization.bearer_token = "test-secret-token"
        service = CommunicationService(AgentRuntime(config, model_client=object()))
        server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        created = service.task_api.execute(
            "create", {"objective": "inspect authenticated A2A binding", "start": False}
        )
        worker_id = created["worker"]["worker_id"]

        async def http(raw: bytes) -> tuple[str, dict[str, str], bytes]:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(raw)
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            head, body = response.split(b"\r\n\r\n", 1)
            lines = head.decode().split("\r\n")
            headers = {
                name.casefold(): value.strip()
                for name, value in (line.split(":", 1) for line in lines[1:])
            }
            return lines[0], headers, body

        status, _headers, body = await http(
            b"GET /.well-known/agent-card.json HTTP/1.1\r\nHost: localhost\r\n\r\n"
        )
        card = json.loads(body)
        assert status == "HTTP/1.1 200 OK"
        assert card["supportedInterfaces"][0]["url"] == (
            "https://agents.example.test/swaag/a2a/v1"
        )
        assert card["supportedInterfaces"][1]["url"] == (
            "https://agents.example.test/swaag/a2a/rest"
        )
        assert card["securityRequirements"] == [{"schemes": {"swaagBearer": {"list": []}}}]
        assert card["securitySchemes"]["swaagBearer"]["httpAuthSecurityScheme"]["scheme"] == "Bearer"

        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "auth-get",
                "method": "GetTask",
                "params": {"id": worker_id},
            }
        ).encode()
        base = (
            b"POST /a2a/v1 HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"A2A-Version: 1.0\r\n"
            b"Content-Type: application/json\r\n"
        )
        status, headers, body = await http(
            base + f"Content-Length: {len(request)}\r\n\r\n".encode() + request
        )
        assert status == "HTTP/1.1 401 Unauthorized"
        assert headers["www-authenticate"] == 'Bearer realm="swaag-a2a"'
        assert json.loads(body) == {"error": "unauthorized"}

        status, _headers, body = await http(
            base
            + b"Authorization: Bearer wrong-token\r\n"
            + f"Content-Length: {len(request)}\r\n\r\n".encode()
            + request
        )
        assert status == "HTTP/1.1 401 Unauthorized"

        status, _headers, body = await http(
            base
            + b"Authorization: Bearer test-secret-token\r\n"
            + f"Content-Length: {len(request)}\r\n\r\n".encode()
            + request
        )
        assert status == "HTTP/1.1 200 OK"
        assert json.loads(body)["result"]["id"] == worker_id

        server.close()
        await server.wait_closed()
        service.workers.shutdown()

    asyncio.run(exercise())


def test_a2a_authorization_requires_https_public_url_and_token(tmp_path):
    from swaag.config import load_config

    base_env = {
        "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
        "SWAAG__TOOLS__READ_ROOTS": f'["{tmp_path}"]',
        "SWAAG__TOOLS__STAGED_DISCOVERY": "false",
        "SWAAG__RUNTIME__COMPLETION_EVALUATION_ENABLED": "false",
        "SWAAG__MODEL__BASE_URL": "http://127.0.0.1:9999",
        "SWAAG__MODEL__CACHE_ENABLED": "false",
        "SWAAG__RETRIEVAL__BACKEND": "unavailable",
        "SWAAG__A2A__AUTHORIZATION__ENABLED": "true",
    }
    with pytest.raises(ValueError, match="absolute HTTPS"):
        load_config(
            env={
                **base_env,
                "SWAAG__A2A__AUTHORIZATION__PUBLIC_BASE_URL": "http://agents.example.test",
                "SWAAG__A2A__AUTHORIZATION__BEARER_TOKEN": "secret",
            }
        )
    with pytest.raises(ValueError, match="bearer_token"):
        load_config(
            env={
                **base_env,
                "SWAAG__A2A__AUTHORIZATION__PUBLIC_BASE_URL": "https://agents.example.test",
                "SWAAG__A2A__AUTHORIZATION__BEARER_TOKEN": "",
            }
        )
    for invalid_url in (
        "https://",
        "https://user:secret@agents.example.test",
        "https://agents.example.test?token=secret",
        "https://agents.example.test#internal",
    ):
        with pytest.raises(ValueError, match="absolute HTTPS"):
            load_config(
                env={
                    **base_env,
                    "SWAAG__A2A__AUTHORIZATION__PUBLIC_BASE_URL": invalid_url,
                    "SWAAG__A2A__AUTHORIZATION__BEARER_TOKEN": "secret",
                }
            )


def test_a2a_http_creates_unary_and_streaming_tasks_without_client_ids(
    make_config, monkeypatch
):
    async def exercise() -> None:
        service = CommunicationService(
            AgentRuntime(make_config(), model_client=object())
        )

        def queue_without_executor(worker_id: str):
            return service.workers.store.transition(
                worker_id,
                "queued",
                expected={"created"},
                event_type="worker_queued",
            )

        monkeypatch.setattr(service.workers, "start", queue_without_executor)
        server = await asyncio.start_server(
            service.handle_client, "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]

        def request(method: str, request_id: str) -> bytes:
            body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": {
                        "message": {
                            "messageId": request_id + "-message",
                            "role": "ROLE_USER",
                            "parts": [{"text": "Create a server-owned task."}],
                        },
                        "configuration": {"returnImmediately": True},
                    },
                }
            ).encode()
            return (
                b"POST /a2a/v1 HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"A2A-Version: 1.0\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )

        try:
            unary_reader, unary_writer = await asyncio.open_connection(
                "127.0.0.1", port
            )
            unary_writer.write(request("SendMessage", "unary-new"))
            await unary_writer.drain()
            unary_response = await asyncio.wait_for(unary_reader.read(), timeout=2)
            unary_writer.close()
            await unary_writer.wait_closed()
            unary_payload = json.loads(unary_response.split(b"\r\n\r\n", 1)[1])
            unary_task = unary_payload["result"]["task"]
            assert unary_task["id"]
            assert unary_task["contextId"]
            assert unary_task["status"]["state"] == "TASK_STATE_SUBMITTED"
            service.workers.cancel(unary_task["id"], reason="finish unary task")

            stream_reader, stream_writer = await asyncio.open_connection(
                "127.0.0.1", port
            )
            stream_writer.write(request("SendStreamingMessage", "stream-new"))
            await stream_writer.drain()
            stream_head = await stream_reader.readuntil(b"\r\n\r\n")
            initial_event = await stream_reader.readuntil(b"\n\n")
            initial_payload = json.loads(initial_event.removeprefix(b"data: "))
            stream_task = initial_payload["result"]["task"]
            service.workers.cancel(stream_task["id"], reason="finish streamed task")
            remaining = await asyncio.wait_for(stream_reader.read(), timeout=2)
            stream_writer.close()
            await stream_writer.wait_closed()

            updates = [
                json.loads(block.removeprefix(b"data: "))
                for block in remaining.split(b"\n\n")
                if block.startswith(b"data: ")
            ]
            assert b"Content-Type: text/event-stream" in stream_head
            assert stream_task["id"]
            assert stream_task["contextId"]
            assert stream_task["status"]["state"] == "TASK_STATE_SUBMITTED"
            assert updates[-1]["result"]["statusUpdate"]["status"]["state"] == (
                "TASK_STATE_CANCELED"
            )
        finally:
            server.close()
            await server.wait_closed()
            service.workers.shutdown()

    asyncio.run(exercise())


def test_a2a_http_json_binding_uses_rest_shapes_queries_and_errors(
    make_config, monkeypatch
):
    async def exercise() -> None:
        service = CommunicationService(
            AgentRuntime(make_config(), model_client=object())
        )

        def queue_without_executor(worker_id: str):
            return service.workers.store.transition(
                worker_id,
                "queued",
                expected={"created"},
                event_type="worker_queued",
            )

        monkeypatch.setattr(service.workers, "start", queue_without_executor)
        server = await asyncio.start_server(
            service.handle_client, "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]

        async def http(
            method: str,
            target: str,
            body: dict | None = None,
            *,
            version: str = "1.0",
            content_type: str = "application/a2a+json",
        ) -> tuple[str, dict[str, str], bytes]:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            encoded = b"" if body is None else json.dumps(body).encode()
            headers = (
                f"{method} {target} HTTP/1.1\r\n"
                "Host: localhost\r\n"
                f"A2A-Version: {version}\r\n"
                "Accept: application/a2a+json\r\n"
            ).encode()
            if body is not None:
                headers += (
                    f"Content-Type: {content_type}\r\n".encode()
                    + f"Content-Length: {len(encoded)}\r\n".encode()
                )
            writer.write(headers + b"\r\n" + encoded)
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), timeout=2)
            writer.close()
            await writer.wait_closed()
            head, response_body = response.split(b"\r\n\r\n", 1)
            lines = head.decode().split("\r\n")
            response_headers = {
                name.casefold(): value.strip()
                for name, value in (line.split(":", 1) for line in lines[1:])
            }
            return lines[0], response_headers, response_body

        message = {
            "message": {
                "messageId": "rest-unary-message",
                "role": "ROLE_USER",
                "parts": [{"text": "Exercise HTTP plus JSON."}],
            },
            "configuration": {"returnImmediately": True},
        }
        worker_id: str | None = None
        try:
            status, headers, body = await http(
                "POST", "/a2a/rest/message:send", message
            )
            created = json.loads(body)["task"]
            worker_id = created["id"]
            assert status == "HTTP/1.1 200 OK"
            assert headers["content-type"] == "application/a2a+json"
            assert created["status"]["state"] == "TASK_STATE_SUBMITTED"

            status, headers, body = await http(
                "GET",
                "/a2a/rest/tasks?status=TASK_STATE_SUBMITTED&"
                "pageSize=10&historyLength=0&includeArtifacts=true",
            )
            listed = json.loads(body)
            assert status == "HTTP/1.1 200 OK"
            assert headers["content-type"] == "application/a2a+json"
            assert listed["tasks"][0]["id"] == worker_id
            assert listed["totalSize"] == 1

            status, _headers, body = await http(
                "GET", f"/a2a/rest/tasks/{worker_id}?historyLength=0"
            )
            assert status == "HTTP/1.1 200 OK"
            assert json.loads(body)["id"] == worker_id

            async def subscribe(method: str):
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", port
                )
                writer.write(
                    (
                        f"{method} /a2a/rest/tasks/{worker_id}:subscribe "
                        "HTTP/1.1\r\n"
                        "Host: localhost\r\n"
                        "A2A-Version: 1.0\r\n"
                        "Accept: text/event-stream\r\n\r\n"
                    ).encode()
                )
                await writer.drain()
                head = await reader.readuntil(b"\r\n\r\n")
                initial = json.loads(
                    (await reader.readuntil(b"\n\n")).removeprefix(b"data: ")
                )
                return reader, writer, head, initial

            get_subscriber = await subscribe("GET")
            post_subscriber = await subscribe("POST")
            for _reader, _writer, head, initial in (
                get_subscriber,
                post_subscriber,
            ):
                assert b"Content-Type: text/event-stream" in head
                assert initial["task"]["id"] == worker_id
                assert "jsonrpc" not in initial

            status, _headers, body = await http(
                "POST", f"/a2a/rest/tasks/{worker_id}:cancel"
            )
            assert status == "HTTP/1.1 200 OK"
            assert json.loads(body)["status"]["state"] == "TASK_STATE_CANCELED"
            for reader, writer, _head, _initial in (
                get_subscriber,
                post_subscriber,
            ):
                terminal = await asyncio.wait_for(reader.read(), timeout=2)
                writer.close()
                await writer.wait_closed()
                assert b"TASK_STATE_CANCELED" in terminal

            stream_reader, stream_writer = await asyncio.open_connection(
                "127.0.0.1", port
            )
            stream_body = json.dumps(
                {
                    **message,
                    "message": {
                        **message["message"],
                        "messageId": "rest-stream-message",
                    },
                }
            ).encode()
            stream_writer.write(
                b"POST /a2a/rest/message:stream HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"A2A-Version: 1.0\r\n"
                b"Content-Type: application/a2a+json\r\n"
                b"Accept: text/event-stream\r\n"
                + f"Content-Length: {len(stream_body)}\r\n\r\n".encode()
                + stream_body
            )
            await stream_writer.drain()
            stream_head = await stream_reader.readuntil(b"\r\n\r\n")
            initial = json.loads(
                (await stream_reader.readuntil(b"\n\n")).removeprefix(b"data: ")
            )
            streamed_id = initial["task"]["id"]
            cancel_status, cancel_headers, cancel_body = await http(
                "POST", f"/a2a/rest/tasks/{streamed_id}:cancel"
            )
            remaining = await asyncio.wait_for(stream_reader.read(), timeout=2)
            stream_writer.close()
            await stream_writer.wait_closed()
            updates = [
                json.loads(block.removeprefix(b"data: "))
                for block in remaining.split(b"\n\n")
                if block.startswith(b"data: ")
            ]
            assert b"Content-Type: text/event-stream" in stream_head
            assert "jsonrpc" not in initial
            assert cancel_status == "HTTP/1.1 200 OK"
            assert cancel_headers["content-type"] == "application/a2a+json"
            assert json.loads(cancel_body)["status"]["state"] == (
                "TASK_STATE_CANCELED"
            )
            assert updates[-1]["statusUpdate"]["status"]["state"] == (
                "TASK_STATE_CANCELED"
            )

            status, headers, body = await http(
                "GET", "/a2a/rest/tasks/missing-task"
            )
            missing = json.loads(body)["error"]
            assert status == "HTTP/1.1 404 Not Found"
            assert headers["content-type"] == "application/a2a+json"
            assert missing["status"] == "NOT_FOUND"
            assert missing["details"][0]["reason"] == "TASK_NOT_FOUND"

            status, _headers, body = await http(
                "GET", "/a2a/rest/tasks?pageSize=one"
            )
            invalid = json.loads(body)["error"]
            assert status == "HTTP/1.1 400 Bad Request"
            assert invalid["status"] == "INVALID_ARGUMENT"
            assert invalid["details"][0]["@type"].endswith("BadRequest")

            status, _headers, body = await http(
                "POST",
                "/a2a/rest/message:send",
                message,
                content_type="text/plain",
            )
            unsupported = json.loads(body)["error"]
            assert status == "HTTP/1.1 400 Bad Request"
            assert unsupported["details"][0]["reason"] == (
                "CONTENT_TYPE_NOT_SUPPORTED"
            )
        finally:
            if (
                worker_id is not None
                and service.workers.store.get(worker_id).status == "queued"
            ):
                service.workers.cancel(worker_id, reason="finish REST test")
            server.close()
            await server.wait_closed()
            service.workers.shutdown()

    asyncio.run(exercise())


def test_serve_tcp_agent_card_advertises_the_effective_bound_endpoint(make_config):
    async def exercise() -> None:
        config = make_config()
        service = CommunicationService(AgentRuntime(config, model_client=object()))
        serve_task = asyncio.create_task(service.serve_tcp("127.0.0.1", 0))
        for _ in range(100):
            if service._advertised_port != config.communication.port:
                break
            await asyncio.sleep(0.01)
        else:
            serve_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await serve_task
            pytest.fail("communication listener did not bind")

        reader, writer = await asyncio.open_connection(
            "127.0.0.1", service._advertised_port
        )
        writer.write(
            b"GET /.well-known/agent-card.json HTTP/1.1\r\n"
            b"Host: localhost\r\n\r\n"
        )
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        card = json.loads(response.split(b"\r\n\r\n", 1)[1])

        serve_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await serve_task

        assert card["supportedInterfaces"][0]["url"] == (
            f"http://127.0.0.1:{service._advertised_port}/a2a/v1"
        )

    asyncio.run(exercise())


def test_http_adapter_extracts_w3c_trace_context(make_config) -> None:
    async def exercise() -> None:
        exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
        meter_provider = MeterProvider()
        telemetry = OperationalTelemetry(
            tracer=tracer_provider.get_tracer("swaag-http-test"),
            meter=meter_provider.get_meter("swaag-http-test"),
        )
        service = CommunicationService(
            AgentRuntime(
                make_config(),
                model_client=object(),
                telemetry=telemetry,
            )
        )
        server = await asyncio.start_server(
            service.handle_client, "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"GET /.well-known/agent-card.json HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"TraceParent: 00-1234567890abcdef1234567890abcdef-1234567890abcdef-01\r\n"
            b"\r\n"
        )
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()
        service.workers.shutdown()
        tracer_provider.force_flush()

        assert response.startswith(b"HTTP/1.1 200 OK")
        span = exporter.get_finished_spans()[0]
        assert span.name == "GET /.well-known/agent-card.json"
        assert span.kind.name == "SERVER"
        assert f"{span.context.trace_id:032x}" == (
            "1234567890abcdef1234567890abcdef"
        )
        assert span.parent is not None
        assert f"{span.parent.span_id:016x}" == "1234567890abcdef"
        assert span.attributes["http.response.status_code"] == 200

        meter_provider.shutdown()
        tracer_provider.shutdown()

    asyncio.run(exercise())


def test_communication_transport_serves_durable_ag_ui_sse(
    make_config, monkeypatch
):
    async def exercise() -> None:
        config = make_config()
        service = CommunicationService(AgentRuntime(config, model_client=object()))

        def complete_without_model(worker_id: str):
            service.workers.store.transition(
                worker_id,
                "queued",
                expected={"created"},
                event_type="worker_queued",
            )
            service.workers.store.transition(
                worker_id,
                "working",
                expected={"queued"},
                event_type="worker_started",
            )
            return service.workers.store.transition(
                worker_id,
                "completed",
                expected={"working"},
                result="exact AG-UI result",
                event_type="worker_completed",
            )

        monkeypatch.setattr(service.workers, "start", complete_without_model)
        server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        async def run_request(payload: dict) -> tuple[dict[str, str], list[dict]]:
            body = json.dumps(payload).encode()
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"POST /ag-ui HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Accept: text/event-stream\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            head, stream = response.split(b"\r\n\r\n", 1)
            lines = head.decode().split("\r\n")
            headers = {
                name.casefold(): value.strip()
                for name, value in (line.split(":", 1) for line in lines[1:])
            }
            events = [
                json.loads(block.removeprefix(b"data: "))
                for block in stream.split(b"\n\n")
                if block.startswith(b"data: ")
            ]
            return headers, events

        request = _ag_ui_input()
        request["messages"][0]["content"] = [
            {"type": "text", "text": "Complete the run."},
            {
                "type": "document",
                "source": {
                    "type": "data",
                    "value": "ZXhhY3QgYnl0ZXM=",
                    "mimeType": "application/pdf",
                },
                "metadata": {"filename": "facts.pdf"},
            },
        ]
        request["state"] = {
            "selectedRecord": {"id": "record-7", "revision": 3},
            "filters": ["active", "reviewed"],
        }
        headers, events = await run_request(request)
        duplicate_request = json.loads(json.dumps(request))
        duplicate_request["state"] = {"mustNotReplaceBoundState": True}
        _duplicate_headers, duplicate_events = await run_request(duplicate_request)
        workers = service.workers.list()
        attachments = service.workers.attachments(workers[0].worker_id)
        bounds = service.store.protocol_message_bounds("ag_ui", "run-1")
        snapshot = service.store.protocol_state_for_message("ag_ui", "run-1")
        server.close()
        await server.wait_closed()
        service.workers.shutdown()

        assert headers["content-type"] == "text/event-stream"
        assert headers["cache-control"] == "no-store"
        assert headers["x-accel-buffering"] == "no"
        assert [event["type"] for event in events].count("RUN_STARTED") == 1
        assert events[0]["threadId"] == "thread-1"
        assert events[0]["runId"] == "run-1"
        assert events[0]["metadata"]["swaagDuplicateRun"] is False
        assert events[1]["type"] == "STATE_SNAPSHOT"
        assert events[1]["snapshot"] == request["state"]
        assert events[1]["metadata"]["swaagStateRevision"] == 1
        assert events[-1]["type"] == "RUN_FINISHED"
        assert events[-1]["threadId"] == "thread-1"
        assert events[-1]["runId"] == "run-1"
        assert events[-1]["result"] == "exact AG-UI result"
        assert duplicate_events[0]["metadata"]["swaagDuplicateRun"] is True
        assert duplicate_events[1]["snapshot"] == request["state"]
        assert duplicate_events[-1]["result"] == "exact AG-UI result"
        assert len(workers) == 1
        assert attachments[0].original_name == "facts.pdf"
        assert attachments[0].source == "ag_ui"
        assert attachments[0].size_bytes == len(b"exact bytes")
        assert snapshot is not None
        assert snapshot.state == request["state"]
        assert snapshot.client_supplied is True
        assert '"selectedRecord":{"id":"record-7","revision":3}' in workers[
            0
        ].objective
        assert bounds is not None and bounds[3] is not None

    asyncio.run(exercise())


def test_communication_transport_serves_dynamic_ag_ui_capabilities(make_config):
    async def exercise() -> None:
        config = make_config()
        config.tools.enabled = ["calculator", "shared_state"]
        config.runtime.max_total_actions = 7
        service = CommunicationService(AgentRuntime(config, model_client=object()))
        server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        async def request(method: str) -> tuple[str, dict[str, str], dict]:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                (
                    f"{method} /ag-ui/capabilities HTTP/1.1\r\n"
                    "Host: localhost\r\n"
                    "Content-Length: 0\r\n\r\n"
                ).encode()
            )
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            head, body = response.split(b"\r\n\r\n", 1)
            lines = head.decode().split("\r\n")
            headers = {
                name.casefold(): value.strip()
                for name, value in (line.split(":", 1) for line in lines[1:])
            }
            return lines[0], headers, json.loads(body)

        get_status, get_headers, capabilities = await request("GET")
        post_status, post_headers, post_error = await request("POST")
        server.close()
        await server.wait_closed()
        service.workers.shutdown()

        assert get_status == "HTTP/1.1 200 OK"
        assert get_headers["content-type"] == "application/json"
        assert get_headers["cache-control"] == "no-store"
        assert capabilities["transport"] == {
            "streaming": True,
            "websocket": False,
            "httpBinary": False,
            "pushNotifications": False,
            "resumable": False,
        }
        assert capabilities["tools"]["supported"] is True
        assert capabilities["tools"]["clientProvided"] is True
        assert [item["name"] for item in capabilities["tools"]["items"]] == [
            "calculator"
        ]
        assert capabilities["tools"]["items"][0]["parameters"]["type"] == "object"
        assert capabilities["state"] == {
            "snapshots": True,
            "deltas": True,
            "memory": False,
            "persistentState": True,
        }
        assert capabilities["execution"]["maxIterations"] == 7
        assert capabilities["humanInTheLoop"]["interrupts"] is True
        assert get_headers["content-length"] == str(
            len(json.dumps(capabilities, sort_keys=True).encode())
        )
        assert post_status == "HTTP/1.1 405 Method Not Allowed"
        assert post_headers["allow"] == "GET"
        assert "GET only" in post_error["error"]

    asyncio.run(exercise())


def test_ag_ui_client_tool_round_trip_is_durable_without_model_access(
    make_config, monkeypatch
) -> None:
    service = CommunicationService(AgentRuntime(make_config(), model_client=object()))
    client_tool = {
        "name": "select_record",
        "description": "Select one record in the connected client.",
        "parameters": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
            "additionalProperties": False,
        },
        "metadata": {"owner": "client"},
    }

    def request_without_model(worker_id: str):
        service.workers.store.transition(
            worker_id,
            "queued",
            expected={"created"},
            event_type="worker_queued",
        )
        working = service.workers.store.transition(
            worker_id,
            "working",
            expected={"queued"},
            increment_run_count=True,
            event_type="worker_started",
        )
        catalog = service.runtime.delegated_tools.latest_catalog(
            working.session_id
        )
        assert catalog is not None
        state = service.runtime.history.rebuild_from_history(
            working.session_id, write_projections=False
        )
        with pytest.raises(DelegatedToolInputRequired) as wait:
            service.runtime._request_delegated_tool(
                state,
                catalog=catalog,
                spec=catalog.tools[0],
                arguments={"record_id": "record-7"},
            )
        call = wait.value.call
        service.workers._sync_history_events(working)
        return service.workers.store.transition(
            worker_id,
            "input_required",
            expected={"working"},
            event_type="worker_delegated_tool_input_required",
            event_payload={
                "call_id": call.call_id,
                "tool_name": call.tool_name,
                "arguments": call.arguments,
            },
        )

    observed = {}

    def complete_without_model(worker_id: str, message: str, *, source: str, **_):
        observed["message"] = message
        observed["source"] = source
        current = service.workers.store.get(worker_id)
        history = service.runtime.history.read_history(current.session_id)
        observed["tool_result"] = next(
            event
            for event in history
            if event.event_type == "tool_result"
            and event.payload.get("delegated") is True
        )
        service.workers.store.transition(
            worker_id,
            "queued",
            expected={"input_required"},
            event_type="worker_resumed",
        )
        service.workers.store.transition(
            worker_id,
            "working",
            expected={"queued"},
            increment_run_count=True,
            event_type="worker_started",
        )
        return service.workers.store.transition(
            worker_id,
            "completed",
            expected={"working"},
            result="completed after client execution",
            event_type="worker_completed",
        )

    monkeypatch.setattr(service.workers, "start", request_without_model)
    first_payload = _ag_ui_input()
    first_payload["tools"] = [client_tool]
    first_run = AgUiProjectionAdapter().user_run(first_payload)
    first_record, first_start, _, _, _ = service._ag_ui_begin(first_run)
    first_events = AgUiProjectionAdapter().events(
        first_record,
        service.workers.events(
            first_record.worker_id, after_sequence=first_start
        ),
        thread_id=first_run.thread_id,
        run_id=first_run.run_id,
    )
    pending = service.runtime.delegated_tools.pending_call(
        first_record.session_id
    )
    assert pending is not None

    monkeypatch.setattr(service.workers, "message", complete_without_model)
    second_payload = _ag_ui_input(run_id="run-2")
    second_payload["tools"] = [client_tool]
    second_payload["messages"].append(
        {
            "id": "client-tool-result-1",
            "role": "tool",
            "toolCallId": pending.call_id,
            "content": '{"selected":"record-7"}',
            "metadata": {"durationMs": 5},
        }
    )
    second_record, second_start, _, duplicate, _ = service._ag_ui_begin(
        AgUiProjectionAdapter().user_run(second_payload)
    )
    second_events = service.workers.events(
        second_record.worker_id, after_sequence=second_start
    )
    resolved = service.runtime.delegated_tools.call(pending.call_id)
    service.workers.shutdown()

    event_types = [event["type"] for event in first_events]
    assert event_types.count("TOOL_CALL_START") == 1
    assert event_types.count("TOOL_CALL_ARGS") == 1
    assert event_types.count("TOOL_CALL_END") == 1
    assert event_types[-1] == "RUN_FINISHED"
    assert first_events[-1]["outcome"] == {"type": "success"}
    assert first_events[-1]["metadata"]["swaagToolCallId"] == pending.call_id
    assert second_record.status == "completed"
    assert duplicate is False
    assert observed["source"] == "ag_ui:run-2"
    assert pending.call_id in observed["message"]
    assert observed["tool_result"].payload["output"]["content"] == (
        '{"selected":"record-7"}'
    )
    assert resolved is not None and resolved.status == "resolved"
    assert resolved.result_message_id == "client-tool-result-1"
    assert resolved.history_event_hash == observed["tool_result"].hash
    assert not any(
        event.event_type == "worker_history_event"
        and event.payload.get("history_event_type") == "tool_result"
        for event in second_events
    )


def test_ag_ui_rejects_client_tool_name_collisions(make_config) -> None:
    service = CommunicationService(AgentRuntime(make_config(), model_client=object()))
    payload = _ag_ui_input()
    payload["tools"] = [
        {
            "name": "calculator",
            "description": "Shadow the server calculator.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        }
    ]

    with pytest.raises(ValueError, match="collide"):
        service._ag_ui_begin(AgUiProjectionAdapter().user_run(payload))
    service.workers.shutdown()


def test_ag_ui_accepts_only_exact_known_historical_tool_messages(make_config) -> None:
    service = CommunicationService(AgentRuntime(make_config(), model_client=object()))
    worker = service.workers.create("Continue a protocol conversation.")
    service.store.set_protocol_worker("ag_ui", "thread-1", worker.worker_id)
    tool = {
        "name": "select_record",
        "description": "Select one record in the connected client.",
        "parameters": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
            "additionalProperties": False,
        },
    }
    catalog = service.runtime.delegated_tools.bind_catalog(
        worker.session_id,
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="original-tool-run",
        tools=[prepare_delegated_tool_spec(tool)],
    )
    call = service.runtime.delegated_tools.request_call(
        worker.session_id,
        catalog_revision=catalog.revision,
        tool_name="select_record",
        arguments={"record_id": "record-7"},
    )
    result = DelegatedToolResultInput(
        message_id="client-tool-result-1",
        call_id=call.call_id,
        content='{"selected":"record-7"}',
        error=None,
        metadata={"durationMs": 5},
    )
    service.runtime.accept_delegated_tool_result(
        worker.session_id,
        call.call_id,
        source="ag_ui",
        external_request_id="original-result-run",
        result=result,
    )
    payload = _ag_ui_input(run_id="history-run")
    payload["tools"] = [tool]
    payload["messages"].append(
        {
            "id": result.message_id,
            "role": "tool",
            "toolCallId": call.call_id,
            "content": result.content,
            "metadata": result.metadata,
        }
    )

    record, _start, _end, duplicate, _state = service._ag_ui_begin(
        AgUiProjectionAdapter().user_run(payload)
    )
    assert record.worker_id == worker.worker_id
    assert duplicate is False

    changed = _ag_ui_input(run_id="changed-history-run")
    changed["tools"] = [tool]
    changed["messages"].append(
        {
            "id": result.message_id,
            "role": "tool",
            "toolCallId": call.call_id,
            "content": "different",
            "metadata": result.metadata,
        }
    )
    with pytest.raises(ValueError, match="differs from durable exact result"):
        service._ag_ui_begin(AgUiProjectionAdapter().user_run(changed))
    assert (
        service.store.protocol_state_for_message("ag_ui", "changed-history-run")
        is None
    )
    service.workers.shutdown()


def test_ag_ui_rejects_a_terminal_tool_result_from_another_run_before_state_bind(
    make_config,
) -> None:
    service = CommunicationService(AgentRuntime(make_config(), model_client=object()))
    worker = service.workers.create("Continue after connected-client execution.")
    service.store.set_protocol_worker("ag_ui", "thread-1", worker.worker_id)
    service.workers.store.transition(
        worker.worker_id,
        "queued",
        expected={"created"},
        event_type="worker_queued",
    )
    working = service.workers.store.transition(
        worker.worker_id,
        "working",
        expected={"queued"},
        event_type="worker_started",
    )
    tool = prepare_delegated_tool_spec(
        {
            "name": "select_record",
            "description": "Select one record in the connected client.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
                "additionalProperties": False,
            },
        }
    )
    catalog = service.runtime.delegated_tools.bind_catalog(
        worker.session_id,
        source="ag_ui",
        external_context_id="thread-1",
        external_request_id="catalog-run",
        tools=[tool],
    )
    call = service.runtime.delegated_tools.request_call(
        worker.session_id,
        catalog_revision=catalog.revision,
        tool_name=tool.name,
        arguments={"record_id": "record-7"},
    )
    result = DelegatedToolResultInput(
        message_id="client-tool-result-1",
        call_id=call.call_id,
        content='{"selected":"record-7"}',
        error=None,
        metadata={},
    )
    service.runtime.accept_delegated_tool_result(
        worker.session_id,
        call.call_id,
        source="ag_ui",
        external_request_id="accepted-result-run",
        result=result,
    )
    service.workers.store.transition(
        worker.worker_id,
        "input_required",
        expected={working.status},
        event_type="worker_delegated_tool_input_required",
        event_payload={"call_id": call.call_id, "tool_name": call.tool_name},
    )
    payload = _ag_ui_input(run_id="different-result-run")
    payload["tools"] = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    ]
    payload["messages"].append(
        {
            "id": result.message_id,
            "role": "tool",
            "toolCallId": result.call_id,
            "content": result.content,
        }
    )

    with pytest.raises(ValueError, match="different exact result"):
        service._ag_ui_begin(AgUiProjectionAdapter().user_run(payload))
    assert (
        service.store.protocol_state_for_message("ag_ui", "different-result-run")
        is None
    )
    service.workers.shutdown()


def test_ag_ui_resume_validates_and_resolves_the_durable_interrupt(
    make_config, monkeypatch
) -> None:
    service = CommunicationService(AgentRuntime(make_config(), model_client=object()))

    def require_input(worker_id: str):
        service.workers.store.transition(
            worker_id,
            "queued",
            expected={"created"},
            event_type="worker_queued",
        )
        service.workers.store.transition(
            worker_id,
            "working",
            expected={"queued"},
            event_type="worker_started",
        )
        return service.workers.store.transition(
            worker_id,
            "input_required",
            expected={"working"},
            result="Provide the exact approval.",
            event_type="worker_input_required",
        )

    monkeypatch.setattr(service.workers, "start", require_input)
    (
        first_record,
        _first_start,
        _first_end,
        _first_duplicate,
        _first_state,
    ) = service._ag_ui_begin(AgUiProjectionAdapter().user_run(_ag_ui_input()))
    interrupt_event = next(
        item
        for item in reversed(service.workers.store.events(first_record.worker_id))
        if item.event_type == "worker_input_required"
    )
    interrupt_id = f"{first_record.worker_id}-input-{interrupt_event.sequence}"
    observed: dict[str, str] = {}

    def resolve_without_model(worker_id: str, message: str, *, source: str, **_):
        observed["message"] = message
        observed["source"] = source
        service.workers.store.transition(
            worker_id,
            "queued",
            expected={"input_required"},
            event_type="worker_resumed",
        )
        service.workers.store.transition(
            worker_id,
            "working",
            expected={"queued"},
            event_type="worker_started",
        )
        return service.workers.store.transition(
            worker_id,
            "completed",
            expected={"working"},
            result="completed after exact approval",
            event_type="worker_completed",
        )

    monkeypatch.setattr(service.workers, "message", resolve_without_model)
    resolved, second_start, _second_end, duplicate, _second_state = (
        service._ag_ui_begin(
            AgUiProjectionAdapter().user_run(
                _ag_ui_input(
                    run_id="run-2",
                    resume=[
                        {
                            "interruptId": interrupt_id,
                            "status": "resolved",
                            "payload": {
                                "approved": True,
                                "reason": "exact evidence",
                            },
                        }
                    ],
                )
            )
        )
    )
    first_bounds = service.store.protocol_message_bounds("ag_ui", "run-1")
    service.workers.shutdown()

    assert resolved.status == "completed"
    assert duplicate is False
    assert observed["source"] == "ag_ui:run-2"
    assert observed["message"].startswith(
        'AG-UI interrupt response:\n{"approved":true,"reason":"exact evidence"}'
    )
    assert '"state":{}' in observed["message"]
    assert first_bounds is not None and first_bounds[3] == second_start


def test_ag_ui_new_run_supersedes_old_stream_without_misattributing_events(
    make_config, monkeypatch
) -> None:
    async def exercise() -> None:
        service = CommunicationService(
            AgentRuntime(make_config(), model_client=object())
        )

        def queue_without_model(worker_id: str):
            return service.workers.store.transition(
                worker_id,
                "queued",
                expected={"created"},
                event_type="worker_queued",
            )

        def complete_redirect(worker_id: str, _message: str, **_):
            service.workers.store.transition(
                worker_id,
                "working",
                expected={"queued"},
                event_type="worker_started",
            )
            return service.workers.store.transition(
                worker_id,
                "completed",
                expected={"working"},
                result="new run result",
                event_type="worker_completed",
            )

        monkeypatch.setattr(service.workers, "start", queue_without_model)
        monkeypatch.setattr(service.workers, "message", complete_redirect)
        server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        async def connect(payload: dict):
            body = json.dumps(payload).encode()
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"POST /ag-ui HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Accept: text/event-stream\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            await reader.readuntil(b"\r\n\r\n")
            first = json.loads(
                (await reader.readuntil(b"\n\n")).removeprefix(b"data: ")
            )
            return reader, writer, first

        old_reader, old_writer, old_started = await connect(_ag_ui_input())
        new_reader, new_writer, new_started = await connect(
            _ag_ui_input(run_id="run-2")
        )
        old_tail = await asyncio.wait_for(old_reader.read(), timeout=2)
        new_tail = await asyncio.wait_for(new_reader.read(), timeout=2)
        old_events = [
            json.loads(block.removeprefix(b"data: "))
            for block in old_tail.split(b"\n\n")
            if block.startswith(b"data: ")
        ]
        new_events = [
            json.loads(block.removeprefix(b"data: "))
            for block in new_tail.split(b"\n\n")
            if block.startswith(b"data: ")
        ]
        old_writer.close()
        new_writer.close()
        await old_writer.wait_closed()
        await new_writer.wait_closed()
        server.close()
        await server.wait_closed()
        first_bounds = service.store.protocol_message_bounds("ag_ui", "run-1")
        second_bounds = service.store.protocol_message_bounds("ag_ui", "run-2")
        service.workers.shutdown()

        assert old_started["runId"] == "run-1"
        assert old_events[-1]["type"] == "RUN_ERROR"
        assert old_events[-1]["code"] == "SWAAG_RUN_SUPERSEDED"
        assert new_started["runId"] == "run-2"
        assert new_events[-1]["type"] == "RUN_FINISHED"
        assert new_events[-1]["runId"] == "run-2"
        assert new_events[-1]["result"] == "new run result"
        assert first_bounds is not None and second_bounds is not None
        assert first_bounds[3] == second_bounds[2]

    asyncio.run(exercise())
