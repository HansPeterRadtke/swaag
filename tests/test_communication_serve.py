from __future__ import annotations

import asyncio
import json

from swaag.cli import _build_parser
from swaag.communication import CommunicationService
from swaag.heartbeat import watchdog_interval_seconds
from swaag.runtime import AgentRuntime


def test_communication_serve_cli_accepts_config_defaults():
    parser = _build_parser()
    args = parser.parse_args(["communication", "serve"] )
    assert args.communication_command == "serve"
    assert args.host is None
    assert args.port is None


def test_watchdog_interval_uses_half_systemd_window(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "20000000")
    assert watchdog_interval_seconds() == 10.0


def test_watchdog_interval_has_safe_default(monkeypatch):
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)
    assert watchdog_interval_seconds(default_seconds=7.0) == 7.0


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
