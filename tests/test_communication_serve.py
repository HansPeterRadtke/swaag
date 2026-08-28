from __future__ import annotations

import asyncio
import json

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from swaag.cli import _build_parser
from swaag.communication import CommunicationService, require_loopback_bind_host
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
        headers, events = await run_request(request)
        _duplicate_headers, duplicate_events = await run_request(request)
        workers = service.workers.list()
        attachments = service.workers.attachments(workers[0].worker_id)
        bounds = service.store.protocol_message_bounds("ag_ui", "run-1")
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
        assert events[-1]["type"] == "RUN_FINISHED"
        assert events[-1]["threadId"] == "thread-1"
        assert events[-1]["runId"] == "run-1"
        assert events[-1]["result"] == "exact AG-UI result"
        assert duplicate_events[0]["metadata"]["swaagDuplicateRun"] is True
        assert duplicate_events[-1]["result"] == "exact AG-UI result"
        assert len(workers) == 1
        assert attachments[0].original_name == "facts.pdf"
        assert attachments[0].source == "ag_ui"
        assert attachments[0].size_bytes == len(b"exact bytes")
        assert bounds is not None and bounds[3] is not None

    asyncio.run(exercise())


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
    first_record, _first_start, _first_end, _first_duplicate = service._ag_ui_begin(
        AgUiProjectionAdapter().user_run(_ag_ui_input())
    )
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
    resolved, second_start, _second_end, duplicate = service._ag_ui_begin(
        AgUiProjectionAdapter().user_run(
            _ag_ui_input(
                run_id="run-2",
                resume=[
                    {
                        "interruptId": interrupt_id,
                        "status": "resolved",
                        "payload": {"approved": True, "reason": "exact evidence"},
                    }
                ],
            )
        )
    )
    first_bounds = service.store.protocol_message_bounds("ag_ui", "run-1")
    service.workers.shutdown()

    assert resolved.status == "completed"
    assert duplicate is False
    assert observed["source"] == "ag_ui:run-2"
    assert observed["message"] == (
        'AG-UI interrupt response:\n{"approved":true,"reason":"exact evidence"}'
    )
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
