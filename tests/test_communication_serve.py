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
        a2a = await request(
            {"op": "a2a.get", "params": {"worker_id": worker_id}}
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
        assert a2a["result"]["task"]["id"] == worker_id
        assert open_webui["result"]["metadata"]["worker_id"] == worker_id

    asyncio.run(exercise())
