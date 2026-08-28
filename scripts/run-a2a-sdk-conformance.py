#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from swaag.communication import CommunicationService
from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.utils import stable_json_dumps, utc_now_iso


class _NoInferenceClient:
    is_deterministic_test_client = True
    mode = ""

    def __init__(self) -> None:
        self.accesses: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.accesses.append(name)
        raise AssertionError(f"conformance probe attempted model-client access: {name}")


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    config = load_config(env={})
    config.sessions.root = output_dir / "state"
    config.embedding_index.enabled = False
    config.communication.enabled = False
    config.communication.host = "127.0.0.1"
    no_inference = _NoInferenceClient()
    service = CommunicationService(AgentRuntime(config, model_client=no_inference))

    def queue_without_executor(worker_id: str):
        return service.workers.store.transition(
            worker_id,
            "queued",
            expected={"created"},
            event_type="worker_queued",
        )

    # The protocol is under test; model execution is intentionally outside scope.
    service.workers.start = queue_without_executor  # type: ignore[method-assign]
    expected_task_id: str | None = None
    if args.exercise_existing_task:
        seeded = service.task_api.execute(
            "create",
            {
                "objective": "Exercise official existing-task operations.",
                "start": True,
            },
        )
        expected_task_id = str(seeded["worker"]["worker_id"])
    server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
    socket = server.sockets[0]
    port = int(socket.getsockname()[1])
    service._advertised_host = "127.0.0.1"
    service._advertised_port = port
    probe = Path(__file__).with_name("a2a-sdk-conformance.mjs")
    command = [
        "node",
        str(probe),
        str(Path(args.sdk_root).expanduser().resolve()),
        f"http://127.0.0.1:{port}",
    ]
    if expected_task_id is not None:
        command.append(expected_task_id)
    command.extend(
        ["--exercise-new-tasks", f"--transport={args.transport}"]
    )
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=float(args.timeout_seconds)
        )
        stdout = stdout_bytes.decode("utf-8")
        stderr = stderr_bytes.decode("utf-8")
        if process.returncode != 0:
            raise RuntimeError(
                f"official A2A SDK probe exited {process.returncode}: {stderr.strip()}"
            )
        sdk_result = json.loads(stdout)
        if sdk_result.get("newTasks") is None:
            raise RuntimeError("official A2A SDK probe omitted new-task evidence")
        if no_inference.accesses:
            raise RuntimeError(
                "model client was accessed: " + ", ".join(no_inference.accesses)
            )
        result = {
            "completed_at": utc_now_iso(),
            "scope": (
                "official A2A SDK new-task send/stream protocol conformance "
                f"over {args.transport}"
            ),
            "inference_allowed": False,
            "model_client_accesses": list(no_inference.accesses),
            "service": {
                "host": "127.0.0.1",
                "port": port,
                "state_root": str(config.sessions.root),
                "worker_execution": "queue-only",
                "seeded_task_id": expected_task_id,
            },
            "process": {
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
            "sdk_result": sdk_result,
        }
        (output_dir / "result.json").write_text(
            stable_json_dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        server.close()
        await server.wait_closed()
        service.workers.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run official A2A new-task SDK conformance against a queue-only "
            "Swaag service that cannot invoke inference."
        )
    )
    parser.add_argument("sdk_root")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--transport", choices=("jsonrpc", "http-json"), default="jsonrpc"
    )
    parser.add_argument("--exercise-existing-task", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    result = asyncio.run(_run(args))
    print(stable_json_dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
