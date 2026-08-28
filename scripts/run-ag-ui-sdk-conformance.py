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

    def complete_without_executor(worker_id: str):
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
            increment_run_count=True,
            event_type="worker_started",
        )
        return service.workers.store.transition(
            worker_id,
            "completed",
            expected={"working"},
            result=args.expected_result,
            event_type="worker_completed",
        )

    # The protocol is under test; model execution is intentionally outside scope.
    service.workers.start = complete_without_executor  # type: ignore[method-assign]
    server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    probe = Path(__file__).with_name("ag-ui-sdk-conformance.mjs")
    command = [
        "node",
        str(probe),
        str(Path(args.sdk_root).expanduser().resolve()),
        f"http://127.0.0.1:{port}",
        args.thread_id,
        args.run_id,
        args.expected_result,
    ]
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
                f"official AG-UI SDK probe exited {process.returncode}: {stderr.strip()}"
            )
        sdk_result = json.loads(stdout)
        if sdk_result.get("result") != args.expected_result:
            raise RuntimeError("official AG-UI SDK probe returned the wrong result")
        if no_inference.accesses:
            raise RuntimeError(
                "model client was accessed: " + ", ".join(no_inference.accesses)
            )
        result = {
            "completed_at": utc_now_iso(),
            "scope": "official AG-UI SDK new-run protocol conformance",
            "inference_allowed": False,
            "model_client_accesses": list(no_inference.accesses),
            "service": {
                "host": "127.0.0.1",
                "port": port,
                "state_root": str(config.sessions.root),
                "worker_execution": "deterministic-completion",
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
            "Run official AG-UI new-run SDK conformance against a deterministic "
            "Swaag service that cannot invoke inference."
        )
    )
    parser.add_argument("sdk_root")
    parser.add_argument("output_dir")
    parser.add_argument("expected_result")
    parser.add_argument("--thread-id", default="official-new-thread")
    parser.add_argument("--run-id", default="official-new-run")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    result = asyncio.run(_run(args))
    print(stable_json_dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
