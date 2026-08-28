#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from swaag.communication import CommunicationService
from swaag.config import load_config
from swaag.delegated_tools import DelegatedToolInputRequired
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
    runtime = AgentRuntime(config, model_client=no_inference)
    service = CommunicationService(runtime)

    def request_tool_without_executor(worker_id: str):
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
        catalog = runtime.delegated_tools.latest_catalog(working.session_id)
        if catalog is None or len(catalog.tools) != 1:
            raise RuntimeError("official client tool catalog was not bound exactly")
        state = runtime.history.rebuild_from_history(
            working.session_id, write_projections=False
        )
        try:
            runtime._request_delegated_tool(
                state,
                catalog=catalog,
                spec=catalog.tools[0],
                arguments={"record_id": "record-7"},
            )
        except DelegatedToolInputRequired as wait:
            service.workers._sync_history_events(working)
            return service.workers.store.transition(
                worker_id,
                "input_required",
                expected={"working"},
                event_type="worker_delegated_tool_input_required",
                event_payload={
                    "call_id": wait.call.call_id,
                    "tool_name": wait.call.tool_name,
                    "arguments": wait.call.arguments,
                    "arguments_sha256": wait.call.arguments_sha256,
                    "catalog_revision": wait.call.catalog_revision,
                },
            )
        raise RuntimeError("delegated tool request unexpectedly returned")

    def complete_without_executor(
        worker_id: str, _message: str, *, source: str, **_kwargs: Any
    ):
        current = service.workers.store.get(worker_id)
        delegated_result = next(
            (
                event
                for event in runtime.history.iter_history_reverse(
                    current.session_id, event_types=("tool_result", "tool_error")
                )
                if event.payload.get("delegated") is True
            ),
            None,
        )
        if delegated_result is None or delegated_result.event_type != "tool_result":
            raise RuntimeError("exact delegated tool result was not accepted")
        if source != f"ag_ui:{args.second_run_id}":
            raise RuntimeError("post-tool worker source lost AG-UI run provenance")
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
            result=args.expected_result,
            event_type="worker_completed",
        )

    service.workers.start = request_tool_without_executor  # type: ignore[method-assign]
    service.workers.message = complete_without_executor  # type: ignore[method-assign]
    server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    probe = Path(__file__).with_name("ag-ui-client-tool-conformance.mjs")
    command = [
        "node",
        str(probe),
        str(Path(args.sdk_root).expanduser().resolve()),
        f"http://127.0.0.1:{port}",
        args.thread_id,
        args.first_run_id,
        args.second_run_id,
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
                f"official AG-UI client-tool probe exited {process.returncode}: "
                f"{stderr.strip()}"
            )
        sdk_result = json.loads(stdout)
        if sdk_result.get("result") != args.expected_result:
            raise RuntimeError("official AG-UI client-tool probe returned wrong result")
        if no_inference.accesses:
            raise RuntimeError(
                "model client was accessed: " + ", ".join(no_inference.accesses)
            )
        worker = service.workers.list()[0]
        call = next(
            item
            for item in [runtime.delegated_tools.call(sdk_result["toolCall"]["id"])]
            if item is not None
        )
        result = {
            "completed_at": utc_now_iso(),
            "scope": "official AG-UI client-provided tool round-trip conformance",
            "inference_allowed": False,
            "model_client_accesses": list(no_inference.accesses),
            "service": {
                "host": "127.0.0.1",
                "port": port,
                "state_root": str(config.sessions.root),
                "worker_execution": "deterministic-client-tool-round-trip",
            },
            "process": {
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
            "worker": {
                "worker_id": worker.worker_id,
                "session_id": worker.session_id,
                "status": worker.status,
            },
            "delegated_call": {
                "call_id": call.call_id,
                "status": call.status,
                "tool_name": call.tool_name,
                "arguments_sha256": call.arguments_sha256,
                "result_message_id": call.result_message_id,
                "history_event_sequence": call.history_event_sequence,
                "history_event_hash": call.history_event_hash,
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
            "Run official AG-UI client-tool conformance against a deterministic "
            "Swaag service that cannot invoke inference."
        )
    )
    parser.add_argument("sdk_root")
    parser.add_argument("output_dir")
    parser.add_argument("expected_result")
    parser.add_argument("--thread-id", default="official-client-tool-thread")
    parser.add_argument("--first-run-id", default="official-client-tool-run-1")
    parser.add_argument("--second-run-id", default="official-client-tool-run-2")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    result = asyncio.run(_run(args))
    print(stable_json_dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
