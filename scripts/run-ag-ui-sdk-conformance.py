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
from swaag.shared_state import shared_state_event_payload
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

    def complete_without_inference(worker_id: str) -> None:
        working = service.workers.store.transition(
            worker_id,
            "working",
            expected={"queued"},
            increment_run_count=True,
            event_type="worker_started",
        )
        read = service.runtime.execute_tool_once(
            "shared_state",
            {
                "operation": "read",
                "base_revision": None,
                "base_state_sha256": None,
                "patch": None,
            },
            session_id=working.session_id,
        )
        if read.error is not None or read.tool_result is None:
            raise RuntimeError(
                "deterministic shared-state read failed: "
                + stable_json_dumps(read.error, indent=None)
            )
        baseline = read.tool_result.output
        update = service.runtime.execute_tool_once(
            "shared_state",
            {
                "operation": "patch",
                "base_revision": baseline["revision"],
                "base_state_sha256": baseline["state_sha256"],
                "patch": [
                    {
                        "op": "add",
                        "path": "/agentProgress",
                        "value_json": (
                            '{"source":"swaag","status":"verified"}'
                        ),
                    }
                ],
            },
            session_id=working.session_id,
        )
        if update.error is not None or update.tool_result is None:
            raise RuntimeError(
                "deterministic shared-state tool failed: "
                + stable_json_dumps(update.error, indent=None)
            )
        service.workers._sync_history_events(working)
        service.workers.store.transition(
            worker_id,
            "completed",
            expected={"working"},
            result=args.expected_result,
            event_type="worker_completed",
        )

    # Keep normal asynchronous worker admission while replacing only inference work.
    service.workers._run_worker = complete_without_inference  # type: ignore[method-assign]
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
        worker_id = service.store.protocol_worker("ag_ui", args.thread_id)
        if worker_id is None:
            raise RuntimeError("AG-UI probe did not retain its durable worker binding")
        worker = service.workers.store.get(worker_id)
        latest_state = service.store.latest_protocol_state("ag_ui", args.thread_id)
        if latest_state is None or (
            latest_state.source_kind,
            latest_state.source_session_id,
            latest_state.history_event_sequence is not None,
            latest_state.history_event_hash is not None,
            latest_state.patch_sha256 is not None,
        ) != ("agent_patch", worker.session_id, True, True, True):
            raise RuntimeError("AG-UI probe did not retain complete state lineage")
        if latest_state.state != sdk_result.get("state"):
            raise RuntimeError(
                "official client state differs from the durable server state"
            )
        history_sequence = latest_state.history_event_sequence
        history_hash = latest_state.history_event_hash
        if history_sequence is None or history_hash is None:
            raise RuntimeError("AG-UI probe state history lineage disappeared")
        history_event = next(
            service.runtime.history.iter_history(
                worker.session_id,
                start_sequence=history_sequence,
                end_sequence=history_sequence,
            ),
            None,
        )
        if history_event is None or (
            history_event.event_type,
            history_event.hash,
            history_event.payload,
        ) != (
            "shared_state_updated",
            history_hash,
            shared_state_event_payload(latest_state),
        ):
            raise RuntimeError(
                "AG-UI probe state lacks exact canonical history provenance"
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
                "worker_execution": "deterministic-shared-state-tool",
            },
            "process": {
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
            "sdk_result": sdk_result,
            "durable_state": {
                "worker_id": worker.worker_id,
                "session_id": worker.session_id,
                "revision": latest_state.revision,
                "state_sha256": latest_state.state_sha256,
                "patch_sha256": latest_state.patch_sha256,
                "source_call_id": latest_state.source_call_id,
                "history_event_sequence": latest_state.history_event_sequence,
                "history_event_hash": latest_state.history_event_hash,
            },
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
