#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import Any

from swaag.communication import CommunicationService
from swaag.mcp import McpInputRequired
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
        raise AssertionError(
            f"MCP conformance probe attempted model-client access: {name}"
        )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    workspace_root = Path(args.workspace_root).expanduser().resolve()

    config = load_config(env={})
    config.sessions.root = output_dir / "state"
    config.embedding_index.enabled = False
    config.mcp.enabled = True
    config.mcp.transport = "streamable_http"
    config.communication.enabled = False
    config.communication.host = "127.0.0.1"
    config.tools.read_roots = [workspace_root]
    no_inference = _NoInferenceClient()
    runtime = AgentRuntime(config, model_client=no_inference)
    runtime.create_or_load_session()

    calculator = runtime.tools.get("calculator")
    calculator_schema = copy.deepcopy(calculator.input_schema)
    calculator_schema["properties"]["expression"]["x-mcp-header"] = "Expression"
    calculator.input_schema = calculator_schema

    service = CommunicationService(runtime)

    def _calculator_mrtr(arguments: dict[str, Any], context):
        response = context.input_responses.get("multiplier")
        if response is None:
            return McpInputRequired(
                input_requests={
                    "multiplier": {
                        "method": "elicitation/create",
                        "params": {
                            "mode": "form",
                            "message": "Choose a deterministic multiplier",
                            "requestedSchema": {
                                "type": "object",
                                "properties": {
                                    "multiplier": {"type": "number", "title": "Multiplier"}
                                },
                                "required": ["multiplier"],
                            },
                        },
                    }
                },
                request_state="calculator-mrtr-v1",
            )
        if context.request_state != "calculator-mrtr-v1":
            raise ValueError("calculator MRTR requestState mismatch")
        if response.get("action") != "accept":
            raise ValueError("calculator MRTR elicitation was not accepted")
        content = response.get("content")
        if not isinstance(content, dict):
            raise ValueError("calculator MRTR elicitation content is missing")
        multiplier = content.get("multiplier")
        if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
            raise ValueError("calculator MRTR multiplier must be numeric")
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression:
            raise ValueError("calculator expression must be a non-empty string")
        return {"expression": f"({expression}) * ({multiplier})"}

    service.mcp.register_multi_round_trip_handler("calculator", _calculator_mrtr)
    server = await asyncio.start_server(service.handle_client, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    probe = Path(__file__).with_name("mcp-http-sdk-conformance.mjs")
    subscription_ready = output_dir / "subscription-ready"
    command = [
        "node",
        str(probe),
        str(Path(args.sdk_root).expanduser().resolve()),
        f"http://127.0.0.1:{port}/mcp",
        str(workspace_root),
        str(subscription_ready),
    ]
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + min(float(args.timeout_seconds), 15.0)
        while not subscription_ready.exists():
            if process.returncode is not None:
                stdout_bytes, stderr_bytes = await process.communicate()
                raise RuntimeError(
                    "official MCP SDK probe exited before subscription acknowledgement: "
                    + stderr_bytes.decode("utf-8").strip()
                )
            if loop.time() >= deadline:
                raise TimeoutError(
                    "official MCP SDK probe did not acknowledge subscriptions/listen"
                )
            await asyncio.sleep(0.05)
        calculator.description = calculator.description + " [conformance-catalog-revision]"
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=float(args.timeout_seconds)
        )
        stdout = stdout_bytes.decode("utf-8")
        stderr = stderr_bytes.decode("utf-8")
        if process.returncode != 0:
            raise RuntimeError(
                f"official MCP SDK probe exited {process.returncode}: {stderr.strip()}"
            )
        sdk_result = json.loads(stdout)
        calculator_result = sdk_result.get("toolCalls", {}).get("calculator", {})
        if calculator_result.get("result") != 126:
            raise RuntimeError("official MCP SDK probe omitted the verified MRTR tool result")
        if calculator_result.get("elicitationCalls") != 1:
            raise RuntimeError("official MCP SDK probe did not auto-fulfill exactly one elicitation")
        if calculator_result.get("mirroredParameterHeader") != "Expression":
            raise RuntimeError("official MCP SDK probe omitted mirrored-header evidence")
        subscription_result = sdk_result.get("subscription", {})
        if subscription_result.get("honoredFilter") != {"toolsListChanged": True}:
            raise RuntimeError("official MCP SDK probe omitted honored subscription evidence")
        if subscription_result.get("notificationMethod") != "notifications/tools/list_changed":
            raise RuntimeError("official MCP SDK probe omitted tools/list_changed evidence")
        if subscription_result.get("closeCause") != "local":
            raise RuntimeError("official MCP SDK probe did not close its subscription locally")
        if no_inference.accesses:
            raise RuntimeError(
                "model client was accessed: " + ", ".join(no_inference.accesses)
            )
        result = {
            "completed_at": utc_now_iso(),
            "scope": "official MCP Streamable HTTP SDK conformance",
            "inference_allowed": False,
            "model_client_accesses": list(no_inference.accesses),
            "service": {
                "host": "127.0.0.1",
                "port": port,
                "endpoint": f"http://127.0.0.1:{port}/mcp",
                "state_root": str(config.sessions.root),
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
            "Run official MCP Streamable HTTP conformance against an ephemeral "
            "Swaag service that cannot invoke inference."
        )
    )
    parser.add_argument("sdk_root")
    parser.add_argument("output_dir")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    result = asyncio.run(_run(args))
    print(stable_json_dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
