#!/usr/bin/env python3

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
from typing import Any

from swaag.config import ExternalMcpServerConfig
from swaag.external_mcp import ExternalMcpClient
from swaag.utils import stable_json_dumps, utc_now_iso

PROTOCOL_VERSION = "2026-07-28"
SERVER_INFO = {"name": "external-conformance-server", "version": "1.0.0"}


def _response(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    request_id = request.get("id")
    if method == "server/discover":
        result = {
            "resultType": "complete",
            "supportedVersions": [PROTOCOL_VERSION],
            "capabilities": {"tools": {}},
            "ttlMs": 0,
            "cacheScope": "private",
        }
    elif method == "tools/list":
        result = {
            "resultType": "complete",
            "tools": [
                {
                    "name": "external_echo",
                    "description": "Echo external structured evidence.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                }
            ],
            "ttlMs": 0,
            "cacheScope": "private",
        }
    elif method == "tools/call":
        params = request.get("params", {})
        if params.get("name") != "external_echo":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": "Unknown tool"},
            }
        arguments = params.get("arguments", {})
        text = arguments.get("text")
        result = {
            "resultType": "complete",
            "content": [{"type": "text", "text": str(text)}],
            "structuredContent": {"echo": text},
            "isError": False,
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    result["_meta"] = {"io.modelcontextprotocol/serverInfo": dict(SERVER_INFO)}
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sdk_root")
    parser.add_argument("output_dir")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    observed_headers: list[dict[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            request = json.loads(self.rfile.read(length))
            observed_headers.append(
                {
                    "protocol": self.headers.get("MCP-Protocol-Version"),
                    "method": self.headers.get("Mcp-Method"),
                    "name": self.headers.get("Mcp-Name"),
                }
            )
            body = stable_json_dumps(_response(request), indent=None).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/mcp"
    try:
        swaag_client = ExternalMcpClient(
            "conformance",
            ExternalMcpServerConfig(
                enabled=True,
                optional=False,
                transport="streamable_http",
                command=[],
                url=endpoint,
                header_env={},
                credential_command=[],
                credential_refresh_skew_seconds=30.0,
                timeout_seconds=float(args.timeout_seconds),
            ),
        )
        swaag_tools = swaag_client.list_tools()
        if [tool.name for tool in swaag_tools] != ["external_echo"]:
            raise RuntimeError("SWAAG client did not decode external_echo")
        swaag_call = swaag_client.call_tool("external_echo", {"text": "swaag-client"})
        if swaag_call.is_error or swaag_call.structured_content != {"echo": "swaag-client"}:
            raise RuntimeError("SWAAG client did not decode external_echo result")

        probe = Path(__file__).with_name("external-mcp-http-sdk-conformance.mjs")
        completed = subprocess.run(
            ["node", str(probe), str(Path(args.sdk_root).resolve()), endpoint],
            text=True,
            capture_output=True,
            timeout=float(args.timeout_seconds),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"official MCP SDK external HTTP probe failed: {completed.stderr.strip()}"
            )
        official = json.loads(completed.stdout)
        result = {
            "completed_at": utc_now_iso(),
            "scope": "external MCP Streamable HTTP client conformance",
            "endpoint": endpoint,
            "swaag_client": {
                "tool_names": [tool.name for tool in swaag_tools],
                "call": swaag_call.structured_content,
            },
            "official_sdk": official,
            "observed_request_headers": observed_headers,
        }
        (output_dir / "result.json").write_text(
            stable_json_dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(stable_json_dumps(result, indent=2))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
