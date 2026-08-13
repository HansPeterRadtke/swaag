from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any, TextIO

from swaag.runtime import AgentRuntime


class McpAdapter:
    """Minimal MCP JSON-RPC adapter over SWAAG's canonical runtime/tool registry."""

    protocol_version = "2026-07-28"

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    def _result(self, request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _error(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if method == "notifications/initialized":
            return None
        if method == "initialize":
            return self._result(request_id, {
                "protocolVersion": self.protocol_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "swaag", "version": "0.1"},
            })
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            tools = []
            for tool in self.runtime.tools.enabled_tools(self.runtime.config):
                tools.append({
                    "name": tool.name,
                    "description": tool.description + (f" {tool.usage_guidance}" if tool.usage_guidance else ""),
                    "inputSchema": tool.input_schema,
                })
            return self._result(request_id, {"tools": tools})
        if method == "tools/call":
            name = str(params.get("name", ""))
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return self._error(request_id, -32602, "tools/call arguments must be an object")
            session_ref = params.get("session")
            session_id = self.runtime.resolve_session_ref(session_ref, latest_if_none=True)
            try:
                run = self.runtime.execute_tool_once(name, arguments, session_id=session_id)
            except Exception as exc:
                return self._error(request_id, -32000, f"{type(exc).__name__}: {exc}")
            result = run.tool_result
            payload = result.output if result is not None else {}
            display = result.display_text if result is not None else ""
            return self._result(request_id, {
                "content": [{"type": "text", "text": display or json.dumps(payload, sort_keys=True)}],
                "structuredContent": payload,
                "isError": False,
                "session_id": run.session_id,
            })
        return self._error(request_id, -32601, f"Method not found: {method}")

    def serve_stdio(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for line in stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                response = self.handle(request)
            except Exception as exc:
                response = self._error(None, -32700, f"Invalid request: {exc}")
            if response is not None:
                stdout.write(json.dumps(response, sort_keys=True) + "\n")
                stdout.flush()
