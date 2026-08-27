from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from swaag.runtime import AgentRuntime
from swaag.tools.base import ToolValidationError


class McpAdapter:
    """Stateless MCP adapter over SWAAG's canonical capability registry."""

    protocol_version = "2026-07-28"

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    def _result(self, request_id: Any, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            result = {
                "resultType": "complete",
                **result,
                "_meta": {
                    **dict(result.get("_meta", {})),
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "swaag",
                        "version": "0.1",
                    },
                },
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _error(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            return self._error(request_id, -32602, "params must be an object")
        metadata = params.get("_meta")
        if not isinstance(metadata, dict):
            return self._error(request_id, -32602, "Every MCP request requires params._meta")
        requested_version = metadata.get("io.modelcontextprotocol/protocolVersion")
        if requested_version != self.protocol_version:
            return self._error(
                request_id,
                -32001,
                f"UnsupportedProtocolVersion: {requested_version!r}; supported={self.protocol_version}",
            )
        if not isinstance(
            metadata.get("io.modelcontextprotocol/clientCapabilities"), dict
        ):
            return self._error(
                request_id,
                -32602,
                "params._meta.io.modelcontextprotocol/clientCapabilities must be an object",
            )
        client_info = metadata.get("io.modelcontextprotocol/clientInfo")
        if client_info is not None and (
            not isinstance(client_info, dict)
            or not isinstance(client_info.get("name"), str)
            or not client_info["name"]
            or not isinstance(client_info.get("version"), str)
            or not client_info["version"]
        ):
            return self._error(
                request_id,
                -32602,
                (
                    "params._meta.io.modelcontextprotocol/clientInfo, when present, "
                    "must contain non-empty string name and version fields"
                ),
            )
        if method == "server/discover":
            return self._result(
                request_id,
                {
                    "supportedVersions": [self.protocol_version],
                    "capabilities": {"tools": {}},
                    "instructions": (
                        "SWAAG exposes model-controlled capabilities. Worker/task lifecycle "
                        "uses the separate transport-neutral task API. Stateful capability "
                        "calls may carry the explicit com.swaag/sessionId request metadata handle."
                    ),
                    "ttlMs": 0,
                    "cacheScope": "private",
                },
            )
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            tools = []
            for tool in sorted(
                self.runtime.tools.enabled_tools(self.runtime.config), key=lambda item: item.name
            ):
                tools.append({
                    "name": tool.name,
                    "description": tool.description + (f" {tool.usage_guidance}" if tool.usage_guidance else ""),
                    "inputSchema": tool.input_schema,
                })
            return self._result(
                request_id,
                {"tools": tools, "ttlMs": 0, "cacheScope": "private"},
            )
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str) or not name:
                return self._error(request_id, -32602, "tools/call name must be a non-empty string")
            arguments = params.get("arguments", {})
            if not isinstance(arguments, dict):
                return self._error(request_id, -32602, "tools/call arguments must be an object")
            enabled = {
                tool.name: tool for tool in self.runtime.tools.enabled_tools(self.runtime.config)
            }
            tool = enabled.get(name)
            if tool is None:
                return self._error(request_id, -32602, f"Unknown or disabled tool: {name}")
            try:
                tool.validate(arguments)
            except ToolValidationError as exc:
                return self._error(request_id, -32602, str(exc))
            session_ref = metadata.get("com.swaag/sessionId")
            if session_ref is not None and not isinstance(session_ref, str):
                return self._error(
                    request_id, -32602, "com.swaag/sessionId metadata must be a string"
                )
            try:
                session_id = self.runtime.resolve_session_ref(session_ref, latest_if_none=True)
                run = self.runtime.execute_tool_once(name, arguments, session_id=session_id)
            except FileNotFoundError as exc:
                return self._error(request_id, -32602, str(exc))
            except Exception as exc:
                return self._error(request_id, -32000, f"{type(exc).__name__}: {exc}")
            if run.error is not None:
                return self._result(
                    request_id,
                    {
                        "content": [
                            {"type": "text", "text": json.dumps(run.error, sort_keys=True)}
                        ],
                        "structuredContent": {"error": run.error},
                        "isError": True,
                        "_meta": {"com.swaag/sessionId": run.session_id},
                    },
                )
            result = run.tool_result
            if result is None:
                return self._error(request_id, -32000, "Tool finished without a result or error")
            payload = result.output
            display = result.display_text
            return self._result(request_id, {
                "content": [{"type": "text", "text": display or json.dumps(payload, sort_keys=True)}],
                "structuredContent": payload,
                "isError": False,
                "_meta": {"com.swaag/sessionId": run.session_id},
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
