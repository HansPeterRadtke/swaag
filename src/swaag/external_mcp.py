from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import time
from typing import Any, Iterable

import requests

from swaag.config import ExternalMcpServerConfig, ExternalToolsConfig
from swaag.delegated_tools import DelegatedToolSpec, prepare_delegated_tool_spec
from swaag.utils import stable_json_dumps

MCP_PROTOCOL_VERSION = "2026-07-28"


class ExternalMcpError(RuntimeError):
    pass


def _schema_allows_null(schema: dict[str, Any]) -> bool:
    if schema.get("type") == "null":
        return True
    schema_type = schema.get("type")
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    variants = schema.get("anyOf")
    return isinstance(variants, list) and any(
        isinstance(item, dict) and _schema_allows_null(item) for item in variants
    )


def _portable_mcp_schema(schema: Any, *, root: bool = False) -> dict[str, Any]:
    """Project ordinary MCP JSON Schema into SWAAG's constrained portable subset.

    The returned schema is only for model-facing constrained decoding. The exact
    original MCP schema is retained separately and remains authoritative for the
    external server. Optional object properties become required nullable fields;
    a generated null is removed again before tools/call so omission/default
    semantics remain owned by the MCP server.
    """
    if not isinstance(schema, dict):
        raise ExternalMcpError("MCP input schema nodes must be JSON objects")

    if "allOf" in schema:
        variants = schema.get("allOf")
        if not isinstance(variants, list) or not variants:
            raise ExternalMcpError("MCP allOf must be a non-empty list")
        # Safely merge the common object-allOf shape. Conflicting/non-object
        # compositions remain unsupported rather than being weakened silently.
        if all(isinstance(item, dict) and item.get("type", "object") == "object" for item in variants):
            merged_properties: dict[str, Any] = {}
            merged_required: list[str] = []
            for item in variants:
                props = item.get("properties", {})
                if not isinstance(props, dict):
                    raise ExternalMcpError("MCP allOf object properties must be objects")
                overlap = set(merged_properties) & set(props)
                if overlap:
                    raise ExternalMcpError(
                        "MCP allOf with overlapping properties is not safely normalizable: "
                        + ", ".join(sorted(overlap))
                    )
                merged_properties.update(props)
                required = item.get("required", [])
                if isinstance(required, list):
                    merged_required.extend(str(value) for value in required)
            schema = {
                "type": "object",
                "properties": merged_properties,
                "required": sorted(set(merged_required)),
                "additionalProperties": False,
            }
        else:
            raise ExternalMcpError("MCP non-object allOf is not safely normalizable")

    variants = schema.get("anyOf")
    if variants is None:
        variants = schema.get("oneOf")
    if variants is not None:
        if root:
            raise ExternalMcpError("MCP root union schemas are not supported for tool arguments")
        if not isinstance(variants, list) or not variants:
            raise ExternalMcpError("MCP union schema must contain variants")
        return {
            "anyOf": [
                _portable_mcp_schema(item, root=False)
                for item in variants
                if isinstance(item, dict)
            ]
        }

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if root:
            if schema_type != ["object"] and set(schema_type) != {"object"}:
                raise ExternalMcpError("MCP root input schema must resolve to object")
            schema_type = "object"
        else:
            return {
                "anyOf": [
                    _portable_mcp_schema({**schema, "type": item}, root=False)
                    for item in schema_type
                ]
            }

    if schema_type is None and (root or "properties" in schema):
        schema_type = "object"

    if schema_type == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ExternalMcpError("MCP object schema properties must be an object")
        required_raw = schema.get("required", [])
        if not isinstance(required_raw, list):
            raise ExternalMcpError("MCP object schema required must be an array when present")
        originally_required = {str(item) for item in required_raw}
        portable_properties: dict[str, Any] = {}
        for name, child in properties.items():
            if not isinstance(name, str) or not isinstance(child, dict):
                raise ExternalMcpError("MCP object properties must map names to schema objects")
            portable_child = _portable_mcp_schema(child, root=False)
            if name not in originally_required and not _schema_allows_null(portable_child):
                portable_child = {"anyOf": [portable_child, {"type": "null"}]}
            portable_properties[name] = portable_child
        return {
            "type": "object",
            "properties": portable_properties,
            "required": list(portable_properties),
            "additionalProperties": False,
        }

    if schema_type == "array":
        items = schema.get("items", {})
        if not isinstance(items, dict):
            raise ExternalMcpError("MCP array items must be a schema object")
        return {"type": "array", "items": _portable_mcp_schema(items, root=False)}

    if schema_type in {"string", "integer", "number", "boolean", "null"}:
        result: dict[str, Any] = {"type": schema_type}
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            # Keep only enum values compatible with this exact variant.
            result["enum"] = list(enum)
        return result

    # A pure enum without type is common enough to infer safely when homogeneous.
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        values = [value for value in enum if value is not None]
        kinds = {
            "boolean" if isinstance(value, bool) else
            "integer" if isinstance(value, int) else
            "number" if isinstance(value, float) else
            "string" if isinstance(value, str) else
            "unsupported"
            for value in values
        }
        if len(kinds) == 1 and "unsupported" not in kinds:
            kind = next(iter(kinds))
            base = {"type": kind, "enum": [value for value in enum if value is not None]}
            if None in enum:
                return {"anyOf": [base, {"type": "null"}]}
            return base
    raise ExternalMcpError(
        f"MCP schema type is not safely normalizable: {schema_type!r}"
    )


def _mcp_execution_arguments(
    arguments: dict[str, Any], original_schema: dict[str, Any]
) -> dict[str, Any]:
    """Undo only the nullable placeholders introduced by normalization."""
    properties = original_schema.get("properties", {})
    required_raw = original_schema.get("required", [])
    required = {str(item) for item in required_raw} if isinstance(required_raw, list) else set()
    if not isinstance(properties, dict):
        return dict(arguments)
    prepared: dict[str, Any] = {}
    for key, value in arguments.items():
        child_schema = properties.get(key)
        if (
            value is None
            and key not in required
            and isinstance(child_schema, dict)
            and not _schema_allows_null(child_schema)
        ):
            continue
        prepared[key] = value
    return prepared


@dataclass(slots=True, frozen=True)
class ExternalMcpTool:
    server_name: str
    spec: DelegatedToolSpec


@dataclass(slots=True, frozen=True)
class ExternalMcpCallResult:
    server_name: str
    tool_name: str
    structured_content: dict[str, Any]
    content: list[dict[str, Any]]
    is_error: bool
    raw_result: dict[str, Any]


class ExternalMcpClient:
    def __init__(self, server_name: str, config: ExternalMcpServerConfig):
        self.server_name = server_name
        self.config = config
        self._request_index = 0
        self._credential_headers: dict[str, str] = {}
        self._credential_expires_at: float | None = None

    def _request_payload(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._request_index += 1
        payload = dict(params or {})
        payload["_meta"] = {
            "io.modelcontextprotocol/protocolVersion": MCP_PROTOCOL_VERSION,
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {"name": "swaag", "version": "0.1.0"},
        }
        return {
            "jsonrpc": "2.0",
            "id": f"swaag-{self.server_name}-{self._request_index}",
            "method": method,
            "params": payload,
        }

    def _stdio_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.config.command:
            raise ExternalMcpError(f"MCP server {self.server_name} has no stdio command")
        try:
            completed = subprocess.run(
                self.config.command,
                input=stable_json_dumps(request, indent=None) + "\n",
                text=True,
                capture_output=True,
                timeout=float(self.config.timeout_seconds),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExternalMcpError(
                f"MCP stdio server {self.server_name} unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
            raise ExternalMcpError(f"MCP stdio server {self.server_name} failed: {detail}")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise ExternalMcpError(f"MCP stdio server {self.server_name} returned no response")
        try:
            response = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise ExternalMcpError(
                f"MCP stdio server {self.server_name} returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(response, dict):
            raise ExternalMcpError(f"MCP stdio server {self.server_name} returned a non-object response")
        return response


    def _run_credential_provider(self, reason: str) -> dict[str, str]:
        if not self.config.credential_command:
            return {}
        request = {
            "server_name": self.server_name,
            "url": self.config.url,
            "reason": reason,
        }
        try:
            completed = subprocess.run(
                self.config.credential_command,
                input=stable_json_dumps(request, indent=None) + "\n",
                text=True,
                capture_output=True,
                timeout=float(self.config.timeout_seconds),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExternalMcpError(
                f"MCP credential provider for {self.server_name} unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
            raise ExternalMcpError(
                f"MCP credential provider for {self.server_name} failed: {detail}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ExternalMcpError(
                f"MCP credential provider for {self.server_name} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ExternalMcpError(
                f"MCP credential provider for {self.server_name} returned a non-object response"
            )
        raw_headers = payload.get("headers", {})
        if not isinstance(raw_headers, dict) or not raw_headers:
            raise ExternalMcpError(
                f"MCP credential provider for {self.server_name} returned no headers"
            )
        headers: dict[str, str] = {}
        for header_name, value in raw_headers.items():
            if not isinstance(header_name, str) or not header_name.strip() or any(
                ch in header_name for ch in "\r\n:"
            ):
                raise ExternalMcpError(
                    f"MCP credential provider for {self.server_name} returned an invalid header name"
                )
            if not isinstance(value, str) or not value or "\r" in value or "\n" in value:
                raise ExternalMcpError(
                    f"MCP credential provider for {self.server_name} returned an invalid header value"
                )
            headers[header_name] = value
        expires_at = payload.get("expires_at_epoch")
        if expires_at is None:
            self._credential_expires_at = None
        elif isinstance(expires_at, (int, float)) and not isinstance(expires_at, bool):
            self._credential_expires_at = float(expires_at)
        else:
            raise ExternalMcpError(
                f"MCP credential provider for {self.server_name} returned invalid expires_at_epoch"
            )
        self._credential_headers = headers
        return dict(headers)

    def _credential_headers_for_request(self, *, force_refresh: bool = False) -> dict[str, str]:
        if not self.config.credential_command:
            return {}
        now = time.time()
        refresh_due = (
            not self._credential_headers
            or force_refresh
            or (
                self._credential_expires_at is not None
                and self._credential_expires_at
                <= now + float(self.config.credential_refresh_skew_seconds)
            )
        )
        if refresh_due:
            return self._run_credential_provider(
                "unauthorized" if force_refresh else "initial_or_expired"
            )
        return dict(self._credential_headers)

    def _http_headers(self, *, force_refresh: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json, text/event-stream"}
        for header_name, env_name in self.config.header_env.items():
            value = os.environ.get(env_name)
            if not value:
                raise ExternalMcpError(
                    f"MCP HTTP server {self.server_name} requires environment variable {env_name} for header {header_name}"
                )
            headers[header_name] = value
        for header_name, value in self._credential_headers_for_request(
            force_refresh=force_refresh
        ).items():
            if header_name in headers and headers[header_name] != value:
                raise ExternalMcpError(
                    f"MCP credential provider for {self.server_name} conflicts with configured header {header_name}"
                )
            headers[header_name] = value
        return headers

    def _http_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.config.url:
            raise ExternalMcpError(f"MCP server {self.server_name} has no Streamable HTTP URL")
        headers = self._http_headers()
        try:
            response = requests.post(
                self.config.url,
                json=request,
                headers=headers,
                timeout=float(self.config.timeout_seconds),
            )
            if (
                response.status_code in {401, 403}
                and self.config.credential_command
            ):
                response = requests.post(
                    self.config.url,
                    json=request,
                    headers=self._http_headers(force_refresh=True),
                    timeout=float(self.config.timeout_seconds),
                )
        except requests.RequestException as exc:
            raise ExternalMcpError(
                f"MCP HTTP server {self.server_name} unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise ExternalMcpError(
                f"MCP HTTP server {self.server_name} returned HTTP {response.status_code}: {response.text[:1000]}"
            )
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" in content_type:
            payloads: list[dict[str, Any]] = []
            for line in response.text.splitlines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payloads.append(parsed)
            if not payloads:
                raise ExternalMcpError(f"MCP HTTP server {self.server_name} returned no SSE JSON response")
            return payloads[-1]
        try:
            parsed = response.json()
        except ValueError as exc:
            raise ExternalMcpError(f"MCP HTTP server {self.server_name} returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ExternalMcpError(f"MCP HTTP server {self.server_name} returned a non-object response")
        return parsed

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request = self._request_payload(method, params)
        response = (
            self._stdio_request(request)
            if self.config.transport == "stdio"
            else self._http_request(request)
        )
        if response.get("jsonrpc") != "2.0" or response.get("id") != request["id"]:
            raise ExternalMcpError(f"MCP server {self.server_name} returned an invalid JSON-RPC envelope")
        error = response.get("error")
        if isinstance(error, dict):
            raise ExternalMcpError(
                f"MCP server {self.server_name} {method} failed: {error.get('code')}: {error.get('message')}"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise ExternalMcpError(f"MCP server {self.server_name} {method} returned no result object")
        return result

    def list_tools(self) -> tuple[DelegatedToolSpec, ...]:
        result = self.request("tools/list")
        raw_tools = result.get("tools", [])
        if not isinstance(raw_tools, list):
            raise ExternalMcpError(f"MCP server {self.server_name} tools/list returned invalid tools")
        specs: list[DelegatedToolSpec] = []
        for item in raw_tools:
            if not isinstance(item, dict):
                raise ExternalMcpError(f"MCP server {self.server_name} returned a non-object tool")
            input_schema = item.get("inputSchema", {"type": "object", "properties": {}})
            if not isinstance(input_schema, dict):
                raise ExternalMcpError(
                    f"MCP server {self.server_name} tool {item.get('name')} has invalid inputSchema"
                )
            portable_schema = _portable_mcp_schema(input_schema, root=True)
            spec = prepare_delegated_tool_spec(
                {
                    "name": item.get("name"),
                    "description": item.get("description", ""),
                    "parameters": portable_schema,
                    "metadata": {
                        "external_executor": "mcp",
                        "mcp_server": self.server_name,
                        "mcp_input_schema": input_schema,
                    },
                }
            )
            specs.append(spec)
        return tuple(specs)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ExternalMcpCallResult:
        result = self.request("tools/call", {"name": tool_name, "arguments": arguments})
        content = result.get("content", [])
        if not isinstance(content, list):
            content = []
        structured = result.get("structuredContent", {})
        if not isinstance(structured, dict):
            structured = {"value": structured}
        return ExternalMcpCallResult(
            server_name=self.server_name,
            tool_name=tool_name,
            structured_content=structured,
            content=[item for item in content if isinstance(item, dict)],
            is_error=bool(result.get("isError", False)),
            raw_result=result,
        )


class ExternalMcpManager:
    def __init__(self, config: ExternalToolsConfig):
        self.config = config
        self._clients = {
            name: ExternalMcpClient(name, server)
            for name, server in config.mcp_servers.items()
            if server.enabled
        }
        self._tools_by_name: dict[str, ExternalMcpTool] = {}
        self._discovery_errors: dict[str, str] = {}
        self.refresh()

    @property
    def discovery_errors(self) -> dict[str, str]:
        return dict(self._discovery_errors)

    def refresh(self) -> tuple[DelegatedToolSpec, ...]:
        tools: dict[str, ExternalMcpTool] = {}
        errors: dict[str, str] = {}
        for name, client in self._clients.items():
            server_config = self.config.mcp_servers[name]
            try:
                specs = client.list_tools()
            except Exception as exc:
                if server_config.optional:
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    continue
                raise
            for spec in specs:
                if spec.name in tools:
                    other = tools[spec.name].server_name
                    raise ExternalMcpError(
                        f"External MCP tool collision: {spec.name} from {other} and {name}"
                    )
                tools[spec.name] = ExternalMcpTool(server_name=name, spec=spec)
        self._tools_by_name = tools
        self._discovery_errors = errors
        return self.specs()

    def specs(self) -> tuple[DelegatedToolSpec, ...]:
        return tuple(item.spec for item in self._tools_by_name.values())

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools_by_name

    def call(self, tool_name: str, arguments: dict[str, Any]) -> ExternalMcpCallResult:
        try:
            item = self._tools_by_name[tool_name]
        except KeyError as exc:
            raise KeyError(f"Unknown external MCP tool: {tool_name}") from exc
        client = self._clients[item.server_name]
        original_schema = item.spec.metadata.get("mcp_input_schema", {})
        prepared_arguments = (
            _mcp_execution_arguments(arguments, original_schema)
            if isinstance(original_schema, dict)
            else dict(arguments)
        )
        return client.call_tool(tool_name, prepared_arguments)
