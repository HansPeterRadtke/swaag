from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import ipaddress
import json
import re
import sys
import threading

import requests
from typing import Any, Callable, Mapping, TextIO
from urllib.parse import urlsplit

from swaag.config import McpAuthorizationConfig
from swaag.runtime import AgentRuntime
from swaag.tools.base import ToolValidationError


_HEADER_MISMATCH = -32020
_UNSUPPORTED_PROTOCOL_VERSION = -32022
_MCP_HEADER_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_MCP_NAME_METHODS = {
    "prompts/get": "name",
    "resources/read": "uri",
    "tools/call": "name",
}
_MISSING = object()
_MAX_SAFE_INTEGER = 9_007_199_254_740_991


@dataclass(slots=True, frozen=True)
class McpHttpResponse:
    status: int
    payload: dict[str, Any] | None
    headers: Mapping[str, str] | None = None


@dataclass(slots=True, frozen=True)
class McpHttpSubscription:
    request_id: str | int
    honored_filter: dict[str, Any]
    cancelled: threading.Event
    initial_tool_catalog_sha256: str


@dataclass(slots=True, frozen=True)
class McpMultiRoundTripContext:
    input_responses: Mapping[str, dict[str, Any]]
    request_state: str | None


@dataclass(slots=True, frozen=True)
class McpInputRequired:
    input_requests: Mapping[str, dict[str, Any]]
    request_state: str | None = None


McpMultiRoundTripHandler = Callable[
    [dict[str, Any], McpMultiRoundTripContext], McpInputRequired | dict[str, Any]
]


class McpOAuthResourceServer:
    """OAuth 2.x protected-resource boundary for externally exposed MCP HTTP."""

    def __init__(self, config: McpAuthorizationConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def protected_resource_metadata(self) -> dict[str, Any]:
        if not self.enabled:
            raise ValueError("MCP OAuth protected-resource metadata is disabled")
        payload: dict[str, Any] = {
            "resource": self.config.resource_uri,
            "authorization_servers": list(self.config.authorization_servers),
            "bearer_methods_supported": ["header"],
        }
        if self.config.required_scopes:
            payload["scopes_supported"] = list(self.config.required_scopes)
        return payload

    def _challenge(self, *, error: str | None = None, description: str | None = None, scope: str | None = None) -> str:
        metadata_url = self.config.resource_uri.rstrip("/")
        metadata_url = metadata_url.split("://", 1)
        if len(metadata_url) != 2:
            raise ValueError("MCP OAuth resource URI must be absolute")
        scheme, rest = metadata_url
        host, _, path = rest.partition("/")
        well_known = f"{scheme}://{host}/.well-known/oauth-protected-resource"
        if path:
            well_known += "/" + path
        parts = [f'resource_metadata="{well_known}"']
        if error:
            parts.append(f'error="{error}"')
        if description:
            safe = description.replace('"', "'")
            parts.append(f'error_description="{safe}"')
        if scope:
            parts.append(f'scope="{scope}"')
        return "Bearer " + ", ".join(parts)

    @staticmethod
    def _audiences(payload: Mapping[str, Any]) -> set[str]:
        raw = payload.get("aud")
        if isinstance(raw, str):
            return {raw}
        if isinstance(raw, list):
            return {str(item) for item in raw if isinstance(item, str)}
        raw = payload.get("resource")
        return {raw} if isinstance(raw, str) else set()

    @staticmethod
    def _scopes(payload: Mapping[str, Any]) -> set[str]:
        raw = payload.get("scope")
        if isinstance(raw, str):
            return {item for item in raw.split() if item}
        raw = payload.get("scp")
        if isinstance(raw, list):
            return {str(item) for item in raw if isinstance(item, str)}
        return set()

    def authorize(self, headers: Mapping[str, str]) -> McpHttpResponse | None:
        if not self.enabled:
            return None
        authorization = headers.get("authorization", "")
        if not authorization.startswith("Bearer ") or not authorization[7:].strip():
            return McpHttpResponse(
                401,
                {"error": "unauthorized"},
                {"WWW-Authenticate": self._challenge()},
            )
        token = authorization[7:].strip()
        try:
            response = requests.post(
                self.config.introspection_url,
                data={"token": token, "token_type_hint": "access_token"},
                auth=(
                    self.config.introspection_client_id,
                    self.config.introspection_client_secret,
                ),
                headers={"Accept": "application/json"},
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            return McpHttpResponse(
                503,
                {"error": "authorization_server_unavailable"},
                {"Cache-Control": "no-store"},
            )
        if not isinstance(payload, dict) or payload.get("active") is not True:
            return McpHttpResponse(
                401,
                {"error": "invalid_token"},
                {
                    "WWW-Authenticate": self._challenge(
                        error="invalid_token", description="Access token is inactive or invalid"
                    )
                },
            )
        if self.config.resource_uri not in self._audiences(payload):
            return McpHttpResponse(
                401,
                {"error": "invalid_token"},
                {
                    "WWW-Authenticate": self._challenge(
                        error="invalid_token", description="Access token audience does not match this MCP resource"
                    )
                },
            )
        missing = [scope for scope in self.config.required_scopes if scope not in self._scopes(payload)]
        if missing:
            required = " ".join(self.config.required_scopes)
            return McpHttpResponse(
                403,
                {"error": "insufficient_scope", "required_scopes": list(self.config.required_scopes)},
                {
                    "WWW-Authenticate": self._challenge(
                        error="insufficient_scope",
                        description="Access token lacks required scope",
                        scope=required,
                    )
                },
            )
        return None


@dataclass(slots=True, frozen=True)
class _MirroredParameter:
    header_name: str
    path: tuple[str, ...]
    value_type: str


class McpAdapter:
    """Stateless MCP adapter over SWAAG's canonical capability registry."""

    protocol_version = "2026-07-28"

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        multi_round_trip_handlers: Mapping[str, McpMultiRoundTripHandler] | None = None,
    ):
        self.runtime = runtime
        self._subscription_lock = threading.Lock()
        self._subscriptions: dict[str | int, threading.Event] = {}
        self._multi_round_trip_handlers: dict[str, McpMultiRoundTripHandler] = dict(
            multi_round_trip_handlers or {}
        )

    def register_multi_round_trip_handler(
        self, tool_name: str, handler: McpMultiRoundTripHandler
    ) -> None:
        name = str(tool_name).strip()
        if not name:
            raise ValueError("MCP multi-round-trip tool name must be non-empty")
        if not callable(handler):
            raise TypeError("MCP multi-round-trip handler must be callable")
        self._multi_round_trip_handlers[name] = handler

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

    def _error(
        self,
        request_id: Any,
        code: int,
        message: str,
        *,
        data: Any = _MISSING,
    ) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not _MISSING:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": request_id, "error": error}

    def _unsupported_version(
        self, request_id: Any, requested_version: str
    ) -> dict[str, Any]:
        return self._error(
            request_id,
            _UNSUPPORTED_PROTOCOL_VERSION,
            f"Unsupported protocol version: {requested_version!r}",
            data={
                "supported": [self.protocol_version],
                "requested": requested_version,
            },
        )

    def _request_validation_error(
        self, request: dict[str, Any]
    ) -> dict[str, Any] | None:
        request_id = request.get("id")
        if request.get("jsonrpc") != "2.0":
            return self._error(request_id, -32600, "jsonrpc must be '2.0'")
        if "id" in request and (
            isinstance(request_id, bool)
            or request_id is None
            or not isinstance(request_id, (str, int))
        ):
            return self._error(None, -32600, "request id must be a string or integer")
        method = request.get("method")
        if not isinstance(method, str) or not method:
            return self._error(request_id, -32600, "method must be a non-empty string")
        params = request.get("params", {})
        if not isinstance(params, dict):
            return self._error(request_id, -32602, "params must be an object")
        metadata = params.get("_meta")
        if not isinstance(metadata, dict):
            return self._error(request_id, -32602, "Every MCP request requires params._meta")
        requested_version = metadata.get("io.modelcontextprotocol/protocolVersion")
        if not isinstance(requested_version, str) or not requested_version:
            return self._error(
                request_id,
                -32602,
                (
                    "params._meta.io.modelcontextprotocol/protocolVersion "
                    "must be a non-empty string"
                ),
            )
        if requested_version != self.protocol_version:
            return self._unsupported_version(request_id, requested_version)
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
        return None

    @staticmethod
    def _multi_round_trip_context(
        params: Mapping[str, Any], request_id: Any
    ) -> tuple[McpMultiRoundTripContext | None, dict[str, Any] | None]:
        raw_responses = params.get("inputResponses", {})
        if not isinstance(raw_responses, dict):
            return None, {
                "code": -32602,
                "message": "tools/call inputResponses must be an object",
            }
        responses: dict[str, dict[str, Any]] = {}
        for key, value in raw_responses.items():
            if not isinstance(key, str) or not key:
                return None, {
                    "code": -32602,
                    "message": "tools/call inputResponses keys must be non-empty strings",
                }
            if not isinstance(value, dict) or "method" in value or "result" in value:
                return None, {
                    "code": -32602,
                    "message": (
                        "tools/call inputResponses entries must be bare MCP input response objects"
                    ),
                }
            responses[key] = dict(value)
        request_state = params.get("requestState")
        if request_state is not None and not isinstance(request_state, str):
            return None, {
                "code": -32602,
                "message": "tools/call requestState must be a string",
            }
        return McpMultiRoundTripContext(responses, request_state), None

    @staticmethod
    def _validate_input_required(result: McpInputRequired) -> dict[str, Any]:
        input_requests = dict(result.input_requests)
        if not input_requests and result.request_state is None:
            raise ValueError(
                "MCP input_required requires at least one input request or requestState"
            )
        for key, request in input_requests.items():
            if not isinstance(key, str) or not key:
                raise ValueError("MCP input request keys must be non-empty strings")
            if not isinstance(request, dict):
                raise ValueError("MCP input requests must be objects")
            method = request.get("method")
            params = request.get("params")
            if method not in {"elicitation/create", "sampling/createMessage", "roots/list"}:
                raise ValueError(f"Unsupported MCP input request method: {method!r}")
            if not isinstance(params, dict):
                raise ValueError("MCP embedded input request params must be an object")
        if result.request_state is not None and not isinstance(result.request_state, str):
            raise ValueError("MCP input_required requestState must be a string")
        return {
            "resultType": "input_required",
            **({"inputRequests": input_requests} if input_requests else {}),
            **(
                {"requestState": result.request_state}
                if result.request_state is not None
                else {}
            ),
        }

    def handle(
        self,
        request: dict[str, Any],
        *,
        transport: str = "stdio",
    ) -> dict[str, Any] | None:
        validation_error = self._request_validation_error(request)
        if validation_error is not None:
            return validation_error
        request_id = request.get("id")
        method = str(request["method"])
        params = request.get("params", {})
        metadata = params["_meta"]
        if "id" not in request:
            if method == "notifications/cancelled":
                cancelled_id = params.get("requestId")
                if (
                    cancelled_id is not None
                    and not isinstance(cancelled_id, bool)
                    and isinstance(cancelled_id, (str, int))
                ):
                    self.cancel_http_subscription(cancelled_id)
            return None
        if method == "server/discover":
            tool_capabilities: dict[str, Any] = {}
            if transport == "streamable_http":
                tool_capabilities["listChanged"] = True
            return self._result(
                request_id,
                {
                    "supportedVersions": [self.protocol_version],
                    "capabilities": {"tools": tool_capabilities},
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
                self.runtime.tools.enabled_tools(self.runtime.config),
                key=lambda item: item.name,
            ):
                tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description
                        + (f" {tool.usage_guidance}" if tool.usage_guidance else ""),
                        "inputSchema": tool.input_schema,
                    }
                )
            return self._result(
                request_id,
                {"tools": tools, "ttlMs": 0, "cacheScope": "private"},
            )
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str) or not name:
                return self._error(
                    request_id, -32602, "tools/call name must be a non-empty string"
                )
            arguments = params.get("arguments", {})
            if not isinstance(arguments, dict):
                return self._error(
                    request_id, -32602, "tools/call arguments must be an object"
                )
            enabled = {
                tool.name: tool
                for tool in self.runtime.tools.enabled_tools(self.runtime.config)
            }
            tool = enabled.get(name)
            if tool is None:
                return self._error(
                    request_id, -32602, f"Unknown or disabled tool: {name}"
                )
            mrtr_handler = self._multi_round_trip_handlers.get(name)
            if mrtr_handler is not None:
                mrtr_context, mrtr_error = self._multi_round_trip_context(params, request_id)
                if mrtr_error is not None:
                    return self._error(
                        request_id, int(mrtr_error["code"]), str(mrtr_error["message"])
                    )
                assert mrtr_context is not None
                try:
                    mrtr_result = mrtr_handler(dict(arguments), mrtr_context)
                    if isinstance(mrtr_result, McpInputRequired):
                        return self._result(
                            request_id, self._validate_input_required(mrtr_result)
                        )
                    if not isinstance(mrtr_result, dict):
                        raise TypeError(
                            "MCP multi-round-trip handler must return McpInputRequired or tool arguments"
                        )
                    arguments = dict(mrtr_result)
                except (TypeError, ValueError) as exc:
                    return self._error(request_id, -32602, str(exc))
            elif "inputResponses" in params or "requestState" in params:
                return self._error(
                    request_id,
                    -32602,
                    f"Tool does not support MCP multi-round-trip input: {name}",
                )
            try:
                tool.validate(arguments)
            except ToolValidationError as exc:
                return self._error(request_id, -32602, str(exc))
            session_ref = metadata.get("com.swaag/sessionId")
            if session_ref is not None and not isinstance(session_ref, str):
                return self._error(
                    request_id,
                    -32602,
                    "com.swaag/sessionId metadata must be a string",
                )
            try:
                session_id = self.runtime.resolve_session_ref(
                    session_ref, latest_if_none=True
                )
                run = self.runtime.execute_tool_once(
                    name, arguments, session_id=session_id
                )
            except FileNotFoundError as exc:
                return self._error(request_id, -32602, str(exc))
            except Exception as exc:
                return self._error(
                    request_id, -32000, f"{type(exc).__name__}: {exc}"
                )
            if run.error is not None:
                return self._result(
                    request_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(run.error, sort_keys=True),
                            }
                        ],
                        "structuredContent": {"error": run.error},
                        "isError": True,
                        "_meta": {"com.swaag/sessionId": run.session_id},
                    },
                )
            result = run.tool_result
            if result is None:
                return self._error(
                    request_id, -32000, "Tool finished without a result or error"
                )
            payload = result.output
            display = result.display_text
            return self._result(
                request_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": display or json.dumps(payload, sort_keys=True),
                        }
                    ],
                    "structuredContent": payload,
                    "isError": False,
                    "_meta": {"com.swaag/sessionId": run.session_id},
                },
            )
        if method == "subscriptions/listen":
            return self._error(
                request_id,
                -32601,
                "subscriptions/listen requires the Streamable HTTP transport",
            )
        return self._error(request_id, -32601, f"Method not found: {method}")

    def tool_catalog_sha256(self) -> str:
        tools = [
            {
                "name": tool.name,
                "description": tool.description
                + (f" {tool.usage_guidance}" if tool.usage_guidance else ""),
                "inputSchema": tool.input_schema,
            }
            for tool in sorted(
                self.runtime.tools.enabled_tools(self.runtime.config),
                key=lambda item: item.name,
            )
        ]
        payload = json.dumps(
            tools,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def prepare_http_subscription(
        self,
        request: dict[str, Any],
        headers: Mapping[str, str],
    ) -> McpHttpSubscription | McpHttpResponse:
        request_id = request.get("id")
        validation_error = self._request_validation_error(request)
        if validation_error is not None:
            return McpHttpResponse(400, validation_error)
        if request.get("method") != "subscriptions/listen":
            return McpHttpResponse(
                400,
                self._error(request_id, -32600, "Expected subscriptions/listen"),
            )
        if "id" not in request:
            return McpHttpResponse(
                400,
                self._error(None, -32600, "subscriptions/listen requires an id"),
            )
        mismatch = self._validate_request_headers(request, headers)
        if mismatch is not None:
            return McpHttpResponse(
                400, self._error(request_id, _HEADER_MISMATCH, mismatch)
            )
        params = request["params"]
        extra_params = set(params) - {"_meta", "notifications"}
        if extra_params:
            return McpHttpResponse(
                200,
                self._error(
                    request_id,
                    -32602,
                    "subscriptions/listen has unsupported parameters: "
                    + ", ".join(sorted(extra_params)),
                ),
            )
        notifications = params.get("notifications")
        if not isinstance(notifications, dict):
            return McpHttpResponse(
                200,
                self._error(
                    request_id,
                    -32602,
                    "subscriptions/listen notifications must be an object",
                ),
            )
        allowed = {
            "toolsListChanged",
            "promptsListChanged",
            "resourcesListChanged",
            "resourceSubscriptions",
        }
        unknown = set(notifications) - allowed
        if unknown:
            return McpHttpResponse(
                200,
                self._error(
                    request_id,
                    -32602,
                    "subscriptions/listen has unknown notification filters: "
                    + ", ".join(sorted(unknown)),
                ),
            )
        for name in (
            "toolsListChanged",
            "promptsListChanged",
            "resourcesListChanged",
        ):
            value = notifications.get(name)
            if value is not None and not isinstance(value, bool):
                return McpHttpResponse(
                    200,
                    self._error(
                        request_id,
                        -32602,
                        f"subscriptions/listen {name} must be a boolean",
                    ),
                )
        resource_subscriptions = notifications.get("resourceSubscriptions")
        if resource_subscriptions is not None and (
            not isinstance(resource_subscriptions, list)
            or any(not isinstance(item, str) for item in resource_subscriptions)
        ):
            return McpHttpResponse(
                200,
                self._error(
                    request_id,
                    -32602,
                    "subscriptions/listen resourceSubscriptions must be a string array",
                ),
            )

        honored_filter = (
            {"toolsListChanged": True}
            if notifications.get("toolsListChanged") is True
            else {}
        )
        cancelled = threading.Event()
        catalog_sha256 = self.tool_catalog_sha256()
        with self._subscription_lock:
            if request_id in self._subscriptions:
                return McpHttpResponse(
                    200,
                    self._error(
                        request_id,
                        -32603,
                        "A subscriptions/listen request with this id is already active",
                    ),
                )
            self._subscriptions[request_id] = cancelled
        return McpHttpSubscription(
            request_id=request_id,
            honored_filter=honored_filter,
            cancelled=cancelled,
            initial_tool_catalog_sha256=catalog_sha256,
        )

    def cancel_http_subscription(self, request_id: str | int) -> bool:
        with self._subscription_lock:
            event = self._subscriptions.get(request_id)
        if event is None:
            return False
        event.set()
        return True

    def finish_http_subscription(self, request_id: str | int) -> None:
        with self._subscription_lock:
            self._subscriptions.pop(request_id, None)

    def http_preflight(
        self, headers: Mapping[str, str]
    ) -> McpHttpResponse | None:
        origin_response = self.http_origin_preflight(headers)
        if origin_response is not None:
            return origin_response
        content_type = headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type.casefold() != "application/json":
            return McpHttpResponse(
                415,
                self._error(None, -32600, "Content-Type must be application/json"),
            )
        accepted = self._accepted_media_types(headers.get("accept", ""))
        if not {"application/json", "text/event-stream"}.issubset(accepted):
            return McpHttpResponse(
                406,
                self._error(
                    None,
                    -32600,
                    "Accept must list application/json and text/event-stream",
                ),
            )
        return None

    def http_origin_preflight(
        self, headers: Mapping[str, str]
    ) -> McpHttpResponse | None:
        origin = headers.get("origin")
        if origin is not None and not self._is_allowed_origin(origin):
            return McpHttpResponse(
                403,
                self._error(
                    None,
                    -32600,
                    "Origin is not allowed by this local MCP endpoint",
                ),
            )
        return None

    def handle_http(
        self,
        request: dict[str, Any],
        headers: Mapping[str, str],
    ) -> McpHttpResponse:
        request_id = request.get("id")
        if request.get("jsonrpc") != "2.0":
            return McpHttpResponse(
                400, self._error(request_id, -32600, "Invalid JSON-RPC request")
            )
        if "id" in request and (
            isinstance(request_id, bool)
            or request_id is None
            or not isinstance(request_id, (str, int))
        ):
            return McpHttpResponse(
                400,
                self._error(None, -32600, "request id must be a string or integer"),
            )
        if "id" not in request:
            response = self.handle(request)
            return McpHttpResponse(202 if response is None else 400, response)

        mismatch = self._validate_request_headers(request, headers)
        if mismatch is not None:
            return McpHttpResponse(
                400, self._error(request_id, _HEADER_MISMATCH, mismatch)
            )
        response = self.handle(request, transport="streamable_http")
        if response is None:
            return McpHttpResponse(202, None)
        error = response.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        if code == _UNSUPPORTED_PROTOCOL_VERSION:
            return McpHttpResponse(400, response)
        if code == -32601:
            return McpHttpResponse(404, response)
        return McpHttpResponse(200, response)

    def _validate_request_headers(
        self,
        request: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> str | None:
        method = request.get("method")
        params = request.get("params")
        metadata = params.get("_meta") if isinstance(params, dict) else None
        body_version = (
            metadata.get("io.modelcontextprotocol/protocolVersion")
            if isinstance(metadata, dict)
            else None
        )
        header_version = headers.get("mcp-protocol-version")
        if (
            not isinstance(body_version, str)
            or header_version is None
            or header_version != body_version
        ):
            return (
                "MCP-Protocol-Version header is missing or does not match "
                "params._meta"
            )
        header_method = headers.get("mcp-method")
        if (
            not isinstance(method, str)
            or header_method is None
            or header_method != method
        ):
            return "Mcp-Method header is missing or does not match method"
        source_field = _MCP_NAME_METHODS.get(method)
        if source_field is not None:
            body_name = params.get(source_field) if isinstance(params, dict) else None
            try:
                header_name = self._decode_header_value(headers.get("mcp-name"))
            except ValueError as exc:
                return f"Mcp-Name header is malformed: {exc}"
            if not isinstance(body_name, str) or header_name != body_name:
                return "Mcp-Name header is missing or does not match the request body"
        if method != "tools/call" or not isinstance(params, dict):
            return None
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        enabled = {
            tool.name: tool
            for tool in self.runtime.tools.enabled_tools(self.runtime.config)
        }
        tool = enabled.get(name)
        if tool is None:
            return None
        try:
            mirrored = self._mirrored_parameters(tool.input_schema)
        except ValueError as exc:
            return f"Tool schema has an invalid x-mcp-header declaration: {exc}"
        for parameter in mirrored:
            body_value = self._value_at_path(arguments, parameter.path)
            header_key = "mcp-param-" + parameter.header_name.casefold()
            encoded_value = headers.get(header_key)
            if body_value is _MISSING or body_value is None:
                if encoded_value is not None:
                    return f"Mcp-Param-{parameter.header_name} must be omitted"
                continue
            if encoded_value is None:
                return f"Mcp-Param-{parameter.header_name} header is required"
            try:
                decoded_value = self._decode_header_value(encoded_value)
                matches = self._mirrored_value_matches(
                    body_value, decoded_value, parameter.value_type
                )
            except ValueError as exc:
                return f"Mcp-Param-{parameter.header_name} is malformed: {exc}"
            if not matches:
                return (
                    f"Mcp-Param-{parameter.header_name} header does not match "
                    "the request body"
                )
        return None

    def _is_allowed_origin(self, origin: str) -> bool:
        if not origin or origin == "null" or any(char.isspace() for char in origin):
            return False
        try:
            parsed = urlsplit(origin)
            parsed.port
        except ValueError:
            return False
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return False
        if parsed.hostname.casefold() == "localhost":
            return True
        try:
            if ipaddress.ip_address(parsed.hostname).is_loopback:
                return True
        except ValueError:
            pass
        if self.runtime.config.mcp.authorization.enabled:
            normalized = origin.rstrip("/")
            return normalized in {item.rstrip("/") for item in self.runtime.config.mcp.authorization.allowed_origins}
        return False

    @staticmethod
    def _accepted_media_types(raw_accept: str) -> set[str]:
        accepted: set[str] = set()
        for part in raw_accept.split(","):
            fields = [field.strip() for field in part.split(";")]
            media_type = fields[0].casefold()
            if not media_type:
                continue
            quality = "1"
            for field in fields[1:]:
                if field.casefold().startswith("q="):
                    quality = field[2:].strip()
            try:
                parsed_quality = Decimal(quality)
                enabled = parsed_quality.is_finite() and 0 < parsed_quality <= 1
            except InvalidOperation:
                enabled = False
            if enabled:
                accepted.add(media_type)
        return accepted

    @staticmethod
    def _decode_header_value(value: str | None) -> str:
        if value is None:
            raise ValueError("header is missing")
        if value.startswith("=?base64?") and value.endswith("?="):
            payload = value[len("=?base64?") : -2]
            try:
                decoded = base64.b64decode(payload, validate=True)
                return decoded.decode("utf-8")
            except (binascii.Error, UnicodeDecodeError) as exc:
                raise ValueError("invalid Base64 UTF-8 sentinel value") from exc
        if not value or value != value.strip() or any(
            ord(char) < 0x20 or ord(char) > 0x7E for char in value
        ):
            raise ValueError("plain value is not safe visible ASCII")
        return value

    @classmethod
    def _mirrored_parameters(
        cls, schema: Mapping[str, Any]
    ) -> tuple[_MirroredParameter, ...]:
        if not isinstance(schema, Mapping):
            return ()
        found: list[_MirroredParameter] = []
        seen: set[str] = set()

        def visit(node: Any, path: tuple[str, ...], *, property_node: bool) -> None:
            if not isinstance(node, Mapping):
                return
            if "x-mcp-header" in node:
                name = node["x-mcp-header"]
                value_type = node.get("type")
                if not property_node:
                    raise ValueError(
                        "annotation is not statically reachable through properties"
                    )
                if not isinstance(name, str) or not _MCP_HEADER_TOKEN.fullmatch(name):
                    raise ValueError("header name must be a non-empty HTTP token")
                normalized = name.casefold()
                if normalized in seen:
                    raise ValueError(
                        "header names must be case-insensitively unique"
                    )
                if value_type not in {"string", "integer", "boolean"}:
                    raise ValueError(
                        "annotated property must have primitive string, integer, "
                        "or boolean type"
                    )
                seen.add(normalized)
                found.append(_MirroredParameter(name, path, value_type))
            properties = node.get("properties")
            if isinstance(properties, Mapping):
                if node.get("type") not in {None, "object"}:
                    if cls._contains_header_annotation(properties):
                        raise ValueError(
                            "annotation is not reachable through object properties"
                        )
                else:
                    for key, child in properties.items():
                        if isinstance(key, str):
                            visit(child, (*path, key), property_node=True)
            for key, child in node.items():
                if key in {"properties", "x-mcp-header"}:
                    continue
                if cls._contains_header_annotation(child):
                    raise ValueError(
                        "annotation is nested beneath a non-properties schema keyword"
                    )

        visit(schema, (), property_node=False)
        return tuple(found)

    @classmethod
    def _contains_header_annotation(cls, value: Any) -> bool:
        if isinstance(value, Mapping):
            return "x-mcp-header" in value or any(
                cls._contains_header_annotation(child) for child in value.values()
            )
        if isinstance(value, list):
            return any(cls._contains_header_annotation(child) for child in value)
        return False

    @staticmethod
    def _value_at_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                return _MISSING
            current = current[key]
        return current

    @staticmethod
    def _mirrored_value_matches(
        body_value: Any, header_value: str, value_type: str
    ) -> bool:
        if value_type == "string":
            if not isinstance(body_value, str):
                raise ValueError("body value is not a string")
            return body_value == header_value
        if value_type == "boolean":
            if not isinstance(body_value, bool):
                raise ValueError("body value is not a boolean")
            return header_value == ("true" if body_value else "false")
        if (
            isinstance(body_value, bool)
            or not isinstance(body_value, int)
            or abs(body_value) > _MAX_SAFE_INTEGER
        ):
            raise ValueError("body value is not a JavaScript-safe integer")
        try:
            header_number = Decimal(header_value)
        except InvalidOperation as exc:
            raise ValueError("header value is not numeric") from exc
        return header_number.is_finite() and header_number == body_value

    def serve_stdio(
        self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout
    ) -> None:
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
