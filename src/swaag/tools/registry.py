from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Iterable

from swaag.config import AgentConfig
from swaag.environment.environment import AgentEnvironment
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.tools.artifacts import ARTIFACT_TOOLS
from swaag.tools.attachments import ATTACHMENT_TOOLS
from swaag.tools.builtin import BUILTIN_TOOLS
from swaag.tools.history import HISTORY_TOOLS
from swaag.tools.terminal import TERMINAL_TOOLS
from swaag.tools.control import CONTROL_TOOLS
from swaag.types import SessionState, ToolExecutionResult, ToolInvocation


class LoadToolsTool(Tool):
    name = "load_tools"
    description = "Load exact schemas for semantically selected capabilities from the compact capability index."
    usage_guidance = (
        "Use when the task needs a capability whose full schema is not currently loaded. "
        "Request only tools relevant to the next work; loaded schemas become available on the next model action."
    )
    kind = "pure"
    repeated_observation_is_redundant = False
    input_schema = {
        "type": "object",
        "properties": {
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["tool_names"],
        "additionalProperties": False,
    }

    def __init__(self, registry: "ToolRegistry"):
        self._registry = registry

    def validate(self, raw_input: dict) -> dict:
        names = raw_input.get("tool_names")
        if not isinstance(names, list) or not names:
            raise ToolValidationError("load_tools.tool_names must be a non-empty array")
        cleaned: list[str] = []
        for value in names:
            if not isinstance(value, str) or not value.strip():
                raise ToolValidationError("load_tools.tool_names must contain non-empty strings")
            name = value.strip()
            if name == self.name:
                continue
            if name not in cleaned:
                cleaned.append(name)
        if not cleaned:
            raise ToolValidationError("load_tools requires at least one non-discovery capability name")
        return {"tool_names": cleaned}

    def execute(self, validated_input: dict, context: ToolContext) -> ToolExecutionResult:
        enabled = {tool.name: tool for tool in self._registry.enabled_domain_tools(context.config)}
        requested = list(validated_input["tool_names"])
        selected = [name for name in requested if name in enabled]
        unavailable = [name for name in requested if name not in enabled]
        output = {
            "selected_tool_names": selected,
            "unavailable_tool_names": unavailable,
        }
        if not selected:
            detail = "No requested capabilities are currently enabled by configuration and policy."
        else:
            detail = "Loaded for subsequent actions: " + ", ".join(selected)
        if unavailable:
            detail += "; unavailable: " + ", ".join(unavailable)
        return ToolExecutionResult(self.name, output, detail)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or [
            *BUILTIN_TOOLS,
            *HISTORY_TOOLS,
            *ARTIFACT_TOOLS,
            *ATTACHMENT_TOOLS,
            *TERMINAL_TOOLS,
            *CONTROL_TOOLS,
        ]:
            self.register(tool)
        if "load_tools" not in self._tools:
            self.register(LoadToolsTool(self))

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool registration: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def enabled_domain_tools(self, config: AgentConfig) -> list[Tool]:
        tools = [self.get(name) for name in config.tools.enabled if name != "load_tools"]
        result: list[Tool] = []
        for tool in tools:
            if tool.kind == "stateful" and not config.tools.allow_stateful_tools:
                continue
            if tool.kind == "side_effect" and not config.tools.allow_side_effect_tools:
                continue
            result.append(tool)
        return result

    def enabled_tools(self, config: AgentConfig) -> list[Tool]:
        return [self.get("load_tools"), *self.enabled_domain_tools(config)]

    def prompt_tuples(self, config: AgentConfig) -> list[tuple[str, str, dict, str]]:
        """Compatibility/full-registry view. Production action loops use staged schemas."""
        return [tool.prompt_tuple() for tool in self.enabled_tools(config)]

    def staged_prompt_tuples(self, config: AgentConfig, selected_names: Iterable[str]) -> list[tuple[str, str, dict, str]]:
        selected = set(selected_names)
        return [
            self.get("load_tools").prompt_tuple(),
            *[tool.prompt_tuple() for tool in self.enabled_domain_tools(config) if tool.name in selected],
        ]

    def capability_index(self, config: AgentConfig) -> list[tuple[str, str, str]]:
        return [(tool.name, tool.description, tool.usage_guidance) for tool in self.enabled_domain_tools(config)]

    def tool_names(self, config: AgentConfig) -> list[str]:
        return [tool.name for tool in self.enabled_tools(config)]

    def prepare(self, name: str, raw_input: dict, config: AgentConfig, session_state: SessionState) -> tuple[Tool, ToolContext, ToolInvocation]:
        tool = self.get(name)
        session_copy = copy.deepcopy(session_state)
        context = ToolContext(config=config, session_state=session_copy, environment=AgentEnvironment(config, session_copy))
        validated = tool.validate(raw_input)
        effective_kind = tool.effective_kind(validated)
        if effective_kind == "stateful" and not config.tools.allow_stateful_tools:
            raise PermissionError(f"Tool disabled by policy: {name}")
        if effective_kind == "side_effect" and not config.tools.allow_side_effect_tools:
            raise PermissionError(f"Tool disabled by policy: {name}")
        invocation = ToolInvocation(tool_name=name, raw_input=raw_input, validated_input=validated)
        return tool, context, invocation

    def execute_prepared(self, tool: Tool, context: ToolContext, invocation: ToolInvocation) -> ToolExecutionResult:
        executor = ThreadPoolExecutor(max_workers=1)
        should_wait = True
        try:
            future = executor.submit(tool.execute, invocation.validated_input, context)
            timeout_seconds = float(tool.execution_timeout_seconds(context))
            if timeout_seconds <= 0:
                raise ValueError(f"Tool execution timeout must be positive: {tool.name}={timeout_seconds}")
            try:
                result = future.result(timeout=timeout_seconds)
            except FuturesTimeoutError as exc:
                future.cancel()
                should_wait = False
                raise TimeoutError(f"Tool timed out after {timeout_seconds:g}s: {tool.name}") from exc
        finally:
            executor.shutdown(wait=should_wait, cancel_futures=not should_wait)
        if not isinstance(result, ToolExecutionResult):
            raise TypeError(f"Tool {tool.name} returned invalid result type: {type(result).__name__}")
        tool.validate_output(result.output)
        return result

    def dispatch(self, name: str, raw_input: dict, config: AgentConfig, session_state: SessionState) -> tuple[ToolInvocation, ToolExecutionResult]:
        tool, context, invocation = self.prepare(name, raw_input, config, session_state)
        result = self.execute_prepared(tool, context, invocation)
        return invocation, result
