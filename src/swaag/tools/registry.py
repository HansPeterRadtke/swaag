from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Iterable

from swaag.config import AgentConfig
from swaag.environment.environment import AgentEnvironment
from swaag.tools.base import Tool, ToolContext
from swaag.tools.artifacts import ARTIFACT_TOOLS
from swaag.tools.builtin import BUILTIN_TOOLS
from swaag.tools.history import HISTORY_TOOLS
from swaag.tools.terminal import TERMINAL_TOOLS
from swaag.tools.control import CONTROL_TOOLS
from swaag.types import SessionState, ToolExecutionResult, ToolInvocation


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or [*BUILTIN_TOOLS, *HISTORY_TOOLS, *ARTIFACT_TOOLS, *TERMINAL_TOOLS, *CONTROL_TOOLS]:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool registration: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def enabled_tools(self, config: AgentConfig) -> list[Tool]:
        tools = [self.get(name) for name in config.tools.enabled]
        result: list[Tool] = []
        for tool in tools:
            if tool.kind == "stateful" and not config.tools.allow_stateful_tools:
                continue
            if tool.kind == "side_effect" and not config.tools.allow_side_effect_tools:
                continue
            result.append(tool)
        return result

    def prompt_tuples(self, config: AgentConfig) -> list[tuple[str, str, dict, str]]:
        return [tool.prompt_tuple() for tool in self.enabled_tools(config)]

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
