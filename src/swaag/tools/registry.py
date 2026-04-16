from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from collections import deque
from typing import Iterable

from swaag.config import AgentConfig
from swaag.environment.environment import AgentEnvironment
from swaag.tools.base import Tool, ToolContext
from swaag.tools.builtin import BUILTIN_TOOLS
from swaag.types import SessionState, ToolExecutionResult, ToolInvocation


@dataclass(slots=True)
class ToolGraphPlan:
    expected_tool: str
    selected_tool: str
    chain: list[str]
    valid: bool
    reason: str


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or BUILTIN_TOOLS:
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

    def prompt_tuples(self, config: AgentConfig) -> list[tuple[str, str, dict]]:
        return [tool.prompt_tuple() for tool in self.enabled_tools(config)]

    def tool_names(self, config: AgentConfig) -> list[str]:
        return [tool.name for tool in self.enabled_tools(config)]

    def capability_graph(self, config: AgentConfig) -> dict[str, list[str]]:
        tools = {tool.name: tool for tool in self.enabled_tools(config)}
        graph: dict[str, list[str]] = {}
        for name, tool in tools.items():
            followers: list[str] = []
            for candidate_name, candidate in tools.items():
                if candidate_name == name:
                    continue
                provides = set(tool.provides_artifacts)
                requires = set(candidate.requires_artifacts)
                if (requires and requires.issubset(provides)) or candidate_name in tool.allowed_followers:
                    followers.append(candidate_name)
            graph[name] = sorted(set(followers))
        return graph

    def can_chain(self, from_tool: str, to_tool: str, config: AgentConfig) -> bool:
        if to_tool == "respond":
            tool = self.get(from_tool)
            return "respond" in tool.allowed_followers
        return to_tool in self.capability_graph(config).get(from_tool, [])

    def shortest_chain(self, start_tool: str, target_tool: str, config: AgentConfig) -> list[str] | None:
        if start_tool == target_tool:
            return [start_tool]
        graph = self.capability_graph(config)
        queue: deque[tuple[str, list[str]]] = deque([(start_tool, [start_tool])])
        seen = {start_tool}
        while queue:
            current, chain = queue.popleft()
            for follower in graph.get(current, []):
                if follower in seen:
                    continue
                next_chain = [*chain, follower]
                if follower == target_tool:
                    return next_chain
                seen.add(follower)
                queue.append((follower, next_chain))
        return None

    def plan_tool_graph(
        self,
        *,
        selected_tool: str,
        expected_tool: str,
        config: AgentConfig,
    ) -> ToolGraphPlan:
        if selected_tool == expected_tool:
            return ToolGraphPlan(
                expected_tool=expected_tool,
                selected_tool=selected_tool,
                chain=[selected_tool],
                valid=True,
                reason="selected_tool_matches_expected_tool",
            )
        chain = self.shortest_chain(selected_tool, expected_tool, config)
        if chain is None:
            return ToolGraphPlan(
                expected_tool=expected_tool,
                selected_tool=selected_tool,
                chain=[selected_tool],
                valid=False,
                reason=f"no_capability_path:{selected_tool}->{expected_tool}",
            )
        return ToolGraphPlan(
            expected_tool=expected_tool,
            selected_tool=selected_tool,
            chain=chain,
            valid=True,
            reason=f"capability_path:{'->'.join(chain)}",
        )

    def validate_tool_chain(self, chain: list[str], config: AgentConfig) -> None:
        enabled = set(self.tool_names(config))
        for tool_name in chain:
            if tool_name not in enabled:
                raise ValueError(f"Unknown or disabled tool in chain: {tool_name}")
        for left, right in zip(chain, chain[1:]):
            if not self.can_chain(left, right, config):
                raise ValueError(f"Invalid tool chain edge: {left} -> {right}")

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
            try:
                result = future.result(timeout=context.config.runtime.tool_timeout_seconds)
            except FuturesTimeoutError as exc:
                future.cancel()
                should_wait = False
                raise TimeoutError(f"Tool timed out after {context.config.runtime.tool_timeout_seconds}s: {tool.name}") from exc
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
