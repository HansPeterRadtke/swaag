from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


class ActionValidationError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class AgentToolCall:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class AgentAction:
    assistant_message: str
    tool_calls: list[AgentToolCall]
    continue_loop: bool

    @property
    def calls_tools(self) -> bool:
        return bool(self.tool_calls)


def action_from_payload(payload: dict[str, Any], *, enabled_tool_names: Iterable[str]) -> AgentAction:
    if not isinstance(payload, dict):
        raise ActionValidationError("Agent action payload must be an object")

    assistant_message = payload.get("assistant_message")
    tool_calls_payload = payload.get("tool_calls")
    continue_loop = payload.get("continue_loop")

    if not isinstance(assistant_message, str):
        raise ActionValidationError("assistant_message must be a string")
    if not isinstance(tool_calls_payload, list):
        raise ActionValidationError("tool_calls must be an array")
    if not isinstance(continue_loop, bool):
        raise ActionValidationError("continue_loop must be a boolean")

    enabled = set(enabled_tool_names)
    tool_calls: list[AgentToolCall] = []
    for index, item in enumerate(tool_calls_payload):
        if not isinstance(item, dict):
            raise ActionValidationError(f"tool_calls[{index}] must be an object")
        tool_name = item.get("tool_name")
        arguments = item.get("arguments")
        if not isinstance(tool_name, str) or tool_name not in enabled:
            raise ActionValidationError(f"tool_calls[{index}].tool_name is not enabled: {tool_name!r}")
        if not isinstance(arguments, dict):
            raise ActionValidationError(f"tool_calls[{index}].arguments must be an object")
        tool_calls.append(AgentToolCall(tool_name=tool_name, arguments=arguments))

    if tool_calls and not continue_loop:
        raise ActionValidationError(
            "continue_loop must be true when tool_calls are present so their exact results are observed"
        )
    if continue_loop and not tool_calls:
        raise ActionValidationError(
            "continue_loop=true requires at least one tool call; use a wait tool when waiting is required"
        )
    if not continue_loop and not tool_calls and not assistant_message.strip():
        raise ActionValidationError(
            "A completed turn without tool calls requires a non-empty assistant_message"
        )

    return AgentAction(
        assistant_message=assistant_message,
        tool_calls=tool_calls,
        continue_loop=continue_loop,
    )
