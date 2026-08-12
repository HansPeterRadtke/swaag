from __future__ import annotations

from pathlib import Path
from typing import Any

from swaag.environment.terminal import TerminalStore
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent, ToolKind
from swaag.utils import stable_json_dumps


class TerminalTool(Tool):
    name = "terminal"
    description = "Manage persistent interactive PTY terminals. Create a shell, send commands or stdin, read incremental terminal output, list terminals, or close one."
    usage_guidance = "Use shell_command for ordinary non-interactive work. Use terminal when a command needs persistent shell state or interactive stdin. After create, copy the returned terminal_id exactly into terminal_ref for send/read/close; do not invent aliases. Keep reads bounded and advance start_offset with next_offset."
    kind = "stateful"
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["create", "send", "read", "list", "close"]},
            "terminal_ref": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "data": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "append_newline": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            "start_offset": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "max_chars": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
        "required": ["operation", "terminal_ref", "name", "data", "append_newline", "start_offset", "max_chars"],
        "additionalProperties": False,
    }

    def effective_kind(self, validated_input: dict[str, Any]) -> ToolKind:
        return "pure" if validated_input["operation"] in {"read", "list"} else "side_effect"

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        operation = raw_input.get("operation")
        if operation not in {"create", "send", "read", "list", "close"}:
            raise ToolValidationError("terminal.operation must be create, send, read, list, or close")
        terminal_ref = raw_input.get("terminal_ref")
        name = raw_input.get("name")
        data = raw_input.get("data")
        append_newline = raw_input.get("append_newline")
        start_offset = raw_input.get("start_offset")
        terminal_ref = "" if terminal_ref is None else terminal_ref
        name = "" if name is None else name
        data = "" if data is None else data
        append_newline = False if append_newline is None else append_newline
        start_offset = 0 if start_offset is None else start_offset
        max_chars = raw_input.get("max_chars")
        if not isinstance(terminal_ref, str) or not isinstance(name, str) or not isinstance(data, str):
            raise ToolValidationError("terminal_ref, name, and data must be strings")
        if not isinstance(append_newline, bool):
            raise ToolValidationError("terminal.append_newline must be boolean")
        if not isinstance(start_offset, int) or isinstance(start_offset, bool) or start_offset < 0:
            raise ToolValidationError("terminal.start_offset must be a non-negative integer")
        if max_chars is not None and (not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0):
            raise ToolValidationError("terminal.max_chars must be a positive integer")
        if operation in {"send", "read", "close"} and not terminal_ref.strip():
            raise ToolValidationError(f"terminal.{operation} requires terminal_ref")
        if operation == "send" and not data:
            raise ToolValidationError("terminal.send requires non-empty data")
        return {
            "operation": operation,
            "terminal_ref": terminal_ref.strip(),
            "name": name.strip(),
            "data": data,
            "append_newline": append_newline,
            "start_offset": start_offset,
            "max_chars": max_chars,
        }

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {f"terminal_{validated_input['operation']}"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        store = TerminalStore(context.config.sessions.root, context.session_state.session_id)
        operation = validated_input["operation"]
        if operation == "create":
            record = store.create(
                cwd=Path(context.environment.current_cwd),
                shell=context.config.environment.shell_executable,
                name=validated_input["name"],
            )
            output = _record_dict(record)
        elif operation == "send":
            record = store.send(validated_input["terminal_ref"], validated_input["data"], append_newline=validated_input["append_newline"])
            output = _record_dict(record) | {"sent_chars": len(validated_input["data"]) + (1 if validated_input["append_newline"] else 0)}
        elif operation == "read":
            max_chars = validated_input["max_chars"] or context.config.reader.default_chunk_chars
            max_chars = min(int(max_chars), int(context.config.reader.max_chunk_chars))
            output = store.read(validated_input["terminal_ref"], start_offset=validated_input["start_offset"], max_chars=max_chars)
        elif operation == "list":
            output = {"terminals": [_record_dict(item) for item in store.list()]}
        else:
            output = _record_dict(store.close(validated_input["terminal_ref"]))
        event = ToolGeneratedEvent(
            f"terminal_{operation}",
            {"operation": operation, "terminal_id": output.get("terminal_id", ""), "terminal_ref": validated_input["terminal_ref"], "name": output.get("name", validated_input["name"]), "active": output.get("active")},
        )
        return ToolExecutionResult(
            tool_name=self.name,
            output=output,
            display_text=f"terminal result: {stable_json_dumps(output, indent=2)}",
            generated_events=[event],
        )


def _record_dict(record) -> dict[str, Any]:
    return {
        "terminal_id": record.terminal_id,
        "name": record.name,
        "cwd": record.cwd,
        "shell": record.shell,
        "worker_pid": record.worker_pid,
        "shell_pid": record.shell_pid,
        "active": record.active,
        "return_code": record.return_code,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "output_chars": record.output_chars,
    }


TERMINAL_TOOLS = [TerminalTool()]
