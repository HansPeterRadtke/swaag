from __future__ import annotations

from dataclasses import asdict
from typing import Any

from swaag.prompt_instructions import (
    PROMPT_INSTRUCTION_SCOPES,
    PromptInstructionError,
    enforce_prompt_instruction_limits,
    make_prompt_instruction,
)
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import stable_json_dumps


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


class PromptInstructionsTool(Tool):
    name = "prompt_instructions"
    description = (
        "List, add, replace, or remove durable model-authored instructions scoped "
        "to explicit LLM call kinds."
    )
    usage_guidance = (
        "Use this for durable corrections or learned operating rules that must become "
        "system instructions on matching future model calls. Choose scopes semantically; "
        "ordinary task facts and temporary plans belong in notes instead. Replace or remove "
        "obsolete/redundant entries rather than appending duplicates. Every matching entry is "
        "included exactly and context-accounted, so storage overflow fails closed."
    )
    kind = "stateful"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "add", "replace", "remove"],
            },
            "instruction_id": _nullable({"type": "string"}),
            "title": _nullable({"type": "string"}),
            "content": _nullable({"type": "string"}),
            "scopes": _nullable(
                {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(PROMPT_INSTRUCTION_SCOPES),
                    },
                }
            ),
        },
        "required": ["action", "instruction_id", "title", "content", "scopes"],
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {
            "instruction_id": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "scopes": {"type": "array", "items": {"type": "string"}},
            "instructions": {"type": "array", "items": {"type": "object"}},
            "removed": {"type": "boolean"},
        },
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        action = raw_input.get("action")
        if action not in {"list", "add", "replace", "remove"}:
            raise ToolValidationError(
                "prompt_instructions.action must be one of list, add, replace, remove"
            )
        instruction_id = raw_input.get("instruction_id")
        title = raw_input.get("title")
        content = raw_input.get("content")
        scopes = raw_input.get("scopes")
        if action in {"replace", "remove"} and (
            not isinstance(instruction_id, str) or not instruction_id.strip()
        ):
            raise ToolValidationError(
                f"prompt_instructions {action} requires instruction_id"
            )
        if action in {"add", "replace"}:
            if not isinstance(title, str) or not isinstance(content, str):
                raise ToolValidationError(
                    f"prompt_instructions {action} requires title and content"
                )
            if not isinstance(scopes, list) or not all(
                isinstance(item, str) for item in scopes
            ):
                raise ToolValidationError(
                    f"prompt_instructions {action} requires a scopes array"
                )
        elif any(value is not None for value in (title, content, scopes)):
            raise ToolValidationError(
                f"prompt_instructions {action} does not accept title, content, or scopes"
            )
        if action in {"list", "add"} and instruction_id is not None:
            raise ToolValidationError(
                f"prompt_instructions {action} does not accept instruction_id"
            )
        return {
            "action": action,
            "instruction_id": (
                instruction_id.strip() if isinstance(instruction_id, str) else None
            ),
            "title": title,
            "content": content,
            "scopes": list(scopes) if isinstance(scopes, list) else None,
        }

    def required_generated_event_types(
        self, validated_input: dict[str, Any]
    ) -> set[str]:
        return {
            "add": {"prompt_instruction_added"},
            "replace": {"prompt_instruction_replaced"},
            "remove": {"prompt_instruction_removed"},
        }.get(str(validated_input["action"]), set())

    def execute(
        self, validated_input: dict[str, Any], context: ToolContext
    ) -> ToolExecutionResult:
        state = context.session_state
        action = validated_input["action"]
        if action == "list":
            output = {
                "instructions": [
                    asdict(item) for item in state.prompt_instructions
                ]
            }
            return ToolExecutionResult(
                self.name,
                output,
                f"prompt_instructions result: {stable_json_dumps(output, indent=2)}",
            )

        existing = next(
            (
                item
                for item in state.prompt_instructions
                if item.instruction_id == validated_input["instruction_id"]
            ),
            None,
        )
        if action in {"replace", "remove"} and existing is None:
            raise ToolValidationError(
                f"Unknown prompt instruction: {validated_input['instruction_id']}"
            )
        if action == "remove":
            assert existing is not None
            output = {"instruction_id": existing.instruction_id, "removed": True}
            return ToolExecutionResult(
                self.name,
                output,
                f"Removed prompt instruction {existing.instruction_id}",
                generated_events=[
                    ToolGeneratedEvent(
                        "prompt_instruction_removed",
                        {"instruction_id": existing.instruction_id},
                    )
                ],
            )

        try:
            instruction = make_prompt_instruction(
                context.config,
                title=validated_input["title"],
                content=validated_input["content"],
                scopes=validated_input["scopes"],
                instruction_id=(
                    existing.instruction_id if existing is not None else None
                ),
            )
            if existing is not None:
                instruction.created_at = existing.created_at
            candidate = [
                instruction if item.instruction_id == instruction.instruction_id else item
                for item in state.prompt_instructions
            ]
            if existing is None:
                candidate.append(instruction)
            enforce_prompt_instruction_limits(context.config, candidate)
        except PromptInstructionError as exc:
            raise ToolValidationError(
                f"prompt_instructions {action} failed without modifying instructions: {exc}"
            ) from exc
        event_type = (
            "prompt_instruction_replaced"
            if existing is not None
            else "prompt_instruction_added"
        )
        output = {
            "instruction_id": instruction.instruction_id,
            "title": instruction.title,
            "content": instruction.content,
            "scopes": instruction.scopes,
        }
        return ToolExecutionResult(
            self.name,
            output,
            f"prompt_instructions result: {stable_json_dumps(output, indent=2)}",
            generated_events=[
                ToolGeneratedEvent(event_type, {"instruction": asdict(instruction)})
            ],
        )


PROMPT_INSTRUCTION_TOOLS = [PromptInstructionsTool()]
