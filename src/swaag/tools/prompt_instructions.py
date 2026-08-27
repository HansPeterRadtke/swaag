from __future__ import annotations

from dataclasses import asdict
from typing import Any

from swaag.prompt_instructions import (
    PROMPT_INSTRUCTION_SCOPES,
    PromptInstructionError,
    enforce_prompt_instruction_limits,
    make_prompt_instruction,
)
from swaag.prompt_instruction_store import PromptInstructionStore
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import stable_json_dumps


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


class PromptInstructionsTool(Tool):
    name = "prompt_instructions"
    description = (
        "List, add, replace, or remove durable model-authored instructions scoped "
        "to explicit LLM call kinds, optional semantic step categories, and either "
        "this session or the local user."
    )
    usage_guidance = (
        "Use this for durable corrections or learned operating rules that must become "
        "system instructions on matching future model calls. Choose broad call-kind scopes "
        "and fine-grained free-form categories semantically. An empty categories array means "
        "the instruction applies to every call in its broad scope; categorized entries are "
        "selected by an LLM from the exact next-call context. "
        "Choose the user store only when the rule should reach independent future sessions, "
        "and the session store when it belongs only to the current task. "
        "Ordinary task facts and temporary plans belong in notes instead. Consolidate "
        "duplicates by replacing one complete entry and removing the obsolete entries rather "
        "than appending. Selected entries are included exactly and context-accounted; selector "
        "failure conservatively includes all broad-scope candidates."
    )
    kind = "stateful"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "add", "replace", "remove"],
            },
            "instruction_store": {
                "type": "string",
                "enum": ["session", "user"],
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
            "categories": _nullable(
                {
                    "type": "array",
                    "items": {"type": "string"},
                }
            ),
        },
        "required": [
            "action",
            "instruction_store",
            "instruction_id",
            "title",
            "content",
            "scopes",
            "categories",
        ],
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {
            "instruction_id": {"type": "string"},
            "instruction_store": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "scopes": {"type": "array", "items": {"type": "string"}},
            "categories": {"type": "array", "items": {"type": "string"}},
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
        instruction_store = raw_input.get("instruction_store")
        if instruction_store not in {"session", "user"}:
            raise ToolValidationError(
                "prompt_instructions.instruction_store must be session or user"
            )
        instruction_id = raw_input.get("instruction_id")
        title = raw_input.get("title")
        content = raw_input.get("content")
        scopes = raw_input.get("scopes")
        categories = raw_input.get("categories")
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
            if categories is None:
                categories = []
            if not isinstance(categories, list) or not all(
                isinstance(item, str) for item in categories
            ):
                raise ToolValidationError(
                    f"prompt_instructions {action} requires a categories array"
                )
        elif any(
            value is not None for value in (title, content, scopes, categories)
        ):
            raise ToolValidationError(
                f"prompt_instructions {action} does not accept title, content, scopes, or categories"
            )
        if action in {"list", "add"} and instruction_id is not None:
            raise ToolValidationError(
                f"prompt_instructions {action} does not accept instruction_id"
            )
        return {
            "action": action,
            "instruction_store": instruction_store,
            "instruction_id": (
                instruction_id.strip() if isinstance(instruction_id, str) else None
            ),
            "title": title,
            "content": content,
            "scopes": list(scopes) if isinstance(scopes, list) else None,
            "categories": (
                list(categories) if isinstance(categories, list) else None
            ),
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
        instruction_store = validated_input["instruction_store"]
        if instruction_store == "user":
            return self._execute_user_store(validated_input, context)
        if action == "list":
            output = {
                "instruction_store": "session",
                "instructions": [
                    asdict(item) | {"instruction_store": "session"}
                    for item in state.prompt_instructions
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
            output = {
                "instruction_id": existing.instruction_id,
                "instruction_store": "session",
                "removed": True,
            }
            return ToolExecutionResult(
                self.name,
                output,
                f"Removed prompt instruction {existing.instruction_id}",
                generated_events=[
                    ToolGeneratedEvent(
                        "prompt_instruction_removed",
                        {
                            "instruction_id": existing.instruction_id,
                            "instruction_store": "session",
                        },
                    )
                ],
            )

        try:
            instruction = make_prompt_instruction(
                context.config,
                title=validated_input["title"],
                content=validated_input["content"],
                scopes=validated_input["scopes"],
                categories=validated_input["categories"],
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
            "instruction_store": "session",
            "title": instruction.title,
            "content": instruction.content,
            "scopes": instruction.scopes,
            "categories": instruction.categories,
        }
        return ToolExecutionResult(
            self.name,
            output,
            f"prompt_instructions result: {stable_json_dumps(output, indent=2)}",
            generated_events=[
                ToolGeneratedEvent(
                    event_type,
                    {
                        "instruction": asdict(instruction),
                        "instruction_store": "session",
                    },
                )
            ],
        )

    def _execute_user_store(
        self,
        validated_input: dict[str, Any],
        context: ToolContext,
    ) -> ToolExecutionResult:
        store = PromptInstructionStore(context.config.sessions.root, context.config)
        action = validated_input["action"]
        if action == "list":
            output = {
                "instruction_store": "user",
                "instructions": [
                    asdict(item) | {"instruction_store": "user"}
                    for item in store.list()
                ],
            }
            return ToolExecutionResult(
                self.name,
                output,
                f"prompt_instructions result: {stable_json_dumps(output, indent=2)}",
            )

        try:
            if action == "add":
                mutation = store.add(
                    title=validated_input["title"],
                    content=validated_input["content"],
                    scopes=validated_input["scopes"],
                    categories=validated_input["categories"],
                    origin_session_id=context.session_state.session_id,
                )
            elif action == "replace":
                mutation = store.replace(
                    instruction_id=validated_input["instruction_id"],
                    title=validated_input["title"],
                    content=validated_input["content"],
                    scopes=validated_input["scopes"],
                    categories=validated_input["categories"],
                    origin_session_id=context.session_state.session_id,
                )
            else:
                mutation = store.remove(
                    instruction_id=validated_input["instruction_id"],
                    origin_session_id=context.session_state.session_id,
                )
        except PromptInstructionError as exc:
            raise ToolValidationError(
                f"prompt_instructions {action} failed without modifying instructions: {exc}"
            ) from exc

        event = mutation.event
        store_reference = {
            "sequence": event.sequence,
            "event_id": event.event_id,
            "event_hash": event.event_hash,
            "previous_hash": event.previous_hash,
        }
        if action == "remove":
            output = {
                "instruction_id": event.instruction_id,
                "instruction_store": "user",
                "removed": True,
            }
            payload = {
                "instruction_id": event.instruction_id,
                "instruction_store": "user",
                "store_event": store_reference,
            }
            event_type = "prompt_instruction_removed"
        else:
            instruction = mutation.instruction
            assert instruction is not None
            output = {
                "instruction_id": instruction.instruction_id,
                "instruction_store": "user",
                "title": instruction.title,
                "content": instruction.content,
                "scopes": instruction.scopes,
                "categories": instruction.categories,
            }
            payload = {
                "instruction": asdict(instruction),
                "instruction_store": "user",
                "store_event": store_reference,
            }
            event_type = (
                "prompt_instruction_replaced"
                if action == "replace"
                else "prompt_instruction_added"
            )
        return ToolExecutionResult(
            self.name,
            output,
            f"prompt_instructions result: {stable_json_dumps(output, indent=2)}",
            generated_events=[ToolGeneratedEvent(event_type, payload)],
        )


PROMPT_INSTRUCTION_TOOLS = [PromptInstructionsTool()]
