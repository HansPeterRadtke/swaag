from __future__ import annotations

import json
import re
from typing import Any, NoReturn, cast

from swaag.shared_state import (
    SharedStateChannel,
    SharedStateConflictError,
    shared_state_event_payload,
)
from swaag.tools.base import Tool, ToolContext, ToolExecutionError, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent, ToolKind
from swaag.utils import stable_json_dumps


def _closed_object(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_VALUE_OPERATION = _closed_object(
    {
        "op": {"type": "string", "enum": ["add", "replace", "test"]},
        "path": {"type": "string"},
        "value_json": {"type": "string"},
    }
)
_REMOVE_OPERATION = _closed_object(
    {
        "op": {"type": "string", "enum": ["remove"]},
        "path": {"type": "string"},
    }
)
_FROM_OPERATION = _closed_object(
    {
        "op": {"type": "string", "enum": ["move", "copy"]},
        "path": {"type": "string"},
        "from": {"type": "string"},
    }
)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=_reject_json_constant,
    )


class SharedStateTool(Tool):
    name = "shared_state"
    description = (
        "Read or atomically update the exact structured state shared with the "
        "connected user interface using revision-guarded RFC 6902 operations."
    )
    usage_guidance = (
        "Use read whenever the current revision is uncertain. For patch, copy the "
        "exact current revision and SHA-256, then provide only semantically intended "
        "RFC 6902 changes. Encode add/replace/test values as strict JSON in value_json. "
        "A stale base fails with the complete current state so you can reconsider; do "
        "not blindly replay a patch after user changes."
    )
    kind = "stateful"
    required_runtime_capability = "shared_state"
    input_schema = _closed_object(
        {
            "operation": {"type": "string", "enum": ["read", "patch"]},
            "base_revision": {
                "anyOf": [{"type": "integer"}, {"type": "null"}]
            },
            "base_state_sha256": {
                "anyOf": [{"type": "string"}, {"type": "null"}]
            },
            "patch": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                _VALUE_OPERATION,
                                _REMOVE_OPERATION,
                                _FROM_OPERATION,
                            ]
                        },
                    },
                    {"type": "null"},
                ]
            },
        }
    )

    def execution_timeout_seconds(self, context: ToolContext) -> float | None:
        return None

    def effective_kind(self, validated_input: dict[str, Any]) -> ToolKind:
        return "pure" if validated_input["operation"] == "read" else "stateful"

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw_input, dict):
            raise ToolValidationError("shared_state input must be an object")
        allowed = {
            "operation",
            "base_revision",
            "base_state_sha256",
            "patch",
        }
        unknown = set(raw_input) - allowed
        if unknown:
            raise ToolValidationError(
                "shared_state received unknown arguments: "
                + ", ".join(sorted(unknown))
            )
        operation = raw_input.get("operation")
        if operation not in {"read", "patch"}:
            raise ToolValidationError("shared_state.operation must be read or patch")
        base_revision = raw_input.get("base_revision")
        base_sha256 = raw_input.get("base_state_sha256")
        raw_patch = raw_input.get("patch")
        if operation == "read":
            if any(value is not None for value in (base_revision, base_sha256, raw_patch)):
                raise ToolValidationError(
                    "shared_state.read requires null base_revision, "
                    "base_state_sha256, and patch"
                )
            return {
                "operation": "read",
                "base_revision": None,
                "base_state_sha256": None,
                "patch": None,
            }

        if (
            not isinstance(base_revision, int)
            or isinstance(base_revision, bool)
            or base_revision < 1
        ):
            raise ToolValidationError(
                "shared_state.patch base_revision must be a positive integer"
            )
        if not isinstance(base_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", base_sha256
        ) is None:
            raise ToolValidationError(
                "shared_state.patch base_state_sha256 must be a lowercase SHA-256"
            )
        if not isinstance(raw_patch, list) or not raw_patch:
            raise ToolValidationError(
                "shared_state.patch patch must be a non-empty operation array"
            )

        patch: list[dict[str, Any]] = []
        for index, raw_operation in enumerate(raw_patch):
            path = f"shared_state.patch[{index}]"
            if not isinstance(raw_operation, dict):
                raise ToolValidationError(f"{path} must be an object")
            patch_operation = raw_operation.get("op")
            pointer = raw_operation.get("path")
            if not isinstance(pointer, str):
                raise ToolValidationError(f"{path}.path must be a string")
            if patch_operation in {"add", "replace", "test"}:
                if set(raw_operation) != {"op", "path", "value_json"}:
                    raise ToolValidationError(
                        f"{path} requires exactly op, path, and value_json"
                    )
                value_json = raw_operation.get("value_json")
                if not isinstance(value_json, str):
                    raise ToolValidationError(f"{path}.value_json must be a string")
                try:
                    value = _strict_json_loads(value_json)
                except (TypeError, ValueError) as exc:
                    raise ToolValidationError(
                        f"{path}.value_json is not strict JSON: {exc}"
                    ) from exc
                patch.append(
                    {"op": patch_operation, "path": pointer, "value": value}
                )
            elif patch_operation == "remove":
                if set(raw_operation) != {"op", "path"}:
                    raise ToolValidationError(f"{path} requires exactly op and path")
                patch.append({"op": "remove", "path": pointer})
            elif patch_operation in {"move", "copy"}:
                if set(raw_operation) != {"op", "path", "from"}:
                    raise ToolValidationError(
                        f"{path} requires exactly op, path, and from"
                    )
                source = raw_operation.get("from")
                if not isinstance(source, str):
                    raise ToolValidationError(f"{path}.from must be a string")
                patch.append(
                    {"op": patch_operation, "path": pointer, "from": source}
                )
            else:
                raise ToolValidationError(f"{path}.op is not an RFC 6902 operation")
        return {
            "operation": "patch",
            "base_revision": base_revision,
            "base_state_sha256": base_sha256,
            "patch": patch,
        }

    @staticmethod
    def _channel(context: ToolContext) -> SharedStateChannel:
        channel = context.runtime_capabilities.get("shared_state")
        if channel is None:
            raise RuntimeError("No shared-state channel is bound to this session")
        return cast(SharedStateChannel, channel)

    def execute(
        self, validated_input: dict[str, Any], context: ToolContext
    ) -> ToolExecutionResult:
        channel = self._channel(context)
        if validated_input["operation"] == "read":
            snapshot = channel.snapshot()
            output = snapshot.tool_payload()
            return ToolExecutionResult(
                tool_name=self.name,
                output=output,
                display_text="shared_state: " + stable_json_dumps(output, indent=2),
            )

        if not context.tool_call_id:
            raise RuntimeError("shared-state mutation requires a durable tool call id")
        try:
            snapshot = channel.apply_patch(
                source_call_id=context.tool_call_id,
                base_revision=validated_input["base_revision"],
                base_state_sha256=validated_input["base_state_sha256"],
                patch=validated_input["patch"],
            )
        except SharedStateConflictError as exc:
            raise ToolExecutionError(
                str(exc),
                error_type="SharedStateConflictError",
                evidence={"current": exc.current.tool_payload()},
            ) from exc
        output = {
            **snapshot.tool_payload(),
            "base_revision": snapshot.base_revision,
            "base_state_sha256": snapshot.base_state_sha256,
            "applied_patch": list(snapshot.patch or ()),
            "patch_sha256": snapshot.patch_sha256,
        }
        generated = ToolGeneratedEvent(
            "shared_state_updated",
            shared_state_event_payload(snapshot),
        )
        return ToolExecutionResult(
            tool_name=self.name,
            output=output,
            display_text="shared_state: " + stable_json_dumps(output, indent=2),
            generated_events=[generated],
        )

    def required_generated_event_types(
        self, validated_input: dict[str, Any]
    ) -> set[str]:
        return (
            {"shared_state_updated"}
            if validated_input["operation"] == "patch"
            else set()
        )

    def generated_event_recorded(
        self,
        generated: ToolGeneratedEvent,
        recorded: Any,
        context: ToolContext,
    ) -> None:
        if generated.event_type != "shared_state_updated":
            return
        source_call_id = generated.payload.get("source_call_id")
        if not isinstance(source_call_id, str) or not source_call_id:
            raise RuntimeError("shared-state update event lost its source call id")
        self._channel(context).link_history(
            source_call_id=source_call_id,
            sequence=recorded.sequence,
            event_hash=recorded.hash,
        )


SHARED_STATE_TOOLS = [SharedStateTool()]
