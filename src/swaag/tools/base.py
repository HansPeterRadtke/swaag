from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

from swaag.config import AgentConfig
from swaag.types import SessionState, ToolExecutionResult, ToolGeneratedEvent, ToolKind, ToolInvocation

if TYPE_CHECKING:
    from swaag.environment.environment import AgentEnvironment


class ToolValidationError(ValueError):
    pass


@dataclass(slots=True)
class ToolContext:
    config: AgentConfig
    session_state: SessionState
    environment: "AgentEnvironment"

    @property
    def read_roots(self) -> list[Path]:
        return self.config.tools.read_roots


class Tool(abc.ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    kind: ToolKind = "pure"
    requires_artifacts: tuple[str, ...] = ()
    provides_artifacts: tuple[str, ...] = ()
    allowed_followers: tuple[str, ...] = ()

    def prompt_tuple(self) -> tuple[str, str, dict[str, Any]]:
        return self.name, self.description, self.input_schema

    def effective_kind(self, validated_input: dict[str, Any]) -> ToolKind:
        return self.kind

    def pre_execute_events(self, validated_input: dict[str, Any], context: ToolContext) -> list[ToolGeneratedEvent]:
        return []

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return set()

    def validate_output(self, output: dict[str, Any]) -> None:
        if self.output_schema is None:
            return
        _validate_schema_value(output, self.output_schema, path=f"{self.name}.output")

    @abc.abstractmethod
    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        raise NotImplementedError


def _validate_schema_value(value: Any, schema: dict[str, Any], *, path: str) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(f"{path} must be an object")
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for key in required:
            if key not in value:
                raise ToolValidationError(f"{path} is missing required key: {key}")
        additional_properties = schema.get("additionalProperties", True)
        if additional_properties is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ToolValidationError(f"{path} contains unknown keys: {', '.join(sorted(unknown))}")
        for key, child_value in value.items():
            child_schema = properties.get(key)
            if child_schema is None:
                continue
            _validate_schema_value(child_value, child_schema, path=f"{path}.{key}")
        return
    if expected_type == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"{path} must be an array")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, path=f"{path}[{index}]")
        return
    if expected_type == "string":
        if not isinstance(value, str):
            raise ToolValidationError(f"{path} must be a string")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolValidationError(f"{path} must be an integer")
    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ToolValidationError(f"{path} must be a number")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            raise ToolValidationError(f"{path} must be a boolean")
    elif expected_type is not None:
        raise ToolValidationError(f"Unsupported schema type at {path}: {expected_type}")

    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"{path} must be one of: {', '.join(map(str, schema['enum']))}")
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        raise ToolValidationError(f"{path} must be >= {schema['minimum']}")
