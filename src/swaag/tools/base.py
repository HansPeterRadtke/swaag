from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from typing import TYPE_CHECKING

from swaag.config import AgentConfig
from swaag.types import (
    BudgetReport,
    ContractSpec,
    ModelCallKind,
    PromptComponent,
    SessionState,
    ToolExecutionResult,
    ToolGeneratedEvent,
    ToolInvocation,
    ToolKind,
)

if TYPE_CHECKING:
    from swaag.delegated_tools import DelegatedToolSpec
    from swaag.environment.environment import AgentEnvironment


class ToolValidationError(ValueError):
    pass


class ToolExecutionError(RuntimeError):
    """Tool failure whose exact evidence must be committed before the error."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str | None = None,
        evidence: dict[str, Any] | None = None,
        generated_events: list[ToolGeneratedEvent] | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type or self.__class__.__name__
        self.evidence = dict(evidence or {})
        self.generated_events = list(generated_events or [])


class SemanticCallContextOverflow(RuntimeError):
    def __init__(self, report: BudgetReport | None):
        super().__init__(
            "The complete semantic-call input does not fit the resolved model context"
        )
        self.report = report


def semantic_sources_cannot_recover_overflow(
    error: SemanticCallContextOverflow,
    source_component_names: set[str],
    *,
    fixed_slack_tokens: int = 32,
) -> bool:
    """Return true when removing every named reducible source still cannot fit."""
    if error.report is None:
        return False
    matched = [
        component
        for component in error.report.breakdown
        if component.name in source_component_names
    ]
    if not matched:
        return False
    overflow = max(
        1,
        int(error.report.required_tokens) - int(error.report.context_limit),
    )
    recoverable = sum(max(0, int(component.tokens)) for component in matched)
    return recoverable <= overflow + max(0, int(fixed_slack_tokens))


@dataclass(slots=True, frozen=True)
class SemanticCallRequest:
    kind: ModelCallKind
    system_instruction: str
    components: list[PromptComponent]
    contract: ContractSpec
    minimum_output_tokens: int
    desired_output_tokens: int | None = None
    prompt_mode: str = "lean"
    allow_prompt_instruction_projection: bool = False
    include_prompt_instructions: bool = True
    prompt_template_names: tuple[str, ...] = ()


@dataclass(slots=True)
class ToolContext:
    config: AgentConfig
    session_state: SessionState
    environment: "AgentEnvironment"
    semantic_call: Callable[[SemanticCallRequest], dict[str, Any]] | None = None
    delegated_tools: tuple["DelegatedToolSpec", ...] = ()

    @property
    def read_roots(self) -> list[Path]:
        return self.config.tools.read_roots

    def call_semantic(self, request: SemanticCallRequest) -> dict[str, Any]:
        if self.semantic_call is None:
            raise RuntimeError(
                "This model-backed capability requires the AgentRuntime semantic-call service"
            )
        return self.semantic_call(request)


class Tool(abc.ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    usage_guidance: str = ""
    kind: ToolKind = "pure"
    repeated_observation_is_redundant: bool = False

    def prompt_tuple(self) -> tuple[str, str, dict[str, Any], str]:
        return self.name, self.description, self.input_schema, self.usage_guidance

    def available(self, config: AgentConfig) -> bool:
        return True

    def effective_kind(self, validated_input: dict[str, Any]) -> ToolKind:
        return self.kind

    def execution_timeout_seconds(self, context: ToolContext) -> float | None:
        return float(context.config.runtime.tool_timeout_seconds)

    def pre_execute_events(self, validated_input: dict[str, Any], context: ToolContext) -> list[ToolGeneratedEvent]:
        return []

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return set()

    def validate_output(self, output: dict[str, Any]) -> None:
        if self.output_schema is None:
            return
        _validate_schema_value(output, self.output_schema, path=f"{self.name}.output")

    def verify_effect(
        self,
        result: ToolExecutionResult,
        environment: "AgentEnvironment",
    ) -> tuple[bool, dict[str, Any]] | None:
        """Verify a persisted side effect after its derived writes are committed."""
        return None

    @abc.abstractmethod
    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        raise NotImplementedError


def _validate_schema_value(value: Any, schema: dict[str, Any], *, path: str) -> None:
    any_of = schema.get("anyOf")
    if any_of is not None:
        if not isinstance(any_of, list) or not any_of:
            raise ToolValidationError(f"{path}.anyOf must be a non-empty list")
        errors: list[str] = []
        for variant in any_of:
            if not isinstance(variant, dict):
                errors.append(f"{path}.anyOf variant must be a schema object")
                continue
            try:
                _validate_schema_value(value, variant, path=path)
            except ToolValidationError as exc:
                errors.append(str(exc))
                continue
            return
        raise ToolValidationError(
            f"{path} must match at least one anyOf schema: {'; '.join(errors)}"
        )
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        errors: list[str] = []
        for candidate_type in expected_type:
            candidate_schema = dict(schema)
            candidate_schema["type"] = candidate_type
            try:
                _validate_schema_value(value, candidate_schema, path=path)
            except ToolValidationError as exc:
                errors.append(str(exc))
                continue
            return
        allowed = ", ".join(str(item) for item in expected_type)
        raise ToolValidationError(f"{path} must match one of the allowed schema types: {allowed}")
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
    elif expected_type == "null":
        if value is not None:
            raise ToolValidationError(f"{path} must be null")
    elif expected_type is not None:
        raise ToolValidationError(f"Unsupported schema type at {path}: {expected_type}")

    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"{path} must be one of: {', '.join(map(str, schema['enum']))}")
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        raise ToolValidationError(f"{path} must be >= {schema['minimum']}")
