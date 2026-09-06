from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from swaag.delegated_tools import DelegatedToolSpec


class RuntimeExternalToolError(RuntimeError):
    def __init__(self, message: str, *, evidence: dict[str, Any] | None = None):
        super().__init__(message)
        self.evidence = dict(evidence or {})


@dataclass(slots=True, frozen=True)
class RuntimeExternalToolCallResult:
    provider_id: str
    tool_name: str
    structured_content: dict[str, Any]
    content: list[dict[str, Any]]
    is_error: bool
    raw_result: dict[str, Any]


class RuntimeExternalToolAdapter(Protocol):
    adapter_id: str

    @property
    def discovery_errors(self) -> dict[str, str]: ...

    def specs(self) -> tuple[DelegatedToolSpec, ...]: ...

    def has_tool(self, tool_name: str) -> bool: ...

    def call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> RuntimeExternalToolCallResult: ...


class RuntimeExternalToolManager:
    """Provider-neutral catalog/executor for layer-three tools run by SWAAG."""

    def __init__(self, adapters: Iterable[RuntimeExternalToolAdapter] = ()):
        self._adapters = tuple(adapters)
        self._adapter_by_tool: dict[str, RuntimeExternalToolAdapter] = {}
        self.refresh()

    @property
    def discovery_errors(self) -> dict[str, str]:
        errors: dict[str, str] = {}
        for adapter in self._adapters:
            for key, value in adapter.discovery_errors.items():
                errors[f"{adapter.adapter_id}:{key}"] = value
        return errors

    def refresh(self) -> tuple[DelegatedToolSpec, ...]:
        routed: dict[str, RuntimeExternalToolAdapter] = {}
        for adapter in self._adapters:
            for spec in adapter.specs():
                if spec.name in routed:
                    raise RuntimeExternalToolError(
                        "Runtime external tool collision: "
                        f"{spec.name} from {routed[spec.name].adapter_id} and {adapter.adapter_id}"
                    )
                routed[spec.name] = adapter
        self._adapter_by_tool = routed
        return self.specs()

    def specs(self) -> tuple[DelegatedToolSpec, ...]:
        specs: list[DelegatedToolSpec] = []
        for adapter in self._adapters:
            specs.extend(adapter.specs())
        return tuple(specs)

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._adapter_by_tool

    def call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> RuntimeExternalToolCallResult:
        try:
            adapter = self._adapter_by_tool[tool_name]
        except KeyError as exc:
            raise KeyError(f"Unknown runtime external tool: {tool_name}") from exc
        return adapter.call(tool_name, arguments)
