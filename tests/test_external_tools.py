from __future__ import annotations

from dataclasses import dataclass

from swaag.delegated_tools import DelegatedToolSpec, prepare_delegated_tool_spec
from swaag.external_tools import RuntimeExternalToolCallResult, RuntimeExternalToolManager
from swaag.runtime import AgentRuntime


@dataclass
class DummyRuntimeAdapter:
    adapter_id: str = "dummy"

    @property
    def discovery_errors(self) -> dict[str, str]:
        return {}

    def specs(self) -> tuple[DelegatedToolSpec, ...]:
        return (
            prepare_delegated_tool_spec(
                {
                    "name": "dummy_runtime_lookup",
                    "description": "Fetch opaque evidence from a dummy runtime provider.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "additionalProperties": False,
                    },
                    "metadata": {
                        "external_execution_mode": "runtime",
                        "external_provider_id": "dummy:provider",
                    },
                }
            ),
        )

    def has_tool(self, tool_name: str) -> bool:
        return tool_name == "dummy_runtime_lookup"

    def call(self, tool_name: str, arguments: dict) -> RuntimeExternalToolCallResult:
        assert tool_name == "dummy_runtime_lookup"
        value = f"dummy:{arguments['key']}"
        return RuntimeExternalToolCallResult(
            provider_id="dummy:provider",
            tool_name=tool_name,
            structured_content={"value": value},
            content=[{"type": "text", "text": value}],
            is_error=False,
            raw_result={"provider_payload": value},
        )


def test_core_runtime_executes_non_mcp_external_adapter_without_protocol_semantics(
    make_config,
) -> None:
    runtime = AgentRuntime(make_config(), model_client=object())
    runtime.runtime_external_tools = RuntimeExternalToolManager([DummyRuntimeAdapter()])
    state = runtime.create_or_load_session()
    spec = runtime.runtime_external_tools.specs()[0]

    assert spec.name not in runtime.tools.system_tool_names()
    index = runtime.tools.capability_index(runtime.config, (spec,))
    assert any(name == spec.name for name, _description, _guidance in index)
    staged = runtime.tools.staged_prompt_tuples(runtime.config, [spec.name], (spec,))
    guidance = next(item[3] for item in staged if item[0] == spec.name)
    assert "external tool" in guidance.lower()
    assert "provider adapter" in guidance.lower()
    assert "mcp" not in guidance.lower()

    result = runtime._execute_runtime_external_tool(
        state, spec=spec, arguments={"key": "alpha"}
    )
    assert result is not None
    assert result.output["structured_content"] == {"value": "dummy:alpha"}
    assert result.output["external_provider_id"] == "dummy:provider"

    events = runtime.history.read_history(state.session_id)
    called = next(event for event in events if event.event_type == "tool_called")
    completed = next(event for event in events if event.event_type == "tool_result")
    assert called.payload["executor"] == "external_runtime"
    assert called.payload["external_provider_id"] == "dummy:provider"
    assert completed.payload["executor"] == "external_runtime"
    assert completed.payload["external_provider_id"] == "dummy:provider"
    serialized = repr(called.payload).lower() + repr(completed.payload).lower()
    assert "mcp" not in serialized
