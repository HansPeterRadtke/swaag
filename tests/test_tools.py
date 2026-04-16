from __future__ import annotations

import time
from pathlib import Path

import pytest

from swaag.notes import make_note
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.tools.registry import ToolRegistry
from swaag.types import SessionState, ToolExecutionResult



def _empty_state() -> SessionState:
    return SessionState(session_id="s", created_at="t", updated_at="t", config_fingerprint="cfg", model_base_url="http://example.test")



def test_calculator_tool_executes(make_config) -> None:
    registry = ToolRegistry()
    config = make_config()
    invocation, result = registry.dispatch("calculator", {"expression": "2 + 3 * 4"}, config, _empty_state())
    assert invocation.validated_input == {"expression": "2 + 3 * 4"}
    assert result.output["result"] == 14



def test_unknown_tool_raises(make_config) -> None:
    with pytest.raises(KeyError):
        ToolRegistry().dispatch("missing", {}, make_config(), _empty_state())



def test_side_effect_tool_blocked_by_policy(make_config, tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_text("hello", encoding="utf-8")
    registry = ToolRegistry()
    config = make_config(tools__allow_side_effect_tools=False)
    with pytest.raises(PermissionError):
        registry.dispatch("edit_text", {"path": str(path), "operation": "replace_pattern_all", "pattern": "hello", "replacement": "world"}, config, _empty_state())



def test_notes_tool_add_returns_generated_events(make_config) -> None:
    registry = ToolRegistry()
    config = make_config()
    _, result = registry.dispatch("notes", {"action": "add", "title": "Todo", "content": "Check file"}, config, _empty_state())
    assert result.generated_events
    assert result.generated_events[0].event_type == "note_added"



def test_read_text_tool_reads_chunk(make_config, tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    registry = ToolRegistry()
    config = make_config(reader__default_chunk_chars=4, reader__max_chunk_chars=10)
    _, result = registry.dispatch("read_text", {"path": str(path)}, config, _empty_state())
    assert result.output["text"] == "abcd"
    assert any(event.event_type == "file_chunk_read" for event in result.generated_events)



def test_edit_tool_dry_run_preview(make_config, tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    registry = ToolRegistry()
    config = make_config(tools__allow_side_effect_tools=False)
    _, result = registry.dispatch(
        "edit_text",
        {"path": str(path), "operation": "replace_pattern_all", "pattern": "hello", "replacement": "world", "dry_run": True},
        config,
        _empty_state(),
    )
    assert any(event.event_type == "edit_previewed" for event in result.generated_events)



def test_edit_tool_write_blocked_by_editor_policy(make_config, tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    registry = ToolRegistry()
    config = make_config(tools__allow_side_effect_tools=True, editor__allow_writes=False)
    with pytest.raises(PermissionError):
        registry.dispatch(
            "edit_text",
            {"path": str(path), "operation": "replace_pattern_all", "pattern": "hello", "replacement": "world"},
            config,
            _empty_state(),
        )



def test_malformed_arguments_raise_validation_error(make_config) -> None:
    with pytest.raises(ToolValidationError):
        ToolRegistry().dispatch("calculator", {"expression": "__import__('os')"}, make_config(), _empty_state())


class MutatingTool(Tool):
    name = "mutator"
    description = "Mutate the provided session snapshot."
    kind = "stateful"
    input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    def validate(self, raw_input):
        return {}

    def execute(self, validated_input, context: ToolContext) -> ToolExecutionResult:
        context.session_state.notes.clear()
        return ToolExecutionResult(tool_name=self.name, output={"ok": True}, display_text="ok")



def test_tool_execution_context_is_isolated_copy(make_config) -> None:
    registry = ToolRegistry(tools=[MutatingTool()])
    config = make_config(tools__enabled=["mutator"], tools__allow_stateful_tools=True)
    state = _empty_state()
    state.notes.append(make_note(config, title="Keep", content="original"))

    _, result = registry.dispatch("mutator", {}, config, state)

    assert result.output == {"ok": True}
    assert len(state.notes) == 1


class InvalidOutputTool(Tool):
    name = "bad_output"
    description = "Return invalid output."
    kind = "pure"
    input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    output_schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}

    def validate(self, raw_input):
        return {}

    def execute(self, validated_input, context: ToolContext) -> ToolExecutionResult:
        return ToolExecutionResult(tool_name=self.name, output={"ok": "not-bool"}, display_text="bad")


class SlowTool(Tool):
    name = "slow"
    description = "Sleep longer than the timeout."
    kind = "pure"
    input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    output_schema = {"type": "object", "properties": {"done": {"type": "boolean"}}, "required": ["done"], "additionalProperties": False}

    def validate(self, raw_input):
        return {}

    def execute(self, validated_input, context: ToolContext) -> ToolExecutionResult:
        time.sleep(0.2)
        return ToolExecutionResult(tool_name=self.name, output={"done": True}, display_text="done")


def test_tool_invalid_output_is_rejected(make_config) -> None:
    registry = ToolRegistry(tools=[InvalidOutputTool()])
    config = make_config(tools__enabled=["bad_output"])
    with pytest.raises(ToolValidationError):
        registry.dispatch("bad_output", {}, config, _empty_state())


def test_tool_timeout_is_enforced(make_config) -> None:
    registry = ToolRegistry(tools=[SlowTool()])
    config = make_config(tools__enabled=["slow"], runtime__tool_timeout_seconds=1)
    config.runtime.tool_timeout_seconds = 0.05
    with pytest.raises(TimeoutError):
        registry.dispatch("slow", {}, config, _empty_state())


def test_tool_registry_exposes_capability_graph(make_config) -> None:
    registry = ToolRegistry()
    config = make_config(tools__allow_side_effect_tools=True)

    graph = registry.capability_graph(config)

    assert "edit_text" in graph["read_text"]
    assert "calculator" in graph["notes"]


def test_invalid_tool_chain_is_rejected(make_config) -> None:
    registry = ToolRegistry()
    config = make_config()

    with pytest.raises(ValueError):
        registry.validate_tool_chain(["calculator", "read_text"], config)


def test_tool_graph_planner_returns_shortest_valid_chain(make_config) -> None:
    registry = ToolRegistry()
    config = make_config(tools__allow_side_effect_tools=True)

    plan = registry.plan_tool_graph(selected_tool="read_text", expected_tool="edit_text", config=config)

    assert plan.valid is True
    assert plan.chain == ["read_text", "edit_text"]


def test_tool_graph_planner_rejects_unreachable_chain(make_config) -> None:
    registry = ToolRegistry()
    config = make_config()

    plan = registry.plan_tool_graph(selected_tool="calculator", expected_tool="read_text", config=config)

    assert plan.valid is False
    assert plan.reason.startswith("no_capability_path:")
