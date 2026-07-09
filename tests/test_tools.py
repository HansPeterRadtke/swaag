from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from swaag.notes import make_note
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.tools.builtin import EditTextTool, ReadTextTool, RunTestsTool
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



def test_edit_tool_replace_pattern_requires_replacement(make_config, tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    registry = ToolRegistry()
    config = make_config(tools__allow_side_effect_tools=True)

    with pytest.raises(ToolValidationError, match="requires pattern and replacement"):
        registry.dispatch(
            "edit_text",
            {"path": str(path), "operation": "replace_pattern_once", "pattern": "hello"},
            config,
            _empty_state(),
        )



def test_malformed_arguments_raise_validation_error(make_config) -> None:
    with pytest.raises(ToolValidationError):
        ToolRegistry().dispatch("calculator", {"expression": "__import__('os')"}, make_config(), _empty_state())


def test_shell_command_rejects_placeholder_text(make_config) -> None:
    registry = ToolRegistry()
    config = make_config(tools__allow_side_effect_tools=True)

    with pytest.raises(ToolValidationError, match="placeholder"):
        registry.dispatch(
            "shell_command",
            {"command": "git apply -v <patch_file> && git diff --cached"},
            config,
            _empty_state(),
        )


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


def test_run_tests_tool_normalizes_bare_pytest_to_current_python() -> None:
    validated = RunTestsTool().validate({"command": ["pytest", "test_sample.py", "-q"]})
    assert validated["command"] == [sys.executable, "-m", "pytest", "test_sample.py", "-q"]


def test_read_text_prefers_explicit_path_over_stale_reader_id() -> None:
    validated = ReadTextTool().validate({"path": "policy.md", "reader_id": "stale_reader", "chunk_chars": 100})
    assert validated["path"] == "policy.md"
    assert validated["reader_id"] is None


def test_edit_tool_strips_fields_from_pattern_operations(make_config, tmp_path: Path) -> None:
    path = tmp_path / "stats.py"
    path.write_text("def moving_total(values):\n    total = 0\n    return total\n", encoding="utf-8")
    validated = EditTextTool().validate(
        {
            "path": str(path),
            "operation": "replace_pattern_once",
            "pattern": "return total",
            "replacement": "return total + values[-1]",
            "start": 0,
            "end": 64,
        }
    )
    assert "start" not in validated
    assert "end" not in validated
    registry = ToolRegistry()
    _, result = registry.dispatch(
        "edit_text",
        {
            "path": str(path),
            "operation": "replace_pattern_once",
            "pattern": "return total",
            "replacement": "return total + values[-1]",
            "start": 0,
            "end": 64,
        },
        make_config(tools__allow_side_effect_tools=True, editor__allow_writes=True),
        _empty_state(),
    )
    assert result.output["changed"] is True
    assert "return total + values[-1]" in result.output["diff"]


def test_edit_tool_decodes_model_escaped_newlines_before_matching(make_config, tmp_path: Path) -> None:
    path = tmp_path / "stats.py"
    path.write_text("def moving_total(values):\n    total = 0\n    return total\n", encoding="utf-8")
    registry = ToolRegistry()
    _, result = registry.dispatch(
        "edit_text",
        {
            "path": str(path),
            "operation": "replace_pattern_once",
            "pattern": "return total\\n",
            "replacement": "return total + values[-1]\\n",
        },
        make_config(tools__allow_side_effect_tools=True, editor__allow_writes=True),
        _empty_state(),
    )
    assert result.output["changed"] is True
    assert "return total + values[-1]" in result.output["diff"]


def test_edit_tool_accepts_replace_alias_for_pattern_replacement(make_config, tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.py"
    path.write_text("def tokenize(text: str):\n    return text.split(',')\n", encoding="utf-8")
    registry = ToolRegistry()
    _, result = registry.dispatch(
        "edit_text",
        {
            "path": str(path),
            "operation": "replace",
            "pattern": "return text.split(',')",
            "replacement": "return text.split('|')",
            "start": 0,
            "end": 64,
        },
        make_config(tools__allow_side_effect_tools=True, editor__allow_writes=True),
        _empty_state(),
    )
    assert result.output["changed"] is True
    assert "return text.split('|')" in result.output["diff"]


def test_read_text_prefers_explicit_path_over_stale_note_and_reader() -> None:
    tool = ReadTextTool()

    validated = tool.validate({"path": "pkg/file.py", "note_id": "note_old", "reader_id": "reader_old"})

    assert validated["path"] == "pkg/file.py"
    assert validated["note_id"] is None
    assert validated["reader_id"] is None


def test_edit_tool_accepts_replace_pattern_alias(make_config, tmp_path: Path) -> None:
    path = tmp_path / "formatter.py"
    path.write_text("CURRENCY = 'USD-1'\n", encoding="utf-8")
    registry = ToolRegistry()
    _, result = registry.dispatch(
        "edit_text",
        {
            "path": str(path),
            "operation": "replace_pattern",
            "pattern": "CURRENCY = 'USD-1'",
            "replacement": "CURRENCY = 'USD'",
        },
        make_config(tools__allow_side_effect_tools=True, editor__allow_writes=True),
        _empty_state(),
    )
    assert result.output["changed"] is True
    assert "CURRENCY = 'USD'" in result.output["diff"]


def test_read_text_accepts_path_list_for_multi_file_buffer() -> None:
    tool = ReadTextTool()

    validated = tool.validate({"path": ["pkg/a.py", "pkg/b.py"], "chunk_chars": 1000})

    assert validated["paths"] == ["pkg/a.py", "pkg/b.py"]
    assert validated["note_id"] is None
    assert validated["reader_id"] is None


def test_read_text_prefers_paths_over_conflicting_path_field() -> None:
    tool = ReadTextTool()

    validated = tool.validate(
        {
            "path": "pkg/a.py\npkg/b.py",
            "paths": ["pkg/a.py", "pkg/b.py"],
            "note_id": None,
            "reader_id": None,
            "chunk_chars": None,
            "overlap_chars": None,
        }
    )

    assert validated["paths"] == ["pkg/a.py", "pkg/b.py"]
    assert validated["path"] == "pkg/a.py\npkg/b.py"
    assert validated["note_id"] is None
    assert validated["reader_id"] is None


def test_read_text_accepts_paths_alias(tmp_path, make_config) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")
    tool = ReadTextTool()
    validated = tool.validate({"paths": [str(first), str(second)]})

    assert validated["paths"] == [str(first), str(second)]
    assert str(first) in validated["path"]
    assert str(second) in validated["path"]


def test_read_text_accepts_offset_hints(tmp_path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("alpha", encoding="utf-8")
    tool = ReadTextTool()

    validated = tool.validate({"path": str(path), "start_offset": 0, "end_offset": 5})

    assert validated["path"] == str(path)
