from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from swaag.environment.environment import AgentEnvironment
from swaag.notes import make_note
from swaag.tools.base import Tool, ToolContext, ToolValidationError, _validate_schema_value
from swaag.tools.builtin import EditTextTool, ReadTextTool, RunTestsTool, ShellCommandTool, WriteFileTool
from swaag.tools.registry import ToolRegistry
from swaag.types import SessionState, ToolExecutionResult



def _empty_state() -> SessionState:
    return SessionState(session_id="s", created_at="t", updated_at="t", config_fingerprint="cfg", model_base_url="http://example.test")



def test_schema_validator_accepts_union_types_and_null() -> None:
    schema = {"type": ["string", "integer", "null"]}

    _validate_schema_value("ready", schema, path="value")
    _validate_schema_value(7, schema, path="value")
    _validate_schema_value(None, schema, path="value")
    with pytest.raises(ToolValidationError):
        _validate_schema_value([], schema, path="value")


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
        {"path": str(path), "operation": "replace_exact", "old_text": "hello", "new_text": "world", "dry_run": True},
        config,
        _empty_state(),
    )
    assert any(event.event_type == "edit_previewed" for event in result.generated_events)
    assert path.read_text(encoding="utf-8") == "hello"


def test_edit_tool_guidance_explains_full_match_replacement() -> None:
    assert "Prefer replace_exact" in EditTextTool.usage_guidance
    assert "exactly one match" in EditTextTool.usage_guidance
    assert "replace the entire matched text" in EditTextTool.usage_guidance
    assert "expected_text" in EditTextTool.usage_guidance
    assert "Use replace_range only as a low-level fallback" in EditTextTool.usage_guidance
    assert "absence fails closed" in EditTextTool.usage_guidance
    assert "tool_effect_verified" in EditTextTool.usage_guidance
    assert "current file exactly matches" in EditTextTool.usage_guidance


def test_edit_tool_applies_exact_replacement_without_offsets(make_config, tmp_path: Path) -> None:
    path = tmp_path / "release.yaml"
    path.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    registry = ToolRegistry()

    _, result = registry.dispatch(
        "edit_text",
        {
            "path": str(path),
            "operation": "replace_exact",
            "dry_run": False,
            "old_text": "status: draft",
            "new_text": "status: ready",
        },
        make_config(tools__allow_side_effect_tools=True, editor__allow_writes=True),
        _empty_state(),
    )

    assert result.output["changed"] is True
    assert result.output["before_sha256"] != result.output["after_sha256"]
    assert result.output["details"]["match_count"] == 1
    assert result.output["details"]["precondition"] == "exactly_one_old_text_match"
    applied = next(event for event in result.generated_events if event.event_type == "edit_applied")
    assert applied.derived_writes
    assert applied.derived_writes[0].content == "name: report-62\nstatus: ready\nowner: team-6\n"
    assert path.read_text(encoding="utf-8") == "name: report-62\nstatus: draft\nowner: team-6\n"


def test_edit_tool_exact_replacement_fails_closed_on_zero_or_multiple_matches(make_config, tmp_path: Path) -> None:
    path = tmp_path / "release.yaml"
    original = "status: draft\nstatus: draft\n"
    path.write_text(original, encoding="utf-8")
    registry = ToolRegistry()
    config = make_config(tools__allow_side_effect_tools=True, editor__allow_writes=True)

    with pytest.raises(ToolValidationError, match="old_text is ambiguous; match_count=2"):
        registry.dispatch(
            "edit_text",
            {
                "path": str(path),
                "operation": "replace_exact",
                "dry_run": False,
                "old_text": "status: draft",
                "new_text": "status: ready",
            },
            config,
            _empty_state(),
        )

    with pytest.raises(ToolValidationError, match="old_text not found; match_count=0"):
        registry.dispatch(
            "edit_text",
            {
                "path": str(path),
                "operation": "replace_exact",
                "dry_run": False,
                "old_text": "status: missing",
                "new_text": "status: ready",
            },
            config,
            _empty_state(),
        )

    assert path.read_text(encoding="utf-8") == original


def test_edit_tool_rejects_range_when_expected_text_mismatches(make_config, tmp_path: Path) -> None:
    path = tmp_path / "release.yaml"
    original = "name: report-62\nstatus: draft\nowner: team-6\n"
    path.write_text(original, encoding="utf-8")
    registry = ToolRegistry()

    with pytest.raises(
        ToolValidationError,
        match=r"selected='s: dr'.*expected='draft'.*range_units=zero_based_character_offsets.*expected_text_matching_ranges=\[{\"end\":29,\"start\":24}\]",
    ):
        registry.dispatch(
            "edit_text",
            {
                "path": str(path),
                "operation": "replace_range",
                "dry_run": False,
                "start": 21,
                "end": 26,
                "position": None,
                "expected_text": "draft",
                "replacement": "ready",
                "insertion": None,
                "pattern": None,
            },
            make_config(tools__allow_side_effect_tools=True, editor__allow_writes=True),
            _empty_state(),
        )

    assert path.read_text(encoding="utf-8") == original


def test_edit_tool_applies_range_when_expected_text_matches(make_config, tmp_path: Path) -> None:
    path = tmp_path / "release.yaml"
    path.write_text("name: report-62\nstatus: draft\nowner: team-6\n", encoding="utf-8")
    registry = ToolRegistry()

    _, result = registry.dispatch(
        "edit_text",
        {
            "path": str(path),
            "operation": "replace_range",
            "dry_run": False,
            "start": 24,
            "end": 29,
            "position": None,
            "expected_text": "draft",
            "replacement": "ready",
            "insertion": None,
            "pattern": None,
        },
        make_config(tools__allow_side_effect_tools=True, editor__allow_writes=True),
        _empty_state(),
    )

    assert result.output["changed"] is True
    assert "+status: ready" in result.output["diff"]


def test_edit_tool_pattern_not_found_exposes_current_text_for_model_recovery(make_config, tmp_path: Path) -> None:
    path = tmp_path / "release.yaml"
    path.write_text("name: report-62\nready\nowner: team-6\n", encoding="utf-8")
    registry = ToolRegistry()

    with pytest.raises(ToolValidationError) as excinfo:
        registry.dispatch(
            "edit_text",
            {
                "path": str(path),
                "operation": "replace_pattern_once",
                "dry_run": False,
                "start": None,
                "end": None,
                "position": None,
                "expected_text": None,
                "replacement": "ready",
                "insertion": None,
                "pattern": "status: draft",
            },
            make_config(tools__allow_side_effect_tools=True),
            _empty_state(),
        )

    error = str(excinfo.value)
    assert "pattern not found" in error
    assert "current_text" in error
    assert "name: report-62\\nready\\nowner: team-6\\n" in error
    assert path.read_text(encoding="utf-8") == "name: report-62\nready\nowner: team-6\n"


def test_write_file_guidance_uses_registered_persisted_hash_verification() -> None:
    assert "persisted-hash tool_effect_verified check" in WriteFileTool.usage_guidance
    assert "command_success only for a distinct executable correctness test" in WriteFileTool.usage_guidance




def test_write_file_reports_and_verifies_persisted_hash(make_config, tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    state = _empty_state()
    config = make_config(tools__allow_side_effect_tools=True)
    environment = AgentEnvironment(config, state)
    result = environment.write_file(str(target), "after\n", create=False)

    assert result.output["changed"] is True
    assert result.output["existed_before"] is True
    assert result.output["before_sha256"] != result.output["after_sha256"]
    target.write_text("after\n", encoding="utf-8")
    passed, evidence = WriteFileTool().verify_effect(result, environment)
    assert passed is True
    assert evidence["persisted"] is True
    assert evidence["real_change"] is True


def test_write_file_effect_rejects_noop_write(make_config, tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("same\n", encoding="utf-8")
    state = _empty_state()
    config = make_config(tools__allow_side_effect_tools=True)
    environment = AgentEnvironment(config, state)
    result = environment.write_file(str(target), "same\n", create=False)

    passed, evidence = WriteFileTool().verify_effect(result, environment)
    assert passed is False
    assert evidence["persisted"] is True
    assert evidence["real_change"] is False


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


def test_shell_command_validation_does_not_classify_placeholder_like_text() -> None:
    validated = ShellCommandTool().validate({"command": "git apply -v <patch_file> && git diff --cached"})

    assert validated == {"command": "git apply -v <patch_file> && git diff --cached", "background": False}


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


def test_tool_registry_has_no_semantic_tool_chain_helpers(make_config) -> None:
    registry = ToolRegistry()
    for name in ["capability_graph", "can_chain", "shortest_chain", "plan_tool_graph", "validate_tool_chain"]:
        assert not hasattr(registry, name)


def test_run_tests_tool_normalizes_bare_pytest_to_current_python() -> None:
    validated = RunTestsTool().validate({"command": ["pytest", "test_sample.py", "-q"]})
    assert validated["command"] == [sys.executable, "-m", "pytest", "test_sample.py", "-q"]


def test_read_text_rejects_explicit_path_with_stale_reader_id() -> None:
    with pytest.raises(ToolValidationError, match="exactly one"):
        ReadTextTool().validate({"path": "policy.md", "reader_id": "stale_reader", "chunk_chars": 100})


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


def test_edit_tool_rejects_replace_alias_for_pattern_replacement(make_config, tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.py"
    path.write_text("def tokenize(text: str):\n    return text.split(',')\n", encoding="utf-8")
    registry = ToolRegistry()
    with pytest.raises(ToolValidationError, match="operation is invalid"):
        registry.dispatch(
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


def test_read_text_rejects_explicit_path_with_stale_note_and_reader() -> None:
    tool = ReadTextTool()

    with pytest.raises(ToolValidationError, match="exactly one"):
        tool.validate({"path": "pkg/file.py", "note_id": "note_old", "reader_id": "reader_old"})


def test_edit_tool_rejects_replace_pattern_alias(make_config, tmp_path: Path) -> None:
    path = tmp_path / "formatter.py"
    path.write_text("CURRENCY = 'USD-1'\n", encoding="utf-8")
    registry = ToolRegistry()
    with pytest.raises(ToolValidationError, match="operation is invalid"):
        registry.dispatch(
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


def test_read_text_accepts_path_list_for_multi_file_buffer() -> None:
    tool = ReadTextTool()

    validated = tool.validate(
        {
            "path": None,
            "paths": ["pkg/a.py", "pkg/b.py"],
            "note_id": None,
            "reader_id": None,
            "chunk_chars": 1000,
            "overlap_chars": None,
            "start_offset": None,
            "end_offset": None,
        }
    )

    assert validated["paths"] == ["pkg/a.py", "pkg/b.py"]
    assert validated["note_id"] is None
    assert validated["reader_id"] is None


def test_read_text_rejects_paths_with_conflicting_path_field() -> None:
    tool = ReadTextTool()

    with pytest.raises(ToolValidationError, match="exactly one"):
        tool.validate(
            {
                "path": "pkg/a.py\npkg/b.py",
                "paths": ["pkg/a.py", "pkg/b.py"],
                "note_id": None,
                "reader_id": None,
                "chunk_chars": None,
                "overlap_chars": None,
            }
        )


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
