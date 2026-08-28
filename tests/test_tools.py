from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from swaag.environment.environment import AgentEnvironment
from swaag.notes import make_note
from swaag.runtime import AgentRuntime
from swaag.tools.base import (
    SemanticCallContextOverflow,
    Tool,
    ToolContext,
    ToolValidationError,
    _validate_schema_value,
)
from swaag.tools.builtin import EditTextTool, ReadFileTool, ReadTextTool, RunTestsTool, ShellCommandTool, WriteFileTool
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


def test_schema_validator_enforces_any_of_variants() -> None:
    schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}

    _validate_schema_value("ready", schema, path="value")
    _validate_schema_value(None, schema, path="value")
    with pytest.raises(ToolValidationError, match="anyOf"):
        _validate_schema_value(7, schema, path="value")



def test_read_file_guidance_requires_exact_file_discovery() -> None:
    assert "Read one exact file per call" in ReadFileTool.usage_guidance
    assert "list_files or search_repo" in ReadFileTool.usage_guidance


def test_run_tests_guidance_distinguishes_required_success_from_diagnostics() -> None:
    assert "require tool_result_success" in RunTestsTool.usage_guidance
    assert "diagnostics where failure is acceptable" in RunTestsTool.usage_guidance
    assert RunTestsTool.objective_verification_check_types == (
        "tool_result_success",
        "tool_output_nonempty",
        "tool_output_schema_valid",
    )

def test_calculator_tool_executes(make_config) -> None:
    registry = ToolRegistry()
    config = make_config()
    invocation, result = registry.dispatch("calculator", {"expression": "2 + 3 * 4"}, config, _empty_state())
    assert invocation.validated_input == {"expression": "2 + 3 * 4"}
    assert result.output["result"] == 14


def test_calculator_tool_supports_safe_round(make_config) -> None:
    registry = ToolRegistry()
    config = make_config()
    _, result = registry.dispatch(
        "calculator",
        {"expression": "round(42 * 142 / 100 * (100 + 22) / 100)"},
        config,
        _empty_state(),
    )
    assert result.output["result"] == 73
    _, result = registry.dispatch("calculator", {"expression": "round(72.7608, 2)"}, config, _empty_state())
    assert result.output["result"] == 72.76


def test_calculator_rejects_arbitrary_function_calls(make_config) -> None:
    registry = ToolRegistry()
    config = make_config()
    for expression in ("abs(-2)", "__import__('os')", "round(1, ndigits=2)"):
        with pytest.raises(ToolValidationError):
            registry.dispatch("calculator", {"expression": expression}, config, _empty_state())



def test_unknown_tool_raises(make_config) -> None:
    with pytest.raises(KeyError):
        ToolRegistry().dispatch("missing", {}, make_config(), _empty_state())


def test_registered_but_unconfigured_tool_is_blocked(make_config) -> None:
    config = make_config(tools__enabled=["echo"])

    with pytest.raises(PermissionError, match="not enabled by configuration"):
        ToolRegistry().dispatch(
            "calculator",
            {"expression": "1 + 1"},
            config,
            _empty_state(),
        )



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
    _, result = registry.dispatch(
        "notes",
        {
            "action": "add",
            "note_id": None,
            "title": "Todo",
            "content": "Check file",
            "categories": ["software investigation"],
        },
        config,
        _empty_state(),
    )
    assert result.generated_events
    assert result.generated_events[0].event_type == "note_added"
    assert result.output["categories"] == ["software investigation"]


def test_notes_tool_remove_is_durable(make_config) -> None:
    registry = ToolRegistry()
    config = make_config()
    state = _empty_state()
    note = make_note(config, title="Obsolete", content="retired workaround")
    state.notes.append(note)

    _, result = registry.dispatch(
        "notes",
        {
            "action": "remove",
            "note_id": note.note_id,
            "title": None,
            "content": None,
            "categories": None,
        },
        config,
        state,
    )

    assert result.output == {"note_id": note.note_id, "removed": True}
    assert result.generated_events[0].event_type == "note_removed"


def test_notes_tool_add_fails_closed_at_capacity_without_generated_mutation(
    make_config,
) -> None:
    registry = ToolRegistry()
    config = make_config(notes__max_notes=1)
    state = _empty_state()
    state.notes.append(make_note(config, title="Existing", content="exact fact"))

    with pytest.raises(ToolValidationError, match="compact existing notes"):
        registry.dispatch(
            "notes",
            {
                "action": "add",
                "note_id": None,
                "title": "New",
                "content": "new exact fact",
                "categories": [],
            },
            config,
            state,
        )

    assert len(state.notes) == 1
    assert state.notes[0].content == "exact fact"


def test_notes_tool_compaction_is_central_semantic_call_with_exact_sources(
    make_config,
) -> None:
    registry = ToolRegistry()
    config = make_config()
    state = _empty_state()
    state.notes.extend(
        [
            make_note(config, title="Constraint", content="Never remove marker-17"),
            make_note(config, title="Evidence", content="Tool check passed at 12:30"),
        ]
    )
    observed = {}

    def semantic_call(request):
        observed["request"] = request
        exact = "".join(component.text for component in request.components)
        assert "marker-17" in exact
        assert "12:30" in exact
        return {
            "title": "Exact durable state",
            "content": "Never remove marker-17. Tool check passed at 12:30.",
            "categories": ["constraints", "verified evidence"],
        }

    _, result = registry.dispatch(
        "notes",
        {
            "action": "compact",
            "note_id": None,
            "title": None,
            "content": None,
            "categories": None,
        },
        config,
        state,
        semantic_call=semantic_call,
    )

    request = observed["request"]
    assert request.kind == "notes_compaction"
    assert request.contract.name == "notes_compaction"
    assert result.output["compacted"] is True
    assert result.output["compacted_note"]["content"].endswith("12:30.")
    assert result.output["compacted_note"]["categories"] == [
        "constraints",
        "verified evidence",
    ]
    event = next(
        item for item in result.generated_events if item.event_type == "notes_compacted"
    )
    assert event.payload["semantic"] is True
    assert event.payload["source_note_ids"] == [note.note_id for note in state.notes]


def test_notes_tool_compaction_recovers_from_measured_context_overflow(
    make_config,
) -> None:
    registry = ToolRegistry()
    config = make_config()
    state = _empty_state()
    state.notes.extend(
        [
            make_note(config, title="Left", content="left-marker"),
            make_note(config, title="Right", content="right-marker"),
        ]
    )
    calls = []

    def semantic_call(request):
        source = request.components[-1].text
        calls.append(source)
        if len(calls) == 1:
            raise SemanticCallContextOverflow(None)
        if "Semantic fragment projections" in source:
            assert "left-fragment" in source
            assert "right-fragment" in source
            return {
                "title": "Recovered",
                "content": "left-marker and right-marker are both preserved",
                "categories": ["recovered state"],
            }
        if "left-marker" in source:
            return {
                "title": "Left",
                "content": "left-fragment",
                "categories": ["left"],
            }
        return {
            "title": "Right",
            "content": "right-fragment",
            "categories": ["right"],
        }

    _, result = registry.dispatch(
        "notes",
        {
            "action": "compact",
            "note_id": None,
            "title": None,
            "content": None,
            "categories": None,
        },
        config,
        state,
        semantic_call=semantic_call,
    )

    assert len(calls) == 4
    assert result.output["compacted_note"]["content"] == (
        "left-marker and right-marker are both preserved"
    )


def test_notes_tool_compaction_repairs_mechanical_storage_failure(
    make_config,
) -> None:
    registry = ToolRegistry()
    config = make_config(notes__max_note_chars=40, model__max_retries=1)
    state = _empty_state()
    state.notes.extend(
        [
            make_note(config, title="Left", content="left-marker"),
            make_note(config, title="Right", content="right-marker"),
        ]
    )
    requests = []

    def semantic_call(request):
        requests.append(request)
        if len(requests) == 1:
            return {
                "title": "Too large",
                "content": "x" * 41,
                "categories": ["state"],
            }
        feedback = next(
            component
            for component in request.components
            if component.name == "notes_compaction_validation_feedback"
        )
        assert "max_note_chars" in feedback.text
        exact_sources = next(
            component
            for component in request.components
            if component.name == "notes_compaction_sources"
        )
        assert "left-marker" in exact_sources.text
        assert "right-marker" in exact_sources.text
        return {
            "title": "Repaired",
            "content": "left-marker; right-marker",
            "categories": ["state"],
        }

    _, result = registry.dispatch(
        "notes",
        {
            "action": "compact",
            "note_id": None,
            "title": None,
            "content": None,
            "categories": None,
        },
        config,
        state,
        semantic_call=semantic_call,
    )

    assert len(requests) == 2
    assert result.output["compacted_note"]["content"] == "left-marker; right-marker"



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


def test_runtime_records_successful_persisted_effect_verification(make_config, tmp_path: Path) -> None:
    target = tmp_path / "runtime-write.txt"
    target.write_text("before\n", encoding="utf-8")
    config = make_config(
        tools__enabled=["write_file"],
        tools__allow_side_effect_tools=True,
        editor__allow_writes=True,
        editor__allowed_write_paths=[str(target)],
    )
    runtime = AgentRuntime(config, model_client=object())

    run = runtime.execute_tool_once(
        "write_file",
        {"path": str(target), "content": "after\n", "create": False},
    )

    assert run.error is None
    assert run.tool_result is not None
    events = runtime.history.read_history(run.session_id)
    verification = [
        event for event in events if event.event_type == "tool_effect_verified"
    ]
    assert len(verification) == 1
    assert verification[0].payload["tool_name"] == "write_file"
    assert verification[0].payload["evidence"]["persisted"] is True
    assert target.read_text(encoding="utf-8") == "after\n"


def test_runtime_fails_closed_when_persisted_effect_verification_fails(
    make_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "runtime-write-failure.txt"
    target.write_text("before\n", encoding="utf-8")
    config = make_config(
        tools__enabled=["write_file"],
        tools__allow_side_effect_tools=True,
        editor__allow_writes=True,
        editor__allowed_write_paths=[str(target)],
    )
    runtime = AgentRuntime(config, model_client=object())
    write_tool = runtime.tools.get("write_file")
    monkeypatch.setattr(
        write_tool,
        "verify_effect",
        lambda _result, _environment: (
            False,
            {"reason": "simulated_persistence_mismatch"},
        ),
    )

    run = runtime.execute_tool_once(
        "write_file",
        {"path": str(target), "content": "after\n", "create": False},
    )

    assert run.tool_result is None
    assert run.error is not None
    assert run.error["error_type"] == "ToolEffectVerificationError"
    events = runtime.history.read_history(run.session_id)
    event_types = [event.event_type for event in events]
    assert "tool_effect_verification_failed" in event_types
    assert "tool_error" in event_types
    assert "tool_result" not in event_types


def test_write_file_effect_rejects_noop_write(make_config, tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("same\n", encoding="utf-8")
    state = _empty_state()
    config = make_config(tools__allow_side_effect_tools=True)
    environment = AgentEnvironment(config, state)
    with pytest.raises(ToolValidationError, match="would make no change"):
        environment.write_file(str(target), "same\n", create=False)


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


def test_write_file_blocked_by_editor_policy(make_config, tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    registry = ToolRegistry()
    config = make_config(
        tools__allow_side_effect_tools=True,
        editor__allow_writes=False,
    )

    with pytest.raises(PermissionError, match="write_file"):
        registry.dispatch(
            "write_file",
            {"path": str(path), "content": "world", "create": False},
            config,
            _empty_state(),
        )

    assert path.read_text(encoding="utf-8") == "hello"



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


def test_shell_command_allows_test_runners_as_general_shell_work() -> None:
    tool = ShellCommandTool()
    for command in (
        "pytest -q",
        "python3 -m pytest -q tests",
        "python -m unittest -q test_mod.py",
        "cd pkg && python3 -m unittest test_x",
        "tox -q",
        "nox -s tests",
    ):
        validated = tool.validate({"command": command, "background": False})
        assert validated == {"command": command, "background": False}
    assert tool.validate({"command": "python3 script.py", "background": False})["command"] == "python3 script.py"


def test_edit_text_noop_does_not_require_edit_applied_event() -> None:
    tool = EditTextTool()
    validated = tool.validate({
        "path": "x.txt",
        "operation": "replace_exact",
        "old_text": "a",
        "new_text": "a",
        "dry_run": False,
    })
    assert tool.required_generated_event_types(validated) == {"file_read_for_edit"}


def test_edit_text_rejects_python_syntax_regression(make_config, tmp_path: Path) -> None:
    from swaag.environment.environment import AgentEnvironment
    from swaag.history import HistoryStore

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.tools.allow_side_effect_tools = True
    config.editor.allow_writes = True
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config.tools.read_roots = [workspace]
    target = workspace / "mod.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    env = AgentEnvironment(config, state)
    context = ToolContext(config=config, session_state=state, environment=env)

    validated = EditTextTool().validate({
        "path": "mod.py",
        "operation": "replace_exact",
        "dry_run": False,
        "old_text": "    return 1",
        "new_text": "return 2",
        "start": None,
        "end": None,
        "position": None,
        "expected_text": None,
        "replacement": None,
        "insertion": None,
        "pattern": None,
    })
    with pytest.raises(ToolValidationError, match="regress a syntactically valid Python file") as exc:
        EditTextTool().execute(validated, context)
    assert '"line":2' in str(exc.value)
    assert target.read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_edit_text_allows_repair_of_already_invalid_python(make_config, tmp_path: Path) -> None:
    from swaag.environment.environment import AgentEnvironment
    from swaag.history import HistoryStore

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.tools.allow_side_effect_tools = True
    config.editor.allow_writes = True
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config.tools.read_roots = [workspace]
    target = workspace / "mod.py"
    target.write_text("def f():\nreturn 1\n", encoding="utf-8")
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    env = AgentEnvironment(config, state)
    context = ToolContext(config=config, session_state=state, environment=env)

    validated = EditTextTool().validate({
        "path": "mod.py",
        "operation": "replace_exact",
        "dry_run": False,
        "old_text": "return 1",
        "new_text": "    return 1",
        "start": None,
        "end": None,
        "position": None,
        "expected_text": None,
        "replacement": None,
        "insertion": None,
        "pattern": None,
    })
    result = EditTextTool().execute(validated, context)
    assert result.output["changed"] is True
    # Apply derived write the same way runtime history does not occur in direct execute;
    # syntax guard itself must permit the proposed repair.
    assert any(event.event_type == "edit_applied" for event in result.generated_events)


def test_write_file_rejects_python_syntax_regression(make_config, tmp_path: Path) -> None:
    from swaag.environment.environment import AgentEnvironment
    from swaag.history import HistoryStore

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.tools.allow_side_effect_tools = True
    config.editor.allow_writes = True
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config.tools.read_roots = [workspace]
    target = workspace / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    env = AgentEnvironment(config, state)

    with pytest.raises(ToolValidationError, match="regress a syntactically valid Python file"):
        env.write_file("mod.py", "def broken(:\n", create=False)
    assert target.read_text(encoding="utf-8") == "x = 1\n"


def test_edit_text_rejects_return_outside_function_regression(make_config, tmp_path: Path) -> None:
    from swaag.environment.environment import AgentEnvironment
    from swaag.history import HistoryStore

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.tools.allow_side_effect_tools = True
    config.editor.allow_writes = True
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config.tools.read_roots = [workspace]
    target = workspace / "stats.py"
    original = (
        "def moving_total(values: list[int]) -> int:\n"
        "    total = 0\n"
        "    for value in values[:-1]:\n"
        "        total += value\n"
        "    return total\n"
    )
    target.write_text(original, encoding="utf-8")
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    env = AgentEnvironment(config, state)
    context = ToolContext(config=config, session_state=state, environment=env)
    validated = EditTextTool().validate({
        "path": "stats.py",
        "operation": "replace_exact",
        "dry_run": False,
        "old_text": "total += value\\n    return total\\n",
        "new_text": "total += value\\nreturn total\\n",
        "start": None,
        "end": None,
        "position": None,
        "expected_text": None,
        "replacement": None,
        "insertion": None,
        "pattern": None,
    })
    with pytest.raises(ToolValidationError, match="return.*outside function"):
        EditTextTool().execute(validated, context)
    assert target.read_text(encoding="utf-8") == original


def test_terminal_guidance_requires_exact_returned_terminal_id() -> None:
    from swaag.tools.terminal import TerminalTool
    assert "returned terminal_id" in TerminalTool.usage_guidance
    assert "do not invent aliases" in TerminalTool.usage_guidance
    assert "later actions" in TerminalTool.usage_guidance


def test_artifact_guidance_requires_exact_returned_artifact_id() -> None:
    from swaag.tools.artifacts import ReadArtifactTool
    assert "exact stdout_artifact_id or stderr_artifact_id" in ReadArtifactTool.usage_guidance
    assert "never use a filename" in ReadArtifactTool.usage_guidance
    assert "later action" in ReadArtifactTool.usage_guidance


def test_wait_seconds_emits_completed_event(make_config, tmp_path) -> None:
    from swaag.environment.environment import AgentEnvironment
    from swaag.history import HistoryStore
    from swaag.tools.builtin import WaitSecondsTool

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    env = AgentEnvironment(config, state)
    result = WaitSecondsTool().execute({"seconds": 0.0, "duration": "0 ms"}, ToolContext(config=config, session_state=state, environment=env))
    assert [event.event_type for event in result.generated_events] == ["wait_entered", "wait_resumed", "wait_completed"]
    assert result.generated_events[-1].payload["elapsed_seconds"] >= 0.0


def test_read_artifact_resolves_latest_symbolic_stdout_handle(make_config, tmp_path) -> None:
    from swaag.environment.artifacts import TextArtifactStore
    from swaag.environment.environment import AgentEnvironment
    from swaag.history import HistoryStore
    from swaag.tools.artifacts import ReadArtifactTool
    from swaag.types import Message

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    env = AgentEnvironment(config, state)
    artifact = TextArtifactStore(config.sessions.root, state.session_id).create("abcdef", kind="shell_command_stdout")
    state.messages.append(Message(role="tool", name="shell_command", content="ok", created_at="2026-01-01T00:00:00+00:00", metadata={"output": {"stdout_artifact_id": artifact.artifact_id}}))
    result = ReadArtifactTool().execute(
        {"artifact_id": "stdout_artifact_id", "start_offset": 0, "max_chars": 3},
        ToolContext(config=config, session_state=state, environment=env),
    )
    assert result.output["artifact_id"] == artifact.artifact_id
    assert result.output["text"] == "abc"
    assert result.output["next_offset"] == 3


def test_read_artifact_accepts_stdout_alias(make_config, tmp_path) -> None:
    from swaag.environment.artifacts import TextArtifactStore
    from swaag.environment.environment import AgentEnvironment
    from swaag.history import HistoryStore
    from swaag.tools.artifacts import ReadArtifactTool
    from swaag.types import Message

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    env = AgentEnvironment(config, state)
    artifact = TextArtifactStore(config.sessions.root, state.session_id).create("marker\n", kind="shell_command_stdout")
    state.messages.append(Message(role="tool", name="shell_command", content="ok", created_at="2026-01-01T00:00:00+00:00", metadata={"output": {"stdout_artifact_id": artifact.artifact_id}}))
    result = ReadArtifactTool().execute(
        {"artifact_id": "stdout", "start_offset": 0, "max_chars": 100},
        ToolContext(config=config, session_state=state, environment=env),
    )
    assert result.output["artifact_id"] == artifact.artifact_id
    assert result.output["text"] == "marker\n"


def test_terminal_store_resolves_terminal1_when_one_active_terminal_exists(tmp_path, monkeypatch) -> None:
    from swaag.environment.terminal import TerminalRecord, TerminalStore

    store = TerminalStore(tmp_path, "session_x")
    record = TerminalRecord(
        terminal_id="terminal_real", name="persistent_terminal", root="/tmp/x", cwd="/tmp", shell="/bin/bash",
        worker_pid=1, shell_pid=2, active=True, return_code=None, created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00", output_chars=0,
    )
    monkeypatch.setattr(store, "list", lambda: [record])
    monkeypatch.setattr(store, "_dir", lambda ref: tmp_path / "missing")
    assert store.resolve("terminal1") == "terminal_real"


def test_read_artifact_guidance_requires_pagination_when_unfinished() -> None:
    from swaag.tools.artifacts import ReadArtifactTool
    assert "If finished=false, unread exact data remains" in ReadArtifactTool.usage_guidance
    assert "next_offset" in ReadArtifactTool.usage_guidance


def test_tool_specific_execution_timeout_override_is_honored(make_config) -> None:
    import time
    from swaag.history import HistoryStore
    from swaag.tools.base import Tool, ToolContext
    from swaag.tools.registry import ToolRegistry
    from swaag.types import ToolExecutionResult

    class SlowButAllowedTool(Tool):
        name = "slow_allowed"
        description = "slow test"
        input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

        def validate(self, raw_input):
            return {}

        def execution_timeout_seconds(self, context: ToolContext) -> float:
            return 0.25

        def execute(self, validated_input, context: ToolContext) -> ToolExecutionResult:
            time.sleep(0.08)
            return ToolExecutionResult(tool_name=self.name, output={"ok": True}, display_text="ok")

    config = make_config(tools__enabled=["slow_allowed"], runtime__tool_timeout_seconds=0.02)
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint="cfg", model_base_url="http://model")
    registry = ToolRegistry([SlowButAllowedTool()])
    _invocation, result = registry.dispatch("slow_allowed", {}, config, state)
    assert result.output == {"ok": True}


def test_caller_managed_tool_timeout_runs_on_dispatch_thread(make_config) -> None:
    import threading

    from swaag.history import HistoryStore
    from swaag.tools.base import Tool, ToolContext
    from swaag.tools.registry import ToolRegistry
    from swaag.types import ToolExecutionResult

    dispatch_thread = threading.get_ident()

    class CallerManagedTool(Tool):
        name = "caller_managed"
        description = "caller-managed test"
        input_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

        def validate(self, raw_input):
            return {}

        def execution_timeout_seconds(self, context: ToolContext) -> None:
            return None

        def execute(self, validated_input, context: ToolContext) -> ToolExecutionResult:
            return ToolExecutionResult(
                tool_name=self.name,
                output={"thread": threading.get_ident()},
                display_text="ok",
            )

    config = make_config(tools__enabled=["caller_managed"])
    state = HistoryStore(config.sessions.root).create(
        config_fingerprint="cfg", model_base_url="http://model"
    )
    _invocation, result = ToolRegistry([CallerManagedTool()]).dispatch(
        "caller_managed", {}, config, state
    )
    assert result.output["thread"] == dispatch_thread


def test_history_analyze_uses_runtime_managed_model_timeouts(make_config) -> None:
    from swaag.environment.environment import AgentEnvironment
    from swaag.history import HistoryStore
    from swaag.tools.base import ToolContext
    from swaag.tools.history import HistoryAnalyzeTool

    config = make_config(runtime__tool_timeout_seconds=10)
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    context = ToolContext(config=config, session_state=state, environment=AgentEnvironment(config, state))
    assert HistoryAnalyzeTool().execution_timeout_seconds(context) is None


def test_editor_write_allowlist_blocks_protected_files_and_allows_declared_target(make_config, tmp_path: Path) -> None:
    allowed = tmp_path / "allowed.py"
    protected = tmp_path / "test_protected.py"
    allowed.write_text("value = 1\n", encoding="utf-8")
    protected.write_text("assert True\n", encoding="utf-8")
    state = _empty_state()
    config = make_config(tools__allow_side_effect_tools=True)
    config.editor.allow_writes = True
    config.editor.allowed_write_paths = [str(allowed)]
    environment = AgentEnvironment(config, state)

    changed = environment.preview_or_apply_edit(
        {"path": str(allowed), "operation": "replace_exact", "old_text": "value = 1", "new_text": "value = 2", "dry_run": False},
        ToolContext(config=config, session_state=state, environment=environment),
    )
    assert changed.output["changed"] is True
    with pytest.raises(PermissionError, match="forbidden by the active editor allowlist"):
        environment.preview_or_apply_edit(
            {"path": str(protected), "operation": "replace_exact", "old_text": "assert True", "new_text": "assert False", "dry_run": False},
            ToolContext(config=config, session_state=state, environment=environment),
        )
    assert protected.read_text(encoding="utf-8") == "assert True\n"


def test_real_noop_edit_and_write_are_rejected_as_no_progress(make_config, tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("same\n", encoding="utf-8")
    state = _empty_state()
    config = make_config(tools__allow_side_effect_tools=True)
    config.editor.allow_writes = True
    environment = AgentEnvironment(config, state)
    context = ToolContext(config=config, session_state=state, environment=environment)

    with pytest.raises(ToolValidationError, match="would make no change"):
        environment.preview_or_apply_edit(
            {"path": str(target), "operation": "replace_exact", "old_text": "same", "new_text": "same", "dry_run": False},
            context,
        )
    with pytest.raises(ToolValidationError, match="would make no change"):
        environment.write_file(str(target), "same\n", create=False)
    preview = environment.preview_or_apply_edit(
        {"path": str(target), "operation": "replace_exact", "old_text": "same", "new_text": "same", "dry_run": True},
        context,
    )
    assert preview.output["changed"] is False
