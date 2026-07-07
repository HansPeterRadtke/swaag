from __future__ import annotations

import json
from pathlib import Path

import pytest

from swaag.types import HistoryEvent, Plan, PlanStep, SessionState, ToolExecutionResult
from swaag.verification import BenchmarkVerificationReport, VerificationArtifacts, VerificationEngine, VerificationError, verify_benchmark_contract


class _RuntimeStub:
    def __init__(self, payload: dict | None = None):
        self.payload = payload or {"criteria": []}

    def _run_llm_verification(self, state, *, step, criteria, assistant_text, evidence):  # noqa: ANN001
        return self.payload


def _plan(step: PlanStep) -> Plan:
    return Plan(
        plan_id="plan_1",
        goal="goal",
        steps=[step],
        success_criteria="done",
        fallback_strategy="replan",
        status="active",
        created_at="t0",
        updated_at="t0",
        current_step_id=step.step_id,
    )


def _state() -> SessionState:
    return SessionState(session_id="session", created_at="t0", updated_at="t0", config_fingerprint="cfg", model_base_url="http://x")


def test_execution_verification_passes_for_zero_exit_and_passing_pytest(tmp_path: Path) -> None:
    test_file = tmp_path / "test_ok.py"
    test_file.write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    step = PlanStep(
        step_id="step_exec",
        title="Run tests",
        goal="Run tests",
        kind="reasoning",
        expected_tool=None,
        input_text="run",
        expected_output="tests pass",
        done_condition="reasoning_result_nonempty",
        success_criteria="tests pass",
        verification_type="execution",
        verification_checks=[
            {
                "name": "pytest_green",
                "check_type": "command_success",
                "command": ["python3", "-m", "pytest", str(test_file), "-q"],
                "cwd": str(tmp_path),
                "framework": "pytest",
            }
        ],
        required_conditions=["pytest_green"],
        optional_conditions=[],
    )
    result = VerificationEngine().verify_step(runtime=_RuntimeStub(), state=_state(), plan=_plan(step), step=step, artifacts=VerificationArtifacts())
    assert result.verification_passed is True
    assert result.verification_type_used == "execution"


def test_execution_verification_normalizes_bare_pytest_to_current_python(tmp_path: Path) -> None:
    test_file = tmp_path / "test_ok.py"
    test_file.write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    step = PlanStep(
        step_id="step_exec",
        title="Run tests",
        goal="Run tests",
        kind="reasoning",
        expected_tool=None,
        input_text="run",
        expected_output="tests pass",
        done_condition="reasoning_result_nonempty",
        success_criteria="tests pass",
        verification_type="execution",
        verification_checks=[
            {
                "name": "pytest_green",
                "check_type": "command_success",
                "command": ["pytest", str(test_file), "-q"],
                "cwd": str(tmp_path),
                "framework": "pytest",
            }
        ],
        required_conditions=["pytest_green"],
        optional_conditions=[],
    )
    result = VerificationEngine().verify_step(runtime=_RuntimeStub(), state=_state(), plan=_plan(step), step=step, artifacts=VerificationArtifacts())
    assert result.verification_passed is True
    assert result.evidence["pytest_green"]["command"][0].endswith("python3.11")
    assert result.evidence["pytest_green"]["command"][1:3] == ["-m", "pytest"]


def test_execution_verification_fails_for_failing_pytest(tmp_path: Path) -> None:
    test_file = tmp_path / "test_fail.py"
    test_file.write_text("def test_fail():\n    assert 1 == 2\n", encoding="utf-8")
    step = PlanStep(
        step_id="step_exec",
        title="Run tests",
        goal="Run tests",
        kind="reasoning",
        expected_tool=None,
        input_text="run",
        expected_output="tests pass",
        done_condition="reasoning_result_nonempty",
        success_criteria="tests pass",
        verification_type="execution",
        verification_checks=[
            {
                "name": "pytest_green",
                "check_type": "command_success",
                "command": ["python3", "-m", "pytest", str(test_file), "-q"],
                "cwd": str(tmp_path),
                "framework": "pytest",
            }
        ],
        required_conditions=["pytest_green"],
        optional_conditions=[],
    )
    result = VerificationEngine().verify_step(runtime=_RuntimeStub(), state=_state(), plan=_plan(step), step=step, artifacts=VerificationArtifacts())
    assert result.verification_passed is False
    assert "pytest_green" in result.conditions_failed


def test_structural_verification_detects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    step = PlanStep(
        step_id="step_struct",
        title="Check file",
        goal="Check file",
        kind="reasoning",
        expected_tool=None,
        input_text="check",
        expected_output="file exists",
        done_condition="reasoning_result_nonempty",
        success_criteria="file exists",
        verification_type="structural",
        verification_checks=[{"name": "file_exists", "check_type": "file_exists", "path": str(missing)}],
        required_conditions=["file_exists"],
        optional_conditions=[],
    )
    result = VerificationEngine().verify_step(runtime=_RuntimeStub(), state=_state(), plan=_plan(step), step=step, artifacts=VerificationArtifacts())
    assert result.verification_passed is False


def test_structural_verification_supports_schema_and_symbol_checks(tmp_path: Path) -> None:
    module = tmp_path / "sample_module.py"
    module.write_text("VALUE = 4\n\ndef hello():\n    return 'hi'\n", encoding="utf-8")
    step = PlanStep(
        step_id="step_struct",
        title="Check structure",
        goal="Check structure",
        kind="tool",
        expected_tool="echo",
        input_text="check",
        expected_output="structure ok",
        done_condition="tool_result:echo",
        success_criteria="structure is valid",
        verification_type="structural",
        verification_checks=[
            {"name": "module_exists", "check_type": "file_exists", "path": str(module)},
            {"name": "module_contains_function", "check_type": "function_exists", "path": str(module), "function_name": "hello"},
            {"name": "module_contains_symbol", "check_type": "symbol_exists", "path": str(module), "symbol": "VALUE"},
            {
                "name": "output_schema_valid",
                "check_type": "json_schema_valid",
                "actual_source": "tool_output",
                "schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        ],
        required_conditions=["module_exists", "module_contains_function", "module_contains_symbol", "output_schema_valid"],
        optional_conditions=[],
    )
    artifacts = VerificationArtifacts(tool_results=[ToolExecutionResult(tool_name="echo", output={"text": "ok"}, display_text="ok")])
    result = VerificationEngine().verify_step(runtime=_RuntimeStub(), state=_state(), plan=_plan(step), step=step, artifacts=artifacts)
    assert result.verification_passed is True


def test_composite_verification_fails_failed_run_tests_even_without_exit_zero_check() -> None:
    step = PlanStep(
        step_id="step_tests",
        title="Run tests",
        goal="Run tests",
        kind="tool",
        expected_tool="run_tests",
        input_text="run tests",
        expected_output="test output",
        done_condition="tool_result:run_tests",
        success_criteria="tests ran",
        verification_type="composite",
        verification_checks=[
            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
            {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
            {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "run_tests"},
            {"name": "tool_output_nonempty", "check_type": "tool_output_nonempty"},
            {"name": "tool_output_schema_valid", "check_type": "tool_output_schema_valid"},
        ],
        required_conditions=[
            "dependencies_completed",
            "tool_result_present",
            "tool_name_matches",
            "tool_output_nonempty",
            "tool_output_schema_valid",
        ],
        optional_conditions=[],
    )
    artifacts = VerificationArtifacts(
        tool_results=[
            ToolExecutionResult(
                tool_name="run_tests",
                output={"stdout": "", "stderr": "FAILED", "exit_code": 1, "passed": False},
                display_text="failed",
            )
        ]
    )
    result = VerificationEngine().verify_step(runtime=_RuntimeStub(), state=_state(), plan=_plan(step), step=step, artifacts=artifacts)
    assert result.verification_passed is False
    assert "perspective:structural" in result.conditions_failed
    assert result.evidence["perspectives"]["structural"]["passed"] is False


def test_composite_verification_allows_optional_failure(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    data.write_text(json.dumps({"value": 4}), encoding="utf-8")
    step = PlanStep(
        step_id="step_composite",
        title="Check output",
        goal="Check output",
        kind="tool",
        expected_tool="calculator",
        input_text="2 + 2",
        expected_output="4",
        done_condition="tool_result:calculator",
        success_criteria="value is 4",
        verification_type="composite",
        verification_checks=[
            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
            {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
            {"name": "exact_result", "check_type": "exact_match", "actual_source": "tool_output.result", "expected": 4},
            {"name": "missing_file", "check_type": "file_exists", "path": str(tmp_path / "nope.txt")},
        ],
        required_conditions=["dependencies_completed", "tool_result_present", "exact_result"],
        optional_conditions=["missing_file"],
    )
    artifacts = VerificationArtifacts(tool_results=[ToolExecutionResult(tool_name="calculator", output={"result": 4}, display_text="4")])
    result = VerificationEngine().verify_step(runtime=_RuntimeStub(), state=_state(), plan=_plan(step), step=step, artifacts=artifacts)
    assert result.verification_passed is True
    assert "missing_file" in result.conditions_failed


def test_composite_verification_fails_when_required_condition_fails() -> None:
    step = PlanStep(
        step_id="step_value",
        title="Check output",
        goal="Check output",
        kind="tool",
        expected_tool="calculator",
        input_text="2 + 2",
        expected_output="4",
        done_condition="tool_result:calculator",
        success_criteria="value is 4",
        verification_type="composite",
        verification_checks=[
            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
            {"name": "wrong_result", "check_type": "exact_match", "actual_source": "tool_output.result", "expected": 5},
        ],
        required_conditions=["dependencies_completed", "wrong_result"],
        optional_conditions=[],
    )
    artifacts = VerificationArtifacts(tool_results=[ToolExecutionResult(tool_name="calculator", output={"result": 4}, display_text="4")])
    result = VerificationEngine().verify_step(runtime=_RuntimeStub(), state=_state(), plan=_plan(step), step=step, artifacts=artifacts)
    assert result.verification_passed is False
    assert "wrong_result" in result.conditions_failed


def test_tool_files_changed_accepts_edit_text_changed_path(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("old\n", encoding="utf-8")
    step = PlanStep(
        step_id="step_edit",
        title="Patch source",
        goal="Patch source",
        kind="write",
        expected_tool="edit_text",
        input_text="edit sample.py",
        expected_output="sample.py updated",
        done_condition="tool_result:edit_text",
        success_criteria="sample.py updated",
        verification_type="composite",
        verification_checks=[
            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
            {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
            {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
            {"name": "tool_files_changed", "check_type": "tool_files_changed"},
        ],
        required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tool_files_changed"],
        optional_conditions=[],
    )
    artifacts = VerificationArtifacts(
        tool_results=[
            ToolExecutionResult(
                tool_name="edit_text",
                output={"path": str(target), "changed": True, "operation": "replace_pattern_once", "diff": "--- before\n+++ after\n"},
                display_text="edited",
            )
        ]
    )

    result = VerificationEngine().verify_step(runtime=_RuntimeStub(), state=_state(), plan=_plan(step), step=step, artifacts=artifacts)

    assert result.verification_passed is True
    assert "tool_files_changed" in result.conditions_met


def test_value_verification_supports_numeric_tolerance_and_string_match() -> None:
    step = PlanStep(
        step_id="step_value",
        title="Check values",
        goal="Check values",
        kind="tool",
        expected_tool="echo",
        input_text="echo",
        expected_output="value checks",
        done_condition="tool_result:echo",
        success_criteria="values match",
        verification_type="value",
        verification_checks=[
            {"name": "numeric_match", "check_type": "numeric_tolerance", "actual_source": "tool_output.score", "expected": 1.0, "tolerance": 0.01},
            {"name": "string_match", "check_type": "string_match", "actual_source": "tool_output.text", "expected": "done"},
        ],
        required_conditions=["numeric_match", "string_match"],
        optional_conditions=[],
    )
    artifacts = VerificationArtifacts(tool_results=[ToolExecutionResult(tool_name="echo", output={"score": 1.005, "text": "done"}, display_text="done")])
    result = VerificationEngine().verify_step(runtime=_RuntimeStub(), state=_state(), plan=_plan(step), step=step, artifacts=artifacts)
    assert result.verification_passed is True


def test_reviewer_perspective_accepts_exact_literal_expected_output() -> None:
    step = PlanStep(
        step_id="step_answer",
        title="Answer",
        goal="Answer",
        kind="respond",
        expected_tool=None,
        input_text="reply",
        expected_output="written",
        done_condition="assistant_response_nonempty",
        success_criteria="The assistant replies written.",
        verification_type="llm_fallback",
        verification_checks=[
            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
            {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
            {"name": "meets_success_criteria", "check_type": "criterion", "criterion": "The assistant replies written."},
        ],
        required_conditions=["dependencies_completed", "assistant_text_nonempty", "meets_success_criteria"],
        optional_conditions=[],
    )
    result = VerificationEngine().verify_step(
        runtime=_RuntimeStub(
            payload={
                "criteria": [
                    {
                        "name": "meets_success_criteria",
                        "passed": True,
                        "evidence": "candidate exactly matches the expected literal",
                    }
                ],
                "overall_passed": True,
            }
        ),
        state=_state(),
        plan=_plan(step),
        step=step,
        artifacts=VerificationArtifacts(assistant_text="written"),
    )

    assert result.verification_passed is True
    assert "perspective:reviewer" in result.conditions_met


def test_llm_fallback_requires_structured_criteria_results() -> None:
    step = PlanStep(
        step_id="step_answer",
        title="Answer",
        goal="Answer",
        kind="respond",
        expected_tool=None,
        input_text="say hello",
        expected_output="hello",
        done_condition="assistant_response_nonempty",
        success_criteria="say hello",
        verification_type="llm_fallback",
        verification_checks=[
            {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
            {"name": "matches_goal", "check_type": "criterion", "criterion": "reply says hello"},
        ],
        required_conditions=["assistant_text_nonempty", "matches_goal"],
        optional_conditions=[],
    )
    result = VerificationEngine().verify_step(
        runtime=_RuntimeStub(payload={"freeform": "looks good"}),
        state=_state(),
        plan=_plan(step),
        step=step,
        artifacts=VerificationArtifacts(assistant_text="hello"),
    )
    assert result.verification_passed is False
    assert "matches_goal" in result.conditions_failed


def test_reviewer_perspective_accepts_expected_semantic_output() -> None:
    step = PlanStep(
        step_id="step_answer",
        title="Answer",
        goal="Return the owner",
        kind="respond",
        expected_tool=None,
        input_text="respond",
        expected_output="owner=carol",
        done_condition="assistant_response_nonempty",
        success_criteria="return owner=carol exactly",
        verification_type="value",
        verification_checks=[
            {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
        ],
        required_conditions=["assistant_text_nonempty"],
        optional_conditions=[],
    )
    result = VerificationEngine().verify_step(
        runtime=_RuntimeStub(),
        state=_state(),
        plan=_plan(step),
        step=step,
        artifacts=VerificationArtifacts(assistant_text="owner=carol"),
    )
    assert result.verification_passed is True
    assert "perspective:reviewer" in result.conditions_met


def test_llm_fallback_reviewer_perspective_is_advisory_when_explicit_criteria_pass() -> None:
    step = PlanStep(
        step_id="step_answer",
        title="Answer",
        goal="Produce the final response",
        kind="respond",
        expected_tool=None,
        input_text="reply",
        expected_output="Final assistant response",
        done_condition="assistant_response_nonempty",
        success_criteria="The user receives a complete direct answer.",
        verification_type="llm_fallback",
        verification_checks=[
            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
            {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
            {"name": "meets_success_criteria", "check_type": "criterion", "criterion": "The user receives a complete direct answer."},
            {"name": "satisfies_done_condition", "check_type": "criterion", "criterion": "assistant_response_nonempty"},
        ],
        required_conditions=[
            "dependencies_completed",
            "assistant_text_nonempty",
            "meets_success_criteria",
            "satisfies_done_condition",
        ],
        optional_conditions=[],
    )
    result = VerificationEngine().verify_step(
        runtime=_RuntimeStub(
            payload={
                "criteria": [
                    {"name": "meets_success_criteria", "passed": True, "evidence": "17"},
                    {"name": "satisfies_done_condition", "passed": True, "evidence": "17"},
                ]
            }
        ),
        state=_state(),
        plan=_plan(step),
        step=step,
        artifacts=VerificationArtifacts(assistant_text="17"),
    )

    assert result.verification_passed is True
    assert "meets_success_criteria" in result.conditions_met
    assert "satisfies_done_condition" in result.conditions_met
    assert "perspective:reviewer" not in result.conditions_failed


def test_llm_fallback_fails_when_criterion_missing() -> None:
    step = PlanStep(
        step_id="step_answer",
        title="Answer",
        goal="Answer",
        kind="respond",
        expected_tool=None,
        input_text="say hello",
        expected_output="hello",
        done_condition="assistant_response_nonempty",
        success_criteria="say hello",
        verification_type="llm_fallback",
        verification_checks=[
            {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
            {"name": "matches_goal", "check_type": "criterion", "criterion": "reply says hello"},
        ],
        required_conditions=["assistant_text_nonempty", "matches_goal"],
        optional_conditions=[],
    )
    result = VerificationEngine().verify_step(
        runtime=_RuntimeStub(payload={"criteria": []}),
        state=_state(),
        plan=_plan(step),
        step=step,
        artifacts=VerificationArtifacts(assistant_text="hello"),
    )
    assert result.verification_passed is False
    assert "matches_goal" in result.conditions_failed


def test_benchmark_verification_validates_coding_task_outputs(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def value():\n    return 42\n", encoding="utf-8")
    test_file = tmp_path / "test_module.py"
    test_file.write_text(
        "import unittest\n\nfrom module import value\n\n\nclass ModuleTests(unittest.TestCase):\n    def test_value(self) -> None:\n        self.assertEqual(value(), 42)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )
    contract = type(
        "Contract",
        (),
        {
            "task_type": "coding",
            "expected_answer": "implemented",
            "expected_files": {str(module): "def value():\n    return 42\n"},
            "command": ["python3", "-m", "unittest", "-q", "test_module.py"],
            "command_cwd": str(tmp_path),
            "command_framework": "unittest",
            "required_history_events": ["verification_passed"],
        },
    )()
    state = _state()
    state.metrics.last_reasoning_reason = "answered"
    events = [
        HistoryEvent(id="1", sequence=1, session_id="session", timestamp="t1", type="verification_passed", version=1, payload={"step_id": "x"}),
    ]

    report = verify_benchmark_contract(contract, assistant_text="implemented", state=state, events=events)

    assert isinstance(report, BenchmarkVerificationReport)
    assert report.passed is True
    assert report.checks["command"] is True


def test_benchmark_verification_detects_failure_contract_state() -> None:
    contract = type(
        "Contract",
        (),
        {
            "task_type": "failure",
            "required_history_events": ["tool_graph_rejected", "verification_failed"],
        },
    )()
    state = _state()
    state.metrics.steps_failed = 1
    state.metrics.last_reasoning_reason = "no_progress_possible"
    events = [
        HistoryEvent(id="1", sequence=1, session_id="session", timestamp="t1", type="tool_graph_rejected", version=1, payload={"selected_tool": "calculator"}),
        HistoryEvent(id="2", sequence=2, session_id="session", timestamp="t2", type="verification_failed", version=1, payload={"step_id": "x"}),
    ]

    report = verify_benchmark_contract(contract, assistant_text="", state=state, events=events)

    assert report.passed is True
    assert report.checks["failure_signals"] is True


def test_benchmark_verification_enforces_tool_usage_and_workspace_scope(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("alpha\nbeta\n", encoding="utf-8")
    backup = tmp_path / "document.txt.bak"
    backup.write_text("alpha\nbeta\n", encoding="utf-8")
    contract = type(
        "Contract",
        (),
        {
            "task_type": "file_edit",
            "expected_answer": "updated",
            "expected_files": {str(source): "alpha\ngamma\n"},
            "required_tools_used": ["edit_text"],
            "forbidden_tools_used": ["calculator"],
            "allowed_modified_files": [str(source)],
            "required_event_counts": {"tool_called": 1},
        },
    )()
    state = _state()
    state.metrics.last_reasoning_reason = "answered"
    source.write_text("alpha\ngamma\n", encoding="utf-8")
    events = [
        HistoryEvent(id="1", sequence=1, session_id="session", timestamp="t1", type="tool_called", version=1, payload={"tool_name": "edit_text", "tool_input": {"path": str(source)}}),
        HistoryEvent(id="2", sequence=2, session_id="session", timestamp="t2", type="tool_result", version=1, payload={"tool_name": "edit_text", "raw_input": {}, "validated_input": {}, "output": {"changed": True}}),
    ]

    report = verify_benchmark_contract(
        contract,
        assistant_text="updated",
        state=state,
        events=events,
        workspace_before={"document.txt": "alpha\nbeta\n"},
        workspace_after={"document.txt": "alpha\ngamma\n", "document.txt.bak": "alpha\nbeta\n"},
    )

    assert report.passed is True
    assert report.checks["allowed_modified_files"] is True
    assert report.checks["required_tools_used"] is True


def test_tool_name_verification_treats_read_tools_as_equivalent() -> None:
    step = PlanStep(
        step_id="read_source",
        title="Read source",
        goal="Read source",
        kind="read",
        expected_tool="read_file",
        input_text="read source",
        expected_output="source text",
        done_condition="tool_result:read_file",
        success_criteria="source is read",
        verification_type="composite",
        verification_checks=[
            {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
            {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "read_file"},
            {"name": "tool_output_nonempty", "check_type": "tool_output_nonempty"},
            {"name": "tool_output_schema_valid", "check_type": "tool_output_schema_valid"},
        ],
        required_conditions=["tool_result_present", "tool_name_matches", "tool_output_nonempty", "tool_output_schema_valid"],
        optional_conditions=[],
    )
    artifacts = VerificationArtifacts(
        tool_results=[
            ToolExecutionResult(
                tool_name="read_text",
                output={"text": "content", "source_ref": "source.py", "reader_id": "reader", "source_kind": "file", "start_offset": 0, "end_offset": 7, "next_offset": 7, "finished": True},
                display_text="content",
            )
        ]
    )
    result = VerificationEngine().verify_step(runtime=_RuntimeStub(), state=_state(), plan=_plan(step), step=step, artifacts=artifacts)
    assert result.verification_passed is True


def test_coding_contract_accepts_alternate_implementation_when_tests_pass(tmp_path: Path) -> None:
    package = tmp_path / "pkg_alt"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = package / "stats.py"
    source.write_text("def moving_total(values: list[int]) -> int:\n    return sum(values)\n", encoding="utf-8")
    test_file = tmp_path / "test_pkg_alt.py"
    test_file.write_text(
        "import unittest\n\nfrom pkg_alt.stats import moving_total\n\n\n"
        "class StatsTests(unittest.TestCase):\n"
        "    def test_moving_total(self):\n        self.assertEqual(moving_total([7, 7, 15]), 29)\n",
        encoding="utf-8",
    )
    contract = type(
        "Contract",
        (),
        {
            "task_type": "coding",
            "expected_file_patterns": {str(source): ["for value in values:", "return total"]},
            "command": ["python3", "-m", "unittest", "-q", test_file.name],
            "command_cwd": str(tmp_path),
            "command_framework": "unittest",
            "required_history_events": ["reasoning_completed"],
            "allowed_modified_files": [str(source)],
            "forbid_unexpected_workspace_changes": True,
        },
    )()
    state = _state()
    state.metrics.last_reasoning_reason = "answered"
    events = [HistoryEvent(id="1", sequence=1, session_id="session", timestamp="t", type="reasoning_completed", version=1, payload={})]

    report = verify_benchmark_contract(
        contract,
        assistant_text="Fixed and tests pass.",
        state=state,
        events=events,
        workspace_before={str(source): "def moving_total(values):\n    return 0\n"},
        workspace_after={str(source): source.read_text(encoding="utf-8")},
    )

    assert report.passed is True
    assert report.checks["command"] is True
    assert report.checks["expected_file_patterns"] is True
    assert report.evidence["expected_file_patterns"]["advisory_for_coding"] is True


def test_benchmark_contract_accepts_live_semantic_backend_configuration() -> None:
    contract = type("Contract", (), {"task_type": "reading", "expected_answer_contains": ["ok"]})()
    state = _state()
    report = verify_benchmark_contract(
        contract,
        assistant_text="ok",
        state=state,
        events=[],
        semantic_backend_mode="degraded_lexical",
        semantic_base_url=None,
        semantic_seed=123,
        semantic_connect_timeout_seconds=1,
        semantic_read_timeout_seconds=1,
    )
    assert report.passed is True
