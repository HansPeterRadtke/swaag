from __future__ import annotations

import json

import pytest

from swaag.planner import (
    PlanValidationError,
    plan_from_payload,
    ready_steps,
    transition_step,
)
from swaag.runtime import AgentRuntime, FatalSemanticEngineError

from tests.helpers import FakeModelClient, plan_response, plan_step


def _payload(goal: str, steps: list[dict]) -> dict:
    return json.loads(plan_response(goal=goal, steps=steps))


def test_plan_from_payload_validates_ordered_steps(make_config) -> None:
    payload = _payload(
        "Read a file, update it, then answer.",
        [
            plan_step("step_read", "Read the file", "read", expected_tool="read_text", expected_output="File contents", success_criteria="The file is read."),
            plan_step("step_write", "Edit the file", "write", expected_tool="edit_text", expected_output="Updated file", success_criteria="The file is updated.", depends_on=["step_read"]),
            plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user receives the result.", depends_on=["step_write"]),
        ],
    )

    plan = plan_from_payload(payload, available_tools=["read_text", "edit_text", "notes", "calculator"])

    assert [step.expected_tool for step in plan.steps[:-1]] == ["read_text", "edit_text"]
    assert plan.steps[-1].kind == "respond"
    assert plan.steps[-1].depends_on == ["step_write"]
    with pytest.raises(PlanValidationError):
        transition_step(plan, "step_write", "running")


def test_plan_from_payload_preserves_model_declared_steps_without_semantic_repair() -> None:
    payload = _payload(
        "Fix pkg_261/slugify.py and run tests.",
        [
            plan_step("read", "Read pkg_261/slugify.py", "read", expected_tool="read_text", expected_output="source", success_criteria="read"),
            plan_step("edit", "Edit pkg_261/slugify.py", "write", expected_tool="edit_text", expected_output="edited", success_criteria="edited", depends_on=["read"]),
            plan_step("write", "Write pkg_261/slugify.py", "write", expected_tool="write_file", expected_output="written", success_criteria="written", depends_on=["edit"]),
            plan_step("test", "Test pkg_261/slugify.py", "tool", expected_tool="run_tests", expected_output="tests pass", success_criteria="tests pass", depends_on=["write"]),
            plan_step("answer", "Answer", "respond", expected_output="done", success_criteria="done", depends_on=["test"]),
        ],
    )

    plan = plan_from_payload(payload, available_tools=["read_text", "edit_text", "write_file", "run_tests"])

    assert [step.expected_tool for step in plan.steps[:-1]] == ["read_text", "edit_text", "write_file", "run_tests"]
    test_step = next(step for step in plan.steps if step.step_id == "test")
    assert test_step.depends_on == ["write"]


def test_plan_from_payload_keeps_placeholder_like_input_text_as_instruction() -> None:
    payload = _payload(
        "Edit a file and answer.",
        [
            plan_step(
                "edit",
                "Edit file",
                "write",
                expected_tool="edit_text",
                input_text=json.dumps({"instruction": "Use prior content {{edited_file_content}} only as context."}),
                expected_output="edited_file_content",
                success_criteria="file edited",
                output_refs=["edited_file_content"],
            ),
            plan_step("answer", "Answer", "respond", expected_output="done", success_criteria="done", depends_on=["edit"]),
        ],
    )

    plan = plan_from_payload(payload, available_tools=["edit_text"])

    edit_step = next(step for step in plan.steps if step.step_id == "edit")
    assert "{{edited_file_content}}" in edit_step.input_text


def test_plan_from_payload_rejects_invalid_tool() -> None:
    payload = _payload(
        "Try an invalid tool.",
        [
            plan_step("step_bad", "Run missing tool", "tool", expected_tool="missing_tool", expected_output="x", success_criteria="x"),
            plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="done", depends_on=["step_bad"]),
        ],
    )

    with pytest.raises(PlanValidationError):
        plan_from_payload(payload, available_tools=["calculator"])


def test_plan_from_payload_rejects_missing_terminal_response_step() -> None:
    payload = _payload(
        "Inspect a file and edit it.",
        [
            plan_step("step_read", "Read the file", "read", expected_tool="read_text", expected_output="File contents", success_criteria="The file is read."),
            plan_step("step_edit", "Edit the file", "write", expected_tool="edit_text", expected_output="Edited file", success_criteria="The file is edited.", depends_on=["step_read"]),
        ],
    )

    with pytest.raises(PlanValidationError, match="end with a respond step"):
        plan_from_payload(payload, available_tools=["read_text", "edit_text"])


def test_plan_from_payload_rejects_missing_verification_contract() -> None:
    payload = {
        "goal": "Do something",
        "success_criteria": "done",
        "fallback_strategy": "replan",
        "steps": [
            {
                "step_id": "step_1",
                "title": "Compute",
                "goal": "Compute",
                "kind": "tool",
                "expected_tool": "calculator",
                "input_text": "2 + 2",
                "expected_output": "4",
                "done_condition": "tool_result:calculator",
                "success_criteria": "tool returns 4",
                "input_refs": [],
                "output_refs": [],
                "fallback_strategy": "retry",
                "depends_on": [],
            },
            {
                "step_id": "step_2",
                "title": "Answer",
                "goal": "Answer",
                "kind": "respond",
                "expected_tool": "",
                "input_text": "Answer",
                "expected_output": "done",
                "done_condition": "assistant_response_nonempty",
                "success_criteria": "done",
                "depends_on": ["step_1"],
            },
        ],
    }

    with pytest.raises(PlanValidationError, match="must declare expected_outputs"):
        plan_from_payload(payload, available_tools=["calculator"])


def test_plan_from_payload_rejects_sparse_step_fields_and_condition_refs() -> None:
    payload = {
        "goal": "Reply with exactly 17.",
        "success_criteria": "The user receives exactly 17.",
        "fallback_strategy": "Replan safely.",
        "steps": [
            {
                "step_id": "step_calc",
                "title": "Compute final value",
                "goal": "",
                "kind": "tool",
                "expected_tool": "calculator",
                "input_text": "",
                "expected_output": "",
                "expected_outputs": [],
                "done_condition": "",
                "success_criteria": "",
                "verification_type": "composite",
                "verification_checks": [
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "calculator"},
                ],
                "required_conditions": ["unknown"],
                "optional_conditions": ["tool_result_present", "unknown_check"],
                "input_refs": [],
                "output_refs": ["calculator"],
                "fallback_strategy": "",
                "depends_on": [],
            },
            {
                "step_id": "step_answer",
                "title": "Answer the user",
                "goal": "",
                "kind": "respond",
                "expected_tool": "",
                "input_text": "",
                "expected_output": "",
                "expected_outputs": [],
                "done_condition": "",
                "success_criteria": "",
                "verification_type": "llm_fallback",
                "verification_checks": [{"name": "", "check_type": ""}],
                "required_conditions": ["nonsense"],
                "optional_conditions": [],
                "input_refs": ["calculator"],
                "output_refs": [],
                "fallback_strategy": "",
                "depends_on": ["step_calc"],
            },
        ],
    }

    with pytest.raises(PlanValidationError, match="missing required model fields"):
        plan_from_payload(payload, available_tools=["calculator"])


@pytest.mark.parametrize(
    ("check", "message"),
    [
        ({"name": "schema", "check_type": "json_schema_valid", "actual_source": "tool_output", "schema_json": ""}, "non-empty schema_json"),
        ({"name": "function", "check_type": "function_exists", "path": "module.py", "function_name": ""}, "non-empty function_name"),
        ({"name": "symbol", "check_type": "symbol_exists", "path": "module.py", "symbol": ""}, "non-empty symbol"),
        ({"name": "value", "check_type": "exact_match", "actual_source": "tool_output.result", "expected": ""}, "non-empty expected"),
        ({"name": "number", "check_type": "numeric_tolerance", "actual_source": "tool_output.result", "expected": "not-a-number", "tolerance": 0.1}, "expected must be numeric text"),
    ],
)
def test_plan_from_payload_rejects_incomplete_verification_check_payloads(check, message) -> None:
    payload = _payload(
        "Read and verify.",
        [
            plan_step(
                "step_read",
                "Read",
                "read",
                expected_tool="read_file",
                expected_output="contents",
                success_criteria="verified",
                verification_checks=[check],
                required_conditions=[check["name"]],
                optional_conditions=[],
            ),
            plan_step("answer", "Answer", "respond", expected_output="done", success_criteria="done", depends_on=["step_read"]),
        ],
    )
    with pytest.raises(PlanValidationError, match=message):
        plan_from_payload(payload, available_tools=["read_file"])


def test_plan_from_payload_rejects_tool_name_equals_without_expected_tool() -> None:
    payload = _payload(
        "Read a file and answer.",
        [
            plan_step(
                "step_read",
                "Read the file",
                "read",
                expected_tool="read_text",
                expected_output="contents",
                success_criteria="The file is read.",
                verification_checks=[
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals"},
                ],
                required_conditions=["tool_result_present", "tool_name_matches"],
                optional_conditions=[],
            ),
            plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="done", depends_on=["step_read"]),
        ],
    )

    with pytest.raises(PlanValidationError, match="must declare a non-empty expected tool name"):
        plan_from_payload(payload, available_tools=["read_text"])


def test_plan_from_payload_rejects_tool_name_equals_that_contradicts_step_tool() -> None:
    payload = _payload(
        "Read a file and answer.",
        [
            plan_step(
                "step_read",
                "Read the file",
                "read",
                expected_tool="read_text",
                expected_output="contents",
                success_criteria="The file is read.",
                verification_checks=[
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                ],
                required_conditions=["tool_result_present", "tool_name_matches"],
                optional_conditions=[],
            ),
            plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="done", depends_on=["step_read"]),
        ],
    )

    with pytest.raises(PlanValidationError, match="but the step declares expected_tool='read_text'"):
        plan_from_payload(payload, available_tools=["read_text", "edit_text"])


def test_plan_from_payload_rejects_tool_llm_fallback_verification() -> None:
    payload = _payload(
        "Edit the file and answer.",
        [
            plan_step(
                "step_write",
                "Write the file",
                "write",
                expected_tool="write_file",
                expected_output="updated",
                success_criteria="file updated",
                verification_type="llm_fallback",
                verification_checks=[{"name": "output_matches", "check_type": "equals", "expected": "updated"}],
                required_conditions=["output_matches"],
            ),
            plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="reply done", depends_on=["step_write"]),
        ],
    )

    with pytest.raises(PlanValidationError, match="must use composite verification"):
        plan_from_payload(payload, available_tools=["write_file"])


def test_plan_from_payload_rejects_planned_llm_fallback_verification() -> None:
    payload = _payload(
        "Answer with the final state.",
        [
            plan_step(
                "step_answer",
                "Answer",
                "respond",
                expected_output="done",
                success_criteria="The answer describes the final state.",
                verification_type="llm_fallback",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
                    {
                        "name": "matches_goal",
                        "check_type": "criterion",
                        "actual_source": "assistant_text",
                        "criterion": "The answer describes the final state.",
                    },
                ],
                required_conditions=["dependencies_completed", "assistant_text_nonempty"],
                optional_conditions=["matches_goal"],
            ),
        ],
    )

    with pytest.raises(PlanValidationError, match="must use composite verification"):
        plan_from_payload(payload, available_tools=["calculator"])


def test_plan_from_payload_accepts_declared_response_criterion_for_runtime_repair() -> None:
    payload = _payload(
        "Answer with the final state.",
        [
            plan_step(
                "step_answer",
                "Answer",
                "respond",
                expected_output="done",
                success_criteria="The answer describes the final state.",
                verification_type="composite",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
                    {
                        "name": "matches_goal",
                        "check_type": "criterion",
                        "actual_source": "assistant_text",
                        "criterion": "The answer describes the final state.",
                    },
                ],
                required_conditions=["dependencies_completed", "assistant_text_nonempty", "matches_goal"],
                optional_conditions=[],
            ),
        ],
    )

    plan = plan_from_payload(payload, available_tools=["calculator"])

    assert plan.steps[0].required_conditions == ["dependencies_completed", "assistant_text_nonempty", "matches_goal"]
    assert plan.steps[0].optional_conditions == []


def test_plan_from_payload_accepts_response_with_intrinsic_success_criteria_semantics() -> None:
    payload = _payload(
        "Answer with the final state.",
        [
            plan_step(
                "step_answer",
                "Answer",
                "respond",
                expected_output="done",
                success_criteria="The answer describes the final state.",
                verification_type="composite",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
                ],
                required_conditions=["dependencies_completed", "assistant_text_nonempty"],
                optional_conditions=[],
            ),
        ],
    )

    plan = plan_from_payload(payload, available_tools=["calculator"])

    assert plan.steps[0].success_criteria == "The answer describes the final state."
    assert plan.steps[0].required_conditions == ["dependencies_completed", "assistant_text_nonempty"]


def test_plan_from_payload_accepts_optional_additional_response_semantic_check() -> None:
    payload = _payload(
        "Answer with the final state.",
        [
            plan_step(
                "step_answer",
                "Answer",
                "respond",
                expected_output="done",
                success_criteria="The answer describes the final state.",
                verification_type="composite",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
                    {
                        "name": "matches_goal",
                        "check_type": "criterion",
                        "actual_source": "assistant_text",
                        "criterion": "The answer describes the final state.",
                    },
                ],
                required_conditions=["dependencies_completed", "assistant_text_nonempty"],
                optional_conditions=["matches_goal"],
            ),
        ],
    )

    plan = plan_from_payload(payload, available_tools=["calculator"])

    assert plan.steps[0].required_conditions == ["dependencies_completed", "assistant_text_nonempty"]
    assert plan.steps[0].optional_conditions == ["matches_goal"]


def test_plan_from_payload_rejects_assistant_response_actual_source_for_response_verification() -> None:
    payload = _payload(
        "Answer with the final state.",
        [
            plan_step(
                "step_answer",
                "Answer",
                "respond",
                expected_output="done",
                success_criteria="The answer describes the final state.",
                verification_type="composite",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {
                        "name": "answer_exact",
                        "check_type": "exact_match",
                        "actual_source": "assistant_response",
                        "expected_json": '"done"',
                    },
                ],
                required_conditions=["dependencies_completed", "answer_exact"],
                optional_conditions=[],
            ),
        ],
    )

    with pytest.raises(PlanValidationError, match="actual_source='assistant_text'"):
        plan_from_payload(payload, available_tools=["calculator"])


def test_plan_from_payload_rejects_file_contains_without_textual_target() -> None:
    payload = _payload(
        "Write a report and answer.",
        [
            plan_step(
                "write_report",
                "Write report",
                "write",
                expected_tool="write_file",
                expected_output="report written",
                success_criteria="The report contains the model-selected final content.",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {
                        "name": "file_written",
                        "check_type": "file_contains",
                        "expected_json": '{"path":"capacity_report.txt","content":"computed report","create":true}',
                    },
                ],
                required_conditions=["dependencies_completed", "file_written"],
                optional_conditions=[],
            ),
            plan_step("answer", "Answer", "respond", expected_output="done", success_criteria="reply done", depends_on=["write_report"]),
        ],
    )

    with pytest.raises(PlanValidationError, match="file_contains check file_written expected_json must decode"):
        plan_from_payload(payload, available_tools=["write_file"])


def test_plan_from_payload_rejects_bare_text_file_contains_expected_json() -> None:
    payload = _payload(
        "Write a report and answer.",
        [
            plan_step(
                "write_report",
                "Write report",
                "write",
                expected_tool="write_file",
                expected_output="report written",
                success_criteria="The report contains the model-selected final content.",
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "file_written", "check_type": "file_contains", "expected_json": "ready"},
                ],
                required_conditions=["dependencies_completed", "file_written"],
                optional_conditions=[],
            ),
            plan_step("answer", "Answer", "respond", expected_output="done", success_criteria="reply done", depends_on=["write_report"]),
        ],
    )

    with pytest.raises(PlanValidationError, match="expected_json must be JSON text"):
        plan_from_payload(payload, available_tools=["write_file"])


def test_plan_from_payload_maps_dependency_artifact_conditions_mechanically() -> None:
    payload = _payload(
        "Read, edit, answer.",
        [
            plan_step(
                "step_read",
                "Read",
                "read",
                expected_tool="read_file",
                expected_output="file_content",
                success_criteria="file read",
                verification_checks=[{"name": "file_content", "check_type": "tool_output_nonempty"}],
                required_conditions=["file_content"],
                optional_conditions=[],
            ),
            plan_step(
                "step_edit",
                "Edit",
                "write",
                expected_tool="edit_text",
                expected_output="edited_file_content",
                success_criteria="file edited",
                depends_on=["step_read"],
                input_refs=["file_content"],
                verification_checks=[{"name": "edited_file_content", "check_type": "tool_output_nonempty"}],
                required_conditions=["file_content"],
                optional_conditions=[],
            ),
            plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="reply done", depends_on=["step_edit"]),
        ],
    )

    plan = plan_from_payload(payload, available_tools=["read_file", "edit_text"])
    edit_step = next(step for step in plan.steps if step.step_id == "step_edit")

    assert "dependencies_completed" in edit_step.required_conditions
    assert "dependencies_completed" in {check["name"] for check in edit_step.verification_checks}
    assert "file_content" not in edit_step.required_conditions


def test_plan_from_payload_rejects_empty_required_conditions_without_repair() -> None:
    payload = _payload(
        "Read and answer.",
        [
            plan_step(
                "step_read",
                "Read",
                "read",
                expected_tool="read_file",
                expected_output="file_content",
                success_criteria="file read",
                verification_checks=[
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "output_nonempty", "check_type": "tool_output_nonempty"},
                ],
                required_conditions=[],
                optional_conditions=[],
            ),
            plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="reply done", depends_on=["step_read"]),
        ],
    )

    with pytest.raises(PlanValidationError, match="must declare required_conditions"):
        plan_from_payload(payload, available_tools=["read_file"])


def test_plan_from_payload_adds_explicit_dependency_condition_check() -> None:
    payload = _payload(
        "Read, edit, answer.",
        [
            plan_step(
                "step_read",
                "Read",
                "read",
                expected_tool="read_file",
                expected_output="file_content",
                success_criteria="file read",
            ),
            plan_step(
                "step_edit",
                "Edit",
                "write",
                expected_tool="edit_text",
                expected_output="edited_file_content",
                success_criteria="file edited",
                depends_on=["step_read"],
                verification_checks=[{"name": "edited_file_content", "check_type": "tool_output_nonempty"}],
                required_conditions=["dependencies_completed", "edited_file_content"],
                optional_conditions=[],
            ),
            plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="reply done", depends_on=["step_edit"]),
        ],
    )

    plan = plan_from_payload(payload, available_tools=["read_file", "edit_text"])
    edit_step = next(step for step in plan.steps if step.step_id == "step_edit")

    assert edit_step.required_conditions == ["dependencies_completed", "edited_file_content"]
    assert "dependencies_completed" in {check["name"] for check in edit_step.verification_checks}


def test_ready_steps_returns_all_parallel_ready_nodes() -> None:
    payload = _payload(
        "Read and note before answering.",
        [
            plan_step("step_read", "Read", "read", expected_tool="read_text", expected_output="text", success_criteria="read"),
            plan_step("step_note", "Take note", "note", expected_tool="notes", expected_output="note", success_criteria="note"),
            plan_step("step_answer", "Answer", "respond", expected_output="answer", success_criteria="answer", depends_on=["step_read", "step_note"]),
        ],
    )
    plan = plan_from_payload(payload, available_tools=["read_text", "notes", "calculator"])

    assert [step.step_id for step in ready_steps(plan)] == ["step_read", "step_note"]


def test_plan_from_payload_topologically_sorts_out_of_order_dag_steps() -> None:
    payload = _payload(
        "Read and answer.",
        [
            plan_step("step_answer", "Answer", "respond", expected_output="answer", success_criteria="answer", depends_on=["step_read"], input_refs=["read_text"]),
            plan_step("step_read", "Read", "read", expected_tool="read_text", expected_output="text", success_criteria="read", output_refs=["read_text"]),
        ],
    )

    plan = plan_from_payload(payload, available_tools=["read_text", "notes", "calculator"])

    assert [step.step_id for step in plan.steps] == ["step_read", "step_answer"]
    assert plan.current_step_id == "step_read"


def test_runtime_creates_plan_before_tool_execution(make_config) -> None:
    config = make_config(runtime__max_tool_steps=3)
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the value", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the value.", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "4",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    plan_created = next(event for event in events if event.event_type == "plan_created")
    tool_called = next(event for event in events if event.event_type == "tool_called")

    assert result.assistant_text == "4"
    assert plan_created.sequence < tool_called.sequence


def test_replay_restores_completed_plan(make_config) -> None:
    config = make_config(runtime__max_tool_steps=3)
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the value", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the value.", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "4",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    rebuilt = runtime.history.rebuild_from_history(result.session_id)

    assert rebuilt.active_plan is not None
    assert rebuilt.active_plan.status == "completed"
    assert goal in rebuilt.active_plan.goal


def test_runtime_rejects_malformed_plan_and_records_fatal_plan_error(make_config) -> None:
    config = make_config(model__max_retries=0, planner__max_replans=0)
    goal = "Read sample.txt and then reply exactly done."
    fake_client = FakeModelClient(
        responses=[
            json.dumps({"goal": goal, "success_criteria": "x", "fallback_strategy": "y", "steps": []}),
            json.dumps({"goal": goal, "success_criteria": "x", "fallback_strategy": "y", "steps": []}),
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    with pytest.raises(FatalSemanticEngineError):
        runtime.run_turn(goal)

    session_id = next(path.name for path in runtime.history.root.iterdir() if path.is_dir())
    events = runtime.history.read_history(session_id)
    assert any(event.event_type == "fatal_system_error" for event in events)
    assert any(
        event.event_type == "reasoning_completed"
        and event.payload["status"] == "fatal_system_error"
        and event.payload["reason"] == "plan_generation_failed"
        for event in events
    )


def test_plan_parser_derives_internal_condition_lists_from_local_check_status() -> None:
    payload = {
        "goal": "edit and verify",
        "success_criteria": "the edit is verified",
        "fallback_strategy": "replan",
        "steps": [
            {
                "step_id": "edit",
                "title": "Edit file",
                "goal": "persist the edit",
                "kind": "write",
                "expected_tool": "edit_text",
                "input_text": "edit the target",
                "expected_output": "edited file",
                "expected_outputs": ["edited file"],
                "done_condition": "tool_result:edit_text",
                "success_criteria": "the edit is persisted",
                "verification_type": "composite",
                "verification_checks": [
                    {"name": "dependencies_completed", "check_type": "dependencies_completed", "condition": "optional"},
                    {"name": "tool_effect", "check_type": "tool_effect_verified", "condition": "required"},
                ],
                "input_refs": [],
                "output_refs": [],
                "fallback_strategy": "replan",
                "depends_on": [],
            },
            {
                "step_id": "answer",
                "title": "Answer",
                "goal": "report completion",
                "kind": "respond",
                "expected_tool": "",
                "input_text": "answer from the verified result",
                "expected_output": "completion response",
                "expected_outputs": ["completion response"],
                "done_condition": "assistant_response_nonempty",
                "success_criteria": "the response reports completion",
                "verification_type": "composite",
                "verification_checks": [
                    {
                        "name": "answer_nonempty",
                        "check_type": "string_nonempty",
                        "condition": "required",
                        "actual_source": "assistant_text",
                    }
                ],
                "input_refs": [],
                "output_refs": [],
                "fallback_strategy": "replan",
                "depends_on": ["edit"],
            },
        ],
    }
    plan = plan_from_payload(payload, available_tools=["edit_text"])
    assert plan.steps[0].required_conditions == ["tool_effect"]
    assert plan.steps[0].optional_conditions == ["dependencies_completed"]
    assert all("condition" not in check for check in plan.steps[0].verification_checks)


def test_plan_parser_derives_done_condition_when_live_wire_omits_it() -> None:
    payload = {
        "goal": "read then answer",
        "success_criteria": "answered",
        "fallback_strategy": "replan",
        "steps": [
            {
                "step_id": "read",
                "title": "Read",
                "goal": "read",
                "kind": "read",
                "expected_tool": "read_file",
                "input_text": "read the file",
                "expected_output": "contents",
                "expected_outputs": ["contents"],
                "success_criteria": "contents returned",
                "verification_type": "composite",
                "objective_verification_check": _none_objective_check(),
                "verification_checks": [
                    {"name": "tool", "check_type": "tool_name_equals", "condition": "required", "expected": "read_file"}
                ],
                "input_refs": [],
                "output_refs": ["contents"],
                "fallback_strategy": "replan",
                "depends_on": [],
            },
            {
                "step_id": "answer",
                "title": "Answer",
                "goal": "answer",
                "kind": "respond",
                "expected_tool": "",
                "input_text": "answer",
                "expected_output": "response",
                "expected_outputs": ["response"],
                "success_criteria": "response supplied",
                "verification_type": "composite",
                "objective_verification_check": _none_objective_check(),
                "verification_checks": [
                    {"name": "answer", "check_type": "string_nonempty", "condition": "required", "actual_source": "assistant_text"}
                ],
                "input_refs": ["contents"],
                "output_refs": ["response"],
                "fallback_strategy": "replan",
                "depends_on": ["read"],
            },
        ],
    }
    plan = plan_from_payload(payload, available_tools=["read_file"])
    assert plan.steps[0].done_condition == "tool_result:read_file"
    assert plan.steps[1].done_condition == "assistant_response_nonempty"


def _none_objective_check() -> dict[str, object]:
    return {
        "name": "",
        "check_type": "none",
        "path": "",
        "pattern": "",
        "command": [],
        "cwd": "",
    }


def _duplicate_objective_payload() -> dict:
    return _payload(
        "Read a file and answer.",
        [
            plan_step(
                "read",
                "Read file",
                "read",
                expected_tool="read_file",
                expected_output="contents",
                success_criteria="contents returned",
                verification_checks=[
                    {
                        "name": "tool_name",
                        "check_type": "tool_name_equals",
                        "condition": "required",
                        "expected": "read_file",
                    },
                    {
                        "name": "effect",
                        "check_type": "tool_effect_verified",
                        "condition": "required",
                    },
                ],
                required_conditions=["tool_name", "effect"],
                optional_conditions=[],
            ),
            plan_step(
                "answer",
                "Answer",
                "respond",
                expected_output="response",
                success_criteria="response supplied",
                depends_on=["read"],
                verification_checks=[
                    {
                        "name": "answer_nonempty",
                        "check_type": "string_nonempty",
                        "condition": "required",
                        "actual_source": "assistant_text",
                    }
                ],
                required_conditions=["answer_nonempty"],
                optional_conditions=[],
            ),
        ],
    )


def test_plan_parser_collapses_exact_objective_check_duplicate() -> None:
    payload = _duplicate_objective_payload()
    step = payload["steps"][0]
    step["objective_verification_check"] = {
        "name": "effect",
        "check_type": "tool_effect_verified",
        "path": "",
        "pattern": "",
        "command": [],
        "cwd": "",
    }
    step["verification_checks"].append(
        {"name": "effect", "check_type": "tool_effect_verified", "condition": "required"}
    )
    plan = plan_from_payload(payload, available_tools=["read_file"])
    checks = [check for check in plan.steps[0].verification_checks if check["name"] == "effect"]
    assert len(checks) == 1
    assert plan.steps[0].required_conditions.count("effect") == 1


def test_plan_parser_rejects_conflicting_objective_check_duplicate() -> None:
    payload = _duplicate_objective_payload()
    step = payload["steps"][0]
    step["objective_verification_check"] = {
        "name": "effect",
        "check_type": "tool_effect_verified",
        "path": "",
        "pattern": "",
        "command": [],
        "cwd": "",
    }
    step["verification_checks"].append(
        {"name": "effect", "check_type": "string_nonempty", "condition": "required", "actual_source": "tool_output"}
    )
    with pytest.raises(PlanValidationError, match="duplicate verification check name effect"):
        plan_from_payload(payload, available_tools=["read_file"])
