from __future__ import annotations

import json

import pytest

from swaag.planner import (
    PlanValidationError,
    create_shell_recovery_plan,
    plan_from_payload,
    ready_steps,
    transition_step,
)
from swaag.runtime import AgentRuntime, FatalSemanticEngineError

from tests.helpers import FakeModelClient, plan_response, plan_step


def test_plan_from_payload_validates_ordered_steps(make_config) -> None:
    config = make_config()
    payload = json.loads(
        plan_response(
            goal="Read a file, update it, then answer.",
            steps=[
                plan_step("step_read", "Read the file", "read", expected_tool="read_text", expected_output="File contents", success_criteria="The file is read."),
                plan_step(
                    "step_write",
                    "Edit the file",
                    "write",
                    expected_tool="edit_text",
                    expected_output="Updated file",
                    success_criteria="The file is updated.",
                    depends_on=["step_read"],
                ),
                plan_step(
                    "step_answer",
                    "Answer the user",
                    "respond",
                    expected_output="Final answer",
                    success_criteria="The user receives the result.",
                    depends_on=["step_write"],
                ),
            ],
        )
    )

    plan = plan_from_payload(payload, available_tools=["read_text", "edit_text", "notes", "calculator"])

    assert [step.expected_tool for step in plan.steps[:-1]] == ["read_text", "edit_text"]
    assert plan.steps[-1].kind == "respond"
    assert plan.steps[-1].depends_on == ["step_write"]
    with pytest.raises(PlanValidationError):
        transition_step(plan, "step_write", "running")


def test_plan_from_payload_rejects_invalid_tool(make_config) -> None:
    payload = json.loads(
        plan_response(
            goal="Try an invalid tool.",
            steps=[
                plan_step("step_bad", "Run missing tool", "tool", expected_tool="missing_tool", expected_output="x", success_criteria="x"),
                plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="done", depends_on=["step_bad"]),
            ],
        )
    )

    with pytest.raises(PlanValidationError):
        plan_from_payload(payload, available_tools=["calculator"])


def test_create_shell_recovery_plan_builds_read_write_respond_flow() -> None:
    plan = create_shell_recovery_plan("Fix the failing repository test.")

    assert [step.kind for step in plan.steps] == ["read", "write", "tool", "respond"]
    assert [step.expected_tool for step in plan.steps[:-1]] == ["shell_command", "edit_text", "run_tests"]
    assert plan.steps[-1].expected_tool is None
    assert plan.steps[1].depends_on == [plan.steps[0].step_id]
    assert plan.steps[2].depends_on == [plan.steps[1].step_id]
    assert plan.steps[3].depends_on == [plan.steps[2].step_id]
    assert any(check["name"] == "command_exit_zero" for check in plan.steps[0].verification_checks)
    assert any(check["name"] == "tool_files_changed" for check in plan.steps[1].verification_checks)
    assert any(check["name"] == "command_exit_zero" for check in plan.steps[2].verification_checks)
    assert "exact failing test name first" in plan.steps[0].input_text
    assert "Prefer replace_pattern_once or replace_range" in plan.steps[1].input_text


def test_plan_from_payload_derives_missing_verification_contract() -> None:
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
            }
        ],
    }
    plan = plan_from_payload(payload, available_tools=["calculator"])

    step = plan.steps[0]
    assert step.expected_outputs == ["4"]
    assert step.verification_type == "composite"
    assert any(check["name"] == "tool_result_present" for check in step.verification_checks)
    assert "tool_result_present" in step.required_conditions


def test_plan_from_payload_normalizes_invalid_verification_type() -> None:
    payload = json.loads(
        plan_response(
            goal="Compute",
            steps=[
                plan_step(
                    "step_calc",
                    "Compute",
                    "tool",
                    expected_tool="calculator",
                    expected_output="4",
                    success_criteria="tool returns 4",
                    verification_type="nonsense",
                ),
                plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="done", depends_on=["step_calc"]),
            ],
        )
    )
    plan = plan_from_payload(payload, available_tools=["calculator"])

    assert plan.steps[0].verification_type == "composite"


def test_plan_from_payload_normalizes_sparse_step_fields_and_condition_refs() -> None:
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
                "required_conditions": ["goal:Reply with exactly 17."],
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

    plan = plan_from_payload(payload, available_tools=["calculator"])

    calc_step, answer_step = plan.steps
    assert calc_step.goal == "Compute final value"
    assert calc_step.input_text == "Compute final value"
    assert calc_step.expected_output == "Compute final value"
    assert calc_step.done_condition == "tool_result:calculator"
    assert calc_step.required_conditions == [
        "dependencies_completed",
        "tool_result_present",
        "tool_name_matches",
    ]
    assert calc_step.optional_conditions == []
    assert calc_step.fallback_strategy == "If this step fails, replan from the latest valid state."
    assert answer_step.goal == "Answer the user"
    assert answer_step.input_text == "Answer the user"
    assert answer_step.done_condition == "assistant_response_nonempty"
    assert answer_step.required_conditions == [
        "dependencies_completed",
        "assistant_text_nonempty",
        "meets_success_criteria",
        "satisfies_done_condition",
    ]
    assert answer_step.verification_checks[1]["name"] == "assistant_text_nonempty"
    assert answer_step.verification_checks[2]["check_type"] == "criterion"


def test_plan_from_payload_demotes_tool_llm_fallback_to_deterministic_verification() -> None:
    payload = json.loads(
        plan_response(
            goal="Edit the file and answer.",
            steps=[
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
                plan_step(
                    "step_answer",
                    "Answer",
                    "respond",
                    expected_output="done",
                    success_criteria="reply done",
                    depends_on=["step_write"],
                ),
            ],
        )
    )

    plan = plan_from_payload(payload, available_tools=["write_file"])

    step = plan.steps[0]
    assert step.kind == "write"
    assert step.verification_type == "composite"
    assert any(check["check_type"] == "tool_output_schema_valid" for check in step.verification_checks)


def test_plan_from_payload_demotes_tool_execution_verification_to_deterministic_contract() -> None:
    payload = json.loads(
        plan_response(
            goal="Read the file and answer.",
            steps=[
                plan_step(
                    "step_read",
                    "Read the file",
                    "read",
                    expected_tool="read_file",
                    expected_output="contents",
                    success_criteria="file contents available",
                    verification_type="execution",
                    verification_checks=[{"name": "file_exists", "check_type": "criterion", "criterion": "file exists"}],
                    required_conditions=["file_exists"],
                ),
                plan_step(
                    "step_answer",
                    "Answer",
                    "respond",
                    expected_output="done",
                    success_criteria="reply done",
                    depends_on=["step_read"],
                ),
            ],
        )
    )

    plan = plan_from_payload(payload, available_tools=["read_file"])

    step = plan.steps[0]
    assert step.kind == "read"
    assert step.verification_type == "composite"
    assert any(check["name"] == "tool_result_present" for check in step.verification_checks)


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
    # The runtime expands the goal via task_expansion before planning, so the
    # plan's recorded goal includes the expansion suffix.
    assert goal in rebuilt.active_plan.goal


def test_runtime_rejects_malformed_plan_and_records_fatal_plan_error(make_config) -> None:
    config = make_config(planner__max_replans=0)
    goal = "Read sample.txt and then reply exactly done."
    fake_client = FakeModelClient(
        responses=[
            json.dumps({"goal": goal, "success_criteria": "x", "fallback_strategy": "y", "steps": []}),
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    with pytest.raises(FatalSemanticEngineError):
        runtime.run_turn(goal)
    session_dirs = sorted(path for path in runtime.history.root.iterdir() if path.is_dir())
    assert len(session_dirs) == 1
    session_id = session_dirs[0].name
    events = runtime.history.read_history(session_id)

    assert any(event.event_type == "fatal_system_error" for event in events)
    assert any(
        event.event_type == "reasoning_completed"
        and event.payload["status"] == "fatal_system_error"
        and event.payload["reason"] == "plan_generation_failed"
        for event in events
    )


def test_ready_steps_returns_all_parallel_ready_nodes() -> None:
    payload = json.loads(
        plan_response(
            goal="Read and note before answering.",
            steps=[
                plan_step("step_read", "Read", "read", expected_tool="read_text", expected_output="text", success_criteria="read"),
                plan_step("step_note", "Take note", "note", expected_tool="notes", expected_output="note", success_criteria="note"),
                plan_step("step_answer", "Answer", "respond", expected_output="answer", success_criteria="answer", depends_on=["step_read", "step_note"]),
            ],
        )
    )
    plan = plan_from_payload(payload, available_tools=["read_text", "notes", "calculator"])

    assert [step.step_id for step in ready_steps(plan)] == ["step_read", "step_note"]


def test_plan_from_payload_topologically_sorts_out_of_order_dag_steps() -> None:
    payload = json.loads(
        plan_response(
            goal="Read and answer.",
            steps=[
                plan_step(
                    "step_answer",
                    "Answer",
                    "respond",
                    expected_output="answer",
                    success_criteria="answer",
                    depends_on=["step_read"],
                    input_refs=["read_text"],
                ),
                plan_step(
                    "step_read",
                    "Read",
                    "read",
                    expected_tool="read_text",
                    expected_output="text",
                    success_criteria="read",
                    output_refs=["read_text"],
                ),
            ],
        )
    )

    plan = plan_from_payload(payload, available_tools=["read_text", "notes", "calculator"])

    assert [step.step_id for step in plan.steps] == ["step_read", "step_answer"]
    assert plan.current_step_id == "step_read"
