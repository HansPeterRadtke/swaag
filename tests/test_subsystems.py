from __future__ import annotations

import json

from swaag.planner import plan_from_payload
from swaag.runtime import AgentRuntime
from swaag.subsystems import PlanningSubsystem, ReasoningSubsystem, ToolSubsystem
from swaag.types import PlanStep, StrategySelection

from tests.helpers import FakeModelClient, plan_response, plan_step


def test_planning_subsystem_creates_plan_and_records_events(make_config) -> None:
    config = make_config()
    goal = "Use the calculator tool to compute 1 + 1."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            responses=[
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step("step_calc", "Compute the value", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                        plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the answer.", depends_on=["step_calc"]),
                    ],
                ),
            ]
        ),
    )
    state = runtime.create_or_load_session()

    plan = PlanningSubsystem().run(runtime, state, goal)
    events = runtime.history.read_history(state.session_id)

    assert plan.goal == goal
    assert any(event.event_type == "subsystem_started" and event.payload["subsystem"] == "planning" for event in events)
    assert any(event.event_type == "subsystem_completed" and event.payload["subsystem"] == "planning" for event in events)


def test_tool_subsystem_runs_until_done_condition_is_satisfied(make_config) -> None:
    config = make_config(runtime__max_tool_steps=4)
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            responses=[
                json.dumps({"action": "call_tool", "response": "", "tool_name": "notes", "tool_input": {"action": "list"}}),
                json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "3 + 4"}}),
            ]
        ),
    )
    state = runtime.create_or_load_session()
    state.active_strategy = StrategySelection(
        strategy_name="exploratory",
        explore_before_commit=True,
        validate_assumptions=True,
        simplify_if_stuck=True,
        switch_on_failure=True,
        reason="test",
        mode="exploratory",
        tool_chain_depth=2,
    )
    step = PlanStep(
        step_id="step_calc",
        title="Compute the value",
        goal="Compute the value",
        kind="tool",
        expected_tool="calculator",
        input_text="Use tools to compute 3 + 4.",
        expected_output="Calculated value",
        done_condition="tool_result:calculator",
        success_criteria="The calculator returns the value.",
    )

    result = ToolSubsystem().run(runtime, state, step, action_counts={})
    events = runtime.history.read_history(state.session_id)

    assert result.success is True
    assert [item.tool_name for item in result.tool_results] == ["notes", "calculator"]
    assert any(event.event_type == "tool_chain_completed" and event.payload["success"] is True for event in events)


def test_reasoning_subsystem_requires_nonempty_answer(make_config) -> None:
    config = make_config()
    runtime = AgentRuntime(config, model_client=FakeModelClient(responses=["final answer"]))
    state = runtime.create_or_load_session()
    step = PlanStep(
        step_id="step_answer",
        title="Answer the user",
        goal="Answer the user",
        kind="respond",
        expected_tool=None,
        input_text="Answer directly.",
        expected_output="Final answer",
        done_condition="assistant_response_nonempty",
        success_criteria="The user receives a response.",
    )

    result = ReasoningSubsystem().run(runtime, state, step)

    assert result.success is True
    assert result.assistant_text == "final answer"
    assert result.evaluation is None


def test_reasoning_subsystem_does_not_force_not_done_for_incomplete_plan(make_config) -> None:
    config = make_config()
    runtime = AgentRuntime(config, model_client=FakeModelClient(responses=["Tokenizer splits on commas; normalizer uppercases tokens."]))
    state = runtime.create_or_load_session()
    state.active_plan = plan_from_payload(
        {
            "goal": "Fix the text pipeline.",
            "steps": [
                {
                    "step_id": "step_read_tokenizer",
                    "title": "Read tokenizer",
                    "goal": "Read tokenizer",
                    "kind": "read",
                    "expected_tool": "read_text",
                    "input_text": "pkg/tokenizer.py",
                    "expected_output": "Tokenizer source",
                    "success_criteria": "Tokenizer source is available.",
                },
                {
                    "step_id": "step_read_normalizer",
                    "title": "Read normalizer",
                    "goal": "Read normalizer",
                    "kind": "read",
                    "expected_tool": "read_text",
                    "input_text": "pkg/normalizer.py",
                    "expected_output": "Normalizer source",
                    "success_criteria": "Normalizer source is available.",
                },
                {
                    "step_id": "step_reason",
                    "title": "Identify the bugs",
                    "goal": "Identify the bugs",
                    "kind": "reasoning",
                    "expected_tool": "",
                    "input_text": "Tokenizer and normalizer source",
                    "expected_output": "List of bugs",
                    "success_criteria": "The bug list is non-empty.",
                },
                {
                    "step_id": "step_edit",
                    "title": "Edit files",
                    "goal": "Edit files",
                    "kind": "write",
                    "expected_tool": "edit_text",
                    "input_text": "Apply the fixes",
                    "expected_output": "Patched files",
                    "success_criteria": "The files are patched.",
                },
                {
                    "step_id": "step_respond",
                    "title": "Respond",
                    "goal": "Respond",
                    "kind": "respond",
                    "expected_tool": "",
                    "input_text": "Summarize the fixes.",
                    "expected_output": "Summary",
                    "success_criteria": "The user receives a summary.",
                },
            ],
        },
        available_tools=["read_text", "edit_text", "run_tests", "notes"],
    )
    assert state.active_plan is not None
    state.active_plan.steps[0].status = "completed"
    state.active_plan.steps[1].status = "completed"
    state.active_plan.steps[2].status = "running"
    state.active_plan.current_step_id = "step_reason"

    result = ReasoningSubsystem().run(runtime, state, state.active_plan.steps[2])

    assert result.success is True
    assert result.assistant_text == "Tokenizer splits on commas; normalizer uppercases tokens."

def test_tool_subsystem_does_not_blindly_rerun_failed_tests(make_config) -> None:
    config = make_config(runtime__max_tool_steps=4)
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            responses=[
                json.dumps({
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "run_tests",
                    "tool_input": {"command": ["python3", "-c", "import sys; sys.exit(1)"]},
                }),
                json.dumps({
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "run_tests",
                    "tool_input": {"command": ["python3", "-c", "import sys; sys.exit(1)"]},
                }),
            ]
        ),
    )
    state = runtime.create_or_load_session()
    step = PlanStep(
        step_id="step_tests",
        title="Run tests",
        goal="Run tests",
        kind="tool",
        expected_tool="run_tests",
        input_text="Run the failing tests.",
        expected_output="Tests pass",
        done_condition="tool_result:run_tests",
        success_criteria="The tests pass.",
        verification_checks=[
            {"name": "tool_result_present", "check_type": "tool_result_present"},
            {"name": "tool_name_matches", "check_type": "tool_name_matches", "expected": "run_tests"},
            {"name": "command_exit_zero", "check_type": "exact_match", "actual_source": "tool_output.exit_code", "expected": 0},
        ],
        required_conditions=["tool_result_present", "tool_name_matches", "command_exit_zero"],
    )

    result = ToolSubsystem().run(runtime, state, step, action_counts={})
    events = runtime.history.read_history(state.session_id)

    assert result.success is True
    assert [item.tool_name for item in result.tool_results] == ["run_tests"]
    assert sum(1 for event in events if event.event_type == "tool_chain_step") == 1
    assert not any(event.event_type == "duplicate_action_detected" for event in events)
