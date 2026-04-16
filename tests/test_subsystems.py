from __future__ import annotations

import json

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
