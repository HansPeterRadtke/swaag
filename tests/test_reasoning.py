from __future__ import annotations

import json

import pytest

from swaag.runtime import AgentRuntime, FatalSemanticEngineError

from tests.helpers import FakeModelClient, plan_response, plan_step



def test_reasoning_loop_records_plan_and_reasoning_events(make_config) -> None:
    config = make_config(runtime__max_tool_steps=4, runtime__max_reasoning_steps=4)
    goal = "Use the calculator tool to compute 2 + 3."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the result", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the value.", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 3"}}),
            "5",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    event_types = [event.event_type for event in events]

    assert "reasoning_started" in event_types
    assert "plan_created" in event_types
    assert "step_started" in event_types
    assert "step_completed" in event_types
    assert "reasoning_completed" in event_types
    assert "context_built" in event_types
    assert "tool_execution_context" in event_types
    assert "prompt_analyzed" in event_types
    assert "decision_made" in event_types
    assert "strategy_selected" in event_types
    assert "subsystem_started" in event_types
    assert "subsystem_completed" in event_types



def test_reasoning_loop_treats_malformed_decision_output_as_fatal(make_config) -> None:
    config = make_config(runtime__max_reasoning_steps=2, planner__max_replans=0)
    goal = "Use the calculator tool to compute 1 + 1."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the result", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer directly", "respond", expected_output="Final answer", success_criteria="The user gets a reply.", depends_on=["step_calc"]),
                ],
            ),
            "not-json",
            "fallback answer",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    with pytest.raises(FatalSemanticEngineError):
        runtime.run_turn(goal)

    session_id = runtime.history.list_sessions()[0]
    events = runtime.history.read_history(session_id)
    fatal_log = runtime.history.root / "fatal_system_errors.jsonl"

    assert fatal_log.exists()
    assert any(event.event_type == "fatal_system_error" for event in events)
    completed = next(event for event in events if event.event_type == "reasoning_completed")
    assert completed.payload["status"] == "fatal_system_error"



def test_reasoning_loop_stops_at_max_steps_and_falls_back(make_config) -> None:
    config = make_config(runtime__max_reasoning_steps=1, runtime__max_tool_steps=3)
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the result", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the value.", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "fallback final answer",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "4"
    completed = next(event for event in events if event.event_type == "reasoning_completed")
    assert completed.payload["reason"] == "max_iterations_reached"


def test_failing_step_triggers_replan(make_config) -> None:
    config = make_config(runtime__max_reasoning_steps=4, planner__max_replans=1)
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the result", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the value.", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "echo", "tool_input": {"text": "wrong"}}),
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc_replan", "Compute the result correctly", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer_replan", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the value.", depends_on=["step_calc_replan"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "4",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "4"
    assert any(event.event_type == "verification_failed" for event in events)
    assert any(event.event_type == "tool_graph_rejected" for event in events)
    assert sum(1 for event in events if event.event_type == "plan_updated") >= 2
    assert any(event.event_type == "replan_triggered" for event in events)


def test_tool_subsystem_can_chain_helper_tool_before_expected_tool(make_config) -> None:
    config = make_config(runtime__max_reasoning_steps=5, runtime__max_tool_steps=4)
    goal = "Inspect src/app.py and use tools to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the result", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the value.", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "notes", "tool_input": {"action": "list"}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "4",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "4"
    assert [item.tool_name for item in result.tool_results] == ["notes", "calculator"]
    assert any(event.event_type == "tool_chain_started" for event in events)
    assert sum(1 for event in events if event.event_type == "tool_chain_step") == 2
    assert any(event.event_type == "tool_graph_planned" for event in events)


def test_repeated_failures_trigger_drift_recovery(make_config) -> None:
    config = make_config(runtime__max_reasoning_steps=4, runtime__max_tool_steps=2, planner__max_replans=2)
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the result", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the value.", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "echo", "tool_input": {"text": "wrong"}}),
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc_two", "Compute the result again", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer_two", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the value.", depends_on=["step_calc_two"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "echo", "tool_input": {"text": "still wrong"}}),
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc_three", "Compute the result correctly", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer_three", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the value.", depends_on=["step_calc_three"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "fallback",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "4"
    assert any(event.event_type == "drift_detected" for event in events)
    assert any(event.event_type == "recovery_triggered" and event.payload["reason"] == "drift_detected" for event in events)


def test_runtime_stops_when_no_progress_is_possible(make_config) -> None:
    config = make_config(runtime__no_progress_failure_limit=1, runtime__max_tool_steps=1, planner__max_replans=0, runtime__max_reasoning_steps=4)
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the result", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the value.", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "respond", "response": "wrong", "tool_name": "none", "tool_input": {}}),
            "fallback",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "not done"
    completed = next(event for event in events if event.event_type == "reasoning_completed")
    assert completed.payload["reason"] == "no_progress_possible"
