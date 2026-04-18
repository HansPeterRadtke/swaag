from __future__ import annotations

import json
from dataclasses import asdict

from swaag.runtime import AgentRuntime
from swaag.types import ExpandedTask, SessionState
from swaag.working_memory import build_working_memory

from tests.helpers import FakeModelClient, plan_response, plan_step



def test_build_working_memory_uses_plan_and_recent_results(make_config) -> None:
    config = make_config(runtime__max_tool_steps=4)
    goal = "Use the calculator tool to compute 6 * 7."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the value", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the answer.", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "6 * 7"}}),
            "42",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    state = runtime.history.rebuild_from_history(result.session_id)
    working_memory = build_working_memory(state)

    # The runtime expands the goal via the LLM contract before planning, so
    # working memory tracks the effective (expanded) goal — not the original
    # user text.
    assert goal in working_memory.active_goal
    assert "42" in "\n".join(working_memory.recent_results)
    assert working_memory.current_step_title == "" or working_memory.current_step_title == "Answer the user"



def test_working_memory_rebuild_matches_original(make_config) -> None:
    config = make_config(runtime__max_tool_steps=4)
    goal = "Use the calculator tool to compute 6 * 7."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the value", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the answer.", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "6 * 7"}}),
            "42",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    state_before = runtime.history.rebuild_from_history(result.session_id)
    runtime.history.current_state_path(result.session_id).unlink()
    rebuilt = runtime.history.rebuild_from_history(result.session_id)

    assert asdict(rebuilt.working_memory) == asdict(state_before.working_memory)
    events = runtime.history.read_history(result.session_id)
    assert any(event.event_type == "working_memory_updated" for event in events)


def test_consistency_checker_recovers_working_memory(make_config) -> None:
    config = make_config(runtime__max_tool_steps=4)
    goal = "Use the calculator tool to compute 6 * 7."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute the value", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The calculator returns the value."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user sees the answer.", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "6 * 7"}}),
            "42",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    state = runtime.history.rebuild_from_history(result.session_id)
    state.working_memory.active_goal = "corrupted"

    runtime._check_consistency(state)
    events = runtime.history.read_history(result.session_id)

    # After consistency recovery the goal should be the expanded (effective)
    # goal, since that is what the runtime tracks during the turn.
    assert goal in state.working_memory.active_goal
    assert any(event.event_type == "consistency_failed" for event in events)
    assert any(event.event_type == "state_rebuilt" for event in events)


def test_working_memory_uses_expanded_goal_when_present() -> None:
    state = build_working_memory(
        SessionState(
            session_id="s1",
            created_at="t0",
            updated_at="t0",
            config_fingerprint="cfg",
            model_base_url="http://example.test",
            expanded_task=ExpandedTask(
                original_goal="make a game",
                expanded_goal="make a game Build a small arcade prototype with one core mechanic and a playable loop.",
                scope=[],
                constraints=[],
                expected_outputs=[],
                assumptions=[],
            ),
        )
    )

    assert "playable loop" in state.active_goal


def test_working_memory_compacts_benchmark_task_contract_goal() -> None:
    goal = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\"}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "- test_Other\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    state = build_working_memory(
        SessionState(
            session_id="s1",
            created_at="t0",
            updated_at="t0",
            config_fingerprint="cfg",
            model_base_url="http://example.test",
            messages=[],
            active_plan=None,
            expanded_task=ExpandedTask(
                original_goal=goal,
                expanded_goal=goal,
                scope=[],
                constraints=[],
                expected_outputs=[],
                assumptions=[],
            ),
        )
    )

    assert state.active_goal == "Fix the benchmark issue. mathematica_code gives wrong output with Max Verify test_Function, test_Other."
