from __future__ import annotations

import json
from pathlib import Path

from swaag.history import replay_history
from swaag.runtime import AgentRuntime

from tests.helpers import FakeModelClient, plan_response, plan_step


def test_end_to_end_calculator_task_runs_through_verification(make_config) -> None:
    config = make_config()
    goal = "Use the calculator tool to compute 6 * 7. Reply with the numeric result only."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            responses=[
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step(
                            "step_calc",
                            "Compute",
                            "tool",
                            expected_tool="calculator",
                            expected_output="42",
                            success_criteria="The calculator returns 42.",
                            verification_checks=[
                                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "calculator"},
                                {"name": "exact_result", "check_type": "exact_match", "actual_source": "tool_output.result", "expected": 42},
                            ],
                            required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "exact_result"],
                            optional_conditions=[],
                        ),
                        plan_step("step_answer", "Answer", "respond", expected_output="42", success_criteria="Return 42 only.", depends_on=["step_calc"]),
                    ],
                ),
                json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "6 * 7"}}),
                "42",
            ]
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "42"
    assert any(event.event_type == "verification_passed" and event.payload["step_id"] == "step_calc" for event in events)
    rebuilt = replay_history(runtime.history.history_path(result.session_id))
    assert rebuilt.messages[-1].content == "42"


def test_end_to_end_file_edit_task_verifies_written_content(make_config, tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world\n", encoding="utf-8")
    config = make_config(tools__allow_side_effect_tools=True)
    goal = f"Edit {sample} so that hello becomes hi, then answer done."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            responses=[
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step(
                            "step_edit",
                            "Edit file",
                            "write",
                            expected_tool="edit_text",
                            expected_output="File contains hi world",
                            success_criteria="The file contains hi world.",
                            verification_checks=[
                                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                {"name": "file_contains_hi", "check_type": "file_contains", "path": str(sample), "pattern": "hi world"},
                            ],
                            required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_hi"],
                            optional_conditions=[],
                        ),
                        plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="Return done.", depends_on=["step_edit"]),
                    ],
                ),
                json.dumps(
                    {
                        "action": "call_tool",
                        "response": "",
                        "tool_name": "edit_text",
                        "tool_input": {
                            "path": str(sample),
                            "operation": "replace_pattern_all",
                            "pattern": "hello",
                            "replacement": "hi",
                        },
                    }
                ),
                "done",
            ]
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "done"
    assert sample.read_text(encoding="utf-8") == "hi world\n"
    assert any(event.event_type == "verification_passed" and event.payload["step_id"] == "step_edit" for event in events)
    rebuilt = replay_history(runtime.history.history_path(result.session_id))
    assert rebuilt.file_views[str(sample)].content == "hi world\n"
