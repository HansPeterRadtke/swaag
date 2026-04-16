from __future__ import annotations

import json
from pathlib import Path

from swaag.runtime import AgentRuntime

from tests.helpers import FakeModelClient, plan_response, plan_step


def test_runtime_recovers_after_multiple_replans_and_remains_bounded(make_config, tmp_path: Path) -> None:
    module = tmp_path / "counter.py"
    module.write_text("def value() -> int:\n    return 0\n", encoding="utf-8")
    test_file = tmp_path / "test_counter.py"
    test_file.write_text(
        "import unittest\n\nfrom counter import value\n\n\nclass CounterTests(unittest.TestCase):\n    def test_value(self) -> None:\n        self.assertEqual(value(), 3)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )
    config = make_config(tools__allow_side_effect_tools=True, planner__max_replans=2, runtime__max_reasoning_steps=10, runtime__max_total_actions=12)
    goal = f"Read {module}, fix it so the tests pass, and reply fixed."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            responses=[
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step("step_read", "Read module", "read", expected_tool="read_text", expected_output="module source", success_criteria="module loaded"),
                        plan_step(
                            "step_write",
                            "Write first attempt",
                            "write",
                            expected_tool="edit_text",
                            expected_output="updated module",
                            success_criteria="module returns 3",
                            depends_on=["step_read"],
                            verification_checks=[
                                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                {"name": "tests_pass", "check_type": "command_success", "command": ["python3", "-m", "unittest", "-q", "test_counter.py"], "cwd": str(tmp_path), "framework": "unittest"},
                            ],
                            required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tests_pass"],
                            optional_conditions=[],
                        ),
                        plan_step("step_answer", "Answer", "respond", expected_output="fixed", success_criteria="answer returned", depends_on=["step_write"]),
                    ],
                ),
                json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": str(module)}}),
                json.dumps({"action": "call_tool", "response": "", "tool_name": "edit_text", "tool_input": {"path": str(module), "operation": "replace_pattern_once", "pattern": "return 0", "replacement": "return 1"}}),
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step("step_read_retry_1", "Read module again", "read", expected_tool="read_text", expected_output="module source", success_criteria="module loaded again"),
                        plan_step(
                            "step_write_retry_1",
                            "Write second attempt",
                            "write",
                            expected_tool="edit_text",
                            expected_output="updated module",
                            success_criteria="module returns 3",
                            depends_on=["step_read_retry_1"],
                            verification_checks=[
                                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                {"name": "tests_pass", "check_type": "command_success", "command": ["python3", "-m", "unittest", "-q", "test_counter.py"], "cwd": str(tmp_path), "framework": "unittest"},
                            ],
                            required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tests_pass"],
                            optional_conditions=[],
                        ),
                        plan_step("step_answer_retry_1", "Answer", "respond", expected_output="fixed", success_criteria="answer returned", depends_on=["step_write_retry_1"]),
                    ],
                ),
                json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": str(module)}}),
                json.dumps({"action": "call_tool", "response": "", "tool_name": "edit_text", "tool_input": {"path": str(module), "operation": "replace_pattern_once", "pattern": "return 1", "replacement": "return 2"}}),
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step("step_read_retry_2", "Read module again", "read", expected_tool="read_text", expected_output="module source", success_criteria="module loaded again"),
                        plan_step(
                            "step_write_retry_2",
                            "Write final attempt",
                            "write",
                            expected_tool="edit_text",
                            expected_output="updated module",
                            success_criteria="module returns 3",
                            depends_on=["step_read_retry_2"],
                            verification_checks=[
                                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                                {"name": "tests_pass", "check_type": "command_success", "command": ["python3", "-m", "unittest", "-q", "test_counter.py"], "cwd": str(tmp_path), "framework": "unittest"},
                            ],
                            required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tests_pass"],
                            optional_conditions=[],
                        ),
                        plan_step("step_answer_retry_2", "Answer", "respond", expected_output="fixed", success_criteria="answer returned", depends_on=["step_write_retry_2"]),
                    ],
                ),
                json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": str(module)}}),
                json.dumps({"action": "call_tool", "response": "", "tool_name": "edit_text", "tool_input": {"path": str(module), "operation": "replace_pattern_once", "pattern": "return 2", "replacement": "return 3"}}),
                "fixed",
            ]
        ),
    )

    result = runtime.run_turn(goal)
    state = runtime.history.rebuild_from_history(result.session_id)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "fixed"
    assert module.read_text(encoding="utf-8") == "def value() -> int:\n    return 3\n"
    assert state.metrics.replans >= 2
    assert state.metrics.max_iteration_stops == 0
    assert sum(1 for event in events if event.event_type == "replan_triggered") >= 2
    assert sum(1 for event in events if event.event_type == "step_failed") == 2
    assert sum(1 for event in events if event.event_type == "step_completed") >= 2
