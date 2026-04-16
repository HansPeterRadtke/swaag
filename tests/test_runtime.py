from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import pytest
import requests

import swaag.runtime as runtime_module
from swaag.model import ModelClientError
from swaag.planner import create_shell_recovery_plan, plan_from_payload
from swaag.retrieval.embeddings import SemanticBackendProtocolError
from swaag.runtime import AgentRuntime, BudgetExceededError, FatalSemanticEngineError
from swaag.types import CompletionResult, Message

from tests.helpers import FakeModelClient, plan_response, plan_step


def test_runtime_tool_flow_records_budget_reports(make_config) -> None:
    config = make_config()
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        contract_responses={
            "task_decision": [
                json.dumps(
                    {
                        "split_task": False,
                        "expand_task": False,
                        "ask_user": False,
                        "assume_missing": False,
                        "generate_ideas": False,
                        "direct_response": False,
                        "execution_mode": "single_tool",
                        "preferred_tool_name": "calculator",
                        "confidence": 1.0,
                        "reason": "one calculator call plus exact finalization is sufficient",
                    }
                )
            ]
        },
        responses=[
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "4"
    assert [item.tool_name for item in result.tool_results] == ["calculator"]
    assert any(event.event_type == "budget_checked" for event in events)
    assert any(event.event_type == "action_selected" for event in events)
    assert any(event.event_type == "verification_passed" for event in events)
    assert all("budget_report" in event.payload for event in events if event.event_type in {"prompt_built", "budget_checked", "budget_rejected"})


def test_runtime_compacts_history_with_budgeted_summary(make_config) -> None:
    config = make_config(
        model__context_limit=120,
        context__reserved_response_tokens=16,
        context__reserved_summary_tokens=16,
        context__safety_margin_tokens=8,
        context__max_recent_messages=2,
    )
    fake_client = FakeModelClient(responses=[json.dumps({"summary": "Earlier conversation summary."})])
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    for index in range(6):
        runtime.history.record_event(
            state,
            "message_added",
            {"message": asdict(Message(role="user", content=f"old message {index} repeated repeated repeated", created_at=f"t{index}"))},
        )

    assert runtime._compact_once(state) is True
    events = runtime.history.read_history(state.session_id)
    assert any(event.event_type == "summary_created" for event in events)
    assert any(event.event_type == "history_compressed" for event in events)
    compact_event = next(event for event in events if event.event_type == "history_compressed")
    assert "summary_budget_report" in compact_event.payload


def test_runtime_stops_repeated_identical_tool_requests(make_config) -> None:
    config = make_config(runtime__max_repeated_action_occurrences=1, planner__max_replans=0)
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc_1", "Compute the first result", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The first expression is evaluated."),
                    plan_step(
                        "step_calc_2",
                        "Compute the same result again",
                        "tool",
                        expected_tool="calculator",
                        expected_output="Calculated value",
                        success_criteria="The repeated expression is evaluated.",
                        depends_on=["step_calc_1"],
                    ),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The final answer is returned.", depends_on=["step_calc_2"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "4",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "not done"
    assert len(result.tool_results) == 1
    assert any(event.event_type == "duplicate_action_detected" for event in events)


def test_runtime_retries_failed_model_request(make_config) -> None:
    config = make_config(model__max_retries=1)
    goal = "say ok"
    fake_client = FakeModelClient(
        responses=[
            ModelClientError("temporary failure"),
            plan_response(
                goal=goal,
                steps=[
                    plan_step(
                        "step_answer",
                        "Answer the user",
                        "respond",
                        expected_output="ok",
                        success_criteria="The user is greeted with ok.",
                    ),
                ],
            ),
            "ok",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "ok"
    assert any(event.event_type == "model_retry_scheduled" for event in events)
    assert any(event.event_type == "model_call_failed" for event in events)


def test_runtime_raises_budget_error_when_compaction_disabled(make_config) -> None:
    config = make_config(model__context_limit=20, context__reserved_response_tokens=10, context__safety_margin_tokens=5, context__compact_on_overflow=False)
    fake_client = FakeModelClient(responses=[])
    runtime = AgentRuntime(config, model_client=fake_client)
    with pytest.raises(BudgetExceededError):
        runtime.run_turn("word " * 20)


def test_same_input_and_same_model_responses_produce_identical_requests(make_config) -> None:
    config = make_config()
    goal = "Use the calculator tool to compute 2 + 2."
    responses = [
        plan_response(
            goal=goal,
            steps=[
                plan_step("step_calc", "Compute the result", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The expression is evaluated."),
                plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The final answer is returned.", depends_on=["step_calc"]),
            ],
        ),
        json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
        "4",
    ]
    client_one = FakeModelClient(responses=list(responses))
    client_two = FakeModelClient(responses=list(responses))

    result_one = AgentRuntime(config, model_client=client_one).run_turn(goal)
    result_two = AgentRuntime(config, model_client=client_two).run_turn(goal)

    assert result_one.assistant_text == result_two.assistant_text == "4"
    assert client_one.requests == client_two.requests


def test_runtime_updates_project_state_for_file_work(make_config, tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")
    config = make_config(runtime__max_reasoning_steps=3)
    goal = f"Read {sample} and answer done."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_read", "Read the file", "read", expected_tool="read_text", expected_output="File text", success_criteria="The file is read."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user gets done.", depends_on=["step_read"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": str(sample)}}),
            "done",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    state = runtime.history.rebuild_from_history(result.session_id)
    events = runtime.history.read_history(result.session_id)

    assert str(sample) in state.project_state.files_seen
    assert any(event.event_type == "project_state_updated" for event in events)


def test_runtime_keeps_project_state_consistent_when_plan_and_step_status_change(make_config, tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")
    config = make_config(runtime__max_reasoning_steps=3)
    goal = f"Read {sample} and answer done."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_read", "Read the file", "read", expected_tool="read_text", expected_output="File text", success_criteria="The file is read."),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user gets done.", depends_on=["step_read"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": str(sample)}}),
            "done",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert not any(
        event.event_type == "consistency_failed" and "project_state" in str(event.payload.get("component", ""))
        for event in events
    )


def test_runtime_retries_after_verification_failure(make_config) -> None:
    config = make_config(runtime__max_reasoning_steps=5, planner__max_replans=0)
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute", "tool", expected_tool="calculator", expected_output="value", success_criteria="calculator returns a value"),
                    plan_step("step_answer", "Answer", "respond", expected_output="answer", success_criteria="answer returned", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "respond", "response": "wrong", "tool_name": "none", "tool_input": {}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "4",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "4"
    assert sum(1 for event in events if event.event_type == "tool_chain_step") >= 2


def test_runtime_continues_other_ready_work_while_background_process_runs(make_config) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        runtime__background_poll_seconds=0.01,
        runtime__tool_timeout_seconds=2,
        planner__max_replans=0,
    )
    goal = "Start a long shell command, compute 6 * 7, then answer 42."
    no_spawn = json.dumps(
        {
            "spawn": False,
            "subagent_type": "none",
            "reason": "no specialist needed",
            "focus": "",
        }
    )
    fake_client = FakeModelClient(
        contract_responses={
            "subagent_selection": [no_spawn] * 8,
        },
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step(
                        "a_background",
                        "Start the shell command",
                        "tool",
                        expected_tool="shell_command",
                        input_text="Run a background shell command that prints background-ready",
                        expected_output="background command finished",
                        success_criteria="The background command finishes successfully.",
                    ),
                    plan_step(
                        "b_calc",
                        "Compute 6 * 7",
                        "tool",
                        expected_tool="calculator",
                        expected_output="42",
                        success_criteria="The calculator returns 42.",
                    ),
                        plan_step(
                            "c_answer",
                            "Answer the user",
                            "respond",
                            expected_output="42",
                            success_criteria="The user receives 42.",
                            depends_on=["a_background", "b_calc"],
                        ),
                ],
            ),
            json.dumps(
                    {
                        "action": "call_tool",
                        "response": "",
                        "tool_name": "shell_command",
                        "tool_input": {"command": "sleep 0.2; printf background-ready", "background": True},
                    }
                ),
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "calculator",
                    "tool_input": {"expression": "6 * 7"},
                }
            ),
            "42",
        ],
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    event_types = [event.event_type for event in events]

    assert result.assistant_text == "42"
    assert any(tool.tool_name == "calculator" for tool in result.tool_results)
    assert any(tool.tool_name == "shell_command" for tool in result.tool_results)

    calculator_called_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type == "tool_called" and event.payload.get("tool_name") == "calculator"
    )
    shell_completed_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type == "process_completed" and event.payload.get("metadata", {}).get("kind") == "shell_command"
    )
    background_step_completed_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type == "step_completed" and event.payload.get("step_id") == "a_background"
    )

    assert calculator_called_index < shell_completed_index < background_step_completed_index


def test_runtime_enters_wait_state_when_only_background_work_remains(make_config) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        runtime__background_poll_seconds=0.01,
        runtime__tool_timeout_seconds=2,
        planner__max_replans=0,
    )
    goal = "Start a long shell command and then answer ready."
    no_spawn = json.dumps(
        {
            "spawn": False,
            "subagent_type": "none",
            "reason": "no specialist needed",
            "focus": "",
        }
    )
    fake_client = FakeModelClient(
        contract_responses={"subagent_selection": [no_spawn] * 8},
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step(
                        "a_background",
                        "Start the shell command",
                        "tool",
                        expected_tool="shell_command",
                        input_text="Run a background shell command that prints ready",
                        expected_output="background command finished",
                        success_criteria="The background command finishes successfully.",
                    ),
                    plan_step(
                        "b_answer",
                        "Answer the user",
                        "respond",
                        expected_output="ready",
                        success_criteria="The user receives ready.",
                        depends_on=["a_background"],
                    ),
                ],
            ),
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "shell_command",
                    "tool_input": {"command": "sleep 0.2; printf ready", "background": True},
                }
            ),
            "ready",
        ],
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    event_types = [event.event_type for event in runtime.history.read_history(result.session_id)]

    assert result.assistant_text == "ready"
    assert "wait_entered" in event_types
    assert "wait_resumed" in event_types


def test_extract_path_argument_prefers_absolute_path_over_embedded_relative_suffix(make_config, tmp_path: Path) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient(responses=[]))
    target = tmp_path / "result.txt"

    path = runtime._extract_path_argument(
        f"Create {target} containing exactly sum=42 followed by a newline. Reply exactly written.",
        prefer_last=True,
    )

    assert path == str(target)


def test_extract_path_argument_prefers_explicit_path_line_over_paths_in_task_text(make_config, tmp_path: Path) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient(responses=[]))
    core = tmp_path / "pkg" / "core.py"

    path = runtime._extract_path_argument(
        f"read_text path: test_sample.py\nTask: Read {core} and fix it so tests in test_sample.py pass.",
        prefer_last=False,
    )

    assert path == "test_sample.py"

def test_runtime_parse_json_rejects_trailing_text_after_structured_object(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient(responses=[]))

    with pytest.raises(RuntimeError, match="invalid JSON"):
        runtime._parse_json('{"split_task": false, "expand_task": false}\n\n17', contract_name="task_decision")


def test_runtime_parse_json_rejects_fenced_json_object_for_structured_calls(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient(responses=[]))

    with pytest.raises(RuntimeError, match="invalid JSON"):
        runtime._parse_json("```json\n{\"task_type\": \"structured\"}\n```", contract_name="prompt_analysis")


def test_selection_counter_uses_non_recording_tokenization(make_config) -> None:
    client = FakeModelClient(responses=[])
    runtime = AgentRuntime(make_config(), model_client=client)
    state = runtime.create_or_load_session()
    before = len(runtime.history.read_history(state.session_id))

    result = runtime._get_selection_counter().count_text("alpha beta gamma")

    after = len(runtime.history.read_history(state.session_id))
    assert result.tokens == 3
    assert client.tokenize_requests == []
    assert after == before


def test_runtime_metrics_are_derived_from_history(make_config) -> None:
    config = make_config()
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute", "tool", expected_tool="calculator", expected_output="value", success_criteria="calculator returns a value"),
                    plan_step("step_answer", "Answer", "respond", expected_output="answer", success_criteria="answer returned", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "4",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    rebuilt = runtime.history.rebuild_from_history(result.session_id)

    assert rebuilt.metrics.model_calls >= 3
    assert rebuilt.metrics.tool_calls == 1
    assert rebuilt.metrics.verification_passes >= 1
    assert rebuilt.metrics.successful_turns == 1
    assert rebuilt.metrics.total_cost_units > 0
    assert rebuilt.metrics.verification_success_rate > 0.0
    assert rebuilt.metrics.verification_failure_rate == 0.0
    assert rebuilt.metrics.verification_type_distribution["composite"] >= 1
    assert rebuilt.metrics.verification_type_distribution["llm_fallback"] >= 1
    assert rebuilt.metrics.llm_fallback_rate > 0.0


def test_runtime_records_model_request_progress_for_slow_calls(make_config) -> None:
    class SlowClient(FakeModelClient):
        def send_completion(self, payload, *, timeout_seconds: int | None = None):
            time.sleep(0.15)
            return super().send_completion(payload, timeout_seconds=timeout_seconds)

    config = make_config(model__progress_poll_seconds=0.05)
    goal = "Reply with exactly 17. Do not use any tools. Do not add any extra text."
    runtime = AgentRuntime(
        config,
        model_client=SlowClient(
            responses=[
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step(
                            "step_answer",
                            "Answer the user",
                            "respond",
                            expected_output="17",
                            success_criteria="The answer is exactly 17.",
                        ),
                    ],
                ),
                "17",
            ]
        ),
    )

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "17"
    assert any(event.event_type == "model_request_progress" for event in events)


def test_runtime_refines_a_write_step_across_multiple_tool_attempts(make_config, tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    config = make_config(
        tools__allow_side_effect_tools=True,
        runtime__max_tool_steps=3,
        planner__max_replans=0,
    )
    goal = f"Write the final content into {target} and then reply done."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step(
                        "step_write",
                        "Write the target file",
                        "write",
                        expected_tool="write_file",
                        expected_output="file written",
                        success_criteria="the final file content matches exactly",
                        verification_checks=[
                            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                            {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                            {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "write_file"},
                            {"name": "file_has_final_text", "check_type": "file_contains", "path": str(target), "pattern": "final content"},
                        ],
                        required_conditions=[
                            "dependencies_completed",
                            "tool_result_present",
                            "tool_name_matches",
                            "file_has_final_text",
                        ],
                    ),
                    plan_step(
                        "step_answer",
                        "Answer",
                        "respond",
                        expected_output="done",
                        success_criteria="the assistant replies done",
                        depends_on=["step_write"],
                    ),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "write_file", "tool_input": {"path": str(target), "content": "wrong content\n"}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "write_file", "tool_input": {"path": str(target), "content": "final content\n"}}),
            "done",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "done"
    assert target.read_text(encoding="utf-8") == "final content\n"
    assert sum(1 for event in events if event.event_type == "tool_called") == 2
    progress_messages = [event.payload["progress"] for event in events if event.event_type == "subsystem_progress"]
    assert any("preview_passed=False" in message for message in progress_messages)
    assert any("preview_passed=True" in message for message in progress_messages)


def test_runtime_uses_model_driven_frontend_contracts(make_config) -> None:
    config = make_config()
    goal = "Make a game."
    fake_client = FakeModelClient(
        contract_responses={
            "prompt_analysis": [
                json.dumps(
                    {
                        "task_type": "vague",
                        "completeness": "incomplete",
                        "requires_expansion": True,
                        "requires_decomposition": False,
                        "confidence": 0.9,
                        "detected_entities": [],
                        "detected_goals": ["make a game"],
                    }
                )
            ],
            "task_decision": [
                json.dumps(
                    {
                        "split_task": False,
                        "expand_task": True,
                        "ask_user": False,
                        "assume_missing": True,
                        "generate_ideas": True,
                        "confidence": 0.9,
                        "reason": "prompt_is_vague",
                    }
                )
            ],
            "task_expansion": [
                json.dumps(
                    {
                        "original_goal": goal,
                        "expanded_goal": "Make a game. Build a small arcade prototype with one core mechanic and a playable loop.",
                        "scope": ["single playable loop"],
                        "constraints": ["small scope"],
                        "expected_outputs": ["prototype"],
                        "assumptions": ["arcade"],
                    }
                )
            ],
        },
        responses=[
            plan_response(
                goal="Make a game. Build a small arcade prototype with one core mechanic and a playable loop.",
                steps=[
                    plan_step("step_answer", "Answer", "respond", expected_output="answer", success_criteria="answer returned"),
                ],
            ),
            "prototype scoped",
        ],
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    runtime.run_turn(goal)

    contracts = [request["contract"] for request in fake_client.requests]
    semantic_contracts = [contract for contract in contracts if contract != "subagent_selection"]
    assert semantic_contracts[:4] == ["prompt_analysis", "task_decision", "task_expansion", "strategy_selection"]
    assert "task_plan" in contracts


def test_runtime_bypasses_llm_plan_generation_for_semantic_direct_response(make_config) -> None:
    config = make_config()
    goal = "Reply with exactly 17. Do not use any tools."
    fake_client = FakeModelClient(
        contract_responses={
            "prompt_analysis": [
                json.dumps(
                    {
                        "task_type": "structured",
                        "completeness": "complete",
                        "requires_expansion": False,
                        "requires_decomposition": False,
                        "confidence": 0.9,
                        "detected_entities": [],
                        "detected_goals": ["reply exactly 17"],
                    }
                )
            ],
            "task_decision": [
                json.dumps(
                    {
                        "split_task": False,
                        "expand_task": False,
                        "ask_user": False,
                        "assume_missing": False,
                        "generate_ideas": False,
                        "direct_response": True,
                        "confidence": 0.95,
                        "reason": "single direct assistant reply is sufficient",
                    }
                )
            ],
            "strategy_selection": [
                json.dumps(
                    {
                        "task_profile": "generic",
                        "strategy_name": "conservative",
                        "explore_before_commit": False,
                        "tool_chain_depth": 1,
                        "verification_intensity": 1.0,
                        "reason": "direct response path",
                    }
                )
            ],
            "plain_text": ["17"],
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    contracts = [request["contract"] for request in fake_client.requests]

    assert result.assistant_text == "17"
    assert "task_plan" not in contracts
    semantic_contracts = [contract for contract in contracts if contract != "subagent_selection"]
    assert semantic_contracts[:4] == ["prompt_analysis", "task_decision", "strategy_selection", "plain_text"]
    assert any(
        event.event_type == "plan_created" and event.payload.get("plan", {}).get("goal") == goal
        for event in events
    )
    assert any(event.event_type == "verification_passed" for event in events)


def test_runtime_blocks_direct_response_when_prompt_explicitly_requires_named_tool(make_config) -> None:
    config = make_config(runtime__max_reasoning_steps=4)
    goal = "Use the calculator tool to compute 2 + 2. Reply with only the integer."
    fake_client = FakeModelClient(
        responses=[
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "4",
        ],
        contract_responses={
            "prompt_analysis": [
                json.dumps(
                    {
                        "task_type": "structured",
                        "completeness": "complete",
                        "requires_expansion": False,
                        "requires_decomposition": False,
                        "confidence": 1.0,
                        "detected_entities": ["calculator", "2", "2"],
                        "detected_goals": ["compute the expression"],
                    }
                )
            ],
            "task_decision": [
                json.dumps(
                    {
                        "split_task": False,
                        "expand_task": False,
                        "ask_user": False,
                        "assume_missing": False,
                        "generate_ideas": False,
                        "direct_response": True,
                        "confidence": 1.0,
                        "reason": "single answer",
                    }
                )
            ],
            "strategy_selection": [
                json.dumps(
                    {
                        "task_profile": "generic",
                        "strategy_name": "conservative",
                        "explore_before_commit": False,
                        "tool_chain_depth": 1,
                        "verification_intensity": 1.0,
                        "reason": "tool use required",
                    }
                )
            ],
            "task_plan": [
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step("step_calc", "Compute", "tool", expected_tool="calculator", expected_output="value", success_criteria="calculator returns a value"),
                        plan_step("step_answer", "Answer", "respond", expected_output="4", success_criteria="reply with the integer result only", depends_on=["step_calc"]),
                    ],
                )
            ],
        },
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    event_types = [event.event_type for event in events]
    contracts = [request.get("contract") for request in fake_client.requests]

    assert result.assistant_text == "4"
    assert "decision_adjusted" in event_types
    assert "task_plan" in contracts
    assert not any(
        event.event_type == "plan_created"
        and event.payload.get("reason") == "semantic_direct_response"
        for event in events
    )
    assert any(event.event_type == "tool_called" and event.payload.get("tool_name") == "calculator" for event in events)


def test_runtime_blocks_direct_response_when_strategy_requires_write_steps(make_config, tmp_path: Path) -> None:
    config = make_config(
        runtime__max_reasoning_steps=4,
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
    )
    target = tmp_path / "app.py"
    target.write_text("blueprint = name\n", encoding="utf-8")
    goal = f"Fix {target} so blueprint names with dots raise an error."
    fake_client = FakeModelClient(
        contract_responses={
            "prompt_analysis": [
                json.dumps(
                    {
                        "task_type": "structured",
                        "completeness": "complete",
                        "requires_expansion": False,
                        "requires_decomposition": False,
                        "confidence": 1.0,
                        "detected_entities": ["app.py", "blueprint"],
                        "detected_goals": ["fix the file"],
                    }
                )
            ],
            "task_decision": [
                json.dumps(
                    {
                        "split_task": False,
                        "expand_task": False,
                        "ask_user": False,
                        "assume_missing": False,
                        "generate_ideas": False,
                        "direct_response": True,
                        "confidence": 1.0,
                        "reason": "single answer seems enough",
                    }
                )
            ],
            "strategy_selection": [
                json.dumps(
                    {
                        "task_profile": "file_edit",
                        "strategy_name": "conservative",
                        "explore_before_commit": False,
                        "tool_chain_depth": 1,
                        "verification_intensity": 1.0,
                        "reason": "the task requires direct file edits",
                    }
                )
            ],
            "task_plan": [
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step(
                            "step_edit",
                            "Edit",
                            "write",
                            expected_tool="edit_text",
                            expected_output="patched file",
                            success_criteria="app.py contains the validation",
                            input_text=str(target),
                        ),
                        plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="summarize the change", depends_on=["step_edit"]),
                    ],
                )
            ],
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": str(target),
                        "operation": "replace_pattern_once",
                        "pattern": "blueprint = name",
                        "replacement": "if '.' in name: raise ValueError('dots not allowed')\\nblueprint = name",
                    }
                )
            ],
            "plain_text": ["Patched app.py and added the dot-name validation."],
        },
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert "Patched app.py" in result.assistant_text
    assert any(
        event.event_type == "decision_adjusted"
        and event.payload.get("reason") == "strategy_requires_full_plan"
        for event in events
    )
    assert any(event.event_type == "tool_called" and event.payload.get("tool_name") == "edit_text" for event in events)


def test_runtime_repairs_plan_that_omits_explicitly_required_named_tool(make_config) -> None:
    config = make_config(runtime__max_reasoning_steps=4)
    goal = "Use the calculator tool to compute 2 + 2. Reply with only the integer."
    fake_client = FakeModelClient(
        responses=[
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "4",
        ],
        contract_responses={
            "prompt_analysis": [
                json.dumps(
                    {
                        "task_type": "structured",
                        "completeness": "complete",
                        "requires_expansion": False,
                        "requires_decomposition": False,
                        "confidence": 1.0,
                        "detected_entities": ["calculator", "2", "2"],
                        "detected_goals": ["compute the expression"],
                    }
                )
            ],
            "task_decision": [
                json.dumps(
                    {
                        "split_task": False,
                        "expand_task": False,
                        "ask_user": False,
                        "assume_missing": False,
                        "generate_ideas": False,
                        "direct_response": False,
                        "confidence": 1.0,
                        "reason": "tool use required",
                    }
                )
            ],
            "strategy_selection": [
                json.dumps(
                    {
                        "task_profile": "generic",
                        "strategy_name": "conservative",
                        "explore_before_commit": False,
                        "tool_chain_depth": 1,
                        "verification_intensity": 1.0,
                        "reason": "tool use required",
                    }
                )
            ],
            "task_plan": [
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step(
                            "step_answer",
                            "Answer",
                            "respond",
                            expected_output="4",
                            success_criteria="reply with the integer result only",
                        ),
                    ],
                )
            ],
        },
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "4"
    assert any(event.event_type == "tool_called" and event.payload.get("tool_name") == "calculator" for event in events)
    assert "task_plan" in [request.get("contract") for request in fake_client.requests]
    assert any(event.event_type == "plan_repaired" for event in events)


def test_runtime_waits_for_semantic_engine_instead_of_using_fake_semantic_fallback(make_config) -> None:
    config = make_config()
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "prompt_analysis": [
                    requests.ConnectionError("llm down"),
                    requests.ConnectionError("llm still down"),
                ]
            }
        ),
    )
    runtime._sleep = lambda _seconds: None
    runtime._max_model_unavailable_attempts = 1
    state = runtime.create_or_load_session()

    with pytest.raises(ModelClientError, match="semantic_engine_unavailable"):
        runtime._analyze_prompt_frontend(state, "Fix app.py")

    events = runtime.history.read_history(state.session_id)
    assert any(
        event.event_type == "error" and event.payload.get("operation") == "semantic_engine_unavailable"
        for event in events
    )
    assert not any(event.event_type == "prompt_analyzed" for event in events)


def test_runtime_recovers_malformed_coding_plan_with_shell_recovery_plan(make_config) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        planner__max_replans=0,
    )
    goal = "Fix the failing code path, verify it locally, then answer."
    fake_client = FakeModelClient(
        contract_responses={
            "prompt_analysis": [
                json.dumps(
                    {
                        "task_type": "structured",
                        "completeness": "complete",
                        "requires_expansion": False,
                        "requires_decomposition": False,
                        "confidence": 1.0,
                        "detected_entities": ["failing test"],
                        "detected_goals": ["fix the code"],
                    }
                )
            ],
            "task_decision": [
                json.dumps(
                    {
                        "split_task": False,
                        "expand_task": False,
                        "ask_user": False,
                        "assume_missing": False,
                        "generate_ideas": False,
                        "direct_response": False,
                        "execution_mode": "full_plan",
                        "preferred_tool_name": "",
                        "confidence": 1.0,
                        "reason": "planning required",
                    }
                )
            ],
            "strategy_selection": [
                json.dumps(
                    {
                        "task_profile": "coding",
                        "strategy_name": "conservative",
                        "explore_before_commit": False,
                        "tool_chain_depth": 1,
                        "verification_intensity": 1.0,
                        "reason": "explicit code edits required",
                    }
                )
            ],
            "task_plan": [
                "{\n  \"goal\": \"Fix the code\",\n  \"steps\": [\n    {\n      \"step_id\": \"1\",\n      \"title\": \"Inspect\",\n"
            ],
        },
        responses=[
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "shell_command",
                    "tool_input": {"command": "printf inspection"},
                }
            ),
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "shell_command",
                    "tool_input": {"command": "printf verification"},
                }
            ),
            "done",
        ],
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    rebuilt = runtime.history.rebuild_from_history(result.session_id)

    assert result.assistant_text == "done"
    assert rebuilt.active_plan is not None
    assert [step.kind for step in rebuilt.active_plan.steps] == ["read", "write", "respond"]
    assert [step.expected_tool for step in rebuilt.active_plan.steps[:-1]] == ["shell_command", "shell_command"]
    assert any(
        event.event_type == "plan_repaired"
        and event.payload.get("repair") == "shell_recovery_plan"
        and event.payload.get("reason") == "planner_structured_failure_shell_recovery"
        for event in events
    )
    assert sum(
        1
        for event in events
        if event.event_type == "tool_called" and event.payload.get("tool_name") == "shell_command"
    ) == 2
    assert not any(event.event_type == "fatal_system_error" for event in events)


def test_runtime_recovers_strategy_incompatible_coding_plan_with_shell_recovery_plan(make_config) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        planner__max_replans=0,
    )
    goal = "Fix the failing code path, verify it locally, then answer."
    fake_client = FakeModelClient(
        contract_responses={
            "prompt_analysis": [
                json.dumps(
                    {
                        "task_type": "structured",
                        "completeness": "complete",
                        "requires_expansion": False,
                        "requires_decomposition": False,
                        "confidence": 1.0,
                        "detected_entities": ["failing test"],
                        "detected_goals": ["fix the code"],
                    }
                )
            ],
            "task_decision": [
                json.dumps(
                    {
                        "split_task": False,
                        "expand_task": False,
                        "ask_user": False,
                        "assume_missing": False,
                        "generate_ideas": False,
                        "direct_response": False,
                        "execution_mode": "full_plan",
                        "preferred_tool_name": "",
                        "confidence": 1.0,
                        "reason": "planning required",
                    }
                )
            ],
            "strategy_selection": [
                json.dumps(
                    {
                        "task_profile": "coding",
                        "strategy_name": "conservative",
                        "explore_before_commit": False,
                        "tool_chain_depth": 1,
                        "verification_intensity": 1.0,
                        "reason": "explicit code edits required",
                    }
                )
            ],
            "task_plan": [
                plan_response(
                    goal="Fix the code",
                    steps=[
                        plan_step(
                            "step_patch_only",
                            "Patch and verify",
                            "write",
                            expected_tool="shell_command",
                            input_text="apply the fix and verify it",
                            expected_output="patched and verified",
                            success_criteria="apply the fix and verify it",
                        ),
                        plan_step(
                            "step_answer",
                            "Report result",
                            "respond",
                            expected_output="done",
                            success_criteria="reply done",
                        ),
                    ],
                )
            ],
        },
        responses=[
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "shell_command",
                    "tool_input": {"command": "printf inspection"},
                }
            ),
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "shell_command",
                    "tool_input": {"command": "printf verification"},
                }
            ),
            "done",
        ],
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "done"
    assert any(
        event.event_type == "plan_repaired"
        and event.payload.get("repair") == "shell_recovery_plan"
        and event.payload.get("error_type") == "StrategyValidationError"
        for event in events
    )
    assert sum(
        1
        for event in events
        if event.event_type == "tool_called" and event.payload.get("tool_name") == "shell_command"
    ) == 2


def test_runtime_task_contract_marks_benchmark_issue_complete_and_non_expanding(make_config) -> None:
    config = make_config()
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "prompt_analysis": [
                    json.dumps(
                        {
                            "task_type": "structured",
                            "completeness": "partial",
                            "requires_expansion": True,
                            "requires_decomposition": True,
                            "confidence": 0.8,
                            "detected_entities": ["bug"],
                            "detected_goals": ["fix code"],
                        }
                    )
                ],
                "task_decision": [
                    json.dumps(
                        {
                            "split_task": True,
                            "expand_task": True,
                            "ask_user": False,
                            "assume_missing": False,
                            "generate_ideas": False,
                            "direct_response": False,
                            "execution_mode": "full_plan",
                            "preferred_tool_name": "",
                            "confidence": 0.8,
                            "reason": "model wanted expansion",
                        }
                    )
                ],
            },
        ),
    )
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- tests/test_demo.py::test_fix\n"
    )

    analysis = runtime._analyze_prompt_frontend(state, user_text)
    decision = runtime._decide_prompt_frontend(state, user_text, analysis)
    compact_goal = runtime._operational_goal_from_task_contract(user_text)
    turn_prep = runtime._prepare_turn_context(state, user_text)

    assert analysis.completeness == "complete"
    assert analysis.requires_expansion is False
    assert analysis.requires_decomposition is False
    assert decision.expand_task is False
    assert decision.split_task is False
    assert decision.execution_mode == "full_plan"
    assert decision.reason.endswith("task_contract")
    assert compact_goal.startswith("Fix the benchmark issue.")
    assert "Verify tests/test_demo.py::test_fix." in compact_goal
    assert turn_prep.effective_goal == compact_goal


def test_runtime_seeds_shell_recovery_plan_for_local_repo_code_fix_contract(
    make_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        planner__max_replans=0,
    )
    monkeypatch.chdir(tmp_path)
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- tests/test_demo.py::test_fix\n"
    )
    fake_client = FakeModelClient(
        contract_responses={
            "prompt_analysis": [
                json.dumps(
                    {
                        "task_type": "structured",
                        "completeness": "partial",
                        "requires_expansion": True,
                        "requires_decomposition": True,
                        "confidence": 0.6,
                        "detected_entities": ["bug"],
                        "detected_goals": ["fix code"],
                    }
                )
            ],
            "task_decision": [
                json.dumps(
                    {
                        "split_task": True,
                        "expand_task": True,
                        "ask_user": False,
                        "assume_missing": False,
                        "generate_ideas": False,
                        "direct_response": False,
                        "execution_mode": "full_plan",
                        "preferred_tool_name": "",
                        "confidence": 0.6,
                        "reason": "model wanted expansion",
                    }
                )
            ],
            "strategy_selection": [
                json.dumps(
                    {
                        "task_profile": "coding",
                        "strategy_name": "conservative",
                        "explore_before_commit": False,
                        "tool_chain_depth": 1,
                        "verification_intensity": 1.0,
                        "reason": "explicit code edits required",
                    }
                )
            ],
            "tool_input:shell_command": [
                json.dumps({"command": "printf inspection", "background": False}),
                json.dumps(
                    {
                        "command": "python3 - <<'PY'\nfrom pathlib import Path\nPath('patched.txt').write_text('ok\\n', encoding='utf-8')\nprint('verification')\nPY",
                        "background": False,
                    }
                ),
            ],
        },
        responses=[
            "done",
            "done",
        ],
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(user_text)
    events = runtime.history.read_history(result.session_id)
    request_contracts = [request["contract"] for request in fake_client.requests]

    assert result.assistant_text == "done"
    assert "task_plan" not in request_contracts
    assert "tool_input:shell_command" in request_contracts
    assert any(
        event.event_type == "plan_repaired"
        and event.payload.get("reason") == "task_contract_shell_recovery_seed"
        for event in events
    )
    assert any(
        event.event_type == "plan_created"
        and "tool_files_changed"
        in next(
            (
                step.get("required_conditions", [])
                for step in event.payload.get("plan", {}).get("steps", [])
                if step.get("kind") == "write"
            ),
            [],
        )
        for event in events
    )


def test_runtime_skips_retriever_selection_for_shell_recovery_plan(make_config, monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config()
    runtime = AgentRuntime(config, model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    goal = "Fix the failing code path."
    plan = create_shell_recovery_plan(goal)
    state.active_plan = plan

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("subagent selection should be skipped for shell recovery plan context")

    monkeypatch.setattr(runtime, "_select_subagent_frontend", _unexpected)

    bundle = runtime._build_context_bundle(state, goal=goal, kind="decision", prompt_mode="lean")
    events = runtime.history.read_history(state.session_id)

    assert bundle is not None
    assert any(
        event.event_type == "subagent_selection_resolved"
        and event.payload.get("selection", {}).get("reason") == "shell_recovery_context_direct"
        for event in events
    )


def test_runtime_deterministically_finalizes_calculator_tool_result(make_config) -> None:
    config = make_config()
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    runtime._record_message(
        state,
        Message(
            role="user",
            content="Use the calculator tool to compute 2 + 2. Reply with only the integer.",
            created_at="t1",
        ),
    )
    runtime._record_message(
        state,
        Message(
            role="tool",
            name="calculator",
            content="4",
            created_at="t2",
            metadata={"output": {"expression": "2 + 2", "result": 4}},
        ),
    )

    assert runtime._deterministic_answer(state) == "4"


def test_runtime_deterministically_extracts_requested_line_from_file_read(make_config) -> None:
    config = make_config()
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    runtime._record_message(
        state,
        Message(
            role="user",
            content="Read /tmp/example.txt and return exactly the full text on line 3. No extra words.",
            created_at="t1",
        ),
    )
    runtime._record_message(
        state,
        Message(
            role="tool",
            name="read_file",
            content="owner=carol",
            created_at="t2",
            metadata={
                "raw_input": {"path": "/tmp/example.txt", "line_number": 3},
                "output": {"path": "/tmp/example.txt", "text": "line1=ignore\nline2=ignore\nowner=carol\n"},
            },
        ),
    )

    assert runtime._deterministic_answer(state) == "owner=carol"


def test_runtime_logs_fatal_error_when_hard_enforced_structured_output_is_malformed(make_config) -> None:
    config = make_config()
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(contract_responses={"prompt_analysis": ["not-json"]}),
    )
    state = runtime.create_or_load_session()

    with pytest.raises(FatalSemanticEngineError):
        runtime._analyze_prompt_frontend(state, "Fix app.py")

    fatal_log = runtime.history.root / "fatal_system_errors.jsonl"
    assert fatal_log.exists()
    rows = [json.loads(line) for line in fatal_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert rows[-1]["call_kind"] == "analysis"
    assert rows[-1]["contract_name"] == "prompt_analysis"
    events = runtime.history.read_history(state.session_id)
    assert any(event.event_type == "fatal_system_error" for event in events)


def test_runtime_logs_fatal_error_when_retrieval_semantic_schema_fails(
    make_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config()
    runtime = AgentRuntime(config, model_client=FakeModelClient(responses=[]))
    state = runtime.create_or_load_session()

    def _boom(*args, **kwargs):
        raise SemanticBackendProtocolError("structured relevance response violated schema")

    monkeypatch.setattr(runtime_module, "build_context", _boom)

    with pytest.raises(FatalSemanticEngineError):
        runtime._build_context_bundle(
            state,
            goal="Fix app.py",
            kind="analysis",
            prompt_mode="standard",
        )

    fatal_log = runtime.history.root / "fatal_system_errors.jsonl"
    assert fatal_log.exists()
    rows = [json.loads(line) for line in fatal_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert rows[-1]["operation"] == "semantic_retrieval"
    assert rows[-1]["call_kind"] == "analysis"
    events = runtime.history.read_history(state.session_id)
    assert any(event.event_type == "fatal_system_error" for event in events)


def test_runtime_enforces_tool_call_budget(make_config) -> None:
    config = make_config(runtime__tool_call_budget=1, runtime__max_total_actions=6)
    goal = "Use the calculator tool twice and then answer."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc_1", "Compute once", "tool", expected_tool="calculator", expected_output="value", success_criteria="first value returned"),
                    plan_step("step_calc_2", "Compute twice", "tool", expected_tool="calculator", expected_output="value", success_criteria="second value returned", depends_on=["step_calc_1"]),
                    plan_step("step_answer", "Answer", "respond", expected_output="answer", success_criteria="answer returned", depends_on=["step_calc_2"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    rebuilt = runtime.history.rebuild_from_history(result.session_id)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "not done"
    assert sum(1 for event in events if event.event_type == "tool_called") == 1
    completed = next(event for event in events if event.event_type == "reasoning_completed")
    assert completed.payload["reason"] == "tool_call_budget_reached"
    assert rebuilt.metrics.tool_call_budget_hits == 1


def test_runtime_is_deterministic_across_seeded_randomized_calculator_tasks(make_config) -> None:
    config = make_config()
    rng = random.Random(0)
    expressions = [f"{rng.randint(1, 9)} + {rng.randint(1, 9)}" for _ in range(4)]

    outputs: list[str] = []
    for expression in expressions:
        goal = f"Use the calculator tool to compute {expression}."
        expected = str(eval(expression))
        fake_client = FakeModelClient(
            responses=[
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step("step_calc", "Compute", "tool", expected_tool="calculator", expected_output="value", success_criteria="calculator returns a value"),
                        plan_step("step_answer", "Answer", "respond", expected_output="answer", success_criteria="answer returned", depends_on=["step_calc"]),
                    ],
                ),
                json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": expression}}),
                expected,
            ]
        )
        runtime = AgentRuntime(config, model_client=fake_client)
        result = runtime.run_turn(goal)
        rebuilt = runtime.history.rebuild_from_history(result.session_id)
        outputs.append(result.assistant_text)

        assert rebuilt.messages[-1].content == expected
        assert result.assistant_text == expected

    assert outputs == [str(eval(expression)) for expression in expressions]


def test_runtime_rejects_evaluator_override_of_deterministic_verification_failure(make_config, monkeypatch) -> None:
    config = make_config(planner__max_replans=0)
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step(
                        "step_calc",
                        "Compute",
                        "tool",
                        expected_tool="calculator",
                        expected_output="value",
                        success_criteria="calculator returns a value",
                        verification_checks=[
                            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                            {"name": "wrong_result", "check_type": "exact_match", "actual_source": "tool_output.result", "expected": 5},
                        ],
                        required_conditions=["dependencies_completed", "wrong_result"],
                        optional_conditions=[],
                    ),
                    plan_step("step_answer", "Answer", "respond", expected_output="answer", success_criteria="answer returned", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    from swaag.evaluator import EvaluationOutcome
    from swaag.history import HistoryInvariantError
    import swaag.runtime as runtime_module
    from swaag.verification import VerificationOutcome

    monkeypatch.setattr(
        runtime_module,
        "evaluate_verification",
        lambda step, verification: EvaluationOutcome(
            passed=True,
            confidence=1.0,
            reason="forced_override",
            requires_retry=False,
            requires_replan=False,
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_verify_step",
        lambda state, plan, step, artifacts: VerificationOutcome(
            verification_passed=False,
            verification_type_used="composite",
            conditions_met=[],
            conditions_failed=["wrong_result"],
            evidence={"wrong_result": {"actual": 4, "expected": 5}},
            confidence=0.0,
            reason="wrong_result",
            requires_retry=True,
            requires_replan=False,
        ),
    )

    with pytest.raises(HistoryInvariantError):
        runtime.run_turn(goal)


def test_runtime_uses_expected_tool_input_contract_for_profile_optimized_edit_steps(make_config) -> None:
    config = make_config(
        model__profile_name="small_fast",
        model__structured_output_mode="post_validate",
        tools__allow_side_effect_tools=True,
    )
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "sample.py",
                        "operation": "replace_pattern_once",
                        "pattern": "return 0",
                        "replacement": "return 1",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    state.messages = [
        Message(role="user", content="Read sample.py and fix it so return 0 becomes return 1.", created_at="t0"),
        Message(
            role="tool",
            name="read_text",
            content="read_text result: {\"source_ref\":\"sample.py\",\"text\":\"def value():\\n    return 0\\n\"}",
            created_at="t1",
            metadata={"output": {"source_ref": "sample.py", "text": "def value():\n    return 0\n"}},
        ),
    ]
    state.active_plan = plan_from_payload(
        {
            "goal": "Read sample.py and fix it so return 0 becomes return 1.",
            "success_criteria": "fixed",
            "fallback_strategy": "replan",
            "steps": [
                {
                    "step_id": "step_edit",
                    "title": "Fix sample.py",
                    "goal": "Fix sample.py",
                    "kind": "write",
                    "expected_tool": "edit_text",
                    "input_text": "edit_text path: sample.py\nTask: Read sample.py and fix it so return 0 becomes return 1.",
                    "expected_output": "sample.py updated",
                    "expected_outputs": ["sample.py updated"],
                    "done_condition": "tool_result:edit_text",
                    "success_criteria": "sample.py updated",
                    "verification_type": "composite",
                    "verification_checks": [
                        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                        {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                        {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                    ],
                    "required_conditions": ["dependencies_completed", "tool_result_present", "tool_name_matches"],
                    "optional_conditions": [],
                    "fallback_strategy": "replan",
                    "depends_on": [],
                },
                {
                    "step_id": "step_answer",
                    "title": "Answer",
                    "goal": "Answer",
                    "kind": "respond",
                    "expected_tool": "",
                    "input_text": "Respond.",
                    "expected_output": "answer",
                    "expected_outputs": ["answer"],
                    "done_condition": "assistant_response_nonempty",
                    "success_criteria": "answer returned",
                    "verification_type": "composite",
                    "verification_checks": [
                        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
                    ],
                    "required_conditions": ["dependencies_completed", "assistant_text_nonempty"],
                    "optional_conditions": [],
                    "fallback_strategy": "replan",
                    "depends_on": ["step_edit"],
                },
            ],
        },
        available_tools=runtime.tools.tool_names(config),
    )

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "edit_text"
    assert decision.tool_input["pattern"] == "return 0"
    contracts = [request["contract"] for request in fake_client.requests]
    assert contracts[-1] == "tool_input:edit_text"
    assert "subagent_selection" in contracts


def test_runtime_uses_expected_tool_input_contract_for_shell_command_steps(make_config) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:shell_command": [
                json.dumps({"command": "printf inspection", "background": False})
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    goal = "Fix the benchmark issue."
    state.active_plan = create_shell_recovery_plan(goal)

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "shell_command"
    assert isinstance(decision.tool_input["command"], str)
    assert decision.tool_input["command"]
    contracts = [request["contract"] for request in fake_client.requests]
    assert contracts[-1] == "tool_input:shell_command"


def test_runtime_normalizes_trivial_shell_command_into_repo_search(make_config) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:shell_command": [
                json.dumps({"command": "bash", "background": False})
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Known failing tests:\n"
        "- tests/test_demo.py::test_fix\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
    )
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.active_plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "shell_command"
    assert decision.tool_input["command"].startswith("printf 'search_terms:")
    assert "rg -n" in decision.tool_input["command"]
    assert "mathematica_code" in decision.tool_input["command"]


def test_runtime_decomposes_open_ended_answer_into_semantic_units(make_config) -> None:
    config = make_config()
    goal = "Explain the result in two short sections."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step(
                        "step_answer",
                        "Answer the user",
                        "respond",
                        expected_output="Two short sections",
                        success_criteria="The answer is provided in two sections.",
                    ),
                ],
            ),
        ],
        contract_responses={
            "generation_decomposition": [
                json.dumps(
                    {
                        "output_class": "open_ended",
                        "reason": "two bounded semantic sections are clearer",
                        "units": [
                            {"unit_id": "part_1", "title": "Section 1", "instruction": "Write the first section only."},
                            {"unit_id": "part_2", "title": "Section 2", "instruction": "Write the second section only."},
                        ],
                    }
                )
            ],
            "plain_text": ["First section.", "Second section."],
        },
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "First section.\n\nSecond section."
    assert any(event.event_type == "output_decomposition_planned" for event in events)
    assert sum(1 for event in events if event.event_type == "output_unit_generated") == 2
    assert len([request for request in fake_client.requests if request.get("contract") == "plain_text"]) == 2


def test_runtime_uses_overflow_recovery_planning_instead_of_blind_text_continuation(make_config) -> None:
    config = make_config()
    goal = "Write a long structured explanation."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step(
                        "step_answer",
                        "Answer the user",
                        "respond",
                        expected_output="Long explanation",
                        success_criteria="The explanation is complete.",
                    ),
                ],
            ),
        ],
        contract_responses={
            "generation_decomposition": [
                json.dumps(
                    {
                        "output_class": "open_ended",
                        "reason": "start with one draft unit and split only if needed",
                        "units": [
                            {
                                "unit_id": "draft",
                                "title": "Draft answer",
                                "instruction": "Draft the full answer in one unit.",
                            }
                        ],
                    }
                )
            ],
            "plain_text": [
                CompletionResult(
                    text="Partial draft that overflowed.",
                    raw_request={},
                    raw_response={"content": "Partial draft that overflowed."},
                    prompt_tokens=None,
                    completion_tokens=10_000,
                    finish_reason="length",
                ),
                "Recovered section A.",
                "Recovered section B.",
            ],
            "overflow_recovery": [
                json.dumps(
                    {
                        "keep_partial": False,
                        "reason": "split the answer into two smaller semantic units",
                        "next_units": [
                            {"unit_id": "split_a", "title": "Section A", "instruction": "Write section A only."},
                            {"unit_id": "split_b", "title": "Section B", "instruction": "Write section B only."},
                        ],
                    }
                )
            ],
        },
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "Recovered section A.\n\nRecovered section B."
    assert any(event.event_type == "output_overflow_recovery_planned" for event in events)
    assert all("continue this text" not in request.get("prompt", "").lower() for request in fake_client.requests)
