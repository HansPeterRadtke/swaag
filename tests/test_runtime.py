from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import requests

import swaag.runtime as runtime_module
from swaag.evaluator import evaluate_step
from swaag.model import ModelClientError
from swaag.planner import create_shell_recovery_plan, plan_from_payload
from swaag.retrieval.embeddings import SemanticBackendProtocolError
from swaag.runtime import AgentRuntime, BudgetExceededError, FatalSemanticEngineError
from swaag.types import CompletionResult, DecisionOutcome, Message, PlanStep, PromptAnalysis, StrategySelection, ToolExecutionResult
from swaag.verification import VerificationArtifacts

from tests.helpers import FakeModelClient, plan_response, plan_step


class HangingStructuredModelClient(FakeModelClient):
    def select_request_policy(
        self,
        *,
        contract,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ):
        policy = super().select_request_policy(
            contract=contract,
            kind=kind,
            prompt=prompt,
            max_tokens=max_tokens,
            live_mode=live_mode,
        )
        return policy.__class__(
            profile_name=policy.profile_name,
            structured_output_mode=policy.structured_output_mode,
            effective_contract_mode=policy.effective_contract_mode,
            effective_timeout_seconds=1,
            progress_poll_seconds=0.01,
        )

    def send_completion(self, payload: dict[str, Any], *, timeout_seconds: int | None = None) -> CompletionResult:
        del payload, timeout_seconds
        time.sleep(5)
        raise AssertionError("unreachable")


class StreamingHangingStructuredModelClient(HangingStructuredModelClient):
    def select_request_policy(
        self,
        *,
        contract,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ):
        policy = super().select_request_policy(
            contract=contract,
            kind=kind,
            prompt=prompt,
            max_tokens=max_tokens,
            live_mode=live_mode,
        )
        return policy.__class__(
            profile_name=policy.profile_name,
            structured_output_mode=policy.structured_output_mode,
            effective_contract_mode=policy.effective_contract_mode,
            effective_timeout_seconds=0.05,
            progress_poll_seconds=0.01,
        )

    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        stream_callback=None,
    ) -> CompletionResult:
        del payload, timeout_seconds
        for index in range(100):
            if stream_callback is not None:
                stream_callback(
                    {
                        "content": " return total\n",
                        "chunk_index": index,
                        "stop": False,
                        "replayed": False,
                    }
                )
            time.sleep(0.01)
        raise AssertionError("unreachable")


class StreamingThenStallingStructuredModelClient(StreamingHangingStructuredModelClient):
    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        stream_callback=None,
    ) -> CompletionResult:
        del payload, timeout_seconds
        for index in range(3):
            if stream_callback is not None:
                stream_callback(
                    {
                        "content": " partial",
                        "chunk_index": index,
                        "stop": False,
                        "replayed": False,
                    }
                )
            time.sleep(0.01)
        time.sleep(0.2)
        raise AssertionError("unreachable")


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


def test_extract_unconditional_exact_reply_ignores_conditional_reply_clauses(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient(responses=[]))

    assert runtime._extract_unconditional_exact_reply("If tests pass, reply exactly passed.") is None
    assert runtime._extract_unconditional_exact_reply("Read notes.txt and return exactly the full text on line 3.") is None
    assert runtime._extract_unconditional_exact_reply(
        'Read profile.txt. Return exactly this JSON shape with no extra fields: {"name":"...","team":"...","city":"..."}'
    ) is None
    assert (
        runtime._extract_unconditional_exact_reply(
            "Read notes.txt.\nReply exactly beta=2\nDo not add anything else."
        )
        == "beta=2"
    )
    assert runtime._extract_unconditional_exact_reply("Reply with OK only.") == "OK"
    assert runtime._extract_unconditional_exact_reply("Respond with a single word 'OK'") == "OK"
    assert runtime._extract_unconditional_exact_reply("If checks pass, reply with OK only.") is None


def test_runtime_finalizes_unconditional_exact_reply_without_final_model_call(make_config, tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    config = make_config(tools__allow_side_effect_tools=True)
    goal = f"Create {target} containing exactly sum=42 followed by a newline. Reply exactly written."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step(
                        "step_write",
                        "Write result file",
                        "write",
                        expected_tool="write_file",
                        expected_output="file written",
                        success_criteria="the file is written correctly",
                    ),
                    plan_step(
                        "step_answer",
                        "Answer",
                        "respond",
                        expected_output="Final assistant response",
                        success_criteria="the assistant replies to the user",
                        depends_on=["step_write"],
                    ),
                ],
            ),
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "write_file",
                    "tool_input": {"path": str(target), "content": "sum=42\n"},
                }
            ),
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    contracts = [request["contract"] for request in fake_client.requests]

    assert result.assistant_text == "written"
    assert target.read_text(encoding="utf-8") == "sum=42\n"
    assert "plain_text" not in contracts
    assert any(
        event.event_type == "answer_derived" and event.payload.get("source") == "deterministic_finalizer"
        for event in events
    )
    assert any(
        event.event_type == "subagent_selection_resolved"
        and event.payload.get("selection", {}).get("reason") == "deterministic_review_sufficient"
        for event in events
    )


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

    assert "done" in result.assistant_text
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


def test_runtime_skips_result_review_for_inline_evidence_direct_answer(make_config) -> None:
    goal = (
        "Answer this HERB benchmark question using only the evidence below.\n\n"
        "Question: What changed?\n\n"
        "Evidence:\n1. Add AI learning capabilities.\n2. Update SWOT threats."
    )
    fake_client = FakeModelClient(
        contract_responses={
            "prompt_analysis": [json.dumps({
                "task_type": "structured",
                "completeness": "complete",
                "requires_expansion": False,
                "requires_decomposition": False,
                "confidence": 1.0,
                "detected_entities": ["HERB"],
                "detected_goals": ["answer question"],
            })],
            "task_decision": [json.dumps({
                "split_task": False,
                "expand_task": False,
                "ask_user": False,
                "assume_missing": False,
                "generate_ideas": False,
                "direct_response": True,
                "confidence": 1.0,
                "reason": "inline evidence sufficient",
            })],
            "strategy_selection": [json.dumps({
                "task_profile": "reading",
                "strategy_name": "conservative",
                "explore_before_commit": False,
                "tool_chain_depth": 1,
                "verification_intensity": 1.0,
                "reason": "answer directly from evidence",
            })],
            "plain_text": ["Add AI learning capabilities and update SWOT threats."],
        },
    )
    runtime = AgentRuntime(make_config(), model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert "AI learning" in result.assistant_text
    assert not any(
        event.event_type == "subagent_selection_started"
        and event.payload.get("purpose") == "result_review"
        for event in events
    )


def test_runtime_forces_direct_response_for_inline_evidence_even_when_decision_requests_plan(make_config) -> None:
    goal = (
        "Answer this HERB benchmark question using only the evidence below.\n"
        "Answer only the changes suggested by the UX Researcher.\n\n"
        "Question: What are the changes suggested by UX Researcher to improve the Market Research Report?\n\n"
        "Evidence:\n1. UX Researcher: Add a comparison chart.\n2. UX Researcher: Include user feedback."
    )
    fake_client = FakeModelClient(
        contract_responses={
            "prompt_analysis": [json.dumps({
                "task_type": "structured",
                "completeness": "complete",
                "requires_expansion": True,
                "requires_decomposition": False,
                "confidence": 1.0,
                "detected_entities": ["HERB"],
                "detected_goals": ["answer from inline evidence"],
            })],
            "task_decision": [json.dumps({
                "split_task": False,
                "expand_task": True,
                "ask_user": False,
                "assume_missing": False,
                "generate_ideas": False,
                "direct_response": False,
                "confidence": 1.0,
                "reason": "structured task requires expansion",
            })],
            "strategy_selection": [json.dumps({
                "task_profile": "reading",
                "strategy_name": "conservative",
                "explore_before_commit": False,
                "tool_chain_depth": 1,
                "verification_intensity": 1.0,
                "reason": "answer directly from evidence",
            })],
            "plain_text": ["Add a comparison chart and include user feedback."],
        }
    )
    runtime = AgentRuntime(make_config(), model_client=fake_client)

    result = runtime.run_turn(goal)
    contracts = [request["contract"] for request in fake_client.requests]

    assert "comparison chart" in result.assistant_text
    assert "task_expansion" not in contracts
    assert "task_plan" not in contracts


def test_runtime_keeps_direct_response_for_inline_evidence_when_strategy_suggests_read(make_config) -> None:
    config = make_config()
    goal = (
        "Answer this HERB benchmark question using only the evidence below.\n\n"
        "Question: What changed?\n\n"
        "Evidence:\n"
        "1. Add a section on AI learning capabilities.\n"
        "2. Update SWOT threats for rapid technological changes."
    )
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
                        "detected_entities": ["HERB", "evidence"],
                        "detected_goals": ["answer from inline evidence"],
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
                        "reason": "inline evidence is sufficient",
                    }
                )
            ],
            "strategy_selection": [
                json.dumps(
                    {
                        "task_profile": "reading",
                        "strategy_name": "conservative",
                        "explore_before_commit": False,
                        "tool_chain_depth": 1,
                        "verification_intensity": 1.0,
                        "reason": "read the provided evidence and respond",
                    }
                )
            ],
            "plain_text": ["- Add AI learning capabilities.\n- Update SWOT threats for rapid technological changes."],
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    contracts = [request["contract"] for request in fake_client.requests]
    events = runtime.history.read_history(result.session_id)

    assert "AI learning capabilities" in result.assistant_text
    assert "task_plan" not in contracts
    assert not any(event.event_type == "decision_adjusted" for event in events)


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


def test_runtime_progress_polling_enforces_request_timeout(make_config) -> None:
    runtime = AgentRuntime(
        make_config(),
        model_client=HangingStructuredModelClient(),
    )
    runtime._max_model_unavailable_attempts = 0
    state = runtime.create_or_load_session()

    with pytest.raises(ModelClientError, match="semantic_engine_unavailable"):
        runtime._analyze_prompt_frontend(state, "Fix app.py")

    events = runtime.history.read_history(state.session_id)
    assert any(event.event_type == "model_request_progress" for event in events)
    assert not any(event.event_type == "prompt_analyzed" for event in events)


def test_runtime_streaming_progress_polling_enforces_total_request_timeout(make_config) -> None:
    runtime = AgentRuntime(
        make_config(),
        model_client=StreamingHangingStructuredModelClient(),
    )
    runtime._max_model_unavailable_attempts = 0
    state = runtime.create_or_load_session()

    with pytest.raises(ModelClientError, match="semantic_engine_unavailable"):
        runtime._analyze_prompt_frontend(state, "Fix app.py")

    events = runtime.history.read_history(state.session_id)
    assert any(event.event_type == "model_response_chunk" for event in events)
    assert not any(event.event_type == "prompt_analyzed" for event in events)


def test_runtime_streaming_progress_polling_times_out_after_chunk_stream_stalls(make_config) -> None:
    runtime = AgentRuntime(
        make_config(),
        model_client=StreamingThenStallingStructuredModelClient(),
    )
    runtime._max_model_unavailable_attempts = 0
    state = runtime.create_or_load_session()

    with pytest.raises(ModelClientError, match="semantic_engine_unavailable"):
        runtime._analyze_prompt_frontend(state, "Fix app.py")

    events = runtime.history.read_history(state.session_id)
    assert any(event.event_type == "model_response_chunk" for event in events)
    assert any(event.event_type == "model_request_progress" for event in events)
    assert not any(event.event_type == "prompt_analyzed" for event in events)


def test_runtime_recovers_malformed_coding_plan_with_shell_recovery_plan(
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
    (tmp_path / "sample.py").write_text("old\n", encoding="utf-8")
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
                    "tool_name": "edit_text",
                    "tool_input": {
                        "path": "sample.py",
                        "operation": "replace_pattern_once",
                        "pattern": "old",
                        "replacement": "new",
                    },
                }
            ),
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "run_tests",
                    "tool_input": {"command": ["python3", "-c", "print('ok')"]},
                }
            ),
            json.dumps({"action": "respond", "response": "done", "tool_name": "none", "tool_input": {}}),
        ],
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    rebuilt = runtime.history.rebuild_from_history(result.session_id)

    assert "done" in result.assistant_text
    assert rebuilt.active_plan is not None
    assert [step.kind for step in rebuilt.active_plan.steps] == ["read", "write", "tool", "respond"]
    assert [step.expected_tool for step in rebuilt.active_plan.steps[:-1]] == ["shell_command", "edit_text", "run_tests"]
    assert any(
        event.event_type == "plan_repaired"
        and event.payload.get("repair") == "shell_recovery_plan"
        and event.payload.get("reason") == "planner_structured_failure_shell_recovery"
        for event in events
    )
    assert any(event.event_type == "tool_called" and event.payload.get("tool_name") == "shell_command" for event in events)
    assert not any(event.event_type == "fatal_system_error" for event in events)


def test_runtime_recovers_strategy_incompatible_coding_plan_with_shell_recovery_plan(
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
    (tmp_path / "sample.py").write_text("old\n", encoding="utf-8")
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
                    "tool_name": "edit_text",
                    "tool_input": {
                        "path": "sample.py",
                        "operation": "replace_pattern_once",
                        "pattern": "old",
                        "replacement": "new",
                    },
                }
            ),
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "run_tests",
                    "tool_input": {"command": ["python3", "-c", "print('ok')"]},
                }
            ),
            json.dumps({"action": "respond", "response": "done", "tool_name": "none", "tool_input": {}}),
        ],
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert "done" in result.assistant_text
    assert any(
        event.event_type == "plan_repaired"
        and event.payload.get("repair") == "shell_recovery_plan"
        and event.payload.get("error_type") == "StrategyValidationError"
        for event in events
    )
    assert any(event.event_type == "tool_called" and event.payload.get("tool_name") == "shell_command" for event in events)
    assert any(event.event_type == "tool_called" and event.payload.get("tool_name") == "edit_text" for event in events)
    assert any(event.event_type == "tool_called" and event.payload.get("tool_name") == "run_tests" for event in events)


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
    (tmp_path / "sample.py").write_text("old\n", encoding="utf-8")
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
            ],
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "sample.py",
                        "operation": "replace_pattern_once",
                        "pattern": "old",
                        "replacement": "new",
                    }
                )
            ],
            "tool_input:run_tests": [
                json.dumps({"command": ["python3", "-c", "print('ok')"], "background": False})
            ],
        },
        responses=[
            json.dumps({"action": "respond", "response": "done", "tool_name": "none", "tool_input": {}}),
        ],
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    runtime._max_model_unavailable_attempts = 0

    result = runtime.run_turn(user_text)
    events = runtime.history.read_history(result.session_id)
    request_contracts = [request["contract"] for request in fake_client.requests]

    assert "sample.py" in result.assistant_text
    assert "python3 -c" in result.assistant_text
    assert "tests now pass" in result.assistant_text
    assert "task_plan" not in request_contracts
    assert "subagent_selection" not in request_contracts
    assert "tool_input:shell_command" in request_contracts
    assert "tool_input:edit_text" in request_contracts
    assert "tool_input:run_tests" in request_contracts
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


def test_runtime_falls_back_to_deterministic_benchmark_code_fix_frontend_when_semantic_engine_is_unavailable(
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
    package_dir = tmp_path / "pkg_492"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "stats.py").write_text("old\n", encoding="utf-8")
    (tmp_path / "test_pkg_492.py").write_text("import unittest\n", encoding="utf-8")
    user_text = (
        f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
        "Inspect the module and test first, keep the public function name unchanged, and do not modify the test file."
    )
    fake_client = FakeModelClient(
        contract_responses={
            "prompt_analysis": [requests.ConnectionError("llm down")],
            "task_decision": [requests.ConnectionError("llm down")],
            "strategy_selection": [requests.ConnectionError("llm down")],
            "tool_input:shell_command": [
                json.dumps(
                    {
                        "command": (
                            "printf 'source_file: ./pkg_492/stats.py\\n'; "
                            "sed -n '1,220p' ./pkg_492/stats.py; "
                            "printf 'test_file: ./test_pkg_492.py\\n'; "
                            "sed -n '1,220p' ./test_pkg_492.py"
                        ),
                        "background": False,
                    }
                ),
            ],
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "pkg_492/stats.py",
                        "operation": "replace_pattern_once",
                        "pattern": "old",
                        "replacement": "new",
                    }
                )
            ],
            "tool_input:run_tests": [
                json.dumps({"command": ["python3", "-c", "print('ok')"], "background": False})
            ],
        },
        responses=[
            json.dumps({"action": "respond", "response": "done", "tool_name": "none", "tool_input": {}}),
        ],
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    runtime._max_model_unavailable_attempts = 0

    result = runtime.run_turn(user_text)
    events = runtime.history.read_history(result.session_id)
    request_contracts = [request["contract"] for request in fake_client.requests]

    assert "pkg_492/stats.py" in result.assistant_text
    assert "python3 -c" in result.assistant_text
    assert "tests now pass" in result.assistant_text
    assert "task_plan" not in request_contracts
    assert "subagent_selection" not in request_contracts
    assert "tool_input:shell_command" in request_contracts
    assert "tool_input:edit_text" in request_contracts
    assert "tool_input:run_tests" in request_contracts
    assert any(
        event.event_type == "prompt_analyzed"
        and event.payload.get("source") == "deterministic_code_fix_fallback"
        for event in events
    )
    assert any(
        event.event_type == "decision_made"
        and event.payload.get("source") == "deterministic_code_fix_fallback"
        for event in events
    )
    assert any(
        event.event_type == "plan_repaired"
        and event.payload.get("reason") == "benchmark_prompt_shell_recovery_seed"
        for event in events
    )


def test_runtime_locks_benchmark_code_fix_frontend_to_deterministic_shell_recovery(
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
    package_dir = tmp_path / "pkg_492"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "stats.py").write_text("old\n", encoding="utf-8")
    (tmp_path / "test_pkg_492.py").write_text("import unittest\n", encoding="utf-8")
    user_text = (
        f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
        "Inspect the module and test first, keep the public function name unchanged, and do not modify the test file."
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
            "strategy_selection": [
                json.dumps(
                    {
                        "task_profile": "generic",
                        "strategy_name": "conservative",
                        "explore_before_commit": False,
                        "tool_chain_depth": 1,
                        "verification_intensity": 0.8,
                        "reason": "generic",
                    }
                )
            ],
            "tool_input:shell_command": [
                json.dumps(
                    {
                        "command": (
                            "printf 'source_file: ./pkg_492/stats.py\\n'; "
                            "sed -n '1,220p' ./pkg_492/stats.py; "
                            "printf 'test_file: ./test_pkg_492.py\\n'; "
                            "sed -n '1,220p' ./test_pkg_492.py"
                        ),
                        "background": False,
                    }
                ),
            ],
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "pkg_492/stats.py",
                        "operation": "replace_pattern_once",
                        "pattern": "old",
                        "replacement": "new",
                    }
                )
            ],
            "tool_input:run_tests": [
                json.dumps({"command": ["python3", "-c", "print('ok')"], "background": False})
            ],
        },
        responses=[
            json.dumps({"action": "respond", "response": "done", "tool_name": "none", "tool_input": {}}),
        ],
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(user_text)
    events = runtime.history.read_history(result.session_id)
    request_contracts = [request["contract"] for request in fake_client.requests]

    assert "pkg_492/stats.py" in result.assistant_text
    assert "python3 -c" in result.assistant_text
    assert "tests now pass" in result.assistant_text
    assert "task_decision" not in request_contracts
    assert "strategy_selection" not in request_contracts
    assert "task_plan" not in request_contracts
    assert "subagent_selection" not in request_contracts
    assert any(
        event.event_type == "plan_repaired"
        and event.payload.get("reason") == "benchmark_prompt_shell_recovery_seed"
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


def test_shell_recovery_plan_uses_composite_answer_verification() -> None:
    plan = create_shell_recovery_plan("Fix the benchmark issue.")

    answer_step = plan.steps[-1]

    assert answer_step.title == "Report result"
    assert answer_step.verification_type == "composite"
    assert answer_step.required_conditions == ["dependencies_completed", "assistant_text_nonempty"]
    assert answer_step.verification_checks == [
        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
        {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
    ]


def test_shell_recovery_plan_adds_patch_step_per_named_source_hint() -> None:
    plan = create_shell_recovery_plan(
        "Fix the benchmark issue.",
        source_hints=["pkg_850/tokenizer.py", "pkg_850/normalizer.py", "pkg_850/tokenizer.py"],
    )

    assert [step.title for step in plan.steps] == [
        "Inspect failing area",
        "Patch source: pkg_850/tokenizer.py",
        "Patch source: pkg_850/normalizer.py",
        "Verify targeted test",
        "Report result",
    ]
    assert plan.steps[2].depends_on == [plan.steps[1].step_id]
    assert plan.steps[3].depends_on == [plan.steps[2].step_id]


def test_runtime_deterministically_finishes_shell_recovery_code_fix_response_without_model(make_config) -> None:
    config = make_config()
    fake_client = FakeModelClient()
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    goal = (
        "Repository root: /tmp/workspace. Fix `pkg_850/stats.py` so `test_pkg_850.py` passes. "
        "Inspect the module and test first, keep the public function name unchanged, and do not modify the test file. "
        "After the tests pass, give a short plain-language summary of the repair."
    )
    plan = create_shell_recovery_plan(goal)
    for step in plan.steps[:-1]:
        step.status = "completed"
    plan.current_step_id = plan.steps[-1].step_id
    state.active_plan = plan
    state.environment.workspace.root = "/tmp/workspace"
    state.environment.workspace.cwd = "/tmp/workspace"
    state.environment.workspace.modified_files = ["pkg_850/tokenizer.py", "pkg_850/normalizer.py"]
    runtime._record_message(state, Message(role="user", content=goal, created_at="t0"))
    runtime._record_message(
        state,
        Message(
            role="tool",
            name="run_tests",
            content="run_tests result",
            created_at="t1",
            metadata={
                "output": {
                    "command": ["python3", "-m", "unittest", "-q", "test_pkg_850_pipeline.py"],
                    "cwd": "/tmp/workspace",
                    "exit_code": 0,
                    "stdout": "OK",
                    "stderr": "",
                    "passed": True,
                    "background": False,
                }
            },
        ),
    )

    answer = runtime._deterministic_answer(state)

    assert answer is not None
    assert "pkg_850/tokenizer.py" in answer
    assert "pkg_850/normalizer.py" in answer
    assert "python3 -m unittest -q test_pkg_850_pipeline.py" in answer
    assert "tests now pass" in answer
    completed, failed = runtime._finalize_answer_step(state, answer)
    assert (completed, failed) == (True, False)
    assert fake_client.requests == []


def test_runtime_multifile_shell_recovery_targets_remaining_source_hint_after_first_fix(
    make_config,
    tmp_path: Path,
) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    runtime = AgentRuntime(config, model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    package_dir = tmp_path / "pkg_850"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "tokenizer.py").write_text(
        "def tokenize(text: str) -> list[str]:\n"
        "    return [t.strip() for t in text.split('|')]\n",
        encoding="utf-8",
    )
    (package_dir / "normalizer.py").write_text(
        "from pkg_850.tokenizer import tokenize\n\n\n"
        "def normalize(text: str) -> list[str]:\n"
        "    return [t.upper() for t in tokenize(text)]\n",
        encoding="utf-8",
    )
    (tmp_path / "test_pkg_850_pipeline.py").write_text(
        "from pkg_850.normalizer import normalize\n"
        "assert normalize('item-01|item-02|item-03') == ['item-01', 'item-02', 'item-03']\n",
        encoding="utf-8",
    )
    user_text = (
        f"Repository root: {tmp_path}. Fix the two bugs in the `pkg_850` text pipeline. "
        "Inspect `pkg_850/tokenizer.py` and `pkg_850/normalizer.py` — each has an implementation error. "
        "Do not modify `pkg_850/pipeline.py` or `test_pkg_850_pipeline.py`. "
        "Run `python3 -m unittest -q test_pkg_850_pipeline.py` to verify both fixes, then summarize what changed in each module."
    )
    plan = create_shell_recovery_plan(
        "Fix the benchmark issue.",
        source_hints=["pkg_850/tokenizer.py", "pkg_850/normalizer.py"],
    )
    plan.steps[0].status = "completed"
    plan.steps[1].status = "completed"
    plan.steps[2].status = "running"
    plan.current_step_id = plan.steps[2].step_id
    state.active_plan = plan
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.environment.shell.cwd = str(tmp_path)
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.messages.append(
        Message(
            role="tool",
            name="shell_command",
            content="inspection",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": (
                        "source_file: ./pkg_850/tokenizer.py\n"
                        "def tokenize(text: str) -> list[str]:\n"
                        "    return text.split(',')\n"
                        "source_file: ./pkg_850/normalizer.py\n"
                        "from pkg_850.tokenizer import tokenize\n\n\n"
                        "def normalize(text: str) -> list[str]:\n"
                        "    return [t.upper() for t in tokenize(text)]\n"
                        "test_file: ./test_pkg_850_pipeline.py\n"
                        "assert normalize('item-01|item-02|item-03') == ['item-01', 'item-02', 'item-03']\n"
                    )
                }
            },
        )
    )

    payload = runtime._deterministic_expected_tool_input(state, plan.steps[2])

    assert payload == {
        "path": "pkg_850/normalizer.py",
        "operation": "replace_pattern_once",
        "pattern": ".upper()",
        "replacement": ".lower()",
    }


def test_runtime_shell_recovery_source_hints_exclude_do_not_modify_files(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient())
    prompt = (
        "Repository root: /tmp/workspace. Fix the two bugs in the `pkg_850` text pipeline. "
        "Inspect `pkg_850/tokenizer.py` and `pkg_850/normalizer.py`. "
        "Do not modify `pkg_850/pipeline.py` or `test_pkg_850_pipeline.py`."
    )

    assert runtime._shell_recovery_source_hints(prompt) == [
        "pkg_850/tokenizer.py",
        "pkg_850/normalizer.py",
    ]


def test_runtime_strategy_selection_prompt_names_required_fields_and_prefers_standard_mode(make_config) -> None:
    runtime = AgentRuntime(
        make_config(),
        model_client=FakeModelClient(
            contract_responses={
                "strategy_selection": [
                    json.dumps(
                        {
                            "task_profile": "coding",
                            "strategy_name": "conservative",
                            "explore_before_commit": True,
                            "tool_chain_depth": 2,
                            "verification_intensity": 1.0,
                            "reason": "editing task",
                        }
                    )
                ]
            }
        ),
    )
    state = runtime.create_or_load_session()

    runtime._select_strategy_frontend(
        state,
        "Fix app.py",
        PromptAnalysis(
            task_type="structured",
            completeness="complete",
            requires_expansion=False,
            requires_decomposition=False,
            confidence=1.0,
            detected_entities=["app.py"],
            detected_goals=["fix"],
        ),
        DecisionOutcome(
            split_task=False,
            expand_task=False,
            ask_user=False,
            assume_missing=False,
            generate_ideas=False,
            confidence=1.0,
            reason="planning required",
            direct_response=False,
            execution_mode="full_plan",
            preferred_tool_name="",
        ),
    )

    events = [
        event for event in runtime.history.read_history(state.session_id)
        if event.event_type == "prompt_built" and event.payload.get("kind") == "strategy"
    ]

    assert events[0].payload["prompt_mode"] == "standard"
    prompt = events[0].payload["prompt"]
    assert "keys task_profile, strategy_name, explore_before_commit, tool_chain_depth, verification_intensity, and reason" in prompt
    assert "explore_before_commit means inspect/research before editing or committing to a fix" in prompt
    assert "verification_intensity is the amount of checking to do" in prompt


def test_runtime_subagent_selection_prompt_names_required_fields_and_prefers_standard_mode(make_config) -> None:
    runtime = AgentRuntime(
        make_config(),
        model_client=FakeModelClient(
            contract_responses={
                "subagent_selection": [
                    json.dumps(
                        {
                            "spawn": False,
                            "subagent_type": "none",
                            "reason": "main agent is sufficient",
                            "focus": "",
                        }
                    )
                ]
            }
        ),
    )
    state = runtime.create_or_load_session()

    runtime._select_subagent_frontend(
        state,
        goal="Fix app.py",
        purpose="Inspect whether a coding specialist is needed.",
        candidate_types=["code", "verification"],
        detail_lines=["step=inspect"],
    )

    events = [
        event for event in runtime.history.read_history(state.session_id)
        if event.event_type == "prompt_built" and event.payload.get("kind") == "subagent_selection"
    ]

    assert events[0].payload["prompt_mode"] == "standard"
    prompt = events[0].payload["prompt"]
    assert "keys spawn, subagent_type, reason, and focus" in prompt
    assert "subagent_type must be one available specialist or 'none'" in prompt
    assert "focus is the short specialist brief" in prompt


def test_runtime_generation_decomposition_prompt_names_required_fields_and_prefers_standard_mode(make_config) -> None:
    runtime = AgentRuntime(
        make_config(),
        model_client=FakeModelClient(
            contract_responses={
                "generation_decomposition": [
                    json.dumps(
                        {
                            "output_class": "open_ended",
                            "reason": "split long answer",
                            "units": [
                                {
                                    "unit_id": "u1",
                                    "title": "Part 1",
                                    "instruction": "Write the first section.",
                                }
                            ],
                        }
                    )
                ]
            }
        ),
    )
    state = runtime.create_or_load_session()
    runtime._record_message(state, Message(role="user", content="Write a long answer", created_at="t1"))

    runtime._plan_answer_generation_units(state)

    events = [
        event for event in runtime.history.read_history(state.session_id)
        if event.event_type == "prompt_built" and event.payload.get("kind") == "generation_decomposition"
    ]

    assert events[0].payload["prompt_mode"] == "standard"
    prompt = events[0].payload["prompt"]
    assert "keys output_class, reason, and units" in prompt
    assert "units is an array of generation units" in prompt
    assert "unit_id, title, and instruction" in prompt


def test_runtime_overflow_recovery_prompt_names_required_fields_and_prefers_standard_mode(make_config) -> None:
    runtime = AgentRuntime(
        make_config(),
        model_client=FakeModelClient(
            contract_responses={
                "overflow_recovery": [
                    json.dumps(
                        {
                            "keep_partial": True,
                            "reason": "partial text is safe",
                            "next_units": [
                                {
                                    "unit_id": "u2",
                                    "title": "Part 2",
                                    "instruction": "Finish the answer.",
                                }
                            ],
                        }
                    )
                ]
            }
        ),
    )
    state = runtime.create_or_load_session()
    runtime._record_message(state, Message(role="user", content="Write a long answer", created_at="t1"))

    runtime._recover_overflow_unit(
        state,
        unit={"unit_id": "u1", "title": "Part 1", "instruction": "Write the first section."},
        partial_text="Partial answer",
    )

    events = [
        event for event in runtime.history.read_history(state.session_id)
        if event.event_type == "prompt_built" and event.payload.get("kind") == "overflow_recovery"
    ]

    assert events[0].payload["prompt_mode"] == "standard"
    prompt = events[0].payload["prompt"]
    assert "keys keep_partial, reason, and next_units" in prompt
    assert "keep_partial tells whether the existing partial text is safe to keep verbatim" in prompt
    assert "next_units is the remaining work split into smaller unit objects" in prompt


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


def test_runtime_falls_back_when_expected_tool_input_contract_returns_malformed_json(make_config) -> None:
    config = make_config(
        model__profile_name="small_fast",
        model__structured_output_mode="post_validate",
        tools__allow_side_effect_tools=True,
    )
    fake_client = FakeModelClient(
        responses=[
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "edit_text",
                    "tool_input": {
                        "path": "sample.py",
                        "operation": "replace_pattern_once",
                        "pattern": "return 0",
                        "replacement": "return 1",
                    },
                }
            )
        ],
        contract_responses={
            "tool_input:edit_text": [
                '{"path":"sample.py","operation":"replace_pattern_once","pattern":"return 0"'
            ]
        },
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    state.messages = [
        Message(role="user", content="Read sample.py and fix it so return 0 becomes return 1.", created_at="t0"),
        Message(
            role="tool",
            name="read_text",
            content='read_text result: {"source_ref":"sample.py","text":"def value():\\n    return 0\\n"}',
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
    events = runtime.history.read_history(state.session_id)

    assert decision.tool_name == "edit_text"
    assert decision.tool_input["replacement"] == "return 1"
    contracts = [request["contract"] for request in fake_client.requests]
    assert "tool_input:edit_text" in contracts
    assert contracts[-1] == "tool_decision"
    assert any(event.event_type == "error" and event.payload.get("operation") == "tool_input" for event in events)
    assert not any(event.event_type == "fatal_system_error" for event in events)


def test_runtime_normalizes_general_decision_tool_input_for_active_edit_step(make_config) -> None:
    config = make_config(
        model__profile_name="small_fast",
        model__structured_output_mode="post_validate",
        tools__allow_side_effect_tools=True,
    )
    fake_client = FakeModelClient(
        responses=[
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "edit_text",
                    "tool_input": {
                        "path": "/tmp/work/mathematica.py",
                        "operation": "replace_pattern_once",
                        "pattern": "return 0",
                        "replacement": "return 1",
                    },
                }
            )
        ],
        contract_responses={
            "tool_input:edit_text": [
                '{"path":"sample.py","operation":"replace_pattern_once","pattern":"return 0"'
            ]
        },
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    state.messages = [
        Message(role="user", content="Fix the benchmark issue.", created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="inspection",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": "test_file: ./sympy/printing/tests/test_mathematica.py\nsource_file: ./sympy/printing/mathematica.py\n"
                }
            },
        ),
        Message(
            role="tool",
            name="read_text",
            content='read_text result: {"source_ref":"sympy/printing/mathematica.py","text":"def value():\\n    return 0\\n"}',
            created_at="t2",
            metadata={"output": {"source_ref": "sympy/printing/mathematica.py", "text": "def value():\n    return 0\n"}},
        ),
    ]
    state.active_plan = create_shell_recovery_plan("Fix the benchmark issue.")
    state.active_plan.current_step_id = "step_patch_source"
    state.active_plan.steps[0].status = "completed"
    state.active_plan.steps[1].status = "running"

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "edit_text"
    assert decision.tool_input["path"] == "sympy/printing/mathematica.py"
    contracts = [request["contract"] for request in fake_client.requests]
    assert "tool_input:edit_text" in contracts
    assert contracts[-1] == "tool_decision"


def test_runtime_enforces_expected_shell_command_when_fallback_decision_picks_wrong_tool(make_config) -> None:
    config = make_config(
        model__profile_name="small_fast",
        model__structured_output_mode="post_validate",
        tools__allow_side_effect_tools=True,
    )
    fake_client = FakeModelClient(
        responses=[
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "edit_text",
                    "tool_input": {
                        "path": "/tmp/work/mathematica.py",
                        "operation": "replace_pattern_once",
                        "pattern": "wrong",
                    },
                }
            )
        ],
        contract_responses={
            "tool_input:shell_command": [
                '{"command":"printf inspection"'
            ]
        },
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    state.messages = [Message(role="user", content="Known failing tests:\n- test_Function", created_at="t0")]
    state.active_plan = create_shell_recovery_plan("Fix the benchmark issue.")

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "shell_command"
    assert isinstance(decision.tool_input["command"], str)
    assert decision.tool_input["command"].startswith("printf 'search_terms: test_Function")
    contracts = [request["contract"] for request in fake_client.requests]
    assert "tool_input:shell_command" in contracts
    assert contracts[-1] == "tool_decision"


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
    prompt = fake_client.requests[-1]["prompt"]
    assert "Step instructions:" in prompt
    assert "Print exact source and test evidence before any edit" in prompt
    assert "the next patch step needs actual source lines" in prompt


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
    assert "test_fix" in decision.tool_input["command"]
    assert "mathematica_code" not in decision.tool_input["command"]
    assert "-F" in decision.tool_input["command"]


def test_runtime_shell_search_previews_bare_test_and_source_hints(make_config) -> None:
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
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.active_plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "shell_command"
    command = decision.tool_input["command"]
    assert "test_file=''" in command
    assert "test_file=$(" in command
    assert "def test_Function(" in command
    assert "source_file=''" in command
    assert "source_file=$(" in command
    assert "mathematica_code" in command
    assert "mathematica.py" in command
    assert "sed -n '1,220p'" in command
    assert "matches=$(rg -n -F" in command
    assert "head -n 20" in command


def test_runtime_prefers_recent_source_hint_for_edit_text_step(make_config) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "tests/test_Function.py",
                        "operation": "replace_pattern_once",
                        "pattern": "old",
                        "replacement": "new",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": "test_file: ./sympy/printing/tests/test_mathematica.py\nsource_file: ./sympy/printing/mathematica.py\n",
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
        Message(
            role="tool",
            name="read_text",
            content='read_text result: {"source_ref":"sympy/printing/mathematica.py","text":"def value():\\n    old\\n"}',
            created_at="t2",
            metadata={"output": {"source_ref": "sympy/printing/mathematica.py", "text": "def value():\n    old\n"}},
        ),
    ]
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "edit_text"
    assert decision.tool_input["path"] == "sympy/printing/mathematica.py"


def test_runtime_edit_text_prompt_includes_recent_inspection_evidence(make_config) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "sympy/printing/mathematica.py",
                        "operation": "replace_pattern_once",
                        "pattern": "old",
                        "replacement": "new",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": "test_file: ./sympy/printing/tests/test_mathematica.py\nsource_file: ./sympy/printing/mathematica.py\ndef mathematica_code(expr):\n    return expr.func.__name__ + '(%s)' % self.stringify(expr.args, ', ')\n",
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
        Message(
            role="tool",
            name="read_text",
            content=(
                "read_text result: "
                "{\"source_ref\":\"sympy/printing/mathematica.py\","
                "\"text\":\"def mathematica_code(expr):\\n"
                "    old\\ndef mathematica_code(expr):\\n    return expr.func.__name__ + '(%%s)' %% self.stringify(expr.args, ', ')\\n\"}"
            ),
            created_at="t2",
            metadata={
                "output": {
                    "source_ref": "sympy/printing/mathematica.py",
                    "text": "def mathematica_code(expr):\n    old\ndef mathematica_code(expr):\n    return expr.func.__name__ + '(%%s)' %% self.stringify(expr.args, ', ')\n",
                }
            },
        ),
        Message(
            role="tool",
            name="read_text",
            content='read_text result: {"source_ref":"sympy/printing/mathematica.py","text":"def mathematica_code(expr):\\n    old\\n"}',
            created_at="t2",
            metadata={"output": {"source_ref": "sympy/printing/mathematica.py", "text": "def mathematica_code(expr):\n    old\n"}},
        ),
    ]
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "edit_text"
    prompt = fake_client.requests[-1]["prompt"]
    assert "Recent inspection evidence:" in prompt
    assert "source_file: ./sympy/printing/mathematica.py" in prompt
    assert "def mathematica_code(expr):" in prompt


def test_runtime_edit_text_prompt_prefers_source_evidence_over_long_test_preview(make_config) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "sympy/printing/mathematica.py",
                        "operation": "replace_pattern_once",
                        "pattern": "old",
                        "replacement": "new",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    long_test_preview = "test_file: ./sympy/printing/tests/test_mathematica.py\n" + ("assert something\n" * 200)
    source_preview = "source_file: ./sympy/printing/mathematica.py\ndef mathematica_code(expr):\n    return old\n"
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": long_test_preview + source_preview,
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
        Message(
            role="tool",
            name="read_text",
            content=(
                "read_text result: "
                "{\"source_ref\":\"sympy/printing/mathematica.py\","
                "\"text\":\"def mathematica_code(expr):\\n    return old\\n\"}"
            ),
            created_at="t2",
            metadata={
                "output": {
                    "source_ref": "sympy/printing/mathematica.py",
                    "text": "def mathematica_code(expr):\n    return old\n",
                }
            },
        ),
    ]
    state.active_plan = plan

    runtime._decide(state)

    prompt = fake_client.requests[-1]["prompt"]
    evidence = prompt.split("Recent inspection evidence:\n", 1)[1]
    assert evidence.startswith("source_file: ./sympy/printing/mathematica.py")
    assert "def mathematica_code(expr):" in evidence


def test_runtime_edit_text_prompt_omits_duplicate_recent_results_context(make_config) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "sympy/printing/mathematica.py",
                        "operation": "replace_pattern_once",
                        "pattern": "return old",
                        "replacement": "return new",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": (
                        "test_file: ./sympy/printing/tests/test_mathematica.py\n"
                        "source_file: ./sympy/printing/mathematica.py\n"
                        "def mathematica_code(expr):\n"
                        "    return old\n"
                    ),
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
        Message(
            role="tool",
            name="read_text",
            content=(
                "read_text result: "
                "{\"source_ref\":\"sympy/printing/mathematica.py\","
                "\"text\":\"def mathematica_code(expr):\\n    return old\\n\"}"
            ),
            created_at="t2",
            metadata={
                "output": {
                    "source_ref": "sympy/printing/mathematica.py",
                    "text": "def mathematica_code(expr):\n    return old\n",
                }
            },
        ),
    ]
    state.active_plan = plan

    runtime._decide(state)

    prompt = fake_client.requests[-1]["prompt"]
    assert "Recent inspection evidence:" in prompt
    assert "Recent test evidence:" in prompt
    assert "Recent results:" not in prompt


def test_runtime_edit_text_prompt_uses_workspace_source_excerpt_for_long_files(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "sympy/printing/mathematica.py",
                        "operation": "replace_pattern_once",
                        "pattern": "TAIL_MARKER = 2",
                        "replacement": "TAIL_MARKER = 3",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    source_path = tmp_path / "sympy" / "printing" / "mathematica.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "HEADER_MARKER = 1\n"
        + ("mid_line = 1\n" * 300)
        + "TAIL_MARKER = 2\n"
    )
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": "test_file: ./sympy/printing/tests/test_mathematica.py\nsource_file: ./sympy/printing/mathematica.py\n",
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
    ]
    state.active_plan = plan

    runtime._decide(state)

    prompt = fake_client.requests[-1]["prompt"]
    evidence = prompt.split("Recent inspection evidence:\n", 1)[1]
    assert evidence.startswith("source_file: ./sympy/printing/mathematica.py")
    assert "HEADER_MARKER = 1" in evidence
    assert "TAIL_MARKER = 2" in evidence


def test_runtime_edit_text_prompt_includes_all_named_source_hints_and_recent_test_evidence(
    make_config,
    tmp_path: Path,
) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "pkg_545/core.py",
                        "operation": "replace_pattern_once",
                        "pattern": "return 30",
                        "replacement": "return 33",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    workspace = tmp_path
    package_dir = workspace / "pkg_545"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "core.py").write_text("def base_value() -> int:\n    return 30\n", encoding="utf-8")
    (package_dir / "calc.py").write_text(
        "from pkg_545.core import base_value\n\n\ndef total() -> int:\n    return base_value() + 10\n",
        encoding="utf-8",
    )
    (package_dir / "report.py").write_text(
        "import json\nfrom pathlib import Path\n\n\ndef describe() -> str:\n    settings = json.loads(Path('release_settings.json').read_text(encoding='utf-8'))\n    return f\"{settings['label']}:broken:tax={settings['tax_rate']}\"\n",
        encoding="utf-8",
    )
    (package_dir / "compat.py").write_text(
        "from pkg_545.report import describe\n\n\ndef release_summary() -> dict[str, str]:\n    label, total, tax = describe().split(':')\n    return {'label': label, 'total': total, 'tax': tax}\n",
        encoding="utf-8",
    )
    (workspace / "release_settings.json").write_text('{"label": "release-20", "tax_rate": 5}\n', encoding="utf-8")
    (workspace / "test_pkg_545_unit.py").write_text("assert True\n", encoding="utf-8")
    (workspace / "test_pkg_545_compat.py").write_text("assert True\n", encoding="utf-8")
    (workspace / "test_pkg_545_artifacts.py").write_text(
        "import pathlib\n\nnote = pathlib.Path('release_notes.txt').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    state.environment.workspace.root = str(workspace)
    state.environment.workspace.cwd = str(workspace)
    state.environment.shell.cwd = str(workspace)
    state.messages = [
        Message(
            role="user",
            content=(
                f"Repository root: {workspace}. Repair the `pkg_545` release flow. "
                "Inspect `pkg_545/core.py`, `pkg_545/calc.py`, `pkg_545/report.py`, `pkg_545/compat.py`, "
                "`release_settings.json`, and the three test files. "
                "Fix the implementation and the generated release artifact without editing the tests or the settings file. "
                "Run `python3 -m unittest -q test_pkg_545_unit.py test_pkg_545_compat.py test_pkg_545_artifacts.py` before answering."
            ),
            created_at="t0",
        ),
        Message(
            role="tool",
            name="shell_command",
            content="inspection",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": (
                        "source_file: ./pkg_545/core.py\n"
                        "def base_value() -> int:\n    return 30\n"
                        "source_file: ./pkg_545/calc.py\n"
                        "def total() -> int:\n    return base_value() + 10\n"
                        "test_file: ./test_pkg_545_artifacts.py\n"
                        "note = pathlib.Path('release_notes.txt').read_text(encoding='utf-8')\n"
                    )
                }
            },
        ),
    ]
    plan = create_shell_recovery_plan("Fix the benchmark issue.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    state.active_plan = plan

    runtime._decide(state)

    prompt = fake_client.requests[-1]["prompt"]
    assert "source_file: ./pkg_545/core.py" in prompt
    assert "source_file: ./pkg_545/calc.py" in prompt
    assert "source_file: ./pkg_545/report.py" in prompt
    assert "source_file: ./pkg_545/compat.py" in prompt
    assert "source_file: ./release_settings.json" in prompt
    assert "Recent test evidence:" in prompt
    assert "test_file: ./test_pkg_545_artifacts.py" in prompt
    assert "release_notes.txt" in prompt


def test_runtime_edit_text_prompt_focuses_known_mapping_excerpt(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "sympy/printing/mathematica.py",
                        "operation": "replace_pattern_once",
                        "pattern": '"conjugate": [(lambda x: True, "Conjugate")],',
                        "replacement": '"conjugate": [(lambda x: True, "Conjugate")],\n    "Max": [(lambda *x: True, "Max")],',
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    source_path = tmp_path / "sympy" / "printing" / "mathematica.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "\"\"\"\nMathematica code printer\n\"\"\"\n\n"
        "known_functions = {\n"
        '    "exp": [(lambda x: True, "Exp")],\n'
        '    "conjugate": [(lambda x: True, "Conjugate")],\n'
        "}\n\n"
        "class MCodePrinter(object):\n"
        "    pass\n\n"
        "def _print_Function(expr):\n"
        "    return expr.func.__name__ + \"[%s]\"\n"
        + ("padding = 1\n" * 400)
    )
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": "test_file: ./sympy/printing/tests/test_mathematica.py\nsource_file: ./sympy/printing/mathematica.py\n",
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
    ]
    state.active_plan = plan

    runtime._decide(state)

    prompt = fake_client.requests[-1]["prompt"]
    evidence = prompt.split("Recent inspection evidence:\n", 1)[1]
    assert "known_functions = {" in evidence
    assert '"conjugate": [(lambda x: True, "Conjugate")]' in evidence
    assert "def _print_Function(expr):" not in evidence
    assert "padding = 1" not in evidence


def test_runtime_edit_text_retry_instruction_mentions_single_entry_anchor(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                '{"path":"sympy/printing/mathematica.py","operation":"replace_pattern_once","pattern":"known_functions"',
                json.dumps(
                    {
                        "path": "sympy/printing/mathematica.py",
                        "operation": "replace_pattern_once",
                        "pattern": '"conjugate": [(lambda x: True, "Conjugate")],',
                        "replacement": '"conjugate": [(lambda x: True, "Conjugate")],\n    "Max": [(lambda *x: True, "Max")],',
                    }
                ),
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    source_path = tmp_path / "sympy" / "printing" / "mathematica.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        'known_functions = {\n'
        '    "conjugate": [(lambda x: True, "Conjugate")],\n'
        '}\n'
    )
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": "test_file: ./sympy/printing/tests/test_mathematica.py\nsource_file: ./sympy/printing/mathematica.py\n",
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
    ]
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.active_plan = plan

    runtime._decide(state)

    edit_requests = [request for request in fake_client.requests if request.get("contract") == "tool_input:edit_text"]
    assert len(edit_requests) == 2
    assert "use one existing nearby entry line as `pattern`" in edit_requests[-1]["prompt"]


def test_runtime_edit_text_prompt_prefers_focused_excerpt_even_for_short_source(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "sympy/printing/mathematica.py",
                        "operation": "replace_pattern_once",
                        "pattern": '"conjugate": [(lambda x: True, "Conjugate")],',
                        "replacement": '"conjugate": [(lambda x: True, "Conjugate")],\n    "Max": [(lambda *x: True, "Max")],',
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    source_path = tmp_path / "sympy" / "printing" / "mathematica.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "\"\"\"\nMathematica code printer\n\"\"\"\n\n"
        "from __future__ import print_function, division\n\n"
        "known_functions = {\n"
        '    "exp": [(lambda x: True, "Exp")],\n'
        '    "conjugate": [(lambda x: True, "Conjugate")],\n'
        "}\n\n"
        "_default_settings = {\n"
        "    'precision': 15,\n"
        "    'human': True,\n"
        "}\n\n"
        "def mathematica_code(expr, **settings):\n"
        "    return expr\n"
    )
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": "test_file: ./sympy/printing/tests/test_mathematica.py\nsource_file: ./sympy/printing/mathematica.py\n",
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
    ]
    state.active_plan = plan

    runtime._decide(state)

    prompt = fake_client.requests[-1]["prompt"]
    evidence = prompt.split("Recent inspection evidence:\n", 1)[1]
    assert '"conjugate": [(lambda x: True, "Conjugate")]' in evidence
    assert "_default_settings = {" not in evidence
    assert "def mathematica_code" not in evidence


def test_runtime_retries_expected_edit_text_after_invalid_json(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                '{"path":"sympy/printing/mathematica.py","operation":"replace_pattern_once","pattern":"known_functions"',
                json.dumps(
                    {
                        "path": "sympy/printing/mathematica.py",
                        "operation": "replace_pattern_once",
                        "pattern": '"conjugate": [(lambda x: True, "Conjugate")],',
                        "replacement": '"conjugate": [(lambda x: True, "Conjugate")],\n    "Max": [(lambda *x: True, "Max")],',
                    }
                ),
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    source_path = tmp_path / "sympy" / "printing" / "mathematica.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        'known_functions = {\n'
        '    "conjugate": [(lambda x: True, "Conjugate")],\n'
        '}\n'
    )
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": "test_file: ./sympy/printing/tests/test_mathematica.py\nsource_file: ./sympy/printing/mathematica.py\ndef mathematica_code(expr):\n    return expr.func.__name__\n",
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
    ]
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "edit_text"
    edit_requests = [request for request in fake_client.requests if request.get("contract") == "tool_input:edit_text"]
    assert len(edit_requests) == 2
    assert "Previous edit attempt was invalid or incomplete." in edit_requests[-1]["prompt"]


def test_runtime_replaces_directory_edit_path_with_recent_source_hint(make_config) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": ".",
                        "operation": "replace_pattern_once",
                        "pattern": "old",
                        "replacement": "new",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": "test_file: ./sympy/printing/tests/test_mathematica.py\nsource_file: ./sympy/printing/mathematica.py\n",
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
        Message(
            role="tool",
            name="read_text",
            content='read_text result: {"source_ref":"sympy/printing/mathematica.py","text":"def mathematica_code(expr):\\n    old\\n"}',
            created_at="t2",
            metadata={"output": {"source_ref": "sympy/printing/mathematica.py", "text": "def mathematica_code(expr):\n    old\n"}},
        ),
    ]
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "edit_text"
    assert decision.tool_input["path"] == "sympy/printing/mathematica.py"


def test_runtime_resolves_missing_workspace_edit_path_from_problem_symbol(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": str(tmp_path / "mathematica.py"),
                        "operation": "replace_pattern_once",
                        "pattern": "return 'Max[x, 2]'",
                        "replacement": "return 'Max[2, x]'",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    (tmp_path / "sympy" / "parsing").mkdir(parents=True)
    (tmp_path / "sympy" / "printing").mkdir(parents=True)
    (tmp_path / "sympy" / "parsing" / "mathematica.py").write_text("def parse_mathematica(x):\n    return x\n")
    (tmp_path / "sympy" / "printing" / "mathematica.py").write_text(
        "def mathematica_code(expr):\n    return 'Max[x, 2]'\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "edit_text"
    assert decision.tool_input["path"] == "sympy/printing/mathematica.py"


def test_runtime_synthesizes_targeted_run_tests_command_from_recent_hint(make_config) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:run_tests": [
                json.dumps({"command": ["python3", "-m", "pytest"], "background": False})
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        "Task contract:\n"
        "{\"task_kind\":\"local_repo_code_fix\",\"request_completeness\":\"complete\","
        "\"requires_code_changes\":true,\"requires_verification\":true,\"prefer_task_expansion\":false}\n"
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
        "Known failing tests:\n"
        "- test_Function\n"
        "Hints:\n"
        "Check mathematica.py.\n"
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "completed"
    plan.steps[2].status = "running"
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": "test_file: ./sympy/printing/tests/test_mathematica.py\nsource_file: ./sympy/printing/mathematica.py\n",
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
    ]
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "run_tests"
    assert decision.tool_input["command"] == [
        "python3",
        "-m",
        "pytest",
        "sympy/printing/tests/test_mathematica.py",
        "-k",
        "test_Function",
    ]


def test_runtime_shell_search_uses_prompt_file_hints_for_real_source_inspection(make_config, tmp_path: Path) -> None:
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
    package_dir = tmp_path / "pkg_492"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "stats.py").write_text(
        "def moving_total(values: list[int]) -> int:\n"
        "    total = 0\n"
        "    for value in values[:-1]:\n"
        "        total += value\n"
        "    return total\n",
        encoding="utf-8",
    )
    (tmp_path / "test_pkg_492.py").write_text(
        "from pkg_492.stats import moving_total\n",
        encoding="utf-8",
    )
    user_text = (
        f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
        "Inspect the module and test first, keep the public function name unchanged, and do not modify the test file."
    )
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.environment.shell.cwd = str(tmp_path)
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.active_plan = create_shell_recovery_plan("Fix the benchmark issue.")

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "shell_command"
    command = decision.tool_input["command"]
    assert "source_file: ./pkg_492/stats.py" in command
    assert "test_file: ./test_pkg_492.py" in command
    assert "sed -n '1,220p' ./pkg_492/stats.py" in command
    assert "sed -n '1,220p' ./test_pkg_492.py" in command


def test_runtime_shell_search_includes_all_named_prompt_file_hints_for_complex_coding_tasks(
    make_config,
    tmp_path: Path,
) -> None:
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
    package_dir = tmp_path / "pkg_545"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    for name in ("core.py", "calc.py", "report.py", "compat.py"):
        (package_dir / name).write_text(f"# {name}\n", encoding="utf-8")
    (tmp_path / "release_settings.json").write_text('{"label":"release-20"}\n', encoding="utf-8")
    for name in ("test_pkg_545_unit.py", "test_pkg_545_compat.py", "test_pkg_545_artifacts.py"):
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")
    user_text = (
        f"Repository root: {tmp_path}. Repair the `pkg_545` release flow. "
        "Inspect `pkg_545/core.py`, `pkg_545/calc.py`, `pkg_545/report.py`, `pkg_545/compat.py`, "
        "`release_settings.json`, and the three test files. "
        "Fix the implementation and the generated release artifact without editing the tests or the settings file. "
        "Run `python3 -m unittest -q test_pkg_545_unit.py test_pkg_545_compat.py test_pkg_545_artifacts.py` before answering."
    )
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.environment.shell.cwd = str(tmp_path)
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.active_plan = create_shell_recovery_plan("Fix the benchmark issue.")

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "shell_command"
    command = decision.tool_input["command"]
    assert "source_file: ./pkg_545/core.py" in command
    assert "source_file: ./pkg_545/calc.py" in command
    assert "source_file: ./pkg_545/report.py" in command
    assert "source_file: ./pkg_545/compat.py" in command
    assert "source_file: ./release_settings.json" in command
    assert "test_file: ./test_pkg_545_unit.py" in command
    assert "test_file: ./test_pkg_545_compat.py" in command
    assert "test_file: ./test_pkg_545_artifacts.py" in command


def test_runtime_deterministically_falls_back_to_shell_search_when_tool_input_model_is_unavailable(
    make_config,
    tmp_path: Path,
) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:shell_command": [
                requests.ConnectionError("llm down"),
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    runtime._max_model_unavailable_attempts = 0
    state = runtime.create_or_load_session()
    package_dir = tmp_path / "pkg_492"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "stats.py").write_text(
        "def moving_total(values: list[int]) -> int:\n"
        "    total = 0\n"
        "    for value in values[:-1]:\n"
        "        total += value\n"
        "    return total\n",
        encoding="utf-8",
    )
    (tmp_path / "test_pkg_492.py").write_text(
        "from pkg_492.stats import moving_total\n",
        encoding="utf-8",
    )
    user_text = (
        f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
        "Inspect the module and test first, keep the public function name unchanged, and do not modify the test file."
    )
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.environment.shell.cwd = str(tmp_path)
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.active_plan = create_shell_recovery_plan("Fix the benchmark issue.")

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "shell_command"
    command = decision.tool_input["command"]
    assert "source_file: ./pkg_492/stats.py" in command
    assert "test_file: ./test_pkg_492.py" in command


def test_runtime_synthesizes_unittest_command_from_prompt_file_hint(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:run_tests": [
                json.dumps({"command": ["python3", "-m", "pytest"], "background": False})
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
        "Run `python3 -m unittest -q test_pkg_492.py` after the edit."
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "completed"
    plan.steps[2].status = "running"
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "run_tests"
    assert decision.tool_input["command"] == ["python3", "-m", "unittest", "-q", "test_pkg_492.py"]


def test_runtime_normalizes_pytest_command_to_synthesized_unittest_for_named_test_hint(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:run_tests": [
                json.dumps({"command": ["pytest", "test_pkg_492.py"], "background": False})
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
        "Run the provided unit test after the edit."
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "completed"
    plan.steps[2].status = "running"
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "run_tests"
    assert decision.tool_input["command"] == ["python3", "-m", "unittest", "-q", "test_pkg_492.py"]


def test_runtime_deterministically_falls_back_to_run_tests_command_when_tool_input_model_is_unavailable(
    make_config,
    tmp_path: Path,
) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:run_tests": [
                requests.ConnectionError("llm down"),
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    runtime._max_model_unavailable_attempts = 0
    state = runtime.create_or_load_session()
    user_text = (
        f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
        "Run `python3 -m unittest -q test_pkg_492.py` after the edit."
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "completed"
    plan.steps[2].status = "running"
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "run_tests"
    assert decision.tool_input["command"] == ["python3", "-m", "unittest", "-q", "test_pkg_492.py"]


def test_runtime_falls_back_to_plain_text_edit_tool_input_when_structured_edit_call_is_unavailable(
    make_config,
    tmp_path: Path,
) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                requests.ConnectionError("llm down"),
            ],
            "tool_input:edit_text:plain_fallback": [
                json.dumps(
                    {
                        "path": "pkg_492/stats.py",
                        "operation": "replace_pattern_once",
                        "pattern": "for value in values[:-1]:",
                        "replacement": "for value in values:",
                    }
                )
            ],
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    runtime._max_model_unavailable_attempts = 0
    state = runtime.create_or_load_session()
    package_dir = tmp_path / "pkg_492"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "stats.py").write_text(
        "def moving_total(values: list[int]) -> int:\n"
        "    total = 0\n"
        "    for value in values[:-1]:\n"
        "        total += value\n"
        "    return total\n",
        encoding="utf-8",
    )
    (tmp_path / "test_pkg_492.py").write_text(
        "from pkg_492.stats import moving_total\n",
        encoding="utf-8",
    )
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.environment.shell.cwd = str(tmp_path)
    state.messages = [
        Message(
            role="user",
            content=(
                f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
                "Inspect the module and test first."
            ),
            created_at="t0",
        ),
        Message(
            role="tool",
            name="shell_command",
            content="inspection",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": (
                        "source_file: ./pkg_492/stats.py\n"
                        "def moving_total(values: list[int]) -> int:\n"
                        "    total = 0\n"
                        "    for value in values[:-1]:\n"
                        "        total += value\n"
                        "    return total\n"
                        "test_file: ./test_pkg_492.py\n"
                        "from pkg_492.stats import moving_total\n"
                    )
                }
            },
        ),
    ]
    plan = create_shell_recovery_plan("Fix the benchmark issue.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "edit_text"
    assert decision.tool_input["path"] == "pkg_492/stats.py"
    assert decision.tool_input["pattern"] == "for value in values[:-1]:"
    contracts = [request["contract"] for request in fake_client.requests]
    assert "tool_input:edit_text" in contracts
    assert "tool_input:edit_text:plain_fallback" in contracts


def test_runtime_caps_subagent_selection_request_max_tokens(make_config) -> None:
    fake_client = FakeModelClient(
        contract_responses={
            "subagent_selection": [
                json.dumps(
                    {
                        "spawn": False,
                        "subagent_type": "none",
                        "reason": "direct response is sufficient",
                        "focus": "",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(make_config(), model_client=fake_client)
    state = runtime.create_or_load_session()

    selection = runtime._select_subagent_frontend(
        state,
        goal="Reply with OK only.",
        purpose="context_retrieval_focus",
        candidate_types=["retriever"],
        detail_lines=["call_kind=subagent_selection"],
    )

    requests = [request for request in fake_client.requests if request.get("contract") == "subagent_selection"]
    assert selection.spawn is False
    assert requests
    assert 0 < requests[-1]["n_predict"] <= 64


def test_runtime_caps_plan_request_max_tokens(make_config) -> None:
    fake_client = FakeModelClient(
        contract_responses={
            "task_plan": [
                plan_response(
                    goal="Answer from the provided evidence.",
                    steps=[
                        plan_step(
                            "step_answer",
                            "Answer",
                            "respond",
                            expected_output="concise answer",
                            success_criteria="answers from evidence",
                        )
                    ],
                )
            ]
        }
    )
    runtime = AgentRuntime(make_config(), model_client=fake_client)
    state = runtime.create_or_load_session()

    runtime._generate_plan(state, "Answer from the provided evidence.", update_existing=False, replan_reason="")

    requests = [request for request in fake_client.requests if request.get("contract") == "task_plan"]
    assert requests
    assert 0 < requests[-1]["n_predict"] <= 512


def test_runtime_caps_tool_input_request_max_tokens_for_edit_steps(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": "pkg_492/stats.py",
                        "operation": "replace_pattern_once",
                        "pattern": "for value in values[:-1]:",
                        "replacement": "for value in values:",
                    }
                )
            ],
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    package_dir = tmp_path / "pkg_492"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "stats.py").write_text(
        "def moving_total(values: list[int]) -> int:\n"
        "    total = 0\n"
        "    for value in values[:-1]:\n"
        "        total += value\n"
        "    return total\n",
        encoding="utf-8",
    )
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.environment.shell.cwd = str(tmp_path)
    state.messages = [
        Message(
            role="user",
            content=(
                f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
                "Inspect the module and test first."
            ),
            created_at="t0",
        ),
        Message(
            role="tool",
            name="shell_command",
            content="inspection",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": (
                        "source_file: ./pkg_492/stats.py\n"
                        "def moving_total(values: list[int]) -> int:\n"
                        "    total = 0\n"
                        "    for value in values[:-1]:\n"
                        "        total += value\n"
                        "    return total\n"
                    )
                }
            },
        ),
    ]
    plan = create_shell_recovery_plan("Fix the benchmark issue.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "edit_text"
    edit_requests = [request for request in fake_client.requests if request.get("contract") == "tool_input:edit_text"]
    assert edit_requests
    assert 0 < edit_requests[-1]["n_predict"] <= 256


def test_runtime_prefers_deterministic_edit_payload_for_benchmark_locked_frontend(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient()
    fake_client.is_deterministic_test_client = False
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    package_dir = tmp_path / "pkg_492"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "stats.py").write_text(
        "def moving_total(values: list[int]) -> int:\n"
        "    total = 0\n"
        "    for value in values[:-1]:\n"
        "        total += value\n"
        "    return total\n",
        encoding="utf-8",
    )
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.environment.shell.cwd = str(tmp_path)
    state.active_strategy = StrategySelection(
        strategy_name="exploratory",
        explore_before_commit=True,
        validate_assumptions=True,
        simplify_if_stuck=True,
        switch_on_failure=True,
        reason="deterministic_code_fix_fallback:benchmark_code_fix_locked_frontend",
        task_profile="coding",
        required_step_kinds=["read", "write", "respond"],
        expected_flow=["read", "write", "respond"],
    )
    state.messages = [
        Message(
            role="user",
            content=(
                f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
                "Inspect the module and test first."
            ),
            created_at="t0",
        ),
        Message(
            role="tool",
            name="shell_command",
            content="inspection",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": (
                        "source_file: ./pkg_492/stats.py\n"
                        "def moving_total(values: list[int]) -> int:\n"
                        "    total = 0\n"
                        "    for value in values[:-1]:\n"
                        "        total += value\n"
                        "    return total\n"
                        "test_file: ./test_pkg_492.py\n"
                        "import unittest\n\n"
                        "from pkg_492.stats import moving_total\n\n"
                        "class StatsTests(unittest.TestCase):\n"
                        "    def test_moving_total(self) -> None:\n"
                        "        self.assertEqual(moving_total([7, 7, 15]), 29)\n"
                    )
                }
            },
        ),
    ]
    plan = create_shell_recovery_plan("Fix the benchmark issue.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "edit_text"
    assert decision.tool_input["path"] == "pkg_492/stats.py"
    assert decision.tool_input["pattern"] == "values[:-1]"
    assert decision.tool_input["replacement"] == "values"
    assert fake_client.requests == []


def test_runtime_normalizes_run_tests_tool_self_reference(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:run_tests": [
                json.dumps({"command": ["run_tests", "test_pkg_492.py"], "background": False})
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    user_text = (
        f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
        "Run `python3 -m unittest -q test_pkg_492.py` after the edit."
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "completed"
    plan.steps[2].status = "running"
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.environment.shell.cwd = str(tmp_path)
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "run_tests"
    assert decision.tool_input["command"] == ["python3", "-m", "unittest", "-q", "test_pkg_492.py"]


def test_runtime_normalizes_replace_pattern_alias_when_source_is_available(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": str(tmp_path / "stats.py"),
                        "operation": "replace_pattern",
                        "pattern": "for value in values[:-1]:",
                        "replacement": "for value in values:",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    (tmp_path / "pkg_492").mkdir(parents=True)
    (tmp_path / "pkg_492" / "stats.py").write_text(
        "def moving_total(values: list[int]) -> int:\n"
        "    total = 0\n"
        "    for value in values[:-1]:\n"
        "        total += value\n"
        "    return total\n",
        encoding="utf-8",
    )
    user_text = (
        f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
        "Inspect the module and test first, keep the public function name unchanged, and do not modify the test file."
    )
    plan = create_shell_recovery_plan("Fix the benchmark issue.")
    plan.steps[0].status = "completed"
    plan.steps[1].status = "running"
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.environment.shell.cwd = str(tmp_path)
    state.messages = [
        Message(role="user", content=user_text, created_at="t0"),
        Message(
            role="tool",
            name="shell_command",
            content="shell result",
            created_at="t1",
            metadata={
                "output": {
                    "stdout": "source_file: ./pkg_492/stats.py\n"
                    "def moving_total(values: list[int]) -> int:\n"
                    "    total = 0\n"
                    "    for value in values[:-1]:\n"
                    "        total += value\n"
                    "    return total\n",
                    "stderr": "",
                    "exit_code": 0,
                    "modified_files": [],
                    "created_files": [],
                    "deleted_files": [],
                }
            },
        ),
    ]
    state.active_plan = plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "edit_text"
    assert decision.tool_input["path"] == "pkg_492/stats.py"
    assert decision.tool_input["operation"] == "replace_pattern_once"


def test_runtime_normalizes_read_text_payload_that_echoes_output_fields(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    runtime = AgentRuntime(config, model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    (tmp_path / "pkg_492").mkdir(parents=True)
    (tmp_path / "pkg_492" / "stats.py").write_text("def moving_total(values):\n    return 0\n", encoding="utf-8")
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    step_plan = plan_from_payload(
        {
            "goal": "Fix the benchmark issue.",
            "success_criteria": "Resolve the bug safely.",
            "fallback_strategy": "Replan from the latest valid state.",
            "steps": [
                plan_step(
                    "step_read",
                    "Identify the bug",
                    "read",
                    expected_tool="read_text",
                    input_text="pkg_492/stats.py",
                    expected_output="Source text",
                    success_criteria="The source file is read.",
                ),
                plan_step(
                    "step_answer",
                    "Answer the user",
                    "respond",
                    expected_output="Final assistant response",
                    success_criteria="The user gets a final answer.",
                    depends_on=["step_read"],
                ),
            ],
        },
        available_tools=runtime.tools.tool_names(config),
    )
    step = step_plan.steps[0]
    normalized = runtime._normalize_expected_tool_input(
        state,
        step,
        {
            "path": str(tmp_path / "stats.py"),
            "source_ref": str(tmp_path / "pkg_492" / "stats.py"),
            "start_offset": 0,
            "end_offset": 128,
            "next_offset": 128,
            "finished": True,
            "text": "def moving_total(values):\\n    return 0\\n",
        },
    )

    assert normalized == {"path": "pkg_492/stats.py"}


def test_runtime_normalizes_read_file_payload_that_echoes_output_fields(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    runtime = AgentRuntime(config, model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    (tmp_path / "pkg_492").mkdir(parents=True)
    (tmp_path / "pkg_492" / "stats.py").write_text("def moving_total(values):\n    return 0\n", encoding="utf-8")
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    step_plan = plan_from_payload(
        {
            "goal": "Fix the benchmark issue.",
            "success_criteria": "Resolve the bug safely.",
            "fallback_strategy": "Replan from the latest valid state.",
            "steps": [
                plan_step(
                    "step_read",
                    "Identify the bug",
                    "read",
                    expected_tool="read_file",
                    input_text="pkg_492/stats.py",
                    expected_output="Source text",
                    success_criteria="The source file is read.",
                ),
                plan_step(
                    "step_answer",
                    "Answer the user",
                    "respond",
                    expected_output="Final assistant response",
                    success_criteria="The user gets a final answer.",
                    depends_on=["step_read"],
                ),
            ],
        },
        available_tools=runtime.tools.tool_names(config),
    )
    step = step_plan.steps[0]
    normalized = runtime._normalize_expected_tool_input(
        state,
        step,
        {
            "path": str(tmp_path / "stats.py"),
            "source_ref": str(tmp_path / "pkg_492" / "stats.py"),
            "text": "def moving_total(values):\\n    return 0\\n",
            "size_chars": 37,
        },
    )

    assert normalized == {"path": "pkg_492/stats.py"}


def test_runtime_uses_expected_tool_input_contract_for_read_text_step(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:read_text": [
                json.dumps(
                    {
                        "path": str(tmp_path / "stats.py"),
                        "start_offset": 0,
                        "end_offset": 128,
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    (tmp_path / "pkg_492").mkdir(parents=True)
    (tmp_path / "pkg_492" / "stats.py").write_text("def moving_total(values):\n    return 0\n", encoding="utf-8")
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.environment.shell.cwd = str(tmp_path)
    state.messages = [
        Message(
            role="user",
            content=(
                f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
                "Inspect the module and test first."
            ),
            created_at="t0",
        )
    ]
    step_plan = plan_from_payload(
        {
            "goal": "Fix the benchmark issue.",
            "success_criteria": "Resolve the bug safely.",
            "fallback_strategy": "Replan from the latest valid state.",
            "steps": [
                plan_step(
                    "step_read",
                    "Identify the bug",
                    "read",
                    expected_tool="read_text",
                    input_text="pkg_492/stats.py",
                    expected_output="Source text",
                    success_criteria="The source file is read.",
                ),
                plan_step(
                    "step_answer",
                    "Answer the user",
                    "respond",
                    expected_output="Final assistant response",
                    success_criteria="The user gets a final answer.",
                    depends_on=["step_read"],
                ),
            ],
        },
        available_tools=runtime.tools.tool_names(config),
    )
    step_plan.steps[0].status = "running"
    step_plan.current_step_id = step_plan.steps[0].step_id
    state.active_plan = step_plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "read_text"
    assert decision.tool_input["path"] == "pkg_492/stats.py"
    assert "start_offset" not in decision.tool_input
    assert "end_offset" not in decision.tool_input
    request_contracts = [request.get("contract") for request in fake_client.requests]
    assert "tool_input:read_text" in request_contracts


def test_runtime_uses_expected_tool_input_contract_for_read_file_step(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    fake_client = FakeModelClient(
        contract_responses={
            "tool_input:read_file": [
                json.dumps(
                    {
                        "source_ref": str(tmp_path / "pkg_492" / "stats.py"),
                        "text": "def moving_total(values):\\n    return 0\\n",
                    }
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    (tmp_path / "pkg_492").mkdir(parents=True)
    (tmp_path / "pkg_492" / "stats.py").write_text("def moving_total(values):\n    return 0\n", encoding="utf-8")
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    state.environment.shell.cwd = str(tmp_path)
    state.messages = [
        Message(
            role="user",
            content=(
                f"Repository root: {tmp_path}. Fix `pkg_492/stats.py` so `test_pkg_492.py` passes. "
                "Inspect the module and test first."
            ),
            created_at="t0",
        )
    ]
    step_plan = plan_from_payload(
        {
            "goal": "Fix the benchmark issue.",
            "success_criteria": "Resolve the bug safely.",
            "fallback_strategy": "Replan from the latest valid state.",
            "steps": [
                plan_step(
                    "step_read",
                    "Identify the bug",
                    "read",
                    expected_tool="read_file",
                    input_text="pkg_492/stats.py",
                    expected_output="Source text",
                    success_criteria="The source file is read.",
                ),
                plan_step(
                    "step_answer",
                    "Answer the user",
                    "respond",
                    expected_output="Final assistant response",
                    success_criteria="The user gets a final answer.",
                    depends_on=["step_read"],
                ),
            ],
        },
        available_tools=runtime.tools.tool_names(config),
    )
    step_plan.steps[0].status = "running"
    step_plan.current_step_id = step_plan.steps[0].step_id
    state.active_plan = step_plan

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "read_file"
    assert decision.tool_input == {"path": "pkg_492/stats.py"}
    request_contracts = [request.get("contract") for request in fake_client.requests]
    assert "tool_input:read_file" in request_contracts


def test_runtime_read_text_result_satisfies_read_file_verification_and_done_condition(make_config, tmp_path: Path) -> None:
    config = make_config(tools__allow_side_effect_tools=True)
    runtime = AgentRuntime(config, model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    state.environment.workspace.root = str(tmp_path)
    state.environment.workspace.cwd = str(tmp_path)
    step_plan = plan_from_payload(
        {
            "goal": "Inspect the file safely.",
            "success_criteria": "The source is available.",
            "fallback_strategy": "Replan from the latest valid state.",
            "steps": [
                plan_step(
                    "step_read",
                    "Read the source",
                    "read",
                    expected_tool="read_file",
                    input_text="pkg_492/stats.py",
                    expected_output="Source text",
                    success_criteria="The source file is read.",
                ),
                plan_step(
                    "step_answer",
                    "Answer the user",
                    "respond",
                    expected_output="Final assistant response",
                    success_criteria="The user gets a final answer.",
                    depends_on=["step_read"],
                ),
            ],
        },
        available_tools=runtime.tools.tool_names(config),
    )
    step = step_plan.steps[0]
    tool_result = ToolExecutionResult(
        tool_name="read_text",
        output={"source_ref": "pkg_492/stats.py", "text": "def moving_total(values):\n    return 0\n", "finished": True},
        display_text="read_text result",
    )
    verification = runtime._verification.verify_step(
        runtime=runtime,
        state=state,
        plan=step_plan,
        step=step,
        artifacts=VerificationArtifacts(tool_results=[tool_result]),
    )
    evaluation = evaluate_step(step, tool_result=tool_result)

    assert verification.passed is True
    assert evaluation.passed is True


def test_runtime_shell_search_falls_back_to_symbols_without_failing_tests(make_config) -> None:
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
        "Problem statement:\n"
        "mathematica_code gives wrong output with Max\n"
    )
    state.messages = [Message(role="user", content=user_text, created_at="t0")]
    state.active_plan = create_shell_recovery_plan("Fix the benchmark issue. mathematica_code gives wrong output with Max.")

    decision, _ = runtime._decide(state)

    assert decision.tool_name == "shell_command"
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


def test_runtime_failure_classifier_uses_emergency_replan_fallback_when_model_unavailable(make_config) -> None:
    class UnavailableFailureClient(FakeModelClient):
        def send_completion(self, request, *, timeout_seconds=None, stream_callback=None):  # type: ignore[override]
            raise ModelClientError("semantic_engine_unavailable")

    config = make_config()
    runtime = AgentRuntime(config, model_client=UnavailableFailureClient(responses=[]))
    state = runtime.create_or_load_session()
    step = PlanStep(
        step_id="step_tests",
        title="Run tests",
        goal="Run tests",
        kind="tool",
        expected_tool="run_tests",
        input_text="Run tests",
        expected_output="Tests pass",
        done_condition="tool_result:run_tests",
        success_criteria="Tests pass",
    )

    classification = runtime._classify_failure_frontend(state, step=step, reason="verification:command_exit_zero")
    events = runtime.history.read_history(state.session_id)

    assert classification.requires_replan is True
    assert classification.retryable is False
    assert classification.source == "deterministic_fallback_emergency_only"
    assert any(event.event_type == "failure_classification_fallback" for event in events)
