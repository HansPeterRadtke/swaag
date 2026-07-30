from __future__ import annotations

import json
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest
import requests

import swaag.runtime as runtime_module
from swaag.model import ModelClientError
from swaag.planner import PlanValidationError, plan_from_payload
from swaag.retrieval.embeddings import SemanticBackendProtocolError
from swaag.runtime import AgentRuntime, BudgetExceededError, FatalSemanticEngineError
from swaag.tools.base import ToolValidationError
from swaag.types import CompletionResult, DecisionOutcome, ExpandedTask, Message, PlanStep, PromptAnalysis
from swaag.utils import stable_json_dumps

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

    def send_completion(self, payload: dict[str, Any], *, timeout_seconds: int | None = None, progress_callback=None) -> CompletionResult:
        del payload, timeout_seconds, progress_callback
        raise requests.ReadTimeout("No streamed model token/event for 1.0 seconds")


def _normalize_volatile_request_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_volatile_request_fields(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_volatile_request_fields(item) for item in value]
    if isinstance(value, str):
        normalized = re.sub(r"\b[a-z]+_[0-9a-f]{12}\b", "<generated-id>", value)
        normalized = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z)", "<timestamp>", normalized)
        normalized = re.sub(r"elapsed=\d+(?:\.\d+)?s", "elapsed=<duration>", normalized)
        normalized = re.sub(r"avg_tps=(?:None|\d+(?:\.\d+)?)", "avg_tps=<rate>", normalized)
        normalized = re.sub(r"Workspace: [^\n]+", "Workspace: <workspace>", normalized)
        return normalized
    return value


def test_final_objective_accepts_requested_evidence_grounded_clarification(make_config) -> None:
    goal = "Read request.txt and context.txt, then ask the single most useful clarifying question before acting."
    answer = "Which service and rollout risk should the safety plan address, and what criterion defines success?"
    prompts: list[str] = []

    def verify(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        prompts.append(prompt)
        assert "terminal outcome explicitly requested by the original user request" in prompt
        assert "a single evidence-grounded clarification is the completed terminal outcome" in prompt
        assert '"prompt_analysis"' in prompt
        assert '"latest_task_decision"' in prompt
        assert '"evidence_required_before_response":true' in prompt
        assert answer in prompt
        return json.dumps(
            {
                "criteria": [
                    {
                        "name": "final_objective_satisfied",
                        "passed": True,
                        "evidence": "the requested endpoint was an evidence-grounded clarification and it was supplied",
                        "candidate_excerpts": [answer],
                    }
                ]
            }
        )

    runtime = AgentRuntime(
        make_config(model__context_limit=32_000),
        model_client=FakeModelClient(contract_responses={"verification": [verify]}),
    )
    state = runtime.create_or_load_session()
    runtime._record_message(state, Message(role="user", content=goal, created_at="t0"))
    state.prompt_analysis = PromptAnalysis(
        task_type="vague",
        completeness="partial",
        requires_expansion=True,
        requires_decomposition=False,
        confidence=0.9,
        missing_required_information=False,
        detected_entities=["request.txt", "context.txt"],
        detected_goals=["read evidence", "ask clarification"],
    )
    state.latest_decision = DecisionOutcome(
        split_task=False,
        expand_task=False,
        ask_user=True,
        assume_missing=False,
        generate_ideas=False,
        confidence=0.9,
        reason="Read two evidence sources before asking.",
        direct_response=False,
        execution_mode="full_plan",
        preferred_tool_name="",
        evidence_required_before_response=True,
        evidence_call_count=2,
    )
    state.active_plan = plan_from_payload(
        json.loads(
            plan_response(
                goal=goal,
                steps=[
                    plan_step(
                        "read_request",
                        "Read request",
                        "read",
                        expected_tool="read_file",
                        expected_output="request contents",
                        success_criteria="request.txt is read",
                    ),
                    plan_step(
                        "read_context",
                        "Read context",
                        "read",
                        expected_tool="read_file",
                        expected_output="context contents",
                        success_criteria="context.txt is read",
                        depends_on=["read_request"],
                    ),
                    plan_step(
                        "clarify",
                        "Ask clarification",
                        "respond",
                        expected_output="one grounded question",
                        success_criteria="Ask the single most useful grounded clarification question.",
                        depends_on=["read_request", "read_context"],
                    ),
                ],
            )
        ),
        available_tools=["read_file"],
    )
    for item in state.active_plan.steps:
        item.status = "completed"
    state.active_plan.status = "completed"

    outcome = runtime._verify_final_objective(state, state.active_plan.steps[-1], answer)

    assert outcome.verification_passed is True
    assert outcome.conditions_met == ["final_objective_satisfied"]
    assert prompts


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
            ],
            "task_plan": [
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step(
                            "calculate",
                            "Calculate",
                            "tool",
                            expected_tool="calculator",
                            expected_output="4",
                            success_criteria="The calculator returns 4.",
                        ),
                        plan_step(
                            "answer",
                            "Answer",
                            "respond",
                            expected_output="4",
                            success_criteria="Return the calculator result.",
                            depends_on=["calculate"],
                            verification_type="composite",
                            verification_checks=[
                                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                {"name": "answer_exact", "check_type": "exact_match", "actual_source": "assistant_text", "expected": "4"},
                            ],
                            required_conditions=["dependencies_completed", "answer_exact"],
                            optional_conditions=[],
                        ),
                    ],
                )
            ]
        },
        responses=[
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "4",
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


def test_task_decision_correction_accumulates_semantic_and_structural_feedback(make_config) -> None:
    semantic_bad = {
        "split_task": False, "expand_task": False, "ask_user": True,
        "assume_missing": False, "generate_ideas": False, "direct_response": False,
        "execution_mode": "clarification", "preferred_tool_name": "",
        "evidence_required_before_response": False, "evidence_call_count": 0,
        "confidence": 0.8, "reason": "Read request.txt and context.txt before asking.",
    }
    structural_bad = {
        **semantic_bad,
        "evidence_required_before_response": True,
        "evidence_call_count": 2,
    }
    semantic_regression = dict(semantic_bad)
    corrected = {
        **semantic_bad,
        "execution_mode": "full_plan",
        "evidence_required_before_response": True,
        "evidence_call_count": 2,
        "reason": "Read request.txt and context.txt with two calls, then ask one grounded question.",
    }
    failed_review = json.dumps({
        "decision_matches_request": True,
        "decision_is_internally_consistent": True,
        "required_evidence_sources": ["request.txt", "context.txt"],
        "minimum_evidence_call_count": 2,
        "selected_mode_and_tool_can_cover_declared_count": True,
        "feedback": "Both named files require two evidence calls before clarification.",
    })
    passed_review = json.dumps({
        "decision_matches_request": True,
        "decision_is_internally_consistent": True,
        "required_evidence_sources": ["request.txt", "context.txt"],
        "minimum_evidence_call_count": 2,
        "selected_mode_and_tool_can_cover_declared_count": True,
        "feedback": "The full plan covers both sources.",
    })
    client = FakeModelClient(contract_responses={
        "task_decision": [
            json.dumps(semantic_bad),
            json.dumps(structural_bad),
            json.dumps(semantic_regression),
            json.dumps(corrected),
        ],
        "task_decision_semantic_review": [
            failed_review,
            failed_review,
            passed_review,
        ],
    })
    runtime = AgentRuntime(make_config(), model_client=client)
    state = runtime.create_or_load_session()
    analysis = PromptAnalysis(
        task_type="unstructured", completeness="partial",
        requires_expansion=True, requires_decomposition=False,
        confidence=0.8, missing_required_information=False,
        detected_entities=["request.txt", "context.txt"],
        detected_goals=["read both files and clarify"],
    )

    decision = runtime._decide_prompt_frontend(
        state,
        "Read request.txt and context.txt, then ask one clarification.",
        analysis,
    )

    assert decision.execution_mode == "full_plan"
    assert decision.evidence_call_count == 2
    requests = [item for item in client.requests if item["contract"] == "task_decision"]
    assert len(requests) == 4
    final_prompt = requests[-1]["prompt"]
    assert "Attempt 1 semantic review failed" in final_prompt
    assert "Both named files require two evidence calls" in final_prompt
    assert "Attempt 2 structural validation failed" in final_prompt
    assert "requires execution_mode='full_plan' or 'single_tool'" in final_prompt
    assert "Attempt 3 semantic review failed" in final_prompt
    assert final_prompt.rfind("Task-decision correction requirements from all previous attempts:") > final_prompt.rfind("evidence_call_count is the number of tool calls required before the response")
    correction_end = final_prompt.rfind("Keep already-valid fields, but change every field named by the accumulated feedback.")
    assistant_marker = final_prompt.rfind("<|start_header_id|>assistant<|end_header_id|>")
    assert correction_end > 0
    assert assistant_marker > correction_end
    assert "Current user request:" not in final_prompt[correction_end:assistant_marker]
    events = runtime.history.read_history(state.session_id)
    assert sum(
        1 for event in events
        if event.event_type == "model_retry_scheduled" and event.payload.get("kind") == "task_decision"
    ) == 3
    assert not any(event.event_type == "fatal_system_error" for event in events)


def test_task_decision_structural_validation_retries_without_fatal_error(make_config) -> None:
    invalid = {
        "split_task": False, "expand_task": False, "ask_user": True,
        "assume_missing": False, "generate_ideas": False, "direct_response": False,
        "execution_mode": "clarification", "preferred_tool_name": "",
        "evidence_required_before_response": True, "evidence_call_count": 2,
        "confidence": 0.9, "reason": "Read two files before asking.",
    }
    corrected = {
        **invalid,
        "execution_mode": "full_plan",
        "reason": "Read both files under a full plan before asking.",
    }
    passed_review = json.dumps({
        "decision_matches_request": True,
        "decision_is_internally_consistent": True,
        "required_evidence_sources": ["request.txt", "context.txt"],
        "minimum_evidence_call_count": 2,
        "selected_mode_and_tool_can_cover_declared_count": True,
        "feedback": "two file reads are covered by the full plan",
    })
    client = FakeModelClient(contract_responses={
        "task_decision": [json.dumps(invalid), json.dumps(corrected)],
        "task_decision_semantic_review": [passed_review],
    })
    runtime = AgentRuntime(make_config(), model_client=client)
    state = runtime.create_or_load_session()
    analysis = PromptAnalysis(
        task_type="unstructured", completeness="partial",
        requires_expansion=True, requires_decomposition=False,
        confidence=0.8, missing_required_information=False,
        detected_entities=["request.txt", "context.txt"],
        detected_goals=["read files and clarify"],
    )

    decision = runtime._decide_prompt_frontend(
        state,
        "Read request.txt and context.txt, then ask a clarification.",
        analysis,
    )

    assert decision.execution_mode == "full_plan"
    assert decision.evidence_call_count == 2
    assert [item["contract"] for item in client.requests] == [
        "task_decision",
        "task_decision",
        "task_decision_semantic_review",
    ]
    decision_requests = [item for item in client.requests if item["contract"] == "task_decision"]
    assert "Previous rejected task decision JSON:" in decision_requests[1]["prompt"]
    assert "Attempt 1 structural validation failed" in decision_requests[1]["prompt"]
    assert "requires execution_mode='full_plan' or 'single_tool'" in decision_requests[1]["prompt"]
    events = runtime.history.read_history(state.session_id)
    assert any(
        event.event_type == "error"
        and event.payload.get("operation") == "task_decision_validation"
        for event in events
    )
    assert not any(event.event_type == "fatal_system_error" for event in events)


def test_task_decision_semantic_review_retries_contradictory_candidate_with_full_registry(make_config) -> None:
    invalid = {
        "split_task": False, "expand_task": False, "ask_user": True,
        "assume_missing": False, "generate_ideas": False, "direct_response": False,
        "execution_mode": "clarification", "preferred_tool_name": "",
        "evidence_required_before_response": False, "evidence_call_count": 0,
        "confidence": 0.8, "reason": "Need to read request.txt and context.txt before asking.",
    }
    corrected = {
        **invalid,
        "execution_mode": "full_plan",
        "evidence_required_before_response": True,
        "evidence_call_count": 2,
        "reason": "Read both files before asking a grounded clarification.",
    }
    failed_review = json.dumps({
        "decision_matches_request": True,
        "decision_is_internally_consistent": False,
        "required_evidence_sources": ["request.txt", "context.txt"],
        "minimum_evidence_call_count": 2,
        "selected_mode_and_tool_can_cover_declared_count": False,
        "feedback": "read_file accepts one scalar path, so two named files need two calls and full_plan",
    })
    passed_review = json.dumps({
        "decision_matches_request": True,
        "decision_is_internally_consistent": True,
        "required_evidence_sources": ["request.txt", "context.txt"],
        "minimum_evidence_call_count": 2,
        "selected_mode_and_tool_can_cover_declared_count": True,
        "feedback": "two file reads are covered by the full plan",
    })
    client = FakeModelClient(contract_responses={
        "task_decision": [json.dumps(invalid), json.dumps(corrected)],
        "task_decision_semantic_review": [failed_review, passed_review],
    })
    runtime = AgentRuntime(make_config(), model_client=client)
    state = runtime.create_or_load_session()
    analysis = PromptAnalysis(
        task_type="unstructured", completeness="partial",
        requires_expansion=True, requires_decomposition=False,
        confidence=0.8, missing_required_information=False,
        detected_entities=["request.txt", "context.txt"],
        detected_goals=["read files and clarify"],
    )

    decision = runtime._decide_prompt_frontend(
        state,
        "Read request.txt and context.txt, then ask the single most useful clarifying question before acting.",
        analysis,
    )

    assert decision.execution_mode == "full_plan"
    assert decision.evidence_required_before_response is True
    assert decision.evidence_call_count == 2
    decision_requests = [item for item in client.requests if item["contract"] == "task_decision"]
    review_requests = [item for item in client.requests if item["contract"] == "task_decision_semantic_review"]
    assert len(decision_requests) == 2
    assert len(review_requests) == 2
    assert [item["contract"] for item in client.requests] == [
        "task_decision",
        "task_decision_semantic_review",
        "task_decision",
        "task_decision_semantic_review",
    ]
    assert "Previous rejected task decision JSON:" in decision_requests[1]["prompt"]
    assert "read_file accepts one scalar path, so two named files need two calls and full_plan" in decision_requests[1]["prompt"]
    assert "Required evidence sources: request.txt, context.txt" in decision_requests[1]["prompt"]
    assert "Minimum evidence call count: 2" in decision_requests[1]["prompt"]
    review_prompt = review_requests[0]["prompt"]
    for tool in runtime.tools.enabled_tools(runtime.config):
        assert tool.name in review_prompt
        assert tool.description in review_prompt
        assert stable_json_dumps(tool.input_schema) in review_prompt
        assert tool.usage_guidance in review_prompt
    events = runtime.history.read_history(state.session_id)
    assert not any(event.event_type == "subagent_selected" for event in events)
    assert not any(event.event_type == "prompt_analyzed" for event in events)
    assert any(
        event.event_type == "review_completed"
        and event.payload.get("review_kind") == "task_decision_semantic"
        and event.payload.get("passed") is False
        for event in events
    )
    assert any(
        event.event_type == "review_completed"
        and event.payload.get("review_kind") == "task_decision_semantic"
        and event.payload.get("passed") is True
        for event in events
    )


def test_runtime_immediate_clarification_mode_does_not_plan_or_call_tools(make_config) -> None:
    question = "Which service should this rollout concern?"
    client = FakeModelClient(contract_responses={
        "prompt_analysis": [json.dumps({
            "task_type": "incomplete", "completeness": "incomplete",
            "requires_expansion": False, "requires_decomposition": False,
            "missing_required_information": True,
            "confidence": 1.0, "detected_entities": [], "detected_goals": ["clarify service"],
        })],
        "task_decision": [json.dumps({
            "split_task": False, "expand_task": False, "ask_user": True,
            "assume_missing": False, "generate_ideas": False, "direct_response": False,
            "execution_mode": "clarification", "preferred_tool_name": "",
            "evidence_required_before_response": False,
            "evidence_call_count": 0,
            "confidence": 1.0, "reason": "service is missing",
        })],
        "clarification_response": [json.dumps({"text": question})],
    })
    runtime = AgentRuntime(make_config(), model_client=client)

    result = runtime.run_turn("Make the rollout safer.")
    contracts = [request["contract"] for request in client.requests]
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == question
    assert "clarification_response" in contracts
    assert "task_plan" not in contracts
    assert not result.tool_results
    assert any(
        event.event_type == "reasoning_completed"
        and event.payload.get("status") == "clarification_requested"
        for event in events
    )


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


def test_runtime_allows_repeated_identical_tool_requests_across_distinct_steps(make_config) -> None:
    config = make_config(runtime__max_repeated_action_occurrences=1, planner__max_replans=0)
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        contract_responses={
            "answer_response": ["4"],
        },
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
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert result.assistant_text == "4"
    assert len(result.tool_results) == 2
    assert not any(event.event_type == "duplicate_action_detected" for event in events)
    assert any(request["contract"] == "answer_response" for request in fake_client.requests)


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


def test_runtime_caps_structured_answer_generation_tokens(make_config) -> None:
    config = make_config(model__context_limit=32_000)
    fake_client = FakeModelClient(responses=["short answer"])
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()
    runtime._record_message(state, Message(role="user", content="Reply briefly.", created_at="t0"))

    answer, _report = runtime._generate_direct_response_once(state)
    events = runtime.history.read_history(state.session_id)

    assert answer == "short answer"
    answer_request = next(request for request in fake_client.requests if request["contract"] == "answer_response")
    assert answer_request["n_predict"] == 512
    assert any(
        event.event_type == "budget_repaired"
        and event.payload.get("kind") == "answer"
        and event.payload.get("reason") == "cap_answer_generation_tokens"
        and event.payload.get("capped_response_tokens") == 512
        for event in events
    )


def test_runtime_caps_structured_plan_generation_tokens(make_config) -> None:
    config = make_config(model__context_limit=32_000, planner__max_plan_steps=4)
    goal = "Answer ok."
    fake_client = FakeModelClient(
        contract_responses={
            "task_plan": [
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step(
                            "answer",
                            "Answer",
                            "respond",
                            expected_output="ok",
                            success_criteria="Return ok.",
                        )
                    ],
                )
            ]
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    state = runtime.create_or_load_session()

    runtime._generate_plan(state, goal, update_existing=False, replan_reason="")
    events = runtime.history.read_history(state.session_id)

    plan_request = next(request for request in fake_client.requests if request["contract"] == "task_plan")
    assert plan_request["n_predict"] == 3072
    assert any(
        event.event_type == "budget_repaired"
        and event.payload.get("kind") == "plan"
        and event.payload.get("reason") == "cap_plan_generation_tokens"
        and event.payload.get("capped_response_tokens") == 3072
        for event in events
    )


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
    assert _normalize_volatile_request_fields(client_one.requests) == _normalize_volatile_request_fields(client_two.requests)


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


def test_verification_protocol_allows_optional_grounding_for_absence_judgment(make_config) -> None:
    candidate = '{"steps":[{"step_id":"read_request","expected_tool":"read_file"}]}'
    response = json.dumps(
        {
            "criteria": [
                {
                    "name": "plan_uses_current_evidence",
                    "passed": True,
                    "evidence": "No recent failure or snapshot evidence conflicts with the candidate plan.",
                    "candidate_excerpts": [],
                }
            ]
        }
    )
    client = FakeModelClient(contract_responses={"verification": [response]})
    runtime = AgentRuntime(make_config(), model_client=client)
    state = runtime.create_or_load_session()
    step = PlanStep(
        step_id="review",
        title="Plan semantic adequacy review",
        goal="Review plan",
        kind="reasoning",
        expected_tool=None,
        input_text="review",
        expected_output="adequate",
        done_condition="reasoning_result_nonempty",
        success_criteria="The plan uses current evidence.",
        expected_outputs=["adequate"],
        verification_type="composite",
        verification_checks=[],
        required_conditions=[],
        optional_conditions=[],
    )

    payload = runtime._run_llm_verification(
        state,
        step=step,
        criteria=[
            {
                "name": "plan_uses_current_evidence",
                "criterion": "The plan does not depend on stale evidence.",
                "candidate_grounding": "optional",
            }
        ],
        assistant_text=candidate,
        evidence={"recent_failed_tool_or_verification_evidence": []},
    )

    item = payload["criteria"][0]
    assert item["passed"] is True
    assert item["candidate_excerpts"] == []
    assert item["candidate_grounding"] == "optional"
    request = next(item for item in client.requests if item["contract"] == "verification")
    assert '"candidate_grounding":"optional"' in request["prompt"]


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


def test_verification_protocol_retries_criterion_echo_and_requires_grounded_excerpt(make_config) -> None:
    candidate = "Which service is affected, what risk should be reduced, and what success criterion defines a safe rollout?"
    criterion = "Ask one grounded question covering service, risk, and success criterion."
    bad = json.dumps({
        "criteria": [
            {
                "name": "answer_quality",
                "passed": False,
                "evidence": criterion,
                "candidate_excerpts": [candidate],
            }
        ]
    })
    good = json.dumps({
        "criteria": [
            {
                "name": "answer_quality",
                "passed": True,
                "evidence": "The candidate explicitly asks for the service, risk, and success criterion in one question.",
                "candidate_excerpts": [
                    "Which service is affected",
                    "what risk should be reduced",
                    "what success criterion defines a safe rollout?",
                ],
            }
        ]
    })
    client = FakeModelClient(contract_responses={"verification": [bad, good]})
    runtime = AgentRuntime(make_config(), model_client=client)
    state = runtime.create_or_load_session()
    step = PlanStep(
        step_id="clarify",
        title="Ask clarification",
        goal="Ask one grounded clarification question.",
        kind="respond",
        expected_tool=None,
        input_text="ask",
        expected_output="one question",
        done_condition="assistant_response_nonempty",
        success_criteria=criterion,
        expected_outputs=["one grounded question"],
        verification_type="composite",
        verification_checks=[],
        required_conditions=[],
        optional_conditions=[],
    )

    payload = runtime._run_llm_verification(
        state,
        step=step,
        criteria=[{"name": "answer_quality", "criterion": criterion}],
        assistant_text=candidate,
        evidence={},
    )

    assert payload["criteria"][0]["passed"] is True
    assert all(excerpt in candidate for excerpt in payload["criteria"][0]["candidate_excerpts"])
    requests = [item for item in client.requests if item["contract"] == "verification"]
    assert len(requests) == 2
    retry_prompt = requests[1]["prompt"]
    assert "Previous rejected verification JSON:" in retry_prompt
    assert "evidence merely repeats the criterion" in retry_prompt
    assert retry_prompt.rfind("Verification protocol correction requirements from all previous attempts:") > retry_prompt.rfind("Every criterion name must appear exactly once")
    events = runtime.history.read_history(state.session_id)
    assert any(
        event.event_type == "error"
        and event.payload.get("operation") == "verification_protocol_validation"
        for event in events
    )
    assert not any(event.event_type == "fatal_system_error" for event in events)


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


def test_runtime_parse_json_rejects_trailing_text_after_structured_object(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient(responses=[]))

    with pytest.raises(RuntimeError, match="invalid JSON"):
        runtime._parse_json('{"split_task": false, "expand_task": false}\n\n17', contract_name="task_decision")


def test_runtime_parse_json_rejects_fenced_json_for_structured_calls(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient(responses=[]))

    with pytest.raises(RuntimeError, match="invalid JSON"):
        runtime._parse_json("```json\n{\"task_type\": \"structured\"}\n```", contract_name="prompt_analysis")


def test_plan_prompt_uses_configured_max_plan_steps(make_config) -> None:
    config = make_config(planner__max_plan_steps=6)
    runtime = AgentRuntime(config, model_client=FakeModelClient(responses=[]))

    assembly = runtime.prompts.build_plan_prompt(
        "Do the task.",
        prompt_mode="lean",
        context_components=[],
        tools=[],
    )

    assert "at most 6 steps including the final respond step" in assembly.prompt_text
    assert "at most 4 steps including the final respond step" not in assembly.prompt_text
    assert "success_criteria is the authoritative semantic criterion" in assembly.prompt_text
    assert "Require dependencies_completed when dependencies exist" in assembly.prompt_text
    assert "Never emit tool_effect_verified or file_contains" in assembly.prompt_text
    assert "allow the registered persisted-effect check and later whole-goal review" in assembly.prompt_text
    assert "expected_outputs is a non-empty list of output labels for the step, including respond steps" in assembly.prompt_text


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
    assert rebuilt.metrics.verification_type_distribution.get("llm_fallback", 0) == 0
    assert rebuilt.metrics.llm_fallback_rate == 0.0


def test_runtime_records_model_token_progress_for_streaming_calls(make_config) -> None:
    class SlowClient(FakeModelClient):
        def send_completion(self, payload, *, timeout_seconds: int | None = None, progress_callback=None):
            if progress_callback is not None:
                progress_callback({
                    "completion_tokens": 50,
                    "elapsed_seconds": 0.1,
                    "tokens_per_second": 500.0,
                    "first_token_seconds": 0.01,
                    "token_timeout_seconds": timeout_seconds,
                })
            return super().send_completion(payload, timeout_seconds=timeout_seconds, progress_callback=progress_callback)

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
    token_progress = [event for event in events if event.event_type == "model_token_progress"]
    assert token_progress
    assert token_progress[0].payload["tokens_per_second"] == 500.0




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
    assert semantic_contracts[:5] == [
        "prompt_analysis",
        "task_decision",
        "task_decision_semantic_review",
        "task_expansion",
        "strategy_selection",
    ]
    assert "task_plan" in contracts


def test_runtime_uses_model_authored_plan_for_model_selected_direct_answer(make_config) -> None:
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
                        "execution_mode": "direct_response",
                        "preferred_tool_name": "",
                        "confidence": 0.95,
                        "reason": "single direct assistant reply is sufficient",
                    }
                )
            ],
            "task_plan": [
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step(
                            "answer",
                            "Answer",
                            "respond",
                            expected_output="17",
                            success_criteria="Return exactly 17.",
                            verification_type="composite",
                            verification_checks=[
                                {"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"},
                                {"name": "answer_exact", "check_type": "exact_match", "actual_source": "assistant_text", "expected": "17"},
                            ],
                            required_conditions=["assistant_text_nonempty", "answer_exact"],
                            optional_conditions=[],
                        )
                    ],
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
            "answer_response": ["17"],
        }
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    contracts = [request["contract"] for request in fake_client.requests]

    assert result.assistant_text == "17"
    assert "task_plan" in contracts
    assert contracts.index("task_plan") < contracts.index("answer_response")
    semantic_contracts = [contract for contract in contracts if contract != "subagent_selection"]
    assert semantic_contracts[:5] == [
        "prompt_analysis",
        "task_decision",
        "task_decision_semantic_review",
        "strategy_selection",
        "task_plan",
    ]
    assert any(
        event.event_type == "plan_created" and event.payload.get("plan", {}).get("goal") == goal
        for event in events
    )
    assert any(event.event_type == "verification_passed" for event in events)



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
                        "direct_response": False,
                        "execution_mode": "full_plan",
                        "preferred_tool_name": "",
                        "confidence": 1.0,
                        "reason": "the model selected a full edit plan",
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
                            verification_checks=[
                                {"name": "file_contains_validation", "check_type": "file_contains", "path": str(target), "pattern": "dots not allowed"},
                            ],
                            required_conditions=["file_contains_validation"],
                            optional_conditions=[],
                        ),
                        plan_step("step_answer", "Answer", "respond", expected_output="done", success_criteria="summarize the change", depends_on=["step_edit"]),
                    ],
                )
            ],
            "tool_decision": [
                json.dumps(
                    {
                        "action": "call_tool",
                        "response": "",
                        "tool_name": "edit_text",
                        "tool_input": {},
                    }
                )
            ],
            "tool_input:edit_text": [
                json.dumps(
                    {
                        "path": str(target),
                        "operation": "replace_pattern_once",
                        "dry_run": False,
                        "start": None,
                        "end": None,
                        "position": None,
                        "expected_text": None,
                        "pattern": "blueprint = name",
                        "replacement": "if '.' in name: raise ValueError('dots not allowed')\\nblueprint = name",
                        "insertion": None,
                    }
                )
            ],
            "answer_response": ["Patched app.py and added the dot-name validation."],
        },
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert "Patched app.py" in result.assistant_text
    assert not any(event.event_type == "decision_adjusted" for event in events)
    assert any(event.event_type == "tool_called" and event.payload.get("tool_name") == "edit_text" for event in events)



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


def test_runtime_token_timeout_enters_retry_mode(make_config) -> None:
    runtime = AgentRuntime(
        make_config(),
        model_client=HangingStructuredModelClient(),
    )
    runtime._max_model_unavailable_attempts = 0
    state = runtime.create_or_load_session()

    with pytest.raises(ModelClientError, match="semantic_engine_unavailable"):
        runtime._analyze_prompt_frontend(state, "Fix app.py")

    events = runtime.history.read_history(state.session_id)
    assert any(event.event_type == "retry" and event.payload.get("operation") == "model_token_timeout" for event in events)
    assert any(event.event_type == "error" and event.payload.get("retry_mode") == "endless_until_token_progress_or_success" for event in events)
    assert not any(event.event_type == "prompt_analyzed" for event in events)












def test_strategy_validation_retries_with_rejected_json_and_salient_feedback(make_config) -> None:
    invalid = {
        "task_profile": "reading",
        "strategy_name": "conservative",
        "explore_before_commit": False,
        "tool_chain_depth": 0,
        "verification_intensity": 1.0,
        "reason": "read two files and clarify",
    }
    corrected = {**invalid, "tool_chain_depth": 2}
    client = FakeModelClient(
        contract_responses={
            "strategy_selection": [json.dumps(invalid), json.dumps(corrected)],
        }
    )
    runtime = AgentRuntime(make_config(), model_client=client)
    state = runtime.create_or_load_session()
    analysis = PromptAnalysis(
        task_type="unstructured", completeness="partial",
        requires_expansion=False, requires_decomposition=False,
        confidence=0.9, missing_required_information=True,
        detected_entities=["request.txt", "context.txt"],
        detected_goals=["read both files and clarify"],
    )
    decision = DecisionOutcome(
        split_task=False, expand_task=False, ask_user=True, assume_missing=False,
        generate_ideas=False, confidence=0.9, reason="two reads first",
        direct_response=False, execution_mode="full_plan", preferred_tool_name="",
        evidence_required_before_response=True, evidence_call_count=2,
    )

    strategy = runtime._select_strategy_frontend(
        state,
        "Read request.txt and context.txt, then ask one clarification.",
        analysis,
        decision,
    )

    assert strategy.tool_chain_depth == 2
    requests = [item for item in client.requests if item["contract"] == "strategy_selection"]
    assert len(requests) == 2
    prompt = requests[1]["prompt"]
    assert "Previous rejected strategy JSON:" in prompt
    assert "Attempt 1 strategy validation failed: tool_chain_depth must be between 1 and 3" in prompt
    assert prompt.rfind("Strategy correction requirements from all previous attempts:") > prompt.rfind("tool_chain_depth is the expected number of dependent tool steps")
    events = runtime.history.read_history(state.session_id)
    assert any(
        event.event_type == "error"
        and event.payload.get("operation") == "strategy_validation"
        for event in events
    )
    assert not any(event.event_type == "fatal_system_error" for event in events)


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
        detail_lines=["step=inspect"],
    )

    events = [
        event for event in runtime.history.read_history(state.session_id)
        if event.event_type == "prompt_built" and event.payload.get("kind") == "subagent_selection"
    ]

    assert events[0].payload["prompt_mode"] == "standard"
    prompt = events[0].payload["prompt"]
    assert "keys spawn, subagent_type, reason, and focus" in prompt
    assert "complete enabled registry" in prompt
    assert "subagent_type must be one registered enabled specialist or 'none'" in prompt
    assert "reason is one short justification when spawn=true and may be an empty string when spawn=false" in prompt
    assert "focus is the short specialist brief" in prompt
    for spec in runtime._subagents.enabled_specs():
        assert f'"name": "{spec.subagent_type}"' in prompt
        assert spec.purpose in prompt
        assert spec.role_instruction in prompt
        assert spec.usage_guidance in prompt
    assert '"input_schema": {' in prompt
    assert '"context_summary": {' in prompt
    assert '"required": [' in prompt
    for tool in runtime.tools.enabled_tools(runtime.config):
        assert f"- {tool.name}" in prompt
        assert tool.description in prompt
        assert f"input_schema: {stable_json_dumps(tool.input_schema)}" in prompt
        if tool.usage_guidance.strip():
            assert tool.usage_guidance.strip() in prompt




def test_subagent_selection_accepts_empty_reason_when_not_spawning(make_config) -> None:
    runtime = AgentRuntime(
        make_config(),
        model_client=FakeModelClient(
            contract_responses={
                "subagent_selection": [
                    json.dumps({"spawn": False, "subagent_type": "none", "reason": "", "focus": ""})
                ]
            }
        ),
    )
    state = runtime.create_or_load_session()

    decision = runtime._select_subagent_frontend(
        state,
        goal="Edit release.yaml safely.",
        purpose="Review the plan only if specialist delegation is useful.",
    )

    assert decision.spawn is False
    assert decision.subagent_type == "none"
    assert decision.reason == ""
    assert not any(event.event_type == "fatal_system_error" for event in runtime.history.read_history(state.session_id))


def test_subagent_selection_requires_reason_when_spawning(make_config) -> None:
    runtime = AgentRuntime(
        make_config(),
        model_client=FakeModelClient(
            contract_responses={
                "subagent_selection": [
                    json.dumps({"spawn": True, "subagent_type": "reviewer", "reason": "", "focus": "review"})
                ]
            }
        ),
    )
    state = runtime.create_or_load_session()

    with pytest.raises(FatalSemanticEngineError):
        runtime._select_subagent_frontend(
            state,
            goal="Review this plan.",
            purpose="Use a specialist reviewer.",
        )

    events = runtime.history.read_history(state.session_id)
    assert any(
        event.event_type == "fatal_system_error"
        and "Spawned subagent selection reason must not be empty" in event.payload.get("error", "")
        for event in events
    )


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


def test_runtime_degrades_context_when_retrieval_semantic_schema_fails(
    make_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(retrieval__backend="llm_scoring")
    runtime = AgentRuntime(config, model_client=FakeModelClient(responses=[]))
    state = runtime.create_or_load_session()
    original_build_context = runtime_module.build_context
    backends_seen: list[str] = []

    def _flaky_build_context(*args, **kwargs):
        backends_seen.append(args[0].retrieval.backend)
        if len(backends_seen) == 1:
            raise SemanticBackendProtocolError("structured relevance response violated schema")
        return original_build_context(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "build_context", _flaky_build_context)

    bundle = runtime._build_context_bundle(
        state,
        goal="Fix app.py",
        kind="answer",
        prompt_mode="standard",
    )

    assert backends_seen[:2] == ["llm_scoring", "unavailable"]
    assert bundle.retrieval_mode == "unavailable"
    events = runtime.history.read_history(state.session_id)
    assert any(event.event_type == "semantic_retrieval_degraded" for event in events)
    assert not any(event.event_type == "fatal_system_error" for event in events)


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

    assert result.assistant_text == "Task incomplete: tool_call_budget_reached. Verified success was not reached."
    assert sum(1 for event in events if event.event_type == "tool_called") == 1
    completed = next(event for event in events if event.event_type == "reasoning_completed")
    assert completed.payload["reason"] == "tool_call_budget_reached"
    assert rebuilt.metrics.tool_call_budget_hits == 1
    assert not any(request["contract"] == "answer_response" for request in fake_client.requests)


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



def test_runtime_logs_fatal_when_selected_tool_input_contract_returns_malformed_json(make_config) -> None:
    config = make_config(
        model__profile_name="small_fast",
        model__structured_output_mode="server_schema",
        tools__allow_side_effect_tools=True,
    )
    fake_client = FakeModelClient(
        contract_responses={
            "tool_decision": [
                json.dumps(
                    {
                        "action": "call_tool",
                        "response": "",
                        "tool_name": "edit_text",
                        "tool_input": {},
                    }
                )
            ],
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
                        {"name": "answer_exact", "check_type": "exact_match", "actual_source": "assistant_text", "expected": "Done"},
                    ],
                    "required_conditions": ["dependencies_completed", "assistant_text_nonempty", "answer_exact"],
                    "optional_conditions": [],
                    "fallback_strategy": "replan",
                    "depends_on": ["step_edit"],
                },
            ],
        },
        available_tools=runtime.tools.tool_names(config),
    )

    with pytest.raises(FatalSemanticEngineError):
        runtime._decide(state)
    events = runtime.history.read_history(state.session_id)

    contracts = [request["contract"] for request in fake_client.requests]
    assert "tool_input:edit_text" in contracts
    assert contracts[-1] == "tool_input:edit_text"
    assert any(event.event_type == "fatal_system_error" and event.payload.get("category") == "structured_parse_failure" for event in events)





def test_read_text_rejects_conflicting_route_fields(make_config, tmp_path) -> None:
    workspace = tmp_path
    config = make_config(tools__read_roots=[workspace])
    runtime = AgentRuntime(config, model_client=FakeModelClient(responses=[]))
    tool = runtime.tools.get("read_text")

    with pytest.raises(ToolValidationError, match="exactly one"):
        tool.validate(
            {
                "path": "pkg_469/formatter.py",
                "paths": ["pkg_469/formatter.py", "pkg_469/service.py"],
                "note_id": None,
                "reader_id": None,
                "chunk_chars": None,
                "overlap_chars": None,
                "start_offset": None,
                "end_offset": None,
            }
        )


def test_final_objective_uses_bounded_direct_evidence_context(make_config) -> None:
    prompts: list[str] = []

    def verify(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        prompts.append(prompt)
        return json.dumps(
            {
                "criteria": [
                    {
                        "name": "final_objective_satisfied",
                        "passed": True,
                        "evidence": "bounded evidence proves the answer",
                        "candidate_excerpts": ["42"],
                    }
                ]
            }
        )

    runtime = AgentRuntime(
        make_config(model__context_limit=4096),
        model_client=FakeModelClient(contract_responses={"verification": [verify]}),
    )
    state = runtime.create_or_load_session()
    runtime._record_message(state, Message(role="user", content="Compute 6 * 7.", created_at="t0"))
    state.active_plan = plan_from_payload(
        json.loads(
            plan_response(
                goal="Compute 6 * 7.",
                steps=[
                    plan_step(
                        "answer",
                        "Answer the user",
                        "respond",
                        expected_output="42",
                        success_criteria="The user receives 42.",
                    )
                ],
            )
        ),
        available_tools=[],
    )
    step = state.active_plan.steps[0]
    for index in range(40):
        runtime.history.record_event(
            state,
            "verification_passed",
            {
                "step_id": f"old_{index}",
                "conditions_met": ["x"],
                "conditions_failed": [],
                "confidence": 1.0,
                "verification_type_used": "composite",
                "reason": "x" * 4000,
                "evidence": {"nested": "y" * 8000},
            },
        )

    outcome = runtime._verify_final_objective(state, step, "42")

    assert outcome.verification_passed is True
    assert prompts
    assert len(prompts[0]) < 50_000
    assert "y" * 200 not in prompts[0]
    assert "x" * 2_100 not in prompts[0]
    assert not any(request["contract"] == "subagent_selection" for request in runtime.client.requests)


def test_verification_wire_ids_resolve_to_exact_candidate_excerpts(make_config) -> None:
    candidate = "alpha evidence\nbeta evidence\ngamma evidence"

    def verify(payload: dict[str, Any]) -> str:
        schema = payload["json_schema"]
        item = schema["properties"]["criteria"]["items"]
        allowed = item["properties"]["candidate_excerpt_id_1"]["enum"]
        selected = next(value for value in allowed if value)
        assert all(len(value) <= 3 for value in allowed)
        assert "alpha evidence" not in json.dumps(schema)
        return json.dumps(
            {
                "criteria": [
                    {
                        "name": "grounded",
                        "passed": True,
                        "evidence": "The selected ID points to exact candidate evidence.",
                        "candidate_excerpt_id_1": selected,
                        "candidate_excerpt_id_2": "",
                        "candidate_excerpt_id_3": "",
                    }
                ]
            }
        )

    runtime = AgentRuntime(
        make_config(),
        model_client=FakeModelClient(contract_responses={"verification": [verify]}),
    )
    state = runtime.create_or_load_session()
    step = plan_from_payload(
        json.loads(
            plan_response(
                goal="verify bounded evidence IDs",
                steps=[
                    plan_step(
                        "verify",
                        "Verify",
                        "respond",
                        expected_output="grounded result",
                        success_criteria="The result is grounded.",
                    )
                ],
            )
        ),
        available_tools=[],
    ).steps[0]

    result = runtime._run_llm_verification(
        state,
        step=step,
        criteria=[{"name": "grounded", "criterion": "The result is grounded."}],
        assistant_text=candidate,
        evidence={},
        include_context=False,
    )

    excerpts = result["criteria"][0]["candidate_excerpts"]
    assert excerpts
    assert all(excerpt in candidate for excerpt in excerpts)
    assert all("candidate_excerpt_id_" not in item for item in result["criteria"])


def test_runtime_installs_registered_mechanical_objective_check_once(make_config) -> None:
    edit_step = plan_step(
        "edit",
        "Edit file",
        "write",
        expected_tool="edit_text",
        expected_output="edited file",
        success_criteria="The requested edit is applied.",
        verification_checks=[
            {"name": "tool_name", "check_type": "tool_name_equals", "expected": "edit_text"},
        ],
        required_conditions=["tool_name"],
        optional_conditions=[],
    )
    answer_step = plan_step(
        "answer",
        "Answer",
        "respond",
        expected_output="summary",
        success_criteria="The answer summarizes the edit.",
        depends_on=["edit"],
    )
    payload = json.loads(plan_response(goal="Edit the file.", steps=[edit_step, answer_step]))
    plan = plan_from_payload(payload, available_tools=["edit_text"])
    runtime = AgentRuntime(
        make_config(tools__allow_stateful_tools=True, tools__allow_side_effect_tools=True),
        model_client=FakeModelClient(),
    )

    runtime._install_registered_mechanical_objective_checks(plan)
    runtime._install_registered_mechanical_objective_checks(plan)

    checks = [check for check in plan.steps[0].verification_checks if check.get("check_type") == "tool_effect_verified"]
    assert checks == [{"name": "registered_tool_effect_verified", "check_type": "tool_effect_verified"}]
    assert plan.steps[0].required_conditions.count("registered_tool_effect_verified") == 1


def test_runtime_rejects_read_file_step_with_multiple_logical_outputs(make_config) -> None:
    payload = json.loads(
        plan_response(
            goal="Read two files and answer.",
            steps=[
                plan_step(
                    "read_both",
                    "Read both files",
                    "read",
                    expected_tool="read_file",
                    expected_output="both file contents",
                    success_criteria="Both files are inspected.",
                    output_refs=["first_file", "second_file"],
                    verification_checks=[
                        {"name": "first_file", "check_type": "tool_output_nonempty"},
                        {"name": "second_file", "check_type": "tool_output_nonempty"},
                    ],
                    required_conditions=["first_file", "second_file"],
                    optional_conditions=[],
                ),
                plan_step(
                    "answer",
                    "Answer",
                    "respond",
                    expected_output="summary",
                    success_criteria="The answer uses both files.",
                    depends_on=["read_both"],
                ),
            ],
        )
    )
    plan = plan_from_payload(payload, available_tools=["read_file"])
    runtime = AgentRuntime(make_config(tools__allow_stateful_tools=True), model_client=FakeModelClient())

    with pytest.raises(PlanValidationError, match="at most 1 logical output reference"):
        runtime._validate_tool_plan_output_cardinality(plan)


def test_runtime_accepts_canonicalized_response_presence_check(make_config) -> None:
    payload = json.loads(
        plan_response(
            goal="Answer clearly.",
            steps=[
                plan_step(
                    "answer",
                    "Answer",
                    "respond",
                    expected_output="summary",
                    success_criteria="A clear summary is returned.",
                    verification_checks=[
                        {"name": "assistant_text", "check_type": "tool_output_nonempty"},
                    ],
                    required_conditions=["assistant_text"],
                    optional_conditions=[],
                )
            ],
        )
    )
    plan = plan_from_payload(payload, available_tools=[])
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient())

    runtime._validate_step_verification_compatibility(plan)

    assert plan.steps[0].verification_checks == [
        {"name": "assistant_text", "check_type": "string_nonempty", "actual_source": "assistant_text"}
    ]


def test_completed_step_ids_survive_active_plan_replacement(make_config, monkeypatch) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    state.active_plan = plan_from_payload(
        json.loads(
            plan_response(
                goal="Continue after previous work.",
                steps=[
                    plan_step(
                        "current_answer",
                        "Answer",
                        "respond",
                        expected_output="summary",
                        success_criteria="The current plan answers.",
                    )
                ],
            )
        ),
        available_tools=[],
    )
    monkeypatch.setattr(
        runtime,
        "_current_turn_history_events",
        lambda _state: [SimpleNamespace(event_type="step_completed", payload={"step_id": "previous_edit"})],
    )

    assert runtime._completed_step_ids(state) == {"previous_edit"}


def test_plan_semantic_review_evidence_does_not_reembed_prior_review_evidence(make_config, monkeypatch) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    plan = plan_from_payload(
        json.loads(
            plan_response(
                goal="Answer clearly.",
                steps=[
                    plan_step(
                        "answer",
                        "Answer",
                        "respond",
                        expected_output="summary",
                        success_criteria="A clear summary is returned.",
                    )
                ],
            )
        ),
        available_tools=[],
    )
    nested = {
        "review_kind": "plan_semantic",
        "target_id": "old-plan",
        "role": "verifier",
        "passed": True,
        "reason": "plan_semantic_review_passed",
        "evidence": {
            "review_evidence": {
                "recent_events": [
                    {"payload": {"evidence": {"review_evidence": {"recursive": "x" * 20_000}}}}
                ]
            }
        },
    }
    monkeypatch.setattr(
        runtime,
        "_current_turn_history_events",
        lambda _state: [SimpleNamespace(sequence=7, event_type="review_completed", payload=nested)],
    )

    evidence = runtime._plan_semantic_review_evidence(state, plan)
    encoded = json.dumps(evidence["recent_events"], sort_keys=True)

    assert evidence["recent_events"] == [
        {
            "sequence": 7,
            "type": "review_completed",
            "payload": {
                "review_kind": "plan_semantic",
                "target_id": "old-plan",
                "role": "verifier",
                "passed": True,
                "reason": "plan_semantic_review_passed",
            },
        }
    ]
    assert "review_evidence" not in encoded
    assert len(encoded) < 500
