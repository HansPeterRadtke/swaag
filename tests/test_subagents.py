from __future__ import annotations

import json

from swaag.context_builder import build_context
from swaag.planner import create_direct_tool_plan
from swaag.runtime import AgentRuntime
from swaag.subagents import SubagentManager
from swaag.tokens import ConservativeEstimator
from swaag.types import Message, PlanStep, SemanticMemoryItem
from swaag.types import ToolExecutionResult
from swaag.verification import VerificationOutcome

from tests.helpers import FakeModelClient, plan_response, plan_step


def test_subagent_manager_scopes_state_isolation(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    state.messages.append(Message(role="user", content="keep", created_at="t1"))
    manager = SubagentManager(backend_mode="degraded_lexical")

    scoped = manager.scoped_state(state)
    scoped.messages.clear()

    assert len(state.messages) == 1
    assert len(scoped.messages) == 0


def test_reviewer_subagent_rejects_partial_result(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    plan = create_direct_tool_plan("Compute the value", "calculator")
    step = plan.steps[0]
    verification = VerificationOutcome(
        verification_passed=False,
        verification_type_used="value",
        conditions_met=[],
        conditions_failed=["exact_result"],
        evidence={"actual": 3, "expected": 4},
        confidence=1.0,
        reason="exact_result_failed",
        requires_retry=True,
        requires_replan=False,
    )
    subsystem_result = type(
        "SubsystemResult",
        (),
        {"tool_results": [ToolExecutionResult(tool_name="calculator", output={"result": 3}, display_text="3")], "assistant_text": ""},
    )()

    report = runtime._subagents.review_result(state, step, verification=verification, subsystem_result=subsystem_result)

    assert report.accepted is False
    assert report.recommended_action == "retry_or_replan"


def test_reviewer_subagent_accepts_exact_literal_expected_output(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    step = PlanStep(
        step_id="step_answer",
        title="Answer",
        goal="Answer",
        kind="respond",
        expected_tool=None,
        input_text="reply",
        expected_output="written",
        expected_outputs=["written"],
        done_condition="assistant_response_nonempty",
        success_criteria="The assistant replies written.",
        verification_type="llm_fallback",
        verification_checks=[{"name": "assistant_text_nonempty", "check_type": "string_nonempty", "actual_source": "assistant_text"}],
        required_conditions=["assistant_text_nonempty"],
        optional_conditions=[],
    )
    verification = VerificationOutcome(
        verification_passed=True,
        verification_type_used="llm_fallback",
        conditions_met=["assistant_text_nonempty", "perspective:reviewer"],
        conditions_failed=[],
        evidence={"literal_match": True},
        confidence=1.0,
        reason="verified",
        requires_retry=False,
        requires_replan=False,
    )
    subsystem_result = type(
        "SubsystemResult",
        (),
        {"tool_results": [], "assistant_text": "written"},
    )()

    report = runtime._subagents.review_result(state, step, verification=verification, subsystem_result=subsystem_result)

    assert report.accepted is True
    assert report.evidence["literal_exact_match"] is True


def test_runtime_records_subagent_events_during_review(make_config) -> None:
    goal = "Use the calculator tool to compute 2 + 2."
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc", "Compute", "tool", expected_tool="calculator", expected_output="value", success_criteria="calculated"),
                    # The step uses the literal expected_output "4" so the
                    # reviewer perspective can deterministically match the
                    # assistant's response without relying on LLM scoring.
                    plan_step("step_answer", "Answer", "respond", expected_output="4", success_criteria="answer", depends_on=["step_calc"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}}),
            "4",
        ]
    )
    runtime = AgentRuntime(make_config(), model_client=fake_client)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert any(event.event_type == "subagent_spawned" for event in events)
    assert any(event.event_type == "subagent_reported" for event in events)


def test_runtime_records_retriever_subagent_events_during_context_build(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    state.messages.append(Message(role="user", content="The parser bug is described in older notes.", created_at="t1"))
    state.semantic_memory.append(
        SemanticMemoryItem(
            memory_id="m1",
            memory_kind="semantic",
            content="config.py contains the parsing bug",
            source_event_id="e1",
            trust_level="trusted",
            tags=["config.py"],
            created_at="t1",
        )
    )

    runtime._build_context_bundle(
        state,
        goal="Fix config parsing in app.py and verify behavior.",
        kind="plan",
        prompt_mode="standard",
        for_planning=True,
    )
    events = runtime.history.read_history(state.session_id)

    assert any(event.event_type == "subagent_selection_resolved" for event in events)
    assert any(
        event.event_type == "subagent_spawned" and event.payload.get("subagent_type") == "retriever"
        for event in events
    )
    assert any(
        event.event_type == "subagent_reported" and event.payload.get("subagent_type") == "retriever"
        for event in events
    )


def test_retriever_subagent_produces_focused_artifact(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    state.messages.append(Message(role="user", content="Fix config parsing in app.py and verify behavior.", created_at="t1"))
    bundle = build_context(
        runtime.config,
        state,
        ConservativeEstimator(),
        goal="Fix config parsing in app.py and verify behavior.",
        available_tools=runtime.tools.prompt_tuples(runtime.config),
    )

    report = runtime._subagents.retrieve_context(state, goal="Fix config parsing in app.py and verify behavior.", bundle=bundle)

    assert report.accepted is True
    assert report.artifacts
    assert report.artifacts[0].artifact_type == "retrieval_focus"
    assert report.artifacts[0].content["focused_item_ids"]
    assert report.artifacts[0].content["scoped_query"]


def test_planner_subagent_replan_artifact_is_explicit(make_config) -> None:
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    plan = create_direct_tool_plan("Use the calculator tool to compute 2 + 2.", "calculator")
    state.active_plan = plan

    report = runtime._subagents.replan(state, goal=plan.goal, current_plan=plan, failure_reason="verification_failed")

    assert report.accepted is True
    assert report.recommended_action == "replan"
    assert report.artifacts[0].artifact_type == "replan_request"
    assert "replan_guidance" in report.artifacts[0].content
