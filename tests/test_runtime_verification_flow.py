from __future__ import annotations

import json

import swaag.runtime as runtime_module
from swaag.runtime import AgentRuntime
from swaag.verification import VerificationOutcome

from tests.helpers import FakeModelClient, plan_response, plan_step


def test_runtime_calls_verification_before_evaluator(make_config, monkeypatch) -> None:
    config = make_config()
    goal = "Use the calculator tool to compute 2 + 2."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
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
        ),
    )
    call_order: list[str] = []
    observed: dict[str, object] = {}
    original_verify = runtime._verification.verify_step
    original_evaluate = runtime_module.evaluate_verification

    def wrapped_verify(*, runtime, state, plan, step, artifacts):  # noqa: ANN001
        call_order.append("verify")
        observed["artifacts_type"] = artifacts.__class__.__name__
        return original_verify(runtime=runtime, state=state, plan=plan, step=step, artifacts=artifacts)

    def wrapped_evaluate(step, verification):  # noqa: ANN001
        call_order.append("evaluate")
        observed["verification_type"] = verification.__class__.__name__
        observed["verification_payload"] = verification
        return original_evaluate(step, verification)

    monkeypatch.setattr(runtime._verification, "verify_step", wrapped_verify)
    monkeypatch.setattr(runtime_module, "evaluate_verification", wrapped_evaluate)

    result = runtime.run_turn(goal)

    assert result.assistant_text == "4"
    assert call_order[:2] == ["verify", "evaluate"]
    assert observed["artifacts_type"] == "VerificationArtifacts"
    assert observed["verification_type"] == "VerificationOutcome"


def test_runtime_records_verification_events_before_evaluation_effects(make_config) -> None:
    config = make_config()
    goal = "Use the calculator tool to compute 2 + 2."
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
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
        ),
    )
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    verification_started = next(event for event in events if event.event_type == "verification_started")
    verification_completed = next(event for event in events if event.event_type == "verification_completed")
    verification_passed = next(event for event in events if event.event_type == "verification_passed")
    step_completed = next(event for event in events if event.event_type == "step_completed")

    assert verification_started.sequence < verification_completed.sequence < verification_passed.sequence < step_completed.sequence
