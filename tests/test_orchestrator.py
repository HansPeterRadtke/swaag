from __future__ import annotations

from swaag.orchestrator import choose_next_step, select_action
from swaag.types import Plan, PlanStep, SessionMetrics, SessionState, StrategySelection, WorkingMemory
from swaag.verifier import VerificationOutcome


def _strategy() -> StrategySelection:
    return StrategySelection(
        strategy_name="conservative",
        mode="conservative",
        explore_before_commit=False,
        validate_assumptions=True,
        simplify_if_stuck=True,
        switch_on_failure=True,
        reason="test",
        retry_same_action_limit=1,
    )


def test_choose_next_step_respects_dependencies() -> None:
    step_a = PlanStep(
        step_id="a",
        title="Read",
        goal="Read",
        kind="read",
        expected_tool="read_text",
        input_text="f",
        expected_output="text",
        done_condition="tool_result:read_text",
        success_criteria="done",
        status="completed",
    )
    step_b = PlanStep(
        step_id="b",
        title="Compute",
        goal="Compute",
        kind="tool",
        expected_tool="calculator",
        input_text="2+2",
        expected_output="num",
        done_condition="tool_result:calculator",
        success_criteria="done",
        depends_on=["a"],
    )
    step_c = PlanStep(
        step_id="c",
        title="Answer",
        goal="Answer",
        kind="respond",
        expected_tool=None,
        input_text="answer",
        expected_output="text",
        done_condition="assistant_response_nonempty",
        success_criteria="done",
        depends_on=["b"],
    )
    plan = Plan(plan_id="p", goal="goal", steps=[step_a, step_b, step_c], success_criteria="done", fallback_strategy="replan", status="active", created_at="t0", updated_at="t0", current_step_id="b")
    state = SessionState(session_id="s", created_at="t0", updated_at="t0", config_fingerprint="cfg", model_base_url="http://x")

    assert choose_next_step(plan, state).step_id == "b"


def test_orchestrator_stops_for_completed_plan() -> None:
    plan = Plan(plan_id="p", goal="goal", steps=[], success_criteria="done", fallback_strategy="replan", status="completed", created_at="t0", updated_at="t0")
    state = SessionState(session_id="s", created_at="t0", updated_at="t0", config_fingerprint="cfg", model_base_url="http://x")

    decision = select_action(
        state=state,
        plan=plan,
        strategy=_strategy(),
        verification=None,
        failure=None,
        repeated_action_count=0,
        iteration=0,
        max_iterations=4,
    )

    assert decision.action == "stop"
    assert decision.stop_reason == "goal_satisfied"


def test_orchestrator_prefers_replan_after_repeat_limit() -> None:
    step = PlanStep(
        step_id="calc",
        title="Compute",
        goal="Compute",
        kind="tool",
        expected_tool="calculator",
        input_text="2+2",
        expected_output="num",
        done_condition="tool_result:calculator",
        success_criteria="done",
        status="running",
    )
    plan = Plan(plan_id="p", goal="goal", steps=[step], success_criteria="done", fallback_strategy="replan", status="active", created_at="t0", updated_at="t0", current_step_id="calc")
    state = SessionState(
        session_id="s",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url="http://x",
        working_memory=WorkingMemory(current_step_id="calc"),
        metrics=SessionMetrics(tool_failure_counts={"calculator": 2}),
    )

    decision = select_action(
        state=state,
        plan=plan,
        strategy=_strategy(),
        verification=VerificationOutcome(
            verification_passed=False,
            verification_type_used="composite",
            conditions_met=[],
            conditions_failed=["wrong_result"],
            evidence={"wrong_result": {"actual": 4, "expected": 5}},
            confidence=0.1,
            reason="incorrect",
            requires_retry=True,
            requires_replan=False,
        ),
        failure=None,
        repeated_action_count=3,
        iteration=1,
        max_iterations=4,
        current_step=step,
    )

    assert decision.action == "replan"


def test_orchestrator_stops_when_tool_call_budget_is_reached() -> None:
    step = PlanStep(
        step_id="read",
        title="Read",
        goal="Read",
        kind="read",
        expected_tool="read_text",
        input_text="sample.txt",
        expected_output="text",
        done_condition="tool_result:read_text",
        success_criteria="done",
    )
    plan = Plan(plan_id="p", goal="goal", steps=[step], success_criteria="done", fallback_strategy="replan", status="active", created_at="t0", updated_at="t0", current_step_id="read")
    state = SessionState(session_id="s", created_at="t0", updated_at="t0", config_fingerprint="cfg", model_base_url="http://x")

    decision = select_action(
        state=state,
        plan=plan,
        strategy=_strategy(),
        verification=None,
        failure=None,
        repeated_action_count=0,
        iteration=0,
        max_iterations=4,
        turn_tool_calls=2,
        tool_call_budget=2,
    )

    assert decision.action == "stop"
    assert decision.stop_reason == "tool_call_budget_reached"


def test_orchestrator_stops_when_no_progress_is_possible() -> None:
    step = PlanStep(
        step_id="calc",
        title="Compute",
        goal="Compute",
        kind="tool",
        expected_tool="calculator",
        input_text="2+2",
        expected_output="num",
        done_condition="tool_result:calculator",
        success_criteria="done",
    )
    plan = Plan(plan_id="p", goal="goal", steps=[step], success_criteria="done", fallback_strategy="replan", status="active", created_at="t0", updated_at="t0", current_step_id="calc")
    state = SessionState(session_id="s", created_at="t0", updated_at="t0", config_fingerprint="cfg", model_base_url="http://x")

    decision = select_action(
        state=state,
        plan=plan,
        strategy=_strategy(),
        verification=None,
        failure=None,
        repeated_action_count=0,
        iteration=1,
        max_iterations=4,
        no_progress_failures=3,
        no_progress_failure_limit=3,
    )

    assert decision.action == "stop"
    assert decision.stop_reason == "no_progress_possible"


def test_orchestrator_waits_when_only_background_jobs_remain() -> None:
    plan = Plan(
        plan_id="p",
        goal="goal",
        steps=[],
        success_criteria="done",
        fallback_strategy="replan",
        status="active",
        created_at="t0",
        updated_at="t0",
    )
    state = SessionState(session_id="s", created_at="t0", updated_at="t0", config_fingerprint="cfg", model_base_url="http://x")

    decision = select_action(
        state=state,
        plan=plan,
        strategy=_strategy(),
        verification=None,
        failure=None,
        repeated_action_count=0,
        iteration=0,
        max_iterations=4,
        running_background_jobs=1,
    )

    assert decision.action == "wait"
    assert decision.stop_reason is None
