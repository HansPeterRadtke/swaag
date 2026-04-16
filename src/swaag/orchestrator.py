"""Orchestrator: pick the next execution action.

The previous implementation scored every candidate action with hand-crafted
numerical formulas (``expected_gain``, ``cost``, ``uncertainty``,
``redundancy``) and a couple of heuristic signals (``_memory_signal``,
``_tool_success_boost``, ``_budget_pressure``). All of those formulas have
been removed. The orchestrator now does only the things that are *not*
semantic decisions:

* deterministic guards: no plan → replan, plan completed → stop, iteration
  limit exceeded → stop, tool-call budget exhausted → stop, repeated no-
  progress failures → stop;
* a small structural state machine over verification/failure outcomes:
  verification failed-and-requires-retry → retry, failure requires replan →
  replan, repeated-action-limit exceeded → replan, otherwise → execute the
  next ready step.

When the structural state machine is ambiguous (more than one candidate
remains), the runtime is expected to escalate to the LLM via
``action_selection_contract``. The orchestrator surfaces these candidates in
:attr:`OrchestrationDecision.candidate_actions` so the runtime can do that
without re-deriving the structural information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from swaag.failure import FailureClassification
from swaag.planner import ready_steps
from swaag.types import ExecutionAction, Plan, PlanStep, SessionState, StrategySelection
from swaag.verifier import VerificationOutcome


@dataclass(slots=True)
class ActionScore:
    """Lightweight reason record. Kept for backwards compatibility with
    callers/tests that read ``OrchestrationDecision.scores``."""

    action: ExecutionAction
    expected_gain: float = 0.0
    cost: float = 0.0
    uncertainty: float = 0.0
    redundancy: float = 0.0
    score: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class OrchestrationDecision:
    action: ExecutionAction
    step: PlanStep | None
    ready_step_ids: list[str]
    stop_reason: str | None
    scores: list[ActionScore] = field(default_factory=list)
    candidate_actions: list[ExecutionAction] = field(default_factory=list)
    requires_llm_decision: bool = False


def choose_next_step(plan: Plan | None, state: SessionState) -> PlanStep | None:
    """Pick the next ready step. Deterministic structural ordering only."""

    candidates = ready_steps(plan)
    if not candidates:
        return None
    current_id = state.working_memory.current_step_id
    ranked = sorted(
        candidates,
        key=lambda step: (
            0 if step.step_id == current_id else 1,
            0 if step.kind != "respond" else 1,
            len(step.depends_on),
            step.step_id,
        ),
    )
    return ranked[0]


def select_action(
    *,
    state: SessionState,
    plan: Plan | None,
    strategy: StrategySelection,
    verification: VerificationOutcome | None,
    failure: FailureClassification | None,
    repeated_action_count: int,
    iteration: int,
    max_iterations: int,
    turn_tool_calls: int = 0,
    tool_call_budget: int | None = None,
    no_progress_failures: int = 0,
    no_progress_failure_limit: int | None = None,
    current_step: PlanStep | None = None,
    running_background_jobs: int = 0,
) -> OrchestrationDecision:
    step = current_step if current_step is not None and current_step.status == "running" else choose_next_step(plan, state)
    ready_ids = [item.step_id for item in ready_steps(plan)]

    # ── Deterministic guards ───────────────────────────────────────────────
    if plan is None:
        return OrchestrationDecision(
            action="replan",
            step=None,
            ready_step_ids=[],
            stop_reason=None,
            scores=[ActionScore("replan", reason="no_active_plan")],
            candidate_actions=["replan"],
        )
    if plan.status == "completed":
        return OrchestrationDecision(
            action="stop",
            step=None,
            ready_step_ids=ready_ids,
            stop_reason="goal_satisfied",
            scores=[ActionScore("stop", reason="plan_completed")],
            candidate_actions=["stop"],
        )
    if step is None and running_background_jobs > 0:
        return OrchestrationDecision(
            action="wait",
            step=None,
            ready_step_ids=ready_ids,
            stop_reason=None,
            scores=[ActionScore("wait", reason="background_jobs_running")],
            candidate_actions=["wait"],
        )
    if step is None:
        return OrchestrationDecision(
            action="stop",
            step=None,
            ready_step_ids=ready_ids,
            stop_reason="no_ready_steps",
            scores=[ActionScore("stop", reason="no_ready_steps")],
            candidate_actions=["stop"],
        )
    if iteration >= max_iterations:
        return OrchestrationDecision(
            action="stop",
            step=step,
            ready_step_ids=ready_ids,
            stop_reason="max_iterations_reached",
            scores=[ActionScore("stop", reason="max_iterations_reached")],
            candidate_actions=["stop"],
        )
    if tool_call_budget is not None and step.kind in {"tool", "read", "write", "note"} and turn_tool_calls >= tool_call_budget:
        return OrchestrationDecision(
            action="stop",
            step=step,
            ready_step_ids=ready_ids,
            stop_reason="tool_call_budget_reached",
            scores=[ActionScore("stop", reason="tool_call_budget_reached")],
            candidate_actions=["stop"],
        )
    no_progress_limit = (
        no_progress_failure_limit
        if no_progress_failure_limit is not None
        else max(2, strategy.replan_after_failures + 1)
    )
    if no_progress_failures >= no_progress_limit:
        return OrchestrationDecision(
            action="stop",
            step=step,
            ready_step_ids=ready_ids,
            stop_reason="no_progress_possible",
            scores=[ActionScore("stop", reason="no_progress_possible")],
            candidate_actions=["stop"],
        )

    # ── Structural state machine ──────────────────────────────────────────
    candidates: list[ExecutionAction] = []
    scores: list[ActionScore] = []

    forced_replan = repeated_action_count > strategy.retry_same_action_limit
    if forced_replan:
        candidates.append("replan")
        scores.append(ActionScore("replan", reason="repeated_action_limit_exceeded"))

    if verification is not None and not verification.passed:
        if verification.requires_retry:
            candidates.append("retry_step")
            scores.append(ActionScore("retry_step", reason=f"verification_failed={verification.reason}"))
        if verification.requires_replan:
            if "replan" not in candidates:
                candidates.append("replan")
                scores.append(ActionScore("replan", reason=f"verification_requires_replan={verification.reason}"))

    if failure is not None and failure.requires_replan and "replan" not in candidates:
        candidates.append("replan")
        scores.append(ActionScore("replan", reason=f"failure={failure.kind}"))

    default_action: ExecutionAction = "answer_directly" if step.kind == "respond" else "execute_step"
    if default_action not in candidates:
        candidates.append(default_action)
        scores.append(ActionScore(default_action, reason=f"ready_step={step.step_id}"))

    # ── Pick the action ───────────────────────────────────────────────────
    if forced_replan:
        chosen: ExecutionAction = "replan"
    elif verification is not None and not verification.passed and verification.requires_retry:
        chosen = "retry_step"
    elif verification is not None and not verification.passed and verification.requires_replan:
        chosen = "replan"
    elif failure is not None and failure.requires_replan:
        chosen = "replan"
    else:
        chosen = default_action

    requires_llm = len(candidates) > 1 and not forced_replan
    return OrchestrationDecision(
        action=chosen,
        step=step,
        ready_step_ids=ready_ids,
        stop_reason=None,
        scores=scores,
        candidate_actions=candidates,
        requires_llm_decision=requires_llm,
    )


def action_from_payload(payload: dict, *, allowed_actions: list[ExecutionAction] | None = None) -> ExecutionAction:
    """Parse and validate an LLM action_selection response."""

    raw_action = str(payload.get("action", "")).strip()
    valid: tuple[ExecutionAction, ...] = (
        "execute_step",
        "retry_step",
        "replan",
        "wait",
        "stop",
        "answer_directly",
    )
    if raw_action not in valid:
        raise ValueError(f"Unknown action: {raw_action}")
    if allowed_actions is not None and raw_action not in allowed_actions:
        raise ValueError(f"Action {raw_action} not in allowed set {allowed_actions}")
    return raw_action  # type: ignore[return-value]
