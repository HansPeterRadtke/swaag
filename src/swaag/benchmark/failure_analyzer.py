from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swaag.types import HistoryEvent, SessionState


FailureCategory = str


@dataclass(slots=True)
class FailureAnalysis:
    category: FailureCategory
    reason: str
    evidence: dict[str, Any]
    subsystem: str | None = None
    improvement_hints: list[str] | None = None


class FailureAnalyzer:
    def analyze(
        self,
        *,
        state: SessionState,
        events: list[HistoryEvent],
        deterministic_verification_passed: bool,
        runtime_error: Exception | None,
    ) -> FailureAnalysis:
        event_types = [event.event_type for event in events]
        last_reason = state.metrics.last_reasoning_reason or ""
        if any(event.event_type == "tool_graph_rejected" for event in events):
            rejected = next(event for event in events if event.event_type == "tool_graph_rejected")
            return FailureAnalysis(
                category="wrong_tool_usage",
                reason="Selected tool did not satisfy the expected tool graph.",
                evidence={"selected_tool": rejected.payload.get("selected_tool"), "expected_tool": rejected.payload.get("expected_tool")},
                subsystem="tooling",
                improvement_hints=["Tighten tool selection policy for the task type.", "Increase tool graph constraints for file and coding tasks."],
            )
        if any(event.event_type == "duplicate_action_detected" for event in events):
            duplicate = next(event for event in events if event.event_type == "duplicate_action_detected")
            return FailureAnalysis(
                category="loop_no_progress",
                reason="The same action repeated without producing new progress.",
                evidence={"action_key": duplicate.payload.get("action_key"), "count": duplicate.payload.get("count")},
                subsystem="orchestrator",
                improvement_hints=["Force replanning earlier after duplicate actions.", "Tighten repeated-action suppression for helper tool loops."],
            )
        if any(event.event_type == "error" and str(event.payload.get("operation")) in {"plan", "plan_validation"} for event in events):
            error = next(event for event in events if event.event_type == "error" and str(event.payload.get("operation")) in {"plan", "plan_validation"})
            return FailureAnalysis(
                category="bad_planning",
                reason=str(error.payload.get("error", "planner returned an invalid plan")),
                evidence={"operation": error.payload.get("operation"), "error_type": error.payload.get("error_type")},
                subsystem="planner",
                improvement_hints=["Improve plan validation before execution.", "Strengthen planner prompt constraints for mandatory steps and verifier contracts."],
            )
        if "verification_started" in event_types and "verification_completed" not in event_types:
            return FailureAnalysis(
                category="missing_verification",
                reason="Verification started but never completed.",
                evidence={"event_types": event_types},
                subsystem="verification",
                improvement_hints=["Ensure every executable step reaches verification_completed.", "Treat missing verification evidence as a hard runtime error."],
            )
        if state.metrics.no_progress_stops > 0 or last_reason in {"no_progress_possible", "max_iterations_reached"}:
            return FailureAnalysis(
                category="loop_no_progress",
                reason="The agent stopped because it could not make further progress.",
                evidence={"no_progress_stops": state.metrics.no_progress_stops, "last_reason": last_reason},
                subsystem="orchestrator",
                improvement_hints=["Trigger replanning earlier after duplicate actions.", "Reduce retry budget for repeated verifier failures."],
            )
        if runtime_error is not None:
            return FailureAnalysis(
                category="premature_termination",
                reason=str(runtime_error),
                evidence={"error_type": runtime_error.__class__.__name__},
                subsystem="runtime",
                improvement_hints=["Classify runtime exceptions earlier and convert recoverable cases into replans."],
            )
        if not deterministic_verification_passed and state.metrics.verification_failures == 0 and state.metrics.steps_completed > 0:
            return FailureAnalysis(
                category="evaluator_mistake",
                reason="The runtime completed work without any failing verification, but deterministic benchmark verification failed.",
                evidence={"steps_completed": state.metrics.steps_completed},
                subsystem="evaluator",
                improvement_hints=["Tighten evaluator evidence thresholds.", "Reject completion when deterministic benchmark signals are incomplete."],
            )
        if not deterministic_verification_passed and state.metrics.steps_completed == 0:
            return FailureAnalysis(
                category="prompt_misunderstanding",
                reason="The task never progressed to a completed step.",
                evidence={"steps_started": state.metrics.steps_started, "steps_completed": state.metrics.steps_completed},
                subsystem="prompt_analyzer",
                improvement_hints=["Improve prompt understanding for the failing task type.", "Add a stronger clarification/expansion decision for underspecified prompts."],
            )
        return FailureAnalysis(
            category="premature_termination",
            reason="The run ended before satisfying the benchmark verification contract.",
            evidence={"last_reason": last_reason, "verification_failures": state.metrics.verification_failures},
            subsystem="runtime",
            improvement_hints=["Inspect the stop reason and verification evidence for early termination."],
        )
