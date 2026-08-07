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
        metrics = state.metrics
        last_reason = ""
        for event in reversed(events):
            candidate = event.payload.get("reason") or event.payload.get("stop_reason") or event.payload.get("status")
            if candidate:
                last_reason = str(candidate)
                break
        no_progress_stops = int(getattr(metrics, "no_progress_stops", 0))
        steps_started = int(getattr(metrics, "steps_started", getattr(metrics, "action_count", 0)))
        steps_completed = int(getattr(metrics, "steps_completed", getattr(metrics, "successful_turns", 0)))
        verification_failures = int(getattr(metrics, "verification_failures", 0))
        if any(event.event_type == "tool_mismatch_rejected" for event in events):
            rejected = next(event for event in events if event.event_type == "tool_mismatch_rejected")
            return FailureAnalysis(
                category="wrong_tool_usage",
                reason="Selected tool did not exactly match the model-authored plan step.",
                evidence={"selected_tool": rejected.payload.get("selected_tool"), "expected_tool": rejected.payload.get("expected_tool")},
                subsystem="tooling",
                improvement_hints=["Return mismatch evidence to the model and require a corrected constrained tool call."],
            )
        if any(event.event_type == "duplicate_action_detected" for event in events):
            duplicate = next(event for event in events if event.event_type == "duplicate_action_detected")
            return FailureAnalysis(
                category="loop_no_progress",
                reason="The same action repeated without producing new progress.",
                evidence={"action_key": duplicate.payload.get("action_key"), "count": duplicate.payload.get("count")},
                subsystem="action_loop",
                improvement_hints=["Force a different action after duplicate actions.", "Tighten repeated-action suppression for helper tool loops."],
            )
        if "verification_started" in event_types and "verification_completed" not in event_types:
            return FailureAnalysis(
                category="missing_verification",
                reason="Verification started but never completed.",
                evidence={"event_types": event_types},
                subsystem="verification",
                improvement_hints=["Ensure every executable step reaches verification_completed.", "Treat missing verification evidence as a hard runtime error."],
            )
        if no_progress_stops > 0 or last_reason in {"no_progress_possible", "max_iterations_reached"}:
            return FailureAnalysis(
                category="loop_no_progress",
                reason="The agent stopped because it could not make further progress.",
                evidence={"no_progress_stops": no_progress_stops, "last_reason": last_reason},
                subsystem="action_loop",
                improvement_hints=["Change action strategy earlier after duplicate actions.", "Reduce retry budget for repeated verifier failures."],
            )
        if runtime_error is not None:
            return FailureAnalysis(
                category="premature_termination",
                reason=str(runtime_error),
                evidence={"error_type": runtime_error.__class__.__name__},
                subsystem="runtime",
                improvement_hints=["Classify runtime exceptions earlier and return recoverable evidence to the action loop."],
            )
        fallback_reasons = {
            "replan_limit_reached",
            "subsystem_failed",
            "tool_failed",
            "step_failed",
            "fatal_step_error",
            "fatal_system_error",
            "budget_exhausted",
        }
        if not deterministic_verification_passed and (last_reason in fallback_reasons or any(event.event_type == "step_failed" for event in events)):
            return FailureAnalysis(
                category="premature_termination",
                reason="The agent stopped before satisfying the benchmark contract.",
                evidence={"last_reason": last_reason, "steps_completed": steps_completed, "verification_failures": verification_failures},
                subsystem="runtime",
                improvement_hints=["Continue repair loops until the benchmark command/contract passes, or surface the exact blocker instead of a generic fallback."],
            )
        if not deterministic_verification_passed and verification_failures == 0 and steps_completed > 0 and last_reason in {"completed", "all_steps_completed", "response_ready"}:
            return FailureAnalysis(
                category="evaluator_mistake",
                reason="The runtime reported successful completion without failing verification, but deterministic benchmark verification failed.",
                evidence={"steps_completed": steps_completed, "last_reason": last_reason},
                subsystem="evaluator",
                improvement_hints=["Tighten evaluator evidence thresholds.", "Reject completion when deterministic benchmark signals are incomplete."],
            )
        if not deterministic_verification_passed and steps_completed == 0:
            return FailureAnalysis(
                category="prompt_misunderstanding",
                reason="The task never progressed to a completed step.",
                evidence={"steps_started": steps_started, "steps_completed": steps_completed},
                subsystem="prompt_analyzer",
                improvement_hints=["Improve prompt understanding for the failing task type.", "Add a stronger clarification/expansion decision for underspecified prompts."],
            )
        return FailureAnalysis(
            category="premature_termination",
            reason="The run ended before satisfying the benchmark verification contract.",
            evidence={"last_reason": last_reason, "verification_failures": verification_failures},
            subsystem="runtime",
            improvement_hints=["Inspect the stop reason and verification evidence for early termination."],
        )
