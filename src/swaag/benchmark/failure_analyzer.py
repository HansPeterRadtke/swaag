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
        metrics = state.metrics
        rejected = [event for event in events if event.event_type == "agent_action_rejected"]
        tool_errors = [event for event in events if event.event_type == "tool_error"]
        selected_actions = [event for event in events if event.event_type == "agent_action_selected"]
        if runtime_error is not None:
            return FailureAnalysis(
                category="premature_termination",
                reason=str(runtime_error),
                evidence={"error_type": runtime_error.__class__.__name__},
                subsystem="runtime",
                improvement_hints=["Return recoverable runtime evidence to the action loop when possible."],
            )
        if tool_errors:
            last = tool_errors[-1]
            return FailureAnalysis(
                category="wrong_tool_usage",
                reason="A tool call failed before the benchmark contract was satisfied.",
                evidence={
                    "tool_name": last.payload.get("tool_name"),
                    "error": last.payload.get("error"),
                    "error_type": last.payload.get("error_type"),
                },
                subsystem="tooling",
                improvement_hints=["Use the structured tool error as evidence and choose a corrected call."],
            )
        if not deterministic_verification_passed and not selected_actions:
            return FailureAnalysis(
                category="prompt_misunderstanding",
                reason="The task produced no valid agent action before deterministic verification failed.",
                evidence={"model_calls": metrics.model_calls, "tool_calls": metrics.tool_calls},
                subsystem="action_loop",
                improvement_hints=["Improve the action prompt or response validation for the failing task shape."],
            )
        if not deterministic_verification_passed:
            return FailureAnalysis(
                category="premature_termination",
                reason="The run ended before satisfying the deterministic benchmark verification contract.",
                evidence={
                    "actions": len(selected_actions),
                    "tool_calls": metrics.tool_calls,
                    "tool_failures": metrics.tool_failures,
                    "rejected_actions": len(rejected),
                },
                subsystem="runtime",
                improvement_hints=["Continue from the deterministic verifier evidence instead of claiming completion."],
            )
        return FailureAnalysis(
            category="evaluator_mistake",
            reason="Failure analysis was requested even though deterministic verification passed.",
            evidence={"actions": len(selected_actions), "tool_calls": metrics.tool_calls},
            subsystem="evaluator",
            improvement_hints=["Do not classify a deterministically passing seed as failed."],
        )
