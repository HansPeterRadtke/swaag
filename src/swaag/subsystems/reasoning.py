from __future__ import annotations

from swaag.subsystems.base import SubsystemExecutionResult
from swaag.types import PlanStep, SessionState


class ReasoningSubsystem:
    name = "reasoning"

    def run(self, runtime, state: SessionState, step: PlanStep) -> SubsystemExecutionResult:
        runtime.history.record_event(state, "subsystem_started", {"subsystem": self.name, "step_id": step.step_id, "goal": step.goal})
        runtime.history.record_event(
            state,
            "subsystem_progress",
            {"subsystem": self.name, "step_id": step.step_id, "progress": "reasoning_started"},
        )
        assistant_text, report = runtime._answer(state)
        runtime.history.record_event(
            state,
            "subsystem_progress",
            {"subsystem": self.name, "step_id": step.step_id, "progress": "reasoning_finished"},
        )
        runtime.history.record_event(state, "subsystem_completed", {"subsystem": self.name, "step_id": step.step_id, "success": bool(assistant_text.strip()), "result_summary": assistant_text[:120]})
        return SubsystemExecutionResult(
            subsystem_name=self.name,
            success=bool(assistant_text.strip()),
            progress=["final_answer_generated"],
            budget_reports=[report],
            assistant_text=assistant_text,
            evaluation=None,
        )
