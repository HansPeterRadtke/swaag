from __future__ import annotations

from swaag.types import Plan, SessionState


class PlanningSubsystem:
    name = "planning"

    def run(
        self,
        runtime,
        state: SessionState,
        goal: str,
        *,
        replan_reason: str = "",
        replan_attempt: int = 0,
        update_existing: bool = False,
        required_tools: list[str] | None = None,
    ) -> Plan:
        runtime.history.record_event(state, "subsystem_started", {"subsystem": self.name, "step_id": None, "goal": goal})
        runtime.history.record_event(
            state,
            "subsystem_progress",
            {"subsystem": self.name, "step_id": None, "progress": "planning_started"},
        )
        plan = runtime._generate_plan(
            state,
            goal,
            update_existing=update_existing,
            replan_reason=replan_reason,
            replan_attempt=replan_attempt,
            required_tools=required_tools,
        )
        runtime.history.record_event(
            state,
            "subsystem_progress",
            {"subsystem": self.name, "step_id": None, "progress": f"plan_ready:{plan.plan_id}"},
        )
        runtime.history.record_event(state, "subsystem_completed", {"subsystem": self.name, "step_id": None, "success": True, "result_summary": f"plan:{plan.plan_id}"})
        return plan
