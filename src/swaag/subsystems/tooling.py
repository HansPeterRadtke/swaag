from __future__ import annotations

from swaag.subsystems.base import SubsystemExecutionResult
from swaag.types import PlanStep, SessionState
from swaag.utils import stable_json_dumps


class ToolSubsystem:
    name = "tool"

    _REFINEMENT_TOOLS = frozenset({"edit_text", "write_file", "shell_command", "run_tests"})
    _REFINABLE_CHECK_TYPES = frozenset({"file_contains", "file_exists", "exact_match", "string_match", "numeric_tolerance"})

    def _should_preview(self, tool_name: str, *, state: SessionState) -> bool:
        return state.active_plan is not None and tool_name in self._REFINEMENT_TOOLS

    def _can_continue_refinement(self, step: PlanStep, preview) -> bool:
        if not preview.requires_retry:
            return False
        checks_by_name = {str(check.get("name", "")): str(check.get("check_type", "")) for check in step.verification_checks}
        failed_names = [name for name in preview.conditions_failed if not str(name).startswith("perspective:")]
        if not failed_names:
            return False
        failed_types = {checks_by_name.get(name, "") for name in failed_names}
        failed_types.discard("")
        return bool(failed_types) and failed_types.issubset(self._REFINABLE_CHECK_TYPES)

    def run(self, runtime, state: SessionState, step: PlanStep, *, action_counts: dict[str, int]) -> SubsystemExecutionResult:
        runtime.history.record_event(state, "subsystem_started", {"subsystem": self.name, "step_id": step.step_id, "goal": step.goal})
        runtime.history.record_event(state, "tool_chain_started", {"step_id": step.step_id, "expected_tool": step.expected_tool, "max_attempts": runtime.config.runtime.max_tool_steps})
        attempts = 0
        helper_hops = 0
        tool_results = []
        budget_reports = []
        while attempts < runtime.config.runtime.max_tool_steps:
            attempts += 1
            decision, report = runtime._decide(state)
            budget_reports.append(report)
            scope = {}
            if decision.tool_name in {"read_text", "edit_text"}:
                scope["edit_count"] = state.edit_count
            elif decision.tool_name == "notes":
                scope["note_count"] = len(state.notes)
            action_key = stable_json_dumps(
                {
                    "action": decision.action,
                    "tool_name": decision.tool_name,
                    "tool_input": decision.tool_input,
                    "response": decision.response,
                    "scope": scope,
                }
            )
            action_counts[action_key] = action_counts.get(action_key, 0) + 1
            runtime.history.record_event(
                state,
                "tool_chain_step",
                {"step_id": step.step_id, "attempt": attempts, "decision": {"action": decision.action, "tool_name": decision.tool_name, "tool_input": decision.tool_input}},
            )
            runtime.history.record_event(
                state,
                "subsystem_progress",
                {"subsystem": self.name, "step_id": step.step_id, "progress": f"attempt={attempts}; selected={decision.tool_name}:{decision.action}"},
            )
            if decision.action != "call_tool":
                if action_counts[action_key] > runtime.config.runtime.max_repeated_action_occurrences:
                    runtime.history.record_event(
                        state,
                        "duplicate_action_detected",
                        {"action_key": action_key, "count": action_counts[action_key]},
                    )
                    break
                continue
            graph_plan = runtime.tools.plan_tool_graph(
                selected_tool=decision.tool_name,
                expected_tool=step.expected_tool or decision.tool_name,
                config=runtime.config,
            )
            runtime.history.record_event(
                state,
                "tool_graph_planned",
                {
                    "step_id": step.step_id,
                    "selected_tool": decision.tool_name,
                    "expected_tool": step.expected_tool,
                    "chain": graph_plan.chain,
                    "valid": graph_plan.valid,
                    "reason": graph_plan.reason,
                },
            )
            if action_counts[action_key] > runtime.config.runtime.max_repeated_action_occurrences:
                runtime.history.record_event(
                    state,
                    "duplicate_action_detected",
                        {"action_key": action_key, "count": action_counts[action_key]},
                )
                break
            allow_helper_chain = bool(
                (
                    state.active_strategy is None
                    and helper_hops == 0
                )
                or (
                    state.active_strategy is not None
                    and helper_hops < state.active_strategy.tool_chain_depth
                )
            )
            if step.expected_tool and decision.tool_name != step.expected_tool and allow_helper_chain and graph_plan.valid:
                helper_result = runtime._execute_tool(state, decision)
                if helper_result is not None:
                    tool_results.append(helper_result)
                    helper_hops += 1
                    if self._should_preview(helper_result.tool_name, state=state):
                        active_plan = state.active_plan
                        assert active_plan is not None
                        preview = runtime._preview_step_verification(
                            state,
                            active_plan,
                            step,
                            runtime._build_verification_artifacts(step, tool_results=tool_results, assistant_text=""),
                        )
                        runtime.history.record_event(
                            state,
                            "subsystem_progress",
                            {
                                "subsystem": self.name,
                                "step_id": step.step_id,
                                "progress": f"helper_tool={helper_result.tool_name}; preview_passed={preview.passed}; reason={preview.reason}",
                            },
                        )
                continue
            if step.expected_tool and decision.tool_name != step.expected_tool:
                runtime.history.record_event(
                    state,
                    "tool_graph_rejected",
                    {
                        "step_id": step.step_id,
                        "selected_tool": decision.tool_name,
                        "expected_tool": step.expected_tool,
                        "chain": graph_plan.chain,
                        "reason": graph_plan.reason,
                    },
                )
                break
            tool_result = runtime._execute_tool(state, decision)
            if tool_result is None:
                continue
            tool_results.append(tool_result)
            if not tool_result.completed:
                runtime.history.record_event(
                    state,
                    "subsystem_progress",
                    {
                        "subsystem": self.name,
                        "step_id": step.step_id,
                        "progress": f"background_started={tool_result.output.get('process_id', '')}",
                    },
                )
                runtime.history.record_event(state, "tool_chain_completed", {"step_id": step.step_id, "attempts": attempts, "success": True})
                runtime.history.record_event(state, "subsystem_completed", {"subsystem": self.name, "step_id": step.step_id, "success": True, "result_summary": "background_started"})
                return SubsystemExecutionResult(
                    subsystem_name=self.name,
                    success=True,
                    progress=[f"attempt={attempts}", "background_started"],
                    tool_results=tool_results,
                    budget_reports=budget_reports,
                    evaluation=None,
                    background_job_started=True,
                    background_process_id=str(tool_result.output.get("process_id", "")) or None,
                )
            if self._should_preview(decision.tool_name, state=state):
                active_plan = state.active_plan
                assert active_plan is not None
                preview = runtime._preview_step_verification(
                    state,
                    active_plan,
                    step,
                    runtime._build_verification_artifacts(step, tool_results=tool_results, assistant_text=""),
                )
                runtime.history.record_event(
                    state,
                    "subsystem_progress",
                    {
                        "subsystem": self.name,
                        "step_id": step.step_id,
                        "progress": f"attempt={attempts}; preview_passed={preview.passed}; reason={preview.reason}",
                    },
                )
                if not preview.passed and self._can_continue_refinement(step, preview) and attempts < runtime.config.runtime.max_tool_steps:
                    continue
            runtime.history.record_event(state, "tool_chain_completed", {"step_id": step.step_id, "attempts": attempts, "success": True})
            runtime.history.record_event(state, "subsystem_completed", {"subsystem": self.name, "step_id": step.step_id, "success": True, "result_summary": tool_result.tool_name})
            return SubsystemExecutionResult(
                subsystem_name=self.name,
                success=True,
                progress=[f"attempt={attempts}"],
                tool_results=tool_results,
                budget_reports=budget_reports,
                evaluation=None,
            )
        runtime.history.record_event(state, "tool_chain_completed", {"step_id": step.step_id, "attempts": attempts, "success": False})
        runtime.history.record_event(state, "subsystem_completed", {"subsystem": self.name, "step_id": step.step_id, "success": False, "result_summary": "tool_chain_failed"})
        return SubsystemExecutionResult(
            subsystem_name=self.name,
            success=False,
            progress=[f"attempts={attempts}"],
            tool_results=tool_results,
            budget_reports=budget_reports,
            evaluation=None,
        )
