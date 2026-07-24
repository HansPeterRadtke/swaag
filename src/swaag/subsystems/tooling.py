from __future__ import annotations

from swaag.subsystems.base import SubsystemExecutionResult
from swaag.types import Message, PlanStep, SessionState, ToolDecision
from swaag.utils import stable_json_dumps, utc_now_iso


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

    def _record_preview_failure_observation(self, runtime, state: SessionState, step: PlanStep, tool_name: str, preview) -> None:
        payload = {
            "step_id": step.step_id,
            "tool_name": tool_name,
            "verification_type_used": preview.verification_type_used,
            "conditions_met": list(preview.conditions_met),
            "conditions_failed": list(preview.conditions_failed),
            "evidence": preview.evidence,
            "confidence": preview.confidence,
            "reason": preview.reason,
        }
        runtime._record_message(
            state,
            Message(
                role="tool",
                name=tool_name,
                content=f"verification_preview_failed: {stable_json_dumps(payload, indent=2)}",
                created_at=utc_now_iso(),
                metadata=payload,
            ),
        )

    def run(self, runtime, state: SessionState, step: PlanStep, *, action_counts: dict[str, int]) -> SubsystemExecutionResult:
        runtime.history.record_event(state, "subsystem_started", {"subsystem": self.name, "step_id": step.step_id, "goal": step.goal})
        runtime.history.record_event(state, "tool_chain_started", {"step_id": step.step_id, "expected_tool": step.expected_tool, "max_attempts": runtime.config.runtime.max_tool_steps})
        attempts = 0
        tool_results = []
        budget_reports = []
        handoff_after_failed_preview = False
        post_preview_tool_errors = 0
        step_action_counts: dict[str, int] = {}

        def handoff_to_verification(attempt_count: int) -> SubsystemExecutionResult:
            runtime.history.record_event(
                state,
                "subsystem_progress",
                {
                    "subsystem": self.name,
                    "step_id": step.step_id,
                    "progress": f"attempt={attempt_count}; handoff_to_verification_after_preview_failure",
                },
            )
            runtime.history.record_event(
                state,
                "tool_chain_completed",
                {"step_id": step.step_id, "attempts": attempt_count, "success": True, "handoff_to_verification": True},
            )
            runtime.history.record_event(
                state,
                "subsystem_completed",
                {
                    "subsystem": self.name,
                    "step_id": step.step_id,
                    "success": True,
                    "result_summary": "handoff_to_verification_after_preview_failure",
                },
            )
            return SubsystemExecutionResult(
                subsystem_name=self.name,
                success=True,
                progress=[f"attempt={attempt_count}", "handoff_to_verification_after_preview_failure"],
                tool_results=tool_results,
                budget_reports=budget_reports,
                evaluation=None,
                same_step_retry_allowed=False,
            )

        while attempts < runtime.config.runtime.max_tool_steps:
            attempts += 1
            if step.expected_tool:
                tool_input, report = runtime._decide_tool_input_with_report(state, step.expected_tool)
                decision = ToolDecision(action="call_tool", response="", tool_name=step.expected_tool, tool_input=tool_input)
            else:
                decision, report = runtime._decide(state)
            budget_reports.append(report)
            scope = {"edit_count": state.edit_count, "note_count": len(state.notes)}
            step_scoped_duplicate_check = False
            if step.expected_tool:
                if decision.action != "call_tool":
                    step_scoped_duplicate_check = True
                elif decision.tool_name != step.expected_tool:
                    step_scoped_duplicate_check = True
            action_key_payload = {
                "action": decision.action,
                "tool_name": decision.tool_name,
                "tool_input": decision.tool_input,
                "scope": scope,
            }
            if step.expected_tool or step_scoped_duplicate_check:
                action_key_payload["step_id"] = step.step_id
                action_key_payload["expected_tool"] = step.expected_tool or ""
            if not (step.expected_tool and decision.action != "call_tool"):
                action_key_payload["response"] = decision.response
            action_key = stable_json_dumps(action_key_payload)
            action_counts[action_key] = action_counts.get(action_key, 0) + 1
            step_action_key_payload = {
                "action": decision.action,
                "tool_name": decision.tool_name,
                "tool_input": decision.tool_input,
                "step_id": step.step_id,
                "expected_tool": step.expected_tool or "",
            }
            if not (step.expected_tool and decision.action != "call_tool"):
                step_action_key_payload["response"] = decision.response
            step_action_key = stable_json_dumps(step_action_key_payload)
            step_action_counts[step_action_key] = step_action_counts.get(step_action_key, 0) + 1
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
                if step.expected_tool:
                    runtime.history.record_event(
                        state,
                        "tool_mismatch_rejected",
                        {
                            "step_id": step.step_id,
                            "selected_tool": decision.tool_name,
                            "expected_tool": step.expected_tool,
                            "reason": "model_action_did_not_call_required_plan_tool",
                        },
                    )
                    runtime.history.record_event(
                        state,
                        "subsystem_progress",
                        {
                            "subsystem": self.name,
                            "step_id": step.step_id,
                            "progress": f"attempt={attempts}; rejected_action={decision.action}; expected_tool={step.expected_tool}",
                        },
                    )
                if action_counts[action_key] > runtime.config.runtime.max_repeated_action_occurrences:
                    runtime.history.record_event(
                        state,
                        "duplicate_action_detected",
                        {"action_key": action_key, "count": action_counts[action_key]},
                    )
                    if handoff_after_failed_preview and tool_results:
                        return handoff_to_verification(attempts)
                    break
                continue
            if action_counts[action_key] > runtime.config.runtime.max_repeated_action_occurrences:
                runtime.history.record_event(
                    state,
                    "duplicate_action_detected",
                        {"action_key": action_key, "count": action_counts[action_key]},
                )
                if handoff_after_failed_preview and tool_results:
                    return handoff_to_verification(attempts)
                break
            if step_action_counts[step_action_key] > runtime.config.runtime.max_repeated_action_occurrences:
                runtime.history.record_event(
                    state,
                    "duplicate_action_detected",
                    {
                        "action_key": step_action_key,
                        "count": step_action_counts[step_action_key],
                        "scope": "current_step_exact_action",
                    },
                )
                if tool_results:
                    return handoff_to_verification(attempts)
                break
            if step.expected_tool and decision.tool_name != step.expected_tool:
                runtime.history.record_event(
                    state,
                    "tool_mismatch_rejected",
                    {
                        "step_id": step.step_id,
                        "selected_tool": decision.tool_name,
                        "expected_tool": step.expected_tool,
                        "reason": "model_selected_tool_did_not_match_plan_step",
                    },
                )
                runtime.history.record_event(
                    state,
                    "subsystem_progress",
                    {
                        "subsystem": self.name,
                        "step_id": step.step_id,
                        "progress": f"attempt={attempts}; rejected_tool={decision.tool_name}; expected_tool={step.expected_tool}",
                    },
                )
                continue
            tool_result = runtime._execute_tool(state, decision)
            if tool_result is None:
                runtime.history.record_event(
                    state,
                    "subsystem_progress",
                    {
                        "subsystem": self.name,
                        "step_id": step.step_id,
                        "progress": f"attempt={attempts}; tool_error={decision.tool_name}",
                    },
                )
                if handoff_after_failed_preview and tool_results:
                    post_preview_tool_errors += 1
                if handoff_after_failed_preview and tool_results and post_preview_tool_errors >= 2:
                    return handoff_to_verification(attempts)
                if attempts < runtime.config.runtime.max_tool_steps:
                    continue
                break
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
                    handoff_after_failed_preview = True
                    self._record_preview_failure_observation(runtime, state, step, decision.tool_name, preview)
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
