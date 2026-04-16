from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import re
import shlex
import shutil
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterable

import requests

from swaag.compression import decide_history_compression, summary_message_payload
from swaag.budgeting import compute_call_budget, structured_output_token_floor
from swaag.config import AgentConfig, load_config
from swaag.context_builder import ContextBundle, build_context
from swaag.decision import (
    DecisionValidationError,
    decision_from_payload,
)
from swaag.environment.environment import AgentEnvironment, BackgroundProcessUpdate
from swaag.evaluator import evaluate_verification
from swaag.expander import ExpansionValidationError, expanded_task_from_payload
from swaag.failure import (
    FailureClassification,
    FailureValidationError,
    classify_failure_from_payload,
)
from swaag.prompt_analyzer import (
    PromptAnalysisValidationError,
    analysis_from_payload,
)
from swaag.history import HistoryInvariantError, HistoryStore
from swaag.grammar import (
    active_session_control_contract,
    action_selection_contract,
    failure_classification_contract,
    generation_decomposition_contract,
    overflow_recovery_contract,
    plan_contract,
    plain_text_contract,
    prompt_analysis_contract,
    subagent_selection_contract,
    strategy_selection_contract,
    summary_contract,
    task_decision_contract,
    task_expansion_contract,
    tool_decision_contract,
    tool_input_contract,
    verification_contract,
    yes_no_contract,
)
from swaag.memory_semantic import extract_from_event
from swaag.model import LlamaCppClient, ModelClientError
from swaag.orchestrator import action_from_payload, select_action
from swaag.planner import (
    PlanValidationError,
    create_shell_recovery_plan,
    create_direct_response_plan,
    create_direct_tool_plan,
    mark_step_completed,
    mark_step_failed,
    mark_step_in_progress,
    ready_steps,
    next_executable_step,
    plan_as_payload,
    plan_from_payload,
)
from swaag.prompts import PromptBuilder
from swaag.project_state import build_project_state
from swaag.retrieval.embeddings import SemanticBackendProtocolError
from swaag.strategy import (
    StrategyValidationError,
    adapt_strategy,
    reconcile_strategy_to_plan,
    strategy_from_payload,
    validate_plan_against_strategy,
)
from swaag.subagents import SubagentManager
from swaag.subsystems import FileSubsystem, PlanningSubsystem, ReasoningSubsystem, SubsystemExecutionResult, ToolSubsystem
from swaag.tokens import ConservativeEstimator, CountResult, ExactTokenCounter, build_budget
from swaag.tools.registry import ToolRegistry
from swaag.types import (
    BudgetReport,
    CompletionResult,
    ContractSpec,
    DeferredTask,
    DecisionOutcome,
    ExpandedTask,
    Message,
    Plan,
    PlanStep,
    PromptAnalysis,
    PromptAssembly,
    PromptComponent,
    SessionState,
    SubagentSelectionDecision,
    ToolDecision,
    ToolExecutionResult,
    ToolGeneratedEvent,
)
from swaag.utils import new_id, sha256_text, stable_json_dumps, to_jsonable, utc_now_iso
from swaag.working_memory import build_working_memory
from swaag.verification import VerificationArtifacts, VerificationEngine, VerificationError, VerificationOutcome

_PATH_LIKE_RE = re.compile(
    r"""
    (?:
        (?<!\w)
        (?:
            \.{1,2}/[^\s,'"`;]+ |
            /[^\s,'"`;]+ |
            ~/[^\s,'"`;]+
        )
    )
    """,
    re.VERBOSE,
)
_BARE_FILE_RE = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_])")
_URL_RE = re.compile(r"https?://[^\s,'\"`]+")


def _mask_path_like_text(text: str) -> str:
    return _PATH_LIKE_RE.sub(" <PATH> ", text)


def _path_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (_PATH_LIKE_RE, _BARE_FILE_RE):
        for match in pattern.findall(text):
            candidate = match.strip().rstrip(".,)")
            if not candidate:
                continue
            if any(existing == candidate for existing in candidates):
                continue
            if any(
                len(existing) > len(candidate)
                and (
                    existing.endswith(candidate)
                    or existing.endswith("/" + candidate)
                )
                for existing in candidates
            ):
                continue
            candidates.append(candidate)
    return candidates


def _url_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _URL_RE.findall(text):
        candidate = match.strip().rstrip(".,)")
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _truncate_clause(text: str) -> str:
    lowered = text.lower()
    cut_markers = (
        " and ",
        ". ",
        "\n",
        " then ",
        " reply ",
        " return ",
        " answer ",
    )
    end = len(text)
    for marker in cut_markers:
        index = lowered.find(marker)
        if index != -1:
            end = min(end, index)
    return text[:end].strip(" \t\n\r.,)")


class BudgetExceededError(RuntimeError):
    def __init__(self, message: str, report: BudgetReport | None = None):
        super().__init__(message)
        self.report = report


class FatalSemanticEngineError(RuntimeError):
    """Raised when a supposedly hard-constrained semantic call fails impossibly."""


@dataclass(slots=True)
class TurnResult:
    session_id: str
    assistant_text: str
    tool_results: list[ToolExecutionResult]
    budget_reports: list[BudgetReport]


@dataclass(slots=True)
class ToolRunResult:
    session_id: str
    tool_result: ToolExecutionResult | None


@dataclass(slots=True)
class TurnPreparation:
    analysis: PromptAnalysis
    decision: DecisionOutcome
    effective_goal: str
    expanded_task: ExpandedTask | None = None
    clarification_request: str | None = None
    required_named_tools: tuple[str, ...] = ()


@dataclass(slots=True)
class PreparedCall:
    assembly: PromptAssembly
    report: BudgetReport
    prompt_mode: str
    contract: ContractSpec


@dataclass(slots=True)
class BackgroundCycleResult:
    progress_made: bool = False
    completed_steps: int = 0
    failed_steps: int = 0
    no_progress_resolved: bool = False
    last_verification: VerificationOutcome | None = None
    last_failure: FailureClassification | None = None
    replan_reason: str | None = None


@dataclass(slots=True)
class ControlProcessingResult:
    stop_requested: bool = False
    replacement_goal: str | None = None
    replan_requested: bool = False
    assistant_messages: list[str] = field(default_factory=list)


class AgentRuntime:
    def __init__(
        self,
        config: AgentConfig,
        *,
        model_client: LlamaCppClient | None = None,
        tool_registry: ToolRegistry | None = None,
        history_store: HistoryStore | None = None,
        token_counter: ExactTokenCounter | ConservativeEstimator | None = None,
    ):
        self.config = config
        self.client = model_client or LlamaCppClient(config)
        self.tools = tool_registry or ToolRegistry()
        self.history = history_store or HistoryStore(config.sessions.root, write_projections=config.sessions.write_projections)
        self.prompts = PromptBuilder(config)
        self._token_counter = token_counter
        self._verification = VerificationEngine(
            semantic_backend_mode=self.config.retrieval.backend,
            semantic_base_url=self.config.model.base_url,
            semantic_seed=self.config.model.seed,
            semantic_connect_timeout_seconds=self.config.model.connect_timeout_seconds,
            semantic_read_timeout_seconds=self.config.model.verification_timeout_seconds,
        )
        self._planning_subsystem = PlanningSubsystem()
        self._reasoning_subsystem = ReasoningSubsystem()
        self._tool_subsystem = ToolSubsystem()
        self._file_subsystem = FileSubsystem()
        self._subagents = SubagentManager(
            backend_mode=self.config.retrieval.backend,
            base_url=self.config.model.base_url,
            seed=self.config.model.seed,
            connect_timeout_seconds=self.config.model.connect_timeout_seconds,
            read_timeout_seconds=self.config.model.simple_timeout_seconds,
        )
        self._sleep = time.sleep
        self._max_model_unavailable_attempts: int | None = None

    @classmethod
    def from_config_paths(cls, config_paths: list[str] | None = None) -> AgentRuntime:
        return cls(load_config(config_paths))

    def _get_budget_counter(self, state: SessionState | None):
        if self._token_counter is not None:
            return self._token_counter
        if state is None:
            return ConservativeEstimator()
        return _HistoryAwareTokenCounter(self, state)

    def _get_selection_counter(self):
        if self._token_counter is not None:
            return self._token_counter
        return _NonRecordingTokenCounter(self)

    def create_or_load_session(self, session_id: str | None = None) -> SessionState:
        state = self.history.create_or_load(
            config_fingerprint=self.config.config_fingerprint(),
            model_base_url=self.config.model.base_url,
            session_id=session_id,
        )
        self._ensure_environment_initialized(state)
        return state

    def create_or_load_user_session(self, session_ref: str | None = None) -> SessionState:
        state = self.history.create_or_load_user_session(
            config_fingerprint=self.config.config_fingerprint(),
            model_base_url=self.config.model.base_url,
            session_ref=session_ref,
            prefer_latest=True,
        )
        self._ensure_environment_initialized(state)
        return state

    def resolve_session_ref(self, session_ref: str | None, *, latest_if_none: bool = False) -> str | None:
        return self.history.resolve_session_ref(session_ref, latest_if_none=latest_if_none)

    def rebuild_from_history(self, session_id: str) -> SessionState:
        state = self.history.rebuild_from_history(session_id, write_projections=False)
        self.history.record_event(state, "state_rebuilt", {"session_id": session_id, "event_count": state.event_count})
        return state

    def _ensure_environment_initialized(self, state: SessionState) -> None:
        environment = AgentEnvironment(self.config, state)
        for event in environment.initialize_events():
            self.history.record_event(state, event.event_type, event.payload, metadata=event.metadata)

    def run_turn(self, user_text: str, *, session_id: str | None = None) -> TurnResult:
        state = self.create_or_load_session(session_id)
        return self.run_turn_in_session(state, user_text)

    def execute_tool_once(self, tool_name: str, raw_input: dict[str, Any], *, session_id: str | None = None) -> ToolRunResult:
        state = self.create_or_load_session(session_id)
        self._ensure_environment_initialized(state)
        plan = create_direct_tool_plan(f"Execute tool {tool_name} safely", tool_name)
        event_type = "plan_updated" if state.active_plan is not None else "plan_created"
        if event_type == "plan_created":
            plan_event = self.history.record_event(state, event_type, {"goal": plan.goal, "plan": plan_as_payload(plan)})
        else:
            plan_event = self.history.record_event(state, event_type, {"plan": plan_as_payload(plan), "reason": "direct_tool_execution"})
        self._extract_and_store_memory(state, plan_event)
        self._refresh_working_memory(state, reason="direct_tool_execution")
        decision = ToolDecision(action="call_tool", response="", tool_name=tool_name, tool_input=raw_input)
        current_plan = state.active_plan or plan
        step = next_executable_step(current_plan)
        if step is None:
            raise RuntimeError("Direct tool plan has no executable step")
        current_plan = self._start_step(state, current_plan, step)
        result = self._execute_tool(state, decision)
        if result is None:
            self._fail_step(state, current_plan, step, f"Tool {tool_name} failed", "ToolExecutionError")
        elif not result.completed:
            process_id = str(result.output.get("process_id", "")).strip()
            if process_id:
                self._bind_background_process_to_step(
                    state,
                    step=step,
                    process_id=process_id,
                    tool_name=result.tool_name,
                )
        else:
            self._complete_step(state, current_plan, step, outcome=tool_name)
        self._refresh_working_memory(state, reason=f"tool:{tool_name}")
        self._check_consistency(state)
        return ToolRunResult(session_id=state.session_id, tool_result=result)

    def run_turn_in_session(self, state: SessionState, user_text: str) -> TurnResult:
        run_id = f"{state.session_id}:{new_id('run')}"
        self.history.set_active_run(state.session_id, run_id=run_id, user_text=user_text)
        try:
            return self._run_turn_in_session_impl(state, user_text)
        finally:
            self.history.clear_active_run(state.session_id, run_id=run_id)

    def _run_turn_in_session_impl(self, state: SessionState, user_text: str) -> TurnResult:
        self._ensure_environment_initialized(state)
        if not user_text.strip():
            raise ValueError("user_text must not be empty")

        user_text = user_text.strip()
        self.history.ensure_human_readable_name(state, user_text)
        self._record_message(state, Message(role="user", content=user_text, created_at=utc_now_iso()))
        self.history.record_event(
            state,
            "turn_started",
            {"turn_index": state.turn_count + 1, "user_text": user_text},
        )
        self._maybe_compact_history(state)
        turn_prep = self._prepare_turn_context(state, user_text)
        effective_goal = turn_prep.effective_goal
        initial_control = self._process_pending_control_messages(state, effective_goal=effective_goal)
        if initial_control.replacement_goal:
            turn_prep = self._prepare_turn_context(state, initial_control.replacement_goal)
            effective_goal = turn_prep.effective_goal
        if initial_control.stop_requested:
            response_text = initial_control.assistant_messages[-1] if initial_control.assistant_messages else "stopped by user request"
            self.history.record_event(
                state,
                "reasoning_started",
                {"goal": effective_goal, "max_steps": 0},
            )
            self.history.record_event(
                state,
                "reasoning_completed",
                {
                    "goal": effective_goal,
                    "status": "stopped",
                    "completed_steps": 0,
                    "failed_steps": 0,
                    "reason": "user_control_stop",
                },
            )
            return self._finish_turn(state, response_text, [], [])
        plan_ready = True
        if turn_prep.clarification_request is not None:
            self._refresh_working_memory(state, reason="clarification_requested")
            self.history.record_event(
                state,
                "reasoning_started",
                {"goal": effective_goal, "max_steps": 0},
            )
            self.history.record_event(
                state,
                "reasoning_completed",
                {
                    "goal": effective_goal,
                    "status": "clarification_requested",
                    "completed_steps": 0,
                    "failed_steps": 0,
                    "reason": "prompt_incomplete",
                },
            )
            return self._finish_turn(state, turn_prep.clarification_request, [], [])
        if turn_prep.decision.direct_response or turn_prep.decision.execution_mode == "direct_response":
            self._install_direct_response_plan(state, effective_goal)
        required_tools = list(turn_prep.required_named_tools)
        if (
            not turn_prep.decision.direct_response
            and turn_prep.decision.execution_mode == "single_tool"
            and turn_prep.decision.preferred_tool_name in self.tools.tool_names(self.config)
        ):
            self._install_direct_tool_plan(
                state,
                effective_goal,
                turn_prep.decision.preferred_tool_name,
                reason="semantic_single_tool_execution",
            )
        else:
            try:
                self._ensure_plan(state, effective_goal, required_tools=required_tools)
            except FatalSemanticEngineError:
                self.history.record_event(
                    state,
                    "error",
                    {
                        "operation": "plan",
                        "error": "fatal_structured_semantic_failure",
                        "error_type": "FatalSemanticEngineError",
                    },
                )
                self._record_reasoning_completed(
                    state,
                    goal=effective_goal,
                    status="fatal_system_error",
                    completed_steps=0,
                    failed_steps=0,
                    reason="plan_generation_failed",
                )
                raise
            except Exception as exc:
                plan_ready = False
                self.history.record_event(
                    state,
                    "error",
                    {"operation": "plan", "error": str(exc), "error_type": exc.__class__.__name__},
                )
        self._refresh_working_memory(state, reason="turn_started")
        self._check_consistency(state)
        self.history.record_event(
            state,
            "reasoning_started",
            {"goal": effective_goal, "max_steps": min(self.config.runtime.max_total_actions, self.config.runtime.max_reasoning_steps)},
        )

        tool_results: list[ToolExecutionResult] = []
        background_tool_indexes: dict[str, int] = {}
        budget_reports: list[BudgetReport] = []
        action_counts: dict[str, int] = {}
        step_attempts: dict[str, int] = {}
        completed_steps = 0
        failed_steps = 0
        replans_used = 0
        reasoning_status = "completed"
        reasoning_reason = "final_response"
        answer_text = ""
        last_verification: VerificationOutcome | None = None
        last_failure: FailureClassification | None = None
        current_running_step_id: str | None = None
        waiting_on_processes: set[str] | None = (
            set(state.environment.waiting_process_ids)
            if state.environment.waiting and state.environment.waiting_process_ids
            else None
        )
        execution_iterations = 0
        turn_tool_calls = 0
        no_progress_failures = 0
        max_loop_iterations = min(
            self.config.runtime.max_total_actions,
            self.config.runtime.max_reasoning_steps + self.config.planner.max_replans + self.config.runtime.max_tool_steps,
        )

        for _ in range(max_loop_iterations):
            if not plan_ready:
                reasoning_status = "fallback"
                reasoning_reason = "plan_error"
                break
            background_progress = self._poll_background_processes(
                state,
                tool_results=tool_results,
                background_tool_indexes=background_tool_indexes,
            )
            running_background_ids = self._running_background_process_ids(state)
            if waiting_on_processes is not None and (
                background_progress.progress_made
                or set(running_background_ids) != waiting_on_processes
                or not running_background_ids
            ):
                self.history.record_event(
                    state,
                    "wait_resumed",
                    {
                        "reason": "background_progress",
                        "process_ids": sorted(waiting_on_processes),
                    },
                )
                waiting_on_processes = None
            if background_progress.progress_made:
                completed_steps += background_progress.completed_steps
                failed_steps += background_progress.failed_steps
                last_verification = background_progress.last_verification or last_verification
                last_failure = background_progress.last_failure or last_failure
                if background_progress.no_progress_resolved:
                    no_progress_failures = 0
                    last_failure = None
                elif background_progress.failed_steps:
                    no_progress_failures += background_progress.failed_steps
                current_running_step_id = None
                if background_progress.replan_reason:
                    if replans_used >= self.config.planner.max_replans:
                        reasoning_status = "fallback"
                        reasoning_reason = "replan_limit_reached"
                        break
                    replans_used += 1
                    self.history.record_event(
                        state,
                        "replan_triggered",
                        {
                            "step_id": "background",
                            "reason": background_progress.replan_reason,
                            "replan_count": replans_used,
                        },
                    )
                    self._ensure_plan(
                        state,
                        effective_goal,
                        replan_reason=background_progress.replan_reason,
                        force_replan=True,
                    )
                    last_verification = None
                    last_failure = None
                    continue
            control_result = self._process_pending_control_messages(state, effective_goal=effective_goal)
            if control_result.stop_requested:
                answer_text = control_result.assistant_messages[-1] if control_result.assistant_messages else "stopped by user request"
                reasoning_status = "stopped"
                reasoning_reason = "user_control_stop"
                break
            if control_result.replacement_goal:
                turn_prep = self._prepare_turn_context(state, control_result.replacement_goal)
                effective_goal = turn_prep.effective_goal
                current_running_step_id = None
                last_verification = None
                last_failure = None
                required_tools = list(turn_prep.required_named_tools)
                if turn_prep.decision.direct_response or turn_prep.decision.execution_mode == "direct_response":
                    self._install_direct_response_plan(state, effective_goal)
                elif (
                    not turn_prep.decision.direct_response
                    and turn_prep.decision.execution_mode == "single_tool"
                    and turn_prep.decision.preferred_tool_name in self.tools.tool_names(self.config)
                ):
                    self._install_direct_tool_plan(
                        state,
                        effective_goal,
                        turn_prep.decision.preferred_tool_name,
                        reason="control_replacement_single_tool",
                    )
                else:
                    self._ensure_plan(
                        state,
                        effective_goal,
                        replan_reason="user_requested_replacement",
                        force_replan=True,
                        required_tools=required_tools,
                    )
                continue
            if control_result.replan_requested:
                current_running_step_id = None
                last_verification = None
                last_failure = None
                self._ensure_plan(
                    state,
                    effective_goal,
                    replan_reason="control_context_update",
                    force_replan=True,
                )
                continue
            try:
                plan = self._ensure_plan(state, effective_goal)
            except FatalSemanticEngineError:
                self.history.record_event(
                    state,
                    "error",
                    {
                        "operation": "plan",
                        "error": "fatal_structured_semantic_failure",
                        "error_type": "FatalSemanticEngineError",
                    },
                )
                self._record_reasoning_completed(
                    state,
                    goal=effective_goal,
                    status="fatal_system_error",
                    completed_steps=completed_steps,
                    failed_steps=failed_steps,
                    reason="plan_generation_failed",
                )
                raise
            except Exception as exc:
                self.history.record_event(
                    state,
                    "error",
                    {"operation": "plan", "error": str(exc), "error_type": exc.__class__.__name__},
                )
                reasoning_status = "fallback"
                reasoning_reason = "plan_error"
                break
            self._check_consistency(state)

            active_strategy = state.active_strategy
            if active_strategy is None:
                raise HistoryInvariantError("Active strategy is missing before the reasoning loop")
            current_running_step = None
            if current_running_step_id and any(item.step_id == current_running_step_id and item.status == "running" for item in plan.steps):
                current_running_step = next(item for item in plan.steps if item.step_id == current_running_step_id)
            candidate_step = current_running_step or next_executable_step(plan)
            repeated_action_count = step_attempts.get(candidate_step.step_id, 0) if candidate_step is not None else 0
            orchestration = select_action(
                state=state,
                plan=plan,
                strategy=active_strategy,
                verification=last_verification,
                failure=last_failure,
                repeated_action_count=repeated_action_count,
                iteration=execution_iterations,
                max_iterations=min(self.config.runtime.max_reasoning_steps, self.config.runtime.max_total_actions),
                turn_tool_calls=turn_tool_calls,
                tool_call_budget=self.config.runtime.tool_call_budget,
                no_progress_failures=no_progress_failures,
                no_progress_failure_limit=self.config.runtime.no_progress_failure_limit,
                current_step=current_running_step,
                running_background_jobs=len(running_background_ids),
            )
            if orchestration.requires_llm_decision:
                selected_action = self._select_action_frontend(state, orchestration)
                orchestration.action = selected_action
            self._record_action_selection(state, orchestration)

            if orchestration.action == "stop":
                reasoning_status = "completed" if orchestration.stop_reason == "goal_satisfied" else "stopped"
                reasoning_reason = orchestration.stop_reason or "stop_condition"
                break

            if orchestration.action == "replan":
                if replans_used >= self.config.planner.max_replans:
                    reasoning_status = "fallback"
                    reasoning_reason = "replan_limit_reached"
                    break
                if current_running_step is not None and last_failure is not None and self._step_running(plan, current_running_step.step_id):
                    failed_steps += 1
                    self._fail_step(state, plan, current_running_step, last_failure.reason, last_failure.kind)
                    current_running_step_id = None
                if last_failure is not None:
                    observed_failures = max(failed_steps, state.metrics.verification_failures, state.metrics.steps_failed)
                    self._check_drift(state, failed_steps=observed_failures, completed_steps=completed_steps)
                replans_used += 1
                self.history.record_event(
                    state,
                    "replan_triggered",
                    {
                        "step_id": orchestration.step.step_id if orchestration.step is not None else "none",
                        "reason": last_failure.reason if last_failure is not None else "orchestrator_selected_replan",
                        "replan_count": replans_used,
                    },
                )
                plan = self._ensure_plan(state, effective_goal, replan_reason=last_failure.reason if last_failure is not None else "orchestrator_selected_replan", replan_attempt=replans_used, force_replan=True)
                current_running_step_id = None
                last_verification = None
                last_failure = None
                continue

            if orchestration.action == "wait":
                if waiting_on_processes is None:
                    self.history.record_event(
                        state,
                        "wait_entered",
                        {
                            "reason": "background_jobs_running",
                            "process_ids": list(running_background_ids),
                        },
                    )
                    waiting_on_processes = set(running_background_ids)
                if self.config.runtime.background_poll_seconds > 0:
                    self._sleep(self.config.runtime.background_poll_seconds)
                continue

            step = orchestration.step
            if step is None:
                reasoning_status = "stopped"
                reasoning_reason = "no_executable_step"
                break

            try:
                execution_iterations += 1
                if not self._step_running(plan, step.step_id):
                    plan = self._start_step(state, plan, step)
                    step = next(item for item in plan.steps if item.step_id == step.step_id)
                current_running_step_id = step.step_id
                step_attempts[step.step_id] = step_attempts.get(step.step_id, 0) + 1
                if orchestration.action == "retry_step":
                    self.history.record_event(
                        state,
                        "retry_triggered",
                        {
                            "step_id": step.step_id,
                            "reason": last_failure.reason if last_failure is not None else "verification_retry",
                            "attempt": step_attempts[step.step_id],
                            "failure_kind": last_failure.kind if last_failure is not None else "verification_failure",
                        },
                    )
                subsystem_result = self._run_step_subsystem(state, step, action_counts=action_counts)
                budget_reports.extend(subsystem_result.budget_reports)
                tool_results_start = len(tool_results)
                tool_results.extend(subsystem_result.tool_results)
                turn_tool_calls += len(subsystem_result.tool_results)
                if subsystem_result.background_job_started:
                    if subsystem_result.background_process_id is None:
                        raise HistoryInvariantError(f"Background step {step.step_id} did not report a process id")
                    if subsystem_result.tool_results:
                        background_tool_indexes[subsystem_result.background_process_id] = tool_results_start
                    self._bind_background_process_to_step(
                        state,
                        step=step,
                        process_id=subsystem_result.background_process_id,
                        tool_name=subsystem_result.tool_results[-1].tool_name if subsystem_result.tool_results else (step.expected_tool or ""),
                    )
                    no_progress_failures = 0
                    current_running_step_id = None
                    last_verification = None
                    last_failure = None
                    continue
                verification = self._verify_step(
                    state,
                    plan,
                    step,
                    self._build_verification_artifacts(
                        step,
                        assistant_text=subsystem_result.assistant_text,
                        tool_results=list(subsystem_result.tool_results),
                        runtime_artifacts={"subsystem": subsystem_result.subsystem_name},
                    ),
                )
                if verification.passed and verification.confidence < self.config.runtime.verification_confidence_threshold:
                    verification = VerificationOutcome(
                        verification_passed=False,
                        verification_type_used=verification.verification_type_used,
                        conditions_met=list(verification.conditions_met),
                        conditions_failed=[*verification.conditions_failed, "confidence_below_threshold"],
                        evidence=dict(verification.evidence),
                        confidence=verification.confidence,
                        reason=f"{verification.reason};confidence_below_threshold",
                        requires_retry=True,
                        requires_replan=False,
                    )
                review_passed, review_reason, review_evidence = self._review_verification_result(
                    state,
                    step,
                    verification=verification,
                    subsystem_result=subsystem_result,
                )
                if not review_passed:
                    verification = VerificationOutcome(
                        verification_passed=False,
                        verification_type_used=verification.verification_type_used,
                        conditions_met=list(verification.conditions_met),
                        conditions_failed=[*verification.conditions_failed, "review_failed"],
                        evidence={**dict(verification.evidence), "review": review_evidence},
                        confidence=verification.confidence,
                        reason=f"{verification.reason};{review_reason}",
                        requires_retry=True,
                        requires_replan=False,
                    )
                evaluation = evaluate_verification(step, verification)
                if verification.verification_type_used != "llm_fallback" and not verification.passed and evaluation.passed:
                    raise HistoryInvariantError(
                        f"Evaluator attempted to override deterministic verification failure for step {step.step_id}"
                    )
                failure = None if evaluation.passed else self._classify_failure_frontend(
                    state,
                    step=step,
                    reason=f"verification:{evaluation.reason}",
                )
                last_verification = verification
                last_failure = failure
                if evaluation.passed:
                    no_progress_failures = 0
                    outcome = subsystem_result.assistant_text[:120] if subsystem_result.assistant_text else (
                        subsystem_result.tool_results[-1].tool_name if subsystem_result.tool_results else subsystem_result.subsystem_name
                    )
                    self._complete_step(state, plan, step, outcome=outcome)
                    self._refresh_project_state(state, reason=f"step_completed:{step.step_id}")
                    self._check_consistency(state)
                    self._check_drift(state, failed_steps=failed_steps, completed_steps=completed_steps + 1)
                    current_running_step_id = None
                    completed_steps += 1
                    if step.kind == "respond":
                        answer_text = subsystem_result.assistant_text
                        reasoning_reason = "answered"
                        break
                    continue

                updated_strategy = adapt_strategy(active_strategy, failure=failure, metrics=state.metrics, verification_failed=True)
                self._set_strategy(state, updated_strategy, reason=updated_strategy.reason)
                no_progress_failures += 1
                if (
                    evaluation.requires_retry
                    and (failure is None or not failure.requires_replan)
                    and step_attempts[step.step_id] <= updated_strategy.retry_same_action_limit + 1
                ):
                    continue

                failed_steps += 1
                self._fail_step(state, plan, step, evaluation.reason, failure.kind if failure is not None else "VerificationError")
                current_running_step_id = None
                self._check_drift(state, failed_steps=failed_steps, completed_steps=completed_steps)
                if replans_used < self.config.planner.max_replans:
                    replans_used += 1
                    self.history.record_event(
                        state,
                        "replan_triggered",
                        {"step_id": step.step_id, "reason": evaluation.reason, "replan_count": replans_used},
                    )
                    self._ensure_plan(state, effective_goal, replan_reason=f"Step {step.step_id} failed verification: {evaluation.reason}", replan_attempt=replans_used, force_replan=True)
                    last_verification = None
                    last_failure = None
                    continue
                if no_progress_failures >= self.config.runtime.no_progress_failure_limit:
                    reasoning_status = "stopped"
                    reasoning_reason = "no_progress_possible"
                else:
                    reasoning_status = "fallback"
                    reasoning_reason = "step_verification_failed"
                break
            except BudgetExceededError as exc:
                last_failure = self._classify_failure_frontend(state, step=step, error=exc, reason="budget exceeded")
                updated_strategy = adapt_strategy(active_strategy, failure=last_failure, metrics=state.metrics, verification_failed=True)
                self._set_strategy(state, updated_strategy, reason=updated_strategy.reason)
                no_progress_failures += 1
                failed_steps += 1
                self._fail_step(state, plan, step, "Budget exceeded while executing step", "BudgetExceededError")
                current_running_step_id = None
                reasoning_status = "budget_exhausted"
                reasoning_reason = "step_budget_exceeded"
                if replans_used < self.config.planner.max_replans:
                    replans_used += 1
                    self.history.record_event(
                        state,
                        "replan_triggered",
                        {"step_id": step.step_id, "reason": "budget_exceeded", "replan_count": replans_used},
                    )
                    self._ensure_plan(state, effective_goal, replan_reason="Budget exceeded while executing the previous step.", replan_attempt=replans_used, force_replan=True)
                    last_verification = None
                    last_failure = None
                    continue
                if no_progress_failures >= self.config.runtime.no_progress_failure_limit:
                    reasoning_status = "stopped"
                    reasoning_reason = "no_progress_possible"
                break
            except HistoryInvariantError:
                raise
            except FatalSemanticEngineError:
                self.history.record_event(
                    state,
                    "error",
                    {
                        "operation": "step_execution",
                        "error": "fatal_structured_semantic_failure",
                        "error_type": "FatalSemanticEngineError",
                        "step_id": step.step_id,
                    },
                )
                self._record_reasoning_completed(
                    state,
                    goal=effective_goal,
                    status="fatal_system_error",
                    completed_steps=completed_steps,
                    failed_steps=failed_steps,
                    reason=f"fatal_step_error:{step.step_id}",
                )
                raise
            except Exception as exc:
                self.history.record_event(
                    state,
                    "error",
                    {"operation": "step_execution", "error": str(exc), "error_type": exc.__class__.__name__},
                )
                last_failure = self._classify_failure_frontend(state, step=step, error=exc, reason=str(exc))
                updated_strategy = adapt_strategy(active_strategy, failure=last_failure, metrics=state.metrics, verification_failed=False)
                self._set_strategy(state, updated_strategy, reason=updated_strategy.reason)
                no_progress_failures += 1
                if last_failure.retryable and step_attempts[step.step_id] <= updated_strategy.retry_same_action_limit + 1:
                    self.history.record_event(
                        state,
                        "retry_triggered",
                        {
                            "step_id": step.step_id,
                            "reason": str(exc),
                            "attempt": step_attempts[step.step_id],
                            "failure_kind": last_failure.kind,
                        },
                    )
                    continue
                failed_steps += 1
                self._fail_step(state, plan, step, str(exc), exc.__class__.__name__)
                current_running_step_id = None
                self._check_drift(state, failed_steps=failed_steps, completed_steps=completed_steps)
                if replans_used < self.config.planner.max_replans:
                    replans_used += 1
                    self.history.record_event(
                        state,
                        "replan_triggered",
                        {"step_id": step.step_id, "reason": str(exc), "replan_count": replans_used},
                    )
                    self._ensure_plan(state, effective_goal, replan_reason=f"Step {step.step_id} failed: {exc}", replan_attempt=replans_used, force_replan=True)
                    last_verification = None
                    last_failure = None
                    continue
                if no_progress_failures >= self.config.runtime.no_progress_failure_limit:
                    reasoning_status = "stopped"
                    reasoning_reason = "no_progress_possible"
                else:
                    reasoning_status = "fallback"
                    reasoning_reason = "step_failed"
                break
        else:
            reasoning_status = "stopped"
            reasoning_reason = "max_iterations_reached"

        reasoning_recorded = False
        if not answer_text and reasoning_status != "completed":
            self._record_reasoning_completed(
                state,
                goal=effective_goal,
                status=reasoning_status,
                completed_steps=completed_steps,
                failed_steps=failed_steps,
                reason=reasoning_reason,
            )
            reasoning_recorded = True
        if not answer_text:
            answer_text, answer_report = self._answer(state)
            budget_reports.append(answer_report)
            if reasoning_status == "completed":
                answer_completed, answer_failed = self._finalize_answer_step(state, answer_text)
                if answer_completed:
                    completed_steps += 1
                    reasoning_reason = "answered"
                if answer_failed:
                    failed_steps += 1
                    reasoning_status = "fallback"
                    if reasoning_reason == "final_response":
                        reasoning_reason = "answer_verification_failed"
        if not reasoning_recorded:
            self._record_reasoning_completed(
                state,
                goal=effective_goal,
                status=reasoning_status,
                completed_steps=completed_steps,
                failed_steps=failed_steps,
                reason=reasoning_reason,
            )
        return self._finish_turn(state, answer_text, tool_results, budget_reports)

    def budget_demo(self, user_text: str, *, prompt_mode: str = "standard") -> dict[str, Any]:
        messages = [Message(role="user", content=user_text, created_at=utc_now_iso())]
        counter = self._get_budget_counter(None)
        decision_assembly = self.prompts.build_decision_prompt(messages, self.tools.prompt_tuples(self.config), prompt_mode=prompt_mode)
        decision_contract = tool_decision_contract(self.tools.tool_names(self.config))
        decision_report = self._budget_report(None, decision_assembly, decision_contract)
        answer_assembly = self.prompts.build_answer_prompt(messages, prompt_mode=prompt_mode)
        answer_report = self._budget_report(None, answer_assembly, plain_text_contract())
        return {
            "decision": {"prompt_mode": prompt_mode, "budget": asdict(decision_report), "prompt": decision_assembly.prompt_text},
            "answer": {"prompt_mode": prompt_mode, "budget": asdict(answer_report), "prompt": answer_assembly.prompt_text},
        }

    def doctor(self, *, session_id: str | None = None) -> dict[str, Any]:
        state = self.create_or_load_session(session_id)
        self.history.record_event(state, "model_request_sent", {"kind": "doctor_health", "prompt_mode": "n/a", "attempt": 1, "request": {"endpoint": "health"}, "budget_report": None})
        health = self.client.health()
        self.history.record_event(state, "doctor_health_checked", {"health": health})
        token_count = self._tokenize_with_history(state, "doctor probe").tokens
        self.history.record_event(state, "doctor_tokenize_checked", {"probe": "doctor probe", "tokens": token_count})
        grammar_assembly = self.prompts._assemble("doctor", "lean", [PromptComponent(name="doctor", category="instruction", text="Reply yes.")])
        grammar_prepared = PreparedCall(
            assembly=grammar_assembly,
            report=self._budget_report(state, grammar_assembly, yes_no_contract()),
            prompt_mode="lean",
            contract=yes_no_contract(),
        )
        grammar_result = self._execute_model_call(state, grammar_prepared)
        schema_prompt = self.prompts._assemble(
            "doctor",
            "lean",
            [PromptComponent(name="doctor", category="instruction", text='Return JSON only with action="respond", tool_name="none", tool_input={}, and response="ok".')],
        )
        schema_prepared = PreparedCall(
            assembly=schema_prompt,
            report=self._budget_report(state, schema_prompt, tool_decision_contract(self.tools.tool_names(self.config))),
            prompt_mode="lean",
            contract=tool_decision_contract(self.tools.tool_names(self.config)),
        )
        schema_result = self._execute_model_call(state, schema_prepared)
        parsed_schema = self._parse_json(schema_result.text, contract_name="tool_decision")
        return {
            "session_id": state.session_id,
            "health": health,
            "tokenize_probe_tokens": token_count,
            "grammar_probe": grammar_result.text.strip(),
            "schema_probe": parsed_schema,
        }

    def _finish_turn(
        self,
        state: SessionState,
        assistant_text: str,
        tool_results: list[ToolExecutionResult],
        budget_reports: list[BudgetReport],
    ) -> TurnResult:
        self._record_message(state, Message(role="assistant", content=assistant_text.strip(), created_at=utc_now_iso()))
        self.history.record_event(
            state,
            "turn_finished",
            {
                "turn_index": state.turn_count + 1,
                "assistant_text": assistant_text.strip(),
                "tool_steps": len(tool_results),
                "budget_reports": [asdict(item) for item in budget_reports],
            },
        )
        self._refresh_working_memory(state, reason="turn_finished")
        self._check_consistency(state)
        return TurnResult(
            session_id=state.session_id,
            assistant_text=assistant_text.strip(),
            tool_results=tool_results,
            budget_reports=budget_reports,
        )

    def _record_message(self, state: SessionState, message: Message) -> None:
        self.history.record_event(state, "message_added", {"message": asdict(message)})

    def session_status_payload(self, state: SessionState) -> dict[str, Any]:
        active_step = self._current_active_step_text(state)
        running_processes = [
            {
                "process_id": process_id,
                "command": record.command,
                "status": record.status,
            }
            for process_id, record in sorted(state.environment.processes.items())
            if record.status == "running"
        ]
        return {
            "session_id": state.session_id,
            "session_name": state.session_name,
            "active_goal": self._goal_text(state),
            "active_step": active_step,
            "waiting": state.environment.waiting,
            "waiting_reason": state.environment.waiting_reason,
            "running_processes": running_processes,
            "deferred_tasks": [asdict(item) for item in state.deferred_tasks],
            "checkpoint_count": len(state.code_checkpoints),
            "turn_count": state.turn_count,
            "event_count": state.event_count,
        }

    def queue_control_message(self, session_ref: str | None, message: str, *, source: str = "cli") -> dict[str, Any]:
        session_id = self.history.resolve_session_ref(session_ref, latest_if_none=False)
        if session_id is None and session_ref is None:
            active_entries = [entry for entry in self.history.list_session_entries() if entry.get("active")]
            if active_entries:
                session_id = str(active_entries[0]["session_id"])
        if session_id is None:
            session_id = self.history.resolve_session_ref(session_ref, latest_if_none=True)
        if session_id is None:
            raise FileNotFoundError("No session available")
        payload = self.history.enqueue_control_message(session_id, message, source=source)
        state = self.history.rebuild_from_history(session_id, write_projections=False)
        return {
            **payload,
            "active": self.history.read_active_run(session_id) is not None,
            "status": self.session_status_payload(state),
        }

    def query_history_details(
        self,
        *,
        session_ref: str | None,
        query_text: str,
        topic_hint: str = "",
    ) -> dict[str, Any]:
        pol = self.config.selection_policy
        return self.history.query_history_details(
            session_ref,
            query_text,
            topic_hint=topic_hint,
            max_results=pol.history_query_max_results,
            token_score=pol.history_detail_token_score,
            exact_score=pol.history_detail_exact_score,
            type_bonus=pol.history_detail_type_bonus,
            preview_chars=pol.history_detail_preview_chars,
        )

    def pop_next_deferred_task(self, state: SessionState, *, reason: str) -> DeferredTask | None:
        if not state.deferred_tasks:
            return None
        task = state.deferred_tasks[0]
        self.history.record_event(
            state,
            "deferred_task_consumed",
            {"task_id": task.task_id, "reason": reason},
        )
        return task

    def create_code_checkpoint(
        self,
        state: SessionState,
        *,
        label: str = "",
        workspace_root: str | None = None,
    ) -> dict[str, Any]:
        environment = AgentEnvironment(self.config, state)
        root_path = Path(workspace_root).expanduser().resolve() if workspace_root else environment.filesystem.workspace_root.resolve()
        checkpoint_id = new_id("checkpoint")
        checkpoint_dir = self.history.code_checkpoints_dir(state.session_id) / checkpoint_id
        files_dir = checkpoint_dir / "files"
        os.makedirs(str(files_dir), exist_ok=True)
        sessions_root = self.config.sessions.root
        if not sessions_root.is_absolute():
            sessions_root = (root_path / sessions_root).resolve()
        manifest: list[str] = []
        for path in sorted(root_path.rglob("*")):
            if not path.is_file():
                continue
            if path.is_relative_to(checkpoint_dir):
                continue
            if path.is_relative_to(sessions_root):
                continue
            if ".git" in path.parts:
                continue
            relative = path.relative_to(root_path)
            target = files_dir / relative
            os.makedirs(str(target.parent), exist_ok=True)
            shutil.copy2(str(path), str(target))
            manifest.append(str(relative))
        checkpoint_payload = {
            "checkpoint_id": checkpoint_id,
            "label": label.strip() or f"checkpoint-{len(state.code_checkpoints) + 1}",
            "created_at": utc_now_iso(),
            "workspace_root": str(root_path),
            "storage_path": str(checkpoint_dir),
            "file_count": len(manifest),
            "metadata": {"manifest_path": str(checkpoint_dir / "manifest.json")},
        }
        manifest_path = checkpoint_dir / "manifest.json"
        with open(str(manifest_path), "w", encoding="utf-8") as _mf:
            _mf.write(stable_json_dumps({"workspace_root": str(root_path), "files": manifest}, indent=2))
        self.history.record_event(state, "code_checkpoint_created", {"checkpoint": checkpoint_payload})
        return checkpoint_payload

    def restore_code_checkpoint(
        self,
        state: SessionState,
        *,
        checkpoint_ref: str = "latest",
        workspace_root: str | None = None,
    ) -> dict[str, Any]:
        if not state.code_checkpoints:
            raise RuntimeError("No code checkpoints are available")
        if checkpoint_ref in {"latest", ""}:
            checkpoint = state.code_checkpoints[-1]
        else:
            checkpoint = next(
                (
                    item
                    for item in state.code_checkpoints
                    if item.checkpoint_id == checkpoint_ref or item.label == checkpoint_ref
                ),
                None,
            )
            if checkpoint is None:
                raise FileNotFoundError(f"Unknown checkpoint: {checkpoint_ref}")
        root_path = Path(workspace_root).expanduser().resolve() if workspace_root else Path(checkpoint.workspace_root).resolve()
        checkpoint_dir = Path(checkpoint.storage_path)
        manifest_path = checkpoint_dir / "manifest.json"
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot_files = {str(item) for item in manifest_payload.get("files", [])}
        sessions_root = self.config.sessions.root
        if not sessions_root.is_absolute():
            sessions_root = (root_path / sessions_root).resolve()
        for path in sorted(root_path.rglob("*"), reverse=True):
            if not path.exists():
                continue
            if path.is_relative_to(sessions_root) or ".git" in path.parts:
                continue
            if path.is_file():
                rel = str(path.relative_to(root_path))
                if rel not in snapshot_files:
                    os.remove(str(path))
        for rel in sorted(snapshot_files):
            source = checkpoint_dir / "files" / rel
            target = root_path / rel
            os.makedirs(str(target.parent), exist_ok=True)
            shutil.copy2(str(source), str(target))
        self.history.record_event(
            state,
            "code_checkpoint_restored",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "restored_to": checkpoint.label,
                "workspace_root": str(root_path),
            },
        )
        self._refresh_project_state(state, reason=f"checkpoint_restored:{checkpoint.checkpoint_id}")
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "label": checkpoint.label,
            "workspace_root": str(root_path),
            "file_count": len(snapshot_files),
        }

    def _current_active_step_text(self, state: SessionState) -> str:
        if state.active_plan is None or not state.active_plan.current_step_id:
            return ""
        step = next(
            (item for item in state.active_plan.steps if item.step_id == state.active_plan.current_step_id),
            None,
        )
        if step is None:
            return ""
        return f"{step.step_id}: {step.title} [{step.status}]"

    def _record_control_note(self, state: SessionState, text: str, *, title: str = "Control update") -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        timestamp = utc_now_iso()
        note_payload = {
            "note_id": new_id("note"),
            "title": title,
            "content": cleaned,
            "created_at": timestamp,
            "updated_at": timestamp,
            "metadata": {"source": "control"},
        }
        self.history.record_event(state, "note_added", {"note": note_payload})

    def _classify_control_message_frontend(
        self,
        state: SessionState,
        *,
        effective_goal: str,
        message: str,
    ) -> dict[str, str]:
        contract = active_session_control_contract()
        prepared = self._prepare_call(
            state,
            kind="control",
            build_prompt=lambda prompt_mode, bundle: self.prompts.build_active_session_control_prompt(
                session_goal=effective_goal,
                active_step=self._current_active_step_text(state),
                waiting_reason=state.environment.waiting_reason,
                queued_message=message,
                prompt_mode=prompt_mode,
                context_components=bundle.components,
            ),
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
            goal=effective_goal,
        )

        def _validate(payload: dict[str, Any]) -> dict[str, str]:
            action = str(payload.get("action", "")).strip()
            if action not in {
                "status",
                "session_summary",
                "continue_with_note",
                "cancel",
                "stop",
                "replace_task",
                "queue_after_current",
                "clarify_conflict",
            }:
                raise ValueError(f"Unknown control action: {action}")
            validated = {
                "action": action,
                "reason": str(payload.get("reason", "")).strip(),
                "response_text": str(payload.get("response_text", "")).strip(),
                "added_context": str(payload.get("added_context", "")).strip(),
                "replacement_goal": str(payload.get("replacement_goal", "")).strip(),
                "queued_task": str(payload.get("queued_task", "")).strip(),
                "clarification_question": str(payload.get("clarification_question", "")).strip(),
            }
            if not validated["reason"]:
                raise ValueError("Control action reason must not be empty")
            return validated

        _completion, decision = self._execute_structured_call(
            state,
            prepared,
            validator=_validate,
            validation_error_types=(ValueError,),
        )
        return decision

    def _process_pending_control_messages(
        self,
        state: SessionState,
        *,
        effective_goal: str,
    ) -> ControlProcessingResult:
        result = ControlProcessingResult()
        pending = self.history.list_pending_control_messages(state.session_id)
        if not pending:
            return result
        for message_payload in pending:
            control_id = str(message_payload.get("control_id", ""))
            message = str(message_payload.get("message", "")).strip()
            if not control_id or not message:
                continue
            decision = self._classify_control_message_frontend(
                state,
                effective_goal=effective_goal,
                message=message,
            )
            self.history.record_event(
                state,
                "control_message_processed",
                {
                    "control_id": control_id,
                    "session_id": state.session_id,
                    "message": message,
                    "decision": decision,
                },
            )
            action = decision["action"]
            effect = "no_state_change"
            if action in {"status", "session_summary"}:
                if decision["response_text"]:
                    result.assistant_messages.append(decision["response_text"])
                effect = action
            elif action == "continue_with_note":
                note_text = decision["added_context"] or message
                self._record_control_note(state, note_text)
                if decision["response_text"]:
                    result.assistant_messages.append(decision["response_text"])
                effect = "note_added"
            elif action in {"cancel", "stop"}:
                result.stop_requested = True
                result.assistant_messages.append(decision["response_text"] or "stopped by user request")
                effect = action
            elif action == "replace_task":
                replacement_goal = decision["replacement_goal"] or message
                result.replacement_goal = replacement_goal
                result.replan_requested = True
                if decision["response_text"]:
                    result.assistant_messages.append(decision["response_text"])
                effect = "replacement_requested"
            elif action == "queue_after_current":
                queued_text = decision["queued_task"] or message
                if queued_text:
                    task = DeferredTask(
                        task_id=new_id("task"),
                        text=queued_text,
                        queued_at=utc_now_iso(),
                        source="control",
                    )
                    self.history.record_event(state, "deferred_task_queued", {"task": asdict(task)})
                    effect = "deferred_task_queued"
                if decision["response_text"]:
                    result.assistant_messages.append(decision["response_text"])
            elif action == "clarify_conflict":
                clarification = decision["clarification_question"] or "Clarify whether the current task should be replaced."
                result.assistant_messages.append(clarification)
                effect = "clarification_needed"
            self.history.record_event(
                state,
                "control_action_applied",
                {
                    "control_id": control_id,
                    "session_id": state.session_id,
                    "action": action,
                    "effect": effect,
                },
            )
            self.history.mark_control_message_processed(state.session_id, control_id)
        return result

    def _prepare_turn_context(self, state: SessionState, user_text: str) -> TurnPreparation:
        analysis = self._analyze_prompt_frontend(state, user_text)
        decision = self._decide_prompt_frontend(state, user_text, analysis)
        explicit_tools = self._detect_explicit_named_tools(user_text)
        explicit_tool = explicit_tools[0] if explicit_tools else None
        if decision.direct_response and explicit_tool is not None:
            decision = replace(
                decision,
                direct_response=False,
                execution_mode="full_plan",
                preferred_tool_name="",
                reason=f"{decision.reason}; direct_response_blocked=explicit_tool:{explicit_tool}",
            )
            self.history.record_event(
                state,
                "decision_adjusted",
                {
                    "reason": "explicit_named_tool_requirement",
                    "tool_name": explicit_tool,
                    "decision": asdict(decision),
                },
            )
        expanded: ExpandedTask | None = None
        effective_goal = self._operational_goal_from_task_contract(user_text)
        clarification_request: str | None = None
        if decision.expand_task and not decision.direct_response:
            expanded = self._expand_task_frontend(state, user_text, analysis, decision)
            effective_goal = expanded.expanded_goal
        if decision.ask_user and not decision.direct_response:
            clarification_request = self._build_clarification_request(user_text, analysis)
        strategy = self._select_strategy_frontend(state, effective_goal, analysis, decision)
        self._set_strategy(state, strategy, reason=strategy.reason)
        if decision.direct_response and any(kind != "respond" for kind in strategy.required_step_kinds):
            decision = replace(
                decision,
                direct_response=False,
                execution_mode="full_plan",
                preferred_tool_name="",
                reason=(
                    f"{decision.reason}; direct_response_blocked=strategy_requires:"
                    f"{','.join(strategy.required_step_kinds)}"
                ),
            )
            self.history.record_event(
                state,
                "decision_adjusted",
                {
                    "reason": "strategy_requires_full_plan",
                    "tool_name": "",
                    "required_step_kinds": list(strategy.required_step_kinds),
                    "decision": asdict(decision),
                },
            )
        self._refresh_project_state(state, reason="turn_prepared")
        return TurnPreparation(
            analysis=analysis,
            decision=decision,
            effective_goal=effective_goal,
            expanded_task=expanded,
            clarification_request=clarification_request,
            required_named_tools=tuple(explicit_tools),
        )

    def _detect_explicit_named_tools(self, text: str) -> list[str]:
        lowered = text.lower()
        matches: list[str] = []
        for tool_name in self.tools.tool_names(self.config):
            underscored = tool_name.lower()
            spaced = underscored.replace("_", " ")
            phrases = (
                f"use the {underscored} tool",
                f"use the {spaced} tool",
                f"call the {underscored} tool",
                f"call the {spaced} tool",
            )
            if any(phrase in lowered for phrase in phrases):
                matches.append(tool_name)
        return matches

    def _detect_explicit_named_tool_request(self, text: str) -> str | None:
        matches = self._detect_explicit_named_tools(text)
        return matches[0] if matches else None

    def _extract_task_contract(self, text: str) -> dict[str, Any] | None:
        match = re.search(r"Task contract:\s*(\{[^\n]+\})", text)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _task_contract_for_goal(self, state: SessionState, goal: str) -> dict[str, Any] | None:
        contract = self._extract_task_contract(goal)
        if contract is not None:
            return contract
        for message in reversed(state.messages):
            if message.role != "user":
                continue
            contract = self._extract_task_contract(message.content)
            if contract is not None:
                return contract
        return None

    def _apply_task_contract_to_analysis(self, user_text: str, analysis: PromptAnalysis) -> PromptAnalysis:
        contract = self._extract_task_contract(user_text)
        if not contract:
            return analysis
        updated = PromptAnalysis(
            task_type=analysis.task_type,
            completeness=analysis.completeness,
            requires_expansion=analysis.requires_expansion,
            requires_decomposition=analysis.requires_decomposition,
            confidence=analysis.confidence,
            detected_entities=list(analysis.detected_entities),
            detected_goals=list(analysis.detected_goals),
        )
        if str(contract.get("request_completeness", "")).strip() == "complete":
            updated.completeness = "complete"
        if contract.get("prefer_task_expansion") is False:
            updated.requires_expansion = False
            updated.requires_decomposition = False
        if contract.get("requires_code_changes") is True and updated.task_type == "structured":
            updated.requires_decomposition = False
        return updated

    def _apply_task_contract_to_decision(self, user_text: str, decision: DecisionOutcome) -> DecisionOutcome:
        contract = self._extract_task_contract(user_text)
        if not contract:
            return decision
        updated = DecisionOutcome(**asdict(decision))
        if contract.get("prefer_task_expansion") is False:
            updated.expand_task = False
            updated.split_task = False
        if contract.get("requires_code_changes") is True and updated.direct_response:
            updated.direct_response = False
            updated.execution_mode = "full_plan"
            updated.preferred_tool_name = ""
        if updated.reason:
            updated.reason = f"{updated.reason};task_contract"
        else:
            updated.reason = "task_contract"
        return updated

    def _operational_goal_from_task_contract(self, text: str) -> str:
        contract = self._extract_task_contract(text)
        if not contract:
            return text

        def _section_lines(label: str, *, stop_labels: tuple[str, ...]) -> list[str]:
            lines = text.splitlines()
            collected: list[str] = []
            capture = False
            for raw in lines:
                stripped = raw.strip()
                if stripped == label:
                    capture = True
                    continue
                if capture and stripped in stop_labels:
                    break
                if capture:
                    collected.append(raw)
            return collected

        problem_lines = _section_lines(
            "Problem statement:",
            stop_labels=("Known failing tests:", "Hints:", "Benchmark recovery note:"),
        )
        problem_summary = ""
        for raw in problem_lines:
            stripped = raw.strip()
            if not stripped or stripped.startswith("```"):
                continue
            problem_summary = stripped
            break
        failing_tests = [
            raw.strip()[2:].strip()
            for raw in _section_lines(
                "Known failing tests:",
                stop_labels=("Hints:", "Benchmark recovery note:"),
            )
            if raw.strip().startswith("- ")
        ]
        parts: list[str] = []
        if problem_summary:
            parts.append(problem_summary[:220])
        if failing_tests:
            parts.append(f"Verify {', '.join(failing_tests[:2])}.")
        if not parts:
            return text
        return "Fix the benchmark issue. " + " ".join(parts)

    def _install_direct_response_plan(self, state: SessionState, goal: str) -> Plan:
        if state.active_plan is not None and state.active_plan.status == "active" and state.active_plan.goal == goal:
            return state.active_plan
        plan = create_direct_response_plan(goal)
        event_type = "plan_updated" if state.active_plan is not None else "plan_created"
        if event_type == "plan_created":
            event = self.history.record_event(state, event_type, {"goal": goal, "plan": plan_as_payload(plan)})
        else:
            event = self.history.record_event(
                state,
                event_type,
                {"plan": plan_as_payload(plan), "reason": "semantic_direct_response"},
            )
        self._extract_and_store_memory(state, event)
        self._refresh_working_memory(state, reason="direct_response_plan")
        self._refresh_project_state(state, reason="direct_response_plan")
        self._check_consistency(state)
        return state.active_plan or plan

    def _install_direct_tool_plan(self, state: SessionState, goal: str, tool_name: str, *, reason: str) -> Plan:
        if state.active_plan is not None and state.active_plan.status == "active" and state.active_plan.goal == goal:
            if any(step.expected_tool == tool_name for step in state.active_plan.steps):
                return state.active_plan
        plan = create_direct_tool_plan(goal, tool_name)
        event_type = "plan_updated" if state.active_plan is not None else "plan_created"
        if event_type == "plan_created":
            event = self.history.record_event(
                state,
                event_type,
                {"goal": goal, "plan": plan_as_payload(plan)},
            )
        else:
            event = self.history.record_event(
                state,
                event_type,
                {"plan": plan_as_payload(plan), "reason": reason},
            )
        self._extract_and_store_memory(state, event)
        self._refresh_working_memory(state, reason="direct_tool_plan")
        self._refresh_project_state(state, reason="direct_tool_plan")
        self._check_consistency(state)
        return state.active_plan or plan

    def _log_fatal_system_error(
        self,
        state: SessionState,
        *,
        category: str,
        prepared: PreparedCall | None,
        error: Exception,
        raw_response: str | None = None,
        details: dict[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> None:
        if prepared is not None:
            call_kind = prepared.assembly.kind
            operation = call_kind
            contract_name = prepared.contract.name
            contract_mode = prepared.contract.mode
            prompt_hash = sha256_text(prepared.assembly.prompt_text)
            budget_report = asdict(prepared.report)
            request_id = f"{state.session_id}:{call_kind}:{prompt_hash[:16]}"
        else:
            detail_payload = details or {}
            call_kind = str(detail_payload.get("kind") or operation_name or category)
            operation = operation_name or call_kind
            contract_name = "n/a"
            contract_mode = "n/a"
            prompt_hash = ""
            budget_report = None
            request_id = f"{state.session_id}:{call_kind}:{sha256_text(stable_json_dumps(detail_payload))[:16]}"
        payload = {
            "timestamp": utc_now_iso(),
            "request_id": request_id,
            "operation": operation,
            "call_kind": call_kind,
            "contract_name": contract_name,
            "contract_mode": contract_mode,
            "prompt_hash": prompt_hash,
            "model_profile": self.config.model.profile_name,
            "model_base_url": self.config.model.base_url,
            "structured_output_mode": self.config.model.structured_output_mode,
            "context_limit": self.config.model.context_limit,
            "budget_report": budget_report,
            "error": str(error),
            "error_type": error.__class__.__name__,
            "raw_response": (raw_response or "")[:4000],
            "details": details or {},
            "why_fatal": (
                "A core structured semantic call violated its enforced grammar/schema contract. "
                "This is a fundamental semantic-engine failure, not a normal retry/replan case."
            ),
        }
        self.history.append_auxiliary_log("fatal_system_errors.jsonl", payload)
        self.history.record_event(
            state,
            "fatal_system_error",
            {
                "operation": operation,
                "error": str(error),
                "error_type": error.__class__.__name__,
                "category": category,
                "warning": "fatal_structured_semantic_failure",
            },
        )

    def _execute_structured_call(
        self,
        state: SessionState,
        prepared: PreparedCall,
        *,
        validator=None,
        validation_error_types: tuple[type[BaseException], ...] = (),
    ) -> tuple[CompletionResult, Any]:
        completion = self._execute_model_call(state, prepared)
        try:
            payload = self._parse_json(completion.text, contract_name=prepared.contract.name)
        except Exception as exc:
            if prepared.contract.mode in {"json_schema", "gbnf"}:
                self._log_fatal_system_error(
                    state,
                    category="structured_parse_failure",
                    prepared=prepared,
                    error=exc,
                    raw_response=completion.text,
                )
                raise FatalSemanticEngineError(str(exc)) from exc
            raise
        if validator is None:
            return completion, payload
        try:
            validated = validator(payload)
        except validation_error_types as exc:
            if prepared.contract.mode in {"json_schema", "gbnf"}:
                self._log_fatal_system_error(
                    state,
                    category="structured_validation_failure",
                    prepared=prepared,
                    error=exc,
                    raw_response=completion.text,
                    details={"payload": payload},
                )
                raise FatalSemanticEngineError(str(exc)) from exc
            raise
        return completion, validated

    def _validate_summary_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        summary_text = str(payload.get("summary", "")).strip()
        if not summary_text:
            raise ValueError("Summary call returned empty summary")
        return {"summary": summary_text}

    def _validate_verification_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        criteria = payload.get("criteria")
        if not isinstance(criteria, list):
            raise ValueError("Verification call returned invalid criteria payload")
        validated: list[dict[str, Any]] = []
        for item in criteria:
            if not isinstance(item, dict):
                raise ValueError("Verification criteria entry must be an object")
            name = str(item.get("name", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            if not name:
                raise ValueError("Verification criteria entry is missing name")
            validated.append(
                {
                    "name": name,
                    "passed": bool(item.get("passed")),
                    "evidence": evidence,
                }
            )
        return {"criteria": validated}

    def _analyze_prompt_frontend(self, state: SessionState, user_text: str) -> PromptAnalysis:
        contract = prompt_analysis_contract()
        prepared = self._prepare_call(
            state,
            kind="analysis",
            build_prompt=lambda prompt_mode, bundle: self.prompts.build_analysis_prompt(
                user_text,
                prompt_mode=prompt_mode,
                context_components=bundle.components,
            ),
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
            goal=user_text,
        )
        _completion, analysis = self._execute_structured_call(
            state,
            prepared,
            validator=analysis_from_payload,
            validation_error_types=(PromptAnalysisValidationError,),
        )
        analysis = self._apply_task_contract_to_analysis(user_text, analysis)
        source = "model"
        self.history.record_event(state, "prompt_analyzed", {"analysis": asdict(analysis), "source": source})
        return analysis

    def _decide_prompt_frontend(self, state: SessionState, user_text: str, analysis: PromptAnalysis) -> DecisionOutcome:
        contract = task_decision_contract(self.tools.tool_names(self.config))
        prepared = self._prepare_call(
            state,
            kind="task_decision",
            build_prompt=lambda prompt_mode, bundle: self.prompts.build_task_decision_prompt(
                user_text,
                stable_json_dumps(asdict(analysis)),
                prompt_mode=prompt_mode,
                context_components=bundle.components,
            ),
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
            goal=user_text,
        )
        _completion, decision = self._execute_structured_call(
            state,
            prepared,
            validator=lambda payload: decision_from_payload(payload, analysis),
            validation_error_types=(DecisionValidationError,),
        )
        decision = self._apply_task_contract_to_decision(user_text, decision)
        source = "model"
        self.history.record_event(state, "decision_made", {"decision": asdict(decision), "source": source})
        return decision

    def _expand_task_frontend(
        self,
        state: SessionState,
        user_text: str,
        analysis: PromptAnalysis,
        decision: DecisionOutcome,
    ) -> ExpandedTask:
        contract = task_expansion_contract()
        prepared = self._prepare_call(
            state,
            kind="expansion",
            build_prompt=lambda prompt_mode, bundle: self.prompts.build_task_expansion_prompt(
                user_text,
                stable_json_dumps(asdict(analysis)),
                stable_json_dumps(asdict(decision)),
                prompt_mode=prompt_mode,
                context_components=bundle.components,
            ),
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
            goal=user_text,
        )
        _completion, expanded = self._execute_structured_call(
            state,
            prepared,
            validator=lambda payload: expanded_task_from_payload(payload, original_goal=user_text),
            validation_error_types=(ExpansionValidationError,),
        )
        source = "model"
        self.history.record_event(state, "task_expanded", {"expanded_task": asdict(expanded), "source": source})
        return expanded

    def _select_strategy_frontend(
        self,
        state: SessionState,
        effective_goal: str,
        analysis: PromptAnalysis,
        decision: DecisionOutcome,
    ):
        """LLM-driven strategy selection."""

        contract = strategy_selection_contract()
        instruction_text = (
            "Pick the execution strategy profile that best fits the task.\n"
            "Respond with JSON only matching the schema. Choose task_profile from\n"
            "[coding, file_edit, reading, multi_step, generic]. Choose strategy_name\n"
            "from [conservative, exploratory]. tool_chain_depth must be 1..3.\n"
            "Use reading only for information-gathering tasks that do not require repository changes.\n"
            "If the goal explicitly requires code edits, file writes, patches, or running tests, prefer coding or file_edit.\n"
        )

        def _build(prompt_mode: str, bundle: ContextBundle) -> PromptAssembly:
            user_components = [
                PromptComponent(
                    name="current_goal",
                    category="current_user",
                    text=f"Effective goal:\n{effective_goal}\n\n",
                ),
                PromptComponent(
                    name="analysis",
                    category="analysis",
                    text=f"Prompt analysis:\n{stable_json_dumps(asdict(analysis))}\n\n",
                ),
                PromptComponent(
                    name="task_decision",
                    category="decision",
                    text=f"Task decision:\n{stable_json_dumps(asdict(decision))}\n\n",
                ),
                *bundle.components,
                PromptComponent(
                    name="strategy_instruction",
                    category="instruction",
                    text=instruction_text,
                ),
            ]
            return self.prompts._assemble("strategy", prompt_mode, user_components)

        prepared = self._prepare_call(
            state,
            kind="strategy",
            build_prompt=_build,
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
            goal=effective_goal,
        )
        _completion, strategy = self._execute_structured_call(
            state,
            prepared,
            validator=strategy_from_payload,
            validation_error_types=(StrategyValidationError,),
        )
        source = "model"
        self.history.record_event(
            state,
            "strategy_selection_resolved",
            {"strategy": asdict(strategy), "source": source},
        )
        return strategy

    def _select_subagent_frontend(
        self,
        state: SessionState,
        *,
        goal: str,
        purpose: str,
        candidate_types: list[str],
        detail_lines: list[str] | None = None,
    ) -> SubagentSelectionDecision:
        if not candidate_types:
            return SubagentSelectionDecision(spawn=False, subagent_type="none", reason="no_candidates", focus="")
        contract = subagent_selection_contract(candidate_types)
        candidate_text = ", ".join(candidate_types)
        detail_text = "\n".join(line for line in (detail_lines or []) if line.strip())
        instruction_text = (
            "Decide whether an isolated specialist subagent should be spawned for this stage.\n"
            f"Available subagents: {candidate_text}.\n"
            "Choose spawn=true only when a specialist would materially improve the current decision.\n"
            "Choose subagent_type='none' when no specialist is needed.\n"
            "Respond with JSON only matching the schema.\n"
        )

        def _build(prompt_mode: str, bundle: ContextBundle) -> PromptAssembly:
            user_components = [
                PromptComponent(
                    name="subagent_goal",
                    category="current_user",
                    text=f"Goal:\n{goal}\n\n",
                ),
                PromptComponent(
                    name="subagent_purpose",
                    category="subagent",
                    text=f"Subagent purpose:\n{purpose}\n\n",
                ),
                *bundle.components,
            ]
            if detail_text:
                user_components.append(
                    PromptComponent(
                        name="subagent_details",
                        category="subagent",
                        text=f"Stage details:\n{detail_text}\n\n",
                    )
                )
            user_components.append(
                PromptComponent(
                    name="subagent_instruction",
                    category="instruction",
                    text=instruction_text,
                )
            )
            return self.prompts._assemble("subagent_selection", prompt_mode, user_components)

        prepared = self._prepare_call(
            state,
            kind="subagent_selection",
            build_prompt=_build,
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
            goal=goal,
        )

        def _validate(payload: dict[str, Any]) -> SubagentSelectionDecision:
            subagent_type = str(payload.get("subagent_type", "")).strip() or "none"
            if subagent_type not in {"none", *candidate_types}:
                raise ValueError(f"Unknown subagent_type: {subagent_type}")
            reason = str(payload.get("reason", "")).strip()
            if not reason:
                raise ValueError("Subagent selection reason must not be empty")
            spawn = bool(payload.get("spawn"))
            if subagent_type == "none":
                spawn = False
            return SubagentSelectionDecision(
                spawn=spawn,
                subagent_type=subagent_type,
                reason=reason,
                focus=str(payload.get("focus", "")).strip(),
            )

        _completion, selection = self._execute_structured_call(
            state,
            prepared,
            validator=_validate,
            validation_error_types=(ValueError,),
        )
        self.history.record_event(
            state,
            "subagent_selection_resolved",
            {
                "purpose": purpose,
                "candidate_types": candidate_types,
                "selection": asdict(selection),
            },
        )
        return selection

    def _classify_failure_frontend(
        self,
        state: SessionState,
        *,
        step: PlanStep | None,
        error: Exception | None = None,
        error_type: str | None = None,
        reason: str = "",
    ) -> FailureClassification:
        """LLM-driven failure classification."""
        contract = failure_classification_contract()
        normalized_error_type = error_type or (error.__class__.__name__ if error is not None else "")
        instruction_text = (
            "Classify the failure that occurred. Respond with JSON only.\n"
            "kind must be one of [tool_failure, reasoning_failure, planning_failure,\n"
            "missing_information, verification_failure, budget_failure, state_inconsistency,\n"
            "transient_external_wait, retry_now, retry_later_backoff, deterministic_permanent,\n"
            "side_effect_unsafe, needs_replan, needs_clarification, blocked_external,\n"
            "continue_other].\n"
            "suggested_strategy_mode must be one of [conservative, recovery, verification_heavy].\n"
            "Set retryable=true only if a simple retry of the same action is likely to help.\n"
            "Set wait_seconds to the number of seconds to wait before retrying when the kind\n"
            "is transient_external_wait or retry_later_backoff; otherwise 0.\n"
        )
        step_payload = "(no step)"
        if step is not None:
            step_payload = stable_json_dumps(
                {
                    "step_id": step.step_id,
                    "kind": step.kind,
                    "expected_tool": step.expected_tool,
                    "title": step.title,
                }
            )

        def _build(prompt_mode: str, bundle: ContextBundle) -> PromptAssembly:
            user_components = [
                PromptComponent(
                    name="failure_step",
                    category="current_user",
                    text=f"Failed step:\n{step_payload}\n\n",
                ),
                PromptComponent(
                    name="failure_signal",
                    category="failure_signal",
                    text=(
                        f"Reported reason: {reason or '(none)'}\n"
                        f"Error type: {normalized_error_type or '(none)'}\n\n"
                    ),
                ),
                *bundle.components,
                PromptComponent(
                    name="failure_instruction",
                    category="instruction",
                    text=instruction_text,
                ),
            ]
            return self.prompts._assemble("failure", prompt_mode, user_components)

        prepared = self._prepare_call(
            state,
            kind="failure",
            build_prompt=_build,
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
            goal=self._goal_text(state),
        )
        _completion, classification = self._execute_structured_call(
            state,
            prepared,
            validator=classify_failure_from_payload,
            validation_error_types=(FailureValidationError,),
        )
        source = "model"
        self.history.record_event(
            state,
            "failure_classification_resolved",
            {"classification": asdict(classification), "source": source},
        )
        return classification

    def _select_action_frontend(
        self,
        state: SessionState,
        orchestration,
    ):
        """LLM-driven action disambiguation when the orchestrator returns
        multiple candidate actions."""

        if not orchestration.requires_llm_decision or len(orchestration.candidate_actions) <= 1:
            return orchestration.action
        contract = action_selection_contract()
        candidates = list(orchestration.candidate_actions)
        instruction_text = (
            "Pick the next execution action. Respond with JSON only.\n"
            f"action must be one of {candidates}.\n"
            "Use 'execute_step' to run the next ready step, 'retry_step' to retry the\n"
            "same step, 'replan' to rebuild the plan, 'wait' to wait for background work,\n"
            "'stop' to halt, or 'answer_directly' if the user can be answered without\n"
            "further tool use.\n"
        )
        step_payload = "(no step)"
        if orchestration.step is not None:
            step_payload = stable_json_dumps(
                {
                    "step_id": orchestration.step.step_id,
                    "kind": orchestration.step.kind,
                    "expected_tool": orchestration.step.expected_tool,
                    "title": orchestration.step.title,
                }
            )

        def _build(prompt_mode: str, bundle: ContextBundle) -> PromptAssembly:
            user_components = [
                PromptComponent(
                    name="action_step",
                    category="current_user",
                    text=f"Active step:\n{step_payload}\n\n",
                ),
                PromptComponent(
                    name="action_candidates",
                    category="candidates",
                    text=(
                        f"Candidate actions: {candidates}\n"
                        f"Ready step ids: {orchestration.ready_step_ids}\n"
                        f"Default deterministic choice: {orchestration.action}\n\n"
                    ),
                ),
                *bundle.components,
                PromptComponent(
                    name="action_instruction",
                    category="instruction",
                    text=instruction_text,
                ),
            ]
            return self.prompts._assemble("action", prompt_mode, user_components)

        prepared = self._prepare_call(
            state,
            kind="action",
            build_prompt=_build,
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
            goal=self._goal_text(state),
        )
        _completion, chosen = self._execute_structured_call(
            state,
            prepared,
            validator=lambda payload: action_from_payload(payload, allowed_actions=candidates),
            validation_error_types=(ValueError,),
        )
        source = "model"
        self.history.record_event(
            state,
            "action_selection_resolved",
            {"selected_action": chosen, "candidates": candidates, "source": source},
        )
        return chosen

    def _build_clarification_request(self, user_text: str, analysis: PromptAnalysis) -> str:
        missing = []
        if not analysis.detected_goals:
            missing.append("the exact goal")
        if not analysis.detected_entities:
            missing.append("the relevant files, paths, or entities")
        if analysis.task_type == "incomplete":
            missing.append("the missing constraints or inputs")
        if not missing:
            missing.append("the missing details")
        return f"I need clarification before I can continue. Please provide {', '.join(missing)} for: {user_text}"

    def _refresh_project_state(self, state: SessionState, *, reason: str) -> None:
        project_state = build_project_state(state)
        if self._project_state_signature(state.project_state) == self._project_state_signature(project_state):
            return
        self.history.record_event(
            state,
            "project_state_updated",
            {"project_state": asdict(project_state), "reason": reason},
        )

    def _set_strategy(
        self,
        state: SessionState,
        strategy,
        *,
        reason: str,
    ) -> None:
        if state.active_strategy is not None and asdict(state.active_strategy) == asdict(strategy):
            return
        strategy.reason = reason
        self.history.record_event(state, "strategy_selected", {"strategy": asdict(strategy)})

    def _reconcile_strategy_for_plan(
        self,
        state: SessionState,
        plan: Plan,
        *,
        completed_step_kinds: list[str],
    ) -> None:
        if state.active_strategy is None:
            return
        current = state.active_strategy
        try:
            validate_plan_against_strategy(
                plan,
                current,
                completed_step_kinds=completed_step_kinds,
            )
            return
        except StrategyValidationError as exc:
            original_error = exc
        reconciled = reconcile_strategy_to_plan(
            current,
            plan,
            completed_step_kinds=completed_step_kinds,
        )
        if reconciled.task_profile == current.task_profile:
            raise original_error
        self.history.record_event(
            state,
            "strategy_selection_resolved",
            {"strategy": asdict(reconciled), "source": "plan_reconciliation"},
        )
        self._set_strategy(
            state,
            reconciled,
            reason=reconciled.reason,
        )
        validate_plan_against_strategy(
            plan,
            reconciled,
            completed_step_kinds=completed_step_kinds,
        )

    def _switch_role(self, state: SessionState, role_name: str, *, reason: str) -> None:
        if state.active_role == role_name:
            return
        previous_role = state.active_role
        self.history.record_event(
            state,
            "role_switched",
            {"previous_role": previous_role, "new_role": role_name, "reason": reason},
        )

    def _completed_step_kinds(self, state: SessionState) -> list[str]:
        if state.active_plan is None:
            return []
        return [step.kind for step in state.active_plan.steps if step.status == "completed"]

    def _review_plan(self, state: SessionState, plan: Plan) -> None:
        if len(plan.steps) <= 1 and all(step.kind == "respond" for step in plan.steps):
            self.history.record_event(
                state,
                "subagent_selection_resolved",
                {
                    "purpose": "plan_review",
                    "candidate_types": ["reviewer"],
                    "selection": {
                        "spawn": False,
                        "subagent_type": "none",
                        "reason": "structurally_trivial_plan",
                        "focus": "",
                    },
                },
            )
            if state.active_strategy is not None:
                self._reconcile_strategy_for_plan(
                    state,
                    plan,
                    completed_step_kinds=self._completed_step_kinds(state),
                )
            if not all(step.verification_checks and step.required_conditions for step in plan.steps):
                raise PlanValidationError("plan_missing_required_review_properties")
            return
        selection = self._select_subagent_frontend(
            state,
            goal=plan.goal,
            purpose="plan_review",
            candidate_types=["reviewer"],
            detail_lines=[
                f"step_count={len(plan.steps)}",
                f"step_kinds={','.join(step.kind for step in plan.steps)}",
            ],
        )
        if not selection.spawn:
            self.history.record_event(
                state,
                "review_skipped",
                {"review_kind": "plan", "target_id": plan.plan_id, "reason": selection.reason},
            )
            if state.active_strategy is not None:
                self._reconcile_strategy_for_plan(
                    state,
                    plan,
                    completed_step_kinds=self._completed_step_kinds(state),
                )
            if not all(step.verification_checks and step.required_conditions for step in plan.steps):
                raise PlanValidationError("plan_missing_required_review_properties")
            return
        self._switch_role(state, "verifier", reason="plan_review")
        subagent_report = self._subagents.review_plan(state, plan)
        self.history.record_event(
            state,
            "subagent_spawned",
            {
                "subagent_type": subagent_report.spec.subagent_type,
                "purpose": subagent_report.spec.purpose,
                "allowed_tools": subagent_report.spec.allowed_tools,
                "token_budget": subagent_report.spec.token_budget,
                "target_id": plan.plan_id,
            },
        )
        self.history.record_event(
            state,
            "review_started",
            {"review_kind": "plan", "target_id": plan.plan_id, "role": "verifier"},
        )
        evidence = dict(subagent_report.evidence)
        passed = subagent_report.accepted
        reason = subagent_report.reason
        self.history.record_event(
            state,
            "subagent_reported",
            {
                "subagent_type": subagent_report.spec.subagent_type,
                "accepted": subagent_report.accepted,
                "reason": subagent_report.reason,
                "recommended_action": subagent_report.recommended_action,
                "artifacts": [asdict(item) for item in subagent_report.artifacts],
            },
        )
        self.history.record_event(
            state,
            "review_completed",
            {"review_kind": "plan", "target_id": plan.plan_id, "role": "verifier", "passed": passed, "reason": reason, "evidence": evidence},
        )
        self._switch_role(state, "primary", reason="plan_review_finished")
        if not passed:
            raise PlanValidationError(reason)

    def _review_verification_result(
        self,
        state: SessionState,
        step: PlanStep,
        *,
        verification: VerificationOutcome,
        subsystem_result,
    ) -> tuple[bool, str, dict[str, Any]]:
        if step.kind not in {"respond", "reasoning"} and verification.verification_type_used != "llm_fallback":
            self.history.record_event(
                state,
                "subagent_selection_resolved",
                {
                    "purpose": "result_review",
                    "candidate_types": ["reviewer"],
                    "selection": {
                        "spawn": False,
                        "subagent_type": "none",
                        "reason": "deterministic_review_sufficient",
                        "focus": "",
                    },
                },
            )
            return True, "review_skipped_deterministic", {"skipped": True, "reason": "deterministic_review_sufficient"}
        selection = self._select_subagent_frontend(
            state,
            goal=step.goal,
            purpose="result_review",
            candidate_types=["reviewer"],
            detail_lines=[
                f"step_kind={step.kind}",
                f"verification_type={verification.verification_type_used}",
                f"verification_passed={verification.passed}",
                f"assistant_text_present={bool(subsystem_result.assistant_text.strip())}",
            ],
        )
        if not selection.spawn:
            self.history.record_event(
                state,
                "review_skipped",
                {"review_kind": "result", "target_id": step.step_id, "reason": selection.reason},
            )
            return True, "review_skipped", {"skipped": True, "reason": selection.reason}
        self._switch_role(state, "verifier", reason="result_review")
        subagent_report = self._subagents.review_result(
            state,
            step,
            verification=verification,
            subsystem_result=subsystem_result,
        )
        self.history.record_event(
            state,
            "subagent_spawned",
            {
                "subagent_type": subagent_report.spec.subagent_type,
                "purpose": subagent_report.spec.purpose,
                "allowed_tools": subagent_report.spec.allowed_tools,
                "token_budget": subagent_report.spec.token_budget,
                "target_id": step.step_id,
            },
        )
        self.history.record_event(
            state,
            "review_started",
            {"review_kind": "result", "target_id": step.step_id, "role": "verifier"},
        )
        evidence = dict(subagent_report.evidence)
        passed = subagent_report.accepted
        reason = subagent_report.reason
        self.history.record_event(
            state,
            "subagent_reported",
            {
                "subagent_type": subagent_report.spec.subagent_type,
                "accepted": subagent_report.accepted,
                "reason": subagent_report.reason,
                "recommended_action": subagent_report.recommended_action,
                "artifacts": [asdict(item) for item in subagent_report.artifacts],
            },
        )
        self.history.record_event(
            state,
            "review_completed",
            {"review_kind": "result", "target_id": step.step_id, "role": "verifier", "passed": passed, "reason": reason, "evidence": evidence},
        )
        self._switch_role(state, "primary", reason="result_review_finished")
        return passed, reason, evidence

    def _record_action_selection(self, state: SessionState, decision) -> None:
        self.history.record_event(
            state,
            "action_selected",
            {
                "selected_action": decision.action,
                "ready_step_ids": decision.ready_step_ids,
                "scores": [asdict(item) for item in decision.scores],
                "strategy": state.active_strategy.mode if state.active_strategy is not None else "conservative",
                "stop_reason": decision.stop_reason,
                "step_id": decision.step.step_id if decision.step is not None else None,
            },
        )

    def _running_background_process_ids(self, state: SessionState) -> list[str]:
        return sorted(
            process_id
            for process_id, record in state.environment.processes.items()
            if record.status == "running"
        )

    def _bind_background_process_to_step(
        self,
        state: SessionState,
        *,
        step: PlanStep,
        process_id: str,
        tool_name: str,
    ) -> None:
        record = state.environment.processes.get(process_id)
        if record is None:
            raise HistoryInvariantError(f"Background process {process_id} is missing from environment state")
        updated = replace(
            record,
            metadata={
                **dict(record.metadata),
                "step_id": step.step_id,
                "step_kind": step.kind,
                "step_title": step.title,
                "tool_name": tool_name,
            },
        )
        self.history.record_event(
            state,
            "process_polled",
            {**asdict(updated), "completed": False},
        )
        self.history.record_event(
            state,
            "subsystem_progress",
            {
                "subsystem": "scheduler",
                "step_id": step.step_id,
                "progress": f"background_bound={process_id}",
            },
        )

    def _record_background_events(
        self,
        state: SessionState,
        generated_events: list[ToolGeneratedEvent],
    ) -> None:
        for event in generated_events:
            recorded = self.history.record_event(
                state,
                event.event_type,
                event.payload,
                metadata=event.metadata,
                derived_writes=event.derived_writes,
            )
            if event.event_type in {"process_completed", "process_timed_out", "process_killed"}:
                self._extract_and_store_memory(state, recorded)

    def _resolve_background_process_completion(
        self,
        state: SessionState,
        *,
        update: BackgroundProcessUpdate,
        tool_results: list[ToolExecutionResult],
        background_tool_indexes: dict[str, int],
    ) -> BackgroundCycleResult:
        result = BackgroundCycleResult(progress_made=True)
        process_id = update.record.process_id
        if update.tool_result is not None:
            tool_index = background_tool_indexes.get(process_id)
            if tool_index is None:
                background_tool_indexes[process_id] = len(tool_results)
                tool_results.append(update.tool_result)
            else:
                tool_results[tool_index] = update.tool_result

        step_id = str(update.record.metadata.get("step_id", "")).strip()
        if not step_id or state.active_plan is None:
            return result
        plan = state.active_plan
        step = next((item for item in plan.steps if item.step_id == step_id), None)
        if step is None or step.status != "running":
            return result
        if update.tool_result is None:
            self._fail_step(
                state,
                plan,
                step,
                f"Background process {process_id} completed without a tool result",
                "BackgroundProcessResultMissing",
            )
            result.failed_steps = 1
            result.no_progress_resolved = False
            result.replan_reason = f"Background step {step.step_id} completed without a tool result"
            return result

        subsystem_result = SubsystemExecutionResult(
            subsystem_name="background_completion",
            success=update.record.status == "completed",
            tool_results=[update.tool_result],
            progress=[f"process_id={process_id}", f"status={update.record.status}"],
        )
        verification = self._verify_step(
            state,
            plan,
            step,
            self._build_verification_artifacts(
                step,
                assistant_text="",
                tool_results=[update.tool_result],
                runtime_artifacts={
                    "subsystem": subsystem_result.subsystem_name,
                    "process_id": process_id,
                    "process_status": update.record.status,
                },
            ),
        )
        if verification.passed and verification.confidence < self.config.runtime.verification_confidence_threshold:
            verification = VerificationOutcome(
                verification_passed=False,
                verification_type_used=verification.verification_type_used,
                conditions_met=list(verification.conditions_met),
                conditions_failed=[*verification.conditions_failed, "confidence_below_threshold"],
                evidence=dict(verification.evidence),
                confidence=verification.confidence,
                reason=f"{verification.reason};confidence_below_threshold",
                requires_retry=True,
                requires_replan=False,
            )
        review_passed, review_reason, review_evidence = self._review_verification_result(
            state,
            step,
            verification=verification,
            subsystem_result=subsystem_result,
        )
        if not review_passed:
            verification = VerificationOutcome(
                verification_passed=False,
                verification_type_used=verification.verification_type_used,
                conditions_met=list(verification.conditions_met),
                conditions_failed=[*verification.conditions_failed, "review_failed"],
                evidence={**dict(verification.evidence), "review": review_evidence},
                confidence=verification.confidence,
                reason=f"{verification.reason};{review_reason}",
                requires_retry=True,
                requires_replan=False,
            )
        evaluation = evaluate_verification(step, verification)
        if verification.verification_type_used != "llm_fallback" and not verification.passed and evaluation.passed:
            raise HistoryInvariantError(
                f"Evaluator attempted to override deterministic verification failure for background step {step.step_id}"
            )
        result.last_verification = verification
        if evaluation.passed:
            self._complete_step(
                state,
                plan,
                step,
                outcome=update.tool_result.tool_name,
            )
            self._refresh_project_state(state, reason=f"background_step_completed:{step.step_id}")
            self._check_consistency(state)
            result.completed_steps = 1
            result.no_progress_resolved = True
            return result

        failure = self._classify_failure_frontend(
            state,
            step=step,
            reason=f"verification:{evaluation.reason}",
        )
        active_strategy = state.active_strategy
        if active_strategy is None:
            raise HistoryInvariantError("Active strategy is missing while resolving a background step")
        updated_strategy = adapt_strategy(
            active_strategy,
            failure=failure,
            metrics=state.metrics,
            verification_failed=True,
        )
        self._set_strategy(state, updated_strategy, reason=updated_strategy.reason)
        self._fail_step(
            state,
            plan,
            step,
            evaluation.reason,
            failure.kind,
        )
        result.failed_steps = 1
        result.last_failure = failure
        result.no_progress_resolved = False
        result.replan_reason = f"Background step {step.step_id} failed verification: {evaluation.reason}"
        return result

    def _poll_background_processes(
        self,
        state: SessionState,
        *,
        tool_results: list[ToolExecutionResult],
        background_tool_indexes: dict[str, int],
    ) -> BackgroundCycleResult:
        aggregate = BackgroundCycleResult()
        for process_id in self._running_background_process_ids(state):
            update = AgentEnvironment(self.config, state).poll_background_process(process_id)
            self._record_background_events(state, update.generated_events)
            if not update.completed:
                continue
            resolved = self._resolve_background_process_completion(
                state,
                update=update,
                tool_results=tool_results,
                background_tool_indexes=background_tool_indexes,
            )
            aggregate.progress_made = aggregate.progress_made or resolved.progress_made
            aggregate.completed_steps += resolved.completed_steps
            aggregate.failed_steps += resolved.failed_steps
            aggregate.no_progress_resolved = aggregate.no_progress_resolved or resolved.no_progress_resolved
            aggregate.last_verification = resolved.last_verification or aggregate.last_verification
            aggregate.last_failure = resolved.last_failure or aggregate.last_failure
            if aggregate.replan_reason is None and resolved.replan_reason:
                aggregate.replan_reason = resolved.replan_reason
        return aggregate

    def _verify_step(
        self,
        state: SessionState,
        plan: Plan,
        step: PlanStep,
        artifacts: VerificationArtifacts,
    ) -> VerificationOutcome:
        self.history.record_event(
            state,
            "verification_started",
            {
                "step_id": step.step_id,
                "verification_type": step.verification_type,
                "required_conditions": list(step.required_conditions),
                "optional_conditions": list(step.optional_conditions),
            },
        )
        try:
            verification = self._verification.verify_step(
                runtime=self,
                state=state,
                plan=plan,
                step=step,
                artifacts=artifacts,
            )
        except VerificationError as exc:
            verification = VerificationOutcome(
                verification_passed=False,
                verification_type_used=step.verification_type,
                conditions_met=[],
                conditions_failed=["verification_engine_error"],
                evidence={"error": str(exc), "error_type": exc.__class__.__name__},
                confidence=0.0,
                reason=f"verification_engine_error:{exc}",
                requires_retry=False,
                requires_replan=True,
            )
        self._record_verification(state, step, verification)
        return verification

    def _build_verification_artifacts(
        self,
        step: PlanStep,
        *,
        tool_results: list[ToolExecutionResult],
        assistant_text: str,
        runtime_artifacts: dict[str, Any] | None = None,
    ) -> VerificationArtifacts:
        return VerificationArtifacts(
            assistant_text=assistant_text,
            tool_results=list(tool_results),
            runtime_artifacts={
                "step_id": step.step_id,
                **({} if runtime_artifacts is None else dict(runtime_artifacts)),
            },
        )

    def _preview_step_verification(
        self,
        state: SessionState,
        plan: Plan,
        step: PlanStep,
        artifacts: VerificationArtifacts,
    ) -> VerificationOutcome:
        try:
            return self._verification.verify_step(
                runtime=self,
                state=state,
                plan=plan,
                step=step,
                artifacts=artifacts,
            )
        except VerificationError as exc:
            return VerificationOutcome(
                verification_passed=False,
                verification_type_used=step.verification_type,
                conditions_met=[],
                conditions_failed=["verification_engine_error"],
                evidence={"error": str(exc), "error_type": exc.__class__.__name__},
                confidence=0.0,
                reason=f"verification_engine_error:{exc}",
                requires_retry=False,
                requires_replan=True,
            )

    def _record_verification(self, state: SessionState, step: PlanStep, verification: VerificationOutcome) -> None:
        common_payload = {
            "step_id": step.step_id,
            "verification_type_used": verification.verification_type_used,
            "conditions_met": list(verification.conditions_met),
            "conditions_failed": list(verification.conditions_failed),
            "evidence": to_jsonable(verification.evidence),
            "confidence": verification.confidence,
            "reason": verification.reason,
        }
        self.history.record_event(
            state,
            "verification_type_used",
            {"step_id": step.step_id, "verification_type_used": verification.verification_type_used},
        )
        self.history.record_event(
            state,
            "verification_completed",
            {
                **common_payload,
                "verification_passed": verification.verification_passed,
            },
        )
        if verification.passed:
            self.history.record_event(state, "verification_passed", common_payload)
            return
        self.history.record_event(
            state,
            "verification_failed",
            {
                **common_payload,
                "failure_kind": "verification_failure",
            },
        )

    def _step_running(self, plan: Plan, step_id: str) -> bool:
        for item in plan.steps:
            if item.step_id == step_id:
                return item.status == "running"
        return False

    def _run_step_subsystem(
        self,
        state: SessionState,
        step: PlanStep,
        *,
        action_counts: dict[str, int],
    ):
        self._switch_role(state, "executor", reason=f"execute_step:{step.step_id}")
        try:
            if step.kind == "respond" or step.kind == "reasoning":
                return self._reasoning_subsystem.run(self, state, step)
            if step.kind in {"read", "write"}:
                return self._file_subsystem.run(self, state, step, action_counts=action_counts)
            return self._tool_subsystem.run(self, state, step, action_counts=action_counts)
        finally:
            self._switch_role(state, "primary", reason=f"step_execution_finished:{step.step_id}")

    def _ensure_plan(
        self,
        state: SessionState,
        goal: str,
        *,
        replan_reason: str = "",
        replan_attempt: int = 0,
        force_replan: bool = False,
        required_tools: list[str] | None = None,
    ) -> Plan:
        if not force_replan and state.active_plan is not None and state.active_plan.status == "active" and state.active_plan.goal == goal:
            return state.active_plan
        update_existing = state.active_plan is not None and state.active_plan.goal == goal
        if required_tools is None:
            required_tools = self._detect_explicit_named_tools(self._goal_text(state))
        return self._planning_subsystem.run(
            self,
            state,
            goal,
            replan_reason=replan_reason,
            replan_attempt=replan_attempt,
            update_existing=update_existing,
            required_tools=required_tools,
        )

    def _generate_plan(
        self,
        state: SessionState,
        goal: str,
        *,
        update_existing: bool,
        replan_reason: str,
        replan_attempt: int = 0,
        required_tools: list[str] | None = None,
    ) -> Plan:
        self._switch_role(state, "planner", reason="generate_plan")
        planning_goal = self._operational_goal_from_task_contract(goal)
        seeded_plan = self._maybe_seed_shell_recovery_plan(
            state,
            goal=goal,
            planning_goal=planning_goal,
            update_existing=update_existing,
            required_tools=required_tools or [],
        )
        if seeded_plan is not None:
            plan = seeded_plan
            if state.active_strategy is not None:
                self._reconcile_strategy_for_plan(
                    state,
                    plan,
                    completed_step_kinds=self._completed_step_kinds(state),
                )
            event = self.history.record_event(
                state,
                "plan_updated" if update_existing else "plan_created",
                {
                    "goal": goal,
                    "plan": plan_as_payload(plan),
                    **({"reason": "seeded:shell_recovery"} if update_existing else {}),
                },
            )
            self._extract_and_store_memory(state, event)
            self._refresh_working_memory(state, reason="plan_created" if not update_existing else "plan_replanned")
            self._refresh_project_state(state, reason="plan_created" if not update_existing else "plan_replanned")
            self._check_consistency(state)
            self._switch_role(state, "primary", reason="plan_generated")
            return state.active_plan or plan
        planner_replan_guidance = ""
        if update_existing or replan_reason:
            selection = self._select_subagent_frontend(
                state,
                goal=goal,
                purpose="plan_repair",
                candidate_types=["planner"],
                detail_lines=[
                    f"update_existing={update_existing}",
                    f"replan_reason={replan_reason or '(none)'}",
                    f"has_active_plan={state.active_plan is not None}",
                ],
            )
            if selection.spawn:
                replan_report = self._subagents.replan(
                    state,
                    goal=goal,
                    current_plan=state.active_plan,
                    failure_reason=replan_reason or "explicit_replan",
                )
                self.history.record_event(
                    state,
                    "subagent_spawned",
                    {
                        "subagent_type": replan_report.spec.subagent_type,
                        "purpose": replan_report.spec.purpose,
                        "allowed_tools": replan_report.spec.allowed_tools,
                        "token_budget": replan_report.spec.token_budget,
                        "target_id": state.active_plan.plan_id if state.active_plan is not None else None,
                    },
                )
                self.history.record_event(
                    state,
                    "subagent_reported",
                    {
                        "subagent_type": replan_report.spec.subagent_type,
                        "accepted": replan_report.accepted,
                        "reason": replan_report.reason,
                        "recommended_action": replan_report.recommended_action,
                        "artifacts": [asdict(item) for item in replan_report.artifacts],
                    },
                )
                if replan_report.artifacts:
                    planner_replan_guidance = str(replan_report.artifacts[0].content.get("replan_guidance", "")).strip()
        contract = plan_contract(
            self.tools.tool_names(self.config),
            context_limit=self.config.model.context_limit,
            max_steps=self.config.planner.max_plan_steps,
        )
        prepared = self._prepare_call(
            state,
            kind="plan",
            build_prompt=lambda prompt_mode, bundle: self.prompts.build_plan_prompt(
                planning_goal,
                prompt_mode=prompt_mode,
                context_components=bundle.components,
                tools=bundle.tool_prompt_tuples,
                replan_reason="\n".join(part for part in [replan_reason, planner_replan_guidance] if part.strip()),
                replan_attempt=replan_attempt,
                max_replans=self.config.planner.max_replans,
            ),
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
            goal=planning_goal,
            for_planning=True,
        )
        plan_source = "model"
        try:
            completion = self._execute_model_call(state, prepared)
            try:
                payload = self._parse_json(completion.text, contract_name=prepared.contract.name)
            except Exception as exc:
                plan = self._maybe_recover_plan_from_structured_failure(
                    state,
                    goal=planning_goal,
                    prepared=prepared,
                    error=exc,
                    raw_response=completion.text,
                    update_existing=update_existing,
                    required_tools=required_tools or [],
                )
                if plan is None:
                    self._log_fatal_system_error(
                        state,
                        category="structured_parse_failure",
                        prepared=prepared,
                        error=exc,
                        raw_response=completion.text,
                    )
                    raise FatalSemanticEngineError(str(exc)) from exc
                plan_source = "shell_recovery"
            else:
                try:
                    plan = plan_from_payload(
                        payload,
                        available_tools=self.tools.tool_names(self.config),
                        plan_id=state.active_plan.plan_id if update_existing and state.active_plan is not None else None,
                    )
                except PlanValidationError as exc:
                    plan = self._maybe_recover_plan_from_structured_failure(
                        state,
                        goal=planning_goal,
                        prepared=prepared,
                        error=exc,
                        raw_response=completion.text,
                        update_existing=update_existing,
                        required_tools=required_tools or [],
                    )
                    if plan is None:
                        self._log_fatal_system_error(
                            state,
                            category="structured_validation_failure",
                            prepared=prepared,
                            error=exc,
                            raw_response=completion.text,
                            details={"payload": payload},
                        )
                        raise FatalSemanticEngineError(str(exc)) from exc
                    plan_source = "shell_recovery"
            plan.goal = planning_goal
            plan = self._enforce_required_tools_in_plan(
                state,
                plan,
                goal=planning_goal,
                required_tools=required_tools or [],
                update_existing=update_existing,
            )
            if state.active_strategy is not None:
                try:
                    self._reconcile_strategy_for_plan(
                        state,
                        plan,
                        completed_step_kinds=self._completed_step_kinds(state),
                    )
                except StrategyValidationError as exc:
                    recovered = self._maybe_recover_plan_from_structured_failure(
                        state,
                        goal=planning_goal,
                        prepared=prepared,
                        error=exc,
                        raw_response=completion.text,
                        update_existing=update_existing,
                        required_tools=required_tools or [],
                    )
                    if recovered is None:
                        raise
                    plan = recovered
                    plan.goal = planning_goal
                    self._reconcile_strategy_for_plan(
                        state,
                        plan,
                        completed_step_kinds=self._completed_step_kinds(state),
                    )
                    plan_source = "shell_recovery"
        except StrategyValidationError as exc:
            self.history.record_event(
                state,
                "error",
                {"operation": "strategy_validation", "error": str(exc), "error_type": exc.__class__.__name__},
            )
            raise PlanValidationError(str(exc)) from exc
        if len(plan.steps) > self.config.planner.max_plan_steps:
            raise PlanValidationError(f"Planner returned {len(plan.steps)} steps; max is {self.config.planner.max_plan_steps}")
        self._review_plan(state, plan)
        if update_existing:
            event = self.history.record_event(
                state,
                "plan_updated",
                {"plan": plan_as_payload(plan), "reason": replan_reason or f"replanned:{plan_source}"},
            )
        else:
            event = self.history.record_event(
                state,
                "plan_created",
                {"goal": goal, "plan": plan_as_payload(plan)},
            )
        self._extract_and_store_memory(state, event)
        self._refresh_working_memory(state, reason="plan_created" if not update_existing else "plan_replanned")
        self._refresh_project_state(state, reason="plan_created" if not update_existing else "plan_replanned")
        self._check_consistency(state)
        self._switch_role(state, "primary", reason="plan_generated")
        return state.active_plan or plan

    def _maybe_seed_shell_recovery_plan(
        self,
        state: SessionState,
        *,
        goal: str,
        planning_goal: str,
        update_existing: bool,
        required_tools: list[str],
    ) -> Plan | None:
        contract = self._task_contract_for_goal(state, goal)
        if not isinstance(contract, dict) or contract.get("task_kind") != "local_repo_code_fix":
            return None
        strategy = state.active_strategy
        if strategy is None or strategy.task_profile not in {"coding", "file_edit", "multi_step"}:
            return None
        if "shell_command" not in self.tools.tool_names(self.config):
            return None
        normalized_required = [tool for tool in required_tools if tool]
        if normalized_required and any(tool != "shell_command" for tool in normalized_required):
            return None
        plan = create_shell_recovery_plan(planning_goal)
        for step in plan.steps:
            if step.kind != "write" or step.expected_tool != "shell_command":
                continue
            if any(str(check.get("check_type", "")).strip() == "tool_files_changed" for check in step.verification_checks):
                continue
            step.verification_checks.append({"name": "tool_files_changed", "check_type": "tool_files_changed"})
            step.required_conditions.append("tool_files_changed")
        if update_existing and state.active_plan is not None:
            plan.plan_id = state.active_plan.plan_id
        self.history.record_event(
            state,
            "plan_repaired",
            {
                "reason": "task_contract_shell_recovery_seed",
                "required_tools": normalized_required,
                "repair": "shell_recovery_plan",
                "error": "seeded benchmark-local code-fix plan",
                "error_type": "BenchmarkTaskContract",
                "original_plan_id": state.active_plan.plan_id if state.active_plan is not None else "",
                "update_existing": update_existing,
                "contract_name": "task_contract",
                "raw_response_preview": "",
            },
        )
        return plan

    def _maybe_recover_plan_from_structured_failure(
        self,
        state: SessionState,
        *,
        goal: str,
        prepared: PreparedCall,
        error: Exception,
        raw_response: str,
        update_existing: bool,
        required_tools: list[str],
    ) -> Plan | None:
        strategy = state.active_strategy
        if strategy is None or strategy.task_profile not in {"coding", "file_edit", "multi_step"}:
            return None
        if "shell_command" not in self.tools.tool_names(self.config):
            return None
        normalized_required = [tool for tool in required_tools if tool]
        if normalized_required and any(tool != "shell_command" for tool in normalized_required):
            return None
        recovered = create_shell_recovery_plan(goal)
        if update_existing and state.active_plan is not None:
            recovered.plan_id = state.active_plan.plan_id
        self.history.record_event(
            state,
            "plan_repaired",
            {
                "reason": "planner_structured_failure_shell_recovery",
                "required_tools": normalized_required,
                "repair": "shell_recovery_plan",
                "error": str(error),
                "error_type": error.__class__.__name__,
                "original_plan_id": state.active_plan.plan_id if state.active_plan is not None else "",
                "update_existing": update_existing,
                "contract_name": prepared.contract.name,
                "raw_response_preview": raw_response[:400],
            },
        )
        return recovered

    def _enforce_required_tools_in_plan(
        self,
        state: SessionState,
        plan: Plan,
        *,
        goal: str,
        required_tools: list[str],
        update_existing: bool,
    ) -> Plan:
        normalized = [tool for tool in required_tools if tool]
        if not normalized:
            return plan
        present = {step.expected_tool for step in plan.steps if step.expected_tool}
        missing = [tool for tool in normalized if tool not in present]
        if not missing:
            return plan
        if len(normalized) == 1:
            self.history.record_event(
                state,
                "plan_repaired",
                {
                    "reason": "explicit_required_tool_missing",
                    "required_tools": normalized,
                    "repair": "direct_tool_plan",
                    "original_plan_id": plan.plan_id,
                    "update_existing": update_existing,
                },
            )
            return create_direct_tool_plan(goal, normalized[0])
        raise PlanValidationError(f"Plan omitted explicitly required tools: {', '.join(missing)}")

    def _refresh_working_memory(self, state: SessionState, *, reason: str) -> None:
        working_memory = build_working_memory(state)
        self.history.record_event(state, "working_memory_updated", {"working_memory": asdict(working_memory), "reason": reason})

    def _extract_and_store_memory(self, state: SessionState, source_event) -> None:
        items, rejection_reason = extract_from_event(self.config, source_event)
        preview = stable_json_dumps(source_event.payload)[:200]
        if rejection_reason:
            self.history.record_event(
                state,
                "memory_flagged",
                {
                    "source_event_id": source_event.id,
                    "reason": rejection_reason,
                    "trust_level": source_event.metadata.get("trust_level", "derived"),
                    "content_preview": preview,
                },
            )
            self.history.record_event(
                state,
                "memory_rejected",
                {
                    "source_event_id": source_event.id,
                    "reason": rejection_reason,
                    "trust_level": source_event.metadata.get("trust_level", "derived"),
                    "content_preview": preview,
                },
            )
            return
        for item in items:
            if any(existing.memory_kind == item.memory_kind and existing.content == item.content for existing in state.semantic_memory):
                continue
            self.history.record_event(state, "memory_extracted", {"memory": asdict(item), "source_event_id": source_event.id})
            self.history.record_event(state, "memory_stored", {"memory": asdict(item)})

    def _start_step(self, state: SessionState, plan: Plan, step: PlanStep) -> Plan:
        plan = mark_step_in_progress(plan, step.step_id)
        self.history.record_event(
            state,
            "step_started",
            {"plan_id": plan.plan_id, "step_id": step.step_id, "step_title": step.title},
        )
        plan_event = self.history.record_event(
            state,
            "plan_updated",
            {"plan": plan_as_payload(plan), "reason": "step_started"},
        )
        self._extract_and_store_memory(state, plan_event)
        self._refresh_working_memory(state, reason="step_started")
        self._refresh_project_state(state, reason=f"step_started:{step.step_id}")
        return plan

    def _complete_step(self, state: SessionState, plan: Plan, step: PlanStep, *, outcome: str) -> Plan:
        plan = mark_step_completed(plan, step.step_id)
        self.history.record_event(
            state,
            "step_executed",
            {"plan_id": plan.plan_id, "step_id": step.step_id, "step_title": step.title, "outcome": outcome},
        )
        step_completed_event = self.history.record_event(
            state,
            "step_completed",
            {"plan_id": plan.plan_id, "step_id": step.step_id, "step_title": step.title, "outcome": outcome},
        )
        self._extract_and_store_memory(state, step_completed_event)
        plan_event = self.history.record_event(
            state,
            "plan_updated",
            {"plan": plan_as_payload(plan), "reason": "step_completed"},
        )
        self._extract_and_store_memory(state, plan_event)
        if plan.status == "completed":
            self.history.record_event(state, "plan_completed", {"plan_id": plan.plan_id, "status": plan.status})
        self._refresh_working_memory(state, reason="step_completed")
        self._refresh_project_state(state, reason=f"step_completed:{step.step_id}")
        return plan

    def _fail_step(self, state: SessionState, plan: Plan, step: PlanStep, error: str, error_type: str) -> Plan:
        current = next((item for item in plan.steps if item.step_id == step.step_id), None)
        if current is not None and current.status == "failed":
            return plan
        plan = mark_step_failed(plan, step.step_id)
        self.history.record_event(
            state,
            "step_failed",
            {
                "plan_id": plan.plan_id,
                "step_id": step.step_id,
                "step_title": step.title,
                "error": error,
                "error_type": error_type,
            },
        )
        plan_event = self.history.record_event(
            state,
            "plan_updated",
            {"plan": plan_as_payload(plan), "reason": f"step_failed:{step.step_id}"},
        )
        self._extract_and_store_memory(state, plan_event)
        self._refresh_working_memory(state, reason="step_failed")
        self._refresh_project_state(state, reason=f"step_failed:{step.step_id}")
        return plan

    def _check_consistency(self, state: SessionState) -> None:
        expected_working_memory = build_working_memory(state)
        rebuilt = self.history.rebuild_from_history(state.session_id, write_projections=False, prefer_checkpoint=False)
        working_memory_ok = self._working_memory_signature(state.working_memory) == self._working_memory_signature(expected_working_memory)
        semantic_memory_ok = self._semantic_signature(state) == self._semantic_signature(rebuilt)
        environment_ok = self._environment_signature(state) == self._environment_signature(rebuilt)
        project_state_ok = self._project_state_signature(state.project_state) == self._project_state_signature(build_project_state(state))
        if working_memory_ok and semantic_memory_ok and environment_ok and project_state_ok:
            self.history.record_event(
                state,
                "consistency_checked",
                {"working_memory_ok": True, "semantic_memory_ok": True, "environment_ok": True, "project_state_ok": True, "recovered": False},
            )
            return
        components = []
        if not working_memory_ok:
            components.append("working_memory")
        if not semantic_memory_ok:
            components.append("semantic_memory")
        if not environment_ok:
            components.append("environment")
        if not project_state_ok:
            components.append("project_state")
        component = ",".join(components)
        self.history.record_event(
            state,
            "consistency_failed",
            {"component": component, "reason": "State diverged from rebuild-from-history"},
        )
        self.history.record_event(
            state,
            "recovery_triggered",
            {"reason": f"consistency_failed:{component}", "source": "consistency_checker", "event_count": state.event_count},
        )
        rebuilt = self.history.rebuild_from_history(state.session_id, write_projections=False, prefer_checkpoint=False)
        self._sync_state(state, rebuilt)
        self.history.record_event(state, "state_rebuilt", {"session_id": state.session_id, "event_count": rebuilt.event_count})
        self.history.record_event(
            state,
            "consistency_checked",
            {"working_memory_ok": working_memory_ok, "semantic_memory_ok": semantic_memory_ok, "environment_ok": environment_ok, "project_state_ok": project_state_ok, "recovered": True},
        )

    def _working_memory_signature(self, working_memory) -> dict[str, Any]:
        payload = asdict(working_memory)
        payload.pop("updated_at", None)
        return payload

    def _semantic_signature(self, state: SessionState) -> dict[str, Any]:
        return {
            "memory": [asdict(item) for item in state.semantic_memory],
            "entities": {key: asdict(value) for key, value in state.semantic_entities.items()},
            "relationships": [asdict(item) for item in state.semantic_relationships],
            "facts": [asdict(item) for item in state.semantic_facts],
            "procedural": [asdict(item) for item in state.procedural_patterns],
        }

    def _project_state_signature(self, project_state) -> dict[str, Any]:
        payload = asdict(project_state)
        payload.pop("last_updated", None)
        return payload

    def _environment_signature(self, state: SessionState) -> dict[str, Any]:
        payload = asdict(state.environment)
        payload.pop("last_updated", None)
        workspace = payload.get("workspace", {})
        workspace.pop("last_snapshot_at", None)
        shell = payload.get("shell", {})
        shell.pop("updated_at", None)
        for process in payload.get("processes", {}).values():
            process.pop("started_at", None)
            process.pop("ended_at", None)
        return payload

    def _check_drift(self, state: SessionState, *, failed_steps: int, completed_steps: int) -> None:
        if failed_steps < 2:
            return
        self.history.record_event(
            state,
            "drift_detected",
            {
                "reason": "repeated_step_failures",
                "event_count": state.event_count,
                "failed_steps": failed_steps,
                "completed_steps": completed_steps,
            },
        )
        self.history.record_event(
            state,
            "recovery_triggered",
            {"reason": "drift_detected", "source": "reasoning_loop", "event_count": state.event_count},
        )
        rebuilt = self.history.rebuild_from_history(state.session_id, write_projections=False, prefer_checkpoint=False)
        self._sync_state(state, rebuilt)
        self.history.record_event(state, "state_rebuilt", {"session_id": state.session_id, "event_count": rebuilt.event_count})

    def _sync_state(self, target: SessionState, source: SessionState) -> None:
        for field in fields(SessionState):
            setattr(target, field.name, copy.deepcopy(getattr(source, field.name)))

    def _build_context_bundle(
        self,
        state: SessionState,
        *,
        goal: str,
        kind: str,
        prompt_mode: str,
        for_planning: bool = False,
    ) -> ContextBundle:
        try:
            bundle = build_context(
                self.config,
                state,
                self._get_selection_counter(),
                goal=goal,
                call_kind=kind,
                for_planning=for_planning,
                history_events=self.history.read_history(state.session_id),
                available_tools=self.tools.prompt_tuples(self.config),
            )
        except SemanticBackendProtocolError as exc:
            self._log_fatal_system_error(
                state=state,
                category="semantic_retrieval_protocol_violation",
                prepared=None,
                error=exc,
                operation_name="semantic_retrieval",
                details={
                    "kind": kind,
                    "goal": goal,
                    "prompt_mode": prompt_mode,
                    "for_planning": for_planning,
                    "retrieval_backend": self.config.retrieval.backend,
                },
            )
            raise FatalSemanticEngineError(str(exc)) from exc
        contextual_signal_count = sum(
            1
            for count in (
                len(bundle.history_messages),
                len(bundle.semantic_items),
                len(bundle.relevant_environment_files),
                len(bundle.guidance_sources),
            )
            if count
        )
        current_step = None
        if state.active_plan is not None and state.active_plan.current_step_id:
            current_step = next(
                (item for item in state.active_plan.steps if item.step_id == state.active_plan.current_step_id),
                None,
            )
        retrieval_focus_text = ""
        if kind == "subagent_selection":
            self.history.record_event(
                state,
                "subagent_selection_resolved",
                {
                    "purpose": "context_retrieval_focus",
                    "candidate_types": ["retriever"],
                    "selection": {
                        "spawn": False,
                        "subagent_type": "none",
                        "reason": "selection_prompt_recursion_guard",
                        "focus": "",
                    },
                },
            )
        elif (
            state.active_plan is not None
            and state.active_plan.fallback_strategy.startswith("If a shell-based recovery step fails")
            and current_step is not None
            and current_step.expected_tool == "shell_command"
        ):
            self.history.record_event(
                state,
                "subagent_selection_resolved",
                {
                    "purpose": "context_retrieval_focus",
                    "candidate_types": ["retriever"],
                    "selection": {
                        "spawn": False,
                        "subagent_type": "none",
                        "reason": "shell_recovery_context_direct",
                        "focus": "",
                    },
                },
            )
        elif contextual_signal_count <= 1:
            self.history.record_event(
                state,
                "subagent_selection_resolved",
                {
                    "purpose": "context_retrieval_focus",
                    "candidate_types": ["retriever"],
                    "selection": {
                        "spawn": False,
                        "subagent_type": "none",
                        "reason": "context_already_narrow",
                        "focus": "",
                    },
                },
            )
        else:
            selection = self._select_subagent_frontend(
                state,
                goal=goal,
                purpose="context_retrieval_focus",
                candidate_types=["retriever"],
                detail_lines=[
                    f"call_kind={kind}",
                    f"history_messages={len(bundle.history_messages)}",
                    f"semantic_items={len(bundle.semantic_items)}",
                    f"environment_files={len(bundle.relevant_environment_files)}",
                    f"guidance_items={len(bundle.guidance_sources)}",
                ],
            )
            if selection.spawn:
                retriever_report = self._subagents.retrieve_context(state, goal=goal, bundle=bundle)
                self.history.record_event(
                    state,
                    "subagent_spawned",
                    {
                        "subagent_type": retriever_report.spec.subagent_type,
                        "purpose": retriever_report.spec.purpose,
                        "allowed_tools": retriever_report.spec.allowed_tools,
                        "token_budget": retriever_report.spec.token_budget,
                        "target_id": state.active_plan.current_step_id if state.active_plan is not None else None,
                    },
                )
                self.history.record_event(
                    state,
                    "subagent_reported",
                    {
                        "subagent_type": retriever_report.spec.subagent_type,
                        "accepted": retriever_report.accepted,
                        "reason": retriever_report.reason,
                        "recommended_action": retriever_report.recommended_action,
                        "artifacts": [asdict(item) for item in retriever_report.artifacts],
                    },
                )
                retrieval_focus_text = ""
                if retriever_report.artifacts:
                    retrieval_focus_text = str(retriever_report.artifacts[0].content.get("focus_summary", "")).strip()
                if retrieval_focus_text:
                    focus_component = PromptComponent(
                        name="retrieval_focus",
                        category="retrieval_focus",
                        text=f"Retriever focus:\n{retrieval_focus_text}\n\n",
                    )
                    if all(component.name != "retrieval_focus" for component in bundle.components):
                        bundle.components.insert(0, focus_component)
        self.history.record_event(
            state,
            "notes_selected",
            {
                "included_note_ids": bundle.note_ids,
                "omitted_note_ids": bundle.omitted_note_ids,
                "tokens": bundle.note_tokens,
                "exact": bundle.note_tokens_exact,
            },
        )
        self.history.record_event(
            state,
            "memory_retrieved",
            {
                "query": goal,
                "memory_ids": [item.memory_id for item in bundle.semantic_items],
                "count": len(bundle.semantic_items),
            },
        )
        self.history.record_event(
            state,
            "context_built",
            {
                "goal": goal,
                "kind": kind,
                "prompt_mode": prompt_mode,
                "history_message_count": len(bundle.history_messages),
                "note_ids": bundle.note_ids,
                "semantic_memory_ids": [item.memory_id for item in bundle.semantic_items],
                "environment_summary": bundle.environment_text,
                "guidance_sources": bundle.guidance_sources,
                "selected_skill_ids": bundle.selected_skill_ids,
                "exposed_tool_names": bundle.exposed_tool_names,
                "retrieval_mode": bundle.retrieval_mode,
                "retrieval_degraded": bundle.retrieval_degraded,
                "retriever_focus": retrieval_focus_text,
                "call_budget": asdict(self._call_budget(kind)),
                "relevant_environment_files": [
                    item.item_id
                    for item in bundle.selection_trace
                    if item.item_type == "environment_file" and item.selected
                ],
                "plan_id": state.active_plan.plan_id if state.active_plan is not None else None,
                "selection_trace": [asdict(item) for item in bundle.selection_trace],
            },
        )
        return bundle

    def _compact_prompt_message(self, message: Message) -> Message:
        if message.role != "tool":
            return message
        metadata = message.metadata or {}
        output = metadata.get("output") if isinstance(metadata, dict) else None
        if not isinstance(output, dict):
            return message
        if message.name in {"read_text", "read_file"}:
            path = str(output.get("source_ref") or output.get("path") or "")
            text = str(output.get("text") or output.get("content") or "")
            summary = f"{message.name} result for {Path(path).name or path}:\n{text}"
        elif message.name in {"edit_text", "write_file"}:
            path = str(output.get("path") or output.get("target_path") or "")
            changed = output.get("changed")
            summary = f"{message.name} result for {Path(path).name or path}: changed={changed}"
        elif message.name == "run_tests":
            exit_code = output.get("exit_code")
            stdout = str(output.get("stdout") or "")[:240]
            stderr = str(output.get("stderr") or "")[:240]
            summary = f"run_tests exit_code={exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        elif message.name == "calculator":
            summary = f"calculator result: {output.get('result')}"
        else:
            summary = message.content
        return Message(
            role=message.role,
            content=summary,
            name=message.name,
            created_at=message.created_at,
            metadata=message.metadata,
        )

    def _goal_text(self, state: SessionState) -> str:
        if state.expanded_task is not None:
            return state.expanded_task.expanded_goal
        if state.active_plan is not None:
            return state.active_plan.goal
        for message in reversed(state.messages):
            if message.role == "user":
                return message.content
        return ""

    def _decide(self, state: SessionState) -> tuple[ToolDecision, BudgetReport]:
        if self._should_use_expected_tool_input_call(state):
            decision, report = self._decide_expected_tool_input(state)
            if decision is not None:
                return decision, report
        contract = tool_decision_contract(self.tools.tool_names(self.config))
        prepared = self._prepare_call(
            state,
            kind="decision",
            build_prompt=lambda prompt_mode, bundle: self.prompts.build_decision_prompt(
                bundle.history_messages,
                bundle.tool_prompt_tuples,
                prompt_mode=prompt_mode,
                context_components=bundle.components,
            ),
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
        )
        _completion, decision = self._execute_structured_call(
            state,
            prepared,
            validator=self._coerce_decision,
            validation_error_types=(RuntimeError,),
        )
        self.history.record_event(
            state,
            "decision_parsed",
            {"decision": asdict(decision), "prompt_mode": prepared.prompt_mode, "source": "model"},
        )
        return decision, prepared.report

    def _should_use_expected_tool_input_call(self, state: SessionState) -> bool:
        # Structural choice: when the active step already names a known
        # tool with a stable input schema, we can ask the model for a typed
        # tool_input payload via the dedicated contract instead of going
        # through the general decision contract. This is a structural routing
        # decision, not a profile- or vocabulary-based bypass.
        if getattr(self.client, "is_scripted_benchmark_client", False):
            return False
        step = None
        if state.active_plan is not None and state.active_plan.current_step_id:
            step = next((item for item in state.active_plan.steps if item.step_id == state.active_plan.current_step_id), None)
        return bool(step is not None and step.expected_tool in {"edit_text", "write_file", "shell_command", "run_tests"})

    def _decide_expected_tool_input(self, state: SessionState) -> tuple[ToolDecision | None, BudgetReport]:
        step = None
        if state.active_plan is not None and state.active_plan.current_step_id:
            step = next((item for item in state.active_plan.steps if item.step_id == state.active_plan.current_step_id), None)
        if step is None or not step.expected_tool:
            return None, self._empty_budget_report()
        tool = self.tools.get(step.expected_tool)
        contract = tool_input_contract(step.expected_tool, tool.input_schema)
        prepared = self._prepare_call(
            state,
            kind="tool_input",
            build_prompt=lambda prompt_mode, bundle: self.prompts.build_tool_input_prompt(
                bundle.history_messages,
                tool_name=step.expected_tool or "",
                prompt_mode=prompt_mode,
                context_components=bundle.components,
            ),
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
        )
        try:
            _completion, payload = self._execute_structured_call(state, prepared)
            payload = self._normalize_expected_tool_input(state, step, payload)
            validated_input = tool.validate(payload)
        except FatalSemanticEngineError:
            raise
        except Exception as exc:
            self.history.record_event(
                state,
                "error",
                {"operation": "tool_input", "tool_name": step.expected_tool, "error": str(exc), "error_type": exc.__class__.__name__},
            )
            return None, prepared.report
        decision = ToolDecision(action="call_tool", response="", tool_name=step.expected_tool, tool_input=validated_input)
        self.history.record_event(
            state,
            "decision_parsed",
            {"decision": asdict(decision), "prompt_mode": prepared.prompt_mode, "source": "profile_expected_tool_input"},
        )
        return decision, prepared.report

    def _normalize_expected_tool_input(self, state: SessionState, step: PlanStep, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        if step.expected_tool in {"edit_text", "write_file", "read_text", "read_file"}:
            expected_path = self._extract_path_argument(step.input_text or step.goal, prefer_last=step.expected_tool == "write_file")
            candidate = normalized.get("path")
            if expected_path and (not isinstance(candidate, str) or not candidate.strip() or Path(candidate).name == Path(expected_path).name):
                normalized["path"] = expected_path
        elif step.expected_tool == "shell_command":
            synthesized = self._default_shell_command_for_step(state, step)
            candidate = str(normalized.get("command", "") or "").strip()
            if step.kind == "read" and synthesized:
                normalized["command"] = synthesized
            elif candidate in {"", "bash", "sh", "python", "python3"} and synthesized:
                normalized["command"] = synthesized
            normalized["background"] = False if step.kind == "read" else bool(normalized.get("background", False))
        return normalized

    def _default_shell_command_for_step(self, state: SessionState, step: PlanStep) -> str | None:
        if step.kind != "read":
            return None
        terms = self._shell_search_terms(state, step)
        if not terms:
            return None
        pattern = "|".join(terms[:6])
        quoted_terms = " ".join(shlex.quote(term) for term in terms[:6])
        return (
            f"printf 'search_terms: {quoted_terms}\\n'; "
            f"rg -n {shlex.quote(pattern)} . || true"
        )

    def _shell_search_terms(self, state: SessionState, step: PlanStep) -> list[str]:
        for message in reversed(state.messages):
            if message.role == "user":
                texts = [self._operational_goal_from_task_contract(message.content), step.input_text, step.goal, step.title]
                break
        else:
            texts = [step.input_text, step.goal, step.title]
        if state.active_plan is not None:
            texts.append(state.active_plan.goal)
        combined = "\n".join(part for part in texts if part)
        candidates: list[str] = []
        for match in re.findall(r"(tests/[^\s:]+::[^\s]+|[A-Za-z_][A-Za-z0-9_]{2,})", combined):
            token = match.strip()
            lowered = token.lower()
            if lowered in {
                "task",
                "contract",
                "problem",
                "statement",
                "known",
                "failing",
                "tests",
                "hints",
                "benchmark",
                "issue",
                "inspect",
                "inspection",
                "failing_area",
                "patched_code",
                "result_report",
                "area",
                "locate",
                "symbol",
                "symbols",
                "current",
                "step",
                "verify",
                "verification",
                "read",
                "write",
                "respond",
                "tool",
                "repo",
                "task_kind",
                "request_completeness",
                "requires_code_changes",
                "requires_verification",
                "prefer_task_expansion",
                "local_repo_code_fix",
                "complete",
                "true",
                "false",
                "local",
                "code",
                "fix",
                "path",
                "paths",
                "goal",
                "print",
                "before",
                "editing",
                "most",
                "relevant",
                "source",
                "files",
                "file",
                "named",
                "then",
                "use",
                "commands",
                "only",
                "the",
                "and",
            }:
                continue
            if token not in candidates:
                candidates.append(token)

        def _score(token: str) -> tuple[int, int]:
            score = 0
            if "/" in token or "::" in token:
                score += 4
            if "_" in token:
                score += 3
            if any(char.isupper() for char in token):
                score += 2
            if token.islower():
                score -= 1
            return (score, -candidates.index(token))

        return sorted(candidates, key=_score, reverse=True)

    def _validate_generation_units_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        output_class = str(payload.get("output_class", "")).strip()
        if output_class not in {"bounded_structured", "schema_bounded", "open_ended"}:
            raise ValueError("generation output_class must be bounded_structured, schema_bounded, or open_ended")
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise ValueError("generation decomposition reason must not be empty")
        raw_units = payload.get("units")
        if not isinstance(raw_units, list) or not raw_units:
            raise ValueError("generation decomposition must contain at least one unit")
        units: list[dict[str, str]] = []
        for raw in raw_units:
            if not isinstance(raw, dict):
                raise ValueError("generation unit must be an object")
            unit_id = str(raw.get("unit_id", "")).strip()
            title = str(raw.get("title", "")).strip()
            instruction = str(raw.get("instruction", "")).strip()
            if not unit_id or not title or not instruction:
                raise ValueError("generation unit fields must not be empty")
            units.append({"unit_id": unit_id, "title": title, "instruction": instruction})
        return {"output_class": output_class, "reason": reason, "units": units}

    def _validate_overflow_recovery_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise ValueError("overflow recovery reason must not be empty")
        raw_units = payload.get("next_units")
        if not isinstance(raw_units, list):
            raise ValueError("overflow recovery next_units must be a list")
        units: list[dict[str, str]] = []
        for raw in raw_units:
            if not isinstance(raw, dict):
                raise ValueError("overflow recovery unit must be an object")
            unit_id = str(raw.get("unit_id", "")).strip()
            title = str(raw.get("title", "")).strip()
            instruction = str(raw.get("instruction", "")).strip()
            if not unit_id or not title or not instruction:
                raise ValueError("overflow recovery unit fields must not be empty")
            units.append({"unit_id": unit_id, "title": title, "instruction": instruction})
        return {
            "keep_partial": bool(payload.get("keep_partial")),
            "reason": reason,
            "next_units": units,
        }

    def _plan_answer_generation_units(self, state: SessionState) -> tuple[dict[str, Any], BudgetReport]:
        contract = generation_decomposition_contract()
        current_step = None
        if state.active_plan is not None and state.active_plan.current_step_id:
            current_step = next((item for item in state.active_plan.steps if item.step_id == state.active_plan.current_step_id), None)
        detail_lines = []
        if current_step is not None:
            detail_lines.append(f"step_title={current_step.title}")
            detail_lines.append(f"success_criteria={current_step.success_criteria}")
            detail_lines.append(f"expected_output={current_step.expected_output}")

        def _build(prompt_mode: str, bundle: ContextBundle) -> PromptAssembly:
            components = [
                PromptComponent(
                    name="generation_goal",
                    category="current_user",
                    text=f"Goal:\n{self._goal_text(state)}\n\n",
                ),
                *bundle.components,
                PromptComponent(
                    name="generation_instruction",
                    category="instruction",
                    text=(
                        "Plan answer generation units.\n"
                        "Classify the output as bounded_structured, schema_bounded, or open_ended.\n"
                        "For open_ended outputs, split the response into the smallest coherent semantic units that can be generated safely.\n"
                        "Use one unit when a single response is sufficient.\n"
                        f"Details:\n{chr(10).join(detail_lines)}\n"
                    ),
                ),
            ]
            return self.prompts._assemble("generation_decomposition", prompt_mode, components)

        prepared = self._prepare_call(
            state,
            kind="generation_decomposition",
            build_prompt=_build,
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
            goal=self._goal_text(state),
        )
        _completion, payload = self._execute_structured_call(
            state,
            prepared,
            validator=self._validate_generation_units_payload,
            validation_error_types=(ValueError,),
        )
        self.history.record_event(
            state,
            "output_decomposition_planned",
            {
                "output_class": payload["output_class"],
                "reason": payload["reason"],
                "units": payload["units"],
            },
        )
        return payload, prepared.report

    def _generate_answer_unit(
        self,
        state: SessionState,
        *,
        contract: ContractSpec,
        unit: dict[str, str],
    ) -> tuple[str, BudgetReport, bool]:
        prepared = self._prepare_call(
            state,
            kind="answer",
            build_prompt=lambda prompt_mode, bundle: self.prompts._assemble(
                "answer",
                prompt_mode,
                [
                    *self.prompts.build_answer_prompt(
                        bundle.history_messages,
                        prompt_mode=prompt_mode,
                        context_components=bundle.components,
                    ).components,
                    PromptComponent(
                        name="answer_unit",
                        category="instruction",
                        text=(
                            f"Answer unit: {unit['title']}\n"
                            f"Unit id: {unit['unit_id']}\n"
                            f"Instruction:\n{unit['instruction']}\n"
                        ),
                    ),
                ],
            ),
            contract=contract,
            prompt_modes=self._interactive_prompt_modes(),
            goal=self._goal_text(state),
        )
        completion = self._execute_model_call(state, prepared)
        overflowed = bool(
            completion.completion_tokens is not None
            and completion.completion_tokens >= prepared.report.reserved_response_tokens
        )
        self.history.record_event(
            state,
            "output_unit_generated",
            {
                "unit": unit,
                "overflowed": overflowed,
                "text": completion.text.strip(),
            },
        )
        return completion.text.strip(), prepared.report, overflowed

    def _generate_direct_response_once(self, state: SessionState) -> tuple[str, BudgetReport]:
        contract = plain_text_contract()
        prepared = self._prepare_call(
            state,
            kind="answer",
            build_prompt=lambda prompt_mode, bundle: self.prompts.build_answer_prompt(
                state.messages,
                prompt_mode=prompt_mode,
                context_components=bundle.components,
            ),
            contract=contract,
            prompt_modes=self._interactive_prompt_modes(),
            goal=self._goal_text(state),
        )
        completion = self._execute_model_call(state, prepared)
        self.history.record_event(
            state,
            "output_unit_generated",
            {
                "unit": {"unit_id": "direct_response", "title": "Direct response", "instruction": "Answer directly."},
                "overflowed": False,
                "text": completion.text.strip(),
                "source": "semantic_direct_response",
            },
        )
        return completion.text.strip(), prepared.report

    def _recover_overflow_unit(
        self,
        state: SessionState,
        *,
        unit: dict[str, str],
        partial_text: str,
    ) -> tuple[dict[str, Any], BudgetReport]:
        contract = overflow_recovery_contract()

        def _build(prompt_mode: str, bundle: ContextBundle) -> PromptAssembly:
            components = [
                PromptComponent(
                    name="overflow_unit",
                    category="current_user",
                    text=(
                        f"Goal:\n{self._goal_text(state)}\n\n"
                        f"Overflowed unit:\n{stable_json_dumps(unit)}\n\n"
                        f"Partial text:\n{partial_text}\n\n"
                    ),
                ),
                *bundle.components,
                PromptComponent(
                    name="overflow_instruction",
                    category="instruction",
                    text=(
                        "The previous generation unit overflowed its budget.\n"
                        "Decide whether the partial text is safe to keep, and if needed split the remaining work into smaller semantic units.\n"
                        "Do not ask to continue the same text blindly.\n"
                    ),
                ),
            ]
            return self.prompts._assemble("overflow_recovery", prompt_mode, components)

        prepared = self._prepare_call(
            state,
            kind="overflow_recovery",
            build_prompt=_build,
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
            goal=self._goal_text(state),
        )
        _completion, payload = self._execute_structured_call(
            state,
            prepared,
            validator=self._validate_overflow_recovery_payload,
            validation_error_types=(ValueError,),
        )
        self.history.record_event(
            state,
            "output_overflow_recovery_planned",
            {
                "unit": unit,
                "keep_partial": payload["keep_partial"],
                "reason": payload["reason"],
                "next_units": payload["next_units"],
            },
        )
        return payload, prepared.report

    def _answer(self, state: SessionState) -> tuple[str, BudgetReport]:
        contract = plain_text_contract()
        derived_answer = self._deterministic_answer(state)
        if self._should_force_not_done_answer(state, derived_answer=derived_answer):
            report = self._empty_budget_report()
            self.history.record_event(state, "answer_derived", {"answer": "not done", "source": "deterministic_failure_guard"})
            return "not done", report
        if derived_answer is not None and self._can_finalize_exact_reply(state):
            report = self._empty_budget_report()
            self.history.record_event(state, "answer_derived", {"answer": derived_answer, "source": "deterministic_finalizer"})
            return derived_answer, report
        latest_decision = state.latest_decision
        if latest_decision is not None and latest_decision.direct_response and self._can_finalize_exact_reply(state):
            return self._generate_direct_response_once(state)
        unit_plan, plan_report = self._plan_answer_generation_units(state)
        reports = [plan_report]
        unit_texts: list[str] = []
        pending_units = list(unit_plan["units"])
        while pending_units:
            unit = pending_units.pop(0)
            text, unit_report, overflowed = self._generate_answer_unit(state, contract=contract, unit=unit)
            reports.append(unit_report)
            keep_partial = True
            if overflowed:
                recovery, recovery_report = self._recover_overflow_unit(state, unit=unit, partial_text=text)
                reports.append(recovery_report)
                keep_partial = recovery["keep_partial"]
                if recovery["next_units"]:
                    pending_units = list(recovery["next_units"]) + pending_units
            if keep_partial and text.strip():
                unit_texts.append(text.strip())
        answer_text = "\n\n".join(part for part in unit_texts if part)
        return answer_text.strip(), reports[-1]

    def _record_reasoning_completed(
        self,
        state: SessionState,
        *,
        goal: str,
        status: str,
        completed_steps: int,
        failed_steps: int,
        reason: str,
    ) -> None:
        self.history.record_event(
            state,
            "reasoning_completed",
            {
                "goal": goal,
                "status": status,
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
                "reason": reason,
            },
        )

    def _finalize_answer_step(self, state: SessionState, assistant_text: str) -> tuple[bool, bool]:
        plan = state.active_plan
        if plan is None or plan.status != "active":
            return False, False
        step = None
        if plan.current_step_id:
            candidate = next((item for item in plan.steps if item.step_id == plan.current_step_id), None)
            if candidate is not None and candidate.kind == "respond" and candidate.status in {"pending", "running"}:
                step = candidate
        if step is None:
            candidates = ready_steps(plan)
            if len(candidates) == 1 and candidates[0].kind == "respond":
                step = candidates[0]
        if step is None:
            return False, False
        if step.status != "running":
            plan = self._start_step(state, plan, step)
            step = next(item for item in plan.steps if item.step_id == step.step_id)
        subsystem_result = SubsystemExecutionResult(
            subsystem_name="answer_finalizer",
            success=True,
            assistant_text=assistant_text,
        )
        verification = self._verify_step(
            state,
            plan,
            step,
            self._build_verification_artifacts(
                step,
                assistant_text=assistant_text,
                tool_results=[],
                runtime_artifacts={"subsystem": subsystem_result.subsystem_name},
            ),
        )
        if verification.passed and verification.confidence < self.config.runtime.verification_confidence_threshold:
            verification = VerificationOutcome(
                verification_passed=False,
                verification_type_used=verification.verification_type_used,
                conditions_met=list(verification.conditions_met),
                conditions_failed=[*verification.conditions_failed, "confidence_below_threshold"],
                evidence=dict(verification.evidence),
                confidence=verification.confidence,
                reason=f"{verification.reason};confidence_below_threshold",
                requires_retry=True,
                requires_replan=False,
            )
        review_passed, review_reason, review_evidence = self._review_verification_result(
            state,
            step,
            verification=verification,
            subsystem_result=subsystem_result,
        )
        if not review_passed:
            verification = VerificationOutcome(
                verification_passed=False,
                verification_type_used=verification.verification_type_used,
                conditions_met=list(verification.conditions_met),
                conditions_failed=[*verification.conditions_failed, "review_failed"],
                evidence={**dict(verification.evidence), "review": review_evidence},
                confidence=verification.confidence,
                reason=f"{verification.reason};{review_reason}",
                requires_retry=True,
                requires_replan=False,
            )
        evaluation = evaluate_verification(step, verification)
        if verification.verification_type_used != "llm_fallback" and not verification.passed and evaluation.passed:
            raise HistoryInvariantError(
                f"Evaluator attempted to override deterministic verification failure for step {step.step_id}"
            )
        if evaluation.passed:
            self._complete_step(state, plan, step, outcome=assistant_text[:120] or "assistant_response")
            self._refresh_project_state(state, reason=f"step_completed:{step.step_id}")
            self._check_consistency(state)
            return True, False
        failure_kind = "VerificationError"
        self._fail_step(state, plan, step, evaluation.reason, failure_kind)
        self._check_drift(
            state,
            failed_steps=max(state.metrics.steps_failed, state.metrics.verification_failures),
            completed_steps=state.metrics.steps_completed,
        )
        return False, True

    def _run_llm_verification(
        self,
        state: SessionState,
        *,
        step: PlanStep,
        criteria: list[dict[str, str]],
        assistant_text: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        contract = verification_contract(item.get("name", "") for item in criteria)
        prepared = self._prepare_call(
            state,
            kind="verification",
            build_prompt=lambda prompt_mode, bundle: self.prompts.build_verification_prompt(
                step_title=step.title,
                step_goal=step.goal,
                expected_outputs=step.expected_outputs,
                success_criteria=step.success_criteria,
                assistant_text=assistant_text,
                criteria=criteria,
                evidence=evidence,
                prompt_mode=prompt_mode,
                context_components=bundle.components,
            ),
            contract=contract,
            prompt_modes=self._interactive_prompt_modes(),
            goal=step.goal,
        )
        _completion, payload = self._execute_structured_call(
            state,
            prepared,
            validator=self._validate_verification_payload,
            validation_error_types=(ValueError,),
        )
        return payload

    def _execute_tool(self, state: SessionState, decision: ToolDecision) -> ToolExecutionResult | None:
        guard = self.history.guard(state, f"tool:{decision.tool_name}")
        guard.record("tool_called", {"tool_name": decision.tool_name, "tool_input": decision.tool_input})
        try:
            tool, context, invocation = self.tools.prepare(decision.tool_name, decision.tool_input, self.config, state)
            guard.record(
                "tool_execution_context",
                {
                    "tool_name": tool.name,
                    "tool_kind": tool.effective_kind(invocation.validated_input),
                    "isolated": True,
                    "policy": {
                        "allow_stateful_tools": self.config.tools.allow_stateful_tools,
                        "allow_side_effect_tools": self.config.tools.allow_side_effect_tools,
                    },
                },
            )
            for event in tool.pre_execute_events(invocation.validated_input, context):
                guard.record(event.event_type, event.payload, metadata=event.metadata)
            result = self.tools.execute_prepared(tool, context, invocation)
        except Exception as exc:
            error_payload = {
                "tool_name": decision.tool_name,
                "tool_input": decision.tool_input,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }
            guard.record("tool_error", error_payload)
            guard.require_any("tool_called", "tool_error")
            self._record_message(
                state,
                Message(
                    role="tool",
                    name=decision.tool_name,
                    content=f"tool_error: {stable_json_dumps(error_payload, indent=2)}",
                    created_at=utc_now_iso(),
                    metadata=error_payload,
                ),
            )
            return None

        emitted_types: set[str] = set()
        for generated_event in result.generated_events:
            emitted_types.add(generated_event.event_type)
            guard.record(
                generated_event.event_type,
                generated_event.payload,
                metadata=generated_event.metadata,
                derived_writes=generated_event.derived_writes,
            )

        required_generated = tool.required_generated_event_types(invocation.validated_input)
        missing_generated = required_generated - emitted_types
        if (
            decision.tool_name == "edit_text"
            and missing_generated == {"edit_applied"}
            and "edit_previewed" in emitted_types
        ):
            missing_generated = set()
        if missing_generated:
            missing_text = ", ".join(sorted(missing_generated))
            raise HistoryInvariantError(f"Tool {decision.tool_name} completed without required generated events: {missing_text}")

        tool_result_event = guard.record(
            "tool_result",
            {
                "tool_name": result.tool_name,
                "raw_input": invocation.raw_input,
                "validated_input": invocation.validated_input,
                "output": to_jsonable(result.output),
            },
        )
        guard.require_all("tool_called", "tool_result")
        guard.ensure_progress()
        self._record_message(
            state,
            Message(
                role="tool",
                name=result.tool_name,
                content=result.display_text,
                created_at=utc_now_iso(),
                metadata={
                    "raw_input": invocation.raw_input,
                    "validated_input": invocation.validated_input,
                    "output": result.output,
                },
            ),
        )
        self._extract_and_store_memory(state, tool_result_event)
        self._refresh_project_state(state, reason=f"tool:{result.tool_name}")
        return result

    def _prepare_call(
        self,
        state: SessionState,
        *,
        kind: str,
        build_prompt,
        contract: ContractSpec,
        prompt_modes: Iterable[str],
        goal: str | None = None,
        for_planning: bool = False,
    ) -> PreparedCall:
        attempts = 0
        last_report: BudgetReport | None = None
        last_error: str | None = None
        unique_modes = list(dict.fromkeys(prompt_modes))
        goal_text = goal or self._goal_text(state)

        while True:
            for prompt_mode in unique_modes:
                bundle = self._build_context_bundle(
                    state,
                    goal=goal_text,
                    kind=kind,
                    prompt_mode=prompt_mode,
                    for_planning=for_planning,
                )
                assembly = build_prompt(prompt_mode, bundle)
                report = self._budget_report(state, assembly, contract)
                self.history.record_event(
                    state,
                    "prompt_built",
                    {
                        "kind": kind,
                        "prompt_mode": prompt_mode,
                        "contract": to_jsonable(contract),
                        "prompt": assembly.prompt_text,
                        "components": [asdict(component) for component in assembly.components],
                        "budget_report": asdict(report),
                    },
                )
                cap_error = self._cap_error(report)
                self.history.record_event(
                    state,
                    "budget_checked",
                    {
                        "kind": kind,
                        "prompt_mode": prompt_mode,
                        "budget_report": asdict(report),
                        "cap_error": cap_error,
                    },
                )
                if report.fits and cap_error is None:
                    return PreparedCall(assembly=assembly, report=report, prompt_mode=prompt_mode, contract=contract)
                last_report = report
                last_error = cap_error or "budget overflow"
                self.history.record_event(
                    state,
                    "budget_rejected",
                    {"kind": kind, "prompt_mode": prompt_mode, "reason": last_error, "budget_report": asdict(report)},
                )
            if not self.config.context.compact_on_overflow:
                break
            if attempts >= self.config.context.max_compaction_rounds:
                break
            if not self._compact_once(state):
                break
            attempts += 1

        message = f"Prompt does not fit within context budget: {last_error or 'unknown reason'}"
        raise BudgetExceededError(message, last_report)

    def _maybe_compact_history(self, state: SessionState) -> None:
        while decide_history_compression(self.config, state).should_compress:
            if not self._compact_once(state):
                break

    def _compact_once(self, state: SessionState) -> bool:
        keep = min(self.config.context.max_recent_messages, len(state.messages))
        prefix = state.messages[:-keep] if keep else list(state.messages)
        if not prefix:
            return False
        plan = self._largest_summarizable_prefix(state, prefix)
        if plan is None:
            return False
        chunk_size, prepared = plan
        _completion, payload = self._execute_structured_call(
            state,
            prepared,
            validator=self._validate_summary_payload,
            validation_error_types=(ValueError,),
        )
        summary_text = payload["summary"]
        summary_payload = summary_message_payload(summary_text, source_message_count=chunk_size, created_at=utc_now_iso())
        self.history.record_event(
            state,
            "summary_created",
            {
                "source_message_count": chunk_size,
                "summary_message": summary_payload,
                "summary_budget_report": asdict(prepared.report),
            },
        )
        self.history.record_event(
            state,
            "history_compressed",
            {
                "source_message_count": chunk_size,
                "summary_message": summary_payload,
                "summary_budget_report": asdict(prepared.report),
            },
        )
        self._refresh_working_memory(state, reason="history_compressed")
        return True

    def _largest_summarizable_prefix(self, state: SessionState, prefix: list[Message]) -> tuple[int, PreparedCall] | None:
        low = 1
        high = len(prefix)
        best: tuple[int, PreparedCall] | None = None
        contract = summary_contract()
        while low <= high:
            mid = (low + high) // 2
            assembly = self.prompts.build_summary_prompt(prefix[:mid], prompt_mode="lean")
            report = self._budget_report(state, assembly, contract)
            if report.fits and self._cap_error(report) is None:
                best = (mid, PreparedCall(assembly=assembly, report=report, prompt_mode="lean", contract=contract))
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _interactive_prompt_modes(self) -> list[str]:
        modes = ["standard"]
        if self.config.runtime.lean_on_overflow:
            modes.append("lean")
        return modes

    def _call_budget(self, call_kind: str):
        return compute_call_budget(self.config, call_kind=call_kind)

    def _budget_report(self, state: SessionState | None, assembly: PromptAssembly, contract: ContractSpec) -> BudgetReport:
        components = list(assembly.components)
        components.extend(self._contract_components(contract))
        components.append(PromptComponent(name="stop_sequences", category="wrapper", text=stable_json_dumps(self.config.model.stop), include_in_context=False))
        counter = self._get_budget_counter(state)
        call_budget = self._call_budget(assembly.kind)
        try:
            reserved_response_tokens = max(
                call_budget.output_tokens,
                structured_output_token_floor(contract, config=self.config, counter=counter, call_kind=assembly.kind),
            )
            report = build_budget(
                counter,
                components,
                self.config.context,
                self.config.model.context_limit,
                reserved_response_tokens=reserved_response_tokens,
                safety_margin_tokens=call_budget.safety_margin_tokens,
            )
        except Exception as exc:
            if state is None or not self.config.context.allow_estimate_fallback:
                raise
            self.history.record_event(
                state,
                "model_tokenize_failed",
                {"text_hash": "budget-build", "error": str(exc), "error_type": exc.__class__.__name__},
            )
            fallback = ConservativeEstimator()
            self.history.record_event(
                state,
                "token_estimate_used",
                {"text_hash": "budget-build", "tokens": 0, "strategy": "chars_per_token"},
            )
            report = build_budget(
                fallback,
                components,
                self.config.context,
                self.config.model.context_limit,
                reserved_response_tokens=max(
                    call_budget.output_tokens,
                    structured_output_token_floor(contract, config=self.config, counter=fallback, call_kind=assembly.kind),
                ),
                safety_margin_tokens=call_budget.safety_margin_tokens,
            )
        if self.config.runtime.strict_budget and not report.fits:
            return report
        return report

    def _contract_components(self, contract: ContractSpec) -> list[PromptComponent]:
        components: list[PromptComponent] = []
        if contract.grammar:
            components.append(PromptComponent(name="grammar", category="grammar", text=contract.grammar, include_in_context=False))
        if contract.json_schema:
            components.append(PromptComponent(name="json_schema", category="grammar", text=stable_json_dumps(contract.json_schema), include_in_context=False))
        return components

    def _cap_error(self, report: BudgetReport) -> str | None:
        del report
        return None

    def _execute_model_call(self, state: SessionState, prepared: PreparedCall) -> CompletionResult:
        resolved_contract, request_policy = self.client.resolve_contract(
            prepared.contract,
            kind=prepared.assembly.kind,
            prompt=prepared.assembly.prompt_text,
            max_tokens=prepared.report.reserved_response_tokens,
        )
        request = self.client.build_completion_request(
            prepared.assembly.prompt_text,
            max_tokens=prepared.report.reserved_response_tokens,
            contract=resolved_contract,
        )
        last_error: Exception | None = None
        transient_unavailable_attempts = 0
        attempt = 0
        while True:
            guard = self.history.guard(state, f"model_call:{prepared.assembly.kind}")
            guard.record(
                "model_request_sent",
                {
                    "kind": prepared.assembly.kind,
                    "prompt_mode": prepared.prompt_mode,
                    "attempt": attempt + 1,
                    "request": request,
                    "budget_report": asdict(prepared.report),
                    "policy": asdict(request_policy),
                    "requested_contract_mode": prepared.contract.mode,
                    "effective_contract_mode": resolved_contract.mode,
                },
            )
            started = time.monotonic()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.client.send_completion,
                        request,
                        timeout_seconds=request_policy.effective_timeout_seconds,
                    )
                    while True:
                        try:
                            completion = future.result(timeout=request_policy.progress_poll_seconds)
                            break
                        except concurrent.futures.TimeoutError:
                            elapsed = round(time.monotonic() - started, 3)
                            guard.record(
                                "model_request_progress",
                                {
                                    "kind": prepared.assembly.kind,
                                    "prompt_mode": prepared.prompt_mode,
                                    "attempt": attempt + 1,
                                    "elapsed_seconds": elapsed,
                                    "timeout_seconds": request_policy.effective_timeout_seconds,
                                    "policy": asdict(request_policy),
                                },
                            )
            except Exception as exc:
                if self._is_model_server_unavailable(exc):
                    if (
                        self._max_model_unavailable_attempts is not None
                        and transient_unavailable_attempts >= self._max_model_unavailable_attempts
                    ):
                        raise ModelClientError("semantic_engine_unavailable") from exc
                    delay = self._model_unavailable_backoff_seconds(transient_unavailable_attempts)
                    transient_unavailable_attempts += 1
                    guard.record(
                        "retry",
                        {
                            "operation": "model_unavailable",
                            "reason": str(exc),
                            "attempt": transient_unavailable_attempts,
                            "next_attempt": transient_unavailable_attempts + 1,
                        },
                    )
                    self.history.record_event(
                        state,
                        "error",
                        {
                            "operation": "semantic_engine_unavailable",
                            "error": str(exc),
                            "error_type": exc.__class__.__name__,
                        },
                    )
                    self._sleep(delay)
                    continue
                last_error = exc
                guard.record(
                    "model_call_failed",
                    {
                        "kind": prepared.assembly.kind,
                        "prompt_mode": prepared.prompt_mode,
                        "attempt": attempt + 1,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    },
                )
                guard.require_all("model_request_sent", "model_call_failed")
                if attempt < self.config.model.max_retries:
                    attempt += 1
                    guard.record(
                        "model_retry_scheduled",
                        {"kind": prepared.assembly.kind, "prompt_mode": prepared.prompt_mode, "next_attempt": attempt + 1},
                    )
                    continue
                raise
            guard.record(
                "model_response_received",
                {
                    "kind": prepared.assembly.kind,
                    "prompt_mode": prepared.prompt_mode,
                    "attempt": attempt + 1,
                    "completion": to_jsonable(completion),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "policy": asdict(request_policy),
                },
            )
            guard.require_all("model_request_sent", "model_response_received")
            guard.ensure_progress()
            return completion
        raise ModelClientError(f"llama.cpp request failed: {last_error}")

    def _is_model_server_unavailable(self, error: BaseException) -> bool:
        if isinstance(error, requests.ConnectionError):
            return True
        if isinstance(error, requests.Timeout):
            return True
        if isinstance(error, requests.HTTPError):
            response = getattr(error, "response", None)
            if response is not None and getattr(response, "status_code", None) in {502, 503, 504}:
                return True
        return False

    def _model_unavailable_backoff_seconds(self, attempt: int) -> float:
        capped = min(max(attempt, 0), 6)
        return float(min(60, 2**capped))

    def _tokenize_with_history(self, state: SessionState, text: str) -> CountResult:
        text_hash = sha256_text(text)
        guard = self.history.guard(state, "tokenize")
        guard.record("model_tokenize_requested", {"text": text, "text_hash": text_hash})
        try:
            tokens = int(self.client.tokenize(text))
        except Exception as exc:
            guard.record("model_tokenize_failed", {"text_hash": text_hash, "error": str(exc), "error_type": exc.__class__.__name__})
            guard.require_all("model_tokenize_requested", "model_tokenize_failed")
            raise
        guard.record("model_tokenize_result", {"text_hash": text_hash, "tokens": tokens, "exact": True})
        guard.require_all("model_tokenize_requested", "model_tokenize_result")
        return CountResult(tokens=tokens, exact=True, strategy="llama_cpp_server")

    def _parse_json(self, text: str, *, contract_name: str) -> dict[str, Any]:
        stripped = text.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model returned invalid JSON for {contract_name}: {text!r}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Model returned non-object JSON for {contract_name}: {payload!r}")
        return payload

    def _extract_browser_query_argument(self, text: str) -> str | None:
        explicit_match = re.search(r"(?:^|\n)[^\n]*\bbrowser_search\s+query:\s*([^\n]+)", text, re.IGNORECASE)
        if explicit_match is not None:
            candidate = _truncate_clause(explicit_match.group(1))
            if candidate:
                return candidate
        patterns = (
            r"(?:search the web|search web|web search|search online)\s+for\s+(.+)",
            r"(?:look up)\s+(.+)",
            r"(?:browser_search)\s+(.+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match is None:
                continue
            candidate = _truncate_clause(match.group(1))
            if candidate:
                return candidate
        return None

    def _extract_url_argument(self, text: str) -> str | None:
        explicit_match = re.search(r"(?:^|\n)[^\n]*\burl:\s*([^\n]+)", text, re.IGNORECASE)
        if explicit_match is not None:
            candidate = explicit_match.group(1).strip().rstrip(".,)")
            if candidate.startswith(("http://", "https://")):
                return candidate
        candidates = _url_candidates(text)
        return candidates[0] if candidates else None

    def _extract_path_argument(self, text: str, *, prefer_last: bool) -> str | None:
        explicit_match = re.search(r"(?:^|\n)[^\n]*\bpath:\s*([^\n]+)", text)
        if explicit_match is not None:
            candidate = explicit_match.group(1).strip().rstrip(".,)")
            if candidate:
                return candidate
        matches = _path_candidates(text)
        if not matches:
            return None
        return matches[-1] if prefer_last else matches[0]

    def _deterministic_answer(self, state: SessionState) -> str | None:
        goal = self._goal_text(state)
        latest_tool_message = next(
            (message for message in reversed(state.messages) if message.role == "tool" and message.name),
            None,
        )
        if latest_tool_message is None or not isinstance(latest_tool_message.metadata, dict):
            return None
        metadata = latest_tool_message.metadata
        output = metadata.get("output")
        if not isinstance(output, dict):
            return None
        if latest_tool_message.name == "calculator":
            result = output.get("result")
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            if isinstance(result, (int, float)):
                return str(result)
            return None
        if latest_tool_message.name in {"read_file", "read_text"}:
            text = output.get("text") or output.get("content")
            if not isinstance(text, str):
                return None
            raw_input = metadata.get("raw_input") if isinstance(metadata.get("raw_input"), dict) else {}
            line_number = raw_input.get("line_number")
            if not isinstance(line_number, int):
                match = re.search(r"\bline\s+(\d+)\b", goal, re.IGNORECASE)
                line_number = int(match.group(1)) if match is not None else None
            if isinstance(line_number, int) and line_number > 0:
                lines = text.splitlines()
                if line_number <= len(lines):
                    return lines[line_number - 1]
        return None

    def _should_force_not_done_answer(self, state: SessionState, *, derived_answer: str | None = None) -> bool:
        plan = state.active_plan
        can_finalize = self._can_finalize_exact_reply(state)
        if plan is None:
            return False
        non_response_steps = [step for step in plan.steps if step.kind != "respond"]
        if any(step.status == "failed" for step in non_response_steps):
            return True
        if any(step.status in {"pending", "running"} for step in non_response_steps):
            return True
        if state.environment.waiting and state.environment.waiting_process_ids:
            return True
        if derived_answer is not None and can_finalize:
            return False
        return not can_finalize

    def _can_finalize_exact_reply(self, state: SessionState) -> bool:
        plan = state.active_plan
        if plan is None:
            return True
        non_response_steps = [step for step in plan.steps if step.kind != "respond"]
        if not non_response_steps:
            return True
        return all(step.status == "completed" for step in non_response_steps)

    def _deterministic_structured_read_answer(self, goal: str, state: SessionState) -> str | None:
        # The LLM is responsible for assembling structured answers from
        # prior reads. The previous implementation relied on substring
        # vocabulary matching ("json with keys ...", "under unsupported",
        # ...) and has been removed.
        del goal, state
        return None

    def _unsupported_string_default(self, goal: str) -> str | None:
        # Substring-based default extraction removed; LLM owns this.
        del goal
        return None

    def _missing_string_default_for_key(self, goal: str, key: str) -> str | None:
        # Substring-based default extraction removed; LLM owns this.
        del goal, key
        return None

    def _empty_budget_report(self) -> BudgetReport:
        return BudgetReport(
            context_limit=self.config.model.context_limit,
            input_tokens=0,
            reserved_response_tokens=0,
            safety_margin_tokens=0,
            required_tokens=0,
            non_context_tokens=0,
            fits=True,
            exact=True,
            breakdown=[],
        )

    def _key_value_maps_from_reads(self, state: SessionState) -> list[dict[str, str]]:
        def normalize_key(key: str) -> str:
            normalized = re.sub(r"^[Tt]he\s+|^[Aa]n?\s+", "", key.strip())
            normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
            return normalized

        mappings: list[dict[str, str]] = []
        for message in state.messages:
            if message.role != "tool" or message.name not in {"read_text", "read_file"}:
                continue
            metadata = message.metadata or {}
            payload = metadata.get("output") if isinstance(metadata, dict) else None
            if not isinstance(payload, dict):
                continue
            text = payload.get("text")
            if not isinstance(text, str):
                text = payload.get("content")
            if not isinstance(text, str):
                continue
            stripped = text.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    json_payload = json.loads(stripped)
                except json.JSONDecodeError:
                    json_payload = None
                if isinstance(json_payload, dict):
                    mapping = {
                        normalize_key(str(key)): ("true" if value is True else "false" if value is False else str(value))
                        for key, value in json_payload.items()
                        if isinstance(key, str) and isinstance(value, (str, int, float, bool))
                    }
                    if mapping:
                        mappings.append(mapping)
                        continue
            mapping: dict[str, str] = {}
            for line in text.splitlines():
                upper_line = line.strip().upper()
                if upper_line.startswith("WARN"):
                    mapping.setdefault("warning", "true")
                elif upper_line.startswith("ERROR"):
                    mapping.setdefault("error", "true")
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = normalize_key(key)
                value = value.strip()
                if key:
                    mapping[key] = value
            if mapping:
                mappings.append(mapping)
        return mappings

    def _coerce_decision(self, payload: dict[str, Any]) -> ToolDecision:
        action = payload.get("action")
        response = payload.get("response")
        tool_name = payload.get("tool_name")
        tool_input = payload.get("tool_input")
        if action not in {"respond", "call_tool"}:
            raise RuntimeError(f"Invalid tool decision action: {action!r}")
        if not isinstance(response, str):
            raise RuntimeError("tool decision response must be a string")
        if not isinstance(tool_name, str):
            raise RuntimeError("tool decision tool_name must be a string")
        if not isinstance(tool_input, dict):
            raise RuntimeError("tool decision tool_input must be an object")
        if action == "respond" and tool_name != "none":
            raise RuntimeError("tool decision respond action must use tool_name='none'")
        if action == "call_tool":
            if tool_name == "none":
                raise RuntimeError("tool decision call_tool action must select a real tool")
            if tool_name not in self.tools.tool_names(self.config):
                raise RuntimeError(f"tool decision selected unknown tool: {tool_name}")
        return ToolDecision(action=action, response=response, tool_name=tool_name, tool_input=tool_input)

class _HistoryAwareTokenCounter:
    def __init__(self, runtime: AgentRuntime, state: SessionState):
        self._runtime = runtime
        self._state = state

    def count_text(self, text: str) -> CountResult:
        try:
            return self._runtime._tokenize_with_history(self._state, text)
        except Exception:
            if not self._runtime.config.context.allow_estimate_fallback:
                raise
            estimate = ConservativeEstimator().count_text(text)
            self._runtime.history.record_event(
                self._state,
                "token_estimate_used",
                {"text_hash": sha256_text(text), "tokens": estimate.tokens, "strategy": estimate.strategy},
            )
            return estimate


class _NonRecordingTokenCounter:
    def __init__(self, runtime: AgentRuntime):
        self._runtime = runtime
        tokenize = getattr(runtime.client, "tokenize_selection", runtime.client.tokenize)
        self._exact = ExactTokenCounter(tokenize)

    def count_text(self, text: str) -> CountResult:
        try:
            return self._exact.count_text(text)
        except Exception:
            if not self._runtime.config.context.allow_estimate_fallback:
                raise
            return ConservativeEstimator().count_text(text)
