from __future__ import annotations

import copy
import inspect
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterable

import requests

from swaag.artifacts import artifact_labels_from_plan, unresolved_artifact_placeholders
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
    plan_contract,
    prompt_analysis_contract,
    subagent_selection_contract,
    strategy_selection_contract,
    summary_contract,
    task_decision_contract,
    task_decision_semantic_review_contract,
    task_expansion_contract,
    text_response_contract,
    tool_decision_contract,
    tool_input_contract,
    verification_contract,
    yes_no_contract,
)
from swaag.memory_semantic import extract_from_event
from swaag.model import LlamaCppClient, ModelClientError
from swaag.model_cache import build_model_client
from swaag.orchestrator import action_from_payload, select_action
from swaag.planner import (
    PlanValidationError,
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
    strategy_from_payload,
    validate_plan_against_strategy,
)
from swaag.subagents import SubagentManager
from swaag.subsystems import FileSubsystem, PlanningSubsystem, ReasoningSubsystem, SubsystemExecutionResult, ToolSubsystem
from swaag.tokens import ConservativeEstimator, CountResult, ExactTokenCounter, build_budget
from swaag.tools.base import ToolValidationError
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

_VOLATILE_EXCERPT_FIELDS = (
    '"created_at"', '"updated_at"', '"last_updated"',
    '"plan_id"', '"session_id"', '"run_id"',
)
_GENERATED_ID_RE = re.compile(r"\b[a-z]+_[0-9a-f]{12}\b")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z)")
_STRUCTURAL_EXCERPTS = {"{", "}", "[", "]", "},", "],"}


def _verification_candidate_excerpt_options(
    candidate: str,
    *,
    max_options: int = 48,
    max_chars: int = 160,
) -> list[str]:
    """Return exact, stable candidate substrings for constrained evidence selection."""
    text = candidate.strip()
    if not text:
        return []
    options: list[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if not value or value in _STRUCTURAL_EXCERPTS or value in options:
            return
        if any(marker in value for marker in _VOLATILE_EXCERPT_FIELDS):
            return
        if _GENERATED_ID_RE.search(value) or _TIMESTAMP_RE.search(value):
            return
        if len(value) <= max_chars:
            options.append(value)
            return
        for offset in range(0, len(value), max_chars):
            chunk = value[offset : offset + max_chars].strip()
            if chunk and chunk not in _STRUCTURAL_EXCERPTS and chunk not in options:
                options.append(chunk)
            if len(options) >= max_options:
                return

    if len(text) <= max_chars:
        add(text)
    for line in text.splitlines():
        add(line)
        if len(options) >= max_options:
            break
    if not options:
        options.append(text[:max_chars])
    return options[:max_options]


def _verification_candidate_excerpt_catalog(candidate: str) -> dict[str, str]:
    return {
        f"E{index:02d}": excerpt
        for index, excerpt in enumerate(_verification_candidate_excerpt_options(candidate), start=1)
    }


def _normalize_verification_excerpt_ids(
    payload: dict[str, Any],
    catalog: dict[str, str],
) -> dict[str, Any]:
    criteria = payload.get("criteria")
    if not isinstance(criteria, list):
        return payload
    normalized_items: list[Any] = []
    for raw_item in criteria:
        if not isinstance(raw_item, dict):
            normalized_items.append(raw_item)
            continue
        item = dict(raw_item)
        selected_ids = [
            item.pop("candidate_excerpt_id_1", None),
            item.pop("candidate_excerpt_id_2", None),
            item.pop("candidate_excerpt_id_3", None),
        ]
        excerpts: list[str] = []
        for excerpt_id in selected_ids:
            if excerpt_id in {None, ""}:
                continue
            if not isinstance(excerpt_id, str) or excerpt_id not in catalog:
                raise ValueError(f"Unknown candidate excerpt ID: {excerpt_id!r}")
            excerpt = catalog[excerpt_id]
            if excerpt not in excerpts:
                excerpts.append(excerpt)
        item["candidate_excerpts"] = excerpts
        normalized_items.append(item)
    return {**payload, "criteria": normalized_items}


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
        self.client = (
            model_client
            if model_client is not None
            else build_model_client(
                config,
                request_metadata={"cache_scope": "default_agent_runtime"},
            )
        )
        self.tools = tool_registry or ToolRegistry()
        self.history = history_store or HistoryStore(config.sessions.root, write_projections=config.sessions.write_projections)
        self.prompts = PromptBuilder(config)
        self._token_counter = token_counter
        self._token_count_cache: dict[str, int] = {}
        self._verification = VerificationEngine(
            semantic_backend_mode=self.config.retrieval.backend,
            semantic_base_url=self.config.model.base_url,
            semantic_seed=self.config.model.seed,
            semantic_connect_timeout_seconds=self.config.model.connect_timeout_seconds,
            semantic_read_timeout_seconds=self.config.model.verification_timeout_seconds,
            semantic_model_client=self.client,
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
            model_client=self.client,
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
        plan = self._create_explicit_tool_execution_plan(tool_name)
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

    def _create_explicit_tool_execution_plan(self, tool_name: str) -> Plan:
        now = utc_now_iso()
        step = PlanStep(
            step_id=new_id("step"),
            title=f"Execute registered tool {tool_name}",
            goal=f"Execute registered tool {tool_name}",
            kind="tool",
            expected_tool=tool_name,
            input_text="Use the caller-provided validated tool input.",
            expected_output="Tool execution result",
            expected_outputs=["Tool execution result"],
            done_condition=f"tool_result:{tool_name}",
            success_criteria="The explicitly requested tool call completes.",
            verification_type="composite",
            verification_checks=[],
            required_conditions=[],
            optional_conditions=[],
            output_refs=[tool_name],
            fallback_strategy="Report the tool execution failure.",
            status="pending",
            last_updated=now,
        )
        return Plan(
            plan_id=new_id("plan"),
            goal=f"Execute registered tool {tool_name}",
            steps=[step],
            success_criteria="Execute the caller-provided tool request.",
            fallback_strategy="Report the tool execution failure.",
            status="active",
            created_at=now,
            updated_at=now,
            current_step_id=step.step_id,
        )

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
        replan_reason = ""
        if turn_prep.decision.ask_user and turn_prep.decision.execution_mode in {"full_plan", "single_tool"}:
            preferred = (
                f" with preferred_tool_name={turn_prep.decision.preferred_tool_name}"
                if turn_prep.decision.preferred_tool_name
                else ""
            )
            replan_reason = (
                "The task decision selected ask_user=true with "
                f"execution_mode={turn_prep.decision.execution_mode}{preferred}. "
                "Create a model-authored plan that gathers only the necessary evidence, does not assume missing facts, "
                "and ends with one respond step asking the most useful clarification question grounded in that evidence."
            )
        elif turn_prep.decision.direct_response or turn_prep.decision.execution_mode == "direct_response":
            replan_reason = (
                "The task decision selected execution_mode=direct_response. "
                "The model must author any response steps and verification conditions from the full enabled tool registry."
            )
        elif (
            not turn_prep.decision.direct_response
            and turn_prep.decision.execution_mode == "single_tool"
            and turn_prep.decision.preferred_tool_name in self.tools.tool_names(self.config)
        ):
            replan_reason = (
                "The task decision selected execution_mode=single_tool with "
                f"preferred_tool_name={turn_prep.decision.preferred_tool_name}. "
                "Create a complete objective-verifying plan from the full enabled tool registry."
            )
        try:
            self._ensure_plan(state, effective_goal, replan_reason=replan_reason)
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

        def reset_plan_scoped_attempts() -> None:
            step_attempts.clear()
            action_counts.clear()

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
                    reset_plan_scoped_attempts()
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
                replan_reason = "user_requested_replacement"
                if turn_prep.decision.ask_user and turn_prep.decision.execution_mode in {"full_plan", "single_tool"}:
                    preferred = (
                        f" with preferred_tool_name={turn_prep.decision.preferred_tool_name}"
                        if turn_prep.decision.preferred_tool_name
                        else ""
                    )
                    replan_reason = (
                        "The replacement task decision selected ask_user=true with "
                        f"execution_mode={turn_prep.decision.execution_mode}{preferred}. "
                        "Create a model-authored plan that gathers only the necessary evidence, does not assume missing facts, "
                        "and ends with one respond step asking the most useful clarification question grounded in that evidence."
                    )
                elif turn_prep.decision.direct_response or turn_prep.decision.execution_mode == "direct_response":
                    replan_reason = (
                        "The replacement task decision selected execution_mode=direct_response. "
                        "The model must author any response steps and verification conditions from the full enabled tool registry."
                    )
                elif (
                    not turn_prep.decision.direct_response
                    and turn_prep.decision.execution_mode == "single_tool"
                    and turn_prep.decision.preferred_tool_name in self.tools.tool_names(self.config)
                ):
                    replan_reason = (
                        "The replacement task decision selected execution_mode=single_tool with "
                        f"preferred_tool_name={turn_prep.decision.preferred_tool_name}. "
                        "Create a complete objective-verifying plan from the full enabled tool registry."
                    )
                self._ensure_plan(
                    state,
                    effective_goal,
                    replan_reason=replan_reason,
                    force_replan=True,
                )
                reset_plan_scoped_attempts()
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
                reset_plan_scoped_attempts()
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
                reset_plan_scoped_attempts()
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
                if not subsystem_result.success:
                    last_failure = self._classify_failure_frontend(
                        state,
                        step=step,
                        subsystem_result=subsystem_result,
                        reason=f"subsystem_failed:{subsystem_result.subsystem_name}",
                    )
                    updated_strategy = adapt_strategy(active_strategy, failure=last_failure, metrics=state.metrics, verification_failed=False)
                    self._set_strategy(state, updated_strategy, reason=updated_strategy.reason)
                    no_progress_failures += 1
                    if last_failure.retryable and step_attempts[step.step_id] <= updated_strategy.retry_same_action_limit + 1:
                        self.history.record_event(
                            state,
                            "retry_triggered",
                            {
                                "step_id": step.step_id,
                                "reason": last_failure.reason,
                                "attempt": step_attempts[step.step_id],
                                "failure_kind": last_failure.kind,
                            },
                        )
                        continue
                    failed_steps += 1
                    self._fail_step(state, plan, step, f"Subsystem {subsystem_result.subsystem_name} failed", last_failure.kind)
                    current_running_step_id = None
                    self._check_drift(state, failed_steps=failed_steps, completed_steps=completed_steps)
                    if replans_used < self.config.planner.max_replans:
                        replans_used += 1
                        self.history.record_event(
                            state,
                            "replan_triggered",
                            {"step_id": step.step_id, "reason": last_failure.reason, "replan_count": replans_used},
                        )
                        self._ensure_plan(
                            state,
                            effective_goal,
                            replan_reason=f"Step {step.step_id} subsystem failed: {last_failure.reason}",
                            replan_attempt=replans_used,
                            force_replan=True,
                        )
                        reset_plan_scoped_attempts()
                        last_verification = None
                        last_failure = None
                        continue
                    if no_progress_failures >= self.config.runtime.no_progress_failure_limit:
                        reasoning_status = "stopped"
                        reasoning_reason = "no_progress_possible"
                    else:
                        reasoning_status = "fallback"
                        reasoning_reason = "subsystem_failed"
                    break
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
                if evaluation.passed and step.kind == "respond":
                    final_verification = self._verify_final_objective(state, step, subsystem_result.assistant_text)
                    final_evaluation = evaluate_verification(step, final_verification)
                    if not final_evaluation.passed:
                        verification = final_verification
                        evaluation = final_evaluation
                failure = None if evaluation.passed else (
                    self._classify_failure_frontend(
                        state,
                        step=step,
                        verification=verification,
                        subsystem_result=subsystem_result,
                        reason=f"verification:{evaluation.reason}",
                    )
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
                retry_allowed_by_failure = failure is None or (failure.retryable and not failure.requires_replan)
                if (
                    evaluation.requires_retry
                    and retry_allowed_by_failure
                    and subsystem_result.same_step_retry_allowed
                    and step_attempts[step.step_id] <= updated_strategy.retry_same_action_limit + 1
                ):
                    continue
                if evaluation.requires_retry and not subsystem_result.same_step_retry_allowed:
                    self.history.record_event(
                        state,
                        "retry_suppressed",
                        {
                            "step_id": step.step_id,
                            "reason": "subsystem_disallowed_same_step_retry",
                            "verification_reason": evaluation.reason,
                        },
                    )
                if evaluation.requires_retry and not retry_allowed_by_failure:
                    self.history.record_event(
                        state,
                        "retry_suppressed",
                        {
                            "step_id": step.step_id,
                            "reason": "model_disallowed_same_step_retry",
                            "verification_reason": evaluation.reason,
                            "failure_kind": failure.kind if failure is not None else "",
                            "failure_reason": failure.reason if failure is not None else "",
                        },
                    )

                failed_steps += 1
                self._fail_step(state, plan, step, evaluation.reason, failure.kind if failure is not None else "VerificationError")
                current_running_step_id = None
                self._check_drift(state, failed_steps=failed_steps, completed_steps=completed_steps)
                if replans_used < self.config.planner.max_replans:
                    replans_used += 1
                    replan_reason = failure.reason if failure is not None else evaluation.reason
                    self.history.record_event(
                        state,
                        "replan_triggered",
                        {"step_id": step.step_id, "reason": replan_reason, "replan_count": replans_used},
                    )
                    self._ensure_plan(state, effective_goal, replan_reason=f"Step {step.step_id} failed verification: {replan_reason}", replan_attempt=replans_used, force_replan=True)
                    reset_plan_scoped_attempts()
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
                    reset_plan_scoped_attempts()
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
                    reset_plan_scoped_attempts()
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
        if not answer_text and reasoning_status == "completed":
            answer_text, answer_report = self._answer(state)
            budget_reports.append(answer_report)
            answer_completed, answer_failed = self._finalize_answer_step(state, answer_text)
            if answer_completed:
                completed_steps += 1
                reasoning_reason = "answered"
            if answer_failed:
                failed_steps += 1
                reasoning_status = "fallback"
                if reasoning_reason == "final_response":
                    reasoning_reason = "answer_verification_failed"
                answer_text = ""
            if not answer_completed and not answer_failed:
                reasoning_status = "fallback"
                reasoning_reason = "answer_not_verified"
                answer_text = ""
        if not answer_text:
            answer_text = self._incomplete_turn_response(reasoning_status, reasoning_reason)
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
        answer_report = self._budget_report(None, answer_assembly, text_response_contract("answer_response"))
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
        constrained_assembly = self.prompts._assemble("doctor", "lean", [PromptComponent(name="doctor", category="instruction", text='Return {"answer":"yes"}.')])
        constrained_prepared = PreparedCall(
            assembly=constrained_assembly,
            report=self._budget_report(state, constrained_assembly, yes_no_contract()),
            prompt_mode="lean",
            contract=yes_no_contract(),
        )
        _completion, yes_no_payload = self._execute_structured_call(
            state,
            constrained_prepared,
            validator=self._validate_yes_no_payload,
            validation_error_types=(ValueError,),
        )
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
            "json_probe": yes_no_payload["answer"],
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
            prompt_modes=self._interactive_prompt_modes(),
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
        expanded: ExpandedTask | None = None
        effective_goal = user_text
        clarification_request: str | None = None
        if decision.expand_task and not decision.direct_response:
            expanded = self._expand_task_frontend(state, user_text, analysis, decision)
            effective_goal = expanded.expanded_goal
        if decision.ask_user and not decision.direct_response and decision.execution_mode == "clarification":
            clarification_request = self._build_clarification_request(user_text, analysis, state=state)
        strategy = self._select_strategy_frontend(state, effective_goal, analysis, decision)
        self._set_strategy(state, strategy, reason=strategy.reason)
        self._refresh_project_state(state, reason="turn_prepared")
        return TurnPreparation(
            analysis=analysis,
            decision=decision,
            effective_goal=effective_goal,
            expanded_task=expanded,
            clarification_request=clarification_request,
        )














































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
                "A core structured semantic call violated its enforced JSON-schema contract. "
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
        fatal_on_structured_failure: bool = True,
    ) -> tuple[CompletionResult, Any]:
        completion = self._execute_model_call(state, prepared)
        try:
            payload = self._parse_json(completion.text, contract_name=prepared.contract.name)
        except Exception as exc:
            if fatal_on_structured_failure and prepared.contract.mode == "json_schema":
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
            if fatal_on_structured_failure and prepared.contract.mode == "json_schema":
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

    def _validate_text_response_payload(self, payload: dict[str, Any]) -> dict[str, str]:
        raw_text = payload.get("text")
        if not isinstance(raw_text, str):
            raise ValueError("Text response call returned non-string text")
        if not raw_text.strip():
            raise ValueError("Text response call returned empty text")
        return {"text": raw_text}

    def _validate_yes_no_payload(self, payload: dict[str, Any]) -> dict[str, str]:
        answer = str(payload.get("answer", "")).strip()
        if answer not in {"yes", "no"}:
            raise ValueError("yes_no contract returned an invalid answer")
        return {"answer": answer}

    def _validate_verification_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_names: list[str] | None = None,
        criteria_by_name: dict[str, str] | None = None,
        candidate_grounding_by_name: dict[str, str] | None = None,
        assistant_text: str = "",
    ) -> dict[str, Any]:
        criteria = payload.get("criteria")
        if not isinstance(criteria, list):
            raise ValueError("Verification call returned invalid criteria payload")
        validated: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        normalized_candidate = assistant_text.strip()
        criteria_map = dict(criteria_by_name or {})
        grounding_map = dict(candidate_grounding_by_name or {})
        for item in criteria:
            if not isinstance(item, dict):
                raise ValueError("Verification criteria entry must be an object")
            name = str(item.get("name", "")).strip()
            if not name:
                raise ValueError("Verification criteria entry is missing name")
            if name in seen_names:
                raise ValueError(f"Verification criteria entry {name!r} is duplicated")
            seen_names.add(name)
            passed = item.get("passed")
            if not isinstance(passed, bool):
                raise ValueError(f"Verification criterion {name!r} passed must be a boolean")
            evidence = item.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(f"Verification criterion {name!r} requires non-empty evidence")
            evidence = evidence.strip()
            candidate_excerpts = item.get("candidate_excerpts")
            if not isinstance(candidate_excerpts, list) or any(not isinstance(excerpt, str) for excerpt in candidate_excerpts):
                raise ValueError(f"Verification criterion {name!r} candidate_excerpts must be an array of strings")
            candidate_excerpts = [excerpt.strip() for excerpt in candidate_excerpts]
            if any(not excerpt for excerpt in candidate_excerpts):
                raise ValueError(f"Verification criterion {name!r} candidate_excerpts cannot contain empty strings")
            candidate_excerpts = list(dict.fromkeys(candidate_excerpts))
            grounding_policy = str(grounding_map.get(name, "required")).strip() or "required"
            if grounding_policy not in {"required", "optional"}:
                raise ValueError(
                    f"Verification criterion {name!r} has unsupported candidate grounding policy {grounding_policy!r}"
                )
            if normalized_candidate:
                if grounding_policy == "required" and not candidate_excerpts:
                    raise ValueError(
                        f"Verification criterion {name!r} requires at least one exact candidate excerpt"
                    )
                for excerpt in candidate_excerpts:
                    if excerpt not in assistant_text:
                        optional_hint = (
                            " Candidate grounding is optional for this criterion; use an empty candidate_excerpts array "
                            "when the judgment is grounded only in deterministic evidence or absence."
                            if grounding_policy == "optional"
                            else ""
                        )
                        raise ValueError(
                            f"Verification criterion {name!r} candidate excerpt is not an exact substring of the candidate result: {excerpt!r}."
                            f"{optional_hint}"
                        )
            elif candidate_excerpts:
                raise ValueError(
                    f"Verification criterion {name!r} candidate_excerpts must be empty when the candidate result is empty"
                )
            criterion_text = criteria_map.get(name, "").strip()
            if criterion_text:
                normalized_evidence = " ".join(evidence.split())
                normalized_criterion = " ".join(criterion_text.split())
                if normalized_evidence == normalized_criterion:
                    raise ValueError(
                        f"Verification criterion {name!r} evidence merely repeats the criterion instead of judging the candidate"
                    )
            validated.append(
                {
                    "name": name,
                    "passed": passed,
                    "evidence": evidence,
                    "candidate_excerpts": candidate_excerpts,
                    "candidate_grounding": grounding_policy,
                }
            )
        if expected_names is not None:
            if [item["name"] for item in validated] != expected_names:
                raise ValueError(
                    "Verification criteria names must appear exactly once in input order: "
                    f"expected {expected_names!r}, got {[item['name'] for item in validated]!r}"
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
            prompt_modes=self._interactive_prompt_modes(),
            goal=user_text,
        )
        _completion, analysis = self._execute_structured_call(
            state,
            prepared,
            validator=analysis_from_payload,
            validation_error_types=(PromptAnalysisValidationError,),
        )
        source = "model"
        self.history.record_event(state, "prompt_analyzed", {"analysis": asdict(analysis), "source": source})
        return analysis

    @staticmethod
    def _validate_task_decision_semantic_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
        for field in ("decision_matches_request", "decision_is_internally_consistent", "selected_mode_and_tool_can_cover_declared_count"):
            if not isinstance(payload.get(field), bool):
                raise ValueError(f"{field} must be a boolean")
        sources = payload.get("required_evidence_sources")
        if not isinstance(sources, list) or any(not isinstance(item, str) or not item.strip() for item in sources):
            raise ValueError("required_evidence_sources must be an array of non-empty strings")
        normalized_sources = list(dict.fromkeys(item.strip() for item in sources))
        minimum_calls = payload.get("minimum_evidence_call_count")
        if not isinstance(minimum_calls, int) or isinstance(minimum_calls, bool) or minimum_calls < 0:
            raise ValueError("minimum_evidence_call_count must be a non-negative integer")
        feedback = payload.get("feedback")
        if not isinstance(feedback, str):
            raise ValueError("feedback must be a string")
        return {
            "decision_matches_request": payload["decision_matches_request"],
            "decision_is_internally_consistent": payload["decision_is_internally_consistent"],
            "required_evidence_sources": normalized_sources,
            "minimum_evidence_call_count": minimum_calls,
            "selected_mode_and_tool_can_cover_declared_count": payload["selected_mode_and_tool_can_cover_declared_count"],
            "feedback": feedback.strip(),
        }

    def _prepare_direct_task_decision_semantic_review_call(
        self,
        state: SessionState,
        *,
        user_text: str,
        analysis: PromptAnalysis,
        decision: DecisionOutcome,
        contract: ContractSpec,
    ) -> PreparedCall:
        last_report: BudgetReport | None = None
        last_error = "unknown direct semantic-review budget failure"
        complete_tools = self.tools.prompt_tuples(self.config)
        for prompt_mode in self._interactive_prompt_modes():
            assembly = self.prompts.build_task_decision_semantic_review_prompt(
                user_text=user_text,
                analysis_json=stable_json_dumps(asdict(analysis), indent=2),
                decision_json=stable_json_dumps(asdict(decision), indent=2),
                tools=complete_tools,
                prompt_mode=prompt_mode,
                context_components=[],
            )
            report = self._budget_report(state, assembly, contract)
            assembly, report = self._fit_optional_prompt_context(state, assembly, contract, report)
            self.history.record_event(
                state,
                "prompt_built",
                {
                    "kind": "verification",
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
                    "kind": "verification",
                    "prompt_mode": prompt_mode,
                    "budget_report": asdict(report),
                    "cap_error": cap_error,
                },
            )
            if report.fits and cap_error is None:
                return PreparedCall(
                    assembly=assembly,
                    report=report,
                    prompt_mode=prompt_mode,
                    contract=contract,
                )
            last_report = report
            last_error = cap_error or "budget overflow"
            self.history.record_event(
                state,
                "budget_rejected",
                {
                    "kind": "verification",
                    "prompt_mode": prompt_mode,
                    "reason": last_error,
                    "budget_report": asdict(report),
                },
            )
        raise BudgetExceededError(
            f"Direct task-decision semantic review does not fit within context budget: {last_error}",
            last_report,
        )

    def _review_task_decision_semantically(
        self,
        state: SessionState,
        *,
        user_text: str,
        analysis: PromptAnalysis,
        decision: DecisionOutcome,
        attempt: int,
    ) -> tuple[bool, str, dict[str, Any]]:
        target_id = f"task_decision:semantic_review:{attempt}"
        registry_payload = [
            {
                "name": name,
                "description": description,
                "input_schema": schema,
                "usage_guidance": guidance,
            }
            for name, description, schema, guidance in self.tools.prompt_tuples(self.config)
        ]
        self.history.record_event(
            state,
            "review_started",
            {
                "review_kind": "task_decision_semantic",
                "target_id": target_id,
                "role": "verifier",
            },
        )
        contract = task_decision_semantic_review_contract()
        prepared = self._prepare_direct_task_decision_semantic_review_call(
            state,
            user_text=user_text,
            analysis=analysis,
            decision=decision,
            contract=contract,
        )
        _completion, review = self._execute_structured_call(
            state,
            prepared,
            validator=self._validate_task_decision_semantic_review_payload,
            validation_error_types=(ValueError,),
        )
        minimum_calls = int(review["minimum_evidence_call_count"])
        failures: list[str] = []
        if not review["decision_matches_request"]:
            failures.append("The reviewer found that the decision drops or changes an explicit user instruction.")
        if not review["decision_is_internally_consistent"]:
            failures.append("The reviewer found that the decision fields or reason contradict one another.")
        if not review["selected_mode_and_tool_can_cover_declared_count"]:
            failures.append("The reviewer found that the selected execution mode or preferred tool cannot cover the required evidence in the declared count.")
        if minimum_calls > 0 and not decision.evidence_required_before_response:
            failures.append(f"The reviewer requires at least {minimum_calls} evidence call(s), but evidence_required_before_response is false.")
        if decision.evidence_call_count < minimum_calls:
            failures.append(
                f"The reviewer requires at least {minimum_calls} evidence call(s), but the decision declares {decision.evidence_call_count}."
            )
        if minimum_calls > 1 and decision.execution_mode != "full_plan":
            failures.append(
                f"The reviewer requires {minimum_calls} evidence calls, so execution_mode must be full_plan rather than {decision.execution_mode}."
            )
        if minimum_calls == 1 and decision.execution_mode not in {"single_tool", "full_plan"}:
            failures.append("One required evidence call needs single_tool or full_plan execution.")
        if minimum_calls == 0 and decision.evidence_required_before_response and not review["required_evidence_sources"]:
            failures.append("The decision declares required evidence, but the reviewer found no required evidence source.")
        passed = not failures
        feedback_parts = [review["feedback"]] if review["feedback"] else []
        feedback_parts.extend(failures)
        if review["required_evidence_sources"]:
            feedback_parts.append(
                "Required evidence sources: " + ", ".join(review["required_evidence_sources"])
            )
        feedback_parts.append(f"Minimum evidence call count: {minimum_calls}")
        feedback = "; ".join(feedback_parts)
        self.history.record_event(
            state,
            "review_completed",
            {
                "review_kind": "task_decision_semantic",
                "target_id": target_id,
                "role": "verifier",
                "passed": passed,
                "reason": "task_decision_semantic_review_passed" if passed else "task_decision_semantic_review_failed",
                "evidence": {
                    "review": review,
                    "mechanical_failures": failures,
                    "candidate_task_decision": asdict(decision),
                    "complete_enabled_tool_registry": registry_payload,
                },
            },
        )
        return passed, feedback, review

    def _decide_prompt_frontend(self, state: SessionState, user_text: str, analysis: PromptAnalysis) -> DecisionOutcome:
        contract = task_decision_contract(self.tools.tool_names(self.config))
        max_attempts = max(5, int(self.config.model.max_retries) + 3)
        previous_rejected_decision = ""
        correction_feedback: list[str] = []
        last_decision: DecisionOutcome | None = None
        for attempt in range(1, max_attempts + 1):
            semantic_review_feedback = "\n".join(correction_feedback)
            prepared = self._prepare_call(
                state,
                kind="task_decision",
                build_prompt=lambda prompt_mode, bundle, previous=previous_rejected_decision, feedback=semantic_review_feedback: self.prompts.build_task_decision_prompt(
                    user_text,
                    stable_json_dumps(asdict(analysis)),
                    prompt_mode=prompt_mode,
                    context_components=bundle.components,
                    tools=bundle.tool_prompt_tuples,
                    previous_rejected_decision=previous,
                    semantic_review_feedback=feedback,
                ),
                contract=contract,
                prompt_modes=self._interactive_prompt_modes(),
                goal=user_text,
            )
            _completion, decision_payload = self._execute_structured_call(
                state,
                prepared,
            )
            try:
                decision = decision_from_payload(decision_payload)
            except DecisionValidationError as exc:
                previous_rejected_decision = stable_json_dumps(decision_payload)
                correction_feedback.append(
                    f"Attempt {attempt} structural validation failed: {exc}"
                )
                self.history.record_event(
                    state,
                    "error",
                    {
                        "operation": "task_decision_validation",
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "attempt": attempt,
                        "payload": decision_payload,
                    },
                )
                if attempt < max_attempts:
                    self.history.record_event(
                        state,
                        "model_retry_scheduled",
                        {
                            "kind": "task_decision",
                            "prompt_mode": prepared.prompt_mode,
                            "next_attempt": attempt + 1,
                        },
                    )
                    continue
                raise FatalSemanticEngineError(
                    "Task decision failed structural validation after bounded correction attempts: "
                    f"{exc}"
                ) from exc
            last_decision = decision
            passed, feedback, _results = self._review_task_decision_semantically(
                state,
                user_text=user_text,
                analysis=analysis,
                decision=decision,
                attempt=attempt,
            )
            if passed:
                self.history.record_event(
                    state,
                    "decision_made",
                    {"decision": asdict(decision), "source": "model"},
                )
                return decision
            previous_rejected_decision = stable_json_dumps(asdict(decision))
            correction_feedback.append(
                f"Attempt {attempt} semantic review failed: {feedback}"
            )
            self.history.record_event(
                state,
                "error",
                {
                    "operation": "task_decision_semantic_review",
                    "error": feedback,
                    "error_type": "DecisionSemanticReviewError",
                    "attempt": attempt,
                },
            )
            if attempt < max_attempts:
                self.history.record_event(
                    state,
                    "model_retry_scheduled",
                    {
                        "kind": "task_decision",
                        "prompt_mode": prepared.prompt_mode,
                        "next_attempt": attempt + 1,
                    },
                )
        decision_payload = {} if last_decision is None else asdict(last_decision)
        raise FatalSemanticEngineError(
            "Task decision failed semantic review after bounded correction attempts: "
            f"{' | '.join(correction_feedback) or stable_json_dumps(decision_payload)}"
        )

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
            prompt_modes=self._interactive_prompt_modes(),
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
        """LLM-driven strategy selection with bounded correction of local validation failures."""

        contract = strategy_selection_contract()
        instruction_text = (
            "Pick the execution strategy profile that best fits the task.\n"
            "Return one JSON object with keys task_profile, strategy_name, explore_before_commit, "
            "tool_chain_depth, verification_intensity, and reason.\n"
            "task_profile must be one of [coding, file_edit, reading, multi_step, generic].\n"
            "  coding: task requires writing or changing code.\n"
            "  file_edit: task edits non-code files or configs.\n"
            "  reading: information-gathering only, no repo changes.\n"
            "  multi_step: several dependent steps across tool kinds.\n"
            "  generic: does not fit the others.\n"
            "strategy_name must be one of [conservative, exploratory].\n"
            "  conservative: proceed directly with the known approach.\n"
            "  exploratory: inspect or research before committing to a solution.\n"
            "explore_before_commit means inspect/research before editing or committing to a fix; return it as a boolean.\n"
            "tool_chain_depth is the expected number of dependent tool steps (integer 1 to 3).\n"
            "verification_intensity is the amount of checking to do (float 0.0 to 2.0; 1.0 is normal).\n"
            "reason is one short justification.\n"
            "Use reading only for tasks that do not require repository changes.\n"
            "If the goal explicitly requires code edits, file writes, patches, or running tests, prefer coding or file_edit.\n"
        )
        max_attempts = max(5, int(self.config.model.max_retries) + 3)
        previous_rejected_strategy = ""
        correction_feedback: list[str] = []
        last_payload: dict[str, Any] = {}

        for attempt in range(1, max_attempts + 1):
            feedback_text = "\n".join(correction_feedback)

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
                if previous_rejected_strategy:
                    user_components.append(
                        PromptComponent(
                            name="previous_rejected_strategy",
                            category="turn_context",
                            text=f"\nPrevious rejected strategy JSON:\n{previous_rejected_strategy}\n",
                        )
                    )
                if feedback_text:
                    user_components.append(
                        PromptComponent(
                            name="strategy_correction_feedback",
                            category="instruction",
                            text=(
                                "\nStrategy correction requirements from all previous attempts:\n"
                                f"{feedback_text}\n\n"
                                "Return one corrected strategy now. The correction requirements above override the rejected fields. "
                                "Keep already-valid fields, but change every field named by the accumulated feedback.\n"
                            ),
                        )
                    )
                return self.prompts._assemble("strategy", prompt_mode, user_components)

            prepared = self._prepare_call(
                state,
                kind="strategy",
                build_prompt=_build,
                contract=contract,
                prompt_modes=self._interactive_prompt_modes(),
                goal=effective_goal,
            )
            _completion, payload = self._execute_structured_call(state, prepared)
            last_payload = payload
            try:
                strategy = strategy_from_payload(payload)
            except StrategyValidationError as exc:
                previous_rejected_strategy = stable_json_dumps(payload)
                correction_feedback.append(
                    f"Attempt {attempt} strategy validation failed: {exc}"
                )
                self.history.record_event(
                    state,
                    "error",
                    {
                        "operation": "strategy_validation",
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "attempt": attempt,
                        "payload": payload,
                    },
                )
                if attempt < max_attempts:
                    self.history.record_event(
                        state,
                        "model_retry_scheduled",
                        {
                            "kind": "strategy",
                            "prompt_mode": prepared.prompt_mode,
                            "next_attempt": attempt + 1,
                        },
                    )
                    continue
                raise FatalSemanticEngineError(
                    "Strategy selection failed validation after bounded correction attempts: "
                    f"{' | '.join(correction_feedback)}"
                ) from exc
            self.history.record_event(
                state,
                "strategy_selection_resolved",
                {"strategy": asdict(strategy), "source": "model"},
            )
            return strategy

        raise FatalSemanticEngineError(
            "Strategy selection failed without a valid result: "
            f"{stable_json_dumps(last_payload)}"
        )

    def _select_subagent_frontend(
        self,
        state: SessionState,
        *,
        goal: str,
        purpose: str,
        detail_lines: list[str] | None = None,
    ) -> SubagentSelectionDecision:
        if os.environ.get("SWAAG_DISABLE_SUBAGENT_SELECTION", "").lower() in {"1", "true", "yes", "on"}:
            return SubagentSelectionDecision(spawn=False, subagent_type="none", reason="disabled_by_env", focus="")
        subagent_specs = self._subagents.enabled_specs()
        candidate_types = [spec.subagent_type for spec in subagent_specs]
        if not candidate_types:
            return SubagentSelectionDecision(spawn=False, subagent_type="none", reason="no_candidates", focus="")
        contract = subagent_selection_contract(candidate_types)
        registry_payload = [
            {
                "name": spec.subagent_type,
                "description": spec.purpose,
                "capabilities": list(spec.capabilities),
                "role_instruction": spec.role_instruction,
                "input_schema": spec.input_schema,
                "usage_guidance": spec.usage_guidance,
                "metadata": spec.metadata,
            }
            for spec in subagent_specs
        ]
        registry_text = stable_json_dumps(registry_payload, indent=2)
        detail_text = "\n".join(line for line in (detail_lines or []) if line.strip())
        instruction_text = (
            "Decide whether an isolated specialist subagent should be spawned for this stage.\n"
            "Available subagents are the complete enabled registry; do not assume hidden specialists exist:\n"
            f"{registry_text}\n"
            "Return one JSON object with keys spawn, subagent_type, reason, and focus.\n"
            "spawn is a boolean (true/false): true means spawn the named specialist.\n"
            "Choose spawn=true only when a specialist would materially improve the current decision.\n"
            "subagent_type must be one registered enabled specialist or 'none'.\n"
            "reason is one short justification when spawn=true and may be an empty string when spawn=false.\n"
            "focus is the short specialist brief to pass if a subagent is spawned; use an empty string otherwise.\n"
            "Set subagent_type='none' and spawn=false when no specialist is needed.\n"
        )

        def _build(prompt_mode: str, bundle: ContextBundle) -> PromptAssembly:
            tool_catalog = self.prompts.render_tool_catalog(bundle.tool_prompt_tuples, prompt_mode=prompt_mode)
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
            if tool_catalog:
                user_components.append(
                    PromptComponent(
                        name="tool_descriptions",
                        category="tool_descriptions",
                        text=f"Available tools:\n{tool_catalog}\n\n",
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
            prompt_modes=self._interactive_prompt_modes(),
            goal=goal,
        )

        def _validate(payload: dict[str, Any]) -> SubagentSelectionDecision:
            subagent_type = str(payload.get("subagent_type", "")).strip() or "none"
            if subagent_type not in {"none", *candidate_types}:
                raise ValueError(f"Unknown subagent_type: {subagent_type}")
            reason = str(payload.get("reason", "")).strip()
            spawn = bool(payload.get("spawn"))
            if subagent_type == "none":
                spawn = False
            if spawn and not reason:
                raise ValueError("Spawned subagent selection reason must not be empty")
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
                "enabled_registry": registry_payload,
                "selection": asdict(selection),
            },
        )
        return selection

    def _enabled_subagent_names(self) -> list[str]:
        return [spec.subagent_type for spec in self._subagents.enabled_specs()]


    def _classify_failure_frontend(
        self,
        state: SessionState,
        *,
        step: PlanStep | None,
        error: Exception | None = None,
        error_type: str | None = None,
        verification: VerificationOutcome | None = None,
        subsystem_result: SubsystemExecutionResult | None = None,
        reason: str = "",
    ) -> FailureClassification:
        """LLM-driven failure classification."""
        contract = failure_classification_contract()
        normalized_error_type = error_type or (error.__class__.__name__ if error is not None else "")
        instruction_text = (
            "Classify the failure that occurred. Respond with JSON only.\n"
            "Return one JSON object with keys kind, retryable, requires_replan, suggested_strategy_mode, wait_seconds, and reason.\n"
            "kind must be one of [tool_failure, reasoning_failure, planning_failure,\n"
            "missing_information, verification_failure, budget_failure, state_inconsistency,\n"
            "transient_external_wait, retry_now, retry_later_backoff, deterministic_permanent,\n"
            "side_effect_unsafe, needs_replan, needs_clarification, blocked_external,\n"
            "continue_other].\n"
            "requires_replan should be true only when the current plan shape is no longer usable.\n"
            "suggested_strategy_mode must be one of [conservative, recovery, verification_heavy].\n"
            "Set retryable=true only if a simple retry of the same action is likely to help.\n"
            "Set wait_seconds to the number of seconds to wait before retrying when the kind\n"
            "is transient_external_wait or retry_later_backoff; otherwise 0.\n"
            "reason is one short justification.\n"
        )
        step_payload = "(no step)"
        if step is not None:
            step_payload = stable_json_dumps(
                {
                    "step_id": step.step_id,
                    "kind": step.kind,
                    "expected_tool": step.expected_tool,
                    "title": step.title,
                    "goal": step.goal,
                    "success_criteria": step.success_criteria,
                    "done_condition": step.done_condition,
                    "required_conditions": list(step.required_conditions),
                    "optional_conditions": list(step.optional_conditions),
                    "verification_checks": list(step.verification_checks),
                }
            )
        failure_payload: dict[str, Any] = {
            "reported_reason": reason or "",
            "error_type": normalized_error_type or "",
            "error_message": str(error) if error is not None else "",
        }
        if verification is not None:
            failure_payload["verification"] = {
                "verification_passed": verification.verification_passed,
                "verification_type_used": verification.verification_type_used,
                "conditions_met": list(verification.conditions_met),
                "conditions_failed": list(verification.conditions_failed),
                "evidence": to_jsonable(verification.evidence),
                "confidence": verification.confidence,
                "reason": verification.reason,
                "requires_retry": verification.requires_retry,
                "requires_replan": verification.requires_replan,
            }
        if subsystem_result is not None:
            failure_payload["subsystem_result"] = {
                "subsystem_name": subsystem_result.subsystem_name,
                "success": subsystem_result.success,
                "progress": list(subsystem_result.progress),
                "same_step_retry_allowed": subsystem_result.same_step_retry_allowed,
                "background_job_started": subsystem_result.background_job_started,
                "background_process_id": subsystem_result.background_process_id or "",
                "tool_results": [
                    {
                        "tool_name": result.tool_name,
                        "completed": result.completed,
                        "output": to_jsonable(result.output),
                    }
                    for result in subsystem_result.tool_results
                ],
                "assistant_text": subsystem_result.assistant_text,
            }

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
                        "Current failure signal to classify. Use this current failure as primary evidence; "
                        "older history is context only and may describe failures that have already been handled.\n"
                        f"{stable_json_dumps(to_jsonable(failure_payload), indent=2)}\n\n"
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
            "Return one JSON object with keys action and reason.\n"
            f"action must be one of {candidates}.\n"
            "Use 'execute_step' to run the next ready step, 'retry_step' to retry the\n"
            "same step, 'replan' to rebuild the plan, 'wait' to wait for background work,\n"
            "'stop' to halt, or 'answer_directly' if the user can be answered without\n"
            "further tool use.\n"
            "reason is one short justification.\n"
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
                        f"Structural state-machine suggestion: {orchestration.action}\n\n"
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

    def _build_clarification_request(
        self,
        user_text: str,
        analysis: PromptAnalysis,
        *,
        state: SessionState | None = None,
    ) -> str:
        if state is None:
            raise RuntimeError("Clarification generation requires session state")
        contract = text_response_contract("clarification_response")

        def _build(prompt_mode: str, bundle: ContextBundle) -> PromptAssembly:
            components = [
                PromptComponent(
                    name="clarification_goal",
                    category="current_user",
                    text=f"User request:\n{user_text}\n\n",
                ),
                PromptComponent(
                    name="prompt_analysis",
                    category="analysis",
                    text=f"Prompt analysis:\n{stable_json_dumps(asdict(analysis))}\n\n",
                ),
                *bundle.components,
                PromptComponent(
                    name="clarification_instruction",
                    category="instruction",
                    text=(
                        "Write the assistant's clarification request to the user. "
                        "Ask only for the missing information needed to proceed. "
                        "Do not claim completion or invent missing facts.\n"
                    ),
                ),
            ]
            return self.prompts._assemble("clarification", prompt_mode, components)

        prepared = self._prepare_call(
            state,
            kind="clarification",
            build_prompt=_build,
            contract=contract,
            prompt_modes=self._interactive_prompt_modes(),
            goal=user_text,
        )
        _completion, payload = self._execute_structured_call(
            state,
            prepared,
            validator=self._validate_text_response_payload,
            validation_error_types=(ValueError,),
        )
        text = payload["text"]
        if not text:
            raise RuntimeError("Model returned an empty clarification request")
        self.history.record_event(state, "clarification_generated", {"source": "model", "text": text})
        return text

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

    def _validate_strategy_for_plan(
        self,
        state: SessionState,
        plan: Plan,
        *,
        completed_step_kinds: list[str],
    ) -> None:
        if state.active_strategy is None:
            return
        current = state.active_strategy
        validate_plan_against_strategy(
            plan,
            current,
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

    def _validate_tool_objective_verification(self, plan: Plan) -> None:
        for step in plan.steps:
            if not step.expected_tool:
                continue
            tool = self.tools.get(step.expected_tool)
            required_types = tuple(getattr(tool, "objective_verification_check_types", ()) or ())
            if not required_types:
                continue
            checks_by_name = {str(check.get("name", "")).strip(): check for check in step.verification_checks}
            required_checks = [checks_by_name[name] for name in step.required_conditions if name in checks_by_name]
            if any(str(check.get("check_type", "")).strip() in required_types for check in required_checks):
                continue
            allowed = ", ".join(required_types)
            raise PlanValidationError(
                f"Plan step {step.step_id} uses {step.expected_tool} but required_conditions lack an objective state check "
                f"of type: {allowed}"
            )

    def _validate_response_semantic_conditions_required(self, plan: Plan) -> None:
        for step in plan.steps:
            if step.kind not in {"respond", "reasoning"}:
                continue
            checks_by_name = {str(check.get("name", "")).strip(): check for check in step.verification_checks}
            required_names = {str(name).strip() for name in step.required_conditions}
            declared_semantic_names: list[str] = []
            for name, check in checks_by_name.items():
                if not name:
                    continue
                check_type = str(check.get("check_type", "")).strip()
                if (
                    check_type == "criterion"
                    and str(check.get("actual_source", "")).strip() == "assistant_text"
                    and str(check.get("criterion", "")).strip()
                ):
                    declared_semantic_names.append(name)
                    continue
                if check_type not in {"exact_match", "string_match"}:
                    continue
                if str(check.get("actual_source", "")).strip() != "assistant_text":
                    continue
                expected_json = str(check.get("expected_json", "")).strip()
                expected = check.get("expected")
                if expected_json or (isinstance(expected, str) and expected.strip()):
                    declared_semantic_names.append(name)
            unrequired = [name for name in declared_semantic_names if name not in required_names]
            if unrequired:
                raise PlanValidationError(
                    f"Plan step {step.step_id} declares semantic response checks that are not required_conditions: "
                    + ", ".join(unrequired)
                )

    def _required_check_types_for_plan(self, plan: Plan) -> set[str]:
        check_types: set[str] = set()
        for step in plan.steps:
            checks_by_name = {str(check.get("name", "")).strip(): check for check in step.verification_checks}
            for name in step.required_conditions:
                check = checks_by_name.get(str(name).strip())
                if check is None:
                    continue
                check_type = str(check.get("check_type", "")).strip()
                if check_type:
                    check_types.add(check_type)
        return check_types

    def _required_check_types_for_step_payload(self, step_payload: dict[str, Any]) -> dict[str, str]:
        checks_by_name: dict[str, str] = {}
        verification_checks = step_payload.get("verification_checks")
        required_conditions = step_payload.get("required_conditions")
        if not isinstance(verification_checks, list) or not isinstance(required_conditions, list):
            return checks_by_name
        all_checks: dict[str, str] = {}
        for check in verification_checks:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name", "")).strip()
            check_type = str(check.get("check_type", "")).strip()
            if name and check_type:
                all_checks[name] = check_type
        for condition_name in required_conditions:
            name = str(condition_name).strip()
            check_type = all_checks.get(name, "")
            if check_type:
                checks_by_name[name] = check_type
        return checks_by_name

    def _step_payloads_by_id(self, plan_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        raw_steps = plan_payload.get("steps")
        if not isinstance(raw_steps, list):
            return {}
        steps_by_id: dict[str, dict[str, Any]] = {}
        for step in raw_steps:
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("step_id", "")).strip()
            if step_id:
                steps_by_id[step_id] = step
        return steps_by_id

    def _current_turn_history_events(self, state: SessionState):
        events = self.history.read_history(state.session_id)
        turn_start_sequence = 1
        for event in reversed(events):
            if event.event_type == "turn_started":
                turn_start_sequence = event.sequence
                break
        return [event for event in events if event.sequence >= turn_start_sequence]

    def _unresolved_objective_verification_groups(self, state: SessionState) -> set[tuple[str, ...]]:
        unresolved: set[tuple[str, ...]] = set()
        steps_by_id: dict[str, dict[str, Any]] = {}
        for event in self._current_turn_history_events(state):
            if event.event_type in {"plan_created", "plan_updated"}:
                plan_payload = event.payload.get("plan")
                if isinstance(plan_payload, dict):
                    steps_by_id = self._step_payloads_by_id(plan_payload)
                continue
            if event.event_type not in {"verification_failed", "verification_passed"}:
                continue
            step_id = str(event.payload.get("step_id", "")).strip()
            step_payload = steps_by_id.get(step_id)
            if step_payload is None:
                continue
            required_check_types_by_name = self._required_check_types_for_step_payload(step_payload)
            if event.event_type == "verification_failed":
                expected_tool = str(step_payload.get("expected_tool", "")).strip()
                if not expected_tool:
                    continue
                try:
                    tool = self.tools.get(expected_tool)
                except KeyError:
                    continue
                objective_types = tuple(sorted(str(item).strip() for item in getattr(tool, "objective_verification_check_types", ()) or () if str(item).strip()))
                if not objective_types:
                    continue
                failed_names = {str(name).strip() for name in event.payload.get("conditions_failed", [])}
                failed_check_types = {
                    required_check_types_by_name[name]
                    for name in failed_names
                    if name in required_check_types_by_name
                }
                if failed_check_types.intersection(objective_types):
                    unresolved.add(objective_types)
                continue
            passed_names = {str(name).strip() for name in event.payload.get("conditions_met", [])}
            passed_check_types = {
                required_check_types_by_name[name]
                for name in passed_names
                if name in required_check_types_by_name
            }
            for objective_types in list(unresolved):
                if passed_check_types.intersection(objective_types):
                    unresolved.discard(objective_types)
        return unresolved

    def _validate_unresolved_objective_verification_preserved(self, state: SessionState, plan: Plan) -> None:
        unresolved_groups = self._unresolved_objective_verification_groups(state)
        if not unresolved_groups:
            return
        plan_check_types = self._required_check_types_for_plan(plan)
        missing_groups = [group for group in sorted(unresolved_groups) if not plan_check_types.intersection(group)]
        if not missing_groups:
            return
        final_response_step = self._final_response_step_with_required_semantic_check(plan)
        if final_response_step is not None:
            self.history.record_event(
                state,
                "unresolved_objective_verification_deferred",
                {
                    "missing_check_groups": [list(group) for group in missing_groups],
                    "final_step_id": final_response_step.step_id,
                    "reason": "mandatory_final_objective_verification",
                },
            )
            return
        required_descriptions = [" or ".join(group) for group in missing_groups]
        raise PlanValidationError(
            "Replacement plan cannot abandon unresolved objective verification; "
            "it must require an objective state check before success using one of: "
            + "; ".join(required_descriptions)
        )

    def _final_response_step_with_required_semantic_check(self, plan: Plan) -> PlanStep | None:
        if not plan.steps:
            return None
        step = plan.steps[-1]
        if step.kind != "respond":
            return None
        checks_by_name = {str(check.get("name", "")).strip(): check for check in step.verification_checks}
        for condition_name in step.required_conditions:
            check = checks_by_name.get(str(condition_name).strip())
            if not isinstance(check, dict):
                continue
            check_type = str(check.get("check_type", "")).strip()
            if check_type == "criterion" and str(check.get("criterion", "")).strip():
                return step
            if check_type not in {"exact_match", "string_match"}:
                continue
            if str(check.get("actual_source", "")).strip() != "assistant_text":
                continue
            expected_json = str(check.get("expected_json", "")).strip()
            expected = check.get("expected")
            if expected_json or (isinstance(expected, str) and expected.strip()):
                return step
        return None

    def _review_plan(self, state: SessionState, plan: Plan) -> None:
        self._validate_tool_objective_verification(plan)
        self._validate_response_semantic_conditions_required(plan)
        self._validate_unresolved_objective_verification_preserved(state, plan)
        self._review_plan_semantic_adequacy(state, plan)
        if len(plan.steps) <= 1 and all(step.kind == "respond" for step in plan.steps):
            self.history.record_event(
                state,
                "subagent_selection_resolved",
                {
                    "purpose": "plan_review",
                    "candidate_types": self._enabled_subagent_names(),
                    "selection": {
                        "spawn": False,
                        "subagent_type": "none",
                        "reason": "structurally_trivial_plan",
                        "focus": "",
                    },
                },
            )
            if state.active_strategy is not None:
                self._validate_strategy_for_plan(
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
                self._validate_strategy_for_plan(
                    state,
                    plan,
                    completed_step_kinds=self._completed_step_kinds(state),
                )
            if not all(step.verification_checks and step.required_conditions for step in plan.steps):
                raise PlanValidationError("plan_missing_required_review_properties")
            return
        self._switch_role(state, "verifier", reason="plan_review")
        subagent_report = self._subagents.review_plan(state, plan, subagent_type=selection.subagent_type)
        self.history.record_event(
            state,
            "subagent_spawned",
            {
                "subagent_type": subagent_report.spec.subagent_type,
                "purpose": subagent_report.spec.purpose,
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

    def _plan_semantic_review_evidence(self, state: SessionState, plan: Plan) -> dict[str, Any]:
        max_string_chars = max(1000, int(self.config.environment.max_capture_chars))
        current_turn_events = self._current_turn_history_events(state)
        recent_event_types = {
            "tool_result",
            "tool_error",
            "verification_failed",
            "verification_passed",
            "review_completed",
            "step_completed",
            "step_failed",
            "replan_triggered",
            "drift_detected",
            "state_rebuilt",
        }
        recent_events = [
            {
                "sequence": event.sequence,
                "type": event.event_type,
                "payload": self._bounded_evidence_value(
                    event.payload,
                    max_string_chars=max_string_chars,
                ),
            }
            for event in current_turn_events
            if event.event_type in recent_event_types
        ][-24:]
        return {
            "original_user_request": self._original_user_goal_text(state),
            "effective_goal": self._goal_text(state),
            "candidate_plan": self._bounded_evidence_value(plan_as_payload(plan), max_string_chars=max_string_chars),
            "active_strategy": None if state.active_strategy is None else self._bounded_evidence_value(asdict(state.active_strategy), max_string_chars=max_string_chars),
            "recent_failed_tool_or_verification_evidence": self._recent_tool_failure_evidence(state),
            "latest_observed_file_snapshots": self._latest_file_snapshot_evidence(state),
            "recent_events": recent_events,
        }

    def _review_plan_semantic_adequacy(self, state: SessionState, plan: Plan) -> None:
        self._switch_role(state, "verifier", reason="plan_semantic_review")
        review_step = PlanStep(
            step_id=f"{plan.plan_id}:semantic_plan_review",
            title="Plan semantic adequacy review",
            goal=plan.goal,
            kind="reasoning",
            expected_tool=None,
            input_text="Verify whether the candidate plan is semantically adequate before execution.",
            expected_output="The candidate plan preserves the requested objective and can fail closed.",
            done_condition="reasoning_result_nonempty",
            success_criteria=(
                "The candidate plan must preserve the original user request, use current observations and failure evidence, "
                "avoid stale failed targets unless newer evidence proves they apply, and declare objective checks precise "
                "enough to reject partial, corrupted, stale, or weakened artifact states."
            ),
            expected_outputs=["semantically_adequate_plan"],
            verification_type="composite",
            verification_checks=[],
            required_conditions=["plan_satisfies_original_request", "plan_uses_current_evidence", "plan_verifies_exact_requested_state"],
            optional_conditions=[],
        )
        plan_changes_artifacts = any(
            step.kind == "write"
            or (
                step.expected_tool is not None
                and self.tools.get(step.expected_tool).kind == "side_effect"
            )
            for step in plan.steps
        )
        criteria = [
            {
                "name": "plan_satisfies_original_request",
                "criterion": (
                    "The candidate plan is sufficient to satisfy the original user request, not a narrowed or weakened later interpretation."
                ),
            },
            {
                "name": "plan_uses_current_evidence",
                "candidate_grounding": "optional",
                "criterion": (
                    "When recent failures or snapshots show changed current state, the plan uses that evidence and does not depend on stale source snippets, ranges, patterns, or path states unless newer evidence proves they now apply."
                ),
            },
            {
                "name": "plan_verifies_exact_requested_state",
                "candidate_grounding": "required" if plan_changes_artifacts else "optional",
                "criterion": (
                    "For artifact-changing work, the plan's required objective checks are precise enough to reject partial, corrupted, stale, or merely broad-value matches."
                ),
            },
        ]
        evidence = self._plan_semantic_review_evidence(state, plan)
        self.history.record_event(
            state,
            "review_started",
            {"review_kind": "plan_semantic", "target_id": plan.plan_id, "role": "verifier"},
        )
        try:
            payload = self._run_llm_verification(
                state,
                step=review_step,
                criteria=criteria,
                assistant_text=stable_json_dumps(plan_as_payload(plan), indent=2),
                evidence=evidence,
                contract_name="plan_semantic_verification",
            )
        except SemanticBackendProtocolError as exc:
            reason = "plan_semantic_review_protocol_error"
            self.history.record_event(
                state,
                "review_completed",
                {
                    "review_kind": "plan_semantic",
                    "target_id": plan.plan_id,
                    "role": "verifier",
                    "passed": False,
                    "reason": reason,
                    "evidence": {
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "review_backend_degraded": True,
                        "review_evidence": evidence,
                    },
                },
            )
            raise PlanValidationError(f"{reason}: {exc}") from exc
        finally:
            self._switch_role(state, "primary", reason="plan_semantic_review_finished")
        criteria_results = [
            item
            for item in payload.get("criteria", [])
            if isinstance(item, dict)
        ]
        failed = [item for item in criteria_results if item.get("passed") is not True]
        passed = not failed and len(criteria_results) == len(criteria)
        reason = "plan_semantic_review_passed" if passed else "plan_semantic_review_failed"
        self.history.record_event(
            state,
            "review_completed",
            {
                "review_kind": "plan_semantic",
                "target_id": plan.plan_id,
                "role": "verifier",
                "passed": passed,
                "reason": reason,
                "evidence": {"criteria": criteria_results, "review_evidence": evidence},
            },
        )
        if not passed:
            details = "; ".join(
                f"{item.get('name', '')}: {item.get('evidence', '')}".strip()
                for item in failed
            )
            raise PlanValidationError(f"{reason}: {details or 'criteria did not all pass'}")

    def _review_verification_result(
        self,
        state: SessionState,
        step: PlanStep,
        *,
        verification: VerificationOutcome,
        subsystem_result,
    ) -> tuple[bool, str, dict[str, Any]]:
        if verification.passed and self._requires_semantic_result_review(subsystem_result):
            return self._review_mutating_tool_result_semantically(
                state,
                step,
                verification=verification,
                subsystem_result=subsystem_result,
            )
        if (
            step.kind not in {"respond", "reasoning"} and verification.verification_type_used != "llm_fallback"
        ) or self._step_uses_exact_assistant_match(step):
            self.history.record_event(
                state,
                "subagent_selection_resolved",
                {
                    "purpose": "result_review",
                    "candidate_types": self._enabled_subagent_names(),
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
            subagent_type=selection.subagent_type,
        )
        self.history.record_event(
            state,
            "subagent_spawned",
            {
                "subagent_type": subagent_report.spec.subagent_type,
                "purpose": subagent_report.spec.purpose,
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

    def _requires_semantic_result_review(self, subsystem_result) -> bool:
        latest_tool = subsystem_result.tool_results[-1] if subsystem_result.tool_results else None
        if latest_tool is None:
            return False
        try:
            tool = self.tools.get(latest_tool.tool_name)
        except KeyError:
            return False
        return bool(getattr(tool, "semantic_result_review_required", False))

    def _semantic_result_review_evidence(
        self,
        state: SessionState,
        *,
        verification: VerificationOutcome,
        subsystem_result,
    ) -> dict[str, Any]:
        latest_tool = subsystem_result.tool_results[-1] if subsystem_result.tool_results else None
        evidence: dict[str, Any] = {
            "deterministic_verification": {
                "passed": verification.passed,
                "verification_type_used": verification.verification_type_used,
                "conditions_met": list(verification.conditions_met),
                "conditions_failed": list(verification.conditions_failed),
                "evidence": to_jsonable(verification.evidence),
                "confidence": verification.confidence,
                "reason": verification.reason,
            },
            "tool_result": None,
        }
        if latest_tool is None:
            return evidence
        evidence["tool_result"] = {
            "tool_name": latest_tool.tool_name,
            "output": to_jsonable(latest_tool.output),
            "display_text": latest_tool.display_text,
        }
        output = latest_tool.output if isinstance(latest_tool.output, dict) else {}
        path_text = output.get("path")
        if isinstance(path_text, str) and path_text.strip():
            path = Path(path_text).expanduser()
            current_file: dict[str, Any] = {"path": str(path)}
            try:
                current_file["exists"] = path.exists()
                current_file["is_file"] = path.is_file()
                if path.is_file():
                    current_file["text"] = path.read_text(encoding="utf-8")
            except OSError as exc:
                current_file["read_error"] = str(exc)
                current_file["error_type"] = exc.__class__.__name__
            evidence["current_file"] = current_file
        evidence["metrics"] = {
            "tool_calls": state.metrics.tool_calls,
            "verification_failures": state.metrics.verification_failures,
            "steps_completed": state.metrics.steps_completed,
        }
        return evidence

    def _review_mutating_tool_result_semantically(
        self,
        state: SessionState,
        step: PlanStep,
        *,
        verification: VerificationOutcome,
        subsystem_result,
    ) -> tuple[bool, str, dict[str, Any]]:
        latest_tool = subsystem_result.tool_results[-1] if subsystem_result.tool_results else None
        evidence = self._semantic_result_review_evidence(
            state,
            verification=verification,
            subsystem_result=subsystem_result,
        )
        self.history.record_event(
            state,
            "review_started",
            {"review_kind": "semantic_result", "target_id": step.step_id, "role": "verifier"},
        )
        criteria = [
            {
                "name": "result_satisfies_step",
                "criterion": (
                    "Decide whether the observed tool result and current artifact state fully satisfy the step goal, "
                    "expected outputs, and success criteria. Reject partial, over-broad, corrupt, or weakly evidenced "
                    "mutations even when a lower-level deterministic condition passed."
                ),
            }
        ]
        assistant_text = subsystem_result.assistant_text
        if not assistant_text.strip() and latest_tool is not None:
            assistant_text = stable_json_dumps(evidence.get("tool_result", {}), indent=2)
        try:
            payload = self._run_llm_verification(
                state,
                step=step,
                criteria=criteria,
                assistant_text=assistant_text,
                evidence=evidence,
            )
        except Exception as exc:
            review_evidence = {
                **evidence,
                "review_error": str(exc),
                "review_error_type": exc.__class__.__name__,
            }
            passed = False
            reason = f"semantic_result_review_error:{exc}"
        else:
            criteria_results = payload.get("criteria", [])
            result = next(
                (
                    item
                    for item in criteria_results
                    if isinstance(item, dict) and item.get("name") == "result_satisfies_step"
                ),
                None,
            )
            passed = bool(isinstance(result, dict) and result.get("passed") is True)
            reason = "semantic_result_review_passed" if passed else "semantic_result_review_failed"
            review_evidence = {
                **evidence,
                "criteria": criteria_results,
            }
        self.history.record_event(
            state,
            "review_completed",
            {
                "review_kind": "semantic_result",
                "target_id": step.step_id,
                "role": "verifier",
                "passed": passed,
                "reason": reason,
                "evidence": review_evidence,
            },
        )
        if not passed and latest_tool is not None:
            self._record_message(
                state,
                Message(
                    role="tool",
                    name=latest_tool.tool_name,
                    content=f"semantic_result_review_failed: {stable_json_dumps(review_evidence, indent=2)}",
                    created_at=utc_now_iso(),
                    metadata=review_evidence,
                ),
            )
        return passed, reason, review_evidence

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
            verification=verification,
            subsystem_result=subsystem_result,
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
        artifacts: dict[str, Any] = {"step_id": step.step_id}
        latest = tool_results[-1] if tool_results else None
        if latest is not None:
            for alias in [step.expected_output, *step.expected_outputs, *step.output_refs]:
                alias_text = str(alias).strip()
                if alias_text and alias_text not in {"tool_result", "tool_name", "assistant_text"}:
                    artifacts.setdefault(alias_text, latest.output)
        if runtime_artifacts is not None:
            artifacts.update(dict(runtime_artifacts))
        return VerificationArtifacts(
            assistant_text=assistant_text,
            tool_results=list(tool_results),
            runtime_artifacts=artifacts,
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
            if step.kind in {"respond", "reasoning"}:
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
        return self._planning_subsystem.run(
            self,
            state,
            goal,
            replan_reason=replan_reason,
            replan_attempt=replan_attempt,
            update_existing=update_existing,
            required_tools=list(required_tools or []),
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
        del required_tools
        self._switch_role(state, "planner", reason="generate_plan")
        planning_goal = goal
        planner_replan_guidance = ""
        if update_existing or replan_reason:
            selection = self._select_subagent_frontend(
                state,
                goal=goal,
                purpose="plan_repair",
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
                    subagent_type=selection.subagent_type,
                )
                self.history.record_event(
                    state,
                    "subagent_spawned",
                    {
                        "subagent_type": replan_report.spec.subagent_type,
                        "purpose": replan_report.spec.purpose,
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
        plan: Plan | None = None
        validation_feedback = ""
        previous_rejected_plan = ""
        replan_state_guidance = ""
        if update_existing or replan_reason:
            replan_evidence_parts = []
            failure_evidence = self._recent_tool_failure_evidence(state)
            if failure_evidence:
                replan_evidence_parts.append(f"Recent failed tool or verification evidence:\n{failure_evidence}")
            file_snapshot_evidence = self._latest_file_snapshot_evidence(state)
            if file_snapshot_evidence:
                replan_evidence_parts.append(f"Latest observed file snapshots:\n{file_snapshot_evidence}")
            replan_state_guidance = (
                "Replan from current observations, history, and environment state. "
                "Failed steps do not undo prior tool side effects; if current observations already satisfy the requested state, "
                "plan verification and final response rather than repeating the mutation. "
                "If current observations or tool errors show that a previous source snippet, range, pattern, or other target "
                "no longer applies to the current artifact, do not plan another action that depends only on that stale target. "
                "Read or use the current artifact state, then decide the next repair, verification, blocker, or clarification. "
                "Verification checks for artifact changes must prove the exact requested final state with enough surrounding "
                "context to reject partial, corrupted, or merely broad-value matches."
            )
            if replan_evidence_parts:
                replan_state_guidance = f"{replan_state_guidance}\n\n" + "\n\n".join(replan_evidence_parts)
        configured_plan_attempts = int(self.config.model.max_retries) + 1
        planner_validation_attempts = int(self.config.planner.max_replans) + 1
        max_plan_attempts = max(2, configured_plan_attempts, planner_validation_attempts)
        last_prepared: PreparedCall | None = None
        last_raw_response = ""
        last_payload: dict[str, Any] | None = None
        last_category = "structured_validation_failure"
        last_error: Exception | None = None
        plan_validation_errors: list[str] = []

        def compact_plan_error(error: Exception) -> str:
            text = " ".join(str(error).split())
            return text if len(text) <= 800 else f"{text[:797]}..."

        def plan_error_summary() -> str:
            return "; ".join(plan_validation_errors[-4:])

        for plan_attempt in range(max_plan_attempts):
            effective_replan_reason = "\n".join(
                part
                for part in [replan_reason, replan_state_guidance, planner_replan_guidance, validation_feedback]
                if part.strip()
            )
            prepared = self._prepare_call(
                state,
                kind="plan",
                build_prompt=lambda prompt_mode, bundle, reason=effective_replan_reason, attempt=plan_attempt, rejected=previous_rejected_plan: self.prompts.build_plan_prompt(
                    planning_goal,
                    prompt_mode=prompt_mode,
                    context_components=bundle.components,
                    tools=bundle.tool_prompt_tuples,
                    replan_reason=reason,
                    previous_rejected_plan=rejected,
                    replan_attempt=replan_attempt + attempt,
                    max_replans=self.config.planner.max_replans,
                ),
                contract=contract,
                prompt_modes=["lean", *self._interactive_prompt_modes()],
                goal=planning_goal,
                for_planning=True,
            )
            last_prepared = prepared
            completion = self._execute_model_call(state, prepared)
            last_raw_response = completion.text
            try:
                payload = self._parse_json(completion.text, contract_name=prepared.contract.name)
            except Exception as exc:
                last_category = "structured_parse_failure"
                last_error = exc
                plan = None
                self.history.record_event(
                    state,
                    "error",
                    {"operation": "plan_validation", "error": str(exc), "error_type": exc.__class__.__name__},
                )
                plan_validation_errors.append(f"attempt {plan_attempt + 1} JSON parsing: {compact_plan_error(exc)}")
                if plan_attempt < max_plan_attempts - 1:
                    self.history.record_event(
                        state,
                        "model_retry_scheduled",
                        {"kind": "plan", "prompt_mode": prepared.prompt_mode, "next_attempt": plan_attempt + 2},
                    )
                    validation_feedback = (
                        f"Plan correction evidence from this generation cycle: {plan_error_summary()}. "
                        f"Latest plan attempt {plan_attempt + 1} failed JSON parsing: {str(exc)}. "
                        "Return a corrected plan that satisfies the same closed schema and all planning instructions."
                    )
                    continue
                break
            try:
                last_payload = payload
                plan = plan_from_payload(
                    payload,
                    available_tools=self.tools.tool_names(self.config),
                    plan_id=state.active_plan.plan_id if update_existing and state.active_plan is not None else None,
                )
                plan.goal = planning_goal
                if state.active_strategy is not None:
                    self._validate_strategy_for_plan(
                        state,
                        plan,
                        completed_step_kinds=self._completed_step_kinds(state),
                    )
                if len(plan.steps) > self.config.planner.max_plan_steps:
                    raise PlanValidationError(f"Planner returned {len(plan.steps)} steps; max is {self.config.planner.max_plan_steps}")
                self._review_plan(state, plan)
                break
            except (PlanValidationError, StrategyValidationError) as exc:
                last_category = "structured_validation_failure"
                last_error = exc
                plan = None
                self.history.record_event(
                    state,
                    "error",
                    {"operation": "plan_validation", "error": str(exc), "error_type": exc.__class__.__name__},
                )
                plan_validation_errors.append(f"attempt {plan_attempt + 1} validation: {compact_plan_error(exc)}")
                previous_rejected_plan = stable_json_dumps(payload)
                if plan_attempt < max_plan_attempts - 1:
                    self.history.record_event(
                        state,
                        "model_retry_scheduled",
                        {"kind": "plan", "prompt_mode": prepared.prompt_mode, "next_attempt": plan_attempt + 2},
                    )
                    validation_feedback = (
                        f"Plan correction evidence from this generation cycle: {plan_error_summary()}. "
                        f"Latest plan attempt {plan_attempt + 1} failed validation: {str(exc)}. "
                        "Return a corrected plan under the same schema. Conditions must name declared checks. "
                        "Every step, including respond steps, must declare non-empty expected_outputs labels. "
                        "Require objective checks for side-effect tools. For answer/reasoning steps, require a semantic "
                        "assistant_text check: use check_type='criterion' with actual_source='assistant_text' and non-empty "
                        "criterion, or check_type='exact_match'/'string_match' with actual_source='assistant_text' and a "
                        "non-empty expected value; include that semantic check name in required_conditions. "
                        "string_nonempty and assistant_response_nonempty are only presence checks, not semantic checks. "
                        "Give file_contains a non-empty target. For read/list/note context steps, use "
                        "tool_output_nonempty or tool_output_schema_valid unless you are checking concrete file text "
                        "with a non-empty file_contains target. expected_json is a string field containing JSON text; "
                        'for a text target like status: ready set expected_json to "\\"status: ready\\"" or leave '
                        "expected_json empty when pattern is set."
                    )
                    continue
                plan = None
                break
        if plan is None:
            error = last_error or PlanValidationError("planner did not return a valid plan")
            self._log_fatal_system_error(
                state,
                category=last_category,
                prepared=last_prepared,
                error=error,
                raw_response=last_raw_response,
                details={"payload": last_payload} if last_payload is not None else None,
            )
            raise FatalSemanticEngineError(str(error)) from error
        if update_existing:
            event = self.history.record_event(
                state,
                "plan_updated",
                {"plan": plan_as_payload(plan), "reason": replan_reason or "replanned:model"},
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
                model_client=self.client,
            )
        except SemanticBackendProtocolError as exc:
            self.history.record_event(
                state,
                "semantic_retrieval_degraded",
                {
                    "operation": "semantic_retrieval",
                    "kind": kind,
                    "goal": goal,
                    "prompt_mode": prompt_mode,
                    "for_planning": for_planning,
                    "retrieval_backend": self.config.retrieval.backend,
                    "fallback_backend": "unavailable",
                    "error": str(exc),
                },
            )
            fallback_config = copy.deepcopy(self.config)
            fallback_config.retrieval.backend = "unavailable"
            fallback_config.retrieval.allow_degraded_fallback = True
            try:
                bundle = build_context(
                    fallback_config,
                    state,
                    self._get_selection_counter(),
                    goal=goal,
                    call_kind=kind,
                    for_planning=for_planning,
                    history_events=self.history.read_history(state.session_id),
                    available_tools=self.tools.prompt_tuples(self.config),
                    model_client=self.client,
                )
            except SemanticBackendProtocolError as fallback_exc:
                self._log_fatal_system_error(
                    state=state,
                    category="semantic_retrieval_protocol_violation",
                    prepared=None,
                    error=fallback_exc,
                    operation_name="semantic_retrieval",
                    details={
                        "kind": kind,
                        "goal": goal,
                        "prompt_mode": prompt_mode,
                        "for_planning": for_planning,
                        "retrieval_backend": self.config.retrieval.backend,
                        "fallback_backend": "unavailable",
                    },
                )
                raise FatalSemanticEngineError(str(fallback_exc)) from fallback_exc
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
        retrieval_focus_text = ""
        if kind == "subagent_selection":
            self.history.record_event(
                state,
                "subagent_selection_resolved",
                {
                    "purpose": "context_retrieval_focus",
                    "candidate_types": self._enabled_subagent_names(),
                    "selection": {
                        "spawn": False,
                        "subagent_type": "none",
                        "reason": "selection_prompt_recursion_guard",
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
                    "candidate_types": self._enabled_subagent_names(),
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
                detail_lines=[
                    f"call_kind={kind}",
                    f"history_messages={len(bundle.history_messages)}",
                    f"semantic_items={len(bundle.semantic_items)}",
                    f"environment_files={len(bundle.relevant_environment_files)}",
                    f"guidance_items={len(bundle.guidance_sources)}",
                ],
            )
            if selection.spawn:
                retriever_report = self._subagents.retrieve_context(
                    state,
                    goal=goal,
                    bundle=bundle,
                    subagent_type=selection.subagent_type,
                )
                self.history.record_event(
                    state,
                    "subagent_spawned",
                    {
                        "subagent_type": retriever_report.spec.subagent_type,
                        "purpose": retriever_report.spec.purpose,
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
                if retriever_report.evidence.get("retrieval_degraded"):
                    self.history.record_event(
                        state,
                        "semantic_retrieval_degraded",
                        {
                            "operation": "subagent_retrieval_focus",
                            "kind": kind,
                            "goal": goal,
                            "prompt_mode": prompt_mode,
                            "for_planning": for_planning,
                            "retrieval_backend": retriever_report.evidence.get("retrieval_mode", self.config.retrieval.backend),
                            "fallback_backend": "complete_context_bundle",
                            "error": retriever_report.evidence.get("error", ""),
                            "scope": "subagent_retrieval_focus",
                            "reason": retriever_report.reason,
                            "error_type": retriever_report.evidence.get("error_type", ""),
                            "fallback": "complete_context_bundle",
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
        summary = stable_json_dumps(
            {
                "tool_name": message.name,
                "output": output,
            }
        )
        return Message(
            role=message.role,
            content=summary,
            name=message.name,
            created_at=message.created_at,
            metadata=message.metadata,
        )

    def _original_user_goal_text(self, state: SessionState) -> str:
        for message in reversed(state.messages):
            if message.role == "user":
                return message.content
        return ""

    def _goal_text(self, state: SessionState) -> str:
        if state.expanded_task is not None:
            return state.expanded_task.expanded_goal
        if state.active_plan is not None:
            return state.active_plan.goal
        for message in reversed(state.messages):
            if message.role == "user":
                return message.content
        return ""

    def _current_or_next_plan_step(self, state: SessionState) -> Any:
        plan = state.active_plan
        if plan is None:
            return None
        if plan.current_step_id is None:
            running = next((item for item in plan.steps if item.status == "running"), None)
            if running is not None:
                return running
        if plan.current_step_id:
            step = next((item for item in plan.steps if item.step_id == plan.current_step_id), None)
            if step is not None and step.status in {"pending", "running"}:
                return step
        running = next((item for item in plan.steps if item.status == "running"), None)
        if running is not None:
            return running
        return next_executable_step(plan)

    def _decide(self, state: SessionState) -> tuple[ToolDecision, BudgetReport]:
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
        if decision.action == "call_tool":
            tool_input = self._decide_tool_input(state, decision.tool_name)
            decision = ToolDecision(
                action=decision.action,
                response=decision.response,
                tool_name=decision.tool_name,
                tool_input=tool_input,
            )
        self.history.record_event(
            state,
            "decision_parsed",
            {"decision": asdict(decision), "prompt_mode": prepared.prompt_mode, "source": "model"},
        )
        return decision, prepared.report

    def _decide_tool_input(self, state: SessionState, tool_name: str) -> dict[str, Any]:
        tool_input, _report = self._decide_tool_input_with_report(state, tool_name)
        return tool_input

    def _decide_tool_input_with_report(self, state: SessionState, tool_name: str) -> tuple[dict[str, Any], BudgetReport]:
        prepared = self._prepare_tool_input_call(state, tool_name)
        _completion, payload = self._execute_structured_call(state, prepared)
        self.history.record_event(
            state,
            "tool_input_parsed",
            {"tool_name": tool_name, "tool_input": payload, "prompt_mode": prepared.prompt_mode, "source": "model"},
        )
        return payload, prepared.report

    def _prepare_tool_input_call(
        self,
        state: SessionState,
        tool_name: str,
        *,
        extra_instruction: str | None = None,
    ) -> PreparedCall:
        tool = self.tools.get(tool_name)
        contract = tool_input_contract(tool_name, tool.input_schema)
        step = self._current_or_next_plan_step(state)
        step_context: list[PromptComponent] = []
        if step is not None:
            step_context.extend(
                [
                    PromptComponent(name="step_title", category="turn_context", text=f"Active step title:\n{step.title}\n\n"),
                    PromptComponent(name="step_goal", category="turn_context", text=f"Active step goal:\n{step.goal}\n\n"),
                    PromptComponent(name="step_instructions", category="instruction", text=f"Step instructions:\n{step.input_text}\n\n"),
                    PromptComponent(name="step_success_criteria", category="instruction", text=f"Step success criteria:\n{step.success_criteria}\n\n"),
                ]
            )
        step_context.extend(self._tool_input_evidence_components(state, tool_name))
        if extra_instruction:
            step_context.append(
                PromptComponent(name="tool_input_retry_instruction", category="instruction", text=f"{extra_instruction}\n")
            )
        return self._prepare_call(
            state,
            kind="tool_input",
            build_prompt=lambda prompt_mode, bundle: self.prompts.build_tool_input_prompt(
                bundle.history_messages,
                tool_spec=tool.prompt_tuple(),
                prompt_mode=prompt_mode,
                context_components=[*step_context, *bundle.components],
            ),
            contract=contract,
            prompt_modes=["lean", *self._interactive_prompt_modes()],
        )













































    def _tool_input_evidence_components(self, state: SessionState, tool_name: str) -> list[PromptComponent]:
        del tool_name
        components: list[PromptComponent] = []
        failure_evidence = self._recent_tool_failure_evidence(state)
        if failure_evidence:
            components.append(
                PromptComponent(
                    name="recent_tool_failure_evidence",
                    category="turn_context",
                    text=f"Recent failed tool or verification evidence:\n{failure_evidence}\n\n",
                )
            )
        file_snapshot_evidence = self._latest_file_snapshot_evidence(state)
        if file_snapshot_evidence:
            components.append(
                PromptComponent(
                    name="latest_file_snapshot_evidence",
                    category="turn_context",
                    text=f"Latest observed file snapshots:\n{file_snapshot_evidence}\n\n",
                )
            )
        test_evidence = self._recent_failed_run_tests_evidence(state)
        if test_evidence:
            components.append(
                PromptComponent(
                    name="recent_failed_test_evidence",
                    category="turn_context",
                    text=f"Recent failed test evidence:\n{test_evidence}\n\n",
                )
            )
        return components

    def _recent_tool_failure_evidence(self, state: SessionState) -> str:
        items: list[dict[str, Any]] = []
        max_string_chars = 600
        for message in reversed(state.messages):
            if message.role != "tool":
                continue
            content = message.content.strip()
            if not (content.startswith("tool_error:") or content.startswith("verification_preview_failed:")):
                continue
            payload = message.metadata if isinstance(message.metadata, dict) else {"content": content}
            items.append(
                {
                    "tool_name": message.name or "",
                    "content": self._bounded_evidence_value(content, max_string_chars=max_string_chars),
                    "metadata": self._bounded_evidence_value(payload, max_string_chars=max_string_chars),
                }
            )
            if len(items) >= 3:
                break
        if not items:
            return ""
        items.reverse()
        return stable_json_dumps(items)

    def _latest_file_snapshot_evidence(self, state: SessionState) -> str:
        snapshots: list[dict[str, Any]] = []
        max_string_chars = 600
        for path, view in sorted(state.file_views.items())[-2:]:
            content = view.content if view.content is not None else view.last_chunk_text
            if content is None:
                content = ""
            snapshots.append(
                {
                    "path": path,
                    "content": self._bounded_evidence_value(content, max_string_chars=max_string_chars),
                    "last_operation": view.last_operation,
                    "metadata": self._bounded_evidence_value(view.metadata, max_string_chars=max_string_chars, max_items=4),
                }
            )
        seen_paths = {item["path"] for item in snapshots}
        for path, text in sorted(state.environment.workspace.known_files.items())[-2:]:
            if path in seen_paths:
                continue
            snapshots.append(
                {
                    "path": path,
                    "content": self._bounded_evidence_value(text, max_string_chars=max_string_chars),
                    "last_operation": "workspace_known_file",
                    "metadata": {},
                }
            )
        if not snapshots:
            return ""
        return stable_json_dumps(snapshots[-2:])




    def _path_exists_in_workspace(self, state: SessionState, candidate: str) -> bool:
        cwd_text = self._environment_cwd(state)
        if not cwd_text:
            return False
        try:
            candidate_path = Path(candidate)
            if candidate_path.is_absolute():
                return candidate_path.exists()
            return (Path(cwd_text) / candidate).exists()
        except OSError:
            return False

    def _path_is_regular_file_in_workspace(self, state: SessionState, candidate: str) -> bool:
        cwd_text = self._environment_cwd(state)
        if not cwd_text:
            return False
        try:
            candidate_path = Path(candidate)
            resolved = candidate_path if candidate_path.is_absolute() else Path(cwd_text) / candidate_path
            return resolved.is_file()
        except OSError:
            return False

    def _recent_tool_output(self, state: SessionState, tool_name: str) -> dict[str, Any] | None:
        for message in reversed(state.messages):
            if message.role != "tool" or message.name != tool_name or not isinstance(message.metadata, dict):
                continue
            output = message.metadata.get("output")
            if isinstance(output, dict):
                return output
        return None

    def _recent_failed_run_tests_output(self, state: SessionState) -> dict[str, Any] | None:
        for message in reversed(state.messages):
            if message.role != "tool" or message.name != "run_tests" or not isinstance(message.metadata, dict):
                continue
            output = message.metadata.get("output")
            if isinstance(output, dict) and output.get("passed") is False:
                return output
        return None

    def _recent_failed_run_tests_evidence(self, state: SessionState) -> str | None:
        output = self._recent_failed_run_tests_output(state)
        if not isinstance(output, dict):
            return None
        command = output.get("command", "")
        stderr = str(output.get("stderr", "") or "").strip()
        stdout = str(output.get("stdout", "") or "").strip()
        if not stderr and not stdout:
            return None
        return (
            f"command={command!r}\n"
            f"stdout:\n{stdout[:1200]}\n"
            f"stderr:\n{stderr[:1200]}"
        ).strip()


    def _workspace_relative_path(self, workspace: Path, path: Path) -> str:
        try:
            return path.relative_to(workspace).as_posix()
        except ValueError:
            return str(path)

    def _environment_cwd(self, state: SessionState) -> str:
        shell_cwd = getattr(state.environment.shell, "cwd", "") or ""
        if shell_cwd:
            return shell_cwd
        workspace_cwd = getattr(state.environment.workspace, "cwd", "") or ""
        if workspace_cwd:
            return workspace_cwd
        return getattr(state.environment.workspace, "root", "") or ""








    def _generate_direct_response_once(self, state: SessionState) -> tuple[str, BudgetReport]:
        contract = text_response_contract("answer_response")
        prepared = self._prepare_call(
            state,
            kind="answer",
            build_prompt=lambda prompt_mode, bundle: self.prompts.build_answer_prompt(
                state.messages,
                prompt_mode=prompt_mode,
                context_components=[*self._answer_step_context_components(state), *bundle.components],
            ),
            contract=contract,
            prompt_modes=self._interactive_prompt_modes(),
            goal=self._goal_text(state),
        )
        _completion, payload = self._execute_structured_call(
            state,
            prepared,
            validator=self._validate_text_response_payload,
            validation_error_types=(ValueError,),
        )
        self.history.record_event(
            state,
            "output_unit_generated",
            {
                "unit": {"unit_id": "direct_response", "title": "Direct response", "instruction": "Answer directly."},
                "overflowed": False,
                "text": payload["text"],
                "source": "model_answer_response",
            },
        )
        return payload["text"], prepared.report

    def _answer_step_context_components(self, state: SessionState) -> list[PromptComponent]:
        step = self._current_or_next_plan_step(state)
        if step is None or step.kind not in {"respond", "reasoning"}:
            return []
        return [
            PromptComponent(
                name="answer_step_contract",
                category="instruction",
                text=(
                    "Current answer step contract:\n"
                    f"step_id: {step.step_id}\n"
                    f"title: {step.title}\n"
                    f"goal: {step.goal}\n"
                    f"expected_output: {step.expected_output}\n"
                    f"expected_outputs: {stable_json_dumps(step.expected_outputs)}\n"
                    f"success_criteria: {step.success_criteria}\n"
                    f"required_conditions: {stable_json_dumps(step.required_conditions)}\n"
                    f"verification_checks: {stable_json_dumps(step.verification_checks)}\n\n"
                ),
            )
        ]


    def _answer(self, state: SessionState) -> tuple[str, BudgetReport]:
        return self._generate_direct_response_once(state)

    def _incomplete_turn_response(self, status: str, reason: str) -> str:
        return f"Task incomplete: {reason or status}. Verified success was not reached."

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
            final_verification = self._verify_final_objective(state, step, assistant_text)
            final_evaluation = evaluate_verification(step, final_verification)
            if not final_evaluation.passed:
                verification = final_verification
                evaluation = final_evaluation
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

    def _prepare_direct_verification_call(
        self,
        state: SessionState,
        *,
        contract: ContractSpec,
        build_prompt,
    ) -> PreparedCall:
        last_report: BudgetReport | None = None
        last_error = "unknown direct verification budget failure"
        for prompt_mode in self._interactive_prompt_modes():
            assembly = build_prompt(prompt_mode)
            report = self._budget_report(state, assembly, contract)
            self.history.record_event(
                state,
                "prompt_built",
                {
                    "kind": "verification",
                    "prompt_mode": prompt_mode,
                    "contract": to_jsonable(contract),
                    "prompt": assembly.prompt_text,
                    "components": [asdict(component) for component in assembly.components],
                    "budget_report": asdict(report),
                    "direct_evidence_context": True,
                },
            )
            cap_error = self._cap_error(report)
            self.history.record_event(
                state,
                "budget_checked",
                {
                    "kind": "verification",
                    "prompt_mode": prompt_mode,
                    "budget_report": asdict(report),
                    "cap_error": cap_error,
                    "direct_evidence_context": True,
                },
            )
            if report.fits and cap_error is None:
                return PreparedCall(
                    assembly=assembly,
                    report=report,
                    prompt_mode=prompt_mode,
                    contract=contract,
                )
            last_report = report
            last_error = cap_error or "budget overflow"
            self.history.record_event(
                state,
                "budget_rejected",
                {
                    "kind": "verification",
                    "prompt_mode": prompt_mode,
                    "reason": last_error,
                    "budget_report": asdict(report),
                    "direct_evidence_context": True,
                },
            )
        raise BudgetExceededError(
            f"Direct verification prompt does not fit within context budget: {last_error}",
            last_report,
        )

    def _run_llm_verification(
        self,
        state: SessionState,
        *,
        step: PlanStep,
        criteria: list[dict[str, Any]],
        assistant_text: str,
        evidence: dict[str, Any],
        contract_name: str = "verification",
        include_context: bool = True,
    ) -> dict[str, Any]:
        expected_names = [str(item.get("name", "")).strip() for item in criteria]
        criteria_by_name = {
            str(item.get("name", "")).strip(): str(item.get("criterion", "")).strip()
            for item in criteria
        }
        candidate_grounding_by_name = {
            str(item.get("name", "")).strip(): str(item.get("candidate_grounding", "required")).strip() or "required"
            for item in criteria
        }
        candidate_excerpt_catalog = _verification_candidate_excerpt_catalog(assistant_text)
        contract = verification_contract(
            expected_names,
            name=contract_name,
            candidate_excerpt_ids=candidate_excerpt_catalog,
        )
        max_attempts = max(3, int(self.config.model.max_retries) + 1)
        previous_rejected_verification = ""
        correction_feedback: list[str] = []
        last_payload: dict[str, Any] = {}
        for attempt in range(1, max_attempts + 1):
            feedback_text = "\n".join(correction_feedback)
            build_verification_prompt = lambda prompt_mode, context_components, previous=previous_rejected_verification, feedback=feedback_text: self.prompts.build_verification_prompt(
                step_title=step.title,
                step_goal=step.goal,
                expected_outputs=step.expected_outputs,
                success_criteria=step.success_criteria,
                assistant_text=assistant_text,
                criteria=criteria,
                evidence=evidence,
                prompt_mode=prompt_mode,
                candidate_excerpt_catalog=candidate_excerpt_catalog,
                context_components=context_components,
                previous_rejected_verification=previous,
                verification_feedback=feedback,
            )
            if include_context:
                prepared = self._prepare_call(
                    state,
                    kind="verification",
                    build_prompt=lambda prompt_mode, bundle: build_verification_prompt(prompt_mode, bundle.components),
                    contract=contract,
                    prompt_modes=self._interactive_prompt_modes(),
                    goal=step.goal,
                )
            else:
                prepared = self._prepare_direct_verification_call(
                    state,
                    contract=contract,
                    build_prompt=lambda prompt_mode: build_verification_prompt(prompt_mode, []),
                )
            _completion, wire_payload = self._execute_structured_call(state, prepared)
            try:
                payload = _normalize_verification_excerpt_ids(wire_payload, candidate_excerpt_catalog)
            except ValueError as exc:
                payload = wire_payload
                previous_rejected_verification = stable_json_dumps(wire_payload)
                correction_feedback.append(
                    f"Attempt {attempt} verification evidence-ID validation failed: {exc}"
                )
                self.history.record_event(
                    state,
                    "error",
                    {
                        "operation": "verification_evidence_id_validation",
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "attempt": attempt,
                        "payload": wire_payload,
                        "contract_name": contract_name,
                    },
                )
                if attempt < max_attempts:
                    self.history.record_event(
                        state,
                        "model_retry_scheduled",
                        {
                            "kind": "verification",
                            "prompt_mode": prepared.prompt_mode,
                            "next_attempt": attempt + 1,
                        },
                    )
                    continue
                raise FatalSemanticEngineError(
                    "Verification evidence-ID protocol failed after bounded correction attempts: "
                    f"{' | '.join(correction_feedback)}"
                ) from exc
            last_payload = payload
            try:
                return self._validate_verification_payload(
                    payload,
                    expected_names=expected_names,
                    criteria_by_name=criteria_by_name,
                    candidate_grounding_by_name=candidate_grounding_by_name,
                    assistant_text=assistant_text,
                )
            except ValueError as exc:
                previous_rejected_verification = stable_json_dumps(payload)
                correction_feedback.append(
                    f"Attempt {attempt} verification protocol validation failed: {exc}"
                )
                self.history.record_event(
                    state,
                    "error",
                    {
                        "operation": "verification_protocol_validation",
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "attempt": attempt,
                        "payload": payload,
                        "contract_name": contract_name,
                    },
                )
                if attempt < max_attempts:
                    self.history.record_event(
                        state,
                        "model_retry_scheduled",
                        {
                            "kind": "verification",
                            "prompt_mode": prepared.prompt_mode,
                            "next_attempt": attempt + 1,
                        },
                    )
                    continue
                raise FatalSemanticEngineError(
                    "Verification protocol failed after bounded correction attempts: "
                    f"{' | '.join(correction_feedback)}"
                ) from exc
        raise FatalSemanticEngineError(
            "Verification protocol failed without a validated result: "
            f"{stable_json_dumps(last_payload)}"
        )

    def _bounded_evidence_value(
        self,
        value: Any,
        *,
        max_string_chars: int,
        max_items: int = 24,
    ) -> Any:
        if isinstance(value, str):
            if len(value) <= max_string_chars:
                return value
            omitted = len(value) - max_string_chars
            return f"{value[:max_string_chars]}\n[truncated {omitted} chars]"
        if isinstance(value, list):
            bounded = [
                self._bounded_evidence_value(item, max_string_chars=max_string_chars, max_items=max_items)
                for item in value[:max_items]
            ]
            if len(value) > max_items:
                bounded.append({"truncated_items": len(value) - max_items})
            return bounded
        if isinstance(value, tuple):
            return self._bounded_evidence_value(list(value), max_string_chars=max_string_chars, max_items=max_items)
        if isinstance(value, dict):
            bounded_dict: dict[str, Any] = {}
            items = list(value.items())
            for key, item in items[:max_items]:
                bounded_dict[str(key)] = self._bounded_evidence_value(
                    item,
                    max_string_chars=max_string_chars,
                    max_items=max_items,
                )
            if len(items) > max_items:
                bounded_dict["truncated_items"] = len(items) - max_items
            return bounded_dict
        return to_jsonable(value)

    def _final_objective_evidence(self, state: SessionState, assistant_text: str) -> dict[str, Any]:
        workspace = state.environment.workspace
        max_string_chars = min(2_000, max(512, int(self.config.environment.max_capture_chars)))
        current_turn_events = self._current_turn_history_events(state)
        relevant_event_types = {
            "tool_result",
            "tool_error",
            "verification_failed",
            "verification_passed",
            "review_completed",
            "step_completed",
            "step_failed",
            "replan_triggered",
        }
        compact_events: list[dict[str, Any]] = []
        payload_fields = {
            "tool_result": ("tool_name", "output", "exit_code"),
            "tool_error": ("tool_name", "error", "error_type"),
            "verification_failed": ("step_id", "conditions_failed", "reason"),
            "verification_passed": ("step_id", "conditions_met", "reason"),
            "review_completed": ("review_kind", "target_id", "passed", "reason"),
            "step_completed": ("step_id", "step_title", "outcome"),
            "step_failed": ("step_id", "step_title", "error", "error_type"),
            "replan_triggered": ("step_id", "reason", "replan_count"),
        }
        for event in current_turn_events:
            if event.event_type not in relevant_event_types:
                continue
            fields = payload_fields[event.event_type]
            compact_payload = {
                field: self._bounded_evidence_value(
                    event.payload[field],
                    max_string_chars=max_string_chars,
                    max_items=8,
                )
                for field in fields
                if field in event.payload
            }
            compact_events.append(
                {
                    "sequence": event.sequence,
                    "type": event.event_type,
                    "payload": compact_payload,
                }
            )
        active_plan = state.active_plan
        plan_summary = None
        if active_plan is not None:
            plan_summary = {
                "goal": active_plan.goal,
                "status": active_plan.status,
                "current_step_id": active_plan.current_step_id,
                "steps": [
                    {
                        "step_id": item.step_id,
                        "title": item.title,
                        "goal": item.goal,
                        "kind": item.kind,
                        "status": item.status,
                        "expected_tool": item.expected_tool,
                        "expected_outputs": list(item.expected_outputs),
                    }
                    for item in active_plan.steps[:24]
                ],
            }
        known_files = {
            path: self._bounded_evidence_value(
                text,
                max_string_chars=max_string_chars,
                max_items=8,
            )
            for path, text in list(sorted(workspace.known_files.items()))[:24]
        }
        return {
            "original_user_request": self._bounded_evidence_value(
                self._original_user_goal_text(state),
                max_string_chars=max_string_chars,
                max_items=8,
            ),
            "effective_goal": self._bounded_evidence_value(
                self._goal_text(state),
                max_string_chars=max_string_chars,
                max_items=8,
            ),
            "assistant_text": self._bounded_evidence_value(
                assistant_text,
                max_string_chars=max_string_chars,
                max_items=8,
            ),
            "prompt_analysis": None
            if state.prompt_analysis is None
            else self._bounded_evidence_value(
                asdict(state.prompt_analysis),
                max_string_chars=max_string_chars,
                max_items=8,
            ),
            "latest_task_decision": None
            if state.latest_decision is None
            else self._bounded_evidence_value(
                asdict(state.latest_decision),
                max_string_chars=max_string_chars,
                max_items=8,
            ),
            "active_plan": plan_summary,
            "workspace": {
                "root": workspace.root,
                "cwd": workspace.cwd,
                "known_files": known_files,
                "listed_files": list(workspace.listed_files)[:48],
                "created_files": list(workspace.created_files)[:48],
                "modified_files": list(workspace.modified_files)[:48],
                "deleted_files": list(workspace.deleted_files)[:48],
            },
            "recent_events": compact_events[-16:],
        }

    def _verify_final_objective(self, state: SessionState, step: PlanStep, assistant_text: str) -> VerificationOutcome:
        proof_step = replace(
            step,
            step_id=f"{step.step_id}:final_objective",
            title="Final objective verification",
            goal=self._goal_text(state),
            input_text=(
                "Verify whether the terminal outcome explicitly requested by the original user request is complete from "
                "the current evidence. Judge the requested endpoint itself rather than assuming every task must produce "
                "a downstream artifact or fully resolved external action."
            ),
            expected_output=(
                "The requested terminal outcome—an action/artifact, a factual answer, or an evidence-grounded clarification—"
                "is proven by current evidence."
            ),
            expected_outputs=["requested_terminal_outcome", "truthful_candidate_answer"],
            success_criteria=(
                "The original user request's requested endpoint is satisfied by concrete current evidence. For an action or "
                "artifact endpoint, require strict proof of the requested current state. For an answer endpoint, require a "
                "truthful grounded answer. When the original request explicitly asks the assistant to ask a clarification, "
                "a single evidence-grounded clarification is the completed terminal outcome; do not fail it merely because "
                "the unknown information or downstream action remains unresolved. Reject partial, weakened, corrupted, stale, "
                "or merely self-consistent results, and never let later plan wording redefine the original requested endpoint."
            ),
            verification_type="composite",
            verification_checks=[],
            required_conditions=["final_objective_satisfied"],
            optional_conditions=[],
            expected_tool=None,
        )
        criteria = [
            {
                "name": "final_objective_satisfied",
                "criterion": (
                    "Decide whether the assistant's candidate final answer and concrete current evidence prove the terminal "
                    "outcome explicitly requested by the original user request. Do not impose a downstream resolved-state "
                    "requirement when the original endpoint is to ask a clarification: in that case, pass only when the required "
                    "evidence was gathered and the candidate asks the requested grounded question without assuming missing facts. "
                    "For action or artifact endpoints, reject partial, corrupted, weakened, stale, or unsupported completion claims."
                ),
            }
        ]
        evidence = self._final_objective_evidence(state, assistant_text)
        self.history.record_event(
            state,
            "verification_started",
            {
                "step_id": proof_step.step_id,
                "verification_type": proof_step.verification_type,
                "required_conditions": list(proof_step.required_conditions),
                "optional_conditions": list(proof_step.optional_conditions),
            },
        )
        payload = self._run_llm_verification(
            state,
            step=proof_step,
            criteria=criteria,
            assistant_text=assistant_text,
            evidence=evidence,
            include_context=False,
        )
        criteria_results = payload.get("criteria", [])
        result = next(
            (
                item
                for item in criteria_results
                if isinstance(item, dict) and item.get("name") == "final_objective_satisfied"
            ),
            None,
        )
        passed = bool(isinstance(result, dict) and result.get("passed") is True)
        conditions_met = ["final_objective_satisfied"] if passed else []
        conditions_failed = [] if passed else ["final_objective_satisfied"]
        reason = "final_objective_verified" if passed else "final_objective_satisfied"
        verification = VerificationOutcome(
            verification_passed=passed,
            verification_type_used=proof_step.verification_type,
            conditions_met=conditions_met,
            conditions_failed=conditions_failed,
            evidence={
                "criteria": criteria_results,
                "final_objective_evidence": evidence,
            },
            confidence=1.0 if passed else 0.0,
            reason=reason,
            requires_retry=False,
            requires_replan=not passed,
        )
        self._record_verification(state, proof_step, verification)
        return verification

    def _execute_tool(self, state: SessionState, decision: ToolDecision) -> ToolExecutionResult | None:
        guard = self.history.guard(state, f"tool:{decision.tool_name}")
        guard.record("tool_called", {"tool_name": decision.tool_name, "tool_input": decision.tool_input})
        try:
            tool, context, invocation = self.tools.prepare(decision.tool_name, decision.tool_input, self.config, state)
            self._validate_concrete_tool_input(state, tool.effective_kind(invocation.validated_input), invocation.validated_input)
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

    def _validate_concrete_tool_input(self, state: SessionState, tool_kind: str, validated_input: dict[str, Any]) -> None:
        if tool_kind != "side_effect":
            return
        unresolved = unresolved_artifact_placeholders(validated_input, artifact_labels_from_plan(state.active_plan))
        if unresolved:
            raise ToolValidationError(
                "Side-effect tool input contains unresolved artifact placeholder(s): "
                f"{', '.join(sorted(set(unresolved)))}"
            )

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
                assembly, report = self._fit_optional_prompt_context(state, assembly, contract, report)
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

    def _fit_optional_prompt_context(
        self,
        state: SessionState,
        assembly: PromptAssembly,
        contract: ContractSpec,
        report: BudgetReport,
    ) -> tuple[PromptAssembly, BudgetReport]:
        if report.fits or not any(component.optional for component in assembly.components):
            return assembly, report

        components = list(assembly.components)
        optional_indexes = [index for index, component in enumerate(components) if component.optional]
        best_assembly = assembly
        best_report = report
        for index in reversed(optional_indexes):
            del components[index]
            candidate = PromptAssembly(
                kind=assembly.kind,
                prompt_text="".join(component.text for component in components),
                components=list(components),
                prompt_mode=assembly.prompt_mode,
            )
            candidate_report = self._budget_report(state, candidate, contract)
            best_assembly = candidate
            best_report = candidate_report
            if candidate_report.fits:
                return candidate, candidate_report
        return best_assembly, best_report

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
        if contract.json_schema:
            components.append(PromptComponent(name="json_schema", category="json_schema", text=stable_json_dumps(contract.json_schema), include_in_context=False))
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
        generation_caps = {
            "answer": 512,
            "plan": max(1024, min(4096, self.config.planner.max_plan_steps * 768)),
            "tool_input": 512,
        }
        cap = generation_caps.get(prepared.assembly.kind)
        if cap is not None:
            token_key = "n_predict" if "n_predict" in request else "max_tokens" if "max_tokens" in request else ""
            original_n_predict = request.get(token_key) if token_key else None
            if isinstance(original_n_predict, int) and token_key and original_n_predict > cap:
                request = dict(request)
                request[token_key] = cap
                self.history.record_event(
                    state,
                    "budget_repaired",
                    {
                        "kind": prepared.assembly.kind,
                        "reason": f"cap_{prepared.assembly.kind}_generation_tokens",
                        "requested_response_tokens": original_n_predict,
                        "capped_response_tokens": cap,
                    },
                )
        last_error: Exception | None = None
        transient_unavailable_attempts = 0
        semantic_attempt = 0
        total_attempt = 0
        while True:
            total_attempt += 1
            guard = self.history.guard(state, f"model_call:{prepared.assembly.kind}")
            guard.record(
                "model_request_sent",
                {
                    "kind": prepared.assembly.kind,
                    "prompt_mode": prepared.prompt_mode,
                    "attempt": total_attempt,
                    "semantic_attempt": semantic_attempt + 1,
                    "request": request,
                    "budget_report": asdict(prepared.report),
                    "policy": asdict(request_policy),
                    "token_timeout_seconds": request_policy.effective_timeout_seconds,
                    "requested_contract_mode": prepared.contract.mode,
                    "effective_contract_mode": resolved_contract.mode,
                },
            )
            started = time.monotonic()
            last_progress_log = started
            last_progress_tokens = 0

            def _progress_callback(progress: dict[str, Any]) -> None:
                nonlocal last_progress_log, last_progress_tokens
                elapsed = float(progress.get("elapsed_seconds", round(time.monotonic() - started, 3)))
                tokens = int(progress.get("completion_tokens", 0) or 0)
                now = time.monotonic()
                should_log = (now - last_progress_log) >= float(request_policy.progress_poll_seconds) or tokens >= last_progress_tokens + 50
                if not should_log:
                    return
                last_progress_log = now
                last_progress_tokens = tokens
                tokens_per_second = float(progress.get("tokens_per_second", 0.0) or 0.0)
                payload = {
                    "kind": prepared.assembly.kind,
                    "prompt_mode": prepared.prompt_mode,
                    "attempt": total_attempt,
                    "elapsed_seconds": round(elapsed, 3),
                    "completion_tokens": tokens,
                    "tokens_per_second": round(tokens_per_second, 3),
                    "first_token_seconds": progress.get("first_token_seconds"),
                    "token_timeout_seconds": progress.get("token_timeout_seconds", request_policy.effective_timeout_seconds),
                }
                guard.record("model_token_progress", payload)
                print(
                    "[model_token_progress] "
                    f"kind={prepared.assembly.kind} attempt={total_attempt} "
                    f"elapsed={payload['elapsed_seconds']}s tokens={tokens} "
                    f"avg_tps={payload['tokens_per_second']} "
                    f"token_timeout={payload['token_timeout_seconds']}s",
                    flush=True,
                )

            try:
                send_completion = self.client.send_completion
                try:
                    signature = inspect.signature(send_completion)
                    supports_progress = "progress_callback" in signature.parameters or any(
                        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
                    )
                except (TypeError, ValueError):
                    supports_progress = False
                if supports_progress:
                    completion = send_completion(
                        request,
                        timeout_seconds=request_policy.effective_timeout_seconds,
                        progress_callback=_progress_callback,
                    )
                else:
                    completion = send_completion(request, timeout_seconds=request_policy.effective_timeout_seconds)
            except Exception as exc:
                if self._is_model_server_unavailable(exc):
                    transient_unavailable_attempts += 1
                    operation = "model_token_timeout" if isinstance(exc, requests.Timeout) else "model_unavailable"
                    delay = self._model_unavailable_backoff_seconds(transient_unavailable_attempts - 1)
                    payload = {
                        "operation": operation,
                        "reason": str(exc),
                        "attempt": transient_unavailable_attempts,
                        "next_attempt": transient_unavailable_attempts + 1,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "token_timeout_seconds": request_policy.effective_timeout_seconds,
                        "retry_mode": "endless_until_token_progress_or_success",
                    }
                    guard.record("retry", payload)
                    print(
                        "[model_retry] "
                        f"operation={operation} kind={prepared.assembly.kind} total_attempt={total_attempt} "
                        f"elapsed={payload['elapsed_seconds']}s token_timeout={payload['token_timeout_seconds']}s "
                        f"next_attempt={payload['next_attempt']} reason={payload['reason']}",
                        flush=True,
                    )
                    self.history.record_event(
                        state,
                        "error",
                        {
                            "operation": operation,
                            "error": str(exc),
                            "error_type": exc.__class__.__name__,
                            "retry_mode": "endless_until_token_progress_or_success",
                        },
                    )
                    if (
                        self._max_model_unavailable_attempts is not None
                        and transient_unavailable_attempts > self._max_model_unavailable_attempts
                    ):
                        self.history.record_event(
                            state,
                            "error",
                            {
                                "operation": "semantic_engine_unavailable",
                                "error": str(exc),
                                "error_type": exc.__class__.__name__,
                                "retry_mode": "bounded_by_test_escape",
                            },
                        )
                        raise ModelClientError("semantic_engine_unavailable") from exc
                    self._sleep(delay)
                    continue
                last_error = exc
                guard.record(
                    "model_call_failed",
                    {
                        "kind": prepared.assembly.kind,
                        "prompt_mode": prepared.prompt_mode,
                        "attempt": total_attempt,
                        "semantic_attempt": semantic_attempt + 1,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    },
                )
                guard.require_all("model_request_sent", "model_call_failed")
                if semantic_attempt < self.config.model.max_retries:
                    semantic_attempt += 1
                    guard.record(
                        "model_retry_scheduled",
                        {"kind": prepared.assembly.kind, "prompt_mode": prepared.prompt_mode, "next_attempt": semantic_attempt + 1},
                    )
                    continue
                raise
            elapsed_seconds = round(time.monotonic() - started, 3)
            completion_tokens = completion.completion_tokens or 0
            tokens_per_second = completion.tokens_per_second
            if tokens_per_second is None and completion_tokens:
                tokens_per_second = round(completion_tokens / max(elapsed_seconds, 1e-9), 3)
            guard.record(
                "model_response_received",
                {
                    "kind": prepared.assembly.kind,
                    "prompt_mode": prepared.prompt_mode,
                    "attempt": total_attempt,
                    "completion": to_jsonable(completion),
                    "elapsed_seconds": elapsed_seconds,
                    "completion_tokens": completion.completion_tokens,
                    "tokens_per_second": tokens_per_second,
                    "first_token_seconds": completion.first_token_seconds,
                    "token_timeout_seconds": request_policy.effective_timeout_seconds,
                    "policy": asdict(request_policy),
                },
            )
            print(
                "[model_response] "
                f"kind={prepared.assembly.kind} attempt={total_attempt} elapsed={elapsed_seconds}s "
                f"tokens={completion.completion_tokens} avg_tps={tokens_per_second}",
                file=sys.stderr,
                flush=True,
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
        cached = self._token_count_cache.get(text_hash)
        if cached is not None:
            return CountResult(tokens=cached, exact=True, strategy="llama_cpp_server_cache")
        guard = self.history.guard(state, "tokenize")
        # Persist only non-reconstructable telemetry. Recording the full text
        # makes history-fed prompts recursively contain earlier prompts.
        guard.record("model_tokenize_requested", {"text_hash": text_hash, "text_chars": len(text)})
        try:
            tokens = int(self.client.tokenize(text))
        except Exception as exc:
            guard.record("model_tokenize_failed", {"text_hash": text_hash, "error": str(exc), "error_type": exc.__class__.__name__})
            guard.require_all("model_tokenize_requested", "model_tokenize_failed")
            raise
        self._token_count_cache[text_hash] = tokens
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











    def _step_uses_exact_assistant_match(self, step: PlanStep) -> bool:
        if step.kind not in {"respond", "reasoning"}:
            return False
        for check in step.verification_checks:
            check_type = str(check.get("check_type", "")).strip()
            actual_source = str(check.get("actual_source", "")).strip()
            if check_type in {"exact_match", "string_match"} and actual_source == "assistant_text":
                return bool(str(check.get("expected", "")).strip())
        return False




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
        if tool_input:
            raise RuntimeError("tool decision tool_input must be empty; selected-tool arguments are generated by the tool_input contract")
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
