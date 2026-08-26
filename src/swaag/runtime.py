from __future__ import annotations

import inspect
import json
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

import requests

from swaag.action import ActionValidationError, AgentAction, action_from_payload
from swaag.attachments import AttachmentStore
from swaag.compression import message_source_event_references, summary_message_payload
from swaag.context_compiler import ContextCompilation, ContextCompiler
from swaag.config import AgentConfig, load_config
from swaag.embedding_index import (
    AsyncEmbeddingIndexer,
    DerivedEmbeddingIndex,
    OpenAICompatibleEmbeddingProvider,
)
from swaag.environment.environment import AgentEnvironment
from swaag.fsops import ensure_dir, restore_tree, snapshot_tree, write_text
from swaag.grammar import (
    agent_action_contract,
    completion_evaluation_contract,
    summary_contract,
    tool_result_projection_contract,
    yes_no_contract,
)
from swaag.history import HistoryInvariantError, HistoryStore
from swaag.heartbeat import heartbeat_payload, systemd_notify
from swaag.model import LlamaCppClient, ModelClientError
from swaag.preemption import (
    ModelCallPreempted,
    ModelCallStateChanged,
    ModelPreemptionCoordinator,
    RunCancellationRequested,
)
from swaag.model_cache import build_model_client
from swaag.notes import select_notes_for_prompt
from swaag.prompts import PromptBuilder
from swaag.scheduler import WakeupStore
from swaag.tokens import ConservativeEstimator, CountResult, ExactTokenCounter, build_budget
from swaag.tools.registry import ToolRegistry
from swaag.types import (
    AttachmentReference,
    BudgetReport,
    CompletionResult,
    ContractSpec,
    DeferredTask,
    Message,
    PromptAssembly,
    PromptComponent,
    SessionState,
    ToolDecision,
    ToolExecutionResult,
)
from swaag.utils import new_id, sha256_text, stable_json_dumps, to_jsonable, utc_now_iso


class BudgetExceededError(RuntimeError):
    def __init__(self, message: str, report: BudgetReport | None = None):
        super().__init__(message)
        self.report = report


class FatalSemanticEngineError(RuntimeError):
    """Compatibility name for impossible constrained-output failures."""


class OutputBudgetExhaustedError(ValueError):
    """The backend stopped before a constrained response could complete."""

    def __init__(self, finish_reason: str, reserved_tokens: int):
        super().__init__(
            f"Model output ended with {finish_reason!r} after a {reserved_tokens}-token reserve; "
            "rebuild the call with more output headroom and less semantic input if required"
        )
        self.finish_reason = finish_reason
        self.reserved_tokens = int(reserved_tokens)


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
class PreparedCall:
    assembly: PromptAssembly
    report: BudgetReport
    prompt_mode: str
    contract: ContractSpec


class AgentRuntime:
    """A constrained model/tool loop with exact history and context admission."""


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
        self.context_compiler = ContextCompiler(config)
        self.client = model_client or build_model_client(
            config,
            request_metadata={"cache_scope": "default_agent_runtime"},
        )
        self.preemption = ModelPreemptionCoordinator(config.sessions.root)
        self.tools = tool_registry or ToolRegistry()
        self._embedding_indexer: AsyncEmbeddingIndexer | None = None
        event_observer = None
        if config.embedding_index.enabled:
            provider = OpenAICompatibleEmbeddingProvider(
                config.embedding_index.base_url,
                config.embedding_index.endpoint,
                config.embedding_index.model,
                config.embedding_index.timeout_seconds,
            )
            self._embedding_indexer = AsyncEmbeddingIndexer(
                DerivedEmbeddingIndex(config.sessions.root, provider),
                config.embedding_index.fields,
            )
            event_observer = self._embedding_indexer.submit
        self.history = history_store or HistoryStore(
            config.sessions.root,
            write_projections=config.sessions.write_projections,
            event_observer=event_observer,
        )
        if history_store is not None and event_observer is not None:
            history_store.event_observer = event_observer
        self.prompts = PromptBuilder(config)
        self._token_counter = token_counter
        self._token_count_cache: dict[str, int] = {}
        self._sleep = time.sleep
        self._max_model_unavailable_attempts: int = max(1, int(self.config.model.max_retries) + 1)

    @classmethod
    def from_config_paths(cls, config_paths: list[str] | None = None) -> AgentRuntime:
        return cls(load_config(config_paths))

    def create_or_load_session(self, session_id: str | None = None) -> SessionState:
        state = self.history.create_or_load(
            config_fingerprint=self.config.config_fingerprint(),
            model_base_url=self.config.model.base_url,
            session_id=session_id,
        )
        self._ensure_environment_initialized(state)
        self._deliver_due_wakeups(state)
        return state

    def create_or_load_user_session(self, session_ref: str | None = None) -> SessionState:
        state = self.history.create_or_load_user_session(
            config_fingerprint=self.config.config_fingerprint(),
            model_base_url=self.config.model.base_url,
            session_ref=session_ref,
            prefer_latest=True,
        )
        self._ensure_environment_initialized(state)
        self._deliver_due_wakeups(state)
        return state

    def add_attachment(
        self,
        data: bytes,
        *,
        original_name: str,
        media_type: str = "",
        source: str = "api",
        session_id: str | None = None,
    ) -> AttachmentReference:
        if session_id:
            resolved = self.history.resolve_session_ref(session_id, latest_if_none=False)
            if resolved is not None and not self.history.history_path(resolved).exists():
                raise RuntimeError(f"attachments cannot be added to archived session: {resolved}")
        state = self.create_or_load_session(session_id)
        if self.history.read_active_run(state.session_id) is not None:
            raise RuntimeError("attachments cannot be added through the idle-session API during an active run")
        reference = AttachmentStore(
            self.config.sessions.root,
            max_upload_bytes=self.config.attachments.max_upload_bytes,
        ).add_bytes(
            data,
            original_name=original_name,
            media_type=media_type,
            source=source,
        )
        self.history.record_event(state, "attachment_added", {"attachment": asdict(reference)})
        return reference


    def _deliver_due_wakeups(self, state: SessionState) -> None:
        store = WakeupStore(self.config.sessions.root)
        for wakeup in store.claim_due(session_id=state.session_id):
            control = self.history.enqueue_control_message(
                state.session_id,
                f"Scheduled wakeup is due: {wakeup.reason} (scheduled for {wakeup.wake_at})",
                source="scheduler",
                control_id=f"wakeup_{wakeup.wakeup_id}",
            )
            delivered = store.mark_delivered(wakeup_id=wakeup.wakeup_id)
            self.history.record_event(
                state,
                "wakeup_due",
                {"wakeup_id": delivered.wakeup_id, "wake_at": delivered.wake_at, "reason": delivered.reason},
            )

    def resolve_session_ref(self, session_ref: str | None, *, latest_if_none: bool = False) -> str | None:
        return self.history.resolve_session_ref(session_ref, latest_if_none=latest_if_none)

    def rebuild_from_history(self, session_id: str) -> SessionState:
        state = self.history.rebuild_from_history(session_id, write_projections=False)
        self.history.record_event(
            state,
            "state_rebuilt",
            {"session_id": session_id, "event_count": state.event_count},
        )
        return state

    def _ensure_environment_initialized(self, state: SessionState) -> None:
        environment = AgentEnvironment(self.config, state)
        for event in environment.initialize_events():
            self.history.record_event(
                state,
                event.event_type,
                event.payload,
                metadata=event.metadata,
            )

    def run_turn(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        allow_silent_completion: bool = False,
    ) -> TurnResult:
        state = self.create_or_load_session(session_id)
        return self.run_turn_in_session(
            state,
            user_text,
            allow_silent_completion=allow_silent_completion,
        )

    def run_turn_in_session(
        self,
        state: SessionState,
        user_text: str,
        *,
        allow_silent_completion: bool = False,
    ) -> TurnResult:
        run_id = f"{state.session_id}:{new_id('run')}"
        self.history.set_active_run(state.session_id, run_id=run_id, user_text=user_text)
        self._heartbeat(state, run_id=run_id, phase="starting", detail="turn starting")
        try:
            result = self._run_model_tool_loop(
                state,
                user_text,
                record_user_message=True,
                allow_silent_completion=allow_silent_completion,
            )
            self._heartbeat(state, run_id=run_id, phase="completed", detail="turn completed")
            return result
        except RunCancellationRequested as exc:
            self.preemption.complete_run_cancellation(state.session_id, run_id)
            self._heartbeat(state, run_id=run_id, phase="cancelled", detail=str(exc))
            raise
        except Exception as exc:
            self._heartbeat(state, run_id=run_id, phase="failed", detail=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.history.clear_active_run(state.session_id, run_id=run_id)

    def resume_turn_in_session(self, state: SessionState, original_request: str) -> TurnResult:
        """Resume an interrupted durable task without duplicating its user request."""
        objective = original_request.strip()
        if not objective:
            raise ValueError("original_request must not be empty")
        run_id = f"{state.session_id}:{new_id('run')}"
        self.history.set_active_run(state.session_id, run_id=run_id, user_text=objective)
        self._heartbeat(state, run_id=run_id, phase="starting", detail="resuming turn")
        try:
            result = self._run_model_tool_loop(
                state,
                objective,
                record_user_message=False,
            )
            self._heartbeat(state, run_id=run_id, phase="completed", detail="resumed turn completed")
            return result
        except RunCancellationRequested as exc:
            self.preemption.complete_run_cancellation(state.session_id, run_id)
            self._heartbeat(state, run_id=run_id, phase="cancelled", detail=str(exc))
            raise
        except Exception as exc:
            self._heartbeat(state, run_id=run_id, phase="failed", detail=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.history.clear_active_run(state.session_id, run_id=run_id)

    def run_pending_controls_in_session(self, state: SessionState) -> TurnResult | None:
        self._deliver_due_wakeups(state)
        if not self.history.list_pending_control_messages(state.session_id):
            return None
        original_request = next((message.content for message in reversed(state.messages) if message.role == "user" and message.content.strip()), "")
        if not original_request:
            return None
        run_id = f"{state.session_id}:{new_id('run')}"
        self.history.set_active_run(state.session_id, run_id=run_id, user_text=original_request)
        self._heartbeat(state, run_id=run_id, phase="starting", detail="processing pending controls")
        try:
            result = self._run_model_tool_loop(state, original_request, record_user_message=False)
            self._heartbeat(state, run_id=run_id, phase="completed", detail="pending controls completed")
            return result
        except RunCancellationRequested as exc:
            self.preemption.complete_run_cancellation(state.session_id, run_id)
            self._heartbeat(state, run_id=run_id, phase="cancelled", detail=str(exc))
            raise
        except Exception as exc:
            self._heartbeat(state, run_id=run_id, phase="failed", detail=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.history.clear_active_run(state.session_id, run_id=run_id)

    def _heartbeat(
        self,
        state: SessionState,
        *,
        run_id: str | None = None,
        phase: str,
        detail: str = "",
        active_kind: str = "",
        active_id: str = "",
    ) -> None:
        payload = heartbeat_payload(phase=phase, detail=detail, active_kind=active_kind, active_id=active_id)
        self.history.update_active_run(
            state.session_id,
            run_id=run_id,
            phase=payload["phase"],
            detail=payload["detail"],
            active_kind=payload["active_kind"],
            active_id=payload["active_id"],
        )
        systemd_notify(
            "WATCHDOG=1",
            f"STATUS=swaag session={state.session_id} phase={payload['phase']} detail={payload['detail'][:180]}",
        )

    def _refresh_state_from_history(self, state: SessionState) -> None:
        refreshed = self.history.rebuild_from_history(state.session_id, write_projections=False)
        for item in fields(SessionState):
            setattr(state, item.name, getattr(refreshed, item.name))

    def _run_model_tool_loop(
        self,
        state: SessionState,
        user_text: str,
        *,
        record_user_message: bool = True,
        allow_silent_completion: bool = False,
    ) -> TurnResult:
        original_request = user_text.strip()
        if not original_request:
            raise ValueError("user_text must not be empty")

        if record_user_message:
            self.history.ensure_human_readable_name(state, original_request)
            self._record_message(
                state,
                Message(role="user", content=original_request, created_at=utc_now_iso()),
            )
        self.history.record_event(
            state,
            "turn_started",
            {
                "turn_index": state.turn_count + 1,
                "user_text": original_request,
                "execution_loop": "model_tool_loop",
            },
        )

        capability_index = self.tools.capability_index(self.config)
        loaded_tool_names: set[str] = (
            set()
            if self.config.tools.staged_discovery
            else {name for name, _, _ in capability_index}
        )
        tool_results: list[ToolExecutionResult] = []
        budget_reports: list[BudgetReport] = []
        previous_action_signature = ""
        consecutive_action_occurrences = 0
        rejected_signature_counts: dict[str, int] = {}
        rejected_observation_counts: dict[str, int] = {}
        validation_failure_counts: dict[str, int] = {}
        tool_calls_used = 0
        observation_signatures_since_state_change: set[str] = set()
        recovery_feedback = ""
        action_minimum_output_tokens = int(self.config.context.reserved_response_tokens)
        accepted_actions = 0
        max_mechanical_attempts = max(
            self.config.runtime.max_total_actions * 3,
            self.config.runtime.max_total_actions + 8,
        )

        for mechanical_attempt in range(1, max_mechanical_attempts + 1):
            self._raise_if_run_cancelled(state)
            if accepted_actions >= self.config.runtime.max_total_actions:
                break
            action_index = accepted_actions + 1
            pending_payloads = self.history.list_pending_control_messages(state.session_id)
            pending_messages = [
                str(item.get("message", "")).strip()
                for item in pending_payloads
                if str(item.get("message", "")).strip()
            ]
            validation_feedback = recovery_feedback
            recovery_feedback = ""
            selected_action: AgentAction | None = None
            state_changed_during_call = False

            for validation_attempt in range(1, 4):
                remaining_tool_calls = self.config.runtime.tool_call_budget - tool_calls_used
                tool_specs = (
                    self.tools.staged_prompt_tuples(self.config, loaded_tool_names)
                    if remaining_tool_calls > 0
                    else []
                )
                tool_names = [str(item[0]) for item in tool_specs]
                contract = agent_action_contract(
                    tool_specs,
                    allow_silent_completion=allow_silent_completion,
                )
                self._heartbeat(state, phase="context_compilation", detail=f"preparing action {action_index}", active_kind="action", active_id=str(action_index))
                prepared = self._prepare_action_call(
                    state,
                    original_request=original_request,
                    pending_messages=pending_messages,
                    tool_specs=tool_specs,
                    capability_index=capability_index if remaining_tool_calls > 0 else [],
                    contract=contract,
                    validation_feedback=validation_feedback,
                    minimum_output_tokens=action_minimum_output_tokens,
                )
                budget_reports.append(prepared.report)

                def validate(payload: dict[str, Any]) -> AgentAction:
                    action = action_from_payload(payload, enabled_tool_names=tool_names)
                    if action.silent_completion and not allow_silent_completion:
                        raise ActionValidationError(
                            "silent_completion is not permitted for this turn; return the complete user-facing result in assistant_message"
                        )
                    if len(action.tool_calls) > remaining_tool_calls:
                        raise ActionValidationError(
                            f"tool_calls contains {len(action.tool_calls)} calls but only {remaining_tool_calls} remain in the mechanical budget"
                        )
                    for tool_call in action.tool_calls:
                        try:
                            self.tools.get(tool_call.tool_name).validate(tool_call.arguments)
                        except (ValueError, TypeError) as exc:
                            raise ActionValidationError(
                                f"Invalid input for tool {tool_call.tool_name}: {exc}"
                            ) from exc
                    return action

                try:
                    selected_action = self._execute_structured_call(
                        state,
                        prepared,
                        validator=validate,
                        seed_offset=(mechanical_attempt - 1) * 3 + (validation_attempt - 1),
                    )
                    break
                except ModelCallStateChanged:
                    self._refresh_state_from_history(state)
                    recovery_feedback = (
                        "The target session changed while the previous model request was preempted for communication. "
                        "The stale request was discarded. Re-evaluate the current authoritative history and continue from the updated state."
                    )
                    state_changed_during_call = True
                    break
                except OutputBudgetExhaustedError as exc:
                    action_minimum_output_tokens = self._expanded_output_minimum(prepared)
                    validation_feedback = (
                        f"The previous constrained action exhausted its {exc.reserved_tokens}-token output budget "
                        f"with finish reason {exc.finish_reason!r}. Be concise and emit one complete valid JSON action. "
                        f"The reconstructed call now requires at least {action_minimum_output_tokens} output tokens."
                    )
                    self.history.record_event(
                        state,
                        "agent_action_rejected",
                        {
                            "action_index": action_index,
                            "validation_attempt": validation_attempt,
                            "reason": validation_feedback,
                            "finish_reason": exc.finish_reason,
                            "previous_output_tokens": exc.reserved_tokens,
                            "next_minimum_output_tokens": action_minimum_output_tokens,
                        },
                    )
                except (ActionValidationError, ValueError) as exc:
                    validation_feedback = str(exc)
                    self.history.record_event(
                        state,
                        "agent_action_rejected",
                        {
                            "action_index": action_index,
                            "validation_attempt": validation_attempt,
                            "reason": validation_feedback,
                        },
                    )

            if state_changed_during_call:
                continue

            if selected_action is None:
                recovery_feedback = validation_feedback or (
                    "The previous mechanical action could not be validated. Produce a different valid action that follows the exact tool schemas and remaining budget."
                )
                validation_key = recovery_feedback.strip()
                validation_failure_counts[validation_key] = validation_failure_counts.get(validation_key, 0) + 1
                failure_count = validation_failure_counts[validation_key]
                if failure_count > max(1, int(self.config.runtime.max_repeated_action_occurrences)):
                    return self._finish_turn(
                        state,
                        "I stopped because the model repeated the same invalid action beyond the configured validation-recovery limit.",
                        tool_results,
                        budget_reports,
                    )
                recovery_feedback += (
                    f" This same validation failure has now occurred in {failure_count} mechanical cycle(s). "
                    "Do not retry the same invalid tool usage; choose the explicitly recommended valid tool or a materially different action."
                )
                continue

            # Output starvation applies to the call that exhibited it. A
            # successful complete action restores full-fidelity-first admission
            # for the next independently compiled action.
            action_minimum_output_tokens = int(self.config.context.reserved_response_tokens)

            action_payload = asdict(selected_action)
            # Duplicate detection is about repeated mechanical work, not cosmetic
            # changes to status wording or assistant prose. For tool-bearing actions,
            # compare exactly the calls and continuation decision.
            signature_payload = (
                {
                    "tool_calls": action_payload.get("tool_calls", []),
                    "continue_loop": action_payload.get("continue_loop", False),
                }
                if action_payload.get("tool_calls")
                else action_payload
            )
            signature = stable_json_dumps(signature_payload, indent=None)
            if signature == previous_action_signature:
                consecutive_action_occurrences += 1
            else:
                previous_action_signature = signature
                consecutive_action_occurrences = 1
            occurrence = consecutive_action_occurrences
            self.history.record_event(
                state,
                "agent_action_selected",
                {
                    "action_index": action_index,
                    "action": action_payload,
                    "occurrence": occurrence,
                },
            )
            self.history.record_event(
                state,
                "agent_status",
                {
                    "action_index": action_index,
                    "situation": selected_action.status.situation,
                    "action": selected_action.status.action,
                    "reason": selected_action.status.reason,
                    "importance": selected_action.status.importance,
                    "importance_rank": selected_action.status.importance_rank,
                },
            )
            for question in selected_action.questions:
                self.history.record_event(
                    state,
                    "agent_question",
                    {
                        "action_index": action_index,
                        "question": question.question,
                        "criticality": question.criticality,
                        "reason": question.reason,
                        "assumption_if_unanswered": question.assumption_if_unanswered,
                    },
                )

            if occurrence > 1:
                rejected_signature_counts[signature] = rejected_signature_counts.get(signature, 0) + 1
                rejected_count = rejected_signature_counts[signature]
                exact_calls = stable_json_dumps(action_payload.get("tool_calls", []), indent=None)
                edit_paths = [
                    str(call.get("arguments", {}).get("path", ""))
                    for call in action_payload.get("tool_calls", [])
                    if call.get("tool_name") == "edit_text" and call.get("arguments", {}).get("path")
                ]
                recovery_feedback = (
                    "This exact mechanical action was rejected because it repeats the immediately preceding action and would produce no new evidence. "
                    f"It has now been rejected {rejected_count} time(s). Exact rejected tool calls: {exact_calls}. "
                    "Do not emit these exact tool calls again. Choose a materially different next action using the evidence already returned: use materially different arguments or a different tool. "
                )
                if edit_paths:
                    recovery_feedback += (
                        "Because the rejected action contains edit_text, reread the current target file before proposing another edit if the prior edit failed or the expected old_text may be stale. "
                        f"Relevant edit target(s): {', '.join(edit_paths)}. "
                    )
                if rejected_count >= 3:
                    recovery_feedback += (
                        "Repeatedly retrying this signature is a hard no-progress loop. You MUST choose a different mechanical action now; do not merely rephrase status or assistant text. "
                    )
                recovery_feedback += "Do not modify tests or files the user explicitly forbade changing."
                self.history.record_event(
                    state,
                    "agent_action_rejected",
                    {
                        "action_index": action_index,
                        "validation_attempt": 0,
                        "reason": recovery_feedback,
                    },
                )
                if rejected_count > max(1, int(self.config.runtime.max_repeated_action_occurrences)):
                    return self._finish_turn(
                        state,
                        "I stopped because the model repeated the same rejected mechanical action beyond the configured no-progress limit.",
                        tool_results,
                        budget_reports,
                    )
                continue

            visible_observation_signatures = self._visible_observation_signatures(state)
            repeated_observation_calls: list[dict[str, Any]] = []
            for tool_call in selected_action.tool_calls:
                tool = self.tools.get(tool_call.tool_name)
                if not tool.repeated_observation_is_redundant:
                    continue
                observation_signature = stable_json_dumps(
                    {"tool_name": tool_call.tool_name, "arguments": tool_call.arguments},
                    indent=None,
                )
                if (
                    observation_signature in observation_signatures_since_state_change
                    and observation_signature in visible_observation_signatures
                ):
                    rejected_observation_counts[observation_signature] = rejected_observation_counts.get(observation_signature, 0) + 1
                    repeated_observation_calls.append(
                        {
                            "tool_name": tool_call.tool_name,
                            "arguments": tool_call.arguments,
                            "rejected_count": rejected_observation_counts[observation_signature],
                        }
                    )
            if repeated_observation_calls:
                exact_observations = stable_json_dumps(repeated_observation_calls, indent=None)
                recovery_feedback = (
                    "This action was rejected because it repeats observation calls whose exact results are already available and no state-changing tool has run since those observations. "
                    f"Already-observed calls: {exact_observations}. Do NOT issue these same observations again. "
                    "Use the evidence already returned. If the requested answer can be derived from that evidence, synthesize and return the final answer now instead of rereading. "
                    "If evidence is genuinely missing, inspect a different source or use materially different tool arguments. "
                )
                if any(int(item["rejected_count"]) >= 2 for item in repeated_observation_calls):
                    recovery_feedback += (
                        "This observation has already been rejected repeatedly, so another identical read/search/inspection is a hard no-progress loop. You MUST either answer from existing evidence or choose different evidence. "
                    )
                self.history.record_event(
                    state,
                    "agent_action_rejected",
                    {
                        "action_index": action_index,
                        "validation_attempt": 0,
                        "reason": recovery_feedback,
                    },
                )
                if any(
                    int(item["rejected_count"]) > max(1, int(self.config.runtime.max_repeated_action_occurrences))
                    for item in repeated_observation_calls
                ):
                    return self._finish_turn(
                        state,
                        "I stopped because the model repeated an already-visible observation beyond the configured no-progress limit.",
                        tool_results,
                        budget_reports,
                    )
                continue

            self._consume_pending_control_messages(
                state,
                pending_payloads=pending_payloads,
                selected_action=selected_action,
            )
            accepted_actions += 1

            if selected_action.tool_calls:
                self._record_message(
                    state,
                    Message(
                        role="assistant",
                        content=stable_json_dumps(action_payload, indent=2),
                        created_at=utc_now_iso(),
                        metadata={"internal_action": True, "action_index": action_index},
                    ),
                )
                if selected_action.assistant_message.strip():
                    self.history.record_event(
                        state,
                        "assistant_progress",
                        {
                            "action_index": action_index,
                            "assistant_text": selected_action.assistant_message.strip(),
                        },
                    )

                for tool_call_index, tool_call in enumerate(selected_action.tool_calls, start=1):
                    tool = self.tools.get(tool_call.tool_name)
                    effective_kind = tool.effective_kind(tool_call.arguments)
                    repeated_observation_is_redundant = tool.repeated_observation_is_redundant
                    self._heartbeat(state, phase="tool_execution", detail=f"running {tool_call.tool_name}", active_kind="tool", active_id=tool_call.tool_name)
                    result = self._execute_tool(
                        state,
                        ToolDecision(
                            action="call_tool",
                            response=selected_action.assistant_message,
                            tool_name=tool_call.tool_name,
                            tool_input=tool_call.arguments,
                        ),
                    )
                    tool_calls_used += 1
                    if repeated_observation_is_redundant:
                        observation_signatures_since_state_change.add(
                            stable_json_dumps(
                                {"tool_name": tool_call.tool_name, "arguments": tool_call.arguments},
                                indent=None,
                            )
                        )
                    elif result is not None and effective_kind in {"stateful", "side_effect"}:
                        observation_signatures_since_state_change.clear()
                    if result is not None:
                        tool_results.append(result)
                        if tool_call.tool_name == "load_tools":
                            newly_loaded = [
                                str(name)
                                for name in result.output.get("selected_tool_names", [])
                                if isinstance(name, str) and name
                            ]
                            loaded_tool_names.update(newly_loaded)
                            self.history.record_event(
                                state,
                                "tool_capabilities_loaded",
                                {
                                    "action_index": action_index,
                                    "requested_tool_names": list(tool_call.arguments.get("tool_names", [])),
                                    "loaded_tool_names": sorted(loaded_tool_names),
                                },
                            )
                        if tool_call.tool_name == "run_tests" and not bool(result.output.get("passed", False)):
                            recovery_feedback = (
                                "The run_tests result failed. Treat its exact stdout/stderr as evidence. Decide the next action from "
                                "the task and current evidence; do not assume the same verification command is required or sufficient."
                            )
                    self.history.record_event(
                        state,
                        "agent_tool_call_completed",
                        {
                            "action_index": action_index,
                            "tool_call_index": tool_call_index,
                            "tool_name": tool_call.tool_name,
                            "success": result is not None,
                        },
                    )
                continue

            has_blocking_question = any(question.criticality == "blocking" for question in selected_action.questions)
            if has_blocking_question:
                self._heartbeat(state, phase="waiting_for_user", detail="blocking user question", active_kind="question", active_id=str(action_index))
            completion = (
                {"complete": True, "reason": "blocking user input requested", "remaining_work": []}
                if has_blocking_question
                else self._evaluate_completion(state, original_request=original_request, selected_action=selected_action, tool_results=tool_results)
                if self.config.runtime.completion_evaluation_enabled
                else {"complete": True, "reason": "completion evaluation disabled", "remaining_work": []}
            )
            if not completion["complete"]:
                remaining = [str(item).strip() for item in completion.get("remaining_work", []) if str(item).strip()]
                recovery_feedback = (
                    "An independent semantic completion evaluation found that the user's objective is not complete. "
                    f"Reason: {str(completion.get('reason', '')).strip()}. "
                    + ("Remaining work: " + "; ".join(remaining) + ". " if remaining else "")
                    + "Continue with a materially useful next action; do not merely restate the candidate final answer."
                )
                self.history.record_event(state, "completion_rejected", {"action_index": action_index, "reason": str(completion.get("reason", "")), "remaining_work": remaining})
                continue

            self.history.record_event(
                state,
                "agent_action_terminal",
                {
                    "action_index": action_index,
                    "continue_loop": selected_action.continue_loop,
                    "silent_completion": selected_action.silent_completion,
                },
            )
            return self._finish_turn(
                state,
                selected_action.assistant_message.strip(),
                tool_results,
                budget_reports,
            )

        return self._finish_turn(
            state,
            "I stopped because the configured accepted-action or mechanical-retry limit was reached before completion.",
            tool_results,
            budget_reports,
        )

    @staticmethod
    def _visible_observation_signatures(state: SessionState) -> set[str]:
        visible: set[str] = set()
        for message in state.messages:
            if message.role != "tool" or not message.name:
                continue
            metadata = message.metadata if isinstance(message.metadata, dict) else {}
            arguments = metadata.get("validated_input", metadata.get("raw_input", {}))
            if not isinstance(arguments, dict):
                continue
            visible.add(
                stable_json_dumps(
                    {"tool_name": message.name, "arguments": arguments},
                    indent=None,
                )
            )
        return visible

    def _consume_pending_control_messages(
        self,
        state: SessionState,
        *,
        pending_payloads: list[dict[str, Any]],
        selected_action: AgentAction,
    ) -> None:
        for payload in pending_payloads:
            control_id = str(payload.get("control_id", ""))
            message = str(payload.get("message", "")).strip()
            if not control_id or not message:
                continue
            # Controls are not user messages. They are injected through the dedicated
            # pending-control prompt channel and preserved exactly in this event.
            decision = asdict(selected_action)
            self.history.record_event(
                state,
                "control_message_processed",
                {
                    "control_id": control_id,
                    "session_id": state.session_id,
                    "message": message,
                    "decision": decision,
                    "selected_action": decision,
                },
            )
            self.history.mark_control_message_processed(state.session_id, control_id)

    def _prepare_action_call(
        self,
        state: SessionState,
        *,
        original_request: str,
        pending_messages: list[str],
        tool_specs: list[tuple[str, str, dict, str]],
        capability_index: list[tuple[str, str, str]] | None = None,
        contract: ContractSpec,
        validation_feedback: str,
        minimum_output_tokens: int | None = None,
    ) -> PreparedCall:
        last_report: BudgetReport | None = None
        tool_result_projections: dict[int, str] = {}
        max_rounds = max(0, int(self.config.context.max_compaction_rounds))
        for compaction_round in range(max_rounds + 1):
            counter = self._counter(state)
            context_components = self._runtime_context_components(state, counter)
            assembly = self.prompts.build_agent_action_prompt(
                list(state.messages),
                tool_specs,
                original_request=original_request,
                pending_user_messages=pending_messages,
                prompt_mode="standard",
                context_components=context_components,
                capability_index=capability_index,
                tool_result_projections=tool_result_projections,
                validation_feedback=validation_feedback,
            )
            compilation = self._compile_context(
                state,
                assembly,
                contract,
                minimum_output_tokens=(
                    self.config.context.reserved_response_tokens
                    if minimum_output_tokens is None
                    else minimum_output_tokens
                ),
            )
            report = compilation.report
            last_report = report
            self.history.record_event(
                state,
                "context_compiled",
                {
                    "kind": "action",
                    "prompt_mode": "standard",
                    "accounting": compilation.accounting(),
                    "cap_error": "" if report.fits else "context_limit_exceeded",
                },
            )
            self.history.record_event(
                state,
                "budget_checked",
                {
                    "kind": "action",
                    "prompt_mode": "standard",
                    "budget_report": asdict(report),
                    "cap_error": "" if report.fits else "context_limit_exceeded",
                },
            )
            if report.fits:
                self._record_prompt_built(state, assembly, contract, report)
                return PreparedCall(
                    assembly=assembly,
                    report=report,
                    prompt_mode="standard",
                    contract=contract,
                )
            if not self.config.context.compact_on_overflow or compaction_round >= max_rounds:
                break
            projected = self._project_largest_tool_result_for_overflow(
                state,
                original_request=original_request,
                assembly=assembly,
                compilation=compilation,
                existing_projections=tool_result_projections,
            )
            if projected is not None:
                sequence, projection = projected
                tool_result_projections[sequence] = projection
                continue
            if not self._compact_once(state):
                break

        raise BudgetExceededError(
            "The exact action prompt, tool schemas, output reserve, and safety margin do not fit the model context.",
            last_report,
        )

    def _evaluate_completion(
        self,
        state: SessionState,
        *,
        original_request: str,
        selected_action: AgentAction,
        tool_results: list[ToolExecutionResult],
    ) -> dict[str, Any]:
        self._heartbeat(
            state,
            phase="completion_evaluation",
            detail="evaluating task completion",
            active_kind="completion_evaluation",
        )
        contract = completion_evaluation_contract()
        evidence_rows = self._completion_evidence_rows(state, tool_results)
        projections: dict[int, str] = {}
        last_compilation: ContextCompilation | None = None
        max_rounds = max(0, int(self.config.context.max_compaction_rounds))
        for reduction_round in range(max_rounds + 1):
            assembly = self.prompts.build_completion_evaluation_prompt(
                original_request=original_request,
                assistant_message=selected_action.assistant_message,
                status_json=stable_json_dumps(asdict(selected_action.status), indent=None),
                tool_evidence_rows=evidence_rows,
                tool_result_projections=projections,
            )
            compilation = self._compile_context(
                state, assembly, contract, minimum_output_tokens=128
            )
            last_compilation = compilation
            self.history.record_event(
                state,
                "context_compiled",
                {
                    "kind": "completion_evaluation",
                    "prompt_mode": "lean",
                    "accounting": compilation.accounting(),
                    "cap_error": "" if compilation.report.fits else "context_limit_exceeded",
                    "reduction_round": reduction_round,
                },
            )
            self.history.record_event(
                state,
                "budget_checked",
                {
                    "kind": "completion_evaluation",
                    "prompt_mode": "lean",
                    "budget_report": asdict(compilation.report),
                    "cap_error": "" if compilation.report.fits else "context_limit_exceeded",
                },
            )
            if compilation.report.fits:
                self._record_prompt_built(state, assembly, contract, compilation.report)
                payload = self._execute_structured_call(
                    state,
                    PreparedCall(assembly, compilation.report, "lean", contract),
                )
                result = {
                    "complete": bool(payload.get("complete", False)),
                    "reason": str(payload.get("reason", "")).strip(),
                    "remaining_work": [
                        str(item)
                        for item in payload.get("remaining_work", [])
                        if isinstance(item, str)
                    ],
                    "evidence_source_references": [
                        reference
                        for row in evidence_rows
                        for reference in row.get("source_event_references", [])
                    ],
                    "projected_source_event_sequences": sorted(projections),
                }
                self.history.record_event(state, "completion_evaluated", result)
                return result
            if (
                not self.config.context.compact_on_overflow
                or reduction_round >= max_rounds
            ):
                break
            projected = self._project_largest_tool_result_for_overflow(
                state,
                original_request=original_request,
                assembly=assembly,
                compilation=compilation,
                existing_projections=projections,
            )
            if projected is None:
                break
            sequence, projection = projected
            projections[sequence] = projection

        report = asdict(last_compilation.report) if last_compilation is not None else None
        self.history.record_event(
            state,
            "completion_evaluation_unavailable",
            {
                "reason": "evaluation_context_does_not_fit_after_semantic_reduction",
                "budget_report": report,
                "projected_source_event_sequences": sorted(projections),
            },
        )
        return {
            "complete": False,
            "reason": "The completion evaluator could not fit its evidence context after bounded semantic reduction.",
            "remaining_work": [
                "Retrieve or semantically reduce additional evidence before deciding completion."
            ],
        }

    @staticmethod
    def _completion_evidence_rows(
        state: SessionState,
        tool_results: list[ToolExecutionResult],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        current_turn_start = 0
        for index in range(len(state.messages) - 1, -1, -1):
            if state.messages[index].role == "user":
                current_turn_start = index + 1
                break
        tool_messages = [
            message
            for message in state.messages[current_turn_start:]
            if message.role == "tool" and message.name
        ]
        matched_result_indices: set[int] = set()
        for message in tool_messages:
            metadata = message.metadata if isinstance(message.metadata, dict) else {}
            output = metadata.get("output")
            for result_index, result in enumerate(tool_results):
                if result_index in matched_result_indices:
                    continue
                if message.name == result.tool_name and to_jsonable(output) == to_jsonable(result.output):
                    matched_result_indices.add(result_index)
                    break
            sequence = metadata.get("source_event_sequence")
            source_hash = metadata.get("source_event_hash")
            nested_references = metadata.get("source_event_references", [])
            source_references = list(nested_references) if isinstance(nested_references, list) else []
            if isinstance(sequence, int) and isinstance(source_hash, str) and source_hash:
                source_references = [
                    {
                        "session_id": str(metadata.get("source_event_session_id", state.session_id)),
                        "sequence": sequence,
                        "hash": source_hash,
                        "event_type": str(metadata.get("source_event_type", "tool_result")),
                    },
                    *source_references,
                ]
            rows.append(
                {
                    "tool_name": message.name,
                    "output": to_jsonable(output),
                    "display_text": message.content,
                    "success": str(metadata.get("source_event_type", "")) == "tool_result",
                    "source_event_sequence": sequence,
                    "source_event_hash": source_hash,
                    "source_event_references": source_references,
                }
            )
        for result_index, result in enumerate(tool_results):
            if result_index in matched_result_indices:
                continue
            rows.append(
                {
                    "tool_name": result.tool_name,
                    "output": to_jsonable(result.output),
                    "display_text": result.display_text,
                    "success": True,
                    "source_event_sequence": None,
                    "source_event_hash": None,
                    "source_event_references": [],
                }
            )
        return rows

    def _project_largest_tool_result_for_overflow(
        self,
        state: SessionState,
        *,
        original_request: str,
        assembly: PromptAssembly,
        compilation: ContextCompilation,
        existing_projections: dict[int, str],
    ) -> tuple[int, str] | None:
        overflow = compilation.overflow_tokens
        if overflow <= 0:
            return None
        report_by_name = {item.name: item for item in compilation.report.breakdown}
        candidates: list[tuple[int, int, Message]] = []
        for message in state.messages:
            if message.role != "tool":
                continue
            sequence = message.metadata.get("source_event_sequence")
            if not isinstance(sequence, int):
                continue
            existing_projection = existing_projections.get(sequence)
            if existing_projection is not None:
                projected_text_tokens = self._counter(state).count_text(existing_projection).tokens
                if projected_text_tokens <= 64:
                    continue
            matching_tokens = 0
            suffix = f"_tool_event_{sequence}"
            for component in assembly.components:
                if component.name.endswith(suffix):
                    item = report_by_name.get(component.name)
                    if item is not None:
                        matching_tokens = int(item.tokens)
                    break
            if matching_tokens > 0:
                candidates.append((matching_tokens, sequence, message))
        if not candidates:
            return None
        current_tokens, sequence, message = max(candidates, key=lambda item: item[0])
        # Reduce enough to clear the measured overflow plus a small deterministic
        # cushion for JSON/provenance framing, while preserving useful room when possible.
        target_tokens = max(64, current_tokens - overflow - max(32, int(current_tokens * 0.05)))
        if sequence in existing_projections:
            projected_text_tokens = self._counter(state).count_text(
                existing_projections[sequence]
            ).tokens
            target_tokens = min(
                target_tokens,
                max(64, projected_text_tokens - max(16, overflow)),
            )
        if target_tokens >= current_tokens:
            return None
        stored_projection = self._stored_tool_result_projection(
            state,
            source_event_sequence=sequence,
            source_event_hash=str(message.metadata.get("source_event_hash", "")),
            target_tokens=target_tokens,
        )
        if stored_projection is not None:
            projection_event_sequence, projection, projected_tokens = stored_projection
            self.history.record_event(
                state,
                "tool_result_projection_reused",
                {
                    "source_event_sequence": sequence,
                    "source_event_hash": str(message.metadata.get("source_event_hash", "")),
                    "source_event_references": message.metadata.get(
                        "source_event_references", []
                    ),
                    "projection_event_sequence": projection_event_sequence,
                    "target_tokens": target_tokens,
                    "projected_tokens": projected_tokens,
                },
            )
            return sequence, projection
        projection = self._create_tool_result_projection(
            state,
            original_request=original_request,
            message=message,
            target_tokens=target_tokens,
            original_tokens=current_tokens,
            overflow_tokens=overflow,
        )
        if not projection:
            return None
        return sequence, projection

    def _stored_tool_result_projection(
        self,
        state: SessionState,
        *,
        source_event_sequence: int,
        source_event_hash: str,
        target_tokens: int,
    ) -> tuple[int, str, int] | None:
        event = self.history.latest_tool_result_projection(
            state.session_id,
            source_event_sequence=source_event_sequence,
            source_event_hash=source_event_hash,
            max_projected_tokens=target_tokens,
        )
        if event is not None:
            projection = str(event.payload.get("projection", "")).strip()
            projected_tokens = event.payload.get("projected_tokens")
            if projection and isinstance(projected_tokens, int):
                return event.sequence, projection, projected_tokens
        return None

    def _create_tool_result_projection(
        self,
        state: SessionState,
        *,
        original_request: str,
        message: Message,
        target_tokens: int,
        original_tokens: int,
        overflow_tokens: int,
    ) -> str:
        sequence = message.metadata.get("source_event_sequence")
        source_hash = str(message.metadata.get("source_event_hash", ""))
        if not isinstance(sequence, int):
            return ""
        contract = tool_result_projection_contract()
        assembly = self.prompts.build_tool_result_projection_prompt(
            original_request=original_request,
            tool_name=message.name or "tool",
            raw_tool_result=message.content,
            source_event_sequence=sequence,
            source_event_hash=source_hash,
            target_tokens=target_tokens,
        )
        compilation = self._compile_context(
            state,
            assembly,
            contract,
            minimum_output_tokens=min(target_tokens + 64, self.config.context.reserved_response_tokens),
        )
        if not compilation.report.fits:
            self.history.record_event(
                state,
                "tool_result_projection_skipped",
                {
                    "source_event_sequence": sequence,
                    "source_event_hash": source_hash,
                    "reason": "projection_prompt_does_not_fit",
                    "target_tokens": target_tokens,
                    "original_tokens": original_tokens,
                    "overflow_tokens": overflow_tokens,
                    "budget_report": asdict(compilation.report),
                },
            )
            return ""
        self.history.record_event(
            state,
            "context_compiled",
            {
                "kind": "tool_result_projection",
                "prompt_mode": "lean",
                "accounting": compilation.accounting(),
            },
        )
        self._record_prompt_built(state, assembly, contract, compilation.report)
        payload = self._execute_structured_call(
            state,
            PreparedCall(assembly, compilation.report, "lean", contract),
        )
        projection = str(payload.get("projection", "")).strip()
        if not projection:
            return ""
        projected_tokens = self._counter(state).count_text(projection).tokens
        self.history.record_event(
            state,
            "tool_result_projected",
            {
                "source_event_sequence": sequence,
                "source_event_hash": source_hash,
                "source_event_references": message.metadata.get(
                    "source_event_references", []
                ),
                "tool_name": message.name or "tool",
                "target_tokens": target_tokens,
                "original_tokens": original_tokens,
                "projected_tokens": projected_tokens,
                "overflow_tokens": overflow_tokens,
                "projection": projection,
            },
        )
        return projection

    def _runtime_context_components(
        self,
        state: SessionState,
        counter: ExactTokenCounter | ConservativeEstimator | _HistoryAwareTokenCounter,
    ) -> list[PromptComponent]:
        wakeup_store = WakeupStore(self.config.sessions.root)
        latest_handles: dict[str, str] = {}
        latest_artifact_cursor: dict[str, object] = {}
        for event in self.history.iter_history(state.session_id):
            if event.event_type != "tool_result":
                continue
            output = event.payload.get("output", {})
            if not isinstance(output, dict):
                continue
            for key in (
                "stdout_artifact_id",
                "stderr_artifact_id",
                "artifact_id",
                "terminal_id",
                "process_id",
                "wakeup_id",
            ):
                value = output.get(key)
                if isinstance(value, str) and value.strip():
                    latest_handles[key] = value.strip()
            if event.payload.get("tool_name") == "read_artifact":
                artifact_id = output.get("artifact_id")
                next_offset = output.get("next_offset")
                finished = output.get("finished")
                if isinstance(artifact_id, str) and artifact_id.strip() and isinstance(next_offset, int):
                    latest_artifact_cursor = {
                        "artifact_id": artifact_id.strip(),
                        "next_offset": next_offset,
                        "finished": bool(finished),
                    }
        environment = {
            "active_session": {
                "session_id": state.session_id,
                "session_name": state.session_name,
            },
            "workspace_root": state.environment.workspace.root,
            "cwd": state.environment.workspace.cwd,
            "workspace_files": list(state.environment.workspace.listed_files),
            "workspace_listing_truncated": state.environment.workspace.listing_truncated,
            "waiting": state.environment.waiting,
            "waiting_reason": state.environment.waiting_reason,
            "processes": {
                process_id: to_jsonable(record)
                for process_id, record in sorted(state.environment.processes.items())
            },
            "scheduled_wakeups": [
                to_jsonable(item) for item in wakeup_store.list(session_id=state.session_id)
            ],
            "latest_handles": latest_handles,
            "latest_artifact_cursor": latest_artifact_cursor,
        }
        components = [
            PromptComponent(
                name="environment_state",
                category="environment",
                text="Environment state:\n" + stable_json_dumps(environment, indent=2) + "\n\n",
            )
        ]
        if state.attachments:
            references = []
            for attachment in state.attachments:
                payload = asdict(attachment)
                payload.pop("storage_ref", None)
                references.append(payload)
            components.append(
                PromptComponent(
                    name="attachment_references",
                    category="attachments",
                    text=(
                        "Raw attachments available to this task. These are references and cheap mechanical facts only; "
                        "decide whether and how to inspect them with an attachment capability:\n"
                        + stable_json_dumps(references, indent=2)
                        + "\n\n"
                    ),
                )
            )
        selected = select_notes_for_prompt(self.config, state.notes, counter)
        self.history.record_event(
            state,
            "notes_selected",
            {
                "included_note_ids": [note.note_id for note in selected.included_notes],
                "omitted_note_ids": selected.omitted_note_ids,
                "tokens": selected.tokens,
                "exact": selected.exact,
            },
        )
        if selected.rendered_text:
            components.append(
                PromptComponent(
                    name="durable_notes",
                    category="notes",
                    text=(
                        "Durable model-authored notes. These are navigation aids; verbatim user messages and tool results remain authoritative:\n"
                        + selected.rendered_text
                        + "\n\n"
                    ),
                    optional=True,
                )
            )
        return components

    def _compile_context(
        self,
        state: SessionState | None,
        assembly: PromptAssembly,
        contract: ContractSpec,
        *,
        minimum_output_tokens: int,
        context_limit_resolution: tuple[int, str] | None = None,
    ) -> ContextCompilation:
        context_limit, context_limit_source = (
            self._resolve_context_limit()
            if context_limit_resolution is None
            else context_limit_resolution
        )
        return self.context_compiler.compile(
            assembly,
            contract,
            self._counter(state),
            minimum_output_tokens=minimum_output_tokens,
            context_limit=context_limit,
            context_limit_source=context_limit_source,
        )

    def _resolve_context_limit(self) -> tuple[int, str]:
        resolver = getattr(self.client, "context_limit_resolution", None)
        if callable(resolver):
            value, source = resolver()
        else:
            server_resolver = getattr(self.client, "server_context_limit", None)
            if callable(server_resolver):
                value, source = server_resolver(), "server_props:n_ctx"
            else:
                value, source = self.config.model.context_limit, "configured"
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ModelClientError(f"Invalid resolved model context limit: {value!r}")
        return int(value), str(source)

    def _budget_report(
        self,
        state: SessionState | None,
        assembly: PromptAssembly,
        contract: ContractSpec,
    ) -> BudgetReport:
        return self._compile_context(
            state,
            assembly,
            contract,
            minimum_output_tokens=self.config.context.reserved_response_tokens,
        ).report

    def _record_prompt_built(
        self,
        state: SessionState,
        assembly: PromptAssembly,
        contract: ContractSpec,
        report: BudgetReport,
    ) -> None:
        self.history.record_event(
            state,
            "prompt_built",
            {
                "kind": assembly.kind,
                "prompt_mode": assembly.prompt_mode,
                "contract": contract.name,
                "prompt": assembly.prompt_text,
                "components": [asdict(component) for component in assembly.components],
                "budget_report": asdict(report),
            },
        )

    def _compact_once(self, state: SessionState) -> bool:
        if len(state.messages) <= 2:
            return False
        keep = max(2, min(int(self.config.context.max_recent_messages), len(state.messages) - 1))
        maximum_source = len(state.messages) - keep
        if maximum_source <= 0:
            return False

        contract = summary_contract()
        context_limit_resolution = self._resolve_context_limit()
        for source_count in range(maximum_source, 0, -1):
            source_messages = state.messages[:source_count]
            adaptive_cap = min(max(0, source_count - 1), max(0, int(self.config.context.max_recent_messages) * 4))
            summary_context_limit, _summary_context_source = context_limit_resolution
            summary_plan = self.context_compiler.plan(call_kind="summary", context_limit=summary_context_limit)
            target_summary_tokens = max(
                int(self.config.context.reserved_summary_tokens),
                int(summary_plan.output_tokens) - 32,
            )
            assembly = self.prompts.build_summary_prompt(
                source_messages,
                prompt_mode="lean",
                maximum_preserve_recent_messages=adaptive_cap,
                target_summary_tokens=target_summary_tokens,
            )
            compilation = self._compile_context(
                state,
                assembly,
                contract,
                minimum_output_tokens=self.config.context.reserved_summary_tokens,
                context_limit_resolution=context_limit_resolution,
            )
            report = compilation.report
            if not report.fits:
                continue
            self.history.record_event(
                state,
                "context_compiled",
                {
                    "kind": "summary",
                    "prompt_mode": "lean",
                    "accounting": compilation.accounting(),
                    "target_summary_tokens": target_summary_tokens,
                },
            )
            self._record_prompt_built(state, assembly, contract, report)
            payload = self._execute_structured_call(state, PreparedCall(assembly, report, "lean", contract))
            summary_text = str(payload.get("summary", "")).strip()
            if not summary_text:
                raise ValueError("summary must not be empty")
            preserve_recent = self._validated_preserve_recent_messages(
                payload.get("preserve_recent_messages", 0),
                source_count=source_count,
                maximum=adaptive_cap,
            )
            effective_source_count = source_count - preserve_recent
            source_event_references = message_source_event_references(
                source_messages[:effective_source_count]
            )
            summary_payload = summary_message_payload(
                summary_text,
                source_message_count=effective_source_count,
                created_at=utc_now_iso(),
                source_event_references=source_event_references,
            )
            event_payload = {
                "source_message_count": effective_source_count,
                "source_event_references": source_event_references,
                "source_event_ranges": summary_payload["metadata"]["source_event_ranges"],
                "summary_message": summary_payload,
                "summary_budget_report": asdict(report),
                "adaptive_preserve_recent_messages": preserve_recent,
                "candidate_source_message_count": source_count,
            }
            self.history.record_event(state, "summary_created", event_payload)
            self.history.record_event(state, "history_compressed", event_payload)
            return True
        return False


    @staticmethod
    def _validated_preserve_recent_messages(value: Any, *, source_count: int, maximum: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("preserve_recent_messages must be an integer")
        upper = min(max(0, int(maximum)), max(0, int(source_count) - 1))
        if value < 0 or value > upper:
            raise ValueError(f"preserve_recent_messages must be between 0 and {upper}")
        return value

    def _summary_budget_report(
        self,
        state: SessionState,
        assembly: PromptAssembly,
        contract: ContractSpec,
    ) -> BudgetReport:
        return self._compile_context(
            state,
            assembly,
            contract,
            minimum_output_tokens=self.config.context.reserved_summary_tokens,
        ).report

    def _execute_structured_call(
        self,
        state: SessionState,
        prepared: PreparedCall,
        *,
        validator: Callable[[dict[str, Any]], Any] | None = None,
        seed_offset: int = 0,
    ) -> Any:
        completion = self._execute_model_call(state, prepared, seed_offset=seed_offset)
        if completion.finish_reason in {"length", "context_overflow"}:
            self.history.record_event(
                state,
                "model_output_budget_exhausted",
                {
                    "kind": prepared.assembly.kind,
                    "finish_reason": completion.finish_reason,
                    "reserved_response_tokens": prepared.report.reserved_response_tokens,
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                },
            )
            raise OutputBudgetExhaustedError(
                completion.finish_reason,
                prepared.report.reserved_response_tokens,
            )
        try:
            payload = json.loads(completion.text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Contract {prepared.contract.name} returned malformed JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Contract {prepared.contract.name} must return one JSON object")
        return validator(payload) if validator is not None else payload

    @staticmethod
    def _expanded_output_minimum(prepared: PreparedCall) -> int:
        current = int(prepared.report.reserved_response_tokens)
        ceiling = max(
            current,
            int(prepared.report.context_limit)
            - int(prepared.report.safety_margin_tokens),
        )
        return min(ceiling, max(current + 64, (current * 3 + 1) // 2))

    def _execute_model_call(
        self,
        state: SessionState,
        prepared: PreparedCall,
        *,
        seed_offset: int = 0,
    ) -> CompletionResult:
        resolved_contract, policy = self.client.resolve_contract(
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
        # Reproducible but non-identical decoding across semantic action/retry attempts.
        # Reusing one fixed seed caused malformed JSON and exact bad actions to recur
        # deterministically even after validation feedback changed.
        request["seed"] = int(self.config.model.seed) + int(seed_offset)
        call_id = new_id("model_call")
        self._heartbeat(state, phase="queued_inference", detail=f"queued {prepared.assembly.kind}", active_kind="model", active_id=call_id)
        active_call = self.preemption.register_active(
            state.session_id,
            call_id,
            prepared.assembly.kind,
            request,
        )
        frozen_request = active_call.request
        transient_attempts = 0
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
                    "request": frozen_request,
                    "budget_report": asdict(prepared.report),
                    "policy": asdict(policy),
                    "token_timeout_seconds": policy.effective_timeout_seconds,
                    "requested_contract_mode": prepared.contract.mode,
                    "effective_contract_mode": resolved_contract.mode,
                },
            )
            self._heartbeat(state, phase="inference", detail=f"running {prepared.assembly.kind}", active_kind="model", active_id=call_id)
            started = time.monotonic()
            last_progress_log = started
            last_progress_tokens = 0

            def progress_callback(progress: dict[str, Any]) -> None:
                nonlocal last_progress_log, last_progress_tokens
                now = time.monotonic()
                tokens = int(progress.get("completion_tokens", 0) or 0)
                if (
                    now - last_progress_log < float(policy.progress_poll_seconds)
                    and tokens < last_progress_tokens + 50
                ):
                    return
                last_progress_log = now
                last_progress_tokens = tokens
                self._heartbeat(state, phase="inference", detail=f"{prepared.assembly.kind}: {tokens} completion tokens", active_kind="model", active_id=call_id)
                guard.record(
                    "model_token_progress",
                    {
                        "kind": prepared.assembly.kind,
                        "prompt_mode": prepared.prompt_mode,
                        "attempt": total_attempt,
                        "elapsed_seconds": float(progress.get("elapsed_seconds", now - started)),
                        "completion_tokens": tokens,
                        "tokens_per_second": float(progress.get("tokens_per_second", 0.0) or 0.0),
                        "first_token_seconds": progress.get("first_token_seconds"),
                        "token_timeout_seconds": progress.get(
                            "token_timeout_seconds",
                            policy.effective_timeout_seconds,
                        ),
                    },
                )

            try:
                send = self.client.send_completion
                try:
                    signature = inspect.signature(send)
                    has_var_kwargs = any(
                        item.kind == inspect.Parameter.VAR_KEYWORD
                        for item in signature.parameters.values()
                    )
                    supports_progress = "progress_callback" in signature.parameters or has_var_kwargs
                    supports_cancel = "cancel_check" in signature.parameters or has_var_kwargs
                except (TypeError, ValueError):
                    supports_progress = False
                    supports_cancel = False
                kwargs: dict[str, Any] = {"timeout_seconds": policy.effective_timeout_seconds}
                if supports_progress:
                    kwargs["progress_callback"] = progress_callback
                if supports_cancel:
                    kwargs["cancel_check"] = lambda: (
                        self._run_cancellation_requested(state)
                        or self.preemption.pending_for_call(state.session_id, call_id) is not None
                    )
                completion = send(frozen_request, **kwargs)
            except ModelCallPreempted:
                if self._run_cancellation_requested(state):
                    run_id = self._active_run_id(state)
                    guard.record(
                        "model_call_preempted",
                        {
                            "kind": prepared.assembly.kind,
                            "prompt_mode": prepared.prompt_mode,
                            "attempt": total_attempt,
                            "call_id": call_id,
                            "preemption_id": f"run_cancellation:{run_id}",
                            "request_sha256": active_call.request_sha256,
                            "reason": "run_cancellation_requested",
                        },
                    )
                    self.preemption.clear_active(state.session_id, call_id)
                    raise RunCancellationRequested("worker run cancellation requested")
                pending = self.preemption.pending_for_call(state.session_id, call_id)
                if pending is None:
                    self.preemption.clear_active(state.session_id, call_id)
                    raise
                guard.record(
                    "model_call_preempted",
                    {
                        "kind": prepared.assembly.kind,
                        "prompt_mode": prepared.prompt_mode,
                        "attempt": total_attempt,
                        "call_id": call_id,
                        "preemption_id": pending.preemption_id,
                        "request_sha256": active_call.request_sha256,
                    },
                )
                # Publish the coordinator transition only after its canonical event.
                # Communication may append target changes as soon as it observes
                # "interrupted", so reversing this order creates a stale writer race.
                self.preemption.mark_interrupted(pending.preemption_id)
                resolved = self.preemption.wait_for_status(
                    pending.preemption_id,
                    {"completed", "failed"},
                    timeout_seconds=max(1.0, float(policy.effective_timeout_seconds)),
                    poll_seconds=0.02,
                )
                if resolved.status == "failed":
                    self.preemption.clear_active(state.session_id, call_id)
                    raise ModelClientError(f"communication preemption failed: {resolved.reply or 'unknown error'}")
                if resolved.target_changed:
                    self.preemption.clear_active(state.session_id, call_id)
                    self._refresh_state_from_history(state)
                    guard = self.history.guard(state, f"model_call:{prepared.assembly.kind}:preemption")
                    guard.record(
                        "model_call_replay_invalidated",
                        {
                            "kind": prepared.assembly.kind,
                            "call_id": call_id,
                            "preemption_id": pending.preemption_id,
                            "request_sha256": active_call.request_sha256,
                        },
                    )
                    raise ModelCallStateChanged("target session changed during communication; stale model request was not replayed")
                guard.record(
                    "model_call_replayed",
                    {
                        "kind": prepared.assembly.kind,
                        "call_id": call_id,
                        "preemption_id": pending.preemption_id,
                        "request_sha256": active_call.request_sha256,
                        "request": frozen_request,
                    },
                )
                continue
            except Exception as exc:
                if self._is_model_server_unavailable(exc):
                    transient_attempts += 1
                    guard.record(
                        "retry",
                        {
                            "operation": "model_unavailable",
                            "reason": str(exc),
                            "attempt": transient_attempts,
                            "next_attempt": transient_attempts + 1,
                        },
                    )
                    if transient_attempts > self._max_model_unavailable_attempts:
                        raise ModelClientError("model_unavailable") from exc
                    self._sleep(self._model_unavailable_backoff_seconds(transient_attempts - 1))
                    continue
                guard.record(
                    "model_call_failed",
                    {
                        "kind": prepared.assembly.kind,
                        "prompt_mode": prepared.prompt_mode,
                        "attempt": total_attempt,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                )
                guard.require_all("model_request_sent", "model_call_failed")
                if semantic_attempt < self.config.model.max_retries:
                    semantic_attempt += 1
                    guard.record(
                        "model_retry_scheduled",
                        {
                            "kind": prepared.assembly.kind,
                            "prompt_mode": prepared.prompt_mode,
                            "next_attempt": semantic_attempt + 1,
                        },
                    )
                    continue
                self.preemption.clear_active(state.session_id, call_id)
                raise

            self.preemption.clear_active(state.session_id, call_id)
            guard.record(
                "model_response_received",
                {
                    "kind": prepared.assembly.kind,
                    "prompt_mode": prepared.prompt_mode,
                    "attempt": total_attempt,
                    "completion": to_jsonable(completion),
                    "elapsed_seconds": completion.elapsed_seconds,
                    "completion_tokens": completion.completion_tokens,
                    "tokens_per_second": completion.tokens_per_second,
                    "first_token_seconds": completion.first_token_seconds,
                    "token_timeout_seconds": policy.effective_timeout_seconds,
                    "policy": asdict(policy),
                },
            )
            guard.require_all("model_request_sent", "model_response_received")
            guard.ensure_progress()
            print(
                "[model_response] "
                f"kind={prepared.assembly.kind} attempt={total_attempt} "
                f"elapsed={completion.elapsed_seconds}s tokens={completion.completion_tokens} "
                f"avg_tps={completion.tokens_per_second}",
                file=sys.stderr,
                flush=True,
            )
            return completion

    def _active_run_id(self, state: SessionState) -> str:
        active = self.history.read_active_run(state.session_id)
        return "" if active is None else str(active.get("run_id", ""))

    def _run_cancellation_requested(self, state: SessionState) -> bool:
        run_id = self._active_run_id(state)
        return bool(run_id) and self.preemption.cancellation_requested(state.session_id, run_id)

    def _raise_if_run_cancelled(self, state: SessionState) -> None:
        if self._run_cancellation_requested(state):
            raise RunCancellationRequested("worker run cancellation requested")

    def _execute_tool(self, state: SessionState, decision: ToolDecision) -> ToolExecutionResult | None:
        guard = self.history.guard(state, f"tool:{decision.tool_name}")
        guard.record(
            "tool_called",
            {"tool_name": decision.tool_name, "tool_input": decision.tool_input},
        )
        try:
            tool, context, invocation = self.tools.prepare(
                decision.tool_name,
                decision.tool_input,
                self.config,
                state,
            )
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
            tool_error_event = guard.record("tool_error", error_payload)
            guard.require_any("tool_called", "tool_error")
            self._record_message(
                state,
                Message(
                    role="tool",
                    name=decision.tool_name,
                    content=f"tool_error: {stable_json_dumps(error_payload, indent=2)}",
                    created_at=utc_now_iso(),
                    metadata={
                        **error_payload,
                        "source_event_sequence": tool_error_event.sequence,
                        "source_event_hash": tool_error_event.hash,
                        "source_event_type": tool_error_event.event_type,
                        "source_event_session_id": tool_error_event.session_id,
                    },
                ),
            )
            return None

        emitted_types: set[str] = set()
        for event in result.generated_events:
            emitted_types.add(event.event_type)
            guard.record(
                event.event_type,
                event.payload,
                metadata=event.metadata,
                derived_writes=event.derived_writes,
            )
        missing = tool.required_generated_event_types(invocation.validated_input) - emitted_types
        if missing:
            raise HistoryInvariantError(
                f"Tool {decision.tool_name} completed without required generated events: {', '.join(sorted(missing))}"
            )
        nested_source_references = result.output.get("source_event_references", [])
        if not isinstance(nested_source_references, list):
            nested_source_references = []
        tool_result_event = guard.record(
            "tool_result",
            {
                "tool_name": result.tool_name,
                "raw_input": invocation.raw_input,
                "validated_input": invocation.validated_input,
                "output": to_jsonable(result.output),
                "source_event_references": nested_source_references,
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
                    "source_event_sequence": tool_result_event.sequence,
                    "source_event_hash": tool_result_event.hash,
                    "source_event_type": tool_result_event.event_type,
                    "source_event_session_id": tool_result_event.session_id,
                    "source_event_references": nested_source_references,
                },
            ),
        )
        return result

    def execute_tool_once(
        self,
        tool_name: str,
        raw_input: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> ToolRunResult:
        state = self.create_or_load_session(session_id)
        result = self._execute_tool(
            state,
            ToolDecision(
                action="call_tool",
                response="",
                tool_name=tool_name,
                tool_input=raw_input,
            ),
        )
        return ToolRunResult(session_id=state.session_id, tool_result=result)

    def _finish_turn(
        self,
        state: SessionState,
        assistant_text: str,
        tool_results: list[ToolExecutionResult],
        budget_reports: list[BudgetReport],
    ) -> TurnResult:
        text = assistant_text.strip()
        self._record_message(
            state,
            Message(role="assistant", content=text, created_at=utc_now_iso()),
        )
        self.history.record_event(
            state,
            "turn_finished",
            {
                "turn_index": state.turn_count + 1,
                "assistant_text": text,
                "tool_steps": len(tool_results),
                "budget_reports": [asdict(item) for item in budget_reports],
            },
        )
        return TurnResult(
            session_id=state.session_id,
            assistant_text=text,
            tool_results=tool_results,
            budget_reports=budget_reports,
        )

    def _record_message(self, state: SessionState, message: Message) -> None:
        self.history.record_event(state, "message_added", {"message": asdict(message)})

    def _counter(
        self,
        state: SessionState | None,
    ) -> ExactTokenCounter | ConservativeEstimator | _HistoryAwareTokenCounter:
        if self._token_counter is not None:
            return self._token_counter
        if state is None:
            try:
                return ExactTokenCounter(self.client.tokenize)
            except Exception:
                return ConservativeEstimator()
        return _HistoryAwareTokenCounter(self, state)

    def _tokenize_with_history(self, state: SessionState, text: str) -> CountResult:
        text_hash = sha256_text(text)
        if text_hash in self._token_count_cache:
            return CountResult(
                tokens=self._token_count_cache[text_hash],
                exact=True,
                strategy="llama_cpp_server_cache",
            )
        guard = self.history.guard(state, "tokenize")
        guard.record(
            "model_tokenize_requested",
            {"text_hash": text_hash, "text_chars": len(text)},
        )
        try:
            tokens = int(self.client.tokenize(text))
        except Exception as exc:
            guard.record(
                "model_tokenize_failed",
                {
                    "text_hash": text_hash,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            )
            guard.require_all("model_tokenize_requested", "model_tokenize_failed")
            raise
        self._token_count_cache[text_hash] = tokens
        guard.record(
            "model_tokenize_result",
            {"text_hash": text_hash, "tokens": tokens, "exact": True},
        )
        guard.require_all("model_tokenize_requested", "model_tokenize_result")
        return CountResult(tokens=tokens, exact=True, strategy="llama_cpp_server")

    def _is_model_server_unavailable(self, error: BaseException) -> bool:
        if isinstance(error, (requests.ConnectionError, requests.Timeout)):
            return True
        if isinstance(error, requests.HTTPError):
            response = getattr(error, "response", None)
            return response is not None and getattr(response, "status_code", None) in {502, 503, 504}
        return False

    def _model_unavailable_backoff_seconds(self, attempt: int) -> float:
        return float(min(60, 2 ** min(max(attempt, 0), 6)))

    def session_status_payload(self, state: SessionState) -> dict[str, Any]:
        latest_user = next(
            (message.content for message in reversed(state.messages) if message.role == "user"),
            "",
        )
        running_processes = [
            {
                "process_id": process_id,
                "command": record.command,
                "status": record.status,
            }
            for process_id, record in sorted(state.environment.processes.items())
            if record.status == "running"
        ]
        active_run = self.history.read_active_run(state.session_id)
        return {
            "session_id": state.session_id,
            "session_name": state.session_name,
            "active_goal": latest_user,
            "active_step": "" if active_run is None else str(active_run.get("detail", "")),
            "mechanical_phase": "idle" if active_run is None else str(active_run.get("phase", "unknown")),
            "heartbeat_at": "" if active_run is None else str(active_run.get("heartbeat_at", "")),
            "active_run": active_run,
            "waiting": state.environment.waiting,
            "waiting_reason": state.environment.waiting_reason,
            "running_processes": running_processes,
            "pending_user_messages": len(
                self.history.list_pending_control_messages(state.session_id)
            ),
            "deferred_tasks": [asdict(item) for item in state.deferred_tasks],
            "checkpoint_count": len(state.code_checkpoints),
            "turn_count": state.turn_count,
            "event_count": state.event_count,
        }

    def queue_control_message(
        self,
        session_ref: str | None,
        message: str,
        *,
        source: str = "cli",
    ) -> dict[str, Any]:
        session_id = self.history.resolve_session_ref(session_ref, latest_if_none=False)
        if session_id is None and session_ref is None:
            active = [item for item in self.history.list_session_entries() if item.get("active")]
            if active:
                session_id = str(active[0]["session_id"])
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

    def query_history_details(
        self,
        *,
        session_ref: str | None,
        query_text: str,
        topic_hint: str = "",
    ) -> dict[str, Any]:
        policy = self.config.history_search
        return self.history.query_history_details(
            session_ref,
            query_text,
            topic_hint=topic_hint,
            max_results=policy.max_results,
            token_score=policy.token_score,
            exact_score=policy.exact_score,
            type_bonus=policy.type_bonus,
            preview_chars=policy.preview_chars,
        )

    def create_code_checkpoint(
        self,
        state: SessionState,
        *,
        label: str = "",
        workspace_root: str | None = None,
    ) -> dict[str, Any]:
        environment = AgentEnvironment(self.config, state)
        root = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root
            else environment.filesystem.workspace_root.resolve()
        )
        checkpoint_id = new_id("checkpoint")
        checkpoint_dir = self.history.code_checkpoints_dir(state.session_id) / checkpoint_id
        files_dir = ensure_dir(checkpoint_dir / "files")
        sessions_root = self.config.sessions.root
        if not sessions_root.is_absolute():
            sessions_root = (root / sessions_root).resolve()
        manifest = snapshot_tree(
            root,
            files_dir,
            excluded_roots=(checkpoint_dir, sessions_root),
        )
        payload = {
            "checkpoint_id": checkpoint_id,
            "label": label.strip() or f"checkpoint-{len(state.code_checkpoints) + 1}",
            "created_at": utc_now_iso(),
            "workspace_root": str(root),
            "storage_path": str(checkpoint_dir),
            "file_count": len(manifest),
            "metadata": {"manifest_path": str(checkpoint_dir / "manifest.json")},
        }
        write_text(
            checkpoint_dir / "manifest.json",
            stable_json_dumps({"workspace_root": str(root), "files": manifest}, indent=2),
        )
        self.history.record_event(state, "code_checkpoint_created", {"checkpoint": payload})
        return payload

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
        root = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root
            else Path(checkpoint.workspace_root).resolve()
        )
        checkpoint_dir = Path(checkpoint.storage_path)
        manifest_payload = json.loads(
            (checkpoint_dir / "manifest.json").read_text(encoding="utf-8")
        )
        snapshot_files = {str(item) for item in manifest_payload.get("files", [])}
        sessions_root = self.config.sessions.root
        if not sessions_root.is_absolute():
            sessions_root = (root / sessions_root).resolve()
        restore_tree(
            checkpoint_dir / "files",
            root,
            snapshot_files,
            excluded_roots=(sessions_root,),
        )
        self.history.record_event(
            state,
            "code_checkpoint_restored",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "restored_to": checkpoint.label,
                "workspace_root": str(root),
            },
        )
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "label": checkpoint.label,
            "workspace_root": str(root),
            "file_count": len(snapshot_files),
        }

    def budget_demo(self, user_text: str, *, prompt_mode: str = "standard") -> dict[str, Any]:
        state = SessionState(
            session_id="budget_demo",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            config_fingerprint=self.config.config_fingerprint(),
            model_base_url=self.config.model.base_url,
            messages=[Message(role="user", content=user_text, created_at=utc_now_iso())],
        )
        tool_specs = self.tools.prompt_tuples(self.config)
        contract = agent_action_contract(tool_specs)
        assembly = self.prompts.build_agent_action_prompt(
            state.messages,
            tool_specs,
            original_request=user_text,
            pending_user_messages=[],
            prompt_mode=prompt_mode,
            context_components=[],
        )
        report = self._budget_report(None, assembly, contract)
        return {
            "action": {
                "prompt_mode": prompt_mode,
                "budget": asdict(report),
                "prompt": assembly.prompt_text,
                "contract": contract.name,
                "schema": contract.json_schema,
            }
        }

    def doctor(self, *, session_id: str | None = None) -> dict[str, Any]:
        state = self.create_or_load_session(session_id)
        self.history.record_event(
            state,
            "model_request_sent",
            {
                "kind": "doctor_health",
                "prompt_mode": "n/a",
                "attempt": 1,
                "request": {"endpoint": "health"},
                "budget_report": None,
            },
        )
        health = self.client.health()
        self.history.record_event(state, "doctor_health_checked", {"health": health})
        token_count = self._tokenize_with_history(state, "doctor probe").tokens
        self.history.record_event(
            state,
            "doctor_tokenize_checked",
            {"probe": "doctor probe", "tokens": token_count},
        )
        contract = yes_no_contract()
        assembly = self.prompts._assemble(
            "doctor",
            "lean",
            [
                PromptComponent(
                    name="doctor",
                    category="instruction",
                    text='Return one JSON object with "answer":"yes".',
                )
            ],
        )
        report = self._budget_report(state, assembly, contract)
        self._record_prompt_built(state, assembly, contract, report)
        payload = self._execute_structured_call(
            state,
            PreparedCall(assembly, report, "lean", contract),
        )
        if payload.get("answer") != "yes":
            raise ValueError("Doctor constrained-output probe did not return yes")
        return {
            "session_id": state.session_id,
            "health": health,
            "tokenize_probe_tokens": token_count,
            "json_probe": "yes",
            "schema_probe": payload,
        }


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
                {
                    "text_hash": sha256_text(text),
                    "tokens": estimate.tokens,
                    "strategy": estimate.strategy,
                },
            )
            return estimate
