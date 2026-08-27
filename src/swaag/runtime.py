from __future__ import annotations

import copy
import inspect
import json
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import requests

from swaag.action import ActionValidationError, AgentAction, action_from_payload
from swaag.attachments import AttachmentStore, find_attachment
from swaag.compression import message_source_event_references, summary_message_payload
from swaag.context_compiler import ContextCompilation, ContextCompiler
from swaag.config import AgentConfig, load_config
from swaag.embedding_index import (
    AsyncEmbeddingIndexer,
    DerivedEmbeddingIndex,
    OpenAICompatibleEmbeddingProvider,
)
from swaag.environment.environment import AgentEnvironment
from swaag.environment.artifacts import TextArtifactStore
from swaag.fsops import ensure_dir, restore_tree, snapshot_tree, write_text
from swaag.grammar import (
    agent_action_contract,
    audio_rendering_contract,
    communication_status_contract,
    completion_evaluation_contract,
    evidence_projection_contract,
    prompt_instruction_projection_contract,
    presentation_evaluation_contract,
    response_relevance_contract,
    summary_contract,
    tool_result_projection_contract,
    yes_no_contract,
)
from swaag.history import HistoryInvariantError, HistoryStore
from swaag.heartbeat import heartbeat_payload, systemd_notify
from swaag.inference import InferenceRequest, InferenceRequestCoordinator
from swaag.model import LlamaCppClient, ModelClientError, uses_chat_completions_transport
from swaag.preemption import (
    ModelCallPreempted,
    ModelCallStateChanged,
    ModelPreemptionCoordinator,
    RunCancellationRequested,
)
from swaag.prompt_instructions import (
    prompt_instructions_for_kind,
)
from swaag.prompt_instruction_store import PromptInstructionStore
from swaag.model_cache import build_model_client
from swaag.notes import render_notes
from swaag.prompts import PromptBuilder
from swaag.scheduler import WakeupStore
from swaag.schema_portability import assert_portable_json_schema
from swaag.telemetry import OperationalTelemetry, TelemetryOperation
from swaag.tokens import ConservativeEstimator, CountResult, ExactTokenCounter, build_budget
from swaag.tools.base import (
    SemanticCallContextOverflow,
    SemanticCallRequest,
    _validate_schema_value,
)
from swaag.tools.registry import ToolRegistry
from swaag.types import (
    AttachmentReference,
    BudgetReport,
    CompletionResult,
    ContractSpec,
    DeferredTask,
    HistoryEvent,
    Message,
    ModelCallKind,
    PromptAssembly,
    PromptArtifact,
    PromptComponent,
    PromptMessageRange,
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


class _OutputRecoveryContextOverflow(BudgetExceededError):
    def __init__(
        self,
        compilation: ContextCompilation,
        minimum_output_tokens: int,
    ):
        super().__init__(
            "The reconstructed call needs more output space than the current exact input permits",
            compilation.report,
        )
        self.compilation = compilation
        self.minimum_output_tokens = int(minimum_output_tokens)


def _validated_caller_output(
    payload: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    _validate_schema_value(payload, schema, path="caller_structured_output")
    return payload


def _validated_presentation_text(
    payload: dict[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    text = payload.get(field_name)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if field_name == "answer":
        omitted = payload.get("omitted_as_irrelevant")
        if not isinstance(omitted, list) or not all(
            isinstance(item, str) for item in omitted
        ):
            raise ValueError("omitted_as_irrelevant must be an array of strings")
    return payload


def _validated_presentation_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload.get("acceptable"), bool):
        raise ValueError("presentation acceptable must be boolean")
    if not isinstance(payload.get("reason"), str):
        raise ValueError("presentation reason must be a string")
    for field_name in (
        "missing_or_changed_information",
        "irrelevant_operational_details",
    ):
        value = payload.get(field_name)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"{field_name} must be an array of strings")
    return payload


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
    error: dict[str, Any] | None = None


@dataclass(slots=True)
class PreparedCall:
    assembly: PromptAssembly
    report: BudgetReport
    prompt_mode: str
    contract: ContractSpec


@dataclass(slots=True, frozen=True)
class RuntimeContextProjection:
    source_sha256: str
    text: str


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
        telemetry: OperationalTelemetry | None = None,
    ):
        self.config = config
        self.context_compiler = ContextCompiler(config)
        self.client = model_client or build_model_client(
            config,
            request_metadata={"cache_scope": "default_agent_runtime"},
        )
        self.preemption = ModelPreemptionCoordinator(config.sessions.root)
        self.prompt_instruction_store = PromptInstructionStore(
            config.sessions.root,
            config,
        )
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
        self.telemetry = telemetry or OperationalTelemetry()
        self._inference_context = threading.local()
        self.inference = InferenceRequestCoordinator(
            config.sessions.root,
            backend_key=self._inference_backend_key(),
            capacity_resolver=self._inference_capacity,
        )
        self._token_counter = token_counter
        self._token_count_cache: dict[str, int] = {}
        self._sleep = time.sleep
        self._max_model_unavailable_attempts: int = max(1, int(self.config.model.max_retries) + 1)

    @classmethod
    def from_config_paths(cls, config_paths: list[str] | None = None) -> AgentRuntime:
        return cls(load_config(config_paths))

    def _inference_backend_key(self) -> str:
        if getattr(self.client, "is_deterministic_test_client", False):
            return f"deterministic:{id(self.client)}"
        if getattr(self.client, "mode", "") == "replay":
            cassette = getattr(self.client, "cassette_path", "")
            return f"replay:{Path(cassette).expanduser()}"
        return self.config.model.base_url.rstrip("/")

    def _inference_capacity(self) -> tuple[int, str]:
        if getattr(self.client, "is_deterministic_test_client", False):
            return 128, "deterministic_test_client"
        if getattr(self.client, "mode", "") == "replay":
            return 128, "recorded_replay"
        resolver = getattr(self.client, "server_slot_count", None)
        if callable(resolver):
            try:
                slots = int(resolver())
                if slots > 0:
                    return slots, "server_props:total_slots"
            except Exception:
                pass
        return 1, "conservative_fallback"

    @contextmanager
    def inference_priority(
        self,
        priority: int,
        *,
        source: str,
    ) -> Iterator[None]:
        previous = getattr(self._inference_context, "value", None)
        self._inference_context.value = (int(priority), str(source))
        try:
            yield
        finally:
            if previous is None:
                try:
                    del self._inference_context.value
                except AttributeError:
                    pass
            else:
                self._inference_context.value = previous

    def _current_inference_priority(self) -> tuple[int, str]:
        value = getattr(self._inference_context, "value", None)
        if isinstance(value, tuple) and len(value) == 2:
            return int(value[0]), str(value[1])
        return 0, "worker"

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
        with self.telemetry.agent_invocation(
            session_id=state.session_id,
            run_id=run_id,
            model_name=self.config.model.model_identity,
        ):
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
        with self.telemetry.agent_invocation(
            session_id=state.session_id,
            run_id=run_id,
            model_name=self.config.model.model_identity,
        ):
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
        with self.telemetry.agent_invocation(
            session_id=state.session_id,
            run_id=run_id,
            model_name=self.config.model.model_identity,
        ):
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

    @contextmanager
    def _periodic_model_heartbeat(
        self,
        state: SessionState,
        *,
        call_id: str,
        call_kind: str,
        interval_seconds: float = 5.0,
    ) -> Iterator[None]:
        """Keep mechanical liveness current while a backend has not streamed output."""
        stop = threading.Event()

        def pulse() -> None:
            while not stop.wait(max(0.01, float(interval_seconds))):
                self._heartbeat(
                    state,
                    phase="inference",
                    detail=f"waiting for {call_kind} model stream",
                    active_kind="model",
                    active_id=call_id,
                )

        thread = threading.Thread(
            target=pulse,
            name=f"swaag-heartbeat-{call_id}",
            daemon=True,
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=max(1.0, float(interval_seconds) + 0.5))

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
        runtime_context_projections: dict[str, RuntimeContextProjection] = {}
        remaining_runtime_projection_calls = [
            max(8, int(self.config.context.max_compaction_rounds) * 8)
        ]
        max_rounds = max(0, int(self.config.context.max_compaction_rounds))
        for compaction_round in range(max_rounds + 1):
            counter = self._counter(state)
            context_components = self._runtime_context_components(
                state,
                counter,
                projections=runtime_context_projections,
            )
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
            projected_context = self._project_runtime_context_for_overflow(
                state,
                original_request=stable_json_dumps(
                    {
                        "original_request": original_request,
                        "pending_messages": pending_messages,
                        "validation_feedback": validation_feedback,
                    },
                    indent=None,
                ),
                compilation=compilation,
                existing_projections=runtime_context_projections,
                remaining_calls=remaining_runtime_projection_calls,
            )
            if projected_context is not None:
                source_name, projection = projected_context
                runtime_context_projections[source_name] = projection
                continue
            if not self._compact_once(
                state,
                required_recovery_tokens=compilation.overflow_tokens,
            ):
                break

        effective_minimum = (
            self.config.context.reserved_response_tokens
            if minimum_output_tokens is None
            else minimum_output_tokens
        )
        recovered = self._recover_prompt_instruction_overflow(
            state,
            assembly,
            contract,
            compilation,
            minimum_output_tokens=effective_minimum,
        )
        if recovered is not None:
            self.history.record_event(
                state,
                "context_compiled",
                {
                    "kind": "action",
                    "prompt_mode": "standard",
                    "accounting": recovered.accounting(),
                    "cap_error": "",
                    "prompt_instruction_projection": True,
                },
            )
            self.history.record_event(
                state,
                "budget_checked",
                {
                    "kind": "action",
                    "prompt_mode": "standard",
                    "budget_report": asdict(recovered.report),
                    "cap_error": "",
                    "prompt_instruction_projection": True,
                },
            )
            self._record_prompt_built(
                state,
                assembly,
                contract,
                recovered.report,
            )
            return PreparedCall(
                assembly=assembly,
                report=recovered.report,
                prompt_mode="standard",
                contract=contract,
            )

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
        evidence_rows = self._completion_evidence_rows(state, tool_results)
        history_snapshot = self.history.read_history(state.session_id)
        current_user_sequence = next(
            (
                event.sequence
                for event in reversed(history_snapshot)
                if event.event_type == "message_added"
                and isinstance(event.payload.get("message"), dict)
                and event.payload["message"].get("role") == "user"
            ),
            None,
        )
        historical_events = (
            [
                event
                for event in history_snapshot
                if current_user_sequence is not None
                and event.sequence < current_user_sequence
            ]
            if current_user_sequence is not None
            else []
        )
        historical_rows = [
            self._communication_evidence_row(event) for event in historical_events
        ]
        historical_source_references = [
            self._communication_evidence_reference(event)
            for event in historical_events
        ]
        evidence_source_inventory = self._completion_evidence_source_inventory(
            state,
            evidence_rows=evidence_rows,
            historical_events=historical_events,
        )
        historical_evidence = (
            stable_json_dumps(historical_rows, indent=None)
            if historical_rows
            else ""
        )
        historical_evidence_tokens = (
            self._counter(state).count_text(historical_evidence).tokens
            if historical_evidence
            else 0
        )
        historical_evidence_projection = ""
        historical_projection_target_tokens: int | None = None
        historical_projection_budget_report: dict[str, Any] | None = None
        remaining_historical_projection_calls = [
            max(16, int(self.config.context.max_compaction_rounds) * 16)
        ]
        context_limit_resolution = self._resolve_context_limit()
        projections: dict[int, str] = {}
        reexpanded_evidence: dict[str, dict[str, Any]] = {}
        reexpanded_evidence_projections: dict[str, str] = {}
        last_compilation: ContextCompilation | None = None
        minimum_output_tokens = 128
        max_rounds = max(0, int(self.config.context.max_compaction_rounds))
        reduction_round = 0

        def validate_completion_payload(payload: dict[str, Any]) -> dict[str, Any]:
            if contract.json_schema is not None:
                _validate_schema_value(
                    payload,
                    contract.json_schema,
                    path=contract.name,
                )
            return payload

        while reduction_round <= max_rounds:
            available_sources = [
                item
                for item in evidence_source_inventory
                if self._completion_evidence_source_key(item)
                not in reexpanded_evidence
            ]
            contract = completion_evaluation_contract(
                (
                    str(item["source_kind"]),
                    str(item["source_id"]),
                )
                for item in available_sources
            )
            assembly = self.prompts.build_completion_evaluation_prompt(
                original_request=original_request,
                assistant_message=selected_action.assistant_message,
                status_json=stable_json_dumps(asdict(selected_action.status), indent=None),
                tool_evidence_rows=evidence_rows,
                tool_result_projections=projections,
                historical_evidence=(
                    "" if historical_evidence_projection else historical_evidence
                ),
                historical_evidence_projection=historical_evidence_projection,
                evidence_source_inventory=available_sources,
                reexpanded_evidence_rows=list(reexpanded_evidence.values()),
                reexpanded_evidence_projections=reexpanded_evidence_projections,
            )
            compilation = self._compile_context(
                state,
                assembly,
                contract,
                minimum_output_tokens=minimum_output_tokens,
                context_limit_resolution=context_limit_resolution,
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
            if (
                not compilation.report.fits
                and (
                    not self.config.context.compact_on_overflow
                    or reduction_round >= max_rounds
                    or (
                        not historical_evidence
                        and not evidence_rows
                        and not reexpanded_evidence
                    )
                )
            ):
                recovered = self._recover_prompt_instruction_overflow(
                    state,
                    assembly,
                    contract,
                    compilation,
                    minimum_output_tokens=minimum_output_tokens,
                    context_limit_resolution=context_limit_resolution,
                )
                if recovered is not None:
                    compilation = recovered
                    last_compilation = recovered
                    self.history.record_event(
                        state,
                        "context_compiled",
                        {
                            "kind": "completion_evaluation",
                            "prompt_mode": "lean",
                            "accounting": recovered.accounting(),
                            "cap_error": "",
                            "reduction_round": reduction_round,
                            "prompt_instruction_projection": True,
                        },
                    )
                    self.history.record_event(
                        state,
                        "budget_checked",
                        {
                            "kind": "completion_evaluation",
                            "prompt_mode": "lean",
                            "budget_report": asdict(recovered.report),
                            "cap_error": "",
                            "prompt_instruction_projection": True,
                        },
                    )
            if compilation.report.fits:
                self._record_prompt_built(state, assembly, contract, compilation.report)
                try:
                    payload, _final_prepared = self._execute_with_output_recovery(
                        state,
                        PreparedCall(
                            assembly,
                            compilation.report,
                            "lean",
                            contract,
                        ),
                        minimum_output_tokens=minimum_output_tokens,
                        validator=validate_completion_payload,
                        context_limit_resolution=context_limit_resolution,
                    )
                except _OutputRecoveryContextOverflow as exc:
                    compilation = exc.compilation
                    last_compilation = compilation
                    minimum_output_tokens = exc.minimum_output_tokens
                else:
                    requested_sources = payload.get("evidence_requests", [])
                    if requested_sources:
                        inventory_by_key = {
                            self._completion_evidence_source_key(item): item
                            for item in available_sources
                        }
                        for request in requested_sources:
                            source_key = self._completion_evidence_request_key(request)
                            if source_key in reexpanded_evidence:
                                continue
                            source = inventory_by_key.get(source_key)
                            if source is None:
                                raise HistoryInvariantError(
                                    "Completion evaluator requested evidence outside its constrained inventory"
                                )
                            expanded = self._reexpand_completion_evidence_source(
                                state,
                                source=source,
                                purpose=str(request.get("purpose", "")).strip(),
                            )
                            reexpanded_evidence[source_key] = expanded
                            self.history.record_event(
                                state,
                                "completion_evidence_reexpanded",
                                {
                                    key: value
                                    for key, value in expanded.items()
                                    if key != "text"
                                }
                                | {
                                    "purpose": str(
                                        request.get("purpose", "")
                                    ).strip(),
                                    "exact_chars": len(
                                        str(expanded.get("text", ""))
                                    ),
                                },
                            )
                        continue
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
                        ]
                        + historical_source_references,
                        "historical_source_event_references": historical_source_references,
                        "historical_evidence_projected": bool(
                            historical_evidence_projection
                        ),
                        "historical_projection_target_tokens": (
                            historical_projection_target_tokens
                        ),
                        "historical_evidence_projection": (
                            historical_evidence_projection
                        ),
                        "historical_projection_budget_report": (
                            historical_projection_budget_report
                        ),
                        "projected_source_event_sequences": sorted(projections),
                        "reexpanded_evidence_sources": [
                            {
                                key: value
                                for key, value in row.items()
                                if key != "text"
                            }
                            | {
                                "projected": source_key
                                in reexpanded_evidence_projections
                            }
                            for source_key, row in reexpanded_evidence.items()
                        ],
                    }
                    self.history.record_event(state, "completion_evaluated", result)
                    return result
            if (
                not self.config.context.compact_on_overflow
                or reduction_round >= max_rounds
            ):
                break
            report_by_name = {
                item.name: item.tokens for item in compilation.report.breakdown
            }
            historical_component_tokens = int(
                report_by_name.get("completion_historical_evidence", 0)
            )
            largest_tool_component_tokens = max(
                (
                    int(item.tokens)
                    for item in compilation.report.breakdown
                    if item.name.startswith("completion_tool_event_")
                ),
                default=0,
            )
            expanded_component_tokens = [
                (
                    int(
                        report_by_name.get(
                            f"completion_reexpanded_evidence_{index}", 0
                        )
                    ),
                    source_key,
                )
                for index, source_key in enumerate(reexpanded_evidence, start=1)
            ]
            largest_expanded_tokens, largest_expanded_key = max(
                expanded_component_tokens,
                default=(0, ""),
            )
            if largest_expanded_tokens >= max(
                historical_component_tokens,
                largest_tool_component_tokens,
            ):
                expanded_projection = (
                    self._project_completion_evidence_source_for_overflow(
                        state,
                        original_request=original_request,
                        source_key=largest_expanded_key,
                        source_row=reexpanded_evidence.get(
                            largest_expanded_key, {}
                        ),
                        current_tokens=largest_expanded_tokens,
                        overflow_tokens=compilation.overflow_tokens,
                        existing_projection=reexpanded_evidence_projections.get(
                            largest_expanded_key
                        ),
                        context_limit_resolution=context_limit_resolution,
                    )
                )
                if expanded_projection is not None:
                    reexpanded_evidence_projections[largest_expanded_key] = (
                        expanded_projection
                    )
                    reduction_round += 1
                    continue
                largest_expanded_tokens = 0
            if (
                historical_evidence
                and historical_component_tokens
                >= max(largest_tool_component_tokens, largest_expanded_tokens)
            ):
                placeholder = self.prompts.build_completion_evaluation_prompt(
                    original_request=original_request,
                    assistant_message=selected_action.assistant_message,
                    status_json=stable_json_dumps(
                        asdict(selected_action.status), indent=None
                    ),
                    tool_evidence_rows=evidence_rows,
                    tool_result_projections=projections,
                    evidence_source_inventory=available_sources,
                    reexpanded_evidence_rows=list(reexpanded_evidence.values()),
                    reexpanded_evidence_projections=(
                        reexpanded_evidence_projections
                    ),
                    historical_evidence_projection=(
                        "[purpose-specific historical evidence projection]"
                    ),
                )
                base_compilation = self._compile_context(
                    state,
                    placeholder,
                    contract,
                    minimum_output_tokens=minimum_output_tokens,
                    context_limit_resolution=context_limit_resolution,
                )
                projection_capacity = (
                    base_compilation.available_input_tokens
                    - base_compilation.report.input_tokens
                )
                if projection_capacity >= 32:
                    reduction = max(32, compilation.overflow_tokens + 16)
                    previous_target = (
                        historical_evidence_tokens
                        if historical_projection_target_tokens is None
                        else historical_projection_target_tokens
                    )
                    historical_projection_target_tokens = min(
                        projection_capacity,
                        max(32, previous_target - reduction),
                    )
                    historical_evidence_projection, projection_report = (
                        self._reduce_text_hierarchically(
                            state,
                            source_text=historical_evidence,
                            source_label=(
                                "exact durable history events before the current user turn"
                            ),
                            target_tokens=historical_projection_target_tokens,
                            contract=evidence_projection_contract(),
                            output_key="projection",
                            build_assembly=lambda text, label, target: (
                                self.prompts.build_evidence_projection_prompt(
                                    purpose=(
                                        "Evaluate whether this user objective is complete: "
                                        + original_request
                                    ),
                                    source_label=label,
                                    raw_evidence=text,
                                    target_tokens=target,
                                )
                            ),
                            remaining_calls=remaining_historical_projection_calls,
                            context_limit_resolution=context_limit_resolution,
                        )
                    )
                    historical_projection_budget_report = asdict(projection_report)
                    reduction_round += 1
                    continue
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
            reduction_round += 1

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
    def _communication_evidence_row(event: HistoryEvent) -> dict[str, Any]:
        return {
            "session_id": event.session_id,
            "sequence": event.sequence,
            "hash": event.hash,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "payload": to_jsonable(event.payload),
            "metadata": to_jsonable(event.metadata),
        }

    @staticmethod
    def _communication_evidence_reference(event: HistoryEvent) -> dict[str, Any]:
        return {
            "session_id": event.session_id,
            "sequence": event.sequence,
            "hash": event.hash,
            "event_type": event.event_type,
        }

    @staticmethod
    def _communication_status_prompt_state(
        mechanical_status: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        compact = to_jsonable(mechanical_status)
        if not isinstance(compact, dict):
            raise TypeError("mechanical_status must serialize to an object")
        semantic: dict[str, Any] = {}

        active_goal = compact.pop("active_goal", "")
        if isinstance(active_goal, str) and active_goal:
            semantic["active_goal"] = active_goal
            compact["active_goal_reference"] = {
                "chars": len(active_goal),
                "sha256": sha256_text(active_goal),
            }

        active_run = compact.get("active_run")
        if isinstance(active_run, dict):
            run_text = active_run.pop("user_text", "")
            if isinstance(run_text, str) and run_text:
                semantic["active_run_user_text"] = run_text
                active_run["user_text_reference"] = {
                    "chars": len(run_text),
                    "sha256": sha256_text(run_text),
                }
        return compact, semantic

    def generate_communication_status(
        self,
        *,
        target_session_id: str,
        question: str,
        mechanical_status: dict[str, Any],
        source_events: list[HistoryEvent],
    ) -> dict[str, Any]:
        """Interpret a worker snapshot in an independent, separately budgeted call."""
        status_question = question.strip()
        if not status_question:
            raise ValueError("status question must not be empty")
        operation_state = self.create_or_load_session(
            new_id("operation_communication_status")
        )
        run_id = f"{operation_state.session_id}:{new_id('run')}"
        source_references = [
            self._communication_evidence_reference(event) for event in source_events
        ]
        self.history.set_active_run(
            operation_state.session_id,
            run_id=run_id,
            user_text=status_question,
        )
        self.history.record_event(
            operation_state,
            "communication_status_requested",
            {
                "target_session_id": target_session_id,
                "question": status_question,
                "mechanical_status": to_jsonable(mechanical_status),
                "source_event_references": source_references,
            },
        )
        self._heartbeat(
            operation_state,
            run_id=run_id,
            phase="semantic_status",
            detail=f"interpreting status for {target_session_id}",
        )
        with self.telemetry.agent_invocation(
            session_id=operation_state.session_id,
            run_id=run_id,
            model_name=self.config.model.model_identity,
        ):
            try:
                result = self._generate_communication_status(
                    operation_state,
                    target_session_id=target_session_id,
                    question=status_question,
                    mechanical_status=mechanical_status,
                    source_events=source_events,
                )
                self.history.record_event(
                    operation_state,
                    "communication_status_generated",
                    {
                        "target_session_id": target_session_id,
                        "question": status_question,
                        "status": result,
                        "mechanical_status": to_jsonable(mechanical_status),
                        "source_event_references": source_references,
                        "evidence_projected": bool(result["evidence_projected"]),
                    },
                )
                self._heartbeat(
                    operation_state,
                    run_id=run_id,
                    phase="completed",
                    detail="semantic status completed",
                )
                return result
            except RunCancellationRequested as exc:
                self.preemption.complete_run_cancellation(
                    operation_state.session_id, run_id
                )
                self._heartbeat(
                    operation_state,
                    run_id=run_id,
                    phase="cancelled",
                    detail=str(exc),
                )
                raise
            except Exception as exc:
                self.history.record_event(
                    operation_state,
                    "communication_status_unavailable",
                    {
                        "target_session_id": target_session_id,
                        "question": status_question,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "source_event_references": source_references,
                    },
                )
                self._heartbeat(
                    operation_state,
                    run_id=run_id,
                    phase="failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                raise
            finally:
                self.history.clear_active_run(
                    operation_state.session_id, run_id=run_id
                )

    def _generate_communication_status(
        self,
        state: SessionState,
        *,
        target_session_id: str,
        question: str,
        mechanical_status: dict[str, Any],
        source_events: list[HistoryEvent],
    ) -> dict[str, Any]:
        contract = communication_status_contract()
        evidence_rows = [
            self._communication_evidence_row(event) for event in source_events
        ]
        prompt_mechanical_status, runtime_semantic_evidence = (
            self._communication_status_prompt_state(mechanical_status)
        )
        source_references = [
            self._communication_evidence_reference(event) for event in source_events
        ]
        valid_sequences = {event.sequence for event in source_events}
        exact_evidence = stable_json_dumps(
            {
                "runtime_semantic_evidence": runtime_semantic_evidence,
                "durable_events": evidence_rows,
            },
            indent=None,
        )
        exact_evidence_tokens = self._counter(state).count_text(exact_evidence).tokens
        context_limit_resolution = self._resolve_context_limit()
        minimum_output_tokens = 128
        desired_output_tokens = min(512, context_limit_resolution[0])
        evidence_projection = ""
        evidence_projected = False
        projection_target_tokens: int | None = None
        projection_budget_report: dict[str, Any] | None = None
        validation_feedback = ""
        validation_attempt = 0
        reduction_round = 0
        max_validation_attempts = max(1, int(self.config.model.max_retries) + 1)
        max_reduction_rounds = max(0, int(self.config.context.max_compaction_rounds))
        remaining_projection_calls = [max(16, max_reduction_rounds * 16)]

        def validate(payload: dict[str, Any]) -> dict[str, Any]:
            if contract.json_schema is not None:
                _validate_schema_value(
                    payload,
                    contract.json_schema,
                    path="communication_status",
                )
            for key in ("answer", "situation", "action", "reason"):
                if not str(payload.get(key, "")).strip():
                    raise ValueError(f"communication_status.{key} must not be empty")
            cited = payload.get("evidence_sequences", [])
            unknown = sorted({int(sequence) for sequence in cited} - valid_sequences)
            if unknown:
                raise ValueError(
                    "communication_status.evidence_sequences cites unavailable "
                    f"target event sequences: {unknown}"
                )
            return payload

        while True:
            assembly = self.prompts.build_communication_status_prompt(
                question=question,
                mechanical_status=prompt_mechanical_status,
                evidence_rows=[] if evidence_projected else evidence_rows,
                runtime_semantic_evidence=(
                    None if evidence_projected else runtime_semantic_evidence
                ),
                evidence_projection=evidence_projection,
                validation_feedback=validation_feedback,
            )
            compilation = self._compile_context(
                state,
                assembly,
                contract,
                minimum_output_tokens=minimum_output_tokens,
                desired_output_tokens=desired_output_tokens,
                context_limit_resolution=context_limit_resolution,
            )
            cap_error = "" if compilation.report.fits else "context_limit_exceeded"
            self.history.record_event(
                state,
                "context_compiled",
                {
                    "kind": "communication_status",
                    "prompt_mode": "lean",
                    "accounting": compilation.accounting(),
                    "cap_error": cap_error,
                    "reduction_round": reduction_round,
                    "validation_attempt": validation_attempt,
                },
            )
            self.history.record_event(
                state,
                "budget_checked",
                {
                    "kind": "communication_status",
                    "prompt_mode": "lean",
                    "budget_report": asdict(compilation.report),
                    "cap_error": cap_error,
                    "reduction_round": reduction_round,
                    "validation_attempt": validation_attempt,
                },
            )
            if (
                not compilation.report.fits
                and (
                    not self.config.context.compact_on_overflow
                    or reduction_round >= max_reduction_rounds
                    or (not evidence_rows and not runtime_semantic_evidence)
                )
            ):
                recovered = self._recover_prompt_instruction_overflow(
                    state,
                    assembly,
                    contract,
                    compilation,
                    minimum_output_tokens=minimum_output_tokens,
                    desired_output_tokens=desired_output_tokens,
                    context_limit_resolution=context_limit_resolution,
                )
                if recovered is not None:
                    compilation = recovered
                    self.history.record_event(
                        state,
                        "context_compiled",
                        {
                            "kind": "communication_status",
                            "prompt_mode": "lean",
                            "accounting": recovered.accounting(),
                            "cap_error": "",
                            "reduction_round": reduction_round,
                            "validation_attempt": validation_attempt,
                            "prompt_instruction_projection": True,
                        },
                    )
                    self.history.record_event(
                        state,
                        "budget_checked",
                        {
                            "kind": "communication_status",
                            "prompt_mode": "lean",
                            "budget_report": asdict(recovered.report),
                            "cap_error": "",
                            "reduction_round": reduction_round,
                            "validation_attempt": validation_attempt,
                            "prompt_instruction_projection": True,
                        },
                    )
            if compilation.report.fits:
                self._record_prompt_built(
                    state, assembly, contract, compilation.report
                )
                try:
                    payload, final_prepared = self._execute_with_output_recovery(
                        state,
                        PreparedCall(
                            assembly,
                            compilation.report,
                            "lean",
                            contract,
                        ),
                        minimum_output_tokens=minimum_output_tokens,
                        desired_output_tokens=desired_output_tokens,
                        validator=validate,
                        context_limit_resolution=context_limit_resolution,
                    )
                except _OutputRecoveryContextOverflow as exc:
                    compilation = exc.compilation
                    minimum_output_tokens = exc.minimum_output_tokens
                except ValueError as exc:
                    validation_attempt += 1
                    self.history.record_event(
                        state,
                        "communication_status_rejected",
                        {
                            "target_session_id": target_session_id,
                            "attempt": validation_attempt,
                            "reason": str(exc),
                        },
                    )
                    if validation_attempt >= max_validation_attempts:
                        raise
                    validation_feedback = str(exc)
                    continue
                else:
                    importance = str(payload["importance"])
                    importance_rank = {
                        "minor": 1,
                        "normal": 2,
                        "major": 3,
                        "critical": 4,
                    }[importance]
                    cited_sequences = sorted(
                        {int(sequence) for sequence in payload["evidence_sequences"]}
                    )
                    return {
                        "answer": str(payload["answer"]).strip(),
                        "situation": str(payload["situation"]).strip(),
                        "action": str(payload["action"]).strip(),
                        "reason": str(payload["reason"]).strip(),
                        "importance": importance,
                        "importance_rank": importance_rank,
                        "evidence_sequences": cited_sequences,
                        "uncertainty": str(payload["uncertainty"]).strip(),
                        "target_session_id": target_session_id,
                        "generated_at": utc_now_iso(),
                        "source_event_references": source_references,
                        "evidence_projected": evidence_projected,
                        "projection_target_tokens": projection_target_tokens,
                        "projection_budget_report": projection_budget_report,
                        "status_budget_report": asdict(final_prepared.report),
                    }

            if (
                not self.config.context.compact_on_overflow
                or reduction_round >= max_reduction_rounds
                or (not evidence_rows and not runtime_semantic_evidence)
            ):
                raise BudgetExceededError(
                    "Communication status evidence does not fit after bounded semantic reduction",
                    compilation.report,
                )

            placeholder = self.prompts.build_communication_status_prompt(
                question=question,
                mechanical_status=prompt_mechanical_status,
                evidence_rows=[],
                runtime_semantic_evidence=None,
                evidence_projection="[purpose-specific evidence projection]",
                validation_feedback=validation_feedback,
            )
            base_compilation = self._compile_context(
                state,
                placeholder,
                contract,
                minimum_output_tokens=minimum_output_tokens,
                desired_output_tokens=desired_output_tokens,
                context_limit_resolution=context_limit_resolution,
            )
            projection_capacity = (
                base_compilation.available_input_tokens
                - base_compilation.report.input_tokens
            )
            if projection_capacity < 32:
                raise BudgetExceededError(
                    "Communication status has no room for a semantic evidence projection",
                    base_compilation.report,
                )
            reduction = max(32, compilation.overflow_tokens + 16)
            previous_target = (
                exact_evidence_tokens
                if projection_target_tokens is None
                else projection_target_tokens
            )
            projection_target_tokens = min(
                projection_capacity,
                max(32, previous_target - reduction),
            )
            projection, projection_report = self._reduce_text_hierarchically(
                state,
                source_text=exact_evidence,
                source_label=(
                    f"exact authoritative evidence for target session {target_session_id}"
                ),
                target_tokens=projection_target_tokens,
                contract=evidence_projection_contract(),
                output_key="projection",
                build_assembly=lambda text, label, target: (
                    self.prompts.build_evidence_projection_prompt(
                        purpose=question,
                        source_label=label,
                        raw_evidence=text,
                        target_tokens=target,
                    )
                ),
                remaining_calls=remaining_projection_calls,
                context_limit_resolution=context_limit_resolution,
            )
            evidence_projection = projection
            evidence_projected = True
            projection_budget_report = asdict(projection_report)
            reduction_round += 1

    def generate_caller_structured_output(
        self,
        state: SessionState,
        *,
        original_request: str,
        assistant_message: str,
        tool_results: list[ToolExecutionResult],
        semantic_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Run caller-output generation as its own cancellable inference lifecycle."""
        run_id = f"{state.session_id}:{new_id('run')}"
        self.history.set_active_run(
            state.session_id,
            run_id=run_id,
            user_text=original_request,
        )
        try:
            self._heartbeat(
                state,
                run_id=run_id,
                phase="structured_output",
                detail="generating caller-defined semantic fields",
            )
            output = self._generate_caller_structured_output(
                state,
                original_request=original_request,
                assistant_message=assistant_message,
                tool_results=tool_results,
                semantic_schema=semantic_schema,
            )
            self._heartbeat(
                state,
                run_id=run_id,
                phase="completed",
                detail="caller-defined semantic fields completed",
            )
            return output
        except RunCancellationRequested as exc:
            self.preemption.complete_run_cancellation(state.session_id, run_id)
            self._heartbeat(
                state,
                run_id=run_id,
                phase="cancelled",
                detail=str(exc),
            )
            raise
        except Exception as exc:
            self._heartbeat(
                state,
                run_id=run_id,
                phase="failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            self.history.clear_active_run(state.session_id, run_id=run_id)

    def _generate_caller_structured_output(
        self,
        state: SessionState,
        *,
        original_request: str,
        assistant_message: str,
        tool_results: list[ToolExecutionResult],
        semantic_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate only caller-declared semantic fields under a constrained schema."""
        assert_portable_json_schema(
            semantic_schema, schema_name="caller_structured_output"
        )
        if not semantic_schema.get("properties"):
            return {}
        contract = ContractSpec(
            name="caller_structured_output",
            mode="json_schema",
            json_schema=semantic_schema,
        )
        evidence_rows = self._completion_evidence_rows(state, tool_results)
        projections: dict[int, str] = {}
        last_compilation: ContextCompilation | None = None
        minimum_output_tokens = 128
        max_rounds = max(0, int(self.config.context.max_compaction_rounds))
        for reduction_round in range(max_rounds + 1):
            assembly = self.prompts.build_caller_structured_output_prompt(
                original_request=original_request,
                assistant_message=assistant_message,
                tool_evidence_rows=evidence_rows,
                tool_result_projections=projections,
            )
            compilation = self._compile_context(
                state,
                assembly,
                contract,
                minimum_output_tokens=minimum_output_tokens,
            )
            last_compilation = compilation
            cap_error = "" if compilation.report.fits else "context_limit_exceeded"
            self.history.record_event(
                state,
                "context_compiled",
                {
                    "kind": "caller_structured_output",
                    "prompt_mode": "lean",
                    "accounting": compilation.accounting(),
                    "cap_error": cap_error,
                    "reduction_round": reduction_round,
                },
            )
            self.history.record_event(
                state,
                "budget_checked",
                {
                    "kind": "caller_structured_output",
                    "prompt_mode": "lean",
                    "budget_report": asdict(compilation.report),
                    "cap_error": cap_error,
                },
            )
            if (
                not compilation.report.fits
                and (
                    not self.config.context.compact_on_overflow
                    or reduction_round >= max_rounds
                    or not evidence_rows
                )
            ):
                recovered = self._recover_prompt_instruction_overflow(
                    state,
                    assembly,
                    contract,
                    compilation,
                    minimum_output_tokens=minimum_output_tokens,
                )
                if recovered is not None:
                    compilation = recovered
                    last_compilation = recovered
                    self.history.record_event(
                        state,
                        "context_compiled",
                        {
                            "kind": "caller_structured_output",
                            "prompt_mode": "lean",
                            "accounting": recovered.accounting(),
                            "cap_error": "",
                            "reduction_round": reduction_round,
                            "prompt_instruction_projection": True,
                        },
                    )
                    self.history.record_event(
                        state,
                        "budget_checked",
                        {
                            "kind": "caller_structured_output",
                            "prompt_mode": "lean",
                            "budget_report": asdict(recovered.report),
                            "cap_error": "",
                            "prompt_instruction_projection": True,
                        },
                    )
            if compilation.report.fits:
                self._record_prompt_built(state, assembly, contract, compilation.report)
                try:
                    payload, _final_prepared = self._execute_with_output_recovery(
                        state,
                        PreparedCall(
                            assembly,
                            compilation.report,
                            "lean",
                            contract,
                        ),
                        minimum_output_tokens=minimum_output_tokens,
                        validator=lambda value: _validated_caller_output(
                            value, semantic_schema
                        ),
                    )
                except _OutputRecoveryContextOverflow as exc:
                    compilation = exc.compilation
                    last_compilation = compilation
                    minimum_output_tokens = exc.minimum_output_tokens
                else:
                    source_references = [
                        reference
                        for row in evidence_rows
                        for reference in row.get("source_event_references", [])
                    ]
                    self.history.record_event(
                        state,
                        "caller_structured_output_created",
                        {
                            "schema": semantic_schema,
                            "semantic_output": payload,
                            "evidence_source_references": source_references,
                            "projected_source_event_sequences": sorted(projections),
                        },
                    )
                    return payload
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
        raise BudgetExceededError(
            "The caller-defined structured output prompt does not fit after bounded semantic reduction.",
            last_compilation.report if last_compilation is not None else None,
        )

    def generate_response_presentations(
        self,
        state: SessionState,
        *,
        original_request: str,
        assistant_message: str,
        modes: set[str] | list[str] | tuple[str, ...],
    ) -> dict[str, Any]:
        """Create separately verified visual/audio views without changing raw history."""
        requested = {str(mode).strip() for mode in modes if str(mode).strip()}
        unknown = requested - {"visual", "audio"}
        if unknown:
            raise ValueError(
                "response presentation modes must be visual and/or audio: "
                + ", ".join(sorted(unknown))
            )
        source = assistant_message
        result: dict[str, Any] = {
            "raw": source,
            "visual": None,
            "audio": None,
            "requested_modes": sorted(requested),
            "completed_modes": [],
        }
        if not requested:
            return result
        if not source.strip():
            raise ValueError("assistant_message must not be empty for presentation")

        run_id = f"{state.session_id}:{new_id('run')}"
        self.history.set_active_run(
            state.session_id,
            run_id=run_id,
            user_text=original_request,
        )
        try:
            self._heartbeat(
                state,
                run_id=run_id,
                phase="response_presentation",
                detail="selecting user-facing information",
            )
            visual = self._generate_validated_response_presentation(
                state,
                mode="response_relevance",
                original_request=original_request,
                source_answer=source,
            )
            if visual is not None:
                result["visual"] = visual
                if "visual" in requested:
                    result["completed_modes"].append("visual")
            else:
                result["visual"] = source

            if "audio" in requested and visual is not None:
                self._heartbeat(
                    state,
                    run_id=run_id,
                    phase="response_presentation",
                    detail="rendering verified information for audio",
                )
                audio = self._generate_validated_response_presentation(
                    state,
                    mode="audio_rendering",
                    original_request=original_request,
                    source_answer=str(result["visual"]),
                )
                if audio is not None:
                    result["audio"] = audio
                    result["completed_modes"].append("audio")
            elif "audio" in requested:
                source_references = self._presentation_source_event_references(
                    state,
                    source,
                )
                self.history.record_event(
                    state,
                    "response_presentation_unavailable",
                    {
                        "mode": "audio_rendering",
                        "source_answer": source,
                        "source_answer_sha256": sha256_text(source),
                        "source_event_references": source_references,
                        "error": (
                            "verified response-relevance selection was unavailable; "
                            "audio rendering was not allowed to bypass it"
                        ),
                        "error_type": "PresentationPrerequisiteUnavailable",
                    },
                )
            self._heartbeat(
                state,
                run_id=run_id,
                phase="completed",
                detail="response presentation completed",
            )
            return result
        except RunCancellationRequested as exc:
            self.preemption.complete_run_cancellation(state.session_id, run_id)
            self._heartbeat(
                state,
                run_id=run_id,
                phase="cancelled",
                detail=str(exc),
            )
            raise
        except Exception as exc:
            self._heartbeat(
                state,
                run_id=run_id,
                phase="failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            self.history.clear_active_run(state.session_id, run_id=run_id)

    def _generate_validated_response_presentation(
        self,
        state: SessionState,
        *,
        mode: str,
        original_request: str,
        source_answer: str,
    ) -> str | None:
        if mode not in {"response_relevance", "audio_rendering"}:
            raise ValueError(f"Unknown response presentation mode: {mode}")
        source_hash = sha256_text(source_answer)
        source_references = self._presentation_source_event_references(
            state,
            source_answer,
        )
        validation_feedback = ""
        max_attempts = max(1, int(self.config.model.max_retries) + 1)
        last_error = "independent presentation evaluation rejected every candidate"
        last_error_type = "PresentationRejected"
        for attempt in range(1, max_attempts + 1):
            try:
                if mode == "response_relevance":
                    assembly = self.prompts.build_response_relevance_prompt(
                        original_request=original_request,
                        source_answer=source_answer,
                        validation_feedback=validation_feedback,
                    )
                    contract = response_relevance_contract()
                    field_name = "answer"
                else:
                    assembly = self.prompts.build_audio_rendering_prompt(
                        original_request=original_request,
                        source_answer=source_answer,
                        validation_feedback=validation_feedback,
                    )
                    contract = audio_rendering_contract()
                    field_name = "audio_text"
                payload = self._execute_compiled_presentation_call(
                    state,
                    assembly,
                    contract,
                    validator=lambda value: _validated_presentation_text(
                        value,
                        field_name=field_name,
                    ),
                )
                candidate = str(payload[field_name]).strip()
                evaluation = self._evaluate_response_presentation(
                    state,
                    mode=mode,
                    original_request=original_request,
                    source_answer=source_answer,
                    candidate_answer=candidate,
                )
            except (RunCancellationRequested, ModelCallStateChanged):
                raise
            except Exception as exc:
                last_error = str(exc)
                last_error_type = type(exc).__name__
                validation_feedback = (
                    "The prior attempt failed mechanical validation or execution: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if bool(evaluation["acceptable"]):
                self.history.record_event(
                    state,
                    "response_presentation_generated",
                    {
                        "mode": mode,
                        "source_answer": source_answer,
                        "source_answer_sha256": source_hash,
                        "source_event_references": source_references,
                        "presentation_sha256": sha256_text(candidate),
                        "presentation": candidate,
                        "transformation": payload,
                        "evaluation": evaluation,
                        "attempt": attempt,
                    },
                )
                return candidate
            self.history.record_event(
                state,
                "response_presentation_rejected",
                {
                    "mode": mode,
                    "source_answer": source_answer,
                    "source_answer_sha256": source_hash,
                    "source_event_references": source_references,
                    "candidate_sha256": sha256_text(candidate),
                    "candidate": candidate,
                    "transformation": payload,
                    "evaluation": evaluation,
                    "attempt": attempt,
                },
            )
            last_error = str(evaluation.get("reason", last_error)).strip() or last_error
            last_error_type = "PresentationRejected"
            validation_feedback = stable_json_dumps(evaluation, indent=2)
        self.history.record_event(
            state,
            "response_presentation_unavailable",
            {
                "mode": mode,
                "source_answer": source_answer,
                "source_answer_sha256": source_hash,
                "source_event_references": source_references,
                "error": last_error,
                "error_type": last_error_type,
            },
        )
        return None

    def _presentation_source_event_references(
        self,
        state: SessionState,
        source_answer: str,
    ) -> list[dict[str, Any]]:
        for event in reversed(self.history.read_history(state.session_id)):
            candidate: Any = None
            if event.event_type == "turn_finished":
                candidate = event.payload.get("assistant_text")
            elif event.event_type == "response_presentation_generated":
                candidate = event.payload.get("presentation")
            elif event.event_type == "message_added":
                message = event.payload.get("message")
                if isinstance(message, dict) and message.get("role") == "assistant":
                    candidate = message.get("content")
            if candidate == source_answer:
                return [
                    {
                        **self._communication_evidence_reference(event),
                        "relationship": "presentation_source",
                    }
                ]
        return []

    def _evaluate_response_presentation(
        self,
        state: SessionState,
        *,
        mode: str,
        original_request: str,
        source_answer: str,
        candidate_answer: str,
    ) -> dict[str, Any]:
        assembly = self.prompts.build_presentation_evaluation_prompt(
            mode=mode,
            original_request=original_request,
            source_answer=source_answer,
            candidate_answer=candidate_answer,
        )
        return self._execute_compiled_presentation_call(
            state,
            assembly,
            presentation_evaluation_contract(),
            validator=_validated_presentation_evaluation,
        )

    def _execute_compiled_presentation_call(
        self,
        state: SessionState,
        assembly: PromptAssembly,
        contract: ContractSpec,
        *,
        validator: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        minimum_output_tokens = 128
        compilation = self._compile_context(
            state,
            assembly,
            contract,
            minimum_output_tokens=minimum_output_tokens,
        )
        cap_error = "" if compilation.report.fits else "context_limit_exceeded"
        self.history.record_event(
            state,
            "context_compiled",
            {
                "kind": assembly.kind,
                "prompt_mode": assembly.prompt_mode,
                "accounting": compilation.accounting(),
                "cap_error": cap_error,
            },
        )
        self.history.record_event(
            state,
            "budget_checked",
            {
                "kind": assembly.kind,
                "prompt_mode": assembly.prompt_mode,
                "budget_report": asdict(compilation.report),
                "cap_error": cap_error,
            },
        )
        if not compilation.report.fits:
            recovered = self._recover_prompt_instruction_overflow(
                state,
                assembly,
                contract,
                compilation,
                minimum_output_tokens=minimum_output_tokens,
            )
            if recovered is not None:
                compilation = recovered
                self.history.record_event(
                    state,
                    "context_compiled",
                    {
                        "kind": assembly.kind,
                        "prompt_mode": assembly.prompt_mode,
                        "accounting": recovered.accounting(),
                        "cap_error": "",
                        "prompt_instruction_projection": True,
                    },
                )
                self.history.record_event(
                    state,
                    "budget_checked",
                    {
                        "kind": assembly.kind,
                        "prompt_mode": assembly.prompt_mode,
                        "budget_report": asdict(recovered.report),
                        "cap_error": "",
                        "prompt_instruction_projection": True,
                    },
                )
        if not compilation.report.fits:
            raise BudgetExceededError(
                f"The {assembly.kind} prompt does not fit without semantic loss",
                compilation.report,
            )
        self._record_prompt_built(state, assembly, contract, compilation.report)
        try:
            payload, _prepared = self._execute_with_output_recovery(
                state,
                PreparedCall(
                    assembly,
                    compilation.report,
                    assembly.prompt_mode,
                    contract,
                ),
                minimum_output_tokens=minimum_output_tokens,
                validator=validator,
                allow_prompt_instruction_projection=True,
            )
        except _OutputRecoveryContextOverflow as exc:
            raise BudgetExceededError(
                f"The {assembly.kind} retry cannot preserve exact input and output headroom",
                exc.compilation.report,
            ) from exc
        return payload

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

    @staticmethod
    def _completion_evidence_source_key(source: dict[str, Any]) -> str:
        return f"{source.get('source_kind', '')}:{source.get('source_id', '')}"

    @classmethod
    def _completion_evidence_request_key(cls, request: dict[str, Any]) -> str:
        return cls._completion_evidence_source_key(request)

    @staticmethod
    def _string_leaves(value: Any) -> set[str]:
        leaves: set[str] = set()
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                leaves.add(item)
            elif isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, (list, tuple)):
                pending.extend(item)
        return leaves

    @staticmethod
    def _is_generated_id(value: str, prefix: str) -> bool:
        stem = f"{prefix}_"
        suffix = value[len(stem) :] if value.startswith(stem) else ""
        return len(suffix) == 12 and all(
            character in "0123456789abcdef" for character in suffix
        )

    def _completion_evidence_source_inventory(
        self,
        state: SessionState,
        *,
        evidence_rows: list[dict[str, Any]],
        historical_events: list[HistoryEvent],
    ) -> list[dict[str, Any]]:
        referenced_values = self._string_leaves(
            [
                evidence_rows,
                [
                    {
                        "payload": event.payload,
                        "metadata": event.metadata,
                    }
                    for event in historical_events
                ],
            ]
        )
        source_events = self.history.read_history(state.session_id)
        inventory: list[dict[str, Any]] = []
        artifact_ids = sorted(
            value
            for value in referenced_values
            if self._is_generated_id(value, "artifact")
        )
        for artifact_id in artifact_ids:
            references = [
                event
                for event in source_events
                if event.event_type == "artifact_created"
                and event.payload.get("artifact_id") == artifact_id
            ]
            if not references:
                continue
            source_event = references[-1]
            inventory.append(
                {
                    "source_kind": "text_artifact",
                    "source_id": artifact_id,
                    "content_kind": str(source_event.payload.get("kind", "")),
                    "size_chars": int(
                        source_event.payload.get("size_chars", 0)
                    ),
                    "sha256": str(source_event.payload.get("sha256", "")),
                    "source_event_references": [
                        self._communication_evidence_reference(event)
                        for event in references
                    ],
                }
            )

        referenced_attachment_ids = {
            value
            for value in referenced_values
            if self._is_generated_id(value, "attachment")
        }
        for reference in state.attachments:
            if reference.attachment_id not in referenced_attachment_ids:
                continue
            metadata = reference.metadata
            source_references = []
            sequence = metadata.get("source_event_sequence")
            source_hash = metadata.get("source_event_hash")
            if isinstance(sequence, int) and isinstance(source_hash, str):
                source_references.append(
                    {
                        "session_id": str(
                            metadata.get(
                                "source_event_session_id", state.session_id
                            )
                        ),
                        "sequence": sequence,
                        "hash": source_hash,
                        "event_type": str(
                            metadata.get("source_event_type", "attachment_added")
                        ),
                    }
                )
            inventory.append(
                {
                    "source_kind": "raw_attachment",
                    "source_id": reference.attachment_id,
                    "original_name": reference.original_name,
                    "media_type": reference.media_type,
                    "size_bytes": reference.size_bytes,
                    "sha256": reference.sha256,
                    "source_event_references": source_references,
                }
            )
        return sorted(
            inventory,
            key=lambda item: (str(item["source_kind"]), str(item["source_id"])),
        )

    def _reexpand_completion_evidence_source(
        self,
        state: SessionState,
        *,
        source: dict[str, Any],
        purpose: str,
    ) -> dict[str, Any]:
        source_kind = str(source["source_kind"])
        source_id = str(source["source_id"])
        row = dict(source)
        row["requested_purpose"] = purpose
        row["integrity_verified"] = True
        if source_kind == "text_artifact":
            artifact = TextArtifactStore(
                self.config.sessions.root, state.session_id
            ).get(source_id)
            if (
                artifact.sha256 != str(source.get("sha256", ""))
                or artifact.size_chars != int(source.get("size_chars", -1))
            ):
                raise HistoryInvariantError(
                    "Completion evidence artifact metadata differs from its source event"
                )
            row["text"] = Path(artifact.path).read_text(encoding="utf-8")
            return row
        if source_kind == "raw_attachment":
            reference = find_attachment(state.attachments, source_id)
            data = AttachmentStore(
                self.config.sessions.root,
                max_upload_bytes=self.config.attachments.max_upload_bytes,
            ).read_bytes(reference)
            try:
                row["text"] = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                row["integrity_verified"] = True
                row["read_error"] = (
                    "The exact raw bytes are not UTF-8 text; a selected specialist "
                    f"reader is required ({exc})."
                )
                row["text"] = ""
            return row
        raise ValueError(f"Unsupported completion evidence source: {source_kind}")

    def _project_completion_evidence_source_for_overflow(
        self,
        state: SessionState,
        *,
        original_request: str,
        source_key: str,
        source_row: dict[str, Any],
        current_tokens: int,
        overflow_tokens: int,
        existing_projection: str | None,
        context_limit_resolution: tuple[int, str],
    ) -> str | None:
        raw_text = str(source_row.get("text", ""))
        if not raw_text or current_tokens <= 0 or overflow_tokens <= 0:
            return None
        current_semantic_text = existing_projection or raw_text
        semantic_tokens = self._counter(state).count_text(
            current_semantic_text
        ).tokens
        target_tokens = max(64, semantic_tokens - overflow_tokens - 32)
        if target_tokens >= semantic_tokens:
            return None
        projection, projection_report = self._reduce_text_hierarchically(
            state,
            source_text=raw_text,
            source_label=(
                "integrity-checked completion evidence source " + source_key
            ),
            target_tokens=target_tokens,
            contract=evidence_projection_contract(),
            output_key="projection",
            build_assembly=lambda text, label, target: (
                self.prompts.build_evidence_projection_prompt(
                    purpose=(
                        "Evaluate whether this user objective is complete: "
                        + original_request
                    ),
                    source_label=label,
                    raw_evidence=text,
                    target_tokens=target,
                )
            ),
            remaining_calls=[
                max(16, int(self.config.context.max_compaction_rounds) * 16)
            ],
            context_limit_resolution=context_limit_resolution,
        )
        projected_tokens = self._counter(state).count_text(projection).tokens
        self.history.record_event(
            state,
            "completion_evidence_projected",
            {
                "source_kind": str(source_row.get("source_kind", "")),
                "source_id": str(source_row.get("source_id", "")),
                "source_sha256": str(source_row.get("sha256", "")),
                "source_event_references": source_row.get(
                    "source_event_references", []
                ),
                "target_tokens": target_tokens,
                "original_tokens": self._counter(state).count_text(raw_text).tokens,
                "previous_tokens": semantic_tokens,
                "projected_tokens": projected_tokens,
                "overflow_tokens": overflow_tokens,
                "projection": projection,
                "budget_report": asdict(projection_report),
            },
        )
        return projection

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
        # Recover only the measured deficit plus fixed framing slack. The next
        # complete prompt is rebuilt and counted, so no proportional semantic
        # reduction is needed as a substitute for exact admission.
        target_tokens = max(64, current_tokens - overflow - 32)
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

    def _reduce_text_hierarchically(
        self,
        state: SessionState,
        *,
        source_text: str,
        source_label: str,
        target_tokens: int,
        contract: ContractSpec,
        output_key: str,
        build_assembly: Callable[[str, str, int], PromptAssembly],
        remaining_calls: list[int],
        context_limit_resolution: tuple[int, str] | None = None,
        include_prompt_instructions: bool = True,
        depth: int = 0,
    ) -> tuple[str, BudgetReport]:
        minimum_output_tokens = min(
            target_tokens + 64,
            self.config.context.reserved_response_tokens,
        )
        assembly = build_assembly(source_text, source_label, target_tokens)
        compilation = self._compile_context(
            state,
            assembly,
            contract,
            minimum_output_tokens=minimum_output_tokens,
            context_limit_resolution=context_limit_resolution,
            desired_output_tokens=target_tokens + 64,
            include_prompt_instructions=include_prompt_instructions,
        )
        prompt_instruction_projected = False
        if (
            not compilation.report.fits
            and include_prompt_instructions
            and self._counter(state).count_text(source_text).tokens
            <= compilation.overflow_tokens + 32
        ):
            recovered = self._recover_prompt_instruction_overflow(
                state,
                assembly,
                contract,
                compilation,
                minimum_output_tokens=minimum_output_tokens,
                desired_output_tokens=target_tokens + 64,
                context_limit_resolution=context_limit_resolution,
            )
            if recovered is not None:
                compilation = recovered
                prompt_instruction_projected = True
        if compilation.report.fits:
            if remaining_calls[0] <= 0:
                raise BudgetExceededError(
                    f"{assembly.kind} exhausted its bounded semantic call budget",
                    compilation.report,
                )
            remaining_calls[0] -= 1
            self.telemetry.record_semantic_reduction(
                call_kind=assembly.kind,
                target_tokens=target_tokens,
                hierarchical_depth=depth,
            )
            self.history.record_event(
                state,
                "context_compiled",
                {
                    "kind": assembly.kind,
                    "prompt_mode": "lean",
                    "accounting": compilation.accounting(),
                    "hierarchical_depth": depth,
                    "prompt_instruction_projection": (
                        prompt_instruction_projected
                    ),
                },
            )
            self._record_prompt_built(
                state,
                assembly,
                contract,
                compilation.report,
            )

            def validate_reduction(payload: dict[str, Any]) -> dict[str, Any]:
                if contract.json_schema is not None:
                    _validate_schema_value(
                        payload,
                        contract.json_schema,
                        path=contract.name,
                    )
                return payload

            try:
                payload, final_prepared = self._execute_with_output_recovery(
                    state,
                    PreparedCall(
                        assembly,
                        compilation.report,
                        "lean",
                        contract,
                    ),
                    minimum_output_tokens=minimum_output_tokens,
                    desired_output_tokens=target_tokens + 64,
                    validator=validate_reduction,
                    context_limit_resolution=context_limit_resolution,
                    include_prompt_instructions=include_prompt_instructions,
                )
            except _OutputRecoveryContextOverflow:
                pass
            else:
                reduced = str(payload.get(output_key, "")).strip()
                if not reduced:
                    raise ValueError(f"{assembly.kind} output must not be empty")
                return reduced, final_prepared.report

        if depth >= 16 or len(source_text) < 2:
            raise BudgetExceededError(
                f"An exact source cannot be segmented enough to fit {assembly.kind}",
                compilation.report,
            )
        midpoint = len(source_text) // 2
        child_target = max(64, (target_tokens + 1) // 2)
        fragments = []
        for index, fragment_text in enumerate(
            (source_text[:midpoint], source_text[midpoint:]),
            start=1,
        ):
            projection, _report = self._reduce_text_hierarchically(
                state,
                source_text=fragment_text,
                source_label=f"{source_label} exact fragment {index}/2",
                target_tokens=child_target,
                contract=contract,
                output_key=output_key,
                build_assembly=build_assembly,
                remaining_calls=remaining_calls,
                context_limit_resolution=context_limit_resolution,
                include_prompt_instructions=include_prompt_instructions,
                depth=depth + 1,
            )
            fragments.append(projection)
        return self._reduce_text_hierarchically(
            state,
            source_text=(
                "[SEMANTIC PROJECTION OF EXACT FRAGMENT 1]\n"
                + fragments[0]
                + "\n\n[SEMANTIC PROJECTION OF EXACT FRAGMENT 2]\n"
                + fragments[1]
            ),
            source_label=f"{source_label} semantic fragment projections",
            target_tokens=target_tokens,
            contract=contract,
            output_key=output_key,
            build_assembly=build_assembly,
            remaining_calls=remaining_calls,
            context_limit_resolution=context_limit_resolution,
            include_prompt_instructions=include_prompt_instructions,
            depth=depth + 1,
        )

    def _project_tool_result_text_hierarchically(
        self,
        state: SessionState,
        *,
        original_request: str,
        tool_name: str,
        source_text: str,
        source_event_sequence: int,
        source_event_hash: str,
        target_tokens: int,
        remaining_calls: list[int],
    ) -> tuple[str, BudgetReport]:
        return self._reduce_text_hierarchically(
            state,
            source_text=source_text,
            source_label=tool_name,
            target_tokens=target_tokens,
            contract=tool_result_projection_contract(),
            output_key="projection",
            build_assembly=lambda text, label, target: (
                self.prompts.build_tool_result_projection_prompt(
                    original_request=original_request,
                    tool_name=label,
                    raw_tool_result=text,
                    source_event_sequence=source_event_sequence,
                    source_event_hash=source_event_hash,
                    target_tokens=target,
                )
            ),
            remaining_calls=remaining_calls,
        )

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
        try:
            projection, final_report = self._project_tool_result_text_hierarchically(
                state,
                original_request=original_request,
                tool_name=message.name or "tool",
                source_text=message.content,
                source_event_sequence=sequence,
                source_event_hash=source_hash,
                target_tokens=target_tokens,
                remaining_calls=[
                    max(16, int(self.config.context.max_compaction_rounds) * 16)
                ],
            )
        except (BudgetExceededError, OutputBudgetExhaustedError) as exc:
            self.history.record_event(
                state,
                "tool_result_projection_skipped",
                {
                    "source_event_sequence": sequence,
                    "source_event_hash": source_hash,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "target_tokens": target_tokens,
                    "original_tokens": original_tokens,
                    "overflow_tokens": overflow_tokens,
                    "budget_report": (
                        asdict(exc.report)
                        if isinstance(exc, BudgetExceededError)
                        and exc.report is not None
                        else None
                    ),
                },
            )
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
                "projection_budget_report": asdict(final_report),
                "projection": projection,
            },
        )
        return projection

    def _runtime_context_sources(self, state: SessionState) -> dict[str, str]:
        filesystem = AgentEnvironment(self.config, state).filesystem
        workspace_files = filesystem.list_files(".")
        return {
            "workspace_file_manifest": stable_json_dumps(
                {
                    "workspace_root": state.environment.workspace.root,
                    "files": workspace_files,
                    "count": len(workspace_files),
                },
                indent=2,
            ),
            "durable_notes": render_notes(state.notes),
        }

    def _runtime_context_source_locator(
        self,
        state: SessionState,
        source_name: str,
    ) -> dict[str, object]:
        if source_name == "workspace_file_manifest":
            return {
                "authoritative_source": "live_filesystem",
                "workspace_root": state.environment.workspace.root,
                "recovery_tool": "list_files",
                "recovery_arguments": {
                    "path": state.environment.workspace.root,
                },
            }
        if source_name == "durable_notes":
            return {
                "authoritative_source": "durable_note_events",
                "session_id": state.session_id,
                "recovery_tool": "notes",
                "recovery_arguments": {"action": "list"},
            }
        raise ValueError(f"Unknown runtime context source: {source_name}")

    def _project_runtime_context_for_overflow(
        self,
        state: SessionState,
        *,
        original_request: str,
        compilation: ContextCompilation,
        existing_projections: dict[str, RuntimeContextProjection],
        remaining_calls: list[int],
    ) -> tuple[str, RuntimeContextProjection] | None:
        if compilation.overflow_tokens <= 0:
            return None
        sources = self._runtime_context_sources(state)
        report_by_name = {
            item.name: int(item.tokens) for item in compilation.report.breakdown
        }
        candidates = [
            (report_by_name.get(name, 0), name, text)
            for name, text in sources.items()
            if text and report_by_name.get(name, 0) > 0
        ]
        if not candidates:
            return None
        _current_tokens, source_name, source_text = max(candidates)
        source_hash = sha256_text(source_text)
        objective_hash = sha256_text(original_request)
        stored_projection = existing_projections.get(source_name)
        previous_projection = (
            stored_projection.text
            if stored_projection is not None
            and stored_projection.source_sha256 == source_hash
            else None
        )
        counter = self._counter(state)
        source_tokens = counter.count_text(source_text).tokens
        previous_tokens = (
            counter.count_text(previous_projection).tokens
            if previous_projection is not None
            else source_tokens
        )
        target_tokens = max(
            64,
            previous_tokens - max(16, compilation.overflow_tokens) - 32,
        )
        if target_tokens >= previous_tokens:
            return None
        stored = self.history.latest_runtime_context_projection(
            state.session_id,
            source_name=source_name,
            source_sha256=source_hash,
            objective_sha256=objective_hash,
            max_projected_tokens=target_tokens,
        )
        if stored is not None:
            projection = str(stored.payload["projection"]).strip()
            projected_tokens = int(stored.payload["projected_tokens"])
            self.history.record_event(
                state,
                "runtime_context_projection_reused",
                {
                    "source_name": source_name,
                    "source_sha256": source_hash,
                    "objective_sha256": objective_hash,
                    "source_locator": self._runtime_context_source_locator(
                        state, source_name
                    ),
                    "projection_event_sequence": stored.sequence,
                    "target_tokens": target_tokens,
                    "projected_tokens": projected_tokens,
                },
            )
            return source_name, RuntimeContextProjection(source_hash, projection)
        try:
            projection, projection_report = self._reduce_text_hierarchically(
                state,
                source_text=source_text,
                source_label=(
                    "complete current workspace file manifest"
                    if source_name == "workspace_file_manifest"
                    else "all exact durable model-authored notes"
                ),
                target_tokens=target_tokens,
                contract=evidence_projection_contract(),
                output_key="projection",
                build_assembly=lambda text, label, target: (
                    self.prompts.build_evidence_projection_prompt(
                        purpose=(
                            "Preserve the parts of this recoverable runtime context that matter "
                            "for the current user objective: "
                            + original_request
                        ),
                        source_label=label,
                        raw_evidence=text,
                        target_tokens=target,
                    )
                ),
                remaining_calls=remaining_calls,
                context_limit_resolution=self._resolve_context_limit(),
            )
        except (BudgetExceededError, OutputBudgetExhaustedError) as exc:
            self.history.record_event(
                state,
                "runtime_context_projection_skipped",
                {
                    "source_name": source_name,
                    "source_sha256": source_hash,
                    "objective_sha256": objective_hash,
                    "source_locator": self._runtime_context_source_locator(
                        state, source_name
                    ),
                    "previous_tokens": previous_tokens,
                    "target_tokens": target_tokens,
                    "overflow_tokens": compilation.overflow_tokens,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "budget_report": (
                        asdict(exc.report)
                        if isinstance(exc, BudgetExceededError)
                        and exc.report is not None
                        else None
                    ),
                },
            )
            return None
        projected_tokens = counter.count_text(projection).tokens
        if projected_tokens >= previous_tokens:
            self.history.record_event(
                state,
                "runtime_context_projection_skipped",
                {
                    "source_name": source_name,
                    "source_sha256": source_hash,
                    "objective_sha256": objective_hash,
                    "source_locator": self._runtime_context_source_locator(
                        state, source_name
                    ),
                    "previous_tokens": previous_tokens,
                    "target_tokens": target_tokens,
                    "overflow_tokens": compilation.overflow_tokens,
                    "reason": "semantic projection did not reduce the measured source",
                    "budget_report": asdict(projection_report),
                },
            )
            return None
        self.history.record_event(
            state,
            "runtime_context_projected",
            {
                "source_name": source_name,
                "source_sha256": source_hash,
                "objective_sha256": objective_hash,
                "source_locator": self._runtime_context_source_locator(
                    state, source_name
                ),
                "source_tokens": source_tokens,
                "previous_tokens": previous_tokens,
                "target_tokens": target_tokens,
                "projected_tokens": projected_tokens,
                "overflow_tokens": compilation.overflow_tokens,
                "projection_budget_report": asdict(projection_report),
                "projection": projection,
            },
        )
        return source_name, RuntimeContextProjection(source_hash, projection)

    def _runtime_context_components(
        self,
        state: SessionState,
        counter: ExactTokenCounter | ConservativeEstimator | _HistoryAwareTokenCounter,
        *,
        projections: dict[str, RuntimeContextProjection] | None = None,
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
        sources = self._runtime_context_sources(state)
        projection_map = projections or {}
        workspace_source = sources["workspace_file_manifest"]
        workspace_projection = projection_map.get("workspace_file_manifest")
        if (
            workspace_projection is not None
            and workspace_projection.source_sha256
            == sha256_text(workspace_source)
        ):
            workspace_text = (
                "[SEMANTIC PROJECTION; the live filesystem remains authoritative]\n"
                + workspace_projection.text
            )
        else:
            workspace_text = workspace_source
        components.append(
            PromptComponent(
                name="workspace_file_manifest",
                category="environment",
                text=(
                    "Workspace file manifest. Use list_files on workspace_root to recover the exact current listing when needed:\n"
                    + workspace_text
                    + "\n\n"
                ),
                optional=True,
            )
        )
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
        notes_source = sources["durable_notes"]
        notes_count = counter.count_text(notes_source)
        self.history.record_event(
            state,
            "notes_selected",
            {
                "included_note_ids": [note.note_id for note in state.notes],
                "omitted_note_ids": [],
                "tokens": notes_count.tokens,
                "exact": notes_count.exact,
            },
        )
        if notes_source:
            notes_projection = projection_map.get("durable_notes")
            if (
                notes_projection is not None
                and notes_projection.source_sha256 == sha256_text(notes_source)
            ):
                notes_text = (
                    "[SEMANTIC PROJECTION; exact notes remain authoritative and retrievable]\n"
                    + notes_projection.text
                )
            else:
                notes_text = notes_source
            components.append(
                PromptComponent(
                    name="durable_notes",
                    category="notes",
                    text=(
                        "Durable model-authored notes. These are navigation aids; verbatim user messages and tool results remain authoritative:\n"
                        + notes_text
                        + "\n\n"
                    ),
                    optional=True,
                )
            )
        return components

    @staticmethod
    def _assembly_chat_messages(
        assembly: PromptAssembly,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        previous_end = 0
        for message_range in assembly.message_ranges:
            start = int(message_range.component_start)
            end = int(message_range.component_end)
            if start < previous_end or end <= start or end > len(assembly.components):
                raise ModelClientError("Prompt assembly has invalid chat-message ranges")
            content = "".join(
                component.text for component in assembly.components[start:end]
            )
            if not content:
                raise ModelClientError("Prompt assembly contains an empty chat message")
            messages.append({"role": message_range.role, "content": content})
            previous_end = end
        return messages

    def _materialize_prompt_protocol(self, assembly: PromptAssembly) -> None:
        protocol_source = "prompt_protocol:server_chat_template"
        if any(
            artifact.source == protocol_source
            for artifact in assembly.prompt_artifacts
        ):
            return
        renderer = getattr(self.client, "render_chat_prompt", None)
        if not callable(renderer):
            return
        if not assembly.message_ranges:
            raise ModelClientError(
                "Prompt assembly does not expose chat-message component ranges"
            )
        messages = self._assembly_chat_messages(assembly)
        rendering = renderer(messages)
        if not isinstance(rendering, dict):
            raise ModelClientError("Model returned an invalid chat-template rendering")
        rendered = rendering.get("prompt")
        protocol_hash = rendering.get("prompt_protocol_sha256")
        if (
            not isinstance(rendered, str)
            or not rendered
            or not isinstance(protocol_hash, str)
            or len(protocol_hash) != 64
        ):
            raise ModelClientError("Model chat-template rendering lacks prompt identity")
        source_components = assembly.components
        semantic_indices: set[int] = set()
        for message_range in assembly.message_ranges:
            semantic_indices.update(
                range(message_range.component_start, message_range.component_end)
            )
        if any(
            index not in semantic_indices and component.category != "wrapper"
            for index, component in enumerate(source_components)
        ):
            raise ModelClientError(
                "Prompt assembly has unassigned semantic components outside chat messages"
            )

        materialized: list[PromptComponent] = []
        materialized_ranges: list[PromptMessageRange] = []
        cursor = 0
        for index, (message, message_range) in enumerate(
            zip(messages, assembly.message_ranges, strict=True),
            start=1,
        ):
            content = message["content"]
            offset = rendered.find(content, cursor)
            if offset < 0:
                raise ModelClientError(
                    "Model chat template transformed message content; exact component accounting is unavailable"
                )
            materialized.append(
                PromptComponent(
                    name=f"chat_template_wrapper_{index}",
                    category="wrapper",
                    text=rendered[cursor:offset],
                )
            )
            start = len(materialized)
            materialized.extend(
                source_components[
                    message_range.component_start : message_range.component_end
                ]
            )
            materialized_ranges.append(
                PromptMessageRange(
                    role=message_range.role,
                    component_start=start,
                    component_end=len(materialized),
                )
            )
            cursor = offset + len(content)
        materialized.append(
            PromptComponent(
                name="chat_template_generation_suffix",
                category="wrapper",
                text=rendered[cursor:],
            )
        )
        if "".join(component.text for component in materialized) != rendered:
            raise ModelClientError(
                "Materialized chat-template components do not reproduce the exact prompt"
            )
        assembly.components = materialized
        assembly.message_ranges = materialized_ranges
        assembly.prompt_text = rendered
        assembly.prompt_artifacts = [
            artifact
            for artifact in assembly.prompt_artifacts
            if not artifact.source.startswith("prompt_protocol:")
        ] + [PromptArtifact(source=protocol_source, sha256=protocol_hash)]

    def _require_system_prompt(self, assembly: PromptAssembly) -> None:
        messages = self._assembly_chat_messages(assembly)
        if not messages or messages[0]["role"] != "system":
            raise ModelClientError(
                "Every model call requires a leading system message"
            )
        if not messages[0]["content"].strip():
            raise ModelClientError("Model call system message must not be blank")

    def _compile_context(
        self,
        state: SessionState | None,
        assembly: PromptAssembly,
        contract: ContractSpec,
        *,
        minimum_output_tokens: int,
        desired_output_tokens: int | None = None,
        context_limit_resolution: tuple[int, str] | None = None,
        include_prompt_instructions: bool = True,
    ) -> ContextCompilation:
        if include_prompt_instructions:
            self._inject_prompt_instructions(state, assembly)
        self._require_system_prompt(assembly)
        self._materialize_prompt_protocol(assembly)
        context_limit, context_limit_source = (
            self._resolve_context_limit()
            if context_limit_resolution is None
            else context_limit_resolution
        )
        compilation = self.context_compiler.compile(
            assembly,
            contract,
            self._counter(state),
            minimum_output_tokens=minimum_output_tokens,
            desired_output_tokens=desired_output_tokens,
            context_limit=context_limit,
            context_limit_source=context_limit_source,
        )
        self.telemetry.record_context_compilation(
            call_kind=assembly.kind,
            context_limit_source=context_limit_source,
            report=compilation.report,
        )
        return compilation

    def _inject_prompt_instructions(
        self,
        state: SessionState | None,
        assembly: PromptAssembly,
    ) -> None:
        if state is None or any(
            component.name
            in {
                "durable_prompt_instructions",
                "durable_prompt_instruction_projection",
            }
            for component in assembly.components
        ):
            return
        selected_sources = [
            ("user", item)
            for item in prompt_instructions_for_kind(
                self.prompt_instruction_store.list(),
                assembly.kind,
            )
        ] + [
            ("session", item)
            for item in prompt_instructions_for_kind(
                state.prompt_instructions,
                assembly.kind,
            )
        ]
        if not selected_sources:
            return
        rendered_rows = [
            {"instruction_store": instruction_store, **asdict(item)}
            for instruction_store, item in selected_sources
        ]
        rendered = stable_json_dumps(rendered_rows, indent=2)
        component = PromptComponent(
            name="durable_prompt_instructions",
            category="system_prompt_instruction",
            text=(
                "\n\n[DURABLE MODEL-AUTHORED INSTRUCTIONS FOR THIS CALL KIND]\n"
                "Apply every instruction below. Their scopes were chosen semantically by "
                "the agent; deterministic runtime code only matches the current call kind.\n"
                + rendered
            ),
        )
        insert_at = next(
            (
                index
                for index, existing in enumerate(assembly.components)
                if existing.name == "fallback_message_separator"
            ),
            None,
        )
        if insert_at is None:
            raise ModelClientError(
                "Prompt assembly is missing the system/user message separator"
            )
        assembly.components.insert(insert_at, component)
        ranges: list[PromptMessageRange] = []
        for message_range in assembly.message_ranges:
            start = message_range.component_start
            end = message_range.component_end
            if message_range.role == "system" and end == insert_at:
                end += 1
            else:
                if start >= insert_at:
                    start += 1
                if end >= insert_at:
                    end += 1
            ranges.append(
                PromptMessageRange(
                    role=message_range.role,
                    component_start=start,
                    component_end=end,
                )
            )
        assembly.message_ranges = ranges
        assembly.prompt_text = "".join(item.text for item in assembly.components)
        instruction_hashes = [
            {
                "instruction_id": item.instruction_id,
                "instruction_store": instruction_store,
                "sha256": sha256_text(
                    stable_json_dumps(
                        {
                            "instruction_store": instruction_store,
                            **asdict(item),
                        },
                        indent=None,
                    )
                ),
            }
            for instruction_store, item in selected_sources
        ]
        combined_hash = sha256_text(rendered)
        assembly.prompt_artifacts.append(
            PromptArtifact(
                source=f"durable_prompt_instructions:{assembly.kind}",
                sha256=combined_hash,
            )
        )
        self.history.record_event(
            state,
            "prompt_instructions_selected",
            {
                "kind": assembly.kind,
                "instruction_ids": [
                    item.instruction_id for _, item in selected_sources
                ],
                "instruction_sources": [
                    {
                        "instruction_store": instruction_store,
                        "instruction_id": item.instruction_id,
                    }
                    for instruction_store, item in selected_sources
                ],
                "instruction_hashes": instruction_hashes,
                "exact": True,
            },
        )

    def _selected_prompt_instruction_rows(
        self,
        state: SessionState,
        kind: ModelCallKind,
    ) -> list[dict[str, Any]]:
        return [
            {"instruction_store": instruction_store, **asdict(item)}
            for instruction_store, item in (
                [
                    ("user", instruction)
                    for instruction in prompt_instructions_for_kind(
                        self.prompt_instruction_store.list(),
                        kind,
                    )
                ]
                + [
                    ("session", instruction)
                    for instruction in prompt_instructions_for_kind(
                        state.prompt_instructions,
                        kind,
                    )
                ]
            )
        ]

    def _recover_prompt_instruction_overflow(
        self,
        state: SessionState,
        assembly: PromptAssembly,
        contract: ContractSpec,
        failed: ContextCompilation,
        *,
        minimum_output_tokens: int,
        desired_output_tokens: int | None = None,
        context_limit_resolution: tuple[int, str] | None = None,
    ) -> ContextCompilation | None:
        if (
            failed.report.fits
            or not self.config.context.compact_on_overflow
            or assembly.kind == "prompt_instruction_projection"
        ):
            return None
        source_component = next(
            (
                component
                for component in assembly.components
                if component.name == "durable_prompt_instructions"
            ),
            None,
        )
        source_report = next(
            (
                component
                for component in failed.report.breakdown
                if component.name == "durable_prompt_instructions"
            ),
            None,
        )
        if source_component is None or source_report is None:
            return None
        source_tokens = int(source_report.tokens)
        overflow_tokens = max(1, int(failed.overflow_tokens))
        if source_tokens <= overflow_tokens + 32:
            return None

        source_rows = self._selected_prompt_instruction_rows(
            state,
            assembly.kind,
        )
        if not source_rows:
            return None
        exact_source = stable_json_dumps(source_rows, indent=2)
        source_sha256 = sha256_text(exact_source)
        references = [
            {
                "instruction_store": str(row["instruction_store"]),
                "instruction_id": str(row["instruction_id"]),
                "sha256": sha256_text(stable_json_dumps(row, indent=None)),
            }
            for row in source_rows
        ]
        projection_header = (
            "\n\n[DURABLE MODEL-AUTHORED INSTRUCTION PROJECTION FOR THIS CALL KIND]\n"
            "Measured context overflow required this model-authored derived view. "
            "Apply every operative rule below. Exact source instructions remain "
            "authoritative and recoverable through the prompt_instructions capability.\n"
        )
        counter = self._counter(state)
        header_tokens = counter.count_text(projection_header).tokens
        target_tokens = max(
            32,
            source_tokens - overflow_tokens - header_tokens - 16,
        )
        if target_tokens >= source_tokens:
            return None
        remaining_calls = [max(8, int(self.config.context.max_compaction_rounds) * 8)]
        maximum_rounds = max(1, int(self.config.context.max_compaction_rounds) + 1)
        for round_index in range(maximum_rounds):
            projection, projection_report = self._reduce_text_hierarchically(
                state,
                source_text=exact_source,
                source_label=(
                    f"exact durable instructions for {assembly.kind} calls"
                ),
                target_tokens=target_tokens,
                contract=prompt_instruction_projection_contract(),
                output_key="projection",
                build_assembly=lambda text, _label, target: (
                    self.prompts.build_prompt_instruction_projection_prompt(
                        call_kind=assembly.kind,
                        source_instructions=text,
                        source_sha256=sha256_text(text),
                        source_tokens=counter.count_text(text).tokens,
                        overflow_tokens=overflow_tokens,
                        target_tokens=target,
                    )
                ),
                remaining_calls=remaining_calls,
                context_limit_resolution=context_limit_resolution,
                include_prompt_instructions=False,
            )
            projected_tokens = counter.count_text(projection).tokens
            candidate = copy.deepcopy(assembly)
            replacement_index = next(
                index
                for index, component in enumerate(candidate.components)
                if component.name == "durable_prompt_instructions"
            )
            candidate.components[replacement_index] = PromptComponent(
                name="durable_prompt_instruction_projection",
                category="system_prompt_instruction",
                text=projection_header + projection,
            )
            candidate.prompt_text = "".join(
                component.text for component in candidate.components
            )
            candidate.prompt_artifacts = [
                artifact
                for artifact in candidate.prompt_artifacts
                if artifact.source != "prompt_protocol:server_chat_template"
                and not artifact.source.startswith("durable_prompt_instructions:")
                and not artifact.source.startswith(
                    "durable_prompt_instruction_projection:"
                )
            ] + [
                PromptArtifact(
                    source=(
                        f"durable_prompt_instruction_projection:{assembly.kind}:"
                        f"{source_sha256}"
                    ),
                    sha256=sha256_text(projection),
                )
            ]
            recovered = self._compile_context(
                state,
                candidate,
                contract,
                minimum_output_tokens=minimum_output_tokens,
                desired_output_tokens=desired_output_tokens,
                context_limit_resolution=context_limit_resolution,
                include_prompt_instructions=False,
            )
            if recovered.report.fits:
                assembly.components = candidate.components
                assembly.message_ranges = candidate.message_ranges
                assembly.prompt_text = candidate.prompt_text
                assembly.prompt_artifacts = candidate.prompt_artifacts
                self.history.record_event(
                    state,
                    "prompt_instruction_projection_created",
                    {
                        "kind": assembly.kind,
                        "source_instruction_references": references,
                        "source_sha256": source_sha256,
                        "source_tokens": source_tokens,
                        "overflow_tokens": overflow_tokens,
                        "target_tokens": target_tokens,
                        "projected_tokens": projected_tokens,
                        "projection": projection,
                        "projection_sha256": sha256_text(projection),
                        "projection_budget_report": asdict(projection_report),
                        "reduction_round": round_index,
                        "exact_source_recovery": {
                            "session_id": state.session_id,
                            "capability": "prompt_instructions",
                            "instruction_references": references,
                        },
                    },
                )
                return recovered
            reduction = max(
                16,
                int(recovered.overflow_tokens) + 16,
                projected_tokens - target_tokens,
            )
            next_target = max(32, target_tokens - reduction)
            if next_target >= target_tokens:
                break
            target_tokens = next_target
        return None

    def _execute_tool_semantic_call(
        self, state: SessionState, request: SemanticCallRequest
    ) -> dict[str, Any]:
        assembly = self.prompts.build_semantic_operation_prompt(
            kind=request.kind,
            system_instruction=request.system_instruction,
            components=request.components,
            prompt_mode=request.prompt_mode,
        )
        minimum_output_tokens = max(1, int(request.minimum_output_tokens))
        output_retry = 0

        def validate_semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
            if request.contract.json_schema is not None:
                _validate_schema_value(
                    payload,
                    request.contract.json_schema,
                    path=request.contract.name,
                )
            return payload

        while True:
            compilation = self._compile_context(
                state,
                assembly,
                request.contract,
                minimum_output_tokens=minimum_output_tokens,
                desired_output_tokens=request.desired_output_tokens,
            )
            cap_error = "" if compilation.report.fits else "context_limit_exceeded"
            self.history.record_event(
                state,
                "context_compiled",
                {
                    "kind": request.kind,
                    "prompt_mode": request.prompt_mode,
                    "accounting": compilation.accounting(),
                    "cap_error": cap_error,
                    "output_retry": output_retry,
                },
            )
            self.history.record_event(
                state,
                "budget_checked",
                {
                    "kind": request.kind,
                    "prompt_mode": request.prompt_mode,
                    "budget_report": asdict(compilation.report),
                    "cap_error": cap_error,
                },
            )
            if (
                not compilation.report.fits
                and request.allow_prompt_instruction_projection
            ):
                recovered = self._recover_prompt_instruction_overflow(
                    state,
                    assembly,
                    request.contract,
                    compilation,
                    minimum_output_tokens=minimum_output_tokens,
                    desired_output_tokens=request.desired_output_tokens,
                )
                if recovered is not None:
                    compilation = recovered
                    self.history.record_event(
                        state,
                        "context_compiled",
                        {
                            "kind": request.kind,
                            "prompt_mode": request.prompt_mode,
                            "accounting": recovered.accounting(),
                            "cap_error": "",
                            "output_retry": output_retry,
                            "prompt_instruction_projection": True,
                        },
                    )
                    self.history.record_event(
                        state,
                        "budget_checked",
                        {
                            "kind": request.kind,
                            "prompt_mode": request.prompt_mode,
                            "budget_report": asdict(recovered.report),
                            "cap_error": "",
                            "prompt_instruction_projection": True,
                        },
                    )
            if not compilation.report.fits:
                raise SemanticCallContextOverflow(compilation.report)
            self._record_prompt_built(
                state, assembly, request.contract, compilation.report
            )
            prepared = PreparedCall(
                assembly,
                compilation.report,
                request.prompt_mode,
                request.contract,
            )
            try:
                return self._execute_structured_call(
                    state,
                    prepared,
                    validator=validate_semantic_payload,
                    seed_offset=output_retry,
                )
            except OutputBudgetExhaustedError:
                if output_retry >= int(self.config.model.max_retries):
                    raise
                expanded = self._expanded_output_minimum(prepared)
                if expanded <= minimum_output_tokens:
                    raise
                self.history.record_event(
                    state,
                    "budget_repaired",
                    {
                        "kind": request.kind,
                        "reason": "model_output_budget_exhausted",
                        "requested_response_tokens": minimum_output_tokens,
                        "capped_response_tokens": expanded,
                    },
                )
                minimum_output_tokens = expanded
                output_retry += 1

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
                "prompt_sha256": sha256_text(assembly.prompt_text),
                "prompt_artifacts": [
                    asdict(artifact) for artifact in assembly.prompt_artifacts
                ],
                "components": [asdict(component) for component in assembly.components],
                "message_ranges": [
                    asdict(message_range) for message_range in assembly.message_ranges
                ],
                "budget_report": asdict(report),
            },
        )

    def _summarize_oversized_message(
        self,
        state: SessionState,
        message: Message,
        *,
        target_summary_tokens: int,
        context_limit_resolution: tuple[int, str],
        remaining_calls: list[int],
        depth: int = 0,
    ) -> tuple[str, BudgetReport]:
        contract = summary_contract()
        minimum_output_tokens = min(
            int(self.config.context.reserved_summary_tokens),
            max(64, int(target_summary_tokens) + 64),
        )
        assembly = self.prompts.build_summary_prompt(
            [message],
            prompt_mode="lean",
            maximum_preserve_recent_messages=0,
            target_summary_tokens=target_summary_tokens,
        )
        compilation = self._compile_context(
            state,
            assembly,
            contract,
            minimum_output_tokens=minimum_output_tokens,
            desired_output_tokens=target_summary_tokens + 64,
            context_limit_resolution=context_limit_resolution,
        )
        prompt_instruction_projected = False
        if (
            not compilation.report.fits
            and self._counter(state).count_text(message.content).tokens
            <= compilation.overflow_tokens + 32
        ):
            recovered = self._recover_prompt_instruction_overflow(
                state,
                assembly,
                contract,
                compilation,
                minimum_output_tokens=minimum_output_tokens,
                desired_output_tokens=target_summary_tokens + 64,
                context_limit_resolution=context_limit_resolution,
            )
            if recovered is not None:
                compilation = recovered
                prompt_instruction_projected = True
        if compilation.report.fits:
            if remaining_calls[0] <= 0:
                raise BudgetExceededError(
                    "Hierarchical summary exhausted its bounded semantic call budget",
                    compilation.report,
                )
            remaining_calls[0] -= 1
            self.telemetry.record_semantic_reduction(
                call_kind="summary",
                target_tokens=target_summary_tokens,
                hierarchical_depth=depth,
            )
            self.history.record_event(
                state,
                "context_compiled",
                {
                    "kind": "summary",
                    "prompt_mode": "lean",
                    "accounting": compilation.accounting(),
                    "hierarchical_depth": depth,
                    "prompt_instruction_projection": (
                        prompt_instruction_projected
                    ),
                },
            )
            self._record_prompt_built(
                state,
                assembly,
                contract,
                compilation.report,
            )
            try:
                payload, final_prepared = self._execute_with_output_recovery(
                    state,
                    PreparedCall(
                        assembly,
                        compilation.report,
                        "lean",
                        contract,
                    ),
                    minimum_output_tokens=minimum_output_tokens,
                    desired_output_tokens=target_summary_tokens + 64,
                    context_limit_resolution=context_limit_resolution,
                )
            except _OutputRecoveryContextOverflow:
                pass
            else:
                summary_text = str(payload.get("summary", "")).strip()
                if not summary_text:
                    raise ValueError("hierarchical summary must not be empty")
                return summary_text, final_prepared.report

        if depth >= 16 or len(message.content) < 2:
            raise BudgetExceededError(
                "An exact history source cannot be segmented enough to fit the summary operation",
                compilation.report,
            )
        midpoint = len(message.content) // 2
        child_target = max(64, (int(target_summary_tokens) + 1) // 2)
        fragments = []
        for index, content in enumerate(
            (message.content[:midpoint], message.content[midpoint:]),
            start=1,
        ):
            fragment = Message(
                role=message.role,
                content=content,
                created_at=message.created_at,
                name=message.name,
                metadata={
                    **message.metadata,
                    "hierarchical_fragment": f"{index}/2",
                    "hierarchical_fragment_depth": depth + 1,
                },
            )
            summary, _report = self._summarize_oversized_message(
                state,
                fragment,
                target_summary_tokens=child_target,
                context_limit_resolution=context_limit_resolution,
                remaining_calls=remaining_calls,
                depth=depth + 1,
            )
            fragments.append(summary)
        combined = Message(
            role="summary",
            content=(
                "[SEMANTIC SUMMARY OF EXACT FRAGMENT 1]\n"
                + fragments[0]
                + "\n\n[SEMANTIC SUMMARY OF EXACT FRAGMENT 2]\n"
                + fragments[1]
            ),
            created_at=utc_now_iso(),
            metadata={
                "projection_kind": "hierarchical_history_summary",
                "source_event_references": message_source_event_references([message]),
            },
        )
        return self._summarize_oversized_message(
            state,
            combined,
            target_summary_tokens=target_summary_tokens,
            context_limit_resolution=context_limit_resolution,
            remaining_calls=remaining_calls,
            depth=depth + 1,
        )

    def _history_compaction_target(
        self,
        state: SessionState,
        source_messages: list[Message],
        *,
        required_recovery_tokens: int,
    ) -> tuple[int, int] | None:
        counter = self._counter(state)
        source_tokens = counter.count_text(
            self.prompts.render_messages(source_messages)
        ).tokens
        references = message_source_event_references(source_messages)
        empty_summary = Message(
            **summary_message_payload(
                "",
                source_message_count=len(source_messages),
                created_at=utc_now_iso(),
                source_event_references=references,
            )
        )
        replacement_overhead = counter.count_text(
            self.prompts.render_messages([empty_summary])
        ).tokens
        target = (
            int(source_tokens)
            - int(replacement_overhead)
            - max(1, int(required_recovery_tokens))
        )
        if target < 1:
            return None
        return target, int(source_tokens)

    def _compaction_recovery(
        self,
        state: SessionState,
        source_messages: list[Message],
        summary_payload: dict[str, Any],
    ) -> tuple[int, int, int]:
        counter = self._counter(state)
        source_tokens = counter.count_text(
            self.prompts.render_messages(source_messages)
        ).tokens
        replacement_tokens = counter.count_text(
            self.prompts.render_messages([Message(**summary_payload)])
        ).tokens
        return (
            int(source_tokens) - int(replacement_tokens),
            int(source_tokens),
            int(replacement_tokens),
        )

    def _compact_once(
        self,
        state: SessionState,
        *,
        required_recovery_tokens: int = 1,
    ) -> bool:
        if len(state.messages) <= 2:
            return False
        # Keep one message outside the candidate and require at least one exact
        # source to be replaced. Semantic retention within the candidate belongs
        # to the summary model, not a configured age cutoff.
        maximum_source = len(state.messages) - 1
        if maximum_source <= 0:
            return False

        contract = summary_contract()
        context_limit_resolution = self._resolve_context_limit()
        minimum_summary_tokens = int(self.config.context.reserved_summary_tokens)
        required_recovery = max(1, int(required_recovery_tokens))
        hierarchical_target: tuple[int, int] | None = None
        for source_count in range(1, maximum_source + 1):
            source_messages = state.messages[:source_count]
            target = self._history_compaction_target(
                state,
                source_messages,
                required_recovery_tokens=required_recovery,
            )
            if target is None:
                continue
            target_summary_tokens, estimated_source_tokens = target
            if source_count == 1:
                hierarchical_target = target
            adaptive_cap = max(0, source_count - 1)
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
                minimum_output_tokens=minimum_summary_tokens,
                desired_output_tokens=target_summary_tokens + 64,
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
            self.telemetry.record_semantic_reduction(
                call_kind="summary",
                target_tokens=target_summary_tokens,
                hierarchical_depth=0,
            )
            try:
                payload, final_prepared = self._execute_with_output_recovery(
                    state,
                    PreparedCall(assembly, report, "lean", contract),
                    minimum_output_tokens=minimum_summary_tokens,
                    desired_output_tokens=target_summary_tokens + 64,
                    context_limit_resolution=context_limit_resolution,
                )
            except _OutputRecoveryContextOverflow as exc:
                minimum_summary_tokens = exc.minimum_output_tokens
                continue
            report = final_prepared.report
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
            recovered_tokens, actual_source_tokens, replacement_tokens = (
                self._compaction_recovery(
                    state,
                    source_messages[:effective_source_count],
                    summary_payload,
                )
            )
            if recovered_tokens <= 0:
                continue
            event_payload = {
                "source_message_count": effective_source_count,
                "source_event_references": source_event_references,
                "source_event_ranges": summary_payload["metadata"]["source_event_ranges"],
                "summary_message": summary_payload,
                "summary_budget_report": asdict(report),
                "adaptive_preserve_recent_messages": preserve_recent,
                "candidate_source_message_count": source_count,
                "required_recovery_tokens": required_recovery,
                "target_summary_tokens": target_summary_tokens,
                "estimated_source_tokens": estimated_source_tokens,
                "actual_source_tokens": actual_source_tokens,
                "actual_replacement_tokens": replacement_tokens,
                "actual_recovered_tokens": recovered_tokens,
            }
            self.history.record_event(state, "summary_created", event_payload)
            self.history.record_event(state, "history_compressed", event_payload)
            self.telemetry.record_history_compaction(
                source_message_count=effective_source_count,
                hierarchical=False,
            )
            return True
        source_messages = state.messages[:1]
        if source_messages and hierarchical_target is not None:
            target_summary_tokens, estimated_source_tokens = hierarchical_target
            summary_text, report = self._summarize_oversized_message(
                state,
                source_messages[0],
                target_summary_tokens=target_summary_tokens,
                context_limit_resolution=context_limit_resolution,
                remaining_calls=[
                    max(16, int(self.config.context.max_compaction_rounds) * 16)
                ],
            )
            source_event_references = message_source_event_references(source_messages)
            summary_payload = summary_message_payload(
                summary_text,
                source_message_count=1,
                created_at=utc_now_iso(),
                source_event_references=source_event_references,
            )
            recovered_tokens, actual_source_tokens, replacement_tokens = (
                self._compaction_recovery(state, source_messages, summary_payload)
            )
            if recovered_tokens <= 0:
                return False
            event_payload = {
                "source_message_count": 1,
                "source_event_references": source_event_references,
                "source_event_ranges": summary_payload["metadata"]["source_event_ranges"],
                "summary_message": summary_payload,
                "summary_budget_report": asdict(report),
                "adaptive_preserve_recent_messages": 0,
                "candidate_source_message_count": 1,
                "hierarchical": True,
                "required_recovery_tokens": required_recovery,
                "target_summary_tokens": target_summary_tokens,
                "estimated_source_tokens": estimated_source_tokens,
                "actual_source_tokens": actual_source_tokens,
                "actual_replacement_tokens": replacement_tokens,
                "actual_recovered_tokens": recovered_tokens,
            }
            self.history.record_event(state, "summary_created", event_payload)
            self.history.record_event(state, "history_compressed", event_payload)
            self.telemetry.record_history_compaction(
                source_message_count=1,
                hierarchical=True,
            )
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

    def _execute_with_output_recovery(
        self,
        state: SessionState,
        prepared: PreparedCall,
        *,
        minimum_output_tokens: int,
        desired_output_tokens: int | None = None,
        validator: Callable[[dict[str, Any]], Any] | None = None,
        context_limit_resolution: tuple[int, str] | None = None,
        include_prompt_instructions: bool = True,
        allow_prompt_instruction_projection: bool = False,
    ) -> tuple[Any, PreparedCall]:
        current = prepared
        current_minimum = max(1, int(minimum_output_tokens))
        for output_retry in range(int(self.config.model.max_retries) + 1):
            try:
                return (
                    self._execute_structured_call(
                        state,
                        current,
                        validator=validator,
                        seed_offset=output_retry,
                    ),
                    current,
                )
            except OutputBudgetExhaustedError:
                if output_retry >= int(self.config.model.max_retries):
                    raise
                expanded = self._expanded_output_minimum(current)
                if expanded <= current_minimum:
                    raise
                self.history.record_event(
                    state,
                    "budget_repaired",
                    {
                        "kind": current.assembly.kind,
                        "reason": "model_output_budget_exhausted",
                        "requested_response_tokens": current.report.reserved_response_tokens,
                        "capped_response_tokens": expanded,
                        "previous_reserved_response_tokens": current.report.reserved_response_tokens,
                        "next_minimum_output_tokens": expanded,
                        "output_retry": output_retry + 1,
                    },
                )
                current_minimum = expanded
                compilation = self._compile_context(
                    state,
                    current.assembly,
                    current.contract,
                    minimum_output_tokens=current_minimum,
                    desired_output_tokens=desired_output_tokens,
                    context_limit_resolution=context_limit_resolution,
                    include_prompt_instructions=include_prompt_instructions,
                )
                cap_error = (
                    "" if compilation.report.fits else "context_limit_exceeded"
                )
                self.history.record_event(
                    state,
                    "context_compiled",
                    {
                        "kind": current.assembly.kind,
                        "prompt_mode": current.prompt_mode,
                        "accounting": compilation.accounting(),
                        "cap_error": cap_error,
                        "output_retry": output_retry + 1,
                    },
                )
                self.history.record_event(
                    state,
                    "budget_checked",
                    {
                        "kind": current.assembly.kind,
                        "prompt_mode": current.prompt_mode,
                        "budget_report": asdict(compilation.report),
                        "cap_error": cap_error,
                        "output_retry": output_retry + 1,
                    },
                )
                if (
                    not compilation.report.fits
                    and include_prompt_instructions
                    and allow_prompt_instruction_projection
                ):
                    recovered = self._recover_prompt_instruction_overflow(
                        state,
                        current.assembly,
                        current.contract,
                        compilation,
                        minimum_output_tokens=current_minimum,
                        desired_output_tokens=desired_output_tokens,
                        context_limit_resolution=context_limit_resolution,
                    )
                    if recovered is not None:
                        compilation = recovered
                        self.history.record_event(
                            state,
                            "context_compiled",
                            {
                                "kind": current.assembly.kind,
                                "prompt_mode": current.prompt_mode,
                                "accounting": recovered.accounting(),
                                "cap_error": "",
                                "output_retry": output_retry + 1,
                                "prompt_instruction_projection": True,
                            },
                        )
                        self.history.record_event(
                            state,
                            "budget_checked",
                            {
                                "kind": current.assembly.kind,
                                "prompt_mode": current.prompt_mode,
                                "budget_report": asdict(recovered.report),
                                "cap_error": "",
                                "output_retry": output_retry + 1,
                                "prompt_instruction_projection": True,
                            },
                        )
                if not compilation.report.fits:
                    raise _OutputRecoveryContextOverflow(
                        compilation,
                        current_minimum,
                    )
                self._record_prompt_built(
                    state,
                    current.assembly,
                    current.contract,
                    compilation.report,
                )
                current = PreparedCall(
                    current.assembly,
                    compilation.report,
                    current.prompt_mode,
                    current.contract,
                )

        raise AssertionError("unreachable output-recovery loop")

    @staticmethod
    def _expanded_output_minimum(prepared: PreparedCall) -> int:
        current = int(prepared.report.reserved_response_tokens)
        ceiling = max(
            current,
            int(prepared.report.context_limit)
            - int(prepared.report.safety_margin_tokens),
        )
        return min(ceiling, max(current + 64, (current * 3 + 1) // 2))

    def _record_inference_started(
        self,
        state: SessionState,
        request: InferenceRequest,
    ) -> None:
        self.telemetry.record_inference_started(
            call_kind=request.call_kind,
            source=request.source,
            priority=request.priority,
            queue_wait_seconds=request.queue_wait_seconds or 0.0,
            active_count=self.inference.active_count(),
            backend_capacity=request.backend_capacity or 1,
        )
        self.history.record_event(
            state,
            "inference_request_started",
            {
                "request_id": request.request_id,
                "call_id": request.call_id,
                "kind": request.call_kind,
                "source": request.source,
                "priority": request.priority,
                "attempt": request.attempt_count,
                "queue_wait_seconds": request.queue_wait_seconds or 0.0,
                "backend_capacity": request.backend_capacity or 1,
                "capacity_source": request.capacity_source or "unknown",
            },
        )

    def _requeue_inference(
        self,
        state: SessionState,
        request_id: str,
        *,
        reason: str,
    ) -> InferenceRequest:
        before = self.inference.get(request_id)
        request = self.inference.requeue(request_id, reason=reason)
        if before is not None and before.status == "running":
            self.telemetry.record_inference_released(
                call_kind=request.call_kind,
                source=request.source,
                priority=request.priority,
                status="requeued",
            )
        self.history.record_event(
            state,
            "inference_request_requeued",
            {
                "request_id": request.request_id,
                "call_id": request.call_id,
                "kind": request.call_kind,
                "reason": reason,
                "attempt": request.attempt_count,
            },
        )
        return request

    def _finish_inference(
        self,
        state: SessionState,
        request_id: str,
        *,
        status: str,
        error: str = "",
        cancellation_requested_at: str | None = None,
        record_history: bool = True,
    ) -> InferenceRequest:
        before = self.inference.get(request_id)
        if status == "completed":
            request = self.inference.complete(request_id)
        elif status == "cancelled":
            request = self.inference.cancel(
                request_id,
                reason=error,
                requested_at=cancellation_requested_at,
            )
        elif status == "superseded":
            request = self.inference.supersede(request_id, reason=error)
        elif status == "failed":
            request = self.inference.fail(request_id, error=error)
        else:
            raise ValueError(f"unknown inference terminal status: {status}")
        if before is not None and before.status == "running":
            cancellation_latency = None
            if status == "cancelled" and request.cancellation_requested_at:
                cancellation_latency = _iso_elapsed_seconds(
                    request.cancellation_requested_at,
                    request.completed_at,
                )
            self.telemetry.record_inference_released(
                call_kind=request.call_kind,
                source=request.source,
                priority=request.priority,
                status=request.status,
                cancellation_latency_seconds=cancellation_latency,
            )
        if record_history:
            self._record_inference_finished_event(state, request, error=error)
        return request

    def _record_inference_finished_event(
        self,
        state: SessionState,
        request: InferenceRequest,
        *,
        error: str,
    ) -> None:
        self.history.record_event(
            state,
            "inference_request_finished",
            {
                "request_id": request.request_id,
                "call_id": request.call_id,
                "kind": request.call_kind,
                "status": request.status,
                "attempt": request.attempt_count,
                "error": error,
            },
        )

    def _handle_model_preemption(
        self,
        state: SessionState,
        prepared: PreparedCall,
        *,
        call_id: str,
        inference_request_id: str,
        active_call: Any,
        guard: Any,
        attempt: int,
        policy: Any,
        frozen_request: dict[str, Any],
    ) -> None:
        if self._run_cancellation_requested(state):
            run_id = self._active_run_id(state)
            cancellation = self.preemption.run_cancellation(state.session_id, run_id)
            guard.record(
                "model_call_preempted",
                {
                    "kind": prepared.assembly.kind,
                    "prompt_mode": prepared.prompt_mode,
                    "attempt": attempt,
                    "call_id": call_id,
                    "preemption_id": f"run_cancellation:{run_id}",
                    "request_sha256": active_call.request_sha256,
                    "reason": "run_cancellation_requested",
                },
            )
            self._finish_inference(
                state,
                inference_request_id,
                status="cancelled",
                error="worker run cancellation requested",
                cancellation_requested_at=(
                    None if cancellation is None else cancellation.requested_at
                ),
            )
            self.preemption.clear_active(state.session_id, call_id)
            raise RunCancellationRequested("worker run cancellation requested")
        pending = self.preemption.pending_for_call(state.session_id, call_id)
        if pending is None:
            self._finish_inference(
                state,
                inference_request_id,
                status="cancelled",
                error="backend interrupted without a pending coordinator request",
            )
            self.preemption.clear_active(state.session_id, call_id)
            raise ModelCallPreempted("model call interrupted without a pending request")
        guard.record(
            "model_call_preempted",
            {
                "kind": prepared.assembly.kind,
                "prompt_mode": prepared.prompt_mode,
                "attempt": attempt,
                "call_id": call_id,
                "preemption_id": pending.preemption_id,
                "request_sha256": active_call.request_sha256,
            },
        )
        current = self.inference.get(inference_request_id)
        if current is not None and current.status == "running":
            self._requeue_inference(
                state,
                inference_request_id,
                reason=f"preempted:{pending.preemption_id}",
            )
        # Publish interruption only after canonical target-session evidence is durable.
        self.preemption.mark_interrupted(pending.preemption_id)
        resolved = self.preemption.wait_for_status(
            pending.preemption_id,
            {"completed", "failed"},
            timeout_seconds=max(
                1.0,
                float(policy.effective_timeout_seconds),
                float(self.config.model.timeout_seconds),
                float(self.config.model.structured_timeout_seconds),
            ),
            poll_seconds=0.02,
        )
        if resolved.status == "failed":
            self._finish_inference(
                state,
                inference_request_id,
                status="failed",
                error=f"communication preemption failed: {resolved.reply or 'unknown error'}",
            )
            self.preemption.clear_active(state.session_id, call_id)
            raise ModelClientError(
                f"communication preemption failed: {resolved.reply or 'unknown error'}"
            )
        if resolved.target_changed:
            finished = self._finish_inference(
                state,
                inference_request_id,
                status="superseded",
                error="target session changed during communication",
                record_history=False,
            )
            self.preemption.clear_active(state.session_id, call_id)
            self._refresh_state_from_history(state)
            guard = self.history.guard(
                state, f"model_call:{prepared.assembly.kind}:preemption"
            )
            guard.record(
                "model_call_replay_invalidated",
                {
                    "kind": prepared.assembly.kind,
                    "call_id": call_id,
                    "preemption_id": pending.preemption_id,
                    "request_sha256": active_call.request_sha256,
                },
            )
            self._record_inference_finished_event(
                state,
                finished,
                error="target session changed during communication",
            )
            raise ModelCallStateChanged(
                "target session changed during communication; stale model request was not replayed"
            )
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

    def _execute_model_call(
        self,
        state: SessionState,
        prepared: PreparedCall,
        *,
        seed_offset: int = 0,
    ) -> CompletionResult:
        call_id = new_id("model_call")
        with self.telemetry.model_call(
            session_id=state.session_id,
            run_id=self._active_run_id(state),
            call_id=call_id,
            call_kind=prepared.assembly.kind,
            operation_name=(
                "chat"
                if uses_chat_completions_transport(
                    self.config.model.base_url,
                    self.config.model.completion_endpoint,
                )
                else "text_completion"
            ),
            provider_name=self.config.model.provider_name,
            model_name=self.config.model.model_identity,
            base_url=self.config.model.base_url,
            max_tokens=prepared.report.reserved_response_tokens,
            cache_mode=(
                self.config.model.cache_mode
                if self.config.model.cache_enabled
                else "disabled"
            ),
        ) as operation:
            completion = self._execute_model_call_inner(
                state,
                prepared,
                call_id=call_id,
                telemetry_operation=operation,
                seed_offset=seed_offset,
            )
            operation.record_model_completion(
                completion,
                budget_report=prepared.report,
            )
            return completion

    def _execute_model_call_inner(
        self,
        state: SessionState,
        prepared: PreparedCall,
        *,
        call_id: str,
        telemetry_operation: TelemetryOperation,
        seed_offset: int = 0,
    ) -> CompletionResult:
        protocol_artifact = next(
            (
                artifact
                for artifact in prepared.assembly.prompt_artifacts
                if artifact.source == "prompt_protocol:server_chat_template"
            ),
            None,
        )
        verifier = getattr(self.client, "verify_prompt_protocol", None)
        if protocol_artifact is not None and callable(verifier):
            verifier(protocol_artifact.sha256)
        resolved_contract, policy = self.client.resolve_contract(
            prepared.contract,
            kind=prepared.assembly.kind,
            prompt=prepared.assembly.prompt_text,
            max_tokens=prepared.report.reserved_response_tokens,
        )
        request_builder = self.client.build_completion_request
        request_kwargs: dict[str, Any] = {
            "max_tokens": prepared.report.reserved_response_tokens,
            "contract": resolved_contract,
        }
        builder_parameters = inspect.signature(request_builder).parameters.values()
        if prepared.assembly.message_ranges and any(
            parameter.name == "messages"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in builder_parameters
        ):
            request_kwargs["messages"] = self._assembly_chat_messages(
                prepared.assembly
            )
        request = request_builder(
            prepared.assembly.prompt_text,
            **request_kwargs,
        )
        # Reproducible but non-identical decoding across semantic action/retry attempts.
        # Reusing one fixed seed caused malformed JSON and exact bad actions to recur
        # deterministically even after validation feedback changed.
        request["seed"] = int(self.config.model.seed) + int(seed_offset)
        self._heartbeat(state, phase="queued_inference", detail=f"queued {prepared.assembly.kind}", active_kind="model", active_id=call_id)
        active_call = self.preemption.register_active(
            state.session_id,
            call_id,
            prepared.assembly.kind,
            request,
        )
        frozen_request = active_call.request
        inference_priority, inference_source = self._current_inference_priority()
        inference_request = self.inference.enqueue(
            session_id=state.session_id,
            run_id=self._active_run_id(state),
            call_id=call_id,
            call_kind=prepared.assembly.kind,
            priority=inference_priority,
            source=inference_source,
        )
        self.telemetry.record_inference_queued(
            call_kind=inference_request.call_kind,
            source=inference_request.source,
            priority=inference_request.priority,
            queue_depth=self.inference.queue_depth(),
        )
        self.history.record_event(
            state,
            "inference_request_queued",
            {
                "request_id": inference_request.request_id,
                "call_id": inference_request.call_id,
                "kind": inference_request.call_kind,
                "source": inference_request.source,
                "priority": inference_request.priority,
                "backend_key": inference_request.backend_key,
            },
        )
        transient_attempts = 0
        semantic_attempt = 0
        total_attempt = 0
        while True:
            total_attempt += 1
            guard = self.history.guard(state, f"model_call:{prepared.assembly.kind}")
            self._heartbeat(
                state,
                phase="queued_inference",
                detail=f"queued {prepared.assembly.kind}",
                active_kind="model",
                active_id=call_id,
            )
            try:
                acquired = self.inference.acquire(
                    inference_request.request_id,
                    cancel_check=lambda: (
                        self._run_cancellation_requested(state)
                        or self.preemption.pending_for_call(
                            state.session_id, call_id
                        )
                        is not None
                    ),
                )
            except ModelCallPreempted:
                telemetry_operation.record_preemption()
                self._handle_model_preemption(
                    state,
                    prepared,
                    call_id=call_id,
                    inference_request_id=inference_request.request_id,
                    active_call=active_call,
                    guard=guard,
                    attempt=total_attempt,
                    policy=policy,
                    frozen_request=frozen_request,
                )
                continue
            self._record_inference_started(state, acquired)
            guard.record(
                "model_request_sent",
                {
                    "call_id": call_id,
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
                        "call_id": call_id,
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
                heartbeat_interval = max(
                    0.5,
                    min(5.0, float(policy.progress_poll_seconds)),
                )
                with self._periodic_model_heartbeat(
                    state,
                    call_id=call_id,
                    call_kind=prepared.assembly.kind,
                    interval_seconds=heartbeat_interval,
                ):
                    completion = send(frozen_request, **kwargs)
            except ModelCallPreempted:
                telemetry_operation.record_preemption()
                self._handle_model_preemption(
                    state,
                    prepared,
                    call_id=call_id,
                    inference_request_id=inference_request.request_id,
                    active_call=active_call,
                    guard=guard,
                    attempt=total_attempt,
                    policy=policy,
                    frozen_request=frozen_request,
                )
                continue
            except Exception as exc:
                if self._is_model_server_unavailable(exc):
                    transient_attempts += 1
                    telemetry_operation.record_retry()
                    guard.record(
                        "retry",
                        {
                            "call_id": call_id,
                            "operation": "model_unavailable",
                            "reason": str(exc),
                            "attempt": transient_attempts,
                            "next_attempt": transient_attempts + 1,
                        },
                    )
                    if transient_attempts > self._max_model_unavailable_attempts:
                        self._finish_inference(
                            state,
                            inference_request.request_id,
                            status="failed",
                            error="model_unavailable",
                        )
                        raise ModelClientError("model_unavailable") from exc
                    self._requeue_inference(
                        state,
                        inference_request.request_id,
                        reason="model_unavailable_retry",
                    )
                    self._sleep(self._model_unavailable_backoff_seconds(transient_attempts - 1))
                    continue
                guard.record(
                    "model_call_failed",
                    {
                        "call_id": call_id,
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
                    telemetry_operation.record_retry()
                    guard.record(
                        "model_retry_scheduled",
                        {
                            "call_id": call_id,
                            "kind": prepared.assembly.kind,
                            "prompt_mode": prepared.prompt_mode,
                            "next_attempt": semantic_attempt + 1,
                        },
                    )
                    self._requeue_inference(
                        state,
                        inference_request.request_id,
                        reason="semantic_model_retry",
                    )
                    continue
                self._finish_inference(
                    state,
                    inference_request.request_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.preemption.clear_active(state.session_id, call_id)
                raise

            self._finish_inference(
                state,
                inference_request.request_id,
                status="completed",
            )
            self.preemption.clear_active(state.session_id, call_id)
            guard.record(
                "model_response_received",
                {
                    "call_id": call_id,
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

    def _execute_tool_with_error(
        self, state: SessionState, decision: ToolDecision
    ) -> tuple[ToolExecutionResult | None, dict[str, Any] | None]:
        call_id = new_id("tool_call")
        with self.telemetry.tool_execution(
            session_id=state.session_id,
            run_id=self._active_run_id(state),
            call_id=call_id,
            tool_name=decision.tool_name,
        ) as operation:
            result, error = self._execute_tool_with_error_inner(
                state,
                decision,
                call_id=call_id,
            )
            if error is not None:
                operation.record_error(
                    str(error.get("error_type", "_OTHER")),
                    str(error.get("error", "")),
                )
            return result, error

    def _execute_tool_with_error_inner(
        self,
        state: SessionState,
        decision: ToolDecision,
        *,
        call_id: str,
    ) -> tuple[ToolExecutionResult | None, dict[str, Any] | None]:
        guard = self.history.guard(state, f"tool:{decision.tool_name}")
        guard.record(
            "tool_called",
            {
                "call_id": call_id,
                "tool_name": decision.tool_name,
                "tool_input": decision.tool_input,
            },
        )
        try:
            tool, context, invocation = self.tools.prepare(
                decision.tool_name,
                decision.tool_input,
                self.config,
                state,
                semantic_call=lambda request: self._execute_tool_semantic_call(
                    state, request
                ),
            )
            guard.record(
                "tool_execution_context",
                {
                    "call_id": call_id,
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
                "call_id": call_id,
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
            return None, error_payload

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
                "call_id": call_id,
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
        return result, None

    def _execute_tool(self, state: SessionState, decision: ToolDecision) -> ToolExecutionResult | None:
        result, _error = self._execute_tool_with_error(state, decision)
        return result

    def execute_tool_once(
        self,
        tool_name: str,
        raw_input: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> ToolRunResult:
        state = self.create_or_load_session(session_id)
        result, error = self._execute_tool_with_error(
            state,
            ToolDecision(
                action="call_tool",
                response="",
                tool_name=tool_name,
                tool_input=raw_input,
            ),
        )
        return ToolRunResult(session_id=state.session_id, tool_result=result, error=error)

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
            "status_kind": "mechanical",
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

    def latest_semantic_status_payload(
        self, state: SessionState
    ) -> dict[str, Any] | None:
        candidates: list[tuple[str, dict[str, Any]]] = []
        for event in reversed(self.history.read_history(state.session_id)):
            if event.event_type != "agent_status":
                continue
            candidates.append(
                (
                    event.timestamp,
                    {
                        **to_jsonable(event.payload),
                        "status_kind": "worker_action_status",
                        "status_event_sequence": event.sequence,
                        "status_event_hash": event.hash,
                    },
                )
            )
            break

        # Independent status calls use their own append-only operation session,
        # so they never race with or mutate the target worker history.
        event = self.history.latest_communication_status(state.session_id)
        if event is not None:
            status = event.payload.get("status")
            if isinstance(status, dict):
                candidates.append(
                    (
                        event.timestamp,
                        {
                            **to_jsonable(status),
                            "status_kind": "independent_communication_status",
                            "status_operation_session_id": event.session_id,
                            "status_event_sequence": event.sequence,
                            "status_event_hash": event.hash,
                        },
                    )
                )
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

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
        payload, _final_prepared = self._execute_with_output_recovery(
            state,
            PreparedCall(assembly, report, "lean", contract),
            minimum_output_tokens=self.config.context.reserved_response_tokens,
            allow_prompt_instruction_projection=True,
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


def _iso_elapsed_seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        start_time = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except ValueError:
        return None
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    return max(
        0.0,
        (end_time.astimezone(timezone.utc) - start_time.astimezone(timezone.utc)).total_seconds(),
    )
