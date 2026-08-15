from __future__ import annotations

import inspect
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from swaag.action import ActionValidationError, AgentAction, action_from_payload
from swaag.budgeting import compute_call_budget, structured_output_token_floor
from swaag.compression import summary_message_payload
from swaag.config import AgentConfig, load_config
from swaag.environment.environment import AgentEnvironment
from swaag.fsops import ensure_dir, restore_tree, snapshot_tree, write_text
from swaag.grammar import agent_action_contract, summary_contract, yes_no_contract
from swaag.history import HistoryInvariantError, HistoryStore
from swaag.model import LlamaCppClient, ModelClientError
from swaag.model_cache import build_model_client
from swaag.notes import select_notes_for_prompt
from swaag.prompts import PromptBuilder
from swaag.scheduler import WakeupStore
from swaag.tokens import ConservativeEstimator, CountResult, ExactTokenCounter, build_budget
from swaag.tools.registry import ToolRegistry
from swaag.types import (
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
        self.client = model_client or build_model_client(
            config,
            request_metadata={"cache_scope": "default_agent_runtime"},
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

    def run_turn(self, user_text: str, *, session_id: str | None = None) -> TurnResult:
        state = self.create_or_load_session(session_id)
        return self.run_turn_in_session(state, user_text)

    def run_turn_in_session(self, state: SessionState, user_text: str) -> TurnResult:
        run_id = f"{state.session_id}:{new_id('run')}"
        self.history.set_active_run(state.session_id, run_id=run_id, user_text=user_text)
        try:
            return self._run_model_tool_loop(state, user_text, record_user_message=True)
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
        try:
            return self._run_model_tool_loop(state, original_request, record_user_message=False)
        finally:
            self.history.clear_active_run(state.session_id, run_id=run_id)

    def _run_model_tool_loop(self, state: SessionState, user_text: str, *, record_user_message: bool = True) -> TurnResult:
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

        all_tool_specs = self.tools.prompt_tuples(self.config)
        tool_results: list[ToolExecutionResult] = []
        budget_reports: list[BudgetReport] = []
        previous_action_signature = ""
        consecutive_action_occurrences = 0
        rejected_signature_counts: dict[str, int] = {}
        rejected_observation_counts: dict[str, int] = {}
        tool_calls_used = 0
        observation_signatures_since_state_change: set[str] = set()
        recovery_feedback = ""
        accepted_actions = 0
        max_mechanical_attempts = max(
            self.config.runtime.max_total_actions * 3,
            self.config.runtime.max_total_actions + 8,
        )

        for mechanical_attempt in range(1, max_mechanical_attempts + 1):
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

            for validation_attempt in range(1, 4):
                remaining_tool_calls = self.config.runtime.tool_call_budget - tool_calls_used
                tool_specs = all_tool_specs if remaining_tool_calls > 0 else []
                tool_names = [str(item[0]) for item in tool_specs]
                contract = agent_action_contract(tool_specs)
                prepared = self._prepare_action_call(
                    state,
                    original_request=original_request,
                    pending_messages=pending_messages,
                    tool_specs=tool_specs,
                    contract=contract,
                    validation_feedback=validation_feedback,
                )
                budget_reports.append(prepared.report)

                def validate(payload: dict[str, Any]) -> AgentAction:
                    action = action_from_payload(payload, enabled_tool_names=tool_names)
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

            if selected_action is None:
                recovery_feedback = validation_feedback or (
                    "The previous mechanical action could not be validated. Produce a different valid action that follows the exact tool schemas and remaining budget."
                )
                continue

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
                continue

            repeated_observation_calls: list[dict[str, Any]] = []
            for tool_call in selected_action.tool_calls:
                tool = self.tools.get(tool_call.tool_name)
                if not tool.repeated_observation_is_redundant:
                    continue
                observation_signature = stable_json_dumps(
                    {"tool_name": tool_call.tool_name, "arguments": tool_call.arguments},
                    indent=None,
                )
                if observation_signature in observation_signatures_since_state_change:
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

            self.history.record_event(
                state,
                "agent_action_terminal",
                {
                    "action_index": action_index,
                    "continue_loop": selected_action.continue_loop,
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
        contract: ContractSpec,
        validation_feedback: str,
    ) -> PreparedCall:
        last_report: BudgetReport | None = None
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
                validation_feedback=validation_feedback,
            )
            report = self._budget_report(state, assembly, contract)
            last_report = report
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
            if not self._compact_once(state):
                break

        raise BudgetExceededError(
            "The exact action prompt, tool schemas, output reserve, and safety margin do not fit the model context.",
            last_report,
        )

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

    def _budget_report(
        self,
        state: SessionState | None,
        assembly: PromptAssembly,
        contract: ContractSpec,
    ) -> BudgetReport:
        counter = self._counter(state)
        plan = compute_call_budget(self.config, call_kind=assembly.kind)
        structured_floor = structured_output_token_floor(
            contract,
            config=self.config,
            counter=counter,
            call_kind=assembly.kind,
        )
        reserved = max(
            int(self.config.context.reserved_response_tokens),
            int(plan.output_tokens),
            int(structured_floor),
        )
        components = [
            *assembly.components,
            PromptComponent(
                name="constraint_schema",
                category="constraint_schema",
                text=stable_json_dumps(contract.json_schema or {}, indent=None),
                include_in_context=False,
            ),
        ]
        return build_budget(
            counter,
            components,
            self.config.context,
            self.config.model.context_limit,
            reserved_response_tokens=reserved,
            safety_margin_tokens=plan.safety_margin_tokens,
        )

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
        for source_count in range(maximum_source, 0, -1):
            source_messages = state.messages[:source_count]
            adaptive_cap = min(max(0, source_count - 1), max(0, int(self.config.context.max_recent_messages) * 4))
            assembly = self.prompts.build_summary_prompt(
                source_messages,
                prompt_mode="lean",
                maximum_preserve_recent_messages=adaptive_cap,
            )
            report = self._summary_budget_report(state, assembly, contract)
            if not report.fits:
                continue
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
            summary_payload = summary_message_payload(
                summary_text,
                source_message_count=effective_source_count,
                created_at=utc_now_iso(),
            )
            event_payload = {
                "source_message_count": effective_source_count,
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
        counter = self._counter(state)
        floor = structured_output_token_floor(
            contract,
            config=self.config,
            counter=counter,
            call_kind="summary",
        )
        reserved = max(int(self.config.context.reserved_summary_tokens), int(floor))
        return build_budget(
            counter,
            [
                *assembly.components,
                PromptComponent(
                    name="constraint_schema",
                    category="constraint_schema",
                    text=stable_json_dumps(contract.json_schema or {}, indent=None),
                    include_in_context=False,
                ),
            ],
            self.config.context,
            self.config.model.context_limit,
            reserved_response_tokens=reserved,
            safety_margin_tokens=self.config.context.safety_margin_tokens,
        )

    def _execute_structured_call(
        self,
        state: SessionState,
        prepared: PreparedCall,
        *,
        validator: Callable[[dict[str, Any]], Any] | None = None,
        seed_offset: int = 0,
    ) -> Any:
        completion = self._execute_model_call(state, prepared, seed_offset=seed_offset)
        try:
            payload = json.loads(completion.text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Contract {prepared.contract.name} returned malformed JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Contract {prepared.contract.name} must return one JSON object")
        return validator(payload) if validator is not None else payload

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
                    "request": request,
                    "budget_report": asdict(prepared.report),
                    "policy": asdict(policy),
                    "token_timeout_seconds": policy.effective_timeout_seconds,
                    "requested_contract_mode": prepared.contract.mode,
                    "effective_contract_mode": resolved_contract.mode,
                },
            )
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
                    supports_progress = (
                        "progress_callback" in signature.parameters
                        or any(
                            item.kind == inspect.Parameter.VAR_KEYWORD
                            for item in signature.parameters.values()
                        )
                    )
                except (TypeError, ValueError):
                    supports_progress = False
                kwargs: dict[str, Any] = {"timeout_seconds": policy.effective_timeout_seconds}
                if supports_progress:
                    kwargs["progress_callback"] = progress_callback
                completion = send(request, **kwargs)
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
                raise

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
        guard.record(
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
        return {
            "session_id": state.session_id,
            "session_name": state.session_name,
            "active_goal": latest_user,
            "active_step": "",
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
