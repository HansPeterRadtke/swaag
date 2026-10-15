from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

from swaag.history import HistoryStore
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import stable_json_dumps


def _resolve_control_target(context: ToolContext, ref: str) -> tuple[HistoryStore, str]:
    """Resolve exact worker IDs through their owner; names/IDs remain session refs."""
    orchestration = context.runtime_capabilities.get("orchestration")
    manager = getattr(orchestration, "manager", None)
    for workers in getattr(manager, "worker_managers", {}).values():
        try:
            worker = workers.store.get(ref)
        except FileNotFoundError:
            continue
        owner = manager.worker_managers.get(worker.model_key, workers)
        return owner.runtime.history, worker.session_id
    store = HistoryStore(context.config.sessions.root, write_projections=False)
    session_id = store.resolve_session_ref(ref, latest_if_none=False)
    if session_id is None:
        raise FileNotFoundError(f"Unknown target session or worker: {ref}")
    return store, session_id


def _originating_user_authority(context: ToolContext) -> list[dict[str, Any]]:
    # Only the trusted foreground controller supplies original user authority.
    # A worker's own message or tool output cannot promote itself into this lane.
    if "system_snapshot_provider" not in context.runtime_capabilities:
        return []
    history = HistoryStore(context.config.sessions.root, write_projections=False)
    events = history.iter_history_reverse(context.session_state.session_id)
    messages = []
    for event in events:
        if event.event_type == "turn_started":
            original = event.payload.get("user_text")
            if isinstance(original, str) and not any(item["message"] == original for item in messages):
                messages.append({"source_session_id": context.session_state.session_id,
                                 "source_event_sequence": event.sequence,
                                 "message": original})
            break
        if event.event_type != "message_added":
            continue
        message = event.payload.get("message", {})
        if message.get("role") == "user":
            messages.append({"source_session_id": context.session_state.session_id,
                             "source_event_sequence": event.sequence,
                             "control_id": message.get("metadata", {}).get("orchestrator_control_id"),
                             "message": message["content"]})
    # The initial message precedes turn_started in some entry paths. Use the
    # current user turn only, never earlier unrelated conversation.
    if not messages:
        latest = next((m for m in reversed(context.session_state.messages) if m.role == "user"), None)
        if latest is not None:
            messages.append({"source_session_id": context.session_state.session_id,
                             "control_id": latest.metadata.get("orchestrator_control_id"),
                             "message": latest.content})
    return list(reversed(messages))


class AgentStatusLookupTool(Tool):
    name = "agent_status_lookup"
    description = "Read durable status for another SWAAG session without changing it."
    usage_guidance = "Use an exact worker ID or session id/name. Returns active goal, waiting/running state, pending controls, turn/event counts, and the complete durable semantic status history."
    kind = "pure"
    input_schema = {
        "type": "object",
        "properties": {"session_ref": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["session_ref"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        ref = raw_input.get("session_ref")
        if ref is not None and not isinstance(ref, str):
            raise ToolValidationError("agent_status_lookup.session_ref must be a string or null")
        return {"session_ref": (ref or "").strip()}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        ref = validated_input["session_ref"] or context.session_state.session_id
        store, session_id = _resolve_control_target(context, ref)
        state = store.rebuild_from_history(session_id, write_projections=False)
        latest_user = next((m.content for m in reversed(state.messages) if m.role == "user"), "")
        status_history = [
            event.payload
            for event in store.iter_history(session_id)
            if event.event_type == "agent_status"
        ]
        active_run = store.read_active_run(session_id)
        output = {
            "session_id": session_id,
            "session_name": state.session_name,
            "active_goal": latest_user,
            "mechanical_phase": "idle" if active_run is None else str(active_run.get("phase", "unknown")),
            "heartbeat_at": "" if active_run is None else str(active_run.get("heartbeat_at", "")),
            "active_run": active_run,
            "waiting": state.environment.waiting,
            "waiting_reason": state.environment.waiting_reason,
            "running_process_ids": [pid for pid, item in state.environment.processes.items() if item.status == "running"],
            "pending_control_count": len(store.list_pending_control_messages(session_id)),
            "turn_count": state.turn_count,
            "event_count": state.event_count,
            "status_history": status_history,
        }
        return ToolExecutionResult(self.name, output, f"agent_status_lookup result: {stable_json_dumps(output, indent=2)}")


class AgentControlTool(Tool):
    name = "agent_control"
    description = "Queue an exact, ordered, durable control/instruction for another SWAAG session."
    usage_guidance = "Use an exact worker ID or session id/name when the user redirects, pauses, stops, or instructs the target. Preserve exact wording. queued=true confirms durable enqueue only, not acknowledgement or effects. Report the pending state and continue in the background; do not resend to poll. Original user authority is preserved separately from your routing note. Retries for the same original user events and target reuse the first receipt; status reports pending or processed, never proof of completed effects."
    kind = "side_effect"
    input_schema = {
        "type": "object",
        "properties": {
            "session_ref": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["session_ref", "message"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        ref = raw_input.get("session_ref")
        message = raw_input.get("message")
        if not isinstance(ref, str) or not ref.strip():
            raise ToolValidationError("agent_control.session_ref must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise ToolValidationError("agent_control.message must be a non-empty string")
        return {"session_ref": ref.strip(), "message": message}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        store, session_id = _resolve_control_target(context, validated_input["session_ref"])
        authority = _originating_user_authority(context)
        # The original foreground events are the action identity. A generated
        # routing-note paraphrase, target alias, or retry must not deliver them
        # twice. A new user event, changed authority set, or target gets a new ID.
        # Direct/legacy worker controls have no such provenance and stay distinct.
        control_id = None
        if authority:
            identity = stable_json_dumps(
                {"target_session_id": session_id, "authority": authority}, indent=None
            )
            control_id = "control_authority_" + hashlib.sha256(identity.encode()).hexdigest()
        payload = store.enqueue_control_message(
            session_id, validated_input["message"], source="agent_control",
            control_id=control_id, authority=authority,
        )
        # queued retains its durable-acceptance meaning; status is the observed
        # delivery state and may already be processed when returning a retry.
        output = {**payload, "queued": True}
        return ToolExecutionResult(
            self.name, output,
            f"Durable control {payload['control_id']} for {session_id}: {payload['status']}",
        )


CONTROL_TOOLS = [AgentStatusLookupTool(), AgentControlTool()]
