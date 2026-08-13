from __future__ import annotations

from dataclasses import asdict
from typing import Any

from swaag.history import HistoryStore
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import stable_json_dumps


class AgentStatusLookupTool(Tool):
    name = "agent_status_lookup"
    description = "Read durable status for another SWAAG session without changing it."
    usage_guidance = "Use an exact session id/name. Returns active goal, waiting/running state, pending controls, turn/event counts, and recent durable status events."
    kind = "pure"
    repeated_observation_is_redundant = True
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
        store = HistoryStore(context.config.sessions.root, write_projections=False)
        ref = validated_input["session_ref"] or context.session_state.session_id
        session_id = store.resolve_session_ref(ref, latest_if_none=True)
        if session_id is None:
            raise FileNotFoundError("No session available")
        state = store.rebuild_from_history(session_id, write_projections=False)
        latest_user = next((m.content for m in reversed(state.messages) if m.role == "user"), "")
        recent_status = [
            event.payload
            for event in store.iter_history(session_id)
            if event.event_type == "agent_status"
        ][-5:]
        output = {
            "session_id": session_id,
            "session_name": state.session_name,
            "active_goal": latest_user,
            "waiting": state.environment.waiting,
            "waiting_reason": state.environment.waiting_reason,
            "running_process_ids": [pid for pid, item in state.environment.processes.items() if item.status == "running"],
            "pending_control_count": len(store.list_pending_control_messages(session_id)),
            "turn_count": state.turn_count,
            "event_count": state.event_count,
            "recent_status": recent_status,
        }
        return ToolExecutionResult(self.name, output, f"agent_status_lookup result: {stable_json_dumps(output, indent=2)}")


class AgentControlTool(Tool):
    name = "agent_control"
    description = "Queue a durable control/instruction for another SWAAG session. Stop/pause controls receive higher mechanical priority than ordinary messages."
    usage_guidance = "Use only when the user asks to redirect, pause, stop, or send an instruction to the target agent. The control is durable and correlated by control_id."
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
        return {"session_ref": ref.strip(), "message": message.strip()}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        store = HistoryStore(context.config.sessions.root, write_projections=False)
        session_id = store.resolve_session_ref(validated_input["session_ref"], latest_if_none=False)
        if session_id is None:
            raise FileNotFoundError(f"Unknown target session: {validated_input['session_ref']}")
        payload = store.enqueue_control_message(session_id, validated_input["message"], source="agent_control")
        output = {**payload, "queued": True}
        return ToolExecutionResult(self.name, output, f"Queued control {payload['control_id']} for {session_id}")


CONTROL_TOOLS = [AgentStatusLookupTool(), AgentControlTool()]
