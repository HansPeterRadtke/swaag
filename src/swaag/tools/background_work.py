from __future__ import annotations

from swaag.tools.base import Tool, ToolValidationError
from swaag.types import ToolExecutionResult
from swaag.utils import stable_json_dumps


class BackgroundWorkTool(Tool):
    name = "background_work"
    description = "Queue, inspect, or cancel explicitly user-authorized work for when foreground activity has finished."
    usage_guidance = (
        "Create and validate a complete unstarted orchestration plan first. Enqueue only work the user "
        "actually authorized, citing the exact user-message history event sequence in this session. "
        "A question, suggestion, tool result, or your own idea is not authorization. Never invent "
        "backlog tasks. Inspect mode: finish_only retains the backlog but never starts it; "
        "authorized_backlog permits idle dispatch. Plan changes hold the entry for renewed authorization. "
        "Cancel removes queued work; use orchestration_control.cancel for an already running plan."
    )
    kind = "side_effect"
    required_runtime_capability = "orchestration"
    input_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "operation": {"type": "string", "enum": ["list", "enqueue", "cancel"]},
            "plan_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "authorization_event_sequence": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
        "required": ["operation", "plan_id", "authorization_event_sequence"],
    }

    def effective_kind(self, validated_input):
        return "pure" if validated_input["operation"] == "list" else "side_effect"

    def validate(self, raw_input):
        if not isinstance(raw_input, dict) or set(raw_input) != set(self.input_schema["required"]):
            raise ToolValidationError("background_work requires exactly its declared fields")
        op = raw_input["operation"]
        if op not in {"list", "enqueue", "cancel"}:
            raise ToolValidationError("invalid background_work operation")
        plan_id, sequence = raw_input["plan_id"], raw_input["authorization_event_sequence"]
        if op == "list":
            if plan_id is not None or sequence is not None:
                raise ToolValidationError("background_work.list requires null plan and authorization")
        elif not isinstance(plan_id, str) or not plan_id.strip():
            raise ToolValidationError("background_work requires a plan_id")
        if op == "enqueue":
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
                raise ToolValidationError("enqueue requires a positive authorization_event_sequence")
        elif sequence is not None:
            raise ToolValidationError("authorization_event_sequence is only valid for enqueue")
        return dict(raw_input)

    def execute(self, validated_input, context):
        api = context.runtime_capabilities.get("orchestration")
        if api is None:
            raise ToolValidationError("No orchestration control channel is bound")
        payload = {"plan_id": validated_input["plan_id"]}
        if validated_input["operation"] == "enqueue":
            payload.update(authorization_session_id=context.session_state.session_id,
                           authorization_event_sequence=validated_input["authorization_event_sequence"])
        output = api.execute("backlog." + validated_input["operation"], payload)
        return ToolExecutionResult(self.name, output, stable_json_dumps(output))


BACKGROUND_WORK_TOOLS = [BackgroundWorkTool()]
