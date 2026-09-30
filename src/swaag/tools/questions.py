from __future__ import annotations

from swaag.history import HistoryStore
from swaag.questions import RESOLUTIONS
from swaag.tools.base import Tool, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import stable_json_dumps


def nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


class QuestionsTool(Tool):
    name = "questions"
    description = "List unresolved questions or explicitly resolve one using recorded user/research evidence."
    usage_guidance = (
        "Unanswered optional questions do not block safe work. Reconcile questions when user answers, "
        "research answers, or changed circumstances make them superseded, irrelevant, or expired. "
        "Use the exact question_id and history event sequences supporting your decision. A new message "
        "alone is not an answer. Never label a provisional assumption as an answer. Resolution records "
        "remain in exact history after the question leaves the open list."
    )
    kind = "stateful"
    input_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "operation": {"type": "string", "enum": ["list", "resolve"]},
            "question_id": nullable({"type": "string"}),
            "resolution": nullable({"type": "string", "enum": list(RESOLUTIONS)}),
            "answer": nullable({"type": "string"}),
            "reason": nullable({"type": "string"}),
            "evidence_sequences": nullable({"type": "array", "items": {"type": "integer"}}),
        },
        "required": ["operation", "question_id", "resolution", "answer", "reason", "evidence_sequences"],
    }

    def effective_kind(self, validated_input):
        return "pure" if validated_input["operation"] == "list" else "stateful"

    def validate(self, raw_input):
        if not isinstance(raw_input, dict) or set(raw_input) != set(self.input_schema["required"]):
            raise ToolValidationError("questions requires exactly its declared fields")
        if raw_input["operation"] == "list":
            if any(value is not None for key, value in raw_input.items() if key != "operation"):
                raise ToolValidationError("questions.list requires null resolution fields")
            return dict(raw_input)
        if raw_input["operation"] != "resolve" or raw_input["resolution"] not in RESOLUTIONS:
            raise ToolValidationError("invalid question operation or resolution")
        for key in ("question_id", "reason"):
            if not isinstance(raw_input[key], str) or not raw_input[key].strip():
                raise ToolValidationError(f"questions.resolve requires nonempty {key}")
        answer = raw_input["answer"]
        if not isinstance(answer, str):
            raise ToolValidationError("questions.resolve answer must be a string (empty for non-answer resolutions)")
        if raw_input["resolution"].startswith("answered_") and not answer.strip():
            raise ToolValidationError("answered questions require an answer")
        if not raw_input["resolution"].startswith("answered_") and answer:
            raise ToolValidationError("non-answer resolutions require an empty answer")
        evidence = raw_input["evidence_sequences"]
        if (not isinstance(evidence, list) or not evidence or len(evidence) > 32
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in evidence)
                or len(set(evidence)) != len(evidence)):
            raise ToolValidationError("evidence_sequences requires one to thirty-two distinct positive event sequences")
        return dict(raw_input)

    def required_generated_event_types(self, validated_input):
        return {"agent_question_resolved"} if validated_input["operation"] == "resolve" else set()

    def execute(self, validated_input, context):
        state = context.session_state
        if validated_input["operation"] == "list":
            output = {"open_questions": list(state.open_questions)}
            return ToolExecutionResult(self.name, output, stable_json_dumps(output))
        if not any(item["question_id"] == validated_input["question_id"] for item in state.open_questions):
            raise ToolValidationError("Unknown or already resolved question")
        if len(validated_input["answer"]) + len(validated_input["reason"]) > context.config.runtime.max_open_question_chars:
            raise ToolValidationError("Question resolution exceeds configured character budget")
        wanted = set(validated_input["evidence_sequences"])
        history = HistoryStore(context.config.sessions.root, write_projections=False)
        evidence = [event for event in history.iter_history(state.session_id,
                    start_sequence=min(wanted), end_sequence=max(wanted)) if event.sequence in wanted]
        if {event.sequence for event in evidence} != wanted:
            raise ToolValidationError("Question resolution cites missing history evidence")
        if validated_input["resolution"] == "answered_by_user" and not any(
            event.event_type == "message_added" and event.payload.get("message", {}).get("role") == "user"
            for event in evidence
        ):
            raise ToolValidationError("User answer requires a recorded user message")
        if validated_input["resolution"] == "answered_by_research" and not any(
            event.event_type in {"tool_result", "external_source_observed"} for event in evidence
        ):
            raise ToolValidationError("Research answer requires recorded tool/source evidence")
        payload = {key: value for key, value in validated_input.items() if key != "operation"}
        output = {"resolved": True, **payload}
        return ToolExecutionResult(self.name, output, stable_json_dumps(output),
            generated_events=[ToolGeneratedEvent("agent_question_resolved", payload)])


QUESTION_TOOLS = [QuestionsTool()]
