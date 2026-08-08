from __future__ import annotations

from dataclasses import asdict
from typing import Any

from swaag.history import HistoryStore
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import stable_json_dumps, to_jsonable


class HistorySearchTool(Tool):
    name = "history_search"
    description = "Search durable exact agent history for earlier user messages, tool calls/results, decisions, and events. Returns ranked previews and sequence numbers; use history_window for exact surrounding events."
    usage_guidance = "Use when current context or summaries may omit an older exact detail. Search first, then retrieve the relevant exact sequence window."
    kind = "pure"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "topic_hint": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "session_ref": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "max_results": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
        "required": ["query", "topic_hint", "session_ref", "max_results"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        query = raw_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolValidationError("history_search.query must be a non-empty string")
        topic_hint = raw_input.get("topic_hint")
        session_ref = raw_input.get("session_ref")
        max_results = raw_input.get("max_results")
        if topic_hint is not None and not isinstance(topic_hint, str):
            raise ToolValidationError("history_search.topic_hint must be a string or null")
        if session_ref is not None and not isinstance(session_ref, str):
            raise ToolValidationError("history_search.session_ref must be a string or null")
        if max_results is not None and (not isinstance(max_results, int) or isinstance(max_results, bool) or max_results <= 0):
            raise ToolValidationError("history_search.max_results must be a positive integer")
        return {
            "query": query.strip(),
            "topic_hint": (topic_hint or "").strip(),
            "session_ref": (session_ref or "").strip(),
            "max_results": max_results,
        }

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"history_retrieved"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        cfg = context.config.history_search
        requested = validated_input["max_results"] or cfg.max_results
        max_results = max(1, min(int(requested), int(cfg.max_results)))
        store = HistoryStore(context.config.sessions.root, write_projections=False)
        session_ref = validated_input["session_ref"] or context.session_state.session_id
        result = store.query_history_details(
            session_ref,
            validated_input["query"],
            topic_hint=validated_input["topic_hint"],
            max_results=max_results,
            token_score=cfg.token_score,
            exact_score=cfg.exact_score,
            type_bonus=cfg.type_bonus,
            preview_chars=cfg.preview_chars,
        )
        matches = [
            {
                "sequence": item["sequence"],
                "event_type": item["event_type"],
                "timestamp": item["timestamp"],
                "preview": item["preview"],
            }
            for item in result["matches"]
        ]
        output = {
            "session_id": result["session_id"],
            "session_name": result["session_name"],
            "query": result["query"],
            "topic_hint": result["topic_hint"],
            "match_count": len(matches),
            "matches": matches,
        }
        event = ToolGeneratedEvent(
            "history_retrieved",
            {
                "session_id": output["session_id"],
                "query": output["query"],
                "match_count": output["match_count"],
                "sequences": [item["sequence"] for item in matches],
            },
        )
        return ToolExecutionResult(
            tool_name=self.name,
            output=output,
            display_text=f"history_search result: {stable_json_dumps(output, indent=2)}",
            generated_events=[event],
        )


class HistoryWindowTool(Tool):
    name = "history_window"
    description = "Read an exact bounded window of durable history events by sequence number. This is the authoritative retrieval path after history_search identifies relevant events."
    usage_guidance = "Use small windows around relevant sequence numbers. Returned payloads are exact durable event data, not summaries."
    kind = "pure"
    input_schema = {
        "type": "object",
        "properties": {
            "start_sequence": {"type": "integer"},
            "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "session_ref": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["start_sequence", "limit", "session_ref"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        start = raw_input.get("start_sequence")
        limit = raw_input.get("limit")
        session_ref = raw_input.get("session_ref")
        limit = 8 if limit is None else limit
        if not isinstance(start, int) or isinstance(start, bool) or start <= 0:
            raise ToolValidationError("history_window.start_sequence must be a positive integer")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
            raise ToolValidationError("history_window.limit must be between 1 and 20")
        if session_ref is not None and not isinstance(session_ref, str):
            raise ToolValidationError("history_window.session_ref must be a string or null")
        return {"start_sequence": start, "limit": limit, "session_ref": (session_ref or "").strip()}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"history_window_read"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        store = HistoryStore(context.config.sessions.root, write_projections=False)
        session_ref = validated_input["session_ref"] or context.session_state.session_id
        session_id = store.resolve_session_ref(session_ref, latest_if_none=False)
        if session_id is None:
            raise ToolValidationError(f"Unknown session: {session_ref}")
        events = store.read_history_window(
            session_id,
            start_sequence=validated_input["start_sequence"],
            limit=validated_input["limit"],
        )
        output = {
            "session_id": session_id,
            "start_sequence": validated_input["start_sequence"],
            "event_count": len(events),
            "events": [to_jsonable(asdict(event)) for event in events],
        }
        event = ToolGeneratedEvent(
            "history_window_read",
            {
                "session_id": session_id,
                "start_sequence": validated_input["start_sequence"],
                "event_count": len(events),
                "sequences": [item.sequence for item in events],
            },
        )
        return ToolExecutionResult(
            tool_name=self.name,
            output=output,
            display_text=f"history_window result: {stable_json_dumps(output, indent=2)}",
            generated_events=[event],
        )


HISTORY_TOOLS = [HistorySearchTool(), HistoryWindowTool()]
