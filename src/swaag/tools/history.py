from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from swaag.history import HistoryStore
from swaag.grammar import history_analysis_contract
from swaag.model import LlamaCppClient
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import stable_json_dumps, to_jsonable


def _active_session_ref(requested: str, context: ToolContext) -> str:
    ref = requested.strip()
    if not ref:
        return context.session_state.session_id
    lowered = ref.casefold()
    current_aliases = {"current", "current_session", "this", "this_session", "active", "session"}
    if lowered in current_aliases:
        return context.session_state.session_id
    if context.session_state.session_name and lowered == context.session_state.session_name.casefold():
        return context.session_state.session_id
    return ref


class HistorySearchTool(Tool):
    repeated_observation_is_redundant = True
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
        session_ref = _active_session_ref(validated_input["session_ref"], context)
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
    repeated_observation_is_redundant = True
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
        session_ref = _active_session_ref(validated_input["session_ref"], context)
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


class HistoryAnalyzeTool(Tool):
    repeated_observation_is_redundant = True
    name = "history_analyze"
    description = "Analyze durable exact history with a model-backed read-only root-cause pass. Returns goal/constraint reconstruction, failure evidence, candidate root causes, exact source event sequences, the wrong strategy, a materially different recommended strategy, and unresolved uncertainties."
    usage_guidance = "Use when repeated failures, user corrections, or ambiguous prior behavior require deeper root-cause analysis. The analyzer is read-only; use its source_sequences with history_window when you need the exact surrounding evidence."
    kind = "pure"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "session_ref": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "max_events": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
        "required": ["query", "session_ref", "max_events"],
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "query": {"type": "string"},
            "goal_constraints": {"type": "array", "items": {"type": "string"}},
            "failure_evidence": {"type": "array", "items": {"type": "string"}},
            "candidate_root_causes": {"type": "array", "items": {"type": "string"}},
            "source_sequences": {"type": "array", "items": {"type": "integer"}},
            "wrong_strategy": {"type": "string"},
            "recommended_strategy": {"type": "string"},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "session_id", "query", "goal_constraints", "failure_evidence",
            "candidate_root_causes", "source_sequences", "wrong_strategy",
            "recommended_strategy", "uncertainties"
        ],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        query = raw_input.get("query")
        session_ref = raw_input.get("session_ref")
        max_events = raw_input.get("max_events")
        if not isinstance(query, str) or not query.strip():
            raise ToolValidationError("history_analyze.query must be a non-empty string")
        if session_ref is not None and not isinstance(session_ref, str):
            raise ToolValidationError("history_analyze.session_ref must be a string or null")
        if max_events is not None and (not isinstance(max_events, int) or isinstance(max_events, bool) or not 1 <= max_events <= 24):
            raise ToolValidationError("history_analyze.max_events must be between 1 and 24 or null")
        return {
            "query": query.strip(),
            "session_ref": (session_ref or "").strip(),
            "max_events": 12 if max_events is None else max_events,
        }

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"history_analyzed"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        store = HistoryStore(context.config.sessions.root, write_projections=False)
        session_ref = _active_session_ref(validated_input["session_ref"], context)
        details = store.query_history_details(
            session_ref,
            validated_input["query"],
            max_results=validated_input["max_events"],
            token_score=context.config.history_search.token_score,
            exact_score=context.config.history_search.exact_score,
            type_bonus=context.config.history_search.type_bonus,
            preview_chars=max(context.config.history_search.preview_chars, 1200),
        )
        candidates = details["matches"]
        if not candidates:
            # Fall back to the latest exact history so an analyzer can still reason about
            # a failure whose wording differs from the user's diagnostic question.
            session_id = details["session_id"]
            events = store.read_history(session_id)
            candidates = [
                {
                    "sequence": event.sequence,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                    "payload": to_jsonable(event.payload),
                    "preview": stable_json_dumps(event.payload)[:1200],
                }
                for event in events[-validated_input["max_events"]:]
            ]
        allowed_sequences = {int(item["sequence"]) for item in candidates}
        prompt = (
            "You are a read-only root-cause analyzer for an agent session. Reconstruct the actual user goal and constraints, "
            "identify concrete failure/dissatisfaction evidence, identify candidate root causes, explain what prior strategy was wrong or incomplete, "
            "recommend a materially different next strategy, and state unresolved uncertainties. Ground every conclusion only in the exact candidate "
            "events below. source_sequences must contain only sequence numbers from these candidates that support your analysis. Do not invent events.\n\n"
            f"Analysis question: {validated_input['query']}\n\n"
            f"Exact candidate events:\n{stable_json_dumps(candidates, indent=2)}"
        )
        completion = LlamaCppClient(context.config).complete(
            prompt,
            max_tokens=max(384, int(context.config.context.reserved_response_tokens)),
            contract=history_analysis_contract(),
            temperature=0.0,
            kind="history_analysis",
            live_mode=False,
        )
        try:
            analysis = json.loads(completion.text)
        except json.JSONDecodeError as exc:
            raise ToolValidationError(f"history_analyze model returned malformed JSON: {exc}") from exc
        if not isinstance(analysis, dict):
            raise ToolValidationError("history_analyze model result must be an object")
        source_sequences = analysis.get("source_sequences")
        if not isinstance(source_sequences, list) or any(not isinstance(item, int) for item in source_sequences):
            raise ToolValidationError("history_analyze.source_sequences must be an integer array")
        invalid = sorted(set(source_sequences) - allowed_sequences)
        if invalid:
            raise ToolValidationError(f"history_analyze referenced non-candidate source sequences: {invalid}")
        output = {"session_id": details["session_id"], "query": validated_input["query"], **analysis}
        self.validate_output(output)
        event = ToolGeneratedEvent(
            "history_analyzed",
            {
                "session_id": output["session_id"],
                "query": output["query"],
                "source_sequences": output["source_sequences"],
                "candidate_root_cause_count": len(output["candidate_root_causes"]),
            },
        )
        return ToolExecutionResult(
            tool_name=self.name,
            output=output,
            display_text=f"history_analyze result: {stable_json_dumps(output, indent=2)}",
            generated_events=[event],
        )


HISTORY_TOOLS = [HistorySearchTool(), HistoryWindowTool(), HistoryAnalyzeTool()]
