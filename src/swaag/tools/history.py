from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from swaag.history import HistoryStore
from swaag.embedding_index import DerivedEmbeddingIndex, OpenAICompatibleEmbeddingProvider
from swaag.grammar import history_analysis_contract, tool_result_projection_contract
from swaag.tools.base import (
    SemanticCallContextOverflow,
    SemanticCallRequest,
    semantic_sources_cannot_recover_overflow,
    Tool,
    ToolContext,
    ToolValidationError,
)
from swaag.types import PromptComponent, ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import sha256_text, stable_json_dumps, to_jsonable


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


def _search_end_before_current_action(
    store: HistoryStore,
    context: ToolContext,
    *,
    session_ref: str,
    tool_name: str,
) -> int | None:
    target_session = store.resolve_session_ref(session_ref, latest_if_none=False)
    if target_session != context.session_state.session_id:
        return None
    active_events = store.read_history(context.session_state.session_id)
    current_call_sequence = next(
        (
            event.sequence
            for event in reversed(active_events)
            if event.event_type == "tool_called"
            and event.payload.get("tool_name") == tool_name
        ),
        None,
    )
    if current_call_sequence is None:
        return None
    current_action_sequence = next(
        (
            event.sequence
            for event in reversed(active_events)
            if event.sequence < current_call_sequence
            and event.event_type == "agent_action_selected"
        ),
        None,
    )
    boundary = current_action_sequence or current_call_sequence
    return boundary - 1


@dataclass(slots=True)
class _SemanticProjectionBudget:
    remaining_calls: int

    def consume(self) -> None:
        if self.remaining_calls <= 0:
            raise ToolValidationError(
                "history analysis exhausted its bounded semantic projection attempts"
            )
        self.remaining_calls -= 1


def _project_history_text(
    context: ToolContext,
    *,
    question: str,
    source_label: str,
    source_text: str,
    target_tokens: int,
    call_budget: _SemanticProjectionBudget,
    depth: int = 0,
) -> str:
    target = max(32, int(target_tokens))
    request = SemanticCallRequest(
        kind="tool_result_projection",
        system_instruction=(
            "Project exact source material for a later semantic operation. Preserve every fact, "
            "constraint, correction, failure, identifier, and uncertainty relevant to the stated "
            "question. Do not add facts. The raw source remains authoritative and recoverable."
        ),
        components=[
            PromptComponent(
                name="projection_question",
                category="current_user",
                text=f"Later analysis question:\n{question}\n\n",
            ),
            PromptComponent(
                name="projection_target",
                category="instruction",
                text=f"Projection target: at most {target} tokens.\n\n",
            ),
            PromptComponent(
                name="projection_exact_source",
                category="history",
                text=f"{source_label}\n{source_text}",
            ),
        ],
        contract=tool_result_projection_contract(),
        minimum_output_tokens=max(
            64,
            min(
                target + 32,
                int(context.config.context.reserved_response_tokens),
            ),
        ),
        desired_output_tokens=target + 64,
        allow_prompt_instruction_projection=(depth >= 16 or len(source_text) < 2),
    )
    call_budget.consume()
    try:
        payload = context.call_semantic(request)
    except SemanticCallContextOverflow as exc:
        if (
            not request.allow_prompt_instruction_projection
            and semantic_sources_cannot_recover_overflow(
                exc,
                {"projection_exact_source"},
            )
        ):
            try:
                payload = context.call_semantic(
                    replace(
                        request,
                        allow_prompt_instruction_projection=True,
                    )
                )
            except SemanticCallContextOverflow as retry_exc:
                exc = retry_exc
            else:
                projection = str(payload.get("projection", "")).strip()
                if not projection:
                    raise ToolValidationError(
                        "history projection must not be empty"
                    )
                return projection
        if depth >= 16 or len(source_text) < 2:
            raise ToolValidationError(
                "history analysis exact source still overflows after bounded semantic segmentation"
            ) from exc
        midpoint = len(source_text) // 2
        child_target = max(32, (target + 1) // 2)
        left = _project_history_text(
            context,
            question=question,
            source_label=f"{source_label} [exact fragment 1/2]",
            source_text=source_text[:midpoint],
            target_tokens=child_target,
            call_budget=call_budget,
            depth=depth + 1,
        )
        right = _project_history_text(
            context,
            question=question,
            source_label=f"{source_label} [exact fragment 2/2]",
            source_text=source_text[midpoint:],
            target_tokens=child_target,
            call_budget=call_budget,
            depth=depth + 1,
        )
        return _project_history_text(
            context,
            question=question,
            source_label=f"{source_label} [semantic fragment projections]",
            source_text=f"[FRAGMENT 1]\n{left}\n\n[FRAGMENT 2]\n{right}",
            target_tokens=target,
            call_budget=call_budget,
            depth=depth + 1,
        )
    projection = payload.get("projection") if isinstance(payload, dict) else None
    if not isinstance(projection, str) or not projection.strip():
        raise ToolValidationError("semantic source projection must return non-empty text")
    return projection.strip()


class HistorySearchTool(Tool):
    repeated_observation_is_redundant = True
    name = "history_search"
    description = "Search durable exact agent history for earlier user messages, tool calls/results, decisions, and events. Returns ranked previews and sequence numbers; use history_window for exact surrounding events."
    usage_guidance = "Use when current context or summaries may omit an older exact detail. For the active session, pass session_ref=null (or use the exact active session_id/name from environment state); never invent a session label. Search first, then use the exact session_id and sequence returned by history_search with history_window. Optional semantic_matches are only candidate references from a derived embedding index; retrieve the exact event before relying on them."
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
        search_end_sequence = _search_end_before_current_action(
            store,
            context,
            session_ref=session_ref,
            tool_name=self.name,
        )
        result = store.query_history_details(
            session_ref,
            validated_input["query"],
            topic_hint=validated_input["topic_hint"],
            max_results=max_results,
            token_score=cfg.token_score,
            exact_score=cfg.exact_score,
            type_bonus=cfg.type_bonus,
            preview_chars=cfg.preview_chars,
            end_sequence=search_end_sequence,
        )
        matches = [
            {
                "sequence": item["sequence"],
                "hash": item["hash"],
                "event_type": item["event_type"],
                "timestamp": item["timestamp"],
                "preview": item["preview"],
            }
            for item in result["matches"]
        ]
        semantic_matches: list[dict[str, Any]] = []
        semantic_index_error = ""
        embedding_cfg = context.config.embedding_index
        if embedding_cfg.enabled and result["search_backend"] != "archive_fts5":
            try:
                provider = OpenAICompatibleEmbeddingProvider(
                    embedding_cfg.base_url,
                    embedding_cfg.endpoint,
                    embedding_cfg.model,
                    embedding_cfg.timeout_seconds,
                )
                index = DerivedEmbeddingIndex(context.config.sessions.root, provider)
                indexed_through = index.complete_through(result["session_id"])
                highest = indexed_through
                for event in store.iter_history(result["session_id"], start_sequence=max(1, indexed_through + 1)):
                    highest = max(highest, event.sequence)
                    if event.event_type != "agent_status":
                        continue
                    for field in embedding_cfg.fields:
                        value = event.payload.get(field)
                        if isinstance(value, str) and value.strip():
                            index.upsert(result["session_id"], event.sequence, field, value)
                if highest > indexed_through:
                    index.mark_complete_through(result["session_id"], highest)
                semantic_matches = [
                    {"sequence": item.sequence, "field": item.field, "score": item.score}
                    for item in index.search(
                        " ".join(part for part in [validated_input["query"], validated_input["topic_hint"]] if part),
                        session_id=result["session_id"],
                        limit=embedding_cfg.max_results,
                    )
                ]
            except Exception as exc:
                semantic_index_error = f"{type(exc).__name__}: {exc}"
        output = {
            "session_id": result["session_id"],
            "session_name": result["session_name"],
            "search_backend": result["search_backend"],
            "query": result["query"],
            "topic_hint": result["topic_hint"],
            "match_count": len(matches),
            "matches": matches,
            "semantic_matches": semantic_matches,
            "semantic_index_error": semantic_index_error,
            "source_event_references": [
                {
                    "session_id": result["session_id"],
                    "sequence": item["sequence"],
                    "hash": item["hash"],
                    "event_type": item["event_type"],
                    "relationship": "retrieved_history_event",
                }
                for item in matches
            ],
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
    usage_guidance = "Use small windows around relevant sequence numbers. For the active session, pass session_ref=null; after history_search, prefer the exact session_id it returned. Returned payloads are exact durable event data, not summaries."
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
            "source_event_references": [
                {
                    "session_id": event.session_id,
                    "sequence": event.sequence,
                    "hash": event.hash,
                    "event_type": event.event_type,
                    "relationship": "retrieved_history_event",
                }
                for event in events
            ],
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
    description = "Analyze complete durable exact history with a model-backed read-only root-cause pass. Returns goal/constraint reconstruction, failure evidence, candidate root causes, exact source event sequences, the wrong strategy, a materially different recommended strategy, and unresolved uncertainties."
    usage_guidance = "Use when repeated failures, user corrections, or ambiguous prior behavior require deeper root-cause analysis. For the active session, pass session_ref=null (or the exact active session_id/name from environment state), never a guessed label. The analyzer is read-only; use its returned source_sequences and exact session_id with history_window for surrounding evidence."
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
            "source_event_references": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
        "required": [
            "session_id", "query", "goal_constraints", "failure_evidence",
            "candidate_root_causes", "source_sequences", "wrong_strategy",
            "recommended_strategy", "uncertainties", "source_event_references"
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
        if max_events is not None and (
            not isinstance(max_events, int)
            or isinstance(max_events, bool)
            or max_events <= 0
        ):
            raise ToolValidationError(
                "history_analyze.max_events must be a positive legacy hint or null"
            )
        return {
            "query": query.strip(),
            "session_ref": (session_ref or "").strip(),
            # Retained for request compatibility; fixed event-count admission no
            # longer controls semantic history analysis.
            "max_events": max_events,
        }

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"history_analyzed"}

    def execution_timeout_seconds(self, context: ToolContext) -> None:
        # Each model request is bounded and preemptable by AgentRuntime. A second
        # thread timeout cannot stop an in-flight request and would orphan it.
        return None

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        store = HistoryStore(context.config.sessions.root, write_projections=False)
        session_ref = _active_session_ref(validated_input["session_ref"], context)
        search_end_sequence = _search_end_before_current_action(
            store,
            context,
            session_ref=session_ref,
            tool_name=self.name,
        )
        session_id = store.resolve_session_ref(session_ref, latest_if_none=False)
        if session_id is None:
            raise ToolValidationError(f"Unknown session: {session_ref}")
        exact_events = store.read_history(session_id)
        if search_end_sequence is not None and session_id == context.session_state.session_id:
            exact_events = [
                event for event in exact_events if event.sequence <= search_end_sequence
            ]
        allowed_events = {event.sequence: event for event in exact_events}
        exact_history_text = "".join(
            (
                f"Exact durable history event sequence={event.sequence} hash={event.hash}:\n"
                f"{stable_json_dumps(to_jsonable(asdict(event)), indent=2)}\n\n"
            )
            for event in exact_events
        )
        exact_history_hash = sha256_text(exact_history_text)
        raw_history_component = PromptComponent(
            name="history_candidate_events",
            category="history",
            text=exact_history_text,
        )
        components = [
            PromptComponent(
                name="history_analysis_question",
                category="current_user",
                text=f"Analysis question:\n{validated_input['query']}\n\n",
            ),
            raw_history_component,
            PromptComponent(
                name="history_analysis_instruction",
                category="instruction",
                text=(
                    "Return the constrained analysis object. source_sequences must contain only "
                    "exact sequence numbers that directly support the analysis. Do not invent "
                    "events or treat uncertainty as evidence."
                ),
            ),
        ]
        system_instruction = (
            "You are a read-only root-cause analyzer for an agent session. Reconstruct the "
            "actual user goal and constraints, identify concrete failure or dissatisfaction "
            "evidence, identify candidate root causes, explain what prior strategy was wrong "
            "or incomplete, recommend a materially different next strategy, and state "
            "unresolved uncertainties. Ground every conclusion only in the exact durable "
            "history events or explicitly identified semantic projections supplied for this call."
        )
        contract = history_analysis_contract()
        minimum_output_tokens = 384
        projection_records: list[dict[str, Any]] = []
        projection_call_budget = _SemanticProjectionBudget(
            remaining_calls=max(
                64,
                (len(exact_history_text) + 1023) // 1024 * 3,
            )
        )
        max_reduction_rounds = max(
            1,
            int(context.config.context.max_compaction_rounds),
        )
        analysis: dict[str, Any] | None = None
        for reduction_round in range(max_reduction_rounds + 1):
            request = SemanticCallRequest(
                kind="history_analysis",
                system_instruction=system_instruction,
                components=components,
                contract=contract,
                minimum_output_tokens=minimum_output_tokens,
                allow_prompt_instruction_projection=(
                    reduction_round >= max_reduction_rounds
                ),
            )
            try:
                analysis = context.call_semantic(request)
                break
            except SemanticCallContextOverflow as exc:
                if (
                    not request.allow_prompt_instruction_projection
                    and semantic_sources_cannot_recover_overflow(
                        exc,
                        {raw_history_component.name},
                    )
                ):
                    try:
                        analysis = context.call_semantic(
                            replace(
                                request,
                                allow_prompt_instruction_projection=True,
                            )
                        )
                    except SemanticCallContextOverflow as retry_exc:
                        exc = retry_exc
                    else:
                        break
                if (
                    not context.config.context.compact_on_overflow
                    or reduction_round >= max_reduction_rounds
                ):
                    raise
                current_tokens = next(
                    (
                        item.tokens
                        for item in exc.report.breakdown
                        if item.name == raw_history_component.name
                    ),
                    0,
                )
                if current_tokens <= 64:
                    raise
                overflow_tokens = max(
                    1, exc.report.required_tokens - exc.report.context_limit
                )
                target_tokens = max(64, current_tokens - overflow_tokens - 32)
                if target_tokens >= current_tokens:
                    raise ToolValidationError(
                        "history analysis cannot recover the measured overflow from its largest source"
                    )
                projection = _project_history_text(
                    context,
                    question=validated_input["query"],
                    source_label=(
                        "Complete exact durable history source "
                        f"session_id={session_id} source_sha256={exact_history_hash}"
                    ),
                    source_text=raw_history_component.text,
                    target_tokens=target_tokens,
                    call_budget=projection_call_budget,
                )
                projected_component = PromptComponent(
                    name=raw_history_component.name,
                    category=raw_history_component.category,
                    text=(
                        "[SEMANTIC PROJECTION CREATED ONLY AFTER MEASURED OVERFLOW; "
                        "raw durable source remains authoritative]\n"
                        f"session_id={session_id} source_sha256={exact_history_hash} "
                        f"target_tokens={target_tokens}\n{projection}\n\n"
                    ),
                )
                components = [
                    projected_component
                    if item.name == raw_history_component.name
                    else item
                    for item in components
                ]
                projection_records.append({
                    "source_session_id": session_id,
                    "source_event_start_sequence": (
                        exact_events[0].sequence if exact_events else None
                    ),
                    "source_event_end_sequence": (
                        exact_events[-1].sequence if exact_events else None
                    ),
                    "source_event_count": len(exact_events),
                    "source_sha256": exact_history_hash,
                    "target_tokens": target_tokens,
                    "overflow_tokens": overflow_tokens,
                    "projection": projection,
                })
        if analysis is None or not isinstance(analysis, dict):
            raise ToolValidationError("history_analyze model result must be an object")
        source_sequences = analysis.get("source_sequences")
        if not isinstance(source_sequences, list) or any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in source_sequences
        ):
            raise ToolValidationError("history_analyze.source_sequences must be an integer array")
        invalid = sorted(set(source_sequences) - set(allowed_events))
        if invalid:
            raise ToolValidationError(
                f"history_analyze referenced non-candidate source sequences: {invalid}"
            )
        unique_source_sequences = list(dict.fromkeys(source_sequences))
        source_event_references = [
            {
                "session_id": event.session_id,
                "sequence": event.sequence,
                "hash": event.hash,
                "event_type": event.event_type,
                "relationship": "history_analysis_evidence",
            }
            for event in (allowed_events[sequence] for sequence in unique_source_sequences)
        ]
        output = {
            **analysis,
            "session_id": session_id,
            "query": validated_input["query"],
            "source_sequences": unique_source_sequences,
            "source_event_references": source_event_references,
        }
        self.validate_output(output)
        event = ToolGeneratedEvent(
            "history_analyzed",
            {
                "session_id": output["session_id"],
                "query": output["query"],
                "source_sequences": output["source_sequences"],
                "source_event_references": source_event_references,
                "semantic_projections": projection_records,
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
