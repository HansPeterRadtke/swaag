from __future__ import annotations

import re
from dataclasses import asdict
from typing import Iterable

from swaag.config import AgentConfig
from swaag.planner import procedural_memory_from_plan
from swaag.security import provenance_for_event, semantic_item_is_trusted, should_promote_to_semantic, trust_level_for_event
from swaag.types import HistoryEvent, Plan, SemanticMemoryItem, SessionState, TrustLevel
from swaag.utils import new_id, utc_now_iso


_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")



def _tokenize(text: str) -> list[str]:
    return [token for token in _WORD_RE.findall(text) if token]



def _make_item(
    *,
    kind: str,
    content: str,
    source_event_id: str,
    trust_level: TrustLevel,
    tags: list[str],
    entities: list[dict],
    relationships: list[dict],
    facts: list[dict],
    outcome: str,
    confidence: float,
    source_event_type: str,
) -> SemanticMemoryItem:
    return SemanticMemoryItem(
        memory_id=new_id("mem"),
        memory_kind=kind,
        content=content,
        source_event_id=source_event_id,
        trust_level=trust_level,
        tags=tags,
        created_at=utc_now_iso(),
        metadata={
            "entities": entities,
            "relationships": relationships,
            "facts": facts,
            "outcome": outcome,
            "confidence": confidence,
            "provenance": provenance_for_event(source_event_type, {"source_event_id": source_event_id}, source_event_id),
        },
    )



def _calculator_items(event: HistoryEvent, trust_level: TrustLevel) -> list[SemanticMemoryItem]:
    output = event.payload.get("output", {})
    if not isinstance(output, dict):
        return []
    expression = str(output.get("expression", "")).strip()
    result = output.get("result")
    if not expression:
        return []
    expression_entity = {"entity_id": f"expr:{expression}", "name": expression, "entity_type": "expression"}
    result_entity = {"entity_id": f"value:{result}", "name": str(result), "entity_type": "value"}
    relationship = {"relationship_id": new_id("rel"), "source_entity_id": expression_entity["entity_id"], "relation_type": "evaluates_to", "target_entity_id": result_entity["entity_id"]}
    fact = {"fact_id": new_id("fact"), "fact_type": "calculation", "content": f"{expression} = {result}"}
    item = _make_item(
        kind="semantic",
        content=f"Calculator result: {expression} = {result}",
        source_event_id=event.id,
        trust_level=trust_level,
        tags=["calculator", "math"],
        entities=[expression_entity, result_entity],
        relationships=[relationship],
        facts=[fact],
        outcome="calculation_completed",
        confidence=1.0,
        source_event_type=event.event_type,
    )
    return [item]



def _time_items(event: HistoryEvent, trust_level: TrustLevel) -> list[SemanticMemoryItem]:
    output = event.payload.get("output", {})
    if not isinstance(output, dict):
        return []
    utc_time = str(output.get("utc_time", "")).strip()
    if not utc_time:
        return []
    entity = {"entity_id": f"time:{utc_time}", "name": utc_time, "entity_type": "timestamp"}
    fact = {"fact_id": new_id("fact"), "fact_type": "observation", "content": f"UTC time observed: {utc_time}"}
    return [
        _make_item(
            kind="semantic",
            content=f"Observed UTC time: {utc_time}",
            source_event_id=event.id,
            trust_level=trust_level,
            tags=["time"],
            entities=[entity],
            relationships=[],
            facts=[fact],
            outcome="time_observed",
            confidence=1.0,
            source_event_type=event.event_type,
        )
    ]



def _plan_items(event: HistoryEvent) -> list[SemanticMemoryItem]:
    plan_payload = event.payload.get("plan")
    if not isinstance(plan_payload, dict):
        return []
    steps = plan_payload.get("steps", [])
    titles = " -> ".join(str(step.get("title", "")).strip() for step in steps if str(step.get("title", "")).strip())
    entities = [
        {
            "entity_id": f"step:{step.get('step_id', index)}",
            "name": str(step.get("title", "")).strip(),
            "entity_type": "plan_step",
        }
        for index, step in enumerate(steps, start=1)
        if str(step.get("title", "")).strip()
    ]
    relationships = []
    for step in steps:
        for dependency in step.get("depends_on", []):
            relationships.append(
                {
                    "relationship_id": new_id("rel"),
                    "source_entity_id": f"step:{dependency}",
                    "relation_type": "precedes",
                    "target_entity_id": f"step:{step.get('step_id', '')}",
                }
            )
    fact = {"fact_id": new_id("fact"), "fact_type": "strategy", "content": titles}
    return [
        _make_item(
            kind="procedural",
            content=f"Plan strategy: {titles}",
            source_event_id=event.id,
            trust_level="derived",
            tags=["plan", "strategy"],
            entities=entities,
            relationships=relationships,
            facts=[fact],
            outcome="plan_available",
            confidence=1.0,
            source_event_type=event.event_type,
        )
    ]


def _step_outcome_items(event: HistoryEvent) -> list[SemanticMemoryItem]:
    step_id = str(event.payload.get("step_id", "")).strip()
    step_title = str(event.payload.get("step_title", "")).strip()
    outcome = str(event.payload.get("outcome", "")).strip()
    if not step_id or not outcome:
        return []
    step_entity = {"entity_id": f"step:{step_id}", "name": step_title or step_id, "entity_type": "plan_step"}
    outcome_entity = {"entity_id": f"outcome:{step_id}", "name": outcome, "entity_type": "step_outcome"}
    relationship = {
        "relationship_id": new_id("rel"),
        "source_entity_id": step_entity["entity_id"],
        "relation_type": "produced",
        "target_entity_id": outcome_entity["entity_id"],
    }
    fact = {"fact_id": new_id("fact"), "fact_type": "outcome", "content": f"{step_title or step_id} -> {outcome}"}
    return [
        _make_item(
            kind="semantic",
            content=f"Step outcome: {step_title or step_id} produced {outcome}",
            source_event_id=event.id,
            trust_level="derived",
            tags=["step", "outcome"],
            entities=[step_entity, outcome_entity],
            relationships=[relationship],
            facts=[fact],
            outcome="step_completed",
            confidence=0.9,
            source_event_type=event.event_type,
        )
    ]



def extract_from_event(config: AgentConfig, event: HistoryEvent) -> tuple[list[SemanticMemoryItem], str | None]:
    trust_level = event.metadata.get("trust_level") if isinstance(event.metadata, dict) else None
    if trust_level not in {"trusted", "untrusted", "derived"}:
        trust_level = trust_level_for_event(event.event_type, event.payload, event.metadata)
    if event.event_type == "tool_result":
        if not should_promote_to_semantic(config, trust_level=trust_level):
            return [], f"trust level {trust_level} is not promotable"
        tool_name = str(event.payload.get("tool_name", ""))
        if tool_name == "calculator":
            return _calculator_items(event, trust_level), None
        if tool_name == "time_now":
            return _time_items(event, trust_level), None
        return [], f"tool {tool_name} is not promotable"
    if event.event_type in {"plan_created", "plan_updated"}:
        return _plan_items(event), None
    if event.event_type == "step_completed":
        return _step_outcome_items(event), None
    return [], f"event type {event.event_type} does not produce semantic memory"



def retrieve_memory(config: AgentConfig, state: SessionState, query: str, *, limit: int) -> list[SemanticMemoryItem]:
    query_terms = {term.lower() for term in _tokenize(query) if len(term) >= 2}
    scored: list[tuple[float, SemanticMemoryItem]] = []
    for item in state.semantic_memory:
        if not semantic_item_is_trusted(config, item):
            continue
        haystack_terms = {term.lower() for term in _tokenize(item.content + " " + " ".join(item.tags))}
        overlap = query_terms & haystack_terms
        confidence = float(item.metadata.get("confidence", 1.0)) if isinstance(item.metadata, dict) else 1.0
        score = confidence + float(len(overlap))
        if item.memory_kind == "procedural":
            score += 1.5
        if overlap or item.memory_kind == "procedural":
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1].created_at, pair[1].memory_id))
    return [item for _, item in scored[:limit]]



def state_semantic_snapshot(state: SessionState) -> list[dict]:
    return [asdict(item) for item in state.semantic_memory]
