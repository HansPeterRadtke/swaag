from __future__ import annotations

from swaag.config import AgentConfig
from swaag.retrieval.ranker import RetrievalCandidate
from swaag.types import SessionState


def memory_candidates(config: AgentConfig, state: SessionState, *, counter) -> list[RetrievalCandidate]:
    candidates: list[RetrievalCandidate] = []
    for item in state.semantic_memory:
        if config.security.block_untrusted_semantic and item.trust_level == "untrusted":
            continue
        text = item.content
        trust = 1.0 if item.trust_level != "untrusted" else config.selection_policy.retrieval_trust_untrusted_memory
        structural = config.selection_policy.retrieval_structural_procedural_memory if item.memory_kind == "procedural" else 0.0
        confidence = float(item.metadata.get("confidence", 1.0)) if isinstance(item.metadata, dict) else 1.0
        candidates.append(
            RetrievalCandidate(
                item_type="semantic_memory",
                item_id=item.memory_id,
                source="semantic_memory",
                text=text,
                token_cost=counter.count_text(text).tokens,
                payload=item,
                metadata={"trust": trust * confidence, "structural": structural},
            )
        )
    if state.project_state.files_seen or state.project_state.relationships:
        text = "\n".join(
            [
                *(f"file:{item}" for item in state.project_state.files_seen),
                *(f"modified:{item}" for item in state.project_state.files_modified),
                *(f"relation:{item}" for item in state.project_state.relationships),
            ]
        )
        candidates.append(
            RetrievalCandidate(
                item_type="project_state",
                item_id="project_state",
                source="project_state",
                text=text,
                token_cost=counter.count_text(text).tokens,
                payload=state.project_state,
                metadata={"structural": 1.0},
            )
        )
    return candidates
