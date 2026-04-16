"""Retrieval candidate ranking.

The previous ranker built a fixed-weight numerical formula
``semantic * w_sem + lexical * w_lex + recency * w_rec + structural * 0.10
+ dependency * 0.20`` and applied a trust multiplier. That formula has been
removed: the LLM is asked, via the configured semantic backend
(:class:`~swaag.retrieval.embeddings.LlmScoringBackend` by default), how
relevant each candidate is to the retrieval intent.

Structural metadata (recency, structural, trust) is still passed through to
the trace so callers can inspect it, but it no longer participates in the
scoring formula. The single source of relevance is the LLM-driven
``backend.score_query`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from swaag.config import AgentConfig
from swaag.retrieval.embeddings import EmbeddingBackend
from swaag.retrieval.query_builder import RetrievalIntent


@dataclass(slots=True)
class RetrievalCandidate:
    item_type: str
    item_id: str
    source: str
    text: str
    token_cost: int
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RankedCandidate:
    candidate: RetrievalCandidate
    score: float
    reasons: list[str]
    signals: dict[str, float]


def rank_candidates(
    config: AgentConfig,  # noqa: ARG001 - kept for signature compatibility
    intent: RetrievalIntent,
    candidates: list[RetrievalCandidate],
    backend: EmbeddingBackend,
) -> list[RankedCandidate]:
    if not candidates:
        return []
    semantic_scores = backend.score_query(intent.query_text, [candidate.text for candidate in candidates])
    ranked: list[RankedCandidate] = []
    for candidate, semantic_score in zip(candidates, semantic_scores, strict=True):
        score = max(0.0, min(1.0, float(semantic_score)))
        reasons = [f"llm_relevance={score:.3f}"]
        if backend.degraded:
            reasons.append("degraded_mode")
        recency = float(candidate.metadata.get("recency", 0.0))
        structural = float(candidate.metadata.get("structural", 0.0))
        trust = float(candidate.metadata.get("trust", 1.0))
        ranked.append(
            RankedCandidate(
                candidate=candidate,
                score=score,
                reasons=reasons,
                signals={
                    "semantic_similarity": score,
                    "recency": recency,
                    "structural": structural,
                    "trust": trust,
                },
            )
        )
    # Tie-breaker is deterministic structural metadata only:
    #   1. higher recency wins (newer items first) — recency is the
    #      candidate's position in its source list, not a heuristic bonus
    #      added to the relevance score itself.
    #   2. then item_type / item_id for stable ordering
    ranked.sort(
        key=lambda item: (
            -item.score,
            -float(item.candidate.metadata.get("recency", 0.0)),
            item.candidate.item_type,
            item.candidate.item_id,
        )
    )
    return ranked
