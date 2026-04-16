from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from swaag.config import AgentConfig
from swaag.retrieval.embeddings import build_backend
from swaag.retrieval.file_retriever import file_candidates
from swaag.retrieval.history_retriever import history_candidates
from swaag.retrieval.memory_retriever import memory_candidates
from swaag.retrieval.query_builder import RetrievalIntent, build_retrieval_intent
from swaag.retrieval.ranker import RankedCandidate, RetrievalCandidate, rank_candidates
from swaag.retrieval.trace import RetrievalTraceItem
from swaag.types import HistoryEvent, Message, SemanticMemoryItem, SessionState


@dataclass(slots=True)
class RetrievalResult:
    intent: RetrievalIntent
    mode: str
    degraded: bool
    history_messages: list[Message] = field(default_factory=list)
    semantic_items: list[SemanticMemoryItem] = field(default_factory=list)
    environment_files: list[tuple[str, str]] = field(default_factory=list)
    trace: list[RetrievalTraceItem] = field(default_factory=list)


class HybridRetriever:
    """Environment-first narrowing + LLM-judged ranking.

    The retriever performs three steps for each candidate source:

    1. Build a candidate list from concrete state (history, memory, files).
    2. Apply a *purely structural* shortlist when the candidate list exceeds
       ``retrieval.max_candidates_per_source``. This shortlist uses ONLY
       deterministic metadata (recency rank, source, structural priority,
       trust level) — no token-overlap, no keyword matching, no synthetic
       semantic score. The intent is to give the LLM ranker a manageable
       slate while never silently dropping items based on a pseudo-semantic
       heuristic.
    3. Hand the shortlist to :func:`rank_candidates` which queries the LLM
       (via the configured backend) for the actual relevance judgement.
    """

    def __init__(self, config: AgentConfig):
        self._config = config
        self._backend = build_backend(
            config.retrieval.backend,
            base_url=config.model.base_url,
            seed=config.model.seed,
            connect_timeout_seconds=config.model.connect_timeout_seconds,
            read_timeout_seconds=config.model.simple_timeout_seconds,
            max_text_chars=config.selection_policy.retrieval_scoring_text_chars,
        )
        if self._backend.degraded and not config.retrieval.allow_degraded_fallback:
            raise RuntimeError("Configured retrieval backend is unavailable and degraded fallback is disabled")

    def retrieve(
        self,
        state: SessionState,
        *,
        counter,
        goal: str,
        current_step_text: str,
        environment_summary: str,
        guidance_summary: str,
        history_events: Iterable[HistoryEvent] | None = None,
        for_planning: bool = False,
        max_history_tokens: int | None = None,
        max_semantic_tokens: int | None = None,
        max_environment_tokens: int | None = None,
        max_history_items: int | None = None,
        max_semantic_items: int | None = None,
        max_environment_items: int | None = None,
    ) -> RetrievalResult:
        intent = build_retrieval_intent(
            state,
            goal=goal,
            purpose="planning" if for_planning else "execution",
            current_step_text=current_step_text,
            environment_summary=environment_summary,
            guidance_summary=guidance_summary,
        )
        history_candidates_list = history_candidates(
            state,
            config=self._config,
            counter=counter,
            history_events=history_events,
            for_planning=for_planning,
        )
        memory_candidates_list = memory_candidates(self._config, state, counter=counter)
        file_candidates_list = file_candidates(state, config=self._config, counter=counter)
        history_ranked = rank_candidates(
            self._config,
            intent,
            self._shortlist_candidates(history_candidates_list),
            self._backend,
        )
        memory_ranked = rank_candidates(
            self._config,
            intent,
            self._shortlist_candidates(memory_candidates_list),
            self._backend,
        )
        file_ranked = rank_candidates(
            self._config,
            intent,
            self._shortlist_candidates(file_candidates_list),
            self._backend,
        )
        result = RetrievalResult(intent=intent, mode=self._backend.mode, degraded=self._backend.degraded)
        history_item_limit = (
            self._config.context_builder.max_history_messages if max_history_items is None else max_history_items
        )
        semantic_item_limit = (
            self._config.context_builder.max_semantic_items if max_semantic_items is None else max_semantic_items
        )
        environment_item_limit = (
            self._config.context_builder.max_environment_files if max_environment_items is None else max_environment_items
        )
        result.history_messages = self._select_history(
            history_ranked,
            min(
                history_item_limit,
                self._dynamic_item_limit(
                    max_history_tokens if max_history_tokens is not None else 1_000_000,
                    approx_tokens=self._config.selection_policy.retrieval_history_item_token_hint,
                ),
            ),
            max_history_tokens if max_history_tokens is not None else 1_000_000,
        )
        result.semantic_items = self._select_semantic(
            memory_ranked,
            min(
                semantic_item_limit,
                self._dynamic_item_limit(
                    max_semantic_tokens if max_semantic_tokens is not None else 1_000_000,
                    approx_tokens=self._config.selection_policy.retrieval_semantic_item_token_hint,
                ),
            ),
            max_semantic_tokens if max_semantic_tokens is not None else 1_000_000,
        )
        result.environment_files = self._select_files(
            file_ranked,
            min(
                environment_item_limit,
                self._dynamic_item_limit(
                    max_environment_tokens if max_environment_tokens is not None else 1_000_000,
                    approx_tokens=self._config.selection_policy.retrieval_environment_item_token_hint,
                ),
            ),
            max_environment_tokens if max_environment_tokens is not None else 1_000_000,
        )
        selected_ids = {
            *(f"history:{id(item)}" for item in result.history_messages),
            *(f"memory:{item.memory_id}" for item in result.semantic_items),
            *(f"file:{item[0]}" for item in result.environment_files),
        }
        for ranked in [*history_ranked, *memory_ranked, *file_ranked]:
            candidate = ranked.candidate
            selected = False
            if candidate.item_type == "history_message" and isinstance(candidate.payload, Message):
                selected = f"history:{id(candidate.payload)}" in selected_ids
            elif candidate.item_type == "history_event" and any(message.content == candidate.text for message in result.history_messages):
                selected = True
            elif candidate.item_type == "semantic_memory":
                selected = f"memory:{candidate.item_id}" in selected_ids
            elif candidate.item_type == "environment_file":
                selected = f"file:{candidate.item_id}" in selected_ids
            result.trace.append(
                RetrievalTraceItem(
                    item_type=candidate.item_type,
                    item_id=candidate.item_id,
                    score=ranked.score,
                    reasons=ranked.reasons,
                    selected=selected,
                    token_cost=candidate.token_cost,
                    source=candidate.source,
                    signals=ranked.signals,
                    degraded=self._backend.degraded,
                )
            )
        return result

    def _shortlist_candidates(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """Reduce a candidate list to ``max_candidates_per_source`` items.

        IMPORTANT: this shortlist uses ONLY deterministic structural metadata
        (recency, structural priority, trust level). It is not allowed to
        compute any pseudo-semantic score, token-overlap, or keyword match.
        Any semantic relevance judgement is the LLM ranker's job; this
        function is purely about giving the ranker a bounded input slate.
        """

        max_candidates = self._config.retrieval.max_candidates_per_source
        if len(candidates) <= max_candidates:
            return list(candidates)

        def structural_key(candidate: RetrievalCandidate) -> tuple[float, float, float, str, str]:
            recency = float(candidate.metadata.get("recency", 0.0))
            structural = float(candidate.metadata.get("structural", 0.0))
            trust = float(candidate.metadata.get("trust", 1.0))
            return (-recency, -structural, -trust, candidate.item_type, candidate.item_id)

        ordered = sorted(candidates, key=structural_key)
        return ordered[:max_candidates]

    def _dynamic_item_limit(self, max_tokens: int, *, approx_tokens: int) -> int:
        if max_tokens <= 0:
            return 1
        return max(1, max_tokens // max(approx_tokens, 1))

    def _select_history(self, ranked: list[RankedCandidate], max_items: int, max_tokens: int) -> list[Message]:
        used = 0
        selected: list[tuple[int, Message]] = []
        for item in ranked:
            if len(selected) >= max_items:
                break
            candidate = item.candidate
            if used + candidate.token_cost > max_tokens:
                continue
            if candidate.item_type == "history_message" and isinstance(candidate.payload, Message):
                selected.append((int(candidate.item_id.split(":")[1]), candidate.payload))
            elif candidate.item_type == "history_event":
                selected.append((10_000 + int(candidate.item_id.split(":")[1]), Message(role="summary", content=candidate.text, created_at="history")))
            else:
                continue
            used += candidate.token_cost
        selected.sort(key=lambda item: item[0])
        return [message for _, message in selected]

    def _select_semantic(self, ranked: list[RankedCandidate], max_items: int, max_tokens: int) -> list[SemanticMemoryItem]:
        used = 0
        selected: list[SemanticMemoryItem] = []
        for item in ranked:
            if len(selected) >= max_items:
                break
            candidate = item.candidate
            if candidate.item_type != "semantic_memory" or not isinstance(candidate.payload, SemanticMemoryItem):
                continue
            if used + candidate.token_cost > max_tokens:
                continue
            selected.append(candidate.payload)
            used += candidate.token_cost
        return selected

    def _select_files(self, ranked: list[RankedCandidate], max_items: int, max_tokens: int) -> list[tuple[str, str]]:
        used = 0
        selected: list[tuple[str, str]] = []
        for item in ranked:
            if len(selected) >= max_items:
                break
            candidate = item.candidate
            if candidate.item_type != "environment_file" or not isinstance(candidate.payload, tuple):
                continue
            if used + candidate.token_cost > max_tokens:
                continue
            selected.append(candidate.payload)
            used += candidate.token_cost
        return selected
