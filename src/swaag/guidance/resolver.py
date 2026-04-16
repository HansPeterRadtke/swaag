"""Guidance resolution.

Always-on guidance items are merged in deterministically (they are
explicitly marked by the source). Optional guidance items are scored for
relevance by the configured semantic backend (LLM-driven by default) and
included up to the configured budgets.

The previous resolver computed a relevance score from the embedding
backend and then *added* layer-specific bonuses (``+0.35`` for ``task``
layer, ``+0.04`` for ``directory`` layer) and a hand-tuned ``semantic_floor``
based on the best score. Those formula bonuses have been removed: the
LLM-driven score is the only source of relevance.
"""

from __future__ import annotations

from swaag.guidance.types import GuidanceBundle, GuidanceItem, GuidanceTraceItem
from swaag.retrieval.embeddings import build_backend


def resolve_guidance(
    items: list[GuidanceItem],
    *,
    counter,
    max_items: int,
    max_tokens: int,
    query_text: str,
    backend_mode: str,
    base_url: str | None = None,
    seed: int = 11,
    connect_timeout_seconds: int = 10,
    read_timeout_seconds: int = 60,
) -> GuidanceBundle:
    backend = build_backend(
        backend_mode,
        base_url=base_url,
        seed=seed,
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
    )
    selected: list[GuidanceItem] = []
    trace: list[GuidanceTraceItem] = []
    used_tokens = 0
    token_costs = {item.source: counter.count_text(item.text).tokens for item in items}

    always_on = sorted(
        [item for item in items if item.always_on],
        key=lambda item: (item.priority, item.layer, item.source),
    )
    for item in always_on:
        token_cost = token_costs[item.source]
        if used_tokens + token_cost > max_tokens:
            trace.append(GuidanceTraceItem(layer=item.layer, source=item.source, selected=False, reason="always_on_over_budget", token_cost=token_cost))
            continue
        selected.append(item)
        used_tokens += token_cost
        trace.append(GuidanceTraceItem(layer=item.layer, source=item.source, selected=True, reason="always_on", token_cost=token_cost))

    remaining = [item for item in items if item.source not in {guidance.source for guidance in selected} and not item.always_on]
    # Score against the guidance text only, not the filesystem source path
    # (which can leak directory names into lexical scoring and create
    # spurious matches in the degraded fallback).
    semantic_scores = backend.score_query(
        query_text,
        [f"{item.layer} {item.text}" for item in remaining],
    )
    scored = list(zip(semantic_scores, remaining, strict=True))
    scored.sort(key=lambda entry: (-float(entry[0]), entry[1].priority, entry[1].layer, entry[1].source))
    has_query = bool(query_text.strip())
    optional_selected = 0
    for semantic_score, item in scored:
        token_cost = token_costs[item.source]
        if optional_selected >= max_items:
            trace.append(GuidanceTraceItem(layer=item.layer, source=item.source, selected=False, reason="item_limit", token_cost=token_cost))
            continue
        if used_tokens + token_cost > max_tokens:
            trace.append(GuidanceTraceItem(layer=item.layer, source=item.source, selected=False, reason="token_budget", token_cost=token_cost))
            continue
        if has_query and semantic_score <= 0.0:
            trace.append(
                GuidanceTraceItem(
                    layer=item.layer,
                    source=item.source,
                    selected=False,
                    reason="irrelevant",
                    token_cost=token_cost,
                )
            )
            continue
        selected.append(item)
        optional_selected += 1
        used_tokens += token_cost
        reason = f"llm_relevance={float(semantic_score):.3f}"
        if backend.degraded:
            reason += ";degraded_mode"
        trace.append(GuidanceTraceItem(layer=item.layer, source=item.source, selected=True, reason=reason, token_cost=token_cost))

    merged_text = "\n\n".join(f"[{item.layer}] {item.text}" for item in selected)
    return GuidanceBundle(items=selected, merged_text=merged_text, trace=trace)
