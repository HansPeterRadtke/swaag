from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalTraceItem:
    item_type: str
    item_id: str
    score: float
    reasons: list[str]
    selected: bool
    token_cost: int
    source: str = ""
    signals: dict[str, float] = field(default_factory=dict)
    degraded: bool = False
