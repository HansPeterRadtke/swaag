from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GuidanceItem:
    layer: str
    source: str
    text: str
    priority: int
    always_on: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GuidanceTraceItem:
    layer: str
    source: str
    selected: bool
    reason: str
    token_cost: int


@dataclass(slots=True)
class GuidanceBundle:
    items: list[GuidanceItem]
    merged_text: str
    trace: list[GuidanceTraceItem]
