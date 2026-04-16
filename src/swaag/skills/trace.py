from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SkillTraceItem:
    skill_id: str
    selected: bool
    full_loaded: bool
    reason: str
    score: float = 0.0
