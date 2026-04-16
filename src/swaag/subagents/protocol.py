from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from swaag.subagents.specs import SubagentSpec


@dataclass(slots=True)
class SubagentArtifact:
    artifact_type: str
    content: dict[str, Any]


@dataclass(slots=True)
class SubagentReport:
    spec: SubagentSpec
    accepted: bool
    reason: str
    evidence: dict[str, Any]
    recommended_action: str
    artifacts: list[SubagentArtifact] = field(default_factory=list)
