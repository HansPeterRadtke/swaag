from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SubagentSpec:
    subagent_type: str
    purpose: str
    token_budget: int
    role_instruction: str
    capabilities: tuple[str, ...]
    input_schema: dict[str, Any]
    usage_guidance: str
    metadata: dict[str, str] = field(default_factory=dict)


def _subagent_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "purpose": {"type": "string"},
            "focus": {"type": "string"},
            "context_summary": {"type": "string"},
        },
        "required": ["purpose", "focus", "context_summary"],
        "additionalProperties": False,
    }


def default_subagent_specs() -> dict[str, SubagentSpec]:
    common_schema = _subagent_input_schema()
    return {
        "planner": SubagentSpec(
            subagent_type="planner",
            purpose="review or refine plans",
            token_budget=256,
            role_instruction="Review plan structure, dependencies, and verification coverage.",
            capabilities=("plan_review", "plan_repair", "replan_guidance"),
            input_schema=common_schema,
            usage_guidance="Use when an isolated pass over plan structure, dependencies, and verification evidence is useful.",
        ),
        "retriever": SubagentSpec(
            subagent_type="retriever",
            purpose="retrieve focused supporting evidence",
            token_budget=192,
            role_instruction="Select only the most relevant evidence for the scoped question.",
            capabilities=("context_retrieval_focus", "evidence_selection"),
            input_schema=common_schema,
            usage_guidance="Use when the current prompt would benefit from a narrower evidence bundle built from available context.",
        ),
        "coder": SubagentSpec(
            subagent_type="coder",
            purpose="prepare a minimal implementation artifact",
            token_budget=384,
            role_instruction="Make the smallest coherent change and preserve project consistency.",
            capabilities=("implementation_review", "patch_guidance", "quality_check"),
            input_schema=common_schema,
            usage_guidance="Use when a scoped implementation or code-quality perspective would materially improve the next decision.",
        ),
        "reviewer": SubagentSpec(
            subagent_type="reviewer",
            purpose="independently review candidate results",
            token_budget=256,
            role_instruction="Reject unsupported, partial, or weakly evidenced outputs.",
            capabilities=("result_review", "plan_review", "verification_review"),
            input_schema=common_schema,
            usage_guidance="Use when independent review of a candidate result, answer, or evidence trail is useful.",
        ),
        "benchmark_analyst": SubagentSpec(
            subagent_type="benchmark_analyst",
            purpose="classify failures and summarize benchmark quality",
            token_budget=192,
            role_instruction="Separate false positives from ordinary failures and rank root causes.",
            capabilities=("failure_analysis", "benchmark_trace_review", "quality_check"),
            input_schema=common_schema,
            usage_guidance="Use when trace or benchmark evidence needs isolated analysis without changing the runtime plan.",
        ),
    }
