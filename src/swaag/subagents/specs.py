from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SubagentSpec:
    subagent_type: str
    purpose: str
    allowed_tools: list[str]
    token_budget: int
    role_instruction: str
    metadata: dict[str, str] = field(default_factory=dict)


def default_subagent_specs() -> dict[str, SubagentSpec]:
    return {
        "planner": SubagentSpec(
            subagent_type="planner",
            purpose="review or refine plans",
            allowed_tools=[],
            token_budget=256,
            role_instruction="Review plan structure, dependencies, and verification coverage.",
        ),
        "retriever": SubagentSpec(
            subagent_type="retriever",
            purpose="retrieve focused supporting evidence",
            allowed_tools=[],
            token_budget=192,
            role_instruction="Select only the most relevant evidence for the scoped question.",
        ),
        "coder": SubagentSpec(
            subagent_type="coder",
            purpose="prepare a minimal implementation artifact",
            allowed_tools=["read_text", "edit_text", "write_file", "run_tests"],
            token_budget=384,
            role_instruction="Make the smallest coherent change and preserve project consistency.",
        ),
        "reviewer": SubagentSpec(
            subagent_type="reviewer",
            purpose="independently review candidate results",
            allowed_tools=[],
            token_budget=256,
            role_instruction="Reject unsupported, partial, or weakly evidenced outputs.",
        ),
        "benchmark_analyst": SubagentSpec(
            subagent_type="benchmark_analyst",
            purpose="classify failures and summarize benchmark quality",
            allowed_tools=[],
            token_budget=192,
            role_instruction="Separate false positives from ordinary failures and rank root causes.",
        ),
    }
