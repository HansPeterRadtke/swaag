"""Skill catalog.

Each :class:`SkillSpec` describes one workflow the agent can opt into. The
catalog is intentionally small: ``selection_blurb`` is what the LLM-driven
selector reads to decide relevance, and ``full_instructions`` is the
expanded version that gets injected into the prompt only when the skill is
chosen (progressive disclosure).

There are NO trigger terms, vocabulary lists, or substring tags here. The
LLM is the only judge of which skill applies to which goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SkillSpec:
    skill_id: str
    title: str
    selection_blurb: str
    full_instructions: str
    allowed_tools: list[str]
    expected_outputs: list[str] = field(default_factory=list)
    verifier_hints: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


def builtin_skills() -> list[SkillSpec]:
    return [
        SkillSpec(
            skill_id="coding_patch",
            title="Coding Patch",
            selection_blurb="Repair broken code or behavior with a small patch and deterministic verification.",
            full_instructions=(
                "Read the relevant files first, repair broken code or service modules with the smallest coherent "
                "change, then verify behavior with deterministic checks before answering."
            ),
            allowed_tools=["read_text", "edit_text", "write_file", "run_tests", "shell_command", "notes"],
            expected_outputs=["code change", "behavioral verification result"],
            verifier_hints=["tests or checks must pass", "scope must match the request", "repair the targeted module only"],
        ),
        SkillSpec(
            skill_id="file_edit",
            title="Exact File Edit",
            selection_blurb="Make an exact file content change and verify the resulting text precisely.",
            full_instructions="Inspect the current file content, apply only the requested change, and verify the final exact content.",
            allowed_tools=["read_text", "edit_text", "write_file", "notes"],
            expected_outputs=["edited file"],
            verifier_hints=["exact file content must match"],
        ),
        SkillSpec(
            skill_id="browser_research",
            title="Browser Research",
            selection_blurb="Retrieve and ground answers in browser evidence when the task explicitly needs web context.",
            full_instructions="Use browser search or browse only when the task explicitly needs web context, then ground the answer in retrieved content instead of local file evidence.",
            allowed_tools=["browser_search", "browser_browse", "notes", "read_text"],
            expected_outputs=["structured web findings"],
            verifier_hints=["cite retrieved evidence, do not guess"],
        ),
        SkillSpec(
            skill_id="benchmark_analysis",
            title="Benchmark Analysis",
            selection_blurb="Summarize benchmark quality, separate false positives, and rank measured failures.",
            full_instructions="Prefer aggregate evidence, identify failure classes explicitly, and separate false positives from ordinary failures.",
            allowed_tools=["read_text", "notes", "calculator"],
            expected_outputs=["failure analysis", "improvement priorities"],
            verifier_hints=["use measured counts"],
        ),
    ]
