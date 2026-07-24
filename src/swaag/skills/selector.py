"""Skill selection.

The relevance of each skill to the current goal is scored by the LLM via
the configured semantic backend (LlmScoringBackend by default). Skills are
prompt text plus descriptive metadata only: they can add instructions,
examples, expected outputs, and verifier hints, but they never hide,
remove, or preselect tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from swaag.retrieval.embeddings import EmbeddingBackend, build_backend
from swaag.skills.catalog import SkillSpec, builtin_skills
from swaag.skills.trace import SkillTraceItem


@dataclass(slots=True)
class SkillSelection:
    selected_skills: list[SkillSpec]
    metadata_skills: list[SkillSpec]
    trace: list[SkillTraceItem] = field(default_factory=list)


def _skill_query_text(
    *,
    goal: str,
    current_step_text: str,
    guidance_text: str,
    retrieval_text: str,
    role_name: str,
    failure_summary: str,
) -> str:
    parts = [goal, current_step_text, guidance_text, retrieval_text, role_name, failure_summary]
    return "\n".join(part for part in parts if part.strip())


def _skill_candidate_text(skill: SkillSpec) -> str:
    parts = [
        skill.title,
        skill.selection_blurb,
        " ".join(skill.expected_outputs),
        " ".join(skill.verifier_hints),
        " ".join(skill.dependencies),
    ]
    return "\n".join(part for part in parts if part.strip())


def select_skills(
    *,
    goal: str,
    current_step_text: str,
    guidance_text: str,
    retrieval_text: str,
    role_name: str,
    failure_summary: str,
    enabled_tool_names: list[str],
    max_full_instructions: int,
    max_metadata_items: int,
    backend_mode: str,
    skills: list[SkillSpec] | None = None,
    backend: EmbeddingBackend | None = None,
    base_url: str | None = None,
    seed: int = 11,
    connect_timeout_seconds: int = 10,
    read_timeout_seconds: int = 60,
    max_text_chars: int | None = None,
    model_client: Any | None = None,
) -> SkillSelection:
    skills = builtin_skills() if skills is None else list(skills)
    query_text = _skill_query_text(
        goal=goal,
        current_step_text=current_step_text,
        guidance_text=guidance_text,
        retrieval_text=retrieval_text,
        role_name=role_name,
        failure_summary=failure_summary,
    )
    if backend is None:
        backend = build_backend(
            backend_mode,
            base_url=base_url,
            seed=seed,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
            max_text_chars=max_text_chars,
            model_client=model_client,
        )
    del enabled_tool_names

    if not skills:
        return SkillSelection(
            selected_skills=[],
            metadata_skills=[],
            trace=[],
        )

    semantic_scores = backend.score_query(
        query_text,
        [_skill_candidate_text(skill) for skill in skills],
    )

    scored_items: list[tuple[float, SkillSpec]] = list(zip(semantic_scores, skills, strict=True))
    ranked = sorted(scored_items, key=lambda item: (-item[0], item[1].skill_id))

    positive_ranked = [(score, skill) for score, skill in ranked if score > 0.0 or not query_text.strip()]
    selected_skills = [skill for score, skill in positive_ranked[:max_full_instructions]]
    metadata_skills = [skill for score, skill in positive_ranked[:max_metadata_items]]

    selected_ids = {skill.skill_id for skill in selected_skills}
    metadata_ids = {skill.skill_id for skill in metadata_skills}
    traces: list[SkillTraceItem] = []
    for score, skill in ranked:
        if skill.skill_id not in metadata_ids and query_text.strip():
            traces.append(
                SkillTraceItem(
                    skill_id=skill.skill_id,
                    selected=False,
                    full_loaded=False,
                    reason="irrelevant" + (";degraded_mode" if backend.degraded else ""),
                    score=score,
                )
            )
            continue
        if skill.skill_id in selected_ids or skill.skill_id in metadata_ids:
            traces.append(
                SkillTraceItem(
                    skill_id=skill.skill_id,
                    selected=skill.skill_id in selected_ids,
                    full_loaded=skill.skill_id in selected_ids,
                    reason=("llm_ranked" if not backend.degraded else "degraded_ranked"),
                    score=score,
                )
            )
        else:
            traces.append(
                SkillTraceItem(
                    skill_id=skill.skill_id,
                    selected=False,
                    full_loaded=False,
                    reason="not_selected_under_budget",
                    score=score,
                )
            )
    return SkillSelection(
        selected_skills=selected_skills,
        metadata_skills=metadata_skills,
        trace=traces,
    )
