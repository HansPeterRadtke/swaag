from __future__ import annotations

from swaag.skills.catalog import SkillSpec


def render_skill_metadata(skills: list[SkillSpec]) -> str:
    if not skills:
        return ""
    return "\n".join(
        f"- {skill.skill_id}: {skill.selection_blurb} tools={','.join(skill.allowed_tools)} outputs={','.join(skill.expected_outputs)}"
        for skill in skills
    )


def render_skill_instructions(skills: list[SkillSpec]) -> str:
    if not skills:
        return ""
    return "\n\n".join(
        f"[{skill.title}]\nInstructions: {skill.full_instructions}\nExpected outputs: {', '.join(skill.expected_outputs)}\nVerifier hints: {', '.join(skill.verifier_hints)}"
        for skill in skills
    )
