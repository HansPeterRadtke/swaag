from __future__ import annotations

from dataclasses import asdict
from typing import get_args

from swaag.config import AgentConfig
from swaag.types import ModelCallKind, PromptInstruction
from swaag.utils import new_id, stable_json_dumps, utc_now_iso


class PromptInstructionError(ValueError):
    pass


PROMPT_INSTRUCTION_SCOPES = (
    "all",
    *(
        kind
        for kind in get_args(ModelCallKind)
        if kind != "prompt_instruction_selection"
    ),
)
MAX_PROMPT_INSTRUCTION_CATEGORIES = 16
MAX_PROMPT_INSTRUCTION_CATEGORY_CHARS = 120

PROMPT_INSTRUCTION_AUTHORITY_RANK = {
    "learned_model": 0,
    "durable_user_policy": 20,
    "project_policy": 40,
    "voice_recording": 60,
    "explicit_user_correction": 80,
}
TRUSTED_PROMPT_INSTRUCTION_AUTHORITIES = frozenset(
    {"durable_user_policy", "project_policy", "voice_recording", "explicit_user_correction"}
)


def prompt_instruction_authority_rank(item: PromptInstruction) -> int:
    return int(PROMPT_INSTRUCTION_AUTHORITY_RANK.get(item.authority, -1))


def is_trusted_prompt_instruction(item: PromptInstruction) -> bool:
    return item.authority in TRUSTED_PROMPT_INSTRUCTION_AUTHORITIES


def sort_prompt_instructions_by_authority(
    instructions: list[PromptInstruction],
) -> list[PromptInstruction]:
    # Highest authority/specificity/newest first. The prompt header defines conflict precedence.
    return sorted(
        instructions,
        key=lambda item: (
            prompt_instruction_authority_rank(item),
            int(item.specificity),
            str(item.updated_at),
            item.instruction_id,
        ),
        reverse=True,
    )


def validate_prompt_instruction_fields(
    config: AgentConfig,
    *,
    title: str,
    content: str,
    scopes: list[str],
    categories: list[str] | None = None,
) -> tuple[str, str, list[str], list[str]]:
    title = title.strip()
    content = content.strip()
    if not title:
        raise PromptInstructionError("prompt instruction title must not be empty")
    if not content:
        raise PromptInstructionError("prompt instruction content must not be empty")
    if len(title) > 200:
        raise PromptInstructionError(
            "prompt instruction title exceeds the 200-character storage limit"
        )
    if len(content) > config.prompt_instructions.max_instruction_chars:
        raise PromptInstructionError(
            "prompt instruction content exceeds the configured "
            f"max_instruction_chars storage limit: {config.prompt_instructions.max_instruction_chars}"
        )
    normalized: list[str] = []
    for raw_scope in scopes:
        scope = str(raw_scope).strip()
        if scope not in PROMPT_INSTRUCTION_SCOPES:
            raise PromptInstructionError(
                "prompt instruction scope must be one of: "
                + ", ".join(PROMPT_INSTRUCTION_SCOPES)
            )
        if scope not in normalized:
            normalized.append(scope)
    if not normalized:
        raise PromptInstructionError(
            "prompt instruction requires at least one model-call scope"
        )
    normalized_categories: list[str] = []
    for raw_category in categories or []:
        if not isinstance(raw_category, str):
            raise PromptInstructionError(
                "prompt instruction categories must contain only strings"
            )
        category = raw_category.strip()
        if not category:
            raise PromptInstructionError(
                "prompt instruction categories must not contain empty values"
            )
        if len(category) > MAX_PROMPT_INSTRUCTION_CATEGORY_CHARS:
            raise PromptInstructionError(
                "prompt instruction category exceeds the "
                f"{MAX_PROMPT_INSTRUCTION_CATEGORY_CHARS}-character storage limit"
            )
        if category not in normalized_categories:
            normalized_categories.append(category)
    if len(normalized_categories) > MAX_PROMPT_INSTRUCTION_CATEGORIES:
        raise PromptInstructionError(
            "prompt instruction category count exceeds the "
            f"{MAX_PROMPT_INSTRUCTION_CATEGORIES}-category storage limit"
        )
    return title, content, normalized, normalized_categories


def make_prompt_instruction(
    config: AgentConfig,
    *,
    title: str,
    content: str,
    scopes: list[str],
    categories: list[str] | None = None,
    instruction_id: str | None = None,
    authority: str = "learned_model",
    source_kind: str = "model_learned",
    source_ref: str = "",
    specificity: int = 0,
) -> PromptInstruction:
    title, content, scopes, categories = validate_prompt_instruction_fields(
        config,
        title=title,
        content=content,
        scopes=scopes,
        categories=categories,
    )
    if authority not in PROMPT_INSTRUCTION_AUTHORITY_RANK:
        raise PromptInstructionError(f"unknown prompt instruction authority: {authority}")
    specificity = int(specificity)
    if specificity < 0 or specificity > 100:
        raise PromptInstructionError("prompt instruction specificity must be between 0 and 100")
    source_kind = str(source_kind).strip()
    source_ref = str(source_ref).strip()
    if authority in TRUSTED_PROMPT_INSTRUCTION_AUTHORITIES and (not source_kind or not source_ref):
        raise PromptInstructionError("trusted prompt instructions require source_kind and source_ref provenance")
    now = utc_now_iso()
    return PromptInstruction(
        instruction_id=instruction_id or new_id("instruction"),
        title=title,
        content=content,
        scopes=scopes,
        created_at=now,
        updated_at=now,
        categories=categories,
        authority=authority,
        source_kind=source_kind,
        source_ref=source_ref,
        specificity=specificity,
    )


def enforce_prompt_instruction_limits(
    config: AgentConfig,
    instructions: list[PromptInstruction],
) -> list[PromptInstruction]:
    result = list(instructions)
    if len(result) > config.prompt_instructions.max_instructions:
        raise PromptInstructionError(
            "prompt instruction count exceeds the configured max_instructions "
            f"storage limit: {config.prompt_instructions.max_instructions}"
        )
    total = sum(
        len(item.title)
        + len(item.content)
        + sum(len(scope) for scope in item.scopes)
        + sum(len(category) for category in item.categories)
        for item in result
    )
    if total > config.prompt_instructions.max_total_chars:
        raise PromptInstructionError(
            "prompt instructions exceed the configured max_total_chars storage limit: "
            f"{total}>{config.prompt_instructions.max_total_chars}"
        )
    return result


def prompt_instructions_for_kind(
    instructions: list[PromptInstruction],
    kind: ModelCallKind,
) -> list[PromptInstruction]:
    return sort_prompt_instructions_by_authority([
        item for item in instructions if "all" in item.scopes or kind in item.scopes
    ])


def render_prompt_instructions(instructions: list[PromptInstruction]) -> str:
    if not instructions:
        return ""
    return stable_json_dumps([asdict(item) for item in instructions], indent=2)
