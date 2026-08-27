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
) -> PromptInstruction:
    title, content, scopes, categories = validate_prompt_instruction_fields(
        config,
        title=title,
        content=content,
        scopes=scopes,
        categories=categories,
    )
    now = utc_now_iso()
    return PromptInstruction(
        instruction_id=instruction_id or new_id("instruction"),
        title=title,
        content=content,
        scopes=scopes,
        created_at=now,
        updated_at=now,
        categories=categories,
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
    return [
        item for item in instructions if "all" in item.scopes or kind in item.scopes
    ]


def render_prompt_instructions(instructions: list[PromptInstruction]) -> str:
    if not instructions:
        return ""
    return stable_json_dumps([asdict(item) for item in instructions], indent=2)
