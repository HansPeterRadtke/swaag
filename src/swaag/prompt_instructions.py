from __future__ import annotations

from dataclasses import asdict
from typing import get_args

from swaag.config import AgentConfig
from swaag.types import ModelCallKind, PromptInstruction
from swaag.utils import new_id, stable_json_dumps, utc_now_iso


class PromptInstructionError(ValueError):
    pass


PROMPT_INSTRUCTION_SCOPES = ("all", *get_args(ModelCallKind))


def validate_prompt_instruction_fields(
    config: AgentConfig,
    *,
    title: str,
    content: str,
    scopes: list[str],
) -> tuple[str, str, list[str]]:
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
    return title, content, normalized


def make_prompt_instruction(
    config: AgentConfig,
    *,
    title: str,
    content: str,
    scopes: list[str],
    instruction_id: str | None = None,
) -> PromptInstruction:
    title, content, scopes = validate_prompt_instruction_fields(
        config,
        title=title,
        content=content,
        scopes=scopes,
    )
    now = utc_now_iso()
    return PromptInstruction(
        instruction_id=instruction_id or new_id("instruction"),
        title=title,
        content=content,
        scopes=scopes,
        created_at=now,
        updated_at=now,
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
        len(item.title) + len(item.content) + sum(len(scope) for scope in item.scopes)
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
