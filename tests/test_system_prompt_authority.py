from __future__ import annotations

from swaag.config import load_config
from swaag.prompts import PromptBuilder


def test_standard_system_prompt_keeps_user_request_as_semantic_authority(tmp_path) -> None:
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})
    text = PromptBuilder(config).system_text("standard")
    assert "user's actual request and later corrections are the semantic task authority" in text
    assert "Exact observed tool results outrank guesses" in text
    assert "Choose the semantic next step yourself" in text


def test_lean_system_prompt_keeps_same_authority_charter(tmp_path) -> None:
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})
    text = PromptBuilder(config).system_text("lean")
    assert "user's actual request and later corrections are the semantic task authority" in text
    assert "Exact observed tool results outrank guesses" in text
    assert "Choose the semantic next step yourself" in text
