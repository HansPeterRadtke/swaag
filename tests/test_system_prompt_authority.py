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


def _assert_updated_infra_interaction_contract(text: str) -> None:
    assert "A question asks for an answer, not an unsolicited implementation" in text
    assert "statement is not automatically a command" in text
    assert "current explicit instruction overrides these defaults" in text
    assert "Do not silently fill a source gap from pretrained memory" in text
    assert "reporting should save the user's time" in text
    assert "only as much numeric precision as the decision requires" in text


def test_standard_system_prompt_contains_updated_infra_interaction_contract(tmp_path) -> None:
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})
    _assert_updated_infra_interaction_contract(PromptBuilder(config).system_text("standard"))


def test_lean_system_prompt_contains_updated_infra_interaction_contract(tmp_path) -> None:
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})
    _assert_updated_infra_interaction_contract(PromptBuilder(config).system_text("lean"))
