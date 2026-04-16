from __future__ import annotations

from pathlib import Path

import pytest

from swaag.config import load_config
from swaag.live_runtime_profiles import get_documented_final_live_benchmark_recommendation


def test_load_config_applies_env_override(tmp_path: Path) -> None:
    env = {
        "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
        "SWAAG__TOOLS__ENABLED": '["echo","calculator"]',
        "SWAAG__MODEL__CONTEXT_LIMIT": "4096",
        "SWAAG__TOOLS__ALLOW_SIDE_EFFECT_TOOLS": "true",
    }
    config = load_config(env=env)

    assert config.model.context_limit == 4096
    assert config.sessions.root == tmp_path / "sessions"
    assert config.tools.enabled == ["echo", "calculator"]
    assert config.tools.allow_side_effect_tools is True
    assert len(config.config_fingerprint()) == 64


def test_invalid_reader_overlap_is_rejected(tmp_path: Path) -> None:
    env = {
        "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
        "SWAAG__READER__DEFAULT_CHUNK_CHARS": "10",
        "SWAAG__READER__DEFAULT_OVERLAP_CHARS": "10",
    }
    with pytest.raises(ValueError):
        load_config(env=env)


def test_model_profile_and_structured_output_env_overrides_are_loaded(tmp_path: Path) -> None:
    env = {
        "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
        "SWAAG__MODEL__PROFILE_NAME": "mid_context",
        "SWAAG__MODEL__STRUCTURED_OUTPUT_MODE": "post_validate",
        "SWAAG__MODEL__PROGRESS_POLL_SECONDS": "2.5",
    }
    config = load_config(env=env)

    assert config.model.profile_name == "mid_context"
    assert config.model.structured_output_mode == "post_validate"
    assert config.model.progress_poll_seconds == 2.5


def test_default_model_profile_and_mode_match_documented_live_profile(tmp_path: Path) -> None:
    recommendation = get_documented_final_live_benchmark_recommendation()
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})

    assert config.model.profile_name == recommendation.model_profile
    assert config.model.structured_output_mode == recommendation.structured_output_mode


def test_selection_policy_retrieval_weights_are_loaded_from_defaults() -> None:
    """All retrieval structural weights must load from defaults.toml without code fallbacks."""
    config = load_config()

    pol = config.selection_policy
    # Structural weights
    assert pol.retrieval_structural_tool_message == 0.75
    assert pol.retrieval_structural_user_message == 0.35
    assert pol.retrieval_structural_failed_event == 0.20
    assert pol.retrieval_structural_summary_event == 0.12
    assert pol.retrieval_structural_modified_file == 0.50
    assert pol.retrieval_structural_procedural_memory == 0.50
    assert pol.retrieval_trust_untrusted_memory == 0.25
    # Preview lengths
    assert pol.retrieval_history_ranking_chars == 280
    assert pol.retrieval_event_ranking_chars == 240
    assert pol.retrieval_file_ranking_chars == 400
    # History detail scoring
    assert pol.history_detail_token_score == 2
    assert pol.history_detail_exact_score == 4
    assert pol.history_detail_type_bonus == 1
    assert pol.history_detail_preview_chars == 320


def test_selection_policy_retrieval_weights_can_be_overridden_via_env() -> None:
    """Retrieval weights must be overridable via environment variables."""
    env = {
        "SWAAG__SELECTION_POLICY__RETRIEVAL_STRUCTURAL_TOOL_MESSAGE": "0.90",
        "SWAAG__SELECTION_POLICY__HISTORY_DETAIL_TOKEN_SCORE": "3",
        "SWAAG__SELECTION_POLICY__RETRIEVAL_HISTORY_RANKING_CHARS": "400",
    }
    config = load_config(env=env)

    assert config.selection_policy.retrieval_structural_tool_message == 0.90
    assert config.selection_policy.history_detail_token_score == 3
    assert config.selection_policy.retrieval_history_ranking_chars == 400


def test_budget_policy_safe_input_floor_is_loaded_from_defaults() -> None:
    """safe_input_floor_tokens must load from defaults.toml."""
    config = load_config()
    assert config.budget_policy.safe_input_floor_tokens == 128


def test_budget_policy_safe_input_floor_can_be_overridden_via_env() -> None:
    """safe_input_floor_tokens must be overridable via environment variable."""
    env = {"SWAAG__BUDGET_POLICY__SAFE_INPUT_FLOOR_TOKENS": "64"}
    config = load_config(env=env)
    assert config.budget_policy.safe_input_floor_tokens == 64


def test_selection_policy_retrieval_scoring_text_chars_is_loaded_from_defaults() -> None:
    """retrieval_scoring_text_chars must load from defaults.toml."""
    config = load_config()
    assert config.selection_policy.retrieval_scoring_text_chars == 400


def test_selection_policy_retrieval_scoring_text_chars_can_be_overridden_via_env() -> None:
    """retrieval_scoring_text_chars must be overridable via environment variable."""
    env = {"SWAAG__SELECTION_POLICY__RETRIEVAL_SCORING_TEXT_CHARS": "200"}
    config = load_config(env=env)
    assert config.selection_policy.retrieval_scoring_text_chars == 200


def test_environment_aubro_overrides_are_loaded(tmp_path: Path) -> None:
    env = {
        "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
        "SWAAG__ENVIRONMENT__AUBRO_ENTRYPOINT": "/usr/bin/python3 -m aubro.cli",
        "SWAAG__ENVIRONMENT__AUBRO_SRC": str(tmp_path / "aubro_src"),
        "SWAAG__ENVIRONMENT__AUBRO_TIMEOUT_SECONDS": "90",
        "SWAAG__ENVIRONMENT__AUBRO_MAX_TEXT_CHARS": "1234",
        "SWAAG__ENVIRONMENT__AUBRO_MAX_RESULTS": "7",
        "SWAAG__ENVIRONMENT__AUBRO_MAX_LINKS": "9",
    }
    config = load_config(env=env)

    assert config.environment.aubro_entrypoint == "/usr/bin/python3 -m aubro.cli"
    assert config.environment.aubro_src == str(tmp_path / "aubro_src")
    assert config.environment.aubro_timeout_seconds == 90
    assert config.environment.aubro_max_text_chars == 1234
    assert config.environment.aubro_max_results == 7
    assert config.environment.aubro_max_links == 9
