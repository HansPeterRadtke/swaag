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
        "SWAAG__MODEL__MAX_SEMANTIC_RESPONSIBILITIES_PER_CALL": "3",
        "SWAAG__MODEL__STRUCTURED_OUTPUT_MODE": "server_schema",
        "SWAAG__MODEL__PROGRESS_POLL_SECONDS": "2.5",
    }
    config = load_config(env=env)

    assert config.model.profile_name == "mid_context"
    assert config.model.max_semantic_responsibilities_per_call == 3
    assert config.model.structured_output_mode == "server_schema"
    assert config.model.progress_poll_seconds == 2.5


def test_default_model_profile_and_mode_match_documented_live_profile(tmp_path: Path) -> None:
    recommendation = get_documented_final_live_benchmark_recommendation()
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})

    assert config.model.profile_name == recommendation.model_profile
    assert config.model.structured_output_mode == recommendation.structured_output_mode
    assert config.model.cache_enabled is True
    assert config.model.cache_mode == "record"
    assert config.model.stop == []
    assert recommendation.timeout_seconds >= 900


def test_model_specific_stop_sequences_are_optional_and_explicit(tmp_path: Path) -> None:
    config = load_config(
        env={
            "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
            "SWAAG__MODEL__STOP": '["MODEL_SPECIFIC_STOP"]',
        }
    )

    assert config.model.stop == ["MODEL_SPECIFIC_STOP"]


def test_visible_editor_backups_are_disabled_by_default() -> None:
    config = load_config()

    assert config.editor.create_backups is False


def test_attachment_defaults_preserve_raw_bytes_without_automatic_extraction() -> None:
    config = load_config()

    assert config.attachments.max_upload_bytes == 100 * 1024 * 1024
    assert config.attachments.preview_chars == 12000
    assert {"list_attachments", "read_attachment"}.issubset(config.tools.enabled)


def test_safe_resumable_search_tools_are_enabled_by_default() -> None:
    config = load_config()

    assert {"search_in_file", "search_repo"}.issubset(config.tools.enabled)


def test_legacy_note_compaction_target_is_accepted_but_no_longer_controls_semantics(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.toml"
    legacy.write_text("[notes]\ncompact_target_chars = 17\n", encoding="utf-8")

    config = load_config(config_paths=[legacy])

    assert not hasattr(config.notes, "compact_target_chars")
    assert config.notes.max_note_chars == 2000


def test_legacy_recent_message_limit_is_accepted_but_no_longer_controls_semantics(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.toml"
    legacy.write_text("[context]\nmax_recent_messages = 2\n", encoding="utf-8")

    config = load_config(config_paths=[legacy])

    assert not hasattr(config.context, "max_recent_messages")


def test_legacy_runtime_context_caps_are_accepted_but_no_longer_drop_candidates(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy-context.toml"
    legacy.write_text(
        "[context]\nworkspace_manifest_max_files = 2\nnote_prompt_token_cap = 3\n",
        encoding="utf-8",
    )

    config = load_config(config_paths=[legacy])

    assert not hasattr(config.context, "workspace_manifest_max_files")
    assert not hasattr(config.context, "note_prompt_token_cap")




def test_semantic_responsibility_limit_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_semantic_responsibilities_per_call"):
        load_config(env={
            "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
            "SWAAG__MODEL__MAX_SEMANTIC_RESPONSIBILITIES_PER_CALL": "0",
        })


def test_legacy_repeated_action_limit_migrates_to_validation_recovery_limit(tmp_path):
    path = tmp_path / "legacy.toml"
    path.write_text("[runtime]\nmax_repeated_action_occurrences = 5\n", encoding="utf-8")
    config = load_config([path], env={})
    assert config.runtime.max_validation_recovery_cycles == 5
    assert "max_repeated_action_occurrences" not in config.raw["runtime"]
    assert config.raw["runtime"]["max_validation_recovery_cycles"] == 5


def test_new_validation_recovery_env_key_wins_over_legacy_alias():
    config = load_config(env={
        "SWAAG__RUNTIME__MAX_REPEATED_ACTION_OCCURRENCES": "5",
        "SWAAG__RUNTIME__MAX_VALIDATION_RECOVERY_CYCLES": "7",
    })
    assert config.runtime.max_validation_recovery_cycles == 7


def test_default_tool_registry_contains_only_system_layer_capabilities() -> None:
    from swaag.tools.registry import ToolRegistry

    registry = ToolRegistry()
    assert all(registry.get(name).layer == "system" for name in registry.registered_names())
    assert {
        "browser_search",
        "browser_browse",
        "extract_attachment",
        "inspect_attachment_capabilities",
    }.isdisjoint(registry.registered_names())
