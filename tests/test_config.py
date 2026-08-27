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
        "SWAAG__MODEL__STRUCTURED_OUTPUT_MODE": "server_schema",
        "SWAAG__MODEL__PROGRESS_POLL_SECONDS": "2.5",
    }
    config = load_config(env=env)

    assert config.model.profile_name == "mid_context"
    assert config.model.structured_output_mode == "server_schema"
    assert config.model.progress_poll_seconds == 2.5


def test_default_model_profile_and_mode_match_documented_live_profile(tmp_path: Path) -> None:
    recommendation = get_documented_final_live_benchmark_recommendation()
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})

    assert config.model.profile_name == recommendation.model_profile
    assert config.model.structured_output_mode == recommendation.structured_output_mode
    assert config.model.cache_enabled is True
    assert config.model.cache_mode == "record"
    assert recommendation.timeout_seconds >= 900


def test_visible_editor_backups_are_disabled_by_default() -> None:
    config = load_config()

    assert config.editor.create_backups is False


def test_attachment_defaults_preserve_raw_bytes_without_automatic_extraction() -> None:
    config = load_config()

    assert config.attachments.max_upload_bytes == 100 * 1024 * 1024
    assert config.attachments.preview_chars == 12000
    assert config.attachments.all2text_command == "all2text"
    assert {"list_attachments", "read_attachment", "extract_attachment"}.issubset(config.tools.enabled)


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
