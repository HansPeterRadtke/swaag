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
        "SWAAG__MODEL__FAIL_SAFE_TIMEOUT_SECONDS": "21600",
    }
    config = load_config(env=env)

    assert config.model.profile_name == "mid_context"
    assert config.model.max_semantic_responsibilities_per_call == 3
    assert config.model.structured_output_mode == "server_schema"
    assert config.model.progress_poll_seconds == 2.5
    assert config.model.fail_safe_timeout_seconds == 21600


def test_model_fail_safe_timeout_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fail_safe_timeout_seconds"):
        load_config(env={
            "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
            "SWAAG__MODEL__FAIL_SAFE_TIMEOUT_SECONDS": "0",
        })


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


def test_open_webui_artifact_serving_requires_secure_public_url() -> None:
    from swaag.config import load_config

    env = {
        "SWAAG__COMMUNICATION__OPEN_WEBUI_ARTIFACTS__ENABLED": "true",
        "SWAAG__COMMUNICATION__OPEN_WEBUI_ARTIFACTS__PUBLIC_BASE_URL": "http://example.test/swaag",
        "SWAAG__COMMUNICATION__OPEN_WEBUI_ARTIFACTS__SIGNING_SECRET_ENV": "SWAAG_TEST_ARTIFACT_SECRET",
        "SWAAG__COMMUNICATION__OPEN_WEBUI_ARTIFACTS__TTL_SECONDS": "60",
    }
    with pytest.raises(ValueError, match="must use HTTPS unless it is loopback"):
        load_config(env=env)


def test_open_webui_artifact_serving_allows_loopback_http() -> None:
    from swaag.config import load_config

    config = load_config(
        env={
            "SWAAG__COMMUNICATION__OPEN_WEBUI_ARTIFACTS__ENABLED": "true",
            "SWAAG__COMMUNICATION__OPEN_WEBUI_ARTIFACTS__PUBLIC_BASE_URL": "http://127.0.0.1:8765",
            "SWAAG__COMMUNICATION__OPEN_WEBUI_ARTIFACTS__SIGNING_SECRET_ENV": "SWAAG_TEST_ARTIFACT_SECRET",
            "SWAAG__COMMUNICATION__OPEN_WEBUI_ARTIFACTS__TTL_SECONDS": "60",
        }
    )
    artifact = config.communication.open_webui_artifacts
    assert artifact.enabled is True
    assert artifact.public_base_url == "http://127.0.0.1:8765"
    assert artifact.signing_secret_env == "SWAAG_TEST_ARTIFACT_SECRET"
    assert artifact.ttl_seconds == 60


def test_config_precedence_and_source_ledger_match_documented_order(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config/local.toml").write_text(
        "[model]\ncontext_limit = 1111\n", encoding="utf-8"
    )
    explicit = tmp_path / "explicit.toml"
    explicit.write_text("[model]\ncontext_limit = 2222\n", encoding="utf-8")
    env_file = tmp_path / "env.toml"
    env_file.write_text("[model]\ncontext_limit = 3333\n", encoding="utf-8")
    monkeypatch.chdir(project)

    config = load_config(
        config_paths=[explicit],
        env={
            "SWAAG_CONFIG": str(env_file),
            "SWAAG__MODEL__CONTEXT_LIMIT": "4444",
            "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
        },
    )
    assert config.model.context_limit == 4444
    assert config.source_for("model.context_limit") == "environment:SWAAG__MODEL__CONTEXT_LIMIT"
    assert config.source_for("sessions.root") == "environment:SWAAG__SESSIONS__ROOT"


def test_config_source_ledger_records_file_winner_without_per_key_env(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config/local.toml").write_text(
        "[model]\ncontext_limit = 1111\n", encoding="utf-8"
    )
    env_file = tmp_path / "env.toml"
    env_file.write_text("[model]\ncontext_limit = 3333\n", encoding="utf-8")
    monkeypatch.chdir(project)
    config = load_config(
        env={
            "SWAAG_CONFIG": str(env_file),
            "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
        }
    )
    assert config.model.context_limit == 3333
    assert config.source_for("model.context_limit") == f"SWAAG_CONFIG:{env_file}"
    assert config.source_for("model.base_url") == "packaged_defaults"


def test_agent_data_defaults_to_private_runtime_root(tmp_path) -> None:
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})
    assert config.agent_data.root == Path("/data/var/swaag/agent-data")
    assert config.agent_data.sandbox_backend == "bwrap"
    assert "agent_workspace" in config.tools.enabled


def test_a2a_bearer_secret_is_resolved_from_named_environment_only(tmp_path):
    env = {
        "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
        "SWAAG__A2A__AUTHORIZATION__ENABLED": "true",
        "SWAAG__A2A__AUTHORIZATION__PUBLIC_BASE_URL": "https://agent.example.test",
        "SWAAG__A2A__AUTHORIZATION__BEARER_TOKEN_ENV": "SWAAG_TEST_A2A_SECRET",
        "SWAAG_TEST_A2A_SECRET": "runtime-secret",
    }
    config = load_config(env=env)
    assert config.a2a_authorization.bearer_token_env == "SWAAG_TEST_A2A_SECRET"
    assert config.a2a_authorization.bearer_token == "runtime-secret"
    assert "runtime-secret" not in str(config.raw)
    assert "runtime-secret" not in config.config_fingerprint()


def test_a2a_literal_bearer_secret_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        "[a2a.authorization]\nenabled=true\npublic_base_url='https://agent.example.test'\nbearer_token='literal-secret'\n",
        encoding="utf-8",
    )
    import pytest
    with pytest.raises(ValueError, match="literal secrets are not permitted"):
        load_config(
            config_paths=[path],
            env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")},
        )


def test_mcp_introspection_secret_is_resolved_from_named_environment_only(tmp_path):
    env = {
        "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
        "SWAAG__MCP__AUTHORIZATION__ENABLED": "true",
        "SWAAG__MCP__AUTHORIZATION__RESOURCE_URI": "https://agent.example.test/mcp",
        "SWAAG__MCP__AUTHORIZATION__AUTHORIZATION_SERVERS": '["https://auth.example.test"]',
        "SWAAG__MCP__AUTHORIZATION__INTROSPECTION_URL": "https://auth.example.test/introspect",
        "SWAAG__MCP__AUTHORIZATION__INTROSPECTION_CLIENT_ID": "client-id",
        "SWAAG__MCP__AUTHORIZATION__INTROSPECTION_CLIENT_SECRET_ENV": "SWAAG_TEST_MCP_SECRET",
        "SWAAG_TEST_MCP_SECRET": "runtime-mcp-secret",
    }
    config = load_config(env=env)
    assert config.mcp.authorization.introspection_client_secret_env == "SWAAG_TEST_MCP_SECRET"
    assert config.mcp.authorization.introspection_client_secret == "runtime-mcp-secret"
    assert "runtime-mcp-secret" not in str(config.raw)
    assert "runtime-mcp-secret" not in config.config_fingerprint()


def test_mcp_literal_introspection_secret_is_rejected(tmp_path):
    path = tmp_path / "bad-mcp.toml"
    path.write_text(
        "[mcp.authorization]\nenabled=true\nresource_uri='https://agent.example.test/mcp'\nauthorization_servers=['https://auth.example.test']\nintrospection_url='https://auth.example.test/introspect'\nintrospection_client_id='client-id'\nintrospection_client_secret='literal-secret'\n",
        encoding="utf-8",
    )
    import pytest
    with pytest.raises(ValueError, match="literal secrets are not permitted"):
        load_config(config_paths=[path], env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})


def test_high_impact_parameter_metadata_has_operator_semantics(tmp_path) -> None:
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})
    metadata = config.parameter_metadata()
    expected_phrases = {
        "model.context_limit": "context capacity",
        "context.semantic_reduction_max_calls": "Hard ceiling",
        "sessions.root": "Persistent root",
        "agent_data.root": "private root",
        "tools.allow_side_effect_tools": "external/project side effects",
        "logging.max_bytes": "operations-log file size",
        "a2a.authorization.bearer_token_env": "Environment-variable name",
        "communication.host": "Bind host",
    }
    for dotted, phrase in expected_phrases.items():
        assert phrase.casefold() in metadata[dotted]["description"].casefold()
        assert not metadata[dotted]["description"].startswith("Controls ")
        assert len(metadata[dotted]["consequences"]) >= 40


def test_communication_status_output_cap_default_and_validation(tmp_path) -> None:
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})
    assert config.communication.status_max_output_tokens == 192
    import pytest
    with pytest.raises(ValueError, match="communication.status_max_output_tokens"):
        load_config(
            env={
                "SWAAG__SESSIONS__ROOT": str(tmp_path / "bad-sessions"),
                "SWAAG__COMMUNICATION__STATUS_MAX_OUTPUT_TOKENS": "0",
            }
        )


def test_openwebui_signing_secret_parameter_metadata_uses_real_config_key(tmp_path) -> None:
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})
    metadata = config.parameter_metadata()
    dotted = "communication.open_webui_artifacts.signing_secret_env"
    assert dotted in metadata
    assert "HMAC secret" in metadata[dotted]["description"]
    assert "signing_key_env" not in metadata


@pytest.mark.parametrize(
    "key,value",
    [
        ("tool_call_budget", "8"),
        ("max_total_actions", "8"),
        ("verification_confidence_threshold", "0.5"),
        ("capture_model_io", "true"),
        ("strict_budget", "true"),
        ("background_poll_seconds", "0.1"),
    ],
)
def test_obsolete_runtime_pseudo_controls_fail_explicitly(tmp_path: Path, key: str, value: str) -> None:
    path = tmp_path / "obsolete.toml"
    path.write_text(f"[runtime]\n{key} = {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"runtime\.{key} is obsolete"):
        load_config(config_paths=[path])


def test_obsolete_environment_timeout_fails_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "obsolete-environment.toml"
    path.write_text("[environment]\ncommand_timeout_seconds = 30\n", encoding="utf-8")
    with pytest.raises(ValueError, match="environment.command_timeout_seconds is obsolete"):
        load_config(config_paths=[path])


def test_obsolete_archive_policy_fails_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "obsolete-archive.toml"
    path.write_text("[archive]\nenabled = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="archive configuration is obsolete"):
        load_config(config_paths=[path])
