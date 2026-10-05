from __future__ import annotations

import json
import time
from pathlib import Path

from swaag.operations_log import OperationsLog, _redact


def _read_json_lines(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_operations_log_records_redacted_startup_sources_and_shutdown(make_config, tmp_path) -> None:
    path = tmp_path / "operations.jsonl"
    config = make_config(
        logging__file_path=path,
        logging__max_bytes=200_000,
        logging__backup_count=2,
    )
    config.a2a_authorization.bearer_token = "runtime-secret-that-must-not-log"
    runtime = OperationsLog(config, component="test")
    runtime.event("custom", token="direct-token", ordinary="ok")
    runtime.shutdown(reason="test_done")

    rows = _read_json_lines([path])
    assert [row["event"] for row in rows] == ["startup", "custom", "shutdown"]
    startup = rows[0]["attributes"]
    assert startup["config_fingerprint"] == config.config_fingerprint()
    assert startup["config_sources"]["model.base_url"].startswith("environment:")
    assert "model.context_limit" in startup["parameter_metadata"]
    assert startup["parameter_metadata"]["model.context_limit"]["unit"] == "tokens"
    custom = rows[1]["attributes"]
    assert custom["token"].startswith("[REDACTED]:sha256=")
    assert custom["ordinary"] == "ok"
    serialized = path.read_text(encoding="utf-8")
    assert "runtime-secret-that-must-not-log" not in serialized
    assert "direct-token" not in serialized
    assert rows[-1]["attributes"]["reason"] == "test_done"


def test_operations_log_rotates_and_retains_bounded_files(make_config, tmp_path) -> None:
    path = tmp_path / "operations.jsonl"
    config = make_config(
        logging__file_path=path,
        logging__max_bytes=40_000,
        logging__backup_count=2,
        logging__queue_capacity=256,
    )
    runtime = OperationsLog(config, component="rotation-test")
    for index in range(120):
        runtime.event("bulk", index=index, payload="x" * 1200)
    runtime.shutdown(reason="rotation_done")

    existing = [candidate for candidate in [path, Path(str(path) + ".1"), Path(str(path) + ".2")] if candidate.exists()]
    assert len(existing) >= 2
    assert not Path(str(path) + ".3").exists()
    assert all(candidate.stat().st_size <= 50_000 for candidate in existing)
    rows = _read_json_lines(existing)
    assert any(row["event"] == "shutdown" for row in rows)


def test_operations_log_queue_overflow_is_explicit(make_config, tmp_path, monkeypatch, capfd) -> None:
    path = tmp_path / "operations.jsonl"
    config = make_config(
        logging__file_path=path,
        logging__queue_capacity=1,
        logging__max_bytes=200_000,
    )
    runtime = OperationsLog(config, component="overflow-test")
    original = runtime._write_line

    def slow(line: str) -> None:
        time.sleep(0.01)
        original(line)

    monkeypatch.setattr(runtime, "_write_line", slow)
    for index in range(300):
        runtime.event("fast", index=index)
    runtime.shutdown(reason="overflow_done")
    assert runtime.dropped_count > 0
    captured = capfd.readouterr().err
    assert "queue_overflow" in captured


def test_redaction_masks_nested_secret_value_fields() -> None:
    configured = "configured-runtime-secret"
    payload = {
        "safe": "visible",
        "nested": {
            "password": "p",
            "client_secret": "s",
            "credentials": "c",
            "authorization": "Bearer authorization-secret",
            "access_token": "access-secret",
            "service_api_key": "api-secret",
            "bearer_token_env": "SAFE_ENV_NAME",
            "message": (
                "Authorization: Bearer embedded-secret "
                "api_key=embedded-api " + configured
            ),
        },
    }
    result = _redact(payload, secret_values=(configured,))
    assert result["safe"] == "visible"
    for key in (
        "password",
        "client_secret",
        "credentials",
        "authorization",
        "access_token",
        "service_api_key",
    ):
        assert result["nested"][key].startswith("[REDACTED]:sha256=")
    assert result["nested"]["bearer_token_env"] == "SAFE_ENV_NAME"
    serialized = json.dumps(result)
    for secret in (
        "authorization-secret",
        "access-secret",
        "api-secret",
        "embedded-secret",
        "embedded-api",
        configured,
    ):
        assert secret not in serialized


def test_cli_tools_emits_startup_and_shutdown_operations_log(tmp_path, monkeypatch, capsys) -> None:
    from swaag.cli import main

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    log_path = tmp_path / "cli-operations.jsonl"
    config_path = tmp_path / "cli.toml"
    config_path.write_text(
        "\n".join(
            [
                "[sessions]",
                f'root = "{tmp_path / "sessions"}"',
                "[logging]",
                f'file_path = "{log_path}"',
                "max_bytes = 200000",
                "backup_count = 2",
                "queue_capacity = 64",
            ]
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(config_path), "tools"]) == 0
    _ = capsys.readouterr()
    rows = _read_json_lines([log_path])
    assert rows[0]["event"] == "startup"
    assert rows[-1]["event"] == "shutdown"
    assert rows[-1]["attributes"]["reason"] == "cli_exit"
    assert rows[0]["component"] == "cli:tools"


def test_operations_log_shutdown_survives_saturated_queue(make_config, tmp_path, monkeypatch):
    path = tmp_path / "operations.jsonl"
    config = make_config(
        logging__file_path=path,
        logging__queue_capacity=1,
        logging__max_bytes=500000,
    )
    runtime = OperationsLog(config, component="shutdown-pressure")
    original = runtime._write_line

    def slow(line: str) -> None:
        time.sleep(0.01)
        original(line)

    monkeypatch.setattr(runtime, "_write_line", slow)
    for index in range(100):
        runtime.event("fast", index=index)
    runtime.shutdown(reason="must_survive")
    rows = _read_json_lines([path])
    shutdown = [row for row in rows if row["event"] == "shutdown"]
    assert len(shutdown) == 1
    assert shutdown[0]["attributes"]["reason"] == "must_survive"
    assert shutdown[0]["attributes"]["dropped_log_events"] > 0

def test_operations_log_redacts_configured_secret_values_and_sensitive_key_families(make_config, tmp_path) -> None:
    path = tmp_path / "secret-operations.jsonl"
    config = make_config(logging__file_path=path)
    configured = "configured-bearer-secret"
    config.a2a_authorization.bearer_token = configured
    runtime = OperationsLog(config, component="secret-test")
    runtime.event(
        "auth_failure",
        authorization="Bearer live-auth-secret",
        access_token="live-access-secret",
        api_key="live-api-key",
        custom_api_key="custom-api-key",
        detail=(
            "Authorization: Bearer embedded-bearer; "
            "access_token=embedded-access; " + configured
        ),
    )
    runtime.shutdown(reason="done")
    text = path.read_text(encoding="utf-8")
    for secret in (
        configured,
        "live-auth-secret",
        "live-access-secret",
        "live-api-key",
        "custom-api-key",
        "embedded-bearer",
        "embedded-access",
    ):
        assert secret not in text
    assert "[REDACTED]:sha256=" in text


def test_operations_log_level_filters_routine_events_but_keeps_lifecycle(make_config, tmp_path) -> None:
    path = tmp_path / "level-operations.jsonl"
    config = make_config(
        logging__file_path=path,
        logging__level="WARNING",
    )
    runtime = OperationsLog(config, component="level-test")
    runtime.event("routine_info", severity="INFO", value=1)
    runtime.event("warning_event", severity="WARNING", value=2)
    runtime.shutdown(reason="done")

    rows = _read_json_lines([path])
    assert [row["event"] for row in rows] == [
        "startup",
        "warning_event",
        "shutdown",
    ]
