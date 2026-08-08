from __future__ import annotations

from pathlib import Path

from swaag.environment.artifacts import TextArtifactStore
from swaag.history import HistoryStore
from swaag.tools.registry import ToolRegistry


def _state(config, session_id="session_large_output"):
    return HistoryStore(config.sessions.root).create(
        config_fingerprint=config.config_fingerprint(),
        model_base_url=config.model.base_url,
        session_id=session_id,
    )


def test_shell_command_large_stdout_is_bounded_and_preserved(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.environment.max_capture_chars = 64
    config.tools.allow_side_effect_tools = True
    state = _state(config)
    registry = ToolRegistry()

    _, result = registry.dispatch(
        "shell_command",
        {"command": "python3 -c \"print('X' * 500)\"", "background": False},
        config,
        state,
    )

    assert len(result.output["stdout"]) <= 64
    assert result.output["stdout_chars"] > 500
    assert result.output["stdout_truncated"] is True
    artifact_id = result.output["stdout_artifact_id"]
    assert artifact_id
    raw = TextArtifactStore(config.sessions.root, state.session_id).read(artifact_id, max_chars=1000)
    assert raw["text"].startswith("X" * 500)
    assert raw["finished"] is True
    completed = next(event for event in result.generated_events if event.event_type == "process_completed")
    assert len(completed.payload["stdout"]) <= 64
    assert "X" * 200 not in str(completed.payload)


def test_run_tests_large_output_is_bounded_and_preserved(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.environment.max_capture_chars = 80
    state = _state(config, "session_large_tests")

    _, result = ToolRegistry().dispatch(
        "run_tests",
        {"command": ["python3", "-c", "print('Y' * 700)"], "background": False},
        config,
        state,
    )

    assert result.output["passed"] is True
    assert len(result.output["stdout"]) <= 80
    assert result.output["stdout_truncated"] is True
    raw = TextArtifactStore(config.sessions.root, state.session_id).read(result.output["stdout_artifact_id"], max_chars=1000)
    assert raw["text"].startswith("Y" * 700)
    assert raw["finished"] is True


def test_small_output_does_not_create_artifact(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.environment.max_capture_chars = 1000
    config.tools.allow_side_effect_tools = True
    state = _state(config, "session_small_output")

    _, result = ToolRegistry().dispatch(
        "shell_command",
        {"command": "printf 'small'", "background": False},
        config,
        state,
    )

    assert result.output["stdout"] == "small"
    assert result.output["stdout_truncated"] is False
    assert result.output["stdout_artifact_id"] == ""
    assert not list((config.sessions.root / state.session_id / "artifacts").glob("*.txt"))


def test_large_output_artifact_is_readable_via_tool(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.environment.max_capture_chars = 16
    config.tools.allow_side_effect_tools = True
    state = _state(config, "session_read_large")
    registry = ToolRegistry()
    _, shell = registry.dispatch("shell_command", {"command": "printf '%0400d' 0", "background": False}, config, state)

    _, read = registry.dispatch(
        "read_artifact",
        {"artifact_id": shell.output["stdout_artifact_id"], "start_offset": 10, "max_chars": 25},
        config,
        state,
    )

    assert read.output["text"] == "0" * 25
    assert read.output["start_offset"] == 10
