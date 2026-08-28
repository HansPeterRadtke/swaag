from __future__ import annotations

import time
from pathlib import Path

from swaag.environment.artifacts import TextArtifactStore
from swaag.history import HistoryStore
from swaag.runtime import AgentRuntime
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


def test_completed_background_shell_output_is_bounded_and_recoverable(
    make_config, tmp_path: Path
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.environment.max_capture_chars = 32
    config.tools.allow_side_effect_tools = True
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    started = runtime.execute_tool_once(
        "shell_command",
        {
            "command": "python3 -u -c \"print('B' * 500)\"",
            "background": True,
        },
        session_id=state.session_id,
    ).tool_result
    assert started is not None

    poll = None
    for _ in range(200):
        poll = runtime.execute_tool_once(
            "poll_process",
            {"process_id": started.output["process_id"]},
            session_id=state.session_id,
        ).tool_result
        assert poll is not None
        if poll.output["completed"]:
            break
        time.sleep(0.02)

    assert poll is not None
    assert poll.output["completed"] is True
    assert len(poll.output["stdout"]) <= 32
    assert poll.output["stdout_chars"] > 500
    assert poll.output["stdout_truncated"] is True
    assert poll.output["completed_tool_result"]["output"]["stdout_artifact_id"] == poll.output[
        "stdout_artifact_id"
    ]
    raw = TextArtifactStore(config.sessions.root, state.session_id).read(
        poll.output["stdout_artifact_id"], max_chars=1000
    )
    assert raw["text"] == "B" * 500 + "\n"
    events = runtime.history.read_history(state.session_id)
    final_poll = next(event for event in reversed(events) if event.event_type == "process_polled")
    assert len(final_poll.payload["stdout"]) <= 32
    assert "B" * 100 not in str(final_poll.payload)
    assert final_poll.payload["output_artifacts"]["stdout_artifact_id"]


def test_running_poll_and_kill_never_inline_unbounded_process_output(
    make_config, tmp_path: Path
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.environment.max_capture_chars = 24
    config.tools.allow_side_effect_tools = True
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    started = runtime.execute_tool_once(
        "shell_command",
        {
            "command": (
                "python3 -u -c \"import sys,time; print('K' * 500); "
                "sys.stdout.flush(); time.sleep(30)\""
            ),
            "background": True,
        },
        session_id=state.session_id,
    ).tool_result
    assert started is not None
    process_id = started.output["process_id"]
    killed = False
    try:
        poll = None
        for _ in range(200):
            poll = runtime.execute_tool_once(
                "poll_process",
                {"process_id": process_id},
                session_id=state.session_id,
            ).tool_result
            assert poll is not None
            if poll.output["stdout_chars"] > 500:
                break
            time.sleep(0.02)
        assert poll is not None
        assert poll.output["completed"] is False
        assert len(poll.output["stdout"]) <= 24
        assert poll.output["stdout_truncated"] is True
        running_raw = TextArtifactStore(config.sessions.root, state.session_id).read(
            poll.output["stdout_artifact_id"], max_chars=1000
        )
        assert running_raw["text"] == "K" * 500 + "\n"

        stopped = runtime.execute_tool_once(
            "kill_process",
            {"process_id": process_id},
            session_id=state.session_id,
        ).tool_result
        killed = True
        assert stopped is not None
        assert stopped.output["status"] == "killed"
        assert len(stopped.output["stdout"]) <= 24
        assert stopped.output["stdout_chars"] > 500
        assert stopped.output["stdout_truncated"] is True
        killed_raw = TextArtifactStore(config.sessions.root, state.session_id).read(
            stopped.output["stdout_artifact_id"], max_chars=1000
        )
        assert killed_raw["text"] == "K" * 500 + "\n"
        events = runtime.history.read_history(state.session_id)
        killed_event = next(event for event in reversed(events) if event.event_type == "process_killed")
        assert len(killed_event.payload["stdout"]) <= 24
        assert "K" * 100 not in str(killed_event.payload)
        assert killed_event.payload["output_artifacts"]["stdout_artifact_id"]
    finally:
        if not killed:
            runtime.execute_tool_once(
                "kill_process",
                {"process_id": process_id},
                session_id=state.session_id,
            )
