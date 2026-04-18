from __future__ import annotations

import time
from pathlib import Path

from swaag.context_builder import build_context
from swaag.environment.environment import AgentEnvironment
from swaag.runtime import AgentRuntime
from swaag.tokens import ConservativeEstimator
from swaag.types import SessionState

from tests.helpers import FakeModelClient


def _record_generated_events(runtime: AgentRuntime, state: SessionState, result) -> None:
    for event in result.generated_events:
        runtime.history.record_event(
            state,
            event.event_type,
            event.payload,
            metadata=event.metadata,
            derived_writes=event.derived_writes,
        )


def test_environment_shell_session_persists_cwd_and_env(make_config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    subdir = workspace / "subdir"
    subdir.mkdir(parents=True)
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())

    first = runtime.execute_tool_once(
        "shell_command",
        {"command": f"cd {subdir} && export DEMO_VALUE=ready"},
    )
    second = runtime.execute_tool_once(
        "shell_command",
        {"command": "printf '%s:%s' \"$PWD\" \"$DEMO_VALUE\""},
        session_id=first.session_id,
    )

    assert second.tool_result is not None
    assert second.tool_result.output["cwd_after"] == str(subdir)
    assert second.tool_result.output["stdout"] == f"{subdir}:ready"

    rebuilt = runtime.history.rebuild_from_history(first.session_id)
    assert rebuilt.environment.shell.cwd == str(subdir)
    assert rebuilt.environment.shell.env_overrides["DEMO_VALUE"] == "ready"
    events = runtime.history.read_history(first.session_id)
    assert [event.event_type for event in events].count("shell_command_completed") == 2
    assert [event.event_type for event in events].count("process_completed") == 2


def test_shell_command_output_is_trimmed_to_capture_limit(make_config) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        environment__max_capture_chars=16,
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())

    run = runtime.execute_tool_once(
        "shell_command",
        {"command": "python3 - <<'PY'\nimport string\nprint(string.ascii_lowercase)\nPY"},
    )

    assert run.tool_result is not None
    assert run.tool_result.output["stdout"] == "abcdefghijklmnop"
    assert "abcdefghijklmnop" in run.tool_result.display_text
    assert "qrstuvwxyz" not in run.tool_result.display_text


def test_shell_command_invalid_syntax_preserves_session_state(make_config, tmp_path: Path) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())

    run = runtime.execute_tool_once(
        "shell_command",
        {"command": "git apply -v <patch_file> && git diff --cached"},
    )

    assert run.tool_result is None
    events = runtime.history.read_history(run.session_id)
    error_event = next(
        event
        for event in events
        if event.event_type == "tool_error" and event.payload.get("tool_name") == "shell_command"
    )
    assert "placeholder" in error_event.payload["error"]

    rebuilt = runtime.history.rebuild_from_history(run.session_id)
    assert rebuilt.environment.shell.cwd == runtime.create_or_load_session(run.session_id).environment.shell.cwd


def test_environment_file_changes_are_visible_across_steps_and_replay(make_config, tmp_path: Path) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())

    write_run = runtime.execute_tool_once(
        "write_file",
        {"path": "notes/result.txt", "content": "ready\n"},
    )
    read_run = runtime.execute_tool_once(
        "read_file",
        {"path": "notes/result.txt"},
        session_id=write_run.session_id,
    )

    assert read_run.tool_result is not None
    assert read_run.tool_result.output["text"] == "ready\n"

    rebuilt = runtime.history.rebuild_from_history(write_run.session_id)
    assert rebuilt.environment.workspace.known_files["notes/result.txt"] == "ready\n"
    assert "notes/result.txt" in rebuilt.environment.workspace.listed_files


def test_shell_command_workspace_snapshot_records_only_delta(make_config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    untouched = workspace / "untouched.txt"
    touched = workspace / "touched.txt"
    untouched.write_text("keep\n", encoding="utf-8")
    touched.write_text("before\n", encoding="utf-8")
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        tools__read_roots=[workspace],
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())

    run = runtime.execute_tool_once(
        "shell_command",
        {"command": "python3 - <<'PY'\nfrom pathlib import Path\nPath('touched.txt').write_text('after\\n', encoding='utf-8')\nPY"},
    )

    events = runtime.history.read_history(run.session_id)
    snapshot_event = next(event for event in events if event.event_type == "workspace_snapshot")

    assert snapshot_event.payload["snapshot_mode"] == "delta"
    assert snapshot_event.payload["files"] == {"touched.txt": "after\n"}
    rebuilt = runtime.history.rebuild_from_history(run.session_id)
    assert rebuilt.environment.workspace.known_files["touched.txt"] == "after\n"
    assert "untouched.txt" not in rebuilt.environment.workspace.known_files


def test_context_builder_selects_relevant_environment_files(make_config, tmp_path: Path) -> None:
    config = make_config(
        context_builder__max_environment_files=1,
    )
    workspace = tmp_path / "repo"
    state = SessionState(
        session_id="session",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url="http://example.test",
    )
    state.environment.workspace.root = str(workspace)
    state.environment.workspace.cwd = str(workspace)
    state.environment.workspace.known_files = {
        "src/app.py": "def render_report() -> str:\n    return 'report'\n",
        "README.md": "project overview only\n",
    }
    state.environment.workspace.listed_files = sorted(state.environment.workspace.known_files)

    bundle = build_context(config, state, ConservativeEstimator(), goal="Update render_report in src/app.py")

    component_names = [component.name for component in bundle.components if component.text]
    assert "environment" in component_names
    assert "environment_files" in component_names
    assert "src/app.py" in bundle.relevant_files_text
    assert "README.md" not in bundle.relevant_files_text
    assert any(item.item_type == "environment_file" and item.selected for item in bundle.selection_trace)


def test_environment_search_tools_and_snapshot_are_recorded(make_config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "alpha.txt").write_text("owner=carol\nstatus=ready\n", encoding="utf-8")
    (workspace / "beta.txt").write_text("owner=bob\n", encoding="utf-8")
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        tools__read_roots=[workspace],
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())

    search_file = runtime.execute_tool_once(
        "search_in_file",
        {"path": str(workspace / "alpha.txt"), "pattern": "owner=carol"},
    )
    assert search_file.tool_result is not None
    assert search_file.tool_result.output["match_count"] == 1

    search_repo = runtime.execute_tool_once(
        "search_repo",
        {"path": str(workspace), "pattern": "owner=", "max_matches": 10},
        session_id=search_file.session_id,
    )
    assert search_repo.tool_result is not None
    assert sorted(search_repo.tool_result.output["matched_files"]) == ["alpha.txt", "beta.txt"]

    snapshot = runtime.execute_tool_once(
        "workspace_snapshot",
        {},
        session_id=search_repo.session_id,
    )
    assert snapshot.tool_result is not None
    assert snapshot.tool_result.output["file_count"] == 2

    events = [event.event_type for event in runtime.history.read_history(search_file.session_id)]
    assert "filesystem_search" in events
    assert "repository_searched" in events
    assert "workspace_snapshot_inspected" in events


def test_environment_diff_and_change_listing(make_config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        tools__read_roots=[workspace],
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())

    first = runtime.execute_tool_once(
        "edit_text",
        {
            "path": str(target),
            "operation": "replace_pattern_once",
            "pattern": "hello",
            "replacement": "world",
            "dry_run": True,
        },
    )
    diff_run = runtime.execute_tool_once("inspect_diff", {"path": str(target)}, session_id=first.session_id)
    assert diff_run.tool_result is not None
    assert diff_run.tool_result.output["changed"] is True
    assert "-hello" in diff_run.tool_result.output["diff"]
    assert "+world" in diff_run.tool_result.output["diff"]

    write_run = runtime.execute_tool_once(
        "write_file",
        {"path": str(workspace / "created.txt"), "content": "ready\n"},
        session_id=first.session_id,
    )
    changes = runtime.execute_tool_once("list_changes", {}, session_id=write_run.session_id)
    assert changes.tool_result is not None
    assert "created.txt" in changes.tool_result.output["created_files"]


def test_environment_stuck_detection_flags_repeated_useless_actions(make_config, tmp_path: Path) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())

    run = runtime.execute_tool_once("shell_command", {"command": "false"})
    for _ in range(2):
        run = runtime.execute_tool_once("shell_command", {"command": "false"}, session_id=run.session_id)
    history_events = runtime.history.read_history(run.session_id)
    state = runtime.history.rebuild_from_history(run.session_id)
    stuck = AgentEnvironment(config, state).detect_stuck_patterns(history_events)
    assert "repeated_useless_command" in stuck


def test_environment_background_shell_command_persists_and_rebuilds(make_config, tmp_path: Path) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        runtime__background_poll_seconds=0.0,
        runtime__tool_timeout_seconds=2,
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    environment = AgentEnvironment(config, state)

    started = environment.run_shell_command("sleep 0.05; printf background-ready", background=True)
    assert started.completed is False
    process_id = str(started.output["process_id"])
    _record_generated_events(runtime, state, started)

    rebuilt = runtime.history.rebuild_from_history(state.session_id)
    assert rebuilt.environment.processes[process_id].status == "running"

    resumed_environment = AgentEnvironment(config, rebuilt)
    deadline = time.monotonic() + 2.0
    while True:
        update = resumed_environment.poll_background_process(process_id)
        _record_generated_events(runtime, rebuilt, update)
        if update.completed:
            assert update.tool_result is not None
            assert update.tool_result.output["stdout"] == "background-ready"
            break
        if time.monotonic() >= deadline:
            raise AssertionError("background shell command did not complete in time")
        time.sleep(0.01)

    final_state = runtime.history.rebuild_from_history(state.session_id)
    assert final_state.environment.processes[process_id].status == "completed"
    events = [event.event_type for event in runtime.history.read_history(state.session_id)]
    assert "process_started" in events
    assert "process_polled" in events
    assert "process_completed" in events
    assert "shell_command_completed" in events


def test_environment_can_kill_background_test_process(make_config, tmp_path: Path) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        runtime__background_poll_seconds=0.0,
        runtime__tool_timeout_seconds=5,
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    environment = AgentEnvironment(config, state)
    command = [
        "python3",
        "-c",
        "import time; time.sleep(5)",
    ]

    started = environment.run_tests(command, background=True)
    process_id = str(started.output["process_id"])
    _record_generated_events(runtime, state, started)

    killed = environment.kill_background_process(process_id)
    _record_generated_events(runtime, state, killed)

    rebuilt = runtime.history.rebuild_from_history(state.session_id)
    assert rebuilt.environment.processes[process_id].status == "killed"


def test_environment_background_run_tests_completes(make_config, tmp_path: Path) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        runtime__background_poll_seconds=0.0,
        runtime__tool_timeout_seconds=2,
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    environment = AgentEnvironment(config, state)

    started = environment.run_tests(["python3", "-c", "print('ok')"], background=True)
    process_id = str(started.output["process_id"])
    _record_generated_events(runtime, state, started)

    deadline = time.monotonic() + 2.0
    while True:
        update = environment.poll_background_process(process_id)
        _record_generated_events(runtime, state, update)
        if update.completed:
            assert update.tool_result is not None
            assert update.tool_result.tool_name == "run_tests"
            assert update.tool_result.output["passed"] is True
            assert update.tool_result.output["stdout"].strip() == "ok"
            break
        if time.monotonic() >= deadline:
            raise AssertionError("background test process did not complete in time")
        time.sleep(0.01)

    rebuilt = runtime.history.rebuild_from_history(state.session_id)
    assert rebuilt.environment.processes[process_id].status == "completed"


def test_environment_kill_background_shell_command_terminates_child_process(make_config, tmp_path: Path) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        runtime__background_poll_seconds=0.0,
        runtime__tool_timeout_seconds=5,
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    environment = AgentEnvironment(config, state)
    target = tmp_path / "late.txt"

    started = environment.run_shell_command(
        f"sleep 0.3; printf late > {target}",
        background=True,
    )
    process_id = str(started.output["process_id"])
    _record_generated_events(runtime, state, started)

    killed = environment.kill_background_process(process_id)
    _record_generated_events(runtime, state, killed)
    time.sleep(0.45)

    rebuilt = runtime.history.rebuild_from_history(state.session_id)
    assert rebuilt.environment.processes[process_id].status == "killed"
    assert not target.exists()
