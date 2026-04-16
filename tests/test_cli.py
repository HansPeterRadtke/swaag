from __future__ import annotations

import json
from pathlib import Path

from swaag.history import HistoryStore
from swaag.cli import main



def test_cli_tools_command(capsys) -> None:
    rc = main(["tools"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "echo" in out
    assert "read_text" in out



def test_cli_budget_demo(capsys) -> None:
    rc = main(["budget-demo", "hello"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[decision]" in out
    assert "[answer]" in out



def test_cli_edit_dry_run(tmp_path: Path, capsys, monkeypatch) -> None:
    sessions_root = tmp_path / "sessions"
    monkeypatch.setenv("SWAAG__SESSIONS__ROOT", str(sessions_root))
    monkeypatch.setenv("SWAAG__TOOLS__READ_ROOTS", json.dumps([str(tmp_path)]))
    monkeypatch.setenv("SWAAG__MODEL__BASE_URL", "http://127.0.0.1:9999")
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    rc = main(["edit-dry-run", str(path), "replace_pattern_all", "--pattern", "hello", "--replacement", "world"])
    captured = capsys.readouterr()
    out = captured.out
    assert rc == 0
    assert "world" in out
    session_id = captured.err.strip().removeprefix("[session=").removesuffix("]")
    events = HistoryStore(sessions_root).read_history(session_id)
    assert any(event.event_type == "tool_called" for event in events)
    assert any(event.event_type == "file_read_for_edit" for event in events)
    assert any(event.event_type == "edit_previewed" for event in events)



def test_cli_history_show_replay_and_diff(tmp_path: Path, capsys, monkeypatch) -> None:
    sessions_root = tmp_path / "sessions"
    monkeypatch.setenv("SWAAG__SESSIONS__ROOT", str(sessions_root))
    monkeypatch.setenv("SWAAG__TOOLS__READ_ROOTS", json.dumps([str(tmp_path)]))
    monkeypatch.setenv("SWAAG__MODEL__BASE_URL", "http://127.0.0.1:9999")
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    rc = main(["edit-dry-run", str(path), "replace_pattern_all", "--pattern", "hello", "--replacement", "world"])
    assert rc == 0
    session_id = capsys.readouterr().err.strip().removeprefix("[session=").removesuffix("]")

    rc = main(["history", "show", session_id, "--event-type", "tool_called"])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"type":"tool_called"' in out

    rc = main(["history", "replay", session_id])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"session_id":' in out

    rc = main(["history", "diff", session_id])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"match": true' in out.lower()


def test_cli_control_history_detail_and_checkpoint_round_trip(tmp_path: Path, capsys, monkeypatch) -> None:
    sessions_root = tmp_path / "sessions"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sample = workspace / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    monkeypatch.setenv("SWAAG__SESSIONS__ROOT", str(sessions_root))
    monkeypatch.setenv("SWAAG__TOOLS__READ_ROOTS", json.dumps([str(workspace)]))
    monkeypatch.setenv("SWAAG__MODEL__BASE_URL", "http://127.0.0.1:9999")

    store = HistoryStore(sessions_root)
    state = store.create(
        config_fingerprint="cfg",
        model_base_url="http://example.test",
        session_name="alpha",
        session_name_source="explicit",
    )
    store.set_active_run(state.session_id, run_id="run_1", user_text="working")
    rebuilt = store.rebuild_from_history(state.session_id, write_projections=False)
    store.record_event(
        rebuilt,
        "shell_command_completed",
        {
            "command": "cp src.txt dst.txt",
            "cwd_before": str(workspace),
            "cwd_after": str(workspace),
            "env_overrides": {},
            "unset_vars": [],
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        },
    )

    assert main(["control", "Keep going but answer numerically.", "--session", "alpha"]) == 0
    control_out = capsys.readouterr().out
    assert "Queued control message" in control_out
    pending = store.list_pending_control_messages(state.session_id)
    assert len(pending) == 1

    assert main(["history", "detail", "alpha", "What exact command copied src.txt to dst.txt?"]) == 0
    detail_out = capsys.readouterr().out
    assert "cp src.txt dst.txt" in detail_out

    assert main(["checkpoint", "create", "--session", "alpha", "--label", "before-edit", "--workspace", str(workspace)]) == 0
    create_out = capsys.readouterr().out
    assert "Checkpoint created:" in create_out

    sample.write_text("changed", encoding="utf-8")
    assert main(["checkpoint", "restore", "--session", "alpha", "--workspace", str(workspace)]) == 0
    restore_out = capsys.readouterr().out
    assert "Restored checkpoint" in restore_out
    assert sample.read_text(encoding="utf-8") == "hello"
