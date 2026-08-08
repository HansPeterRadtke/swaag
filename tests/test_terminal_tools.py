from __future__ import annotations

import time
from pathlib import Path

import pytest

from swaag.environment.terminal import TerminalStore
from swaag.history import HistoryStore
from swaag.tools.base import ToolValidationError
from swaag.tools.registry import ToolRegistry


def _state(config, session_id="session_terminal"):
    return HistoryStore(config.sessions.root).create(
        config_fingerprint=config.config_fingerprint(),
        model_base_url=config.model.base_url,
        session_id=session_id,
    )


def _wait_for_text(store: TerminalStore, ref: str, needle: str, *, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        last = store.read(ref, start_offset=0, max_chars=20000)
        if needle in last["text"]:
            return last
        time.sleep(0.03)
    raise AssertionError(f"did not observe {needle!r}; last={last!r}")


def test_terminal_store_persists_shell_state_and_reads_incrementally(make_config, tmp_path: Path) -> None:
    config = make_config()
    root = tmp_path / "sessions"
    workspace = tmp_path / "workspace"
    (workspace / "subdir").mkdir(parents=True)
    store = TerminalStore(root, "session_a")
    record = store.create(cwd=workspace, shell="/bin/bash", name="main")
    try:
        assert record.active is True
        store.send("main", "printf 'FIRST\\n'", append_newline=True)
        first = _wait_for_text(store, "main", "FIRST")
        offset = first["next_offset"]
        store.send("main", "cd subdir", append_newline=True)
        store.send("main", "pwd", append_newline=True)
        second = _wait_for_text(store, "main", str(workspace / "subdir"))
        assert str(workspace / "subdir") in second["text"]
        incremental = store.read("main", start_offset=offset, max_chars=20000)
        assert str(workspace / "subdir") in incremental["text"]
    finally:
        closed = store.close("main")
        assert closed.active is False


def test_terminal_supports_interactive_stdin(make_config, tmp_path: Path) -> None:
    store = TerminalStore(tmp_path / "sessions", "session_stdin")
    record = store.create(cwd=tmp_path, shell="/bin/bash", name="interactive")
    try:
        command = "python3 -u -c 'x=input(); print(\"GOT:\" + x)'"
        store.send(record.terminal_id, command, append_newline=True)
        time.sleep(0.1)
        store.send(record.terminal_id, "hello-terminal", append_newline=True)
        result = _wait_for_text(store, record.terminal_id, "GOT:hello-terminal")
        assert "GOT:hello-terminal" in result["text"]
    finally:
        store.close(record.terminal_id)


def test_terminal_list_and_name_resolution(make_config, tmp_path: Path) -> None:
    store = TerminalStore(tmp_path / "sessions", "session_list")
    first = store.create(cwd=tmp_path, shell="/bin/bash", name="alpha")
    second = store.create(cwd=tmp_path, shell="/bin/bash", name="beta")
    try:
        assert store.resolve("alpha") == first.terminal_id
        assert store.resolve(second.terminal_id) == second.terminal_id
        names = {item.name for item in store.list() if item.active}
        assert {"alpha", "beta"} <= names
        with pytest.raises(ValueError, match="already exists"):
            store.create(cwd=tmp_path, shell="/bin/bash", name="alpha")
    finally:
        store.close(first.terminal_id)
        store.close(second.terminal_id)


def test_terminal_send_rejects_closed_terminal(make_config, tmp_path: Path) -> None:
    store = TerminalStore(tmp_path / "sessions", "session_closed")
    record = store.create(cwd=tmp_path, shell="/bin/bash", name="closed")
    store.close(record.terminal_id)
    with pytest.raises(RuntimeError, match="not active"):
        store.send(record.terminal_id, "echo nope", append_newline=True)


def test_terminal_tool_end_to_end(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.tools.allow_side_effect_tools = True
    config.reader.max_chunk_chars = 120
    state = _state(config)
    registry = ToolRegistry()
    _, created = registry.dispatch("terminal", {"operation": "create", "name": "toolterm"}, config, state)
    terminal_id = created.output["terminal_id"]
    try:
        assert created.output["active"] is True
        assert created.generated_events[0].event_type == "terminal_create"
        registry.dispatch(
            "terminal",
            {"operation": "send", "terminal_ref": "toolterm", "data": "printf '%0300d' 0", "append_newline": True},
            config,
            state,
        )
        store = TerminalStore(config.sessions.root, state.session_id)
        _wait_for_text(store, terminal_id, "0" * 100)
        _, read = registry.dispatch(
            "terminal",
            {"operation": "read", "terminal_ref": terminal_id, "start_offset": 0, "max_chars": 1000},
            config,
            state,
        )
        assert len(read.output["text"]) <= 120
        assert read.output["total_chars"] > 120
        _, listed = registry.dispatch("terminal", {"operation": "list"}, config, state)
        assert any(item["terminal_id"] == terminal_id for item in listed.output["terminals"])
    finally:
        registry.dispatch("terminal", {"operation": "close", "terminal_ref": terminal_id}, config, state)


def test_terminal_tool_validation_and_policy(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    state = _state(config, "session_policy")
    tool = ToolRegistry().get("terminal")
    assert tool.effective_kind(tool.validate({"operation": "list"})) == "pure"
    assert tool.effective_kind(tool.validate({"operation": "read", "terminal_ref": "x"})) == "pure"
    assert tool.effective_kind(tool.validate({"operation": "create"})) == "side_effect"
    with pytest.raises(ToolValidationError):
        tool.validate({"operation": "send", "terminal_ref": "x", "data": ""})
    with pytest.raises(PermissionError):
        ToolRegistry().dispatch("terminal", {"operation": "create"}, config, state)


def test_terminal_is_enabled_by_default(make_config) -> None:
    assert "terminal" in ToolRegistry().tool_names(make_config())
