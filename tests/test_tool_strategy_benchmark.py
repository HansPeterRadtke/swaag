from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from swaag.benchmark.tool_strategy import (
    STRATEGIES,
    TASKS,
    _case_config,
    _workspace_snapshot,
    _write_fixture,
    build_execution_matrix,
    run_tool_strategy_benchmark,
    verify_case,
)
from swaag.types import SessionMetrics


def test_tool_strategy_matrix_alternates_first_strategy() -> None:
    matrix = build_execution_matrix()

    assert [(task.task_id, strategy.name) for task, strategy in matrix] == [
        ("inspect_and_calculate", "generic_shell"),
        ("inspect_and_calculate", "structured_tools"),
        ("inspect_edit_and_verify", "structured_tools"),
        ("inspect_edit_and_verify", "generic_shell"),
    ]


def test_tool_strategy_matrix_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown tool strategy"):
        build_execution_matrix(strategy_names=["missing"])
    with pytest.raises(ValueError, match="Unknown tool-strategy task"):
        build_execution_matrix(task_ids=["missing"])


def test_tool_strategy_verifier_requires_evidence_and_exact_workspace(tmp_path: Path) -> None:
    task = TASKS[0]
    workspace = tmp_path / "workspace"
    _write_fixture(workspace, task.fixture_files)
    initial = _workspace_snapshot(workspace)

    passing = verify_case(
        task,
        workspace=workspace,
        initial_snapshot=initial,
        assistant_text="The exact combined count is 423 units.",
        tool_names=["load_tools", "shell_command"],
    )
    no_tool = verify_case(
        task,
        workspace=workspace,
        initial_snapshot=initial,
        assistant_text="423",
        tool_names=[],
    )
    (workspace / "inventory/north.txt").write_text("region=north\nunits=138\n", encoding="utf-8")
    changed = verify_case(
        task,
        workspace=workspace,
        initial_snapshot=initial,
        assistant_text="423",
        tool_names=["read_file"],
    )

    assert passing["passed"] is True
    assert no_tool["checks"]["used_domain_tool"] is False
    assert changed["checks"]["workspace_exact"] is False


def test_tool_strategy_case_config_isolates_workspace_and_disables_replay(make_config, tmp_path: Path) -> None:
    base = make_config()
    workspace = tmp_path / "workspace"
    sessions = tmp_path / "sessions"
    target = workspace / "config/service.conf"
    config = _case_config(
        base,
        workspace=workspace,
        sessions_root=sessions,
        strategy=STRATEGIES["generic_shell"],
        allowed_write_paths=[target],
    )

    assert config.sessions.root == sessions
    assert config.tools.read_roots == [workspace]
    assert config.tools.enabled == ["shell_command"]
    assert config.tools.staged_discovery is True
    assert config.tools.allow_side_effect_tools is True
    assert config.editor.allowed_write_paths == [str(target.resolve())]
    assert config.model.cache_enabled is False
    assert config.runtime.completion_evaluation_enabled is False
    assert base.tools.enabled != config.tools.enabled


class _FakeHistory:
    def __init__(self) -> None:
        self.tool_name = ""

    def read_history(self, _session_id: str):
        return [SimpleNamespace(event_type="tool_called", payload={"tool_name": self.tool_name})]

    def rebuild_from_history(self, _session_id: str, *, prefer_checkpoint: bool):
        del prefer_checkpoint
        return SimpleNamespace(metrics=SessionMetrics(model_calls=2, tool_calls=1, action_count=2))


class _FakeRuntime:
    def __init__(self, config) -> None:
        self.config = config
        self.history = _FakeHistory()

    def create_or_load_session(self):
        return SimpleNamespace(session_id="session-test")

    def run_turn_in_session(self, _state, _prompt: str):
        workspace = self.config.tools.read_roots[0]
        if (workspace / "config/service.conf").exists():
            target = workspace / "config/service.conf"
            target.write_text(target.read_text(encoding="utf-8").replace("=9\n", "=14\n"), encoding="utf-8")
            answer = "Configuration updated and verified."
        else:
            answer = "423"
        self.history.tool_name = self.config.tools.enabled[0]
        return SimpleNamespace(assistant_text=answer)


def test_tool_strategy_runner_checkpoints_complete_report(make_config, tmp_path: Path) -> None:
    output = tmp_path / "results"
    report = run_tool_strategy_benchmark(
        config=make_config(),
        output_dir=output,
        runtime_factory=_FakeRuntime,
        model_identity={"model": "fake"},
    )

    assert report["status"] == "complete"
    assert report["passed"] == report["total"] == 4
    assert report["by_strategy"]["generic_shell"]["passed"] == 2
    assert report["by_strategy"]["structured_tools"]["passed"] == 2
    assert (output / "tool_strategy_results.json").is_file()
    assert len(list((output / "runs").glob("*/workspace"))) == 4
    with pytest.raises(FileExistsError, match="already exists"):
        run_tool_strategy_benchmark(
            config=make_config(),
            output_dir=output,
            runtime_factory=_FakeRuntime,
            model_identity={"model": "fake"},
        )
