from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from swaag.benchmark import benchmark_runner
from swaag.benchmark.system_suite import get_system_benchmark_families, run_system_benchmarks


def test_system_benchmark_families_cover_core_agent_architecture() -> None:
    families = {item.family_id: item for item in get_system_benchmark_families()}

    assert set(families) == {
        "continuation_background",
        "session_control",
        "session_history_checkpoint",
        "artifact_recovery",
    }
    assert any("test_runtime.py::test_runtime_continues_other_ready_work_while_background_process_runs" in nodeid for nodeid in families["continuation_background"].nodeids)
    assert any("test_session_control.py::test_process_pending_control_messages_queues_follow_up_task" in nodeid for nodeid in families["session_control"].nodeids)
    assert any("test_code_checkpoint_create_and_restore" in nodeid for nodeid in families["session_history_checkpoint"].nodeids)
    assert any("test_artifact_tracking_completed_steps_move_to_completed" in nodeid for nodeid in families["artifact_recovery"].nodeids)


def test_run_system_benchmarks_executes_selected_families_and_writes_reports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="2 passed\n", stderr="")

    monkeypatch.setattr("swaag.benchmark.system_suite.subprocess.run", fake_run)

    report = run_system_benchmarks(
        output_dir=tmp_path / "system",
        family_ids=["continuation_background"],
        clean=True,
    )

    assert report["summary"]["passed_families"] == 1
    assert report["summary"]["failed_families"] == 0
    assert commands == [[
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_runtime.py::test_runtime_continues_other_ready_work_while_background_process_runs",
        "tests/test_runtime.py::test_runtime_enters_wait_state_when_only_background_work_remains",
        "tests/test_history.py::test_history_rebuild_tracks_wait_events_and_background_process_completion",
        "-x",
    ]]
    assert (tmp_path / "system" / "system_benchmark_results.json").exists()
    assert (tmp_path / "system" / "system_benchmark_report.md").exists()


def test_benchmark_runner_system_command_uses_system_suite(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_system_benchmarks(*, output_dir: Path, family_ids, clean: bool):
        observed["output_dir"] = output_dir
        observed["family_ids"] = family_ids
        observed["clean"] = clean
        return {"summary": {"total_families": 1, "passed_families": 1, "failed_families": 0}, "results": []}

    monkeypatch.setattr(benchmark_runner, "run_system_benchmarks", fake_run_system_benchmarks)

    exit_code = benchmark_runner.main(
        ["system", "--family", "session_control", "--output", str(tmp_path / "out"), "--clean"]
    )

    assert exit_code == 0
    assert observed["output_dir"] == tmp_path / "out"
    assert observed["family_ids"] == ["session_control"]
    assert observed["clean"] is True
