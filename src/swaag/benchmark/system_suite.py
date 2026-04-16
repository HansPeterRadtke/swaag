from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from swaag.utils import stable_json_dumps


@dataclass(frozen=True, slots=True)
class SystemBenchmarkFamily:
    family_id: str
    description: str
    nodeids: tuple[str, ...]


@dataclass(slots=True)
class SystemBenchmarkFamilyResult:
    family_id: str
    description: str
    command: list[str]
    status: str
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    nodeids: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_system_benchmark_families() -> list[SystemBenchmarkFamily]:
    return [
        SystemBenchmarkFamily(
            family_id="continuation_background",
            description="Background job overlap, wait/resume, and recovered finalization.",
            nodeids=(
                "tests/test_runtime.py::test_runtime_continues_other_ready_work_while_background_process_runs",
                "tests/test_runtime.py::test_runtime_enters_wait_state_when_only_background_work_remains",
                "tests/test_history.py::test_history_rebuild_tracks_wait_events_and_background_process_completion",
            ),
        ),
        SystemBenchmarkFamily(
            family_id="session_control",
            description="Active-session control queuing, non-destructive continuation, and deferred work.",
            nodeids=(
                "tests/test_session_control.py::test_runtime_processes_continue_control_without_stopping_current_work",
                "tests/test_session_control.py::test_process_pending_control_messages_returns_non_destructive_control_responses",
                "tests/test_session_control.py::test_process_pending_control_messages_queues_follow_up_task",
                "tests/test_session_control.py::test_continue_with_note_adds_note_without_forcing_replan",
            ),
        ),
        SystemBenchmarkFamily(
            family_id="session_history_checkpoint",
            description="Human-readable session UX, exact-detail history retrieval, and checkpoints.",
            nodeids=(
                "tests/test_sessions.py::test_runtime_create_or_load_user_session_defaults_to_latest",
                "tests/test_session_control.py::test_cli_sessions_list_and_rename_use_human_names",
                "tests/test_session_control.py::test_history_detail_query_returns_exact_old_command",
                "tests/test_session_control.py::test_history_detail_query_handles_quoted_exact_terms",
                "tests/test_session_control.py::test_code_checkpoint_create_and_restore",
            ),
        ),
        SystemBenchmarkFamily(
            family_id="artifact_recovery",
            description="Artifact-first state tracking and bounded recovery after replans.",
            nodeids=(
                "tests/test_project_state.py::test_artifact_tracking_pending_steps_appear_in_expected_and_pending",
                "tests/test_project_state.py::test_artifact_tracking_completed_steps_move_to_completed",
                "tests/test_project_state.py::test_artifact_state_is_rendered_in_context",
                "tests/test_long_running_tasks.py::test_runtime_recovers_after_multiple_replans_and_remains_bounded",
            ),
        ),
    ]


def run_system_benchmarks(
    *,
    output_dir: Path,
    family_ids: Sequence[str] | None = None,
    clean: bool = False,
) -> dict[str, object]:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    families = get_system_benchmark_families()
    if family_ids:
        selected_map = {item.family_id: item for item in families}
        missing = [family_id for family_id in family_ids if family_id not in selected_map]
        if missing:
            raise SystemExit(f"Unknown system benchmark families: {', '.join(missing)}")
        families = [selected_map[item] for item in family_ids]
    results: list[SystemBenchmarkFamilyResult] = []
    started = time.monotonic()
    repo_root = _repo_root()
    for family in families:
        family_dir = output_dir / family.family_id
        family_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = family_dir / "stdout.txt"
        stderr_path = family_dir / "stderr.txt"
        command = [sys.executable, "-m", "pytest", "-q", *family.nodeids, "-x"]
        run_started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        results.append(
            SystemBenchmarkFamilyResult(
                family_id=family.family_id,
                description=family.description,
                command=command,
                status="passed" if completed.returncode == 0 else "failed",
                duration_seconds=round(time.monotonic() - run_started, 3),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                nodeids=list(family.nodeids),
            )
        )
    payload = {
        "summary": {
            "total_families": len(results),
            "passed_families": sum(1 for item in results if item.status == "passed"),
            "failed_families": sum(1 for item in results if item.status != "passed"),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
        },
        "results": [asdict(item) for item in results],
    }
    (output_dir / "system_benchmark_results.json").write_text(
        stable_json_dumps(payload, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# Agent-system benchmark report",
        "",
        f"- total_families: {payload['summary']['total_families']}",
        f"- passed_families: {payload['summary']['passed_families']}",
        f"- failed_families: {payload['summary']['failed_families']}",
        "",
        "| Family | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for result in results:
        report_lines.append(f"| {result.family_id} | {result.status} | `{result.stdout_path}` |")
    (output_dir / "system_benchmark_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return payload
