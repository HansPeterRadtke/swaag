from __future__ import annotations

from pathlib import Path

from swaag.benchmark.benchmark_runner import run_benchmarks


FALSE_POSITIVE_TASKS = [
    "file_edit_noop_detection",
    "failure_wrong_tool_usage",
    "failure_repeated_action_trap",
    "quality_incomplete_clarification",
]


def test_false_positive_killer_tasks_never_report_success_incorrectly(tmp_path: Path) -> None:
    report = run_benchmarks(output_dir=tmp_path / "benchmark", clean=True, task_ids=FALSE_POSITIVE_TASKS)

    assert report["summary"]["failed_tasks"] == 0
    assert report["summary"]["false_positives"] == 0
    assert report["aggregate_metrics"]["primary"]["false_positive_rate"] == 0.0
    assert {task["task_id"] for task in report["tasks"]} == set(FALSE_POSITIVE_TASKS)
