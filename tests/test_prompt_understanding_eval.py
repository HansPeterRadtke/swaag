from __future__ import annotations

from pathlib import Path

from swaag.benchmark.benchmark_runner import run_benchmarks


UNDERSTANDING_TASKS = [
    "reading_debug_log_summary",
    "quality_vague_expansion",
    "quality_already_decomposed_prompt",
    "quality_incomplete_clarification",
]


def test_prompt_understanding_benchmark_tasks_match_their_oracles(tmp_path: Path) -> None:
    report = run_benchmarks(output_dir=tmp_path / "benchmark", clean=True, task_ids=UNDERSTANDING_TASKS)

    assert report["summary"]["failed_tasks"] == 0
    assert report["aggregate_metrics"]["understanding"]["understanding_success_rate"] == 1.0
    assert report["aggregate_metrics"]["primary"]["correct_task_classification_rate"] == 1.0
    assert report["aggregate_metrics"]["primary"]["correct_decomposition_rate"] == 1.0
    assert all(task["quality_summary"]["passed"] for task in report["tasks"])
