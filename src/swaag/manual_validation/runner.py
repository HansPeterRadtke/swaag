"""Manual validation runner — explicit real-model validation, not a test category."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from swaag.benchmark.benchmark_runner import run_benchmarks
from swaag.benchmark.task_definitions import BENCHMARK_DIFFICULTY_ORDER
from swaag.fsops import ensure_dir, write_text
from swaag.utils import stable_json_dumps


def run_manual_validation(
    *,
    output_dir: Path,
    benchmark_task_ids: list[str] | None = None,
    validation_subset: bool = True,
    model_base_url: str | None = None,
    timeout_seconds: int | None = None,
    connect_timeout_seconds: int | None = None,
    model_profile: str | None = None,
    structured_output_mode: str | None = None,
    progress_poll_seconds: float | None = None,
    seeds: list[int] | None = None,
    clean: bool = False,
) -> dict[str, Any]:

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    report = run_benchmarks(
        output_dir=output_dir / "manual_validation",
        task_ids=benchmark_task_ids,
        clean=True,
        live_subset=validation_subset,
        use_live_model=True,
        model_base_url=model_base_url,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        model_profile=model_profile,
        structured_output_mode=structured_output_mode,
        progress_poll_seconds=progress_poll_seconds,
        seeds=seeds,
        agent_behavior_mode=None,
    )
    difficulty_scores = {
        difficulty: float(report["summary"].get("score_by_difficulty", {}).get(difficulty, 0.0))
        for difficulty in BENCHMARK_DIFFICULTY_ORDER
    }
    payload = {
        **report,
        "category": "manual_validation",
        "is_test_category": False,
        "difficulty_tier_scores": difficulty_scores,
        "percent": float(report["summary"].get("average_task_score_percent", 0.0)),
    }
    write_text(output_dir / "manual_validation_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text(output_dir / "manual_validation_report.md", render_manual_validation_report(payload), encoding="utf-8")
    return payload


def render_manual_validation_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Manual real-model validation",
        "",
        "This is explicit real usage, not a test category.",
        "",
        f"- percent: `{float(payload.get('percent', 0.0)):.2f}%`",
        f"- task_count: `{int(payload.get('summary', {}).get('total_tasks', len(payload.get('tasks', []))))}`",
        "",
        "## Difficulty tier scores",
        "",
    ]
    for difficulty in BENCHMARK_DIFFICULTY_ORDER:
        lines.append(f"- `{difficulty}`: `{float(payload['difficulty_tier_scores'].get(difficulty, 0.0)):.2f}%`")
    lines.extend(["", "## Lowest-scoring tasks", ""])
    tasks = sorted(payload.get("tasks", []), key=lambda item: (float(item.get("score_percent", 0.0)), item.get("task_id", "")))
    for item in tasks[:10]:
        lines.append(f"- `{item.get('task_id', '')}` / `{item.get('difficulty', '')}`: `{float(item.get('score_percent', 0.0)):.2f}%`")
    return "\n".join(lines) + "\n"
