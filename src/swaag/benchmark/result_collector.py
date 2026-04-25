from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from swaag.benchmark.metrics import BenchmarkAggregateMetrics, compute_benchmark_metrics
from swaag.benchmark.report import render_benchmark_report
from swaag.benchmark.scoring import TASK_SCORE_COMPONENT_WEIGHTS, build_task_rubric
from swaag.utils import stable_json_dumps


@dataclass(slots=True)
class BenchmarkTaskResult:
    task_id: str
    task_type: str
    difficulty: str
    tags: list[str]
    description: str
    expected_outcome: str
    success: bool
    false_positive: bool
    session_id: str | None
    assistant_text: str
    deterministic_verification_passed: bool
    verification_summary: dict[str, Any]
    quality_summary: dict[str, Any]
    metrics: dict[str, Any]
    failure_category: str | None
    failure_reason: str | None
    failure_subsystem: str | None
    wall_clock_seconds: float = 0.0
    improvement_hints: list[str] = field(default_factory=list)
    history_path: str | None = None
    workspace: str = ""
    score_percent: float = 0.0
    rubric_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkSummary:
    generated_at: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    false_positives: int
    success_rate_by_type: dict[str, float]
    failure_breakdown: dict[str, int]
    score_by_family: dict[str, float] = field(default_factory=dict)
    score_by_difficulty: dict[str, float] = field(default_factory=dict)
    full_task_success_percent: float = 0.0
    family_group_average_percent: float = 0.0
    difficulty_group_average_percent: float = 0.0
    group_average_percent: float = 0.0
    average_task_score_percent: float = 0.0


@dataclass(slots=True)
class BenchmarkRunReport:
    summary: BenchmarkSummary
    aggregate_metrics: BenchmarkAggregateMetrics
    run_metadata: dict[str, Any] = field(default_factory=dict)
    improvement_hints: list[str] = field(default_factory=list)
    tasks: list[BenchmarkTaskResult] = field(default_factory=list)
    score_weights: dict[str, float] = field(default_factory=lambda: dict(TASK_SCORE_COMPONENT_WEIGHTS))


def _cache_transparency_summary(tasks: list[BenchmarkTaskResult]) -> tuple[dict[str, int], dict[str, int]]:
    seed_cache_mode_counts: dict[str, int] = {}
    task_cache_mode_counts: dict[str, int] = {}
    for task in tasks:
        seed_modes = set()
        for seed_result in task.metrics.get("seed_results", []):
            if not isinstance(seed_result, dict):
                continue
            replay_cache = seed_result.get("replay_cache", {})
            mode = str(replay_cache.get("cache_mode") or ("replay" if replay_cache.get("cassette_path") else "uncached"))
            seed_modes.add(mode)
        if not seed_modes:
            task_mode = "uncached"
        elif len(seed_modes) == 1:
            task_mode = next(iter(seed_modes))
        else:
            task_mode = "mixed"
        task_cache_mode_counts[task_mode] = task_cache_mode_counts.get(task_mode, 0) + 1
        for mode in seed_modes or {"uncached"}:
            seed_cache_mode_counts[mode] = seed_cache_mode_counts.get(mode, 0) + 1
    return dict(sorted(seed_cache_mode_counts.items())), dict(sorted(task_cache_mode_counts.items()))


class ResultCollector:
    def __init__(self) -> None:
        self._results: list[BenchmarkTaskResult] = []

    def add(self, result: BenchmarkTaskResult) -> None:
        self._results.append(result)

    @property
    def results(self) -> list[BenchmarkTaskResult]:
        return list(self._results)

    def build_report(self) -> BenchmarkRunReport:
        by_type_total: dict[str, int] = {}
        by_type_success: dict[str, int] = {}
        failure_breakdown: dict[str, int] = {}
        score_by_family_totals: dict[str, list[float]] = {}
        score_by_difficulty_totals: dict[str, list[float]] = {}
        false_positives = 0
        hints: list[str] = []
        for result in self._results:
            score_percent, rubric_breakdown = build_task_rubric(
                success=result.success,
                verification_summary=result.verification_summary,
                quality_summary=result.quality_summary,
            )
            result.score_percent = score_percent
            result.rubric_breakdown = rubric_breakdown
            by_type_total[result.task_type] = by_type_total.get(result.task_type, 0) + 1
            if result.success:
                by_type_success[result.task_type] = by_type_success.get(result.task_type, 0) + 1
            if result.false_positive:
                false_positives += 1
            if result.failure_category is not None:
                failure_breakdown[result.failure_category] = failure_breakdown.get(result.failure_category, 0) + 1
            score_by_family_totals.setdefault(result.task_type, []).append(float(result.score_percent))
            score_by_difficulty_totals.setdefault(result.difficulty, []).append(float(result.score_percent))
            hints.extend(result.improvement_hints)
        success_rate_by_type = {
            task_type: 0.0 if by_type_total[task_type] == 0 else by_type_success.get(task_type, 0) / by_type_total[task_type]
            for task_type in sorted(by_type_total)
        }
        score_by_family = {
            task_type: round(sum(values) / len(values), 2)
            for task_type, values in sorted(score_by_family_totals.items())
            if values
        }
        score_by_difficulty = {
            difficulty: round(sum(values) / len(values), 2)
            for difficulty, values in sorted(score_by_difficulty_totals.items())
            if values
        }
        full_task_success_percent = round(
            sum(1 for item in self._results if item.success) / len(self._results) * 100.0,
            2,
        ) if self._results else 0.0
        family_group_average_percent = round(
            sum(score_by_family.values()) / len(score_by_family),
            2,
        ) if score_by_family else 0.0
        difficulty_group_average_percent = round(
            sum(score_by_difficulty.values()) / len(score_by_difficulty),
            2,
        ) if score_by_difficulty else 0.0
        group_average_components = [
            value
            for value in (difficulty_group_average_percent, family_group_average_percent)
            if value > 0.0
        ]
        group_average_percent = round(
            sum(group_average_components) / len(group_average_components),
            2,
        ) if group_average_components else 0.0
        summary = BenchmarkSummary(
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_tasks=len(self._results),
            successful_tasks=sum(1 for item in self._results if item.success),
            failed_tasks=sum(1 for item in self._results if not item.success),
            false_positives=false_positives,
            success_rate_by_type=success_rate_by_type,
            failure_breakdown=dict(sorted(failure_breakdown.items())),
            score_by_family=score_by_family,
            score_by_difficulty=score_by_difficulty,
            full_task_success_percent=full_task_success_percent,
            family_group_average_percent=family_group_average_percent,
            difficulty_group_average_percent=difficulty_group_average_percent,
            group_average_percent=group_average_percent,
            average_task_score_percent=round(
                sum(float(item.score_percent) for item in self._results) / len(self._results),
                2,
            ) if self._results else 0.0,
        )
        aggregate_metrics = compute_benchmark_metrics(self._results)
        ranked_hints = [
            item[0]
            for item in sorted(
                ((hint, hints.count(hint)) for hint in set(hints)),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        return BenchmarkRunReport(
            summary=summary,
            aggregate_metrics=aggregate_metrics,
            improvement_hints=ranked_hints,
            tasks=list(self._results),
        )

    def write(self, output_dir: Path, *, prefix: str = "benchmark", run_metadata: dict[str, Any] | None = None) -> BenchmarkRunReport:
        os.makedirs(output_dir, exist_ok=True)
        report = self.build_report()
        results_path = output_dir / f"{prefix}_results.json"
        report_path = output_dir / f"{prefix}_report.md"
        seed_cache_mode_counts, task_cache_mode_counts = _cache_transparency_summary(report.tasks)
        report.run_metadata = {
            **({} if run_metadata is None else dict(run_metadata)),
            "results_path": str(results_path),
            "report_path": str(report_path),
            "seed_cache_mode_counts": seed_cache_mode_counts,
            "task_cache_mode_counts": task_cache_mode_counts,
        }
        results_fd = os.open(results_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(results_fd, (stable_json_dumps(asdict(report), indent=2) + "\n").encode("utf-8"))
        finally:
            os.close(results_fd)
        report_fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(report_fd, render_benchmark_report(report).encode("utf-8"))
        finally:
            os.close(report_fd)
        return report
