from __future__ import annotations

from swaag.benchmark.metrics import compute_benchmark_metrics
from swaag.benchmark.result_collector import BenchmarkTaskResult


def _result(*, task_id: str, task_type: str, difficulty: str, success: bool, false_positive: bool, expected_outcome: str, failure_category: str | None = None, verification_failures: int = 0, verification_types: dict[str, int] | None = None, quality_passed: bool = True, quality_checks: dict[str, bool] | None = None) -> BenchmarkTaskResult:
    return BenchmarkTaskResult(
        task_id=task_id,
        task_type=task_type,
        difficulty=difficulty,
        tags=[task_type],
        description=task_id,
        expected_outcome=expected_outcome,
        success=success,
        false_positive=false_positive,
        session_id=None,
        assistant_text="ok",
        deterministic_verification_passed=success and not false_positive,
        verification_summary={"checks": {"command": success}, "evidence": {}, "reason": "ok" if success else "failed"},
        quality_summary={
            "passed": quality_passed,
            "checks": quality_checks or {"task_type": quality_passed},
            "evidence": {"decision": {"expand_task": False}},
            "oracle": {"expand_task": False},
        },
        metrics={
            "action_count": 3,
            "retries": 1 if not success else 0,
            "replans": 1 if not success else 0,
            "no_progress_stops": 0,
            "tool_failures": 0,
            "llm_fallback_rate": 0.0,
            "verification_failures": verification_failures,
            "verification_type_distribution": verification_types or {"composite": 1},
            "stop_reason_counts": {"answered": 1 if success else 0},
        },
        failure_category=failure_category,
        failure_reason=failure_category,
        failure_subsystem="evaluator" if failure_category else None,
        wall_clock_seconds=1.5,
        improvement_hints=[],
        history_path=None,
        workspace="/tmp/work",
    )


def test_compute_benchmark_metrics_tracks_false_positives_and_understanding() -> None:
    metrics = compute_benchmark_metrics(
        [
            _result(task_id="coding_ok", task_type="coding", difficulty="easy", success=True, false_positive=False, expected_outcome="success"),
            _result(task_id="reading_fp", task_type="reading", difficulty="normal", success=False, false_positive=True, expected_outcome="success", failure_category="evaluator_mistake", verification_failures=0),
            _result(task_id="failure_ok", task_type="failure", difficulty="hard", success=True, false_positive=False, expected_outcome="expected_failure", failure_category=None),
        ]
    )

    assert metrics.primary["task_success_rate"] == 2 / 3
    assert metrics.primary["false_positive_rate"] == 1 / 3
    assert metrics.primary["correct_task_classification_rate"] == 1.0
    assert "environment_usage_correctness" in metrics.benchmark_specific
    assert metrics.secondary["benchmark_coverage_by_task_type"]["coding"] == 1
    assert metrics.coverage_by_difficulty["hard"] == 1
    assert metrics.failure_breakdown["evaluator_mistake"] == 1
