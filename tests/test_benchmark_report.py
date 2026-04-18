from __future__ import annotations

from swaag.benchmark.report import render_benchmark_report
from swaag.benchmark.result_collector import BenchmarkRunReport, BenchmarkSummary, BenchmarkTaskResult
from swaag.benchmark.metrics import BenchmarkAggregateMetrics


def test_render_benchmark_report_includes_quality_sections() -> None:
    report = BenchmarkRunReport(
        summary=BenchmarkSummary(
            generated_at="2026-01-01T00:00:00+00:00",
            total_tasks=1,
            successful_tasks=1,
            failed_tasks=0,
            false_positives=0,
            success_rate_by_type={"coding": 1.0},
            failure_breakdown={},
            score_by_difficulty={"extremely_easy": 100.0},
            average_task_score_percent=100.0,
        ),
        aggregate_metrics=BenchmarkAggregateMetrics(
            primary={"task_success_rate": 1.0, "false_positive_rate": 0.0},
            secondary={"llm_fallback_rate": 0.0, "average_wall_clock_seconds_per_task": 1.0},
            understanding={"understanding_success_rate": 1.0},
            benchmark_specific={
                "environment_usage_correctness": 1.0,
                "seed_success_by_seed": {"11": 1.0, "23": 1.0, "37": 1.0},
                "seed_false_positive_by_seed": {"11": 0.0, "23": 0.0, "37": 0.0},
            },
            success_by_type={"coding": 1.0},
            success_by_difficulty={"extremely_easy": 1.0},
            coverage_by_type={"coding": 1},
            coverage_by_difficulty={"extremely_easy": 1},
            per_verification_type_success={"composite": {"tasks": 1, "successful_tasks": 1}},
            stop_reason_counts={"answered": 1},
            failure_breakdown={},
            subsystem_failure_breakdown={},
            verifier_weakness_breakdown={},
            prompt_understanding_mistakes={},
            planning_mistakes={},
            improvement_priorities=[{"name": "evaluator_mistake", "count": 1, "kind": "failure_class"}],
        ),
        run_metadata={"mode": "full", "wall_clock_seconds": 1.0},
        improvement_hints=["Tighten evaluator evidence thresholds."],
        tasks=[
            BenchmarkTaskResult(
                task_id="coding_ok",
                task_type="coding",
                difficulty="extremely_easy",
                tags=["coding"],
                description="ok",
                expected_outcome="success",
                success=True,
                false_positive=False,
                session_id="session_x",
                assistant_text="done",
                deterministic_verification_passed=True,
                verification_summary={"checks": {"command": True}, "evidence": {}, "reason": "ok"},
                quality_summary={"passed": True, "checks": {}, "evidence": {}, "oracle": {}},
                metrics={
                    "action_count": 2,
                    "retrieval_mode": "semantic_tfidf",
                    "retrieval_degraded": False,
                    "guidance_sources": ["repo:AGENTS.md"],
                    "selected_skill_ids": ["coding_patch"],
                    "exposed_tool_names": ["read_file", "run_tests"],
                    "subagent_usage": ["reviewer"],
                    "verification_trace": {"verification_type_used": "execution", "conditions_met": ["command"], "conditions_failed": [], "reason": "ok"},
                },
                failure_category=None,
                failure_reason=None,
                failure_subsystem=None,
                wall_clock_seconds=1.0,
                improvement_hints=[],
                history_path="/tmp/history.jsonl",
                workspace="/tmp/work",
                score_percent=100.0,
                rubric_breakdown={
                    "final_outcome": {"weight": 50.0, "earned": 50.0, "percent": 100.0},
                    "verification_contract": {"weight": 30.0, "earned": 30.0, "percent": 100.0},
                    "quality_and_planning": {"weight": 20.0, "earned": 20.0, "percent": 100.0},
                },
            )
        ],
        score_weights={"final_outcome": 50.0, "verification_contract": 30.0, "quality_and_planning": 20.0},
    )

    text = render_benchmark_report(report)

    assert "Primary Metrics" in text
    assert "Prompt Understanding Metrics" in text
    assert "Benchmark-Specific Metrics" in text
    assert "Run Metadata" in text
    assert "False Positive Analysis" in text
    assert "Recommended Next Fixes" in text
    assert "evaluator_mistake" in text
    assert "Per-Seed Success" in text
    assert "Trace Samples" in text
    assert "coding_patch" in text
    assert "Score By Difficulty Tier" in text
    assert "Per-Task Scores" in text
