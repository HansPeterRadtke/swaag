from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass(slots=True)
class BenchmarkAggregateMetrics:
    primary: dict[str, Any]
    secondary: dict[str, Any]
    understanding: dict[str, Any]
    benchmark_specific: dict[str, Any]
    success_by_type: dict[str, float]
    success_by_difficulty: dict[str, float]
    coverage_by_type: dict[str, int]
    coverage_by_difficulty: dict[str, int]
    per_verification_type_success: dict[str, dict[str, int]]
    stop_reason_counts: dict[str, int]
    failure_breakdown: dict[str, int]
    subsystem_failure_breakdown: dict[str, int]
    verifier_weakness_breakdown: dict[str, int]
    prompt_understanding_mistakes: dict[str, int]
    planning_mistakes: dict[str, int]
    improvement_priorities: list[dict[str, Any]] = field(default_factory=list)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _mean_metric(results: list[Any], key: str) -> float:
    values = [float(result.metrics.get(key, 0.0)) for result in results]
    return mean(values) if values else 0.0


def compute_benchmark_metrics(results: list[Any]) -> BenchmarkAggregateMetrics:
    total = len(results)
    expected_success = [item for item in results if item.expected_outcome == "success"]
    expected_failure = [item for item in results if item.expected_outcome != "success"]
    successful = [item for item in results if item.success]
    failed = [item for item in results if not item.success]
    false_positives = [item for item in results if item.false_positive]
    false_negatives = [item for item in expected_success if not item.success]
    partial_successes = [
        item
        for item in expected_success
        if not item.success and item.verification_summary.get("checks") and any(item.verification_summary["checks"].values())
    ]
    verifier_blocks = [item for item in failed if item.metrics.get("verification_failures", 0) > 0]
    verifier_escapes = [item for item in false_positives if item.deterministic_verification_passed is False]

    success_by_type_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    success_by_difficulty_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    coverage_by_type: Counter[str] = Counter()
    coverage_by_difficulty: Counter[str] = Counter()
    failure_breakdown: Counter[str] = Counter()
    subsystem_failure_breakdown: Counter[str] = Counter()
    verifier_weakness_breakdown: Counter[str] = Counter()
    prompt_understanding_mistakes: Counter[str] = Counter()
    planning_mistakes: Counter[str] = Counter()
    per_verification_type_success: dict[str, dict[str, int]] = defaultdict(lambda: {"tasks": 0, "successful_tasks": 0})
    stop_reason_counts: Counter[str] = Counter()

    understanding_tasks = [item for item in results if item.quality_summary.get("checks")]
    understanding_passed = [item for item in understanding_tasks if item.quality_summary.get("passed")]
    classification_hits = 0
    classification_total = 0
    decomposition_hits = 0
    decomposition_total = 0
    expansion_tp = 0
    expansion_predicted = 0
    expansion_expected = 0
    unnecessary_expansion = 0
    missed_expansion = 0

    for result in results:
        coverage_by_type[result.task_type] += 1
        coverage_by_difficulty[result.difficulty] += 1
        success_by_type_counts[result.task_type][0] += int(result.success)
        success_by_type_counts[result.task_type][1] += 1
        success_by_difficulty_counts[result.difficulty][0] += int(result.success)
        success_by_difficulty_counts[result.difficulty][1] += 1
        if result.failure_category:
            failure_breakdown[result.failure_category] += 1
        if result.failure_subsystem:
            subsystem_failure_breakdown[result.failure_subsystem] += 1
        for check_name, passed in result.verification_summary.get("checks", {}).items():
            if not passed:
                verifier_weakness_breakdown[check_name] += 1
        for check_name, passed in result.quality_summary.get("checks", {}).items():
            if not passed:
                prompt_understanding_mistakes[check_name] += 1
                if "strategy" in check_name or "split_task" in check_name:
                    planning_mistakes[check_name] += 1
        for verification_type, count in result.metrics.get("verification_type_distribution", {}).items():
            per_verification_type_success[verification_type]["tasks"] += count
            if result.success:
                per_verification_type_success[verification_type]["successful_tasks"] += count
        for stop_reason, count in result.metrics.get("stop_reason_counts", {}).items():
            stop_reason_counts[stop_reason] += int(count)

        quality_checks = result.quality_summary.get("checks", {})
        if "task_type" in quality_checks:
            classification_total += 1
            classification_hits += int(bool(quality_checks["task_type"]))
        if "requires_decomposition" in quality_checks:
            decomposition_total += 1
            decomposition_hits += int(bool(quality_checks["requires_decomposition"]))
        predicted_expand = result.quality_summary.get("evidence", {}).get("decision", {}).get("expand_task")
        expected_expand = result.quality_summary.get("oracle", {}).get("expand_task")
        if expected_expand is not None:
            expansion_expected += int(bool(expected_expand))
        if predicted_expand is not None:
            expansion_predicted += int(bool(predicted_expand))
        if predicted_expand is True and expected_expand is True:
            expansion_tp += 1
        if predicted_expand is True and expected_expand is False:
            unnecessary_expansion += 1
        if predicted_expand is False and expected_expand is True:
            missed_expansion += 1

    primary = {
        "success_rate": _rate(len(successful), total),
        "task_success_rate": _rate(len(successful), total),
        "exact_success_rate": _rate(sum(1 for item in expected_success if item.success and not item.false_positive), len(expected_success)),
        "partial_success_rate": _rate(len(partial_successes), len(expected_success)),
        "failure_rate": _rate(len(failed), total),
        "false_positive_rate": _rate(len(false_positives), total),
        "false_negative_rate": _rate(len(false_negatives), len(expected_success)),
        "verifier_block_rate": _rate(len(verifier_blocks), total),
        "verifier_escape_rate": _rate(len(verifier_escapes), total),
        "verification_rejection_accuracy": _rate(sum(1 for item in expected_failure if item.success), len(expected_failure)),
        "prompt_understanding_accuracy": _rate(len(understanding_passed), len(understanding_tasks)),
        "correct_task_classification_rate": _rate(classification_hits, classification_total),
        "decomposition_accuracy": _rate(decomposition_hits, decomposition_total),
        "correct_decomposition_rate": _rate(decomposition_hits, decomposition_total),
        "correct_final_result_rate": _rate(sum(1 for item in expected_success if item.deterministic_verification_passed), len(expected_success)),
        "multi_step_success_rate": _rate(sum(1 for item in results if item.task_type == "multi_step" and item.success), sum(1 for item in results if item.task_type == "multi_step")),
    }

    secondary = {
        "average_action_cycles_per_task": _mean_metric(results, "action_count"),
        "retries_per_task": _mean_metric(results, "retries"),
        "replans_per_task": _mean_metric(results, "replans"),
        "drift_recoveries_per_task": _mean_metric(results, "no_progress_stops"),
        "no_progress_stops": sum(int(item.metrics.get("no_progress_stops", 0)) for item in results),
        "no_progress_rate": _rate(sum(1 for item in results if int(item.metrics.get("no_progress_stops", 0)) > 0), total),
        "average_model_progress_events_per_task": _mean_metric(results, "model_request_progress_events"),
        "model_retry_rate": _rate(sum(int(item.metrics.get("model_retry_events", 0)) for item in results), total),
        "post_validate_fallbacks": sum(int(item.metrics.get("post_validate_fallbacks", 0)) for item in results),
        "server_schema_requests": sum(int(item.metrics.get("server_schema_requests", 0)) for item in results),
        "timeout_failure_rate": _rate(
            sum(
                int(item.metrics.get("failure_counts", {}).get("TimeoutError", 0))
                + int(item.metrics.get("failure_counts", {}).get("ReadTimeout", 0))
                + int(item.metrics.get("failure_counts", {}).get("ConnectTimeout", 0))
                for item in results
            ),
            total,
        ),
        "tool_failure_rate": _rate(sum(int(item.metrics.get("tool_failures", 0)) for item in results), total),
        "evaluator_disagreement_rate": _rate(failure_breakdown.get("evaluator_mistake", 0), total),
        "llm_fallback_rate": _mean_metric(results, "llm_fallback_rate"),
        "average_wall_clock_seconds_per_task": mean([float(item.wall_clock_seconds) for item in results]) if results else 0.0,
        "total_wall_clock_seconds": sum(float(item.wall_clock_seconds) for item in results),
        "benchmark_coverage_by_task_type": dict(sorted(coverage_by_type.items())),
    }

    understanding = {
        "understanding_success_rate": _rate(len(understanding_passed), len(understanding_tasks)),
        "expansion_precision": _rate(expansion_tp, expansion_predicted),
        "decomposition_precision": _rate(decomposition_hits, decomposition_total),
        "unnecessary_expansion_rate": _rate(unnecessary_expansion, total),
        "missed_expansion_rate": _rate(missed_expansion, total),
    }

    environment_tools = {"list_files", "read_file", "write_file", "run_tests", "shell_command"}

    def _required_tools(result: Any) -> set[str]:
        evidence = result.verification_summary.get("evidence", {})
        required = evidence.get("required_tools_used", {}).get("required", [])
        return {str(item) for item in required}

    environment_tasks = [
        item
        for item in results
        if "environment" in item.tags or bool(_required_tools(item) & environment_tools)
    ]
    iterative_tasks = [
        item
        for item in results
        if "refinement" in item.tags or int(item.metrics.get("tool_calls", 0)) > 1
    ]
    recovery_tasks = [
        item
        for item in results
        if "refinement" in item.tags
        or "recovery" in item.tags
        or int(item.metrics.get("retries", 0)) > 0
        or int(item.metrics.get("replans", 0)) > 0
    ]
    seed_runs = [
        seed_result
        for item in results
        for seed_result in item.metrics.get("seed_results", [])
        if isinstance(seed_result, dict)
    ]
    seed_success_by_seed: Counter[str] = Counter()
    seed_false_positive_by_seed: Counter[str] = Counter()
    seed_total_by_seed: Counter[str] = Counter()
    for seed_result in seed_runs:
        seed_key = str(seed_result.get("seed"))
        seed_total_by_seed[seed_key] += 1
        if seed_result.get("success"):
            seed_success_by_seed[seed_key] += 1
        if seed_result.get("false_positive"):
            seed_false_positive_by_seed[seed_key] += 1
    benchmark_specific = {
        "environment_usage_correctness": _rate(sum(1 for item in environment_tasks if item.success), len(environment_tasks)),
        "iteration_improvement_rate": _rate(sum(1 for item in iterative_tasks if item.success), len(iterative_tasks)),
        "recovery_success_rate": _rate(sum(1 for item in recovery_tasks if item.success), len(recovery_tasks)),
        "tasks_with_environment_usage": len(environment_tasks),
        "tasks_with_iteration": len(iterative_tasks),
        "tasks_with_recovery": len(recovery_tasks),
        "slowest_task_wall_clock_seconds": max((float(item.wall_clock_seconds) for item in results), default=0.0),
        "post_validate_fallback_count": sum(int(item.metrics.get("post_validate_fallbacks", 0)) for item in results),
        "seed_success_rate": _rate(sum(1 for item in seed_runs if item.get("success")), len(seed_runs)),
        "seed_false_positive_rate": _rate(sum(1 for item in seed_runs if item.get("false_positive")), len(seed_runs)),
        "seed_variability_rate": _rate(sum(1 for item in results if bool(item.metrics.get("seed_variation"))), len(results)),
        "seed_run_count": len(seed_runs),
        "seed_success_by_seed": {
            seed: _rate(seed_success_by_seed[seed], seed_total_by_seed[seed])
            for seed in sorted(seed_total_by_seed)
        },
        "seed_false_positive_by_seed": {
            seed: _rate(seed_false_positive_by_seed[seed], seed_total_by_seed[seed])
            for seed in sorted(seed_total_by_seed)
        },
    }

    success_by_type = {key: _rate(value[0], value[1]) for key, value in sorted(success_by_type_counts.items())}
    success_by_difficulty = {key: _rate(value[0], value[1]) for key, value in sorted(success_by_difficulty_counts.items())}

    improvement_candidates: list[tuple[str, int, str]] = []
    for key, count in failure_breakdown.items():
        improvement_candidates.append((key, count, "failure_class"))
    for key, count in verifier_weakness_breakdown.items():
        improvement_candidates.append((key, count, "verifier_weakness"))
    for key, count in prompt_understanding_mistakes.items():
        improvement_candidates.append((key, count, "understanding_mistake"))
    improvement_priorities = [
        {"name": name, "count": count, "kind": kind}
        for name, count, kind in sorted(improvement_candidates, key=lambda item: (-item[1], item[2], item[0]))[:10]
    ]

    return BenchmarkAggregateMetrics(
        primary=primary,
        secondary=secondary,
        understanding=understanding,
        benchmark_specific=benchmark_specific,
        success_by_type=success_by_type,
        success_by_difficulty=success_by_difficulty,
        coverage_by_type=dict(sorted(coverage_by_type.items())),
        coverage_by_difficulty=dict(sorted(coverage_by_difficulty.items())),
        per_verification_type_success=dict(sorted(per_verification_type_success.items())),
        stop_reason_counts=dict(sorted(stop_reason_counts.items())),
        failure_breakdown=dict(sorted(failure_breakdown.items())),
        subsystem_failure_breakdown=dict(sorted(subsystem_failure_breakdown.items())),
        verifier_weakness_breakdown=dict(sorted(verifier_weakness_breakdown.items())),
        prompt_understanding_mistakes=dict(sorted(prompt_understanding_mistakes.items())),
        planning_mistakes=dict(sorted(planning_mistakes.items())),
        improvement_priorities=improvement_priorities,
    )
