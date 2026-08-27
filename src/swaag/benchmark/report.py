from __future__ import annotations

from typing import Any


def _kv_lines(title: str, mapping: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", ""]
    if not mapping:
        lines.append("- none")
        lines.append("")
        return lines
    for key, value in mapping.items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.2%}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return lines


def render_benchmark_report(report) -> str:
    summary = report.summary
    metrics = report.aggregate_metrics
    lines = [
        "# Benchmark Report",
        "",
        f"Generated at: `{summary.generated_at}`",
        "",
    ]
    run_metadata = getattr(report, "run_metadata", {})
    if run_metadata:
        lines.extend(["## Run Metadata", ""])
        for key, value in run_metadata.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    cache_summary = {
        "seed_cache_mode_counts": run_metadata.get("seed_cache_mode_counts", {}),
        "task_cache_mode_counts": run_metadata.get("task_cache_mode_counts", {}),
        "replay_cache_root": run_metadata.get("replay_cache_root", ""),
        "replay_cache_policy": run_metadata.get("replay_cache_policy", ""),
        "artifact_reused_from": run_metadata.get("artifact_reused_from", ""),
    }
    lines.extend(["## Cache / Replay Summary", ""])
    added_cache_line = False
    for key, value in cache_summary.items():
        if value in ({}, "", None):
            continue
        lines.append(f"- `{key}`: `{value}`")
        added_cache_line = True
    if not added_cache_line:
        lines.append("- none")
    lines.append("")
    lines.extend(
        [
            "## Summary",
            "",
            f"- Total tasks: `{summary.total_tasks}`",
            f"- Executed tasks: `{summary.executed_tasks}`",
            f"- Blocked tasks: `{summary.blocked_tasks}`",
            f"- Successful tasks: `{summary.successful_tasks}`",
            f"- Failed tasks: `{summary.failed_tasks}`",
            f"- False positives: `{summary.false_positives}`",
            f"- Full-task success: `{summary.full_task_success_percent:.2f}%`",
            f"- Group average score: `{summary.group_average_percent:.2f}%`",
            f"- Difficulty-group average: `{summary.difficulty_group_average_percent:.2f}%`",
            f"- Family-group average: `{summary.family_group_average_percent:.2f}%`",
            f"- Average task score: `{summary.average_task_score_percent:.2f}%`",
            "",
            "## Success Rates By Task Type",
            "",
        ]
    )
    for task_type, rate in summary.success_rate_by_type.items():
        lines.append(f"- `{task_type}`: `{rate:.2%}`")
    lines.append("")
    lines.extend(["## Score By Family", ""])
    if summary.score_by_family:
        for family, percent in summary.score_by_family.items():
            lines.append(f"- `{family}`: `{percent:.2f}%`")
    else:
        lines.append("- none")
    lines.append("")
    lines.extend(["## Score By Difficulty Tier", ""])
    if summary.score_by_difficulty:
        for difficulty, percent in summary.score_by_difficulty.items():
            lines.append(f"- `{difficulty}`: `{percent:.2f}%`")
    else:
        lines.append("- none")
    lines.append("")
    lines.extend(["## Task Score Weights", ""])
    for key, value in getattr(report, "score_weights", {}).items():
        lines.append(f"- `{key}`: `{value:.1f}`")
    lines.append("")
    lines.extend(_kv_lines("Success Rates By Difficulty", metrics.success_by_difficulty))
    lines.extend(_kv_lines("Primary Metrics", metrics.primary))
    lines.extend(_kv_lines("Secondary Metrics", metrics.secondary))
    lines.extend(_kv_lines("Observable Behavior Quality", metrics.behavior_quality))
    lines.extend(_kv_lines("Benchmark-Specific Metrics", metrics.benchmark_specific))
    lines.extend(_kv_lines("Failure Breakdown", metrics.failure_breakdown))
    lines.extend(_kv_lines("Subsystem Failure Breakdown", metrics.subsystem_failure_breakdown))
    lines.extend(_kv_lines("Verifier Weakness Breakdown", metrics.verifier_weakness_breakdown))
    lines.extend(_kv_lines("Behavior Quality Mistakes", metrics.behavior_quality_mistakes))
    lines.extend(_kv_lines("Verification Type Success", {key: value for key, value in metrics.per_verification_type_success.items()}))
    lines.extend(_kv_lines("Stop Reasons", metrics.stop_reason_counts))
    lines.extend(_kv_lines("Coverage By Task Type", metrics.coverage_by_type))
    lines.extend(_kv_lines("Coverage By Difficulty", metrics.coverage_by_difficulty))
    lines.extend(_kv_lines("Per-Seed Success", metrics.benchmark_specific.get("seed_success_by_seed", {})))
    lines.extend(_kv_lines("Per-Seed False Positives", metrics.benchmark_specific.get("seed_false_positive_by_seed", {})))
    lines.extend(_kv_lines("Failure Categories", summary.failure_breakdown))

    lines.extend(["## Top Failure Diagnostics", ""])
    top_failure_categories = list(summary.failure_breakdown.items())[:5]
    top_verifier_weaknesses = list(metrics.verifier_weakness_breakdown.items())[:5]
    top_behavior_quality_mistakes = list(metrics.behavior_quality_mistakes.items())[:5]
    if top_failure_categories:
        lines.append("- Failure categories:")
        for name, count in top_failure_categories:
            lines.append(f"  - `{name}`: `{count}`")
    if top_verifier_weaknesses:
        lines.append("- Verifier weaknesses:")
        for name, count in top_verifier_weaknesses:
            lines.append(f"  - `{name}`: `{count}`")
    if top_behavior_quality_mistakes:
        lines.append("- Behavior quality mistakes:")
        for name, count in top_behavior_quality_mistakes:
            lines.append(f"  - `{name}`: `{count}`")
    if not top_failure_categories and not top_verifier_weaknesses and not top_behavior_quality_mistakes:
        lines.append("- none")
    lines.append("")

    false_positives = [item for item in report.tasks if item.false_positive]
    lines.extend(["## False Positive Analysis", ""])
    if false_positives:
        for item in false_positives:
            lines.extend(
                [
                    f"### {item.task_id}",
                    f"- Type: `{item.task_type}`",
                    f"- Failure category: `{item.failure_category}`",
                    f"- Verification reason: `{item.verification_summary.get('reason')}`",
                    f"- Deterministic verification passed: `{item.deterministic_verification_passed}`",
                    "",
                ]
            )
    else:
        lines.append("- none")
        lines.append("")

    execution_blockers = [
        item for item in report.tasks if item.metrics.get("execution_blocked")
    ]
    lines.extend(["## Execution Blockers", ""])
    if execution_blockers:
        for item in execution_blockers:
            lines.extend(
                [
                    f"### {item.task_id}",
                    f"- Cache mode: `{item.metrics.get('cache_mode_summary', '')}`",
                    f"- Blockers: `{item.metrics.get('execution_blockers', [])}`",
                    "",
                ]
            )
    else:
        lines.append("- none")
        lines.append("")

    lines.extend(["## Worst Failures", ""])
    failures = [
        item
        for item in report.tasks
        if not item.success and not item.metrics.get("execution_blocked")
    ][:5]
    if failures:
        for item in failures:
            lines.extend(
                [
                    f"### {item.task_id}",
                    f"- Type: `{item.task_type}`",
                    f"- Difficulty: `{item.difficulty}`",
                    f"- Failure category: `{item.failure_category}`",
                    f"- Failure subsystem: `{item.failure_subsystem}`",
                    f"- Reason: `{item.failure_reason}`",
                    f"- Verifier reason: `{item.verification_summary.get('reason')}`",
                    f"- Quality checks passed: `{item.quality_summary.get('passed')}`",
                    "",
                ]
            )
    else:
        lines.append("- none")
        lines.append("")

    traceful_tasks = [
        item for item in report.tasks
        if item.metrics.get("model_call_count") or item.metrics.get("tool_call_count")
    ][:5]
    lines.extend(["## Model/Tool Loop Samples", ""])
    if traceful_tasks:
        for item in traceful_tasks:
            lines.extend(
                [
                    f"### {item.task_id}",
                    f"- Model calls: `{item.metrics.get('model_call_count', 0)}`",
                    f"- Model call kinds: `{item.metrics.get('model_call_kinds', {})}`",
                    f"- Context call explanations: `{len(item.metrics.get('context_call_explanations', []))}`",
                    f"- Constrained actions: `{item.metrics.get('action_count', 0)}`",
                    f"- Tool calls: `{item.metrics.get('tool_call_names', [])}`",
                    f"- Tool errors: `{item.metrics.get('tool_error_count', 0)}`",
                    f"- History compactions: `{item.metrics.get('compaction_count', 0)}`",
                    f"- Environment operations: `{item.metrics.get('environment_operations_summary', {})}`",
                    "",
                ]
            )
    else:
        lines.append("- none")
        lines.append("")

    lines.extend(["## Per-Task Scores", ""])
    if report.tasks:
        for item in report.tasks:
            lines.extend(
                [
                    f"### {item.task_id}",
                    f"- Type: `{item.task_type}`",
                    f"- Difficulty: `{item.difficulty}`",
                    f"- Score: `{item.score_percent:.2f}%`",
                    f"- Success: `{item.success}`",
                    f"- Execution blocked: `{bool(item.metrics.get('execution_blocked'))}`",
                    f"- Failure category: `{item.failure_category}`",
                    f"- Failure subsystem: `{item.failure_subsystem}`",
                    f"- Verification reason: `{item.verification_summary.get('reason')}`",
                    f"- Cache mode: `{item.metrics.get('cache_mode_summary', '')}`",
                ]
            )
            if item.rubric_breakdown:
                lines.append("- Rubric:")
                for rubric_name, rubric in item.rubric_breakdown.items():
                    lines.append(
                        f"  - `{rubric_name}`: `{float(rubric.get('earned', 0.0)):.2f}/{float(rubric.get('weight', 0.0)):.2f}` "
                        f"(`{float(rubric.get('percent', 0.0)):.2f}%`)"
                    )
            lines.append("")
    else:
        lines.append("- none")
        lines.append("")

    lines.extend(["## Repeated Failure Patterns", ""])
    if metrics.improvement_priorities:
        for item in metrics.improvement_priorities:
            lines.append(f"- `{item['kind']}` / `{item['name']}`: `{item['count']}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.extend(["## Recommended Next Fixes", ""])
    added = False
    for item in report.improvement_hints[:10]:
        lines.append(f"- {item}")
        added = True
    if not added:
        lines.append("- Maintain the current benchmark suite and increase task difficulty before broadening scope.")
    lines.append("")
    return "\n".join(lines)
