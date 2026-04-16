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
    lines.extend(
        [
            "## Summary",
            "",
            f"- Total tasks: `{summary.total_tasks}`",
            f"- Successful tasks: `{summary.successful_tasks}`",
            f"- Failed tasks: `{summary.failed_tasks}`",
            f"- False positives: `{summary.false_positives}`",
            "",
            "## Success Rates By Task Type",
            "",
        ]
    )
    for task_type, rate in summary.success_rate_by_type.items():
        lines.append(f"- `{task_type}`: `{rate:.2%}`")
    lines.append("")
    lines.extend(_kv_lines("Success Rates By Difficulty", metrics.success_by_difficulty))
    lines.extend(_kv_lines("Primary Metrics", metrics.primary))
    lines.extend(_kv_lines("Secondary Metrics", metrics.secondary))
    lines.extend(_kv_lines("Prompt Understanding Metrics", metrics.understanding))
    lines.extend(_kv_lines("Benchmark-Specific Metrics", metrics.benchmark_specific))
    lines.extend(_kv_lines("Failure Breakdown", metrics.failure_breakdown))
    lines.extend(_kv_lines("Subsystem Failure Breakdown", metrics.subsystem_failure_breakdown))
    lines.extend(_kv_lines("Verifier Weakness Breakdown", metrics.verifier_weakness_breakdown))
    lines.extend(_kv_lines("Prompt Understanding Mistakes", metrics.prompt_understanding_mistakes))
    lines.extend(_kv_lines("Planning Mistakes", metrics.planning_mistakes))
    lines.extend(_kv_lines("Verification Type Success", {key: value for key, value in metrics.per_verification_type_success.items()}))
    lines.extend(_kv_lines("Stop Reasons", metrics.stop_reason_counts))
    lines.extend(_kv_lines("Coverage By Task Type", metrics.coverage_by_type))
    lines.extend(_kv_lines("Coverage By Difficulty", metrics.coverage_by_difficulty))
    lines.extend(_kv_lines("Per-Seed Success", metrics.benchmark_specific.get("seed_success_by_seed", {})))
    lines.extend(_kv_lines("Per-Seed False Positives", metrics.benchmark_specific.get("seed_false_positive_by_seed", {})))

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

    lines.extend(["## Worst Failures", ""])
    failures = [item for item in report.tasks if not item.success][:5]
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
        item
        for item in report.tasks
        if item.metrics.get("retrieval_trace_sample")
        or item.metrics.get("guidance_sources")
        or item.metrics.get("selected_skill_ids")
        or item.metrics.get("subagent_usage")
    ][:5]
    lines.extend(["## Trace Samples", ""])
    if traceful_tasks:
        for item in traceful_tasks:
            lines.extend(
                [
                    f"### {item.task_id}",
                    f"- Retrieval mode: `{item.metrics.get('retrieval_mode', '')}`",
                    f"- Retrieval degraded: `{item.metrics.get('retrieval_degraded', False)}`",
                    f"- Guidance sources: `{item.metrics.get('guidance_sources', [])}`",
                    f"- Selected skills: `{item.metrics.get('selected_skill_ids', [])}`",
                    f"- Exposed tools: `{item.metrics.get('exposed_tool_names', [])}`",
                    f"- Subagents: `{item.metrics.get('subagent_usage', [])}`",
                    f"- Verification trace: `{item.metrics.get('verification_trace', {})}`",
                    "",
                ]
            )
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
