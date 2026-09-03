from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from swaag.benchmark.compaction_preservation import run_compaction_preservation_benchmark
from swaag.benchmark.context_engineering import run_context_engineering_benchmark
from swaag.config import AgentConfig, load_config
from swaag.redaction import configured_secret_values, redact_for_persistence
from swaag.runtime import AgentRuntime
from swaag.utils import stable_json_dumps


BENCHMARK_VERSION = 1


def _atomic_report(path: Path, payload: dict[str, Any], *, config: AgentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    sanitized = redact_for_persistence(payload, secret_values=configured_secret_values(config))
    raw = stable_json_dumps(sanitized, indent=2) + "\n"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, raw.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _overflow_pass(result: dict[str, Any]) -> bool:
    verification = result.get("verification", {})
    checks = verification.get("checks", {}) if isinstance(verification, dict) else {}
    required = (
        "candidate_overflow_measured",
        "semantic_projection_used",
        "projection_lineage_matches_source",
        "raw_source_recoverable",
        "required_facts_preserved",
        "final_request_fits",
    )
    return bool(verification.get("passed")) and all(bool(checks.get(name)) for name in required)


def run_long_horizon_context_benchmark(
    *,
    output_dir: Path,
    config: AgentConfig | None = None,
    cycles: int = 12,
    overflow_trials: int = 3,
    clean: bool = False,
    compaction_model_client: object | None = None,
    context_runtime_factory: Callable[[AgentConfig], AgentRuntime] = AgentRuntime,
    context_model_identity: Any | None = None,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    if overflow_trials <= 0:
        raise ValueError("overflow_trials must be positive")
    base = deepcopy(config or load_config())
    output_dir = output_dir.expanduser().resolve()
    if clean and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "long_horizon_context_results.json"

    signature = {
        "version": BENCHMARK_VERSION,
        "cycles": int(cycles),
        "overflow_trials": int(overflow_trials),
        "context_limit": int(base.model.context_limit),
        "model_base_url": str(base.model.base_url),
        "structured_output_mode": str(base.model.structured_output_mode),
    }
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError(
                "Long-horizon context checkpoint does not match the current benchmark configuration"
            )
        if previous.get("complete") is True:
            return previous

    compaction_path = output_dir / "compaction_preservation.json"
    compaction = run_compaction_preservation_benchmark(
        config=base,
        cycles=cycles,
        output_path=compaction_path,
        model_client=compaction_model_client,
        resume=True,
        adversarial_conflicts=True,
        semantic_retrieval_probe=True,
    )

    overflow_reports: list[dict[str, Any]] = []
    for trial in range(1, overflow_trials + 1):
        trial_dir = output_dir / "overflow" / f"trial-{trial:03d}"
        report = run_context_engineering_benchmark(
            output_dir=trial_dir,
            config=base,
            case_ids=["measured_overflow_projection"],
            clean=False,
            runtime_factory=context_runtime_factory,
            model_identity=context_model_identity,
        )
        overflow_reports.append(report)
        partial = {
            "benchmark": "long_horizon_context",
            "signature": signature,
            "complete": False,
            "compaction": compaction,
            "overflow_trials_completed": len(overflow_reports),
            "overflow_reports": overflow_reports,
        }
        _atomic_report(report_path, partial, config=base)

    compaction_rows = list(compaction.get("results", []))
    exact_preservation_passed = sum(bool(row.get("passed")) for row in compaction_rows)
    provenance_passed = sum(
        int(row.get("source_reference_count", 0)) > 0
        and int(row.get("required_recovery_tokens", 0)) > 0
        and int(row.get("actual_recovered_tokens", 0)) > 0
        for row in compaction_rows
    )
    semantic_retrieval_passed = sum(
        bool(row.get("semantic_retrieval_passed")) for row in compaction_rows
    )
    adversarial_resistance_passed = sum(
        bool(row.get("passed")) and bool(row.get("semantic_retrieval_passed"))
        for row in compaction_rows
    )
    overflow_rows = [
        result
        for report in overflow_reports
        for result in report.get("results", [])
        if result.get("case_id") == "measured_overflow_projection"
    ]
    overflow_passed = sum(_overflow_pass(row) for row in overflow_rows)
    complete = (
        bool(compaction.get("complete"))
        and len(overflow_reports) == overflow_trials
        and all(report.get("complete") is True for report in overflow_reports)
    )
    all_dimensions_passed = bool(
        complete
        and exact_preservation_passed == cycles
        and provenance_passed == cycles
        and semantic_retrieval_passed == cycles
        and adversarial_resistance_passed == cycles
        and overflow_passed == overflow_trials
    )
    aggregate = {
        "benchmark": "long_horizon_context",
        "signature": signature,
        "complete": complete,
        "all_dimensions_passed": all_dimensions_passed,
        "measurement_scope": {
            "exact_preservation": "Exact authoritative values survive repeated semantic compaction.",
            "provenance_recoverability": "Compacted state retains source references and enough recovery evidence to reconstruct durable originals.",
            "semantic_retrieval": "A separate constrained model probe recovers the exact authoritative values after every compaction cycle.",
            "adversarial_conflict_resistance": "Later explicitly untrusted contradictory values do not displace authoritative facts in semantic retrieval.",
            "measured_overflow_projection": "Independent trials prove actual context overflow, semantic projection, exact lineage, raw-source recovery, preserved required facts, and final fit.",
        },
        "dimensions": {
            "exact_preservation": {"passed": exact_preservation_passed, "total": cycles},
            "provenance_recoverability": {"passed": provenance_passed, "total": cycles},
            "semantic_retrieval": {"passed": semantic_retrieval_passed, "total": cycles},
            "adversarial_conflict_resistance": {
                "passed": adversarial_resistance_passed,
                "total": cycles,
            },
            "measured_overflow_projection": {"passed": overflow_passed, "total": overflow_trials},
        },
        "compaction": compaction,
        "overflow_trials_completed": len(overflow_reports),
        "overflow_reports": overflow_reports,
    }
    _atomic_report(report_path, aggregate, config=base)
    return aggregate
