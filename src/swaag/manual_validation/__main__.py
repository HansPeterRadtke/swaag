"""Manual validation CLI — explicit real-model validation, not a test category.

Usage:
    python -m swaag.manual_validation [options]

Run the curated manual-validation subset against a live llama.cpp server.
This is NOT a test category — it requires a running model server and produces
real-usage artifacts, not test results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from swaag.live_runtime_profiles import get_documented_final_live_benchmark_recommendation


def _parse_seed_list(raw: str | None, *, default: tuple[int, int, int]) -> list[int]:
    if raw is None or not raw.strip():
        return list(default)
    seeds: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        seeds.append(int(value))
    return seeds or list(default)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m swaag.manual_validation",
        description=(
            "Run explicit real-model validation against a live llama.cpp server. "
            "This is NOT a test category."
        ),
    )
    parser.set_defaults(validation_subset=True)
    parser.add_argument("--output", default="manual_validation_output", help="Output directory for manual validation artifacts.")
    parser.add_argument("--clean", action="store_true", help="Delete the output directory before running.")
    parser.add_argument("--task", action="append", default=[], help="Run only the named validation task id. Can be passed multiple times.")
    parser.add_argument("--validation-subset", dest="validation_subset", action="store_true", help="Run the curated manual-validation subset (default).")
    parser.add_argument("--full-catalog", dest="validation_subset", action="store_false", help="Run the full benchmark catalog instead of the curated subset.")
    parser.add_argument("--model-base-url", help="Override the llama.cpp base URL.")
    parser.add_argument("--timeout-seconds", type=int, help="Override the model read timeout.")
    parser.add_argument("--connect-timeout-seconds", type=int, help="Override the model connect timeout.")
    parser.add_argument("--model-profile", help="Record the llama.cpp profile used for validation.")
    parser.add_argument("--structured-output-mode", choices=["server_schema", "post_validate", "auto"], help="Override structured output mode.")
    parser.add_argument("--progress-poll-seconds", type=float, help="Override model progress polling interval.")
    parser.add_argument("--seeds", help="Comma-separated fixed seeds.")
    parser.add_argument("--json", action="store_true", help="Print the full result JSON.")
    args = parser.parse_args(argv)

    from swaag.manual_validation.runner import run_manual_validation
    from swaag.utils import stable_json_dumps

    recommendation = get_documented_final_live_benchmark_recommendation()
    seeds = _parse_seed_list(args.seeds, default=recommendation.seeds) if args.seeds else None

    report = run_manual_validation(
        output_dir=Path(args.output),
        clean=bool(args.clean),
        benchmark_task_ids=list(args.task) or None,
        validation_subset=bool(args.validation_subset),
        model_base_url=args.model_base_url,
        timeout_seconds=args.timeout_seconds,
        connect_timeout_seconds=args.connect_timeout_seconds,
        model_profile=args.model_profile,
        structured_output_mode=args.structured_output_mode,
        progress_poll_seconds=args.progress_poll_seconds,
        seeds=seeds,
    )
    if args.json:
        print(stable_json_dumps(report, indent=2))
    else:
        print("manual_validation_not_test_category=true")
        print(f"percent={report['percent']}")
        print(f"task_count={report['summary']['total_tasks']}")
    return 0 if report["summary"].get("failed_tasks", 0) == 0 and report["summary"].get("false_positives", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
