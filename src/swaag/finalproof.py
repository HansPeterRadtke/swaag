from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from swaag.live_runtime_profiles import get_documented_final_live_benchmark_recommendation


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_finalproof_commands(*, benchmark_output: Path, live_benchmark_output: Path) -> list[list[str]]:
    live = get_documented_final_live_benchmark_recommendation()
    return [
        [sys.executable, "-m", "pytest", "-q", "tests/test_imports.py"],
        [sys.executable, "-m", "pytest", "-q", "tests/test_scaled_catalog.py"],
        [sys.executable, "-m", "pytest", "-q", "tests/test_runtime_verification_flow.py"],
        [sys.executable, "-m", "pytest", "-q", "tests/test_end_to_end.py"],
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "-m", "swaag.testlane", "integration"],
        [sys.executable, "-m", "swaag.testlane", "live"],
        [sys.executable, "-m", "swaag.benchmark", "run", "--clean", "--output", str(benchmark_output), "--json"],
        [
            sys.executable,
            "-m",
            "swaag.benchmark",
            "run",
            "--clean",
            "--live-subset",
            "--model-profile",
            live.model_profile,
            "--structured-output-mode",
            live.structured_output_mode,
            "--seeds",
            ",".join(str(seed) for seed in live.seeds),
            "--timeout-seconds",
            str(live.timeout_seconds),
            "--output",
            str(live_benchmark_output),
            "--json",
        ],
        [sys.executable, "scripts/archive_proof.py"],
    ]


def build_finalproof_environment() -> dict[str, str]:
    live = get_documented_final_live_benchmark_recommendation()
    return {
        "SWAAG_RUN_LIVE": "1",
        "SWAAG_LIVE_MODEL_PROFILE": live.model_profile,
        "SWAAG_LIVE_STRUCTURED_OUTPUT_MODE": live.structured_output_mode,
        "SWAAG_LIVE_SEEDS": ",".join(str(seed) for seed in live.seeds),
        "SWAAG_LIVE_TIMEOUT_SECONDS": str(live.timeout_seconds),
        "SWAAG_LIVE_CONNECT_TIMEOUT_SECONDS": str(live.connect_timeout_seconds),
        "SWAAG_LIVE_PROGRESS_POLL_SECONDS": str(live.progress_poll_seconds),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full final proof loop.")
    parser.add_argument("--benchmark-output", default="/tmp/swaag-benchmark-finalproof", help="Output directory for the large benchmark.")
    parser.add_argument("--live-benchmark-output", default="/tmp/swaag-live-benchmark-finalproof", help="Output directory for the live benchmark subset.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands.")
    args = parser.parse_args(argv)

    commands = build_finalproof_commands(
        benchmark_output=Path(args.benchmark_output),
        live_benchmark_output=Path(args.live_benchmark_output),
    )
    env = os.environ.copy()
    live_env = build_finalproof_environment()
    for key, value in live_env.items():
        env.setdefault(key, value)
    live = get_documented_final_live_benchmark_recommendation()
    print(
        "# finalproof_live_settings",
        f"profile={live.model_profile}",
        f"structured_output_mode={live.structured_output_mode}",
        f"seeds={','.join(str(seed) for seed in live.seeds)}",
        f"timeout_seconds={live.timeout_seconds}",
        f"connect_timeout_seconds={live.connect_timeout_seconds}",
        f"progress_poll_seconds={live.progress_poll_seconds}",
    )
    for command in commands:
        print("$", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=_project_root(), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
