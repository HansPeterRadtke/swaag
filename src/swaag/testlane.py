from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from swaag.finalproof import build_finalproof_environment
from swaag.testlanes import build_lane_command, project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an explicit test lane.")
    parser.add_argument("lane", choices=["fast", "system", "integration", "live", "benchmark_heavy", "proof"])
    parser.add_argument("--dry-run", action="store_true", help="Only print the command.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="For fast/system lanes, build a pytest-testmon baseline instead of selecting only affected tests.",
    )
    args = parser.parse_args(argv)

    use_testmon = args.lane in {"fast", "system"}
    command = build_lane_command(args.lane, root=project_root(), use_testmon=use_testmon, baseline_only=args.baseline)
    env = os.environ.copy()
    if args.lane in {"live", "proof"}:
        env.update(build_finalproof_environment())
    print("$", " ".join(command))
    if args.dry_run:
        return 0
    subprocess.run(command, cwd=Path(project_root()), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
