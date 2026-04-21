from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from swaag.finalproof import build_finalproof_environment
from swaag.test_categories import (
    build_agent_tests_command,
    build_code_correctness_command,
    build_lane_command,
    project_root,
)

# Public authoritative choices exposed to users.
_USER_CHOICES = ["code-correctness", "agent-tests", "combined"]

# Internal devcheck profiles (kept for backward compatibility and devcheck CLI use).
_INTERNAL_CHOICES = ["fast", "system", "integration", "live", "benchmark_heavy", "proof"]

ALL_CHOICES = _USER_CHOICES + _INTERNAL_CHOICES


def _run_combined(*, root: Path, dry_run: bool) -> int:
    """Run code-correctness first; run agent-tests only if code-correctness is fully green."""
    cc_command = build_code_correctness_command(root=root)
    print("$ code-correctness:", " ".join(cc_command))
    if dry_run:
        at_command = build_agent_tests_command(root=root)
        print("$ agent-tests (runs only if code-correctness passes):", " ".join(at_command))
        return 0
    result = subprocess.run(cc_command, cwd=root)
    if result.returncode != 0:
        print("error=code-correctness failed; agent-tests will not run")
        return result.returncode
    at_command = build_agent_tests_command(root=root)
    print("$ agent-tests:", " ".join(at_command))
    result2 = subprocess.run(at_command, cwd=root)
    return result2.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an explicit test category or devcheck profile.\n\n"
            "Authoritative two-category commands:\n"
            "  code-correctness  — run only deterministic code-correctness tests (explicit file list)\n"
            "  agent-tests       — run only agent behavior tests (explicit file list)\n"
            "  combined          — code-correctness first; agent-tests only if code-correctness is green\n\n"
            "Internal devcheck profiles (fast/system/integration/live/benchmark_heavy/proof) are kept\n"
            "for the incremental devcheck inner loop and are not the authoritative test interface."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("profile", choices=ALL_CHOICES)
    parser.add_argument("--dry-run", action="store_true", help="Only print the command.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="For fast/system profiles, build a pytest-testmon baseline instead of selecting only affected tests.",
    )
    args = parser.parse_args(argv)
    root = project_root()

    if args.profile == "code-correctness":
        command = build_code_correctness_command(root=root)
        print("$", " ".join(command))
        if args.dry_run:
            return 0
        result = subprocess.run(command, cwd=root)
        return result.returncode

    if args.profile == "agent-tests":
        command = build_agent_tests_command(root=root)
        env = os.environ.copy()
        env.update(build_finalproof_environment())
        print("$", " ".join(command))
        if args.dry_run:
            return 0
        result = subprocess.run(command, cwd=root, env=env)
        return result.returncode

    if args.profile == "combined":
        return _run_combined(root=root, dry_run=args.dry_run)

    # Internal devcheck profiles
    use_testmon = args.profile in {"fast", "system"}
    command = build_lane_command(args.profile, root=root, use_testmon=use_testmon, baseline_only=args.baseline)
    env = os.environ.copy()
    if args.profile in {"live", "proof"}:
        env.update(build_finalproof_environment())
    print("$", " ".join(command))
    if args.dry_run:
        return 0
    subprocess.run(command, cwd=Path(root), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
