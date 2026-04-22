from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from swaag.test_categories import (
    build_agent_tests_command,
    build_code_correctness_command,
    project_root,
)

# Public authoritative choices exposed to users.
ALL_CHOICES = ["code-correctness", "agent-tests", "combined"]


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
            "Run an explicit SWAAG test category.\n\n"
            "Authoritative two-category commands:\n"
            "  code-correctness  — run only deterministic code-correctness tests (explicit file list)\n"
            "  agent-tests       — run only cached agent tests (explicit file list)\n"
            "  combined          — code-correctness first; agent-tests only if code-correctness is green\n\n"
            "Manual validation is explicit real-model usage, not a test category.\n"
            "Use: python -m swaag.manual_validation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("profile", choices=ALL_CHOICES)
    parser.add_argument("--dry-run", action="store_true", help="Only print the command.")
    parser.add_argument("--baseline", action="store_true", help=argparse.SUPPRESS)
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
        print("$", " ".join(command))
        if args.dry_run:
            return 0
        result = subprocess.run(command, cwd=root)
        return result.returncode

    if args.profile == "combined":
        return _run_combined(root=root, dry_run=args.dry_run)

    raise SystemExit(f"Unhandled profile: {args.profile}")


if __name__ == "__main__":
    raise SystemExit(main())
