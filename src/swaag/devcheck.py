from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from swaag.test_categories import (
    DevcheckPlan,
    build_devcheck_plan,
    detect_testmon,
    project_root,
)


def _repo_parent(root: Path) -> Path:
    return root.parent


def discover_changed_files(*, project_root_override: Path | None = None) -> list[str]:
    root = project_root() if project_root_override is None else project_root_override
    parent = _repo_parent(root)
    result = subprocess.run(
        ["git", "-C", str(parent), "status", "--short", root.name],
        check=True,
        text=True,
        capture_output=True,
    )
    changed: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if path.startswith(f"{root.name}/"):
            path = path[len(root.name) + 1 :]
        changed.append(path.replace("\\", "/"))
    return sorted(dict.fromkeys(changed))


def select_test_targets(changed_files: list[str]) -> list[str]:
    return list(build_devcheck_plan(changed_files).candidate_tests)


def build_pytest_command(plan: DevcheckPlan, *, require_testmon: bool = False) -> list[str]:
    if not plan.candidate_tests:
        return [sys.executable, "-c", 'print("devcheck: no candidate tests selected")']
    command = [sys.executable, "-m", "pytest", "-q"]
    if plan.marker_expression:
        command.extend(["-m", plan.marker_expression])
    if require_testmon and not plan.testmon.available:
        raise RuntimeError("pytest-testmon is required for this run, but the plugin is unavailable")
    if plan.testmon.available:
        if plan.testmon.mode == "forceselect":
            command.append("--testmon-forceselect")
        elif plan.testmon.mode == "noselect":
            command.append("--testmon-noselect")
    command.extend(plan.candidate_tests)
    return command


def _print_plan(plan: DevcheckPlan) -> None:
    print(f"test_profile={plan.lane}")
    print(f"marker_expression={plan.marker_expression}")
    print(f"changed_files={list(plan.changed_files)}")
    print(f"candidate_tests={list(plan.candidate_tests)}")
    print(
        "testmon="
        f"available:{plan.testmon.available},"
        f"baseline:{plan.testmon.baseline_exists},"
        f"mode:{plan.testmon.mode},"
        f"reason:{plan.testmon.reason}"
    )
    if plan.reasons:
        print("selection_reasons=")
        for reason in plan.reasons:
            print(f"  - {reason}")
    if plan.explicit_followup_lanes:
        print(f"followup_profiles={list(plan.explicit_followup_lanes)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the smallest correct changed-area test subset with deterministic profile selection and pytest-testmon."
    )
    parser.add_argument("--changed-file", action="append", default=[], help="Override changed-file detection with explicit repo-relative paths.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the selected pytest command.")
    parser.add_argument("--baseline", action="store_true", help="Build the pytest-testmon baseline for the selected profile instead of selecting only affected tests.")
    parser.add_argument("--allow-live", action="store_true", help="Allow live-only changed files to route into the explicit live profile.")
    parser.add_argument(
        "--allow-benchmark-heavy",
        action="store_true",
        help="Allow benchmark-heavy changed files to route into the heavy benchmark profile.",
    )
    parser.add_argument(
        "--require-testmon",
        action="store_true",
        help="Fail instead of falling back when pytest-testmon is unavailable.",
    )
    args = parser.parse_args(argv)

    changed = list(args.changed_file) if args.changed_file else discover_changed_files()
    plan = build_devcheck_plan(
        changed,
        allow_live=args.allow_live,
        allow_benchmark_heavy=args.allow_benchmark_heavy,
    )

    if plan.explicit_followup_lanes and not args.allow_live and "live" in plan.explicit_followup_lanes:
        _print_plan(plan)
        print(
            "error=live-only changes require explicit live execution with SWAAG_RUN_LIVE=1; "
            "rerun with --allow-live or use SWAAG_RUN_LIVE=1 python3 -m swaag.testprofile agent-tests"
        )
        return 2
    if plan.explicit_followup_lanes and not args.allow_benchmark_heavy and "benchmark_heavy" in plan.explicit_followup_lanes:
        _print_plan(plan)
        print(
            "error=benchmark-heavy-only changes require explicit heavy execution; "
            "rerun with --allow-benchmark-heavy or use python3 -m swaag.testprofile agent-tests"
        )
        return 2

    if args.baseline:
        testmon = detect_testmon(project_root())
        if args.require_testmon and not testmon.available:
            raise RuntimeError("pytest-testmon is required to build a baseline, but the plugin is unavailable")
        if not testmon.available:
            raise RuntimeError("Cannot build an incremental baseline without pytest-testmon")
        # Baselines always rebuild the selected profile rather than selecting only
        # affected tests.
        from swaag.test_categories import build_lane_command

        command = build_lane_command(plan.lane, root=project_root(), use_testmon=True, baseline_only=True)
    else:
        command = build_pytest_command(plan, require_testmon=args.require_testmon)

    _print_plan(plan)
    if not plan.candidate_tests and not plan.explicit_followup_lanes:
        print("$ no tests selected")
        return 0
    print("$", " ".join(command))
    if args.dry_run:
        return 0
    subprocess.run(command, cwd=project_root(), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
