from __future__ import annotations

import argparse
from pathlib import Path

from swaag.benchmark.evaluation_runner import (
    run_agent_test_category,
    run_code_correctness_category,
    run_test_category_evaluation,
)
from swaag.test_categories import (
    build_agent_tests_command,
    build_code_correctness_command,
    project_root,
)

ALL_CHOICES = ["code-correctness", "agent-tests", "combined", "all"]
_OUTPUT_ROOT = Path("/tmp/swaag-testprofile")


def _code_output_dir(profile: str) -> Path:
    return _OUTPUT_ROOT / profile


def _print_code_correctness_summary(payload: dict[str, object], *, output_dir: Path) -> None:
    summary = payload["summary"]
    print("== code_correctness ==")
    print(f"total_checks={summary['executed_tests']}")
    print(f"passed={summary['passed_tests']}")
    print(f"failed={summary['failed_tests']}")
    print(f"skipped={summary['skipped_tests']}")
    print(f"percent={summary['percent']:.2f}")
    print(f"binary_result={'passed' if payload.get('exit_code', 1) == 0 and summary['failed_tests'] == 0 else 'failed'}")
    print(f"artifacts={output_dir}")


def _print_agent_test_summary(payload: dict[str, object], *, output_dir: Path) -> None:
    summary = payload["summary"]
    scores = payload["score_summary"]
    print("== agent_test ==")
    print(f"total_tasks={summary['total_tasks']}")
    print(f"successful_tasks={summary['successful_tasks']}")
    print(f"failed_tasks={summary['failed_tasks']}")
    print(f"false_positives={summary['false_positives']}")
    print(f"full_task_success_percent={scores['full_task_success_percent']:.2f}")
    print(f"group_average_percent={scores['group_average_percent']:.2f}")
    print(f"difficulty_group_average_percent={scores['difficulty_group_average_percent']:.2f}")
    print(f"family_group_average_percent={scores['family_group_average_percent']:.2f}")
    print(f"average_task_score_percent={scores['average_task_score_percent']:.2f}")
    print(f"detailed_substep_score={scores['detailed_substep_score_note']}")
    print(f"artifacts={output_dir}")


def _run_combined(*, root: Path, dry_run: bool) -> int:
    code_command = build_code_correctness_command(root=root)
    agent_command = build_agent_tests_command(root=root)
    if dry_run:
        print("$", " ".join(["python3", "-m", "swaag.testprofile", "code-correctness"]))
        print("$", " ".join(["python3", "-m", "swaag.testprofile", "agent-tests"]))
        print("# canonical benchmark command:", " ".join(agent_command))
        return 0
    output_dir = _code_output_dir("combined")
    payload = run_test_category_evaluation(output_dir=output_dir, clean=False)
    _print_code_correctness_summary(payload["code_correctness"], output_dir=output_dir / "code_correctness")
    if not payload["code_correctness_binary_passed"]:
        print(f"error={payload['skip_reason']}")
        return 1
    _print_agent_test_summary(payload["agent_test"], output_dir=output_dir / "agent_test")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an authoritative SWAAG test category.\n\n"
            "Profiles:\n"
            "  code-correctness  - deterministic binary code-correctness checks\n"
            "  agent-tests       - real cached benchmark with score-based output\n"
            "  combined          - code-correctness first, then agent-tests\n"
            "  all               - alias for combined\n\n"
            "Manual validation is explicit real-model usage and not part of the test categories."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("profile", choices=ALL_CHOICES)
    parser.add_argument("--dry-run", action="store_true", help="Print the canonical command(s) only.")
    parser.add_argument("--baseline", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = project_root()

    if args.profile == "code-correctness":
        if args.dry_run:
            print("$", " ".join(["python3", "-m", "swaag.testprofile", "code-correctness"]))
            print("# underlying pytest command:", " ".join(build_code_correctness_command(root=root)))
            return 0
        output_dir = _code_output_dir("code-correctness")
        payload = run_code_correctness_category(output_dir=output_dir, pytest_args=None)
        _print_code_correctness_summary(payload, output_dir=output_dir)
        return payload.get("exit_code", 1)

    if args.profile == "agent-tests":
        if args.dry_run:
            print("$", " ".join(["python3", "-m", "swaag.testprofile", "agent-tests"]))
            print("# canonical benchmark command:", " ".join(build_agent_tests_command(root=root)))
            return 0
        output_dir = _code_output_dir("agent-tests")
        payload = run_agent_test_category(output_dir=output_dir, clean=False)
        _print_agent_test_summary(payload, output_dir=output_dir)
        return 0

    if args.profile in {"combined", "all"}:
        return _run_combined(root=root, dry_run=args.dry_run)

    raise SystemExit(f"Unhandled profile: {args.profile}")


if __name__ == "__main__":
    raise SystemExit(main())
