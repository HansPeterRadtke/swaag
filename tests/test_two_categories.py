from __future__ import annotations

import ast
import sys
from pathlib import Path

from swaag import testlane
from swaag.test_categories import (
    AGENT_TEST_FILES,
    CODE_CORRECTNESS_TEST_FILES,
    TOP_LEVEL_TEST_CATEGORIES,
    build_agent_tests_command,
    build_code_correctness_command,
    category_for_file,
    category_files,
    validate_test_category_registry,
)


def test_top_level_category_registry_covers_current_tree() -> None:
    validate_test_category_registry()


def test_top_level_category_registry_reports_expected_examples() -> None:
    assert category_for_file("tests/test_imports.py") == "code_correctness"
    assert category_for_file("tests/test_runtime.py") == "code_correctness"
    assert category_for_file("tests/test_benchmark.py") == "code_correctness"


def test_top_level_category_groups_are_disjoint_and_category_names_are_stable() -> None:
    categories = category_files()

    assert categories["code_correctness"]
    assert categories["agent_test"] == ()
    assert TOP_LEVEL_TEST_CATEGORIES == ("code_correctness", "agent_test")
    assert set(categories["code_correctness"]).isdisjoint(categories["agent_test"])
    assert AGENT_TEST_FILES == frozenset()


def test_every_pytest_file_in_tests_belongs_to_code_correctness_or_explicit_agent_registry() -> None:
    """Every pytest file under tests/ must be classified into the authoritative model."""
    from swaag.test_categories import all_test_files, project_root

    known = set(all_test_files(project_root()))
    classified = CODE_CORRECTNESS_TEST_FILES | AGENT_TEST_FILES
    unclassified = sorted(known - classified)
    assert unclassified == [], f"Test files not classified into the test model: {unclassified}"
    phantom = sorted(classified - known)
    assert phantom == [], f"Category registries reference files that do not exist: {phantom}"


def test_authoritative_commands_use_explicit_two_category_commands() -> None:
    code_command = build_code_correctness_command()
    agent_command = build_agent_tests_command()

    assert code_command[:3] == [sys.executable, "-m", "pytest"]
    assert "-m" not in code_command[3:]
    assert any(path.endswith("tests/test_imports.py") for path in code_command)
    assert any(path.endswith("tests/test_benchmark.py") for path in code_command)

    assert agent_command == [sys.executable, "-m", "swaag.benchmark", "agent-tests"]


def test_authoritative_agent_tests_run_real_benchmark_not_pytest_wrapper() -> None:
    agent_command = build_agent_tests_command()

    assert agent_command == [sys.executable, "-m", "swaag.benchmark", "agent-tests"]
    assert not any(part.endswith("tests/test_benchmark.py") for part in agent_command)


def test_full_cached_benchmark_test_does_not_replace_catalog_with_subset() -> None:
    benchmark_test = Path(__file__).resolve().parent / "test_benchmark.py"
    tree = ast.parse(benchmark_test.read_text(encoding="utf-8"), filename=str(benchmark_test))
    target = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_benchmark_runner_executes_full_cached_catalog_and_writes_reports"
    )

    calls = [node for node in ast.walk(target) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "setattr"
        and any(isinstance(arg, ast.Constant) and arg.value == "_load_tasks" for arg in call.args)
        for call in calls
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in {"representative_tasks", "subset_tasks"}
        for node in ast.walk(target)
    )
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_full_catalog_with_artifact_reuse"
    )
    helper_calls = [node for node in ast.walk(helper) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Name)
        and call.func.id == "run_benchmarks"
        and not any(keyword.arg == "task_ids" for keyword in call.keywords)
        for call in helper_calls
    )


def test_testprofile_agent_tests_dry_run_uses_real_benchmark_command(capsys) -> None:
    exit_code = testlane.main(["agent-tests", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "swaag.benchmark agent-tests" in captured.out
    assert "tests/test_benchmark.py" not in captured.out


def test_testprofile_all_alias_is_supported(capsys) -> None:
    exit_code = testlane.main(["all", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "python3 -m swaag.testprofile code-correctness" in captured.out
    assert "python3 -m swaag.testprofile agent-tests" in captured.out


def test_testprofile_agent_tests_prints_real_benchmark_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(testlane, "_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        testlane,
        "run_agent_test_category",
        lambda *, output_dir, clean=False: {
            "summary": {"total_tasks": 53, "successful_tasks": 7, "failed_tasks": 46, "false_positives": 9},
            "score_summary": {
                "group_average_percent": 31.5,
                "full_task_success_percent": 13.21,
                "difficulty_group_average_percent": 30.2,
                "family_group_average_percent": 32.8,
                "average_task_score_percent": 28.4,
                "detailed_substep_score_note": "omitted",
                "group_scores_by_difficulty": {"hard": 18.0, "extremely_hard": 9.5},
                "group_scores_by_family": {"coding": 41.0, "failure": 17.5},
            },
            "aggregate_metrics": {
                "failure_breakdown": {"verification_failure": 14, "wrong_tool": 6},
                "verifier_weakness_breakdown": {"stale_source_failure": 4},
                "prompt_understanding_mistakes": {"authority_selection": 3},
            },
            "run_metadata": {
                "execution_mode": "executed_cached_benchmark",
                "seed_cache_mode_counts": {"replay": 50, "record": 3},
                "task_cache_mode_counts": {"replay": 50, "record": 3},
            },
            "execution_mode": "executed_cached_benchmark",
            "cached_benchmark_results_path": "/tmp/results.json",
            "cached_benchmark_report_path": "/tmp/report.md",
        },
    )

    exit_code = testlane.main(["agent-tests"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "== agent_test ==" in captured.out
    assert "execution_mode=executed_cached_benchmark" in captured.out
    assert "difficulty_scores" in captured.out
    assert "family_scores" in captured.out
    assert "cache_replay_mode=per_seed_record_replay" in captured.out
    assert "top_failure_categories={'verification_failure': 14, 'wrong_tool': 6}" in captured.out
    assert "cached_benchmark_results_path=/tmp/results.json" in captured.out


def test_testprofile_combined_prints_code_correctness_then_benchmark(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(testlane, "_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        testlane,
        "run_test_category_evaluation",
        lambda *, output_dir, clean=False: {
            "code_correctness_binary_passed": True,
            "code_correctness": {
                "summary": {
                    "executed_tests": 10,
                    "passed_tests": 10,
                    "failed_tests": 0,
                    "skipped_tests": 0,
                    "percent": 100.0,
                },
                "exit_code": 0,
            },
            "agent_test": {
                "summary": {"total_tasks": 53, "successful_tasks": 7, "failed_tasks": 46, "false_positives": 9},
                "score_summary": {
                    "group_average_percent": 31.5,
                    "full_task_success_percent": 13.21,
                    "difficulty_group_average_percent": 30.2,
                    "family_group_average_percent": 32.8,
                    "average_task_score_percent": 28.4,
                    "detailed_substep_score_note": "omitted",
                    "group_scores_by_difficulty": {"hard": 18.0},
                    "group_scores_by_family": {"coding": 41.0},
                },
                "aggregate_metrics": {},
                "run_metadata": {},
            },
        },
    )

    exit_code = testlane.main(["combined"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.index("== code_correctness ==") < captured.out.index("== agent_test ==")
