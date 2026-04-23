from __future__ import annotations

import ast
from pathlib import Path

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
    assert category_for_file("tests/test_runtime.py") == "agent_test"


def test_top_level_category_groups_are_non_empty_and_disjoint() -> None:
    categories = category_files()

    assert categories["code_correctness"]
    assert categories["agent_test"]
    assert TOP_LEVEL_TEST_CATEGORIES == ("code_correctness", "agent_test")
    assert set(categories["code_correctness"]).isdisjoint(categories["agent_test"])


def test_every_pytest_file_in_tests_belongs_to_a_category() -> None:
    """Every test_*.py file in tests/ must be in exactly one of the two categories.

    No uncategorised files, no third category, no manual-validation test files
    masquerading in the pytest tree.
    """
    from swaag.test_categories import all_test_files, project_root

    known = set(all_test_files(project_root()))
    classified = CODE_CORRECTNESS_TEST_FILES | AGENT_TEST_FILES
    unclassified = sorted(known - classified)
    assert unclassified == [], f"Test files not classified into either category: {unclassified}"
    phantom = sorted(classified - known)
    assert phantom == [], f"Category registries reference files that do not exist: {phantom}"


def test_authoritative_commands_use_explicit_two_category_file_lists() -> None:
    code_command = build_code_correctness_command()
    agent_command = build_agent_tests_command()

    assert "-m" not in code_command[3:]
    assert "-m" not in agent_command[3:]
    assert any(path.endswith("tests/test_imports.py") for path in code_command)
    assert any(path.endswith("tests/test_runtime.py") for path in agent_command)


def test_authoritative_agent_tests_include_full_cached_benchmark_catalog() -> None:
    agent_command = build_agent_tests_command()

    assert any(path.endswith("tests/test_benchmark.py") for path in agent_command)
    assert not any(path.endswith("tests/test_clean_install_agent.py") for path in agent_command)


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
