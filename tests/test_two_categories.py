from __future__ import annotations

from swaag.test_categories import (
    AGENT_TEST_FILES,
    CODE_CORRECTNESS_TEST_FILES,
    MANUAL_VALIDATION_FILES,
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
    assert "tests/test_live_llamacpp.py" in MANUAL_VALIDATION_FILES
    assert "tests/test_live_llamacpp.py" not in AGENT_TEST_FILES
    assert "tests/test_live_llamacpp.py" not in CODE_CORRECTNESS_TEST_FILES


def test_top_level_category_groups_are_non_empty_and_disjoint() -> None:
    categories = category_files()

    assert categories["code_correctness"]
    assert categories["agent_test"]
    assert TOP_LEVEL_TEST_CATEGORIES == ("code_correctness", "agent_test")
    assert "tests/test_live_llamacpp.py" not in categories["agent_test"]
    assert set(categories["code_correctness"]).isdisjoint(categories["agent_test"])


def test_authoritative_commands_use_explicit_two_category_file_lists() -> None:
    code_command = build_code_correctness_command()
    agent_command = build_agent_tests_command()

    assert "-m" not in code_command[3:]
    assert "-m" not in agent_command[3:]
    assert any(path.endswith("tests/test_imports.py") for path in code_command)
    assert any(path.endswith("tests/test_runtime.py") for path in agent_command)
    assert not any(path.endswith("tests/test_live_llamacpp.py") for path in code_command)
    assert not any(path.endswith("tests/test_live_llamacpp.py") for path in agent_command)
