from __future__ import annotations

from swaag.test_categories import (
    top_level_lane_for_test_file,
    top_level_lane_test_files,
    validate_top_level_lane_registry,
)


def test_top_level_category_registry_covers_current_tree() -> None:
    validate_top_level_lane_registry()


def test_top_level_category_registry_reports_expected_examples() -> None:
    assert top_level_lane_for_test_file("tests/test_imports.py") == "code_correctness"
    assert top_level_lane_for_test_file("tests/test_runtime.py") == "agent_test"
    assert top_level_lane_for_test_file("tests/test_live_llamacpp.py") == "agent_test"


def test_top_level_category_groups_are_non_empty_and_disjoint() -> None:
    categories = top_level_lane_test_files()

    assert categories["code_correctness"]
    assert categories["agent_test"]
    assert "tests/test_live_llamacpp.py" in categories["agent_test"]
    assert set(categories["code_correctness"]).isdisjoint(categories["agent_test"])
