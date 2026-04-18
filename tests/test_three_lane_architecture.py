from __future__ import annotations

from swaag.testlanes import (
    top_level_lane_for_test_file,
    top_level_lane_test_files,
    validate_top_level_lane_registry,
)


def test_top_level_lane_registry_covers_current_tree() -> None:
    validate_top_level_lane_registry()


def test_top_level_lane_registry_reports_expected_examples() -> None:
    assert top_level_lane_for_test_file("tests/test_imports.py") == "deterministic_correctness"
    assert top_level_lane_for_test_file("tests/test_runtime.py") == "agent_loop_regression"
    assert top_level_lane_for_test_file("tests/test_live_llamacpp.py") == "live_agent_evaluation"


def test_top_level_lane_groups_are_non_empty_and_disjoint() -> None:
    lanes = top_level_lane_test_files()

    assert lanes["deterministic_correctness"]
    assert lanes["agent_loop_regression"]
    assert lanes["live_agent_evaluation"] == ("tests/test_live_llamacpp.py",)
    assert set(lanes["deterministic_correctness"]).isdisjoint(lanes["agent_loop_regression"])
