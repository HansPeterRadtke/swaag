from __future__ import annotations

import inspect

import tests.test_live_llamacpp as live_tests
from swaag.live_runtime_profiles import get_live_runtime_recommendation


def test_live_suite_has_minimum_real_task_coverage() -> None:
    test_names = sorted(name for name, value in inspect.getmembers(live_tests) if name.startswith("test_live_") and callable(value))

    assert len(test_names) >= 8
    assert any("direct_response" in name for name in test_names)
    assert sum(1 for name in test_names if "tool" in name or "calculator" in name or "run_tests" in name) >= 2
    assert any("file_read" in name for name in test_names)
    assert any("file_edit" in name for name in test_names)
    assert any("multi_step" in name for name in test_names)
    assert any("missing_file" in name or "not_done" in name or "failure" in name for name in test_names)
    assert any("verification_strength" in name for name in test_names)


def test_live_suite_uses_fixed_three_seed_policy() -> None:
    recommendation = get_live_runtime_recommendation("fast_live_tests")
    assert recommendation.seeds == (11, 23, 37)
    assert live_tests._live_seeds() == recommendation.seeds
    live_test_functions = [
        value
        for name, value in inspect.getmembers(live_tests)
        if name.startswith("test_live_") and callable(value)
    ]
    assert live_test_functions
    for function in live_test_functions:
        assert "_run_across_seeds" in inspect.getsource(function)
