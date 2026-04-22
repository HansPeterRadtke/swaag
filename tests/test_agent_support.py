from __future__ import annotations

from pathlib import Path

from swaag.benchmark import benchmark_runner
from swaag.benchmark.agent_support import get_agent_support_families


def test_agent_test_support_families_cover_replay_and_runtime_behavior() -> None:
    families = {item.family_id: item for item in get_agent_support_families()}

    assert "continuation_background" in families
    assert "runtime_recovery_contracts" in families
    assert "direct_response_guardrails" in families
    assert "tool_routing_context_focus" in families
    assert "subagent_traceability" in families
    assert "record_replay_runtime" in families
    assert "scripted_benchmark_runtime" in families
    assert families["record_replay_runtime"].nodeids == (
        "tests/test_agent_loop_replay.py::test_record_replay_client_replays_runtime_tool_flow",
    )


def test_benchmark_runner_agent_tests_command_uses_cached_pytest_category(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_agent_test_category(**kwargs):
        observed.update(kwargs)
        return {
            "summary": {"percent": 100.0, "passed_tests": 2, "failed_tests": 0, "skipped_tests": 0},
            "exit_code": 0,
        }

    monkeypatch.setattr("swaag.benchmark.evaluation_runner.run_agent_test_category", fake_run_agent_test_category)

    exit_code = benchmark_runner.main(["agent-tests", "--output", str(tmp_path / "out"), "--clean"])

    assert exit_code == 0
    assert observed["output_dir"] == tmp_path / "out"
    assert observed["pytest_args"] is None


def test_benchmark_runner_test_categories_command_uses_two_category_evaluation(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_test_category_evaluation(**kwargs):
        observed.update(kwargs)
        return {"overall_percent": 95.0, "category_scores": {"code_correctness": 100.0, "agent_test": 90.0}}

    monkeypatch.setattr("swaag.benchmark.evaluation_runner.run_test_category_evaluation", fake_run_test_category_evaluation)

    exit_code = benchmark_runner.main(["test-categories", "--output", str(tmp_path / "eval"), "--clean"])

    assert exit_code == 1
    assert observed["output_dir"] == tmp_path / "eval"
    assert observed["clean"] is True
    assert "benchmark_task_ids" not in observed


def test_manual_validation_cli_is_separate_from_tests(monkeypatch, tmp_path: Path) -> None:
    from swaag.manual_validation import __main__ as manual_validation_main

    observed: dict[str, object] = {}

    def fake_run_manual_validation(**kwargs):
        observed.update(kwargs)
        return {
            "is_test_category": False,
            "percent": 100.0,
            "summary": {"total_tasks": 1, "failed_tasks": 0, "false_positives": 0},
        }

    monkeypatch.setattr("swaag.manual_validation.runner.run_manual_validation", fake_run_manual_validation)

    exit_code = manual_validation_main.main(["--output", str(tmp_path / "manual"), "--clean", "--task", "demo"])

    assert exit_code == 0
    assert observed["output_dir"] == tmp_path / "manual"
    assert observed["benchmark_task_ids"] == ["demo"]
    assert observed["validation_subset"] is True
