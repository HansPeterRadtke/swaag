from __future__ import annotations

from pathlib import Path

from swaag.benchmark import benchmark_runner
from swaag.benchmark.agent_support import get_agent_support_families


def test_agent_support_families_cover_replay_and_runtime_behavior() -> None:
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


def test_benchmark_runner_agent_support_command_uses_agent_support_runner(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_agent_behavior_support_checks(*, output_dir: Path, family_ids, clean: bool):
        observed["output_dir"] = output_dir
        observed["family_ids"] = family_ids
        observed["clean"] = clean
        return {
            "summary": {"total_families": 1, "passed_families": 1, "failed_families": 0, "percent": 100.0},
            "results": [],
        }

    monkeypatch.setattr("swaag.benchmark.agent_support.run_agent_behavior_support_checks", fake_run_agent_behavior_support_checks)

    exit_code = benchmark_runner.main(
        ["agent-support", "--family", "record_replay_runtime", "--output", str(tmp_path / "out"), "--clean"]
    )

    assert exit_code == 0
    assert observed["output_dir"] == tmp_path / "out"
    assert observed["family_ids"] == ["record_replay_runtime"]
    assert observed["clean"] is True


def test_benchmark_runner_full_evaluate_command_uses_combined_evaluation(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_combined_test_evaluation(**kwargs):
        observed.update(kwargs)
        return {
            "category_scores": {
                "deterministic_correctness": 100.0,
                "agent_behavior_cached_support": 100.0,
                "agent_behavior_validation": 100.0,
            },
            "overall_percent": 100.0,
        }

    monkeypatch.setattr("swaag.benchmark.evaluation_runner.run_combined_test_evaluation", fake_run_combined_test_evaluation)

    exit_code = benchmark_runner.main(
        ["full-evaluate", "--output", str(tmp_path / "eval"), "--clean", "--task", "live-demo"]
    )

    assert exit_code == 0
    assert observed["output_dir"] == tmp_path / "eval"
    assert observed["clean"] is True
    assert observed["benchmark_task_ids"] == ["live-demo"]


def test_benchmark_runner_agent_tests_command_uses_cached_mode_runner(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_agent_behavior_tests(**kwargs):
        observed.update(kwargs)
        return {
            "mode": "cached",
            "percent": 100.0,
            "summary": {"total_tasks": 1, "successful_tasks": 1, "failed_tasks": 0, "false_positives": 0},
            "run_metadata": {"replay_cache_root": str(tmp_path / "out" / "agent_behavior_cached" / "replay_cache")},
        }

    monkeypatch.setattr("swaag.benchmark.evaluation_runner.run_agent_behavior_tests", fake_run_agent_behavior_tests)

    exit_code = benchmark_runner.main(
        ["agent-tests", "--mode", "cached", "--output", str(tmp_path / "out"), "--clean", "--task", "demo-task"]
    )

    assert exit_code == 0
    assert observed["output_dir"] == tmp_path / "out"
    assert observed["mode"] == "cached"
    assert observed["benchmark_task_ids"] == ["demo-task"]


def test_benchmark_runner_test_categories_command_uses_category_evaluation(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_test_category_evaluation(**kwargs):
        observed.update(kwargs)
        return {
            "overall_percent": 95.0,
            "category_scores": {
                "deterministic_correctness": 100.0,
                "agent_behavior_cached": 95.0,
                "agent_behavior_validation": 90.0,
            },
        }

    monkeypatch.setattr("swaag.benchmark.evaluation_runner.run_test_category_evaluation", fake_run_test_category_evaluation)

    exit_code = benchmark_runner.main(
        ["test-categories", "--output", str(tmp_path / "eval"), "--clean", "--task", "demo-task"]
    )

    assert exit_code == 1
    assert observed["output_dir"] == tmp_path / "eval"
    assert observed["clean"] is True
    assert observed["benchmark_task_ids"] == ["demo-task"]
