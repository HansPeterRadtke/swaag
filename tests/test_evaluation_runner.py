from __future__ import annotations

import subprocess
from pathlib import Path

from swaag.benchmark import benchmark_runner
from swaag.benchmark.evaluation_runner import (
    _full_catalog_cache_key,
    run_agent_test_category,
    run_code_correctness_category,
    run_full_evaluation,
    run_test_category_evaluation,
)
from swaag.benchmark.task_definitions import get_benchmark_tasks
from swaag.manual_validation.runner import run_manual_validation


def _fake_junit(command: list[str], *, failures: int = 0) -> subprocess.CompletedProcess:
    junit_arg = next(item for item in command if item.startswith("--junitxml="))
    junit_path = Path(junit_arg.split("=", 1)[1])
    junit_path.write_text(
        f"""<testsuite tests="2" failures="{failures}" errors="0" skipped="0">
<testcase classname="tests.test_demo" name="test_ok" time="0.1" />
{('<testcase classname="tests.test_demo" name="test_fail" time="0.2"><failure message="boom">boom</failure></testcase>' if failures else '<testcase classname="tests.test_demo" name="test_ok_2" time="0.2" />')}
</testsuite>""",
        encoding="utf-8",
    )
    return subprocess.CompletedProcess(command, 1 if failures else 0, stdout="pytest output", stderr="")


def _fake_benchmark_report() -> dict[str, object]:
    return {
        "summary": {
            "total_tasks": 50,
            "successful_tasks": 20,
            "failed_tasks": 30,
            "false_positives": 4,
            "score_by_family": {"coding": 60.0, "reading": 40.0},
            "score_by_difficulty": {
                "extremely_easy": 90.0,
                "easy": 70.0,
                "normal": 50.0,
                "hard": 30.0,
                "extremely_hard": 10.0,
            },
            "full_task_success_percent": 40.0,
            "difficulty_group_average_percent": 50.0,
            "family_group_average_percent": 50.0,
            "group_average_percent": 50.0,
            "average_task_score_percent": 44.0,
            "failure_breakdown": {"verification_failure": 12, "wrong_tool": 5},
        },
        "aggregate_metrics": {
            "primary": {"task_success_rate": 0.4},
            "success_by_type": {"coding": 0.5, "reading": 0.3},
            "success_by_difficulty": {"extremely_easy": 0.8},
            "failure_breakdown": {"verification_failure": 12, "wrong_tool": 5},
            "verifier_weakness_breakdown": {"schema_failure": 4},
            "prompt_understanding_mistakes": {"authority_selection": 3},
        },
        "run_metadata": {
            "agent_behavior_mode": "cached",
            "seed_cache_mode_counts": {"replay": 120, "record": 30},
            "task_cache_mode_counts": {"mixed": 10, "replay": 40},
            "artifact_reused_from": "/tmp/benchmark-artifact",
        },
        "tasks": [{"task_id": "demo_task", "success": True}],
    }


def test_run_code_correctness_category_writes_reports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("swaag.benchmark.evaluation_runner.subprocess.run", lambda command, **kwargs: _fake_junit(command, failures=1))

    payload = run_code_correctness_category(output_dir=tmp_path / "code")

    assert payload["category"] == "code_correctness"
    assert payload["summary"]["total_tests"] == 2
    assert payload["summary"]["passed_tests"] == 1
    assert payload["summary"]["failed_tests"] == 1
    assert payload["summary"]["percent"] == 50.0
    assert (tmp_path / "code" / "code_correctness_results.json").exists()
    assert (tmp_path / "code" / "code_correctness_report.md").exists()


def test_run_agent_test_category_writes_real_benchmark_reports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(benchmark_runner, "run_benchmarks", lambda **kwargs: _fake_benchmark_report())

    payload = run_agent_test_category(output_dir=tmp_path / "agent", clean=True)

    assert payload["category"] == "agent_test"
    assert payload["summary"]["total_tasks"] == 50
    assert payload["score_summary"]["group_average_percent"] == 50.0
    assert payload["score_summary"]["full_task_success_percent"] == 40.0
    assert payload["score_summary"]["detailed_substep_score_percent"] is None
    assert (tmp_path / "agent" / "agent_test_results.json").exists()
    assert (tmp_path / "agent" / "agent_test_report.md").exists()
    report_text = (tmp_path / "agent" / "agent_test_report.md").read_text(encoding="utf-8")
    assert "Group Scores By Family" in report_text
    assert "Cache / Replay Summary" in report_text
    assert "Top Failure Diagnostics" in report_text
    assert "seed_cache_mode_counts" in report_text
    assert "verification_failure" in report_text


def test_run_agent_test_category_never_reuses_full_catalog_artifact(monkeypatch, tmp_path: Path) -> None:
    """run_agent_test_category must always call run_benchmarks, never skip with a cached artifact."""
    run_benchmarks_called = []

    def tracking_run_benchmarks(**kwargs):
        run_benchmarks_called.append(kwargs)
        return _fake_benchmark_report()

    monkeypatch.setattr(benchmark_runner, "run_benchmarks", tracking_run_benchmarks)

    # Even if a valid cached artifact exists, run_benchmarks must still be called
    monkeypatch.setattr("swaag.benchmark.evaluation_runner._valid_full_catalog_report", lambda report_path, tasks: True)

    run_agent_test_category(output_dir=tmp_path / "agent", clean=True)

    assert len(run_benchmarks_called) == 1, "run_benchmarks must be called exactly once for every authoritative run"
    assert run_benchmarks_called[0].get("agent_behavior_mode") == "cached"


def test_run_agent_test_category_seeds_shared_replay_cache_when_available(monkeypatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact-root"
    cache_dir = artifact_root / _full_catalog_cache_key(get_benchmark_tasks()) / "replay_cache" / "demo_task"
    cache_dir.mkdir(parents=True)
    (cache_dir / "seed_42.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SWAAG_FULL_CACHED_BENCHMARK_ARTIFACT_ROOT", str(artifact_root))
    def fake_run_benchmarks(**kwargs):
        assert (kwargs["output_dir"] / "replay_cache" / "demo_task" / "seed_42.json").exists()
        return _fake_benchmark_report()

    monkeypatch.setattr(benchmark_runner, "run_benchmarks", fake_run_benchmarks)

    run_agent_test_category(output_dir=tmp_path / "agent", clean=False)



def test_run_test_category_evaluation_runs_only_two_categories(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "swaag.benchmark.evaluation_runner.run_code_correctness_category",
        lambda *, output_dir, pytest_args=None, env=None: {
            "summary": {"percent": 100.0, "passed_tests": 5, "failed_tests": 0, "skipped_tests": 0, "total_tests": 5, "executed_tests": 5},
            "tests": [],
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        "swaag.benchmark.evaluation_runner.run_agent_test_category",
        lambda *, output_dir, benchmark_task_ids=None, clean=False, pytest_args=None, env=None: {
            "summary": {"total_tasks": 50, "successful_tasks": 20, "failed_tasks": 30, "false_positives": 4},
            "score_summary": {
                "group_average_percent": 50.0,
                "full_task_success_percent": 40.0,
                "difficulty_group_average_percent": 52.0,
                "family_group_average_percent": 48.0,
                "average_task_score_percent": 44.0,
                "detailed_substep_score_percent": None,
                "detailed_substep_score_note": "omitted",
            },
        },
    )

    payload = run_test_category_evaluation(output_dir=tmp_path / "categories", clean=True)

    assert payload["status"] == "complete"
    assert payload["code_correctness_binary_passed"] is True
    assert payload["agent_test_ran"] is True
    assert payload["agent_test"]["score_summary"]["group_average_percent"] == 50.0
    assert (tmp_path / "categories" / "test_categories_results.json").exists()
    assert (tmp_path / "categories" / "test_categories_report.md").exists()


def test_run_test_category_evaluation_stops_when_code_correctness_fails(monkeypatch, tmp_path: Path) -> None:
    agent_calls: list[str] = []
    monkeypatch.setattr(
        "swaag.benchmark.evaluation_runner.run_code_correctness_category",
        lambda *, output_dir, pytest_args=None, env=None: {
            "summary": {"percent": 80.0, "passed_tests": 4, "failed_tests": 1, "skipped_tests": 0, "total_tests": 5, "executed_tests": 5},
            "tests": [],
            "exit_code": 1,
        },
    )

    def fail_if_called(**kwargs):
        agent_calls.append("called")
        raise AssertionError("agent_test must not run after code_correctness failure")

    monkeypatch.setattr("swaag.benchmark.evaluation_runner.run_agent_test_category", fail_if_called)

    payload = run_test_category_evaluation(output_dir=tmp_path / "categories", clean=True)

    assert agent_calls == []
    assert payload["status"] == "code_correctness_failed"
    assert payload["agent_test_ran"] is False
    assert payload["agent_test"] is None
    assert "code_correctness must be 100%" in (tmp_path / "categories" / "test_categories_report.md").read_text(encoding="utf-8")


def test_run_manual_validation_is_not_a_test_category(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "swaag.manual_validation.runner.run_benchmarks",
        lambda **kwargs: {
            "summary": {
                "average_task_score_percent": 91.5,
                "total_tasks": 1,
                "score_by_difficulty": {
                    "extremely_easy": 100.0,
                    "easy": 95.0,
                    "normal": 92.0,
                    "hard": 88.0,
                    "extremely_hard": 82.5,
                },
            },
            "tasks": [{"task_id": "manual-demo", "difficulty": "normal", "score_percent": 92.0}],
        },
    )

    payload = run_manual_validation(output_dir=tmp_path / "manual", benchmark_task_ids=["manual-demo"])

    assert payload["is_test_category"] is False
    assert payload["category"] == "manual_validation"
    assert payload["percent"] == 91.5
    assert (tmp_path / "manual" / "manual_validation_results.json").exists()
    assert (tmp_path / "manual" / "manual_validation_report.md").exists()


def test_run_full_evaluation_combines_code_correctness_and_benchmark_scores(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "swaag.benchmark.evaluation_runner.run_code_correctness_category",
        lambda *, output_dir, pytest_args=None, env=None: {
            "summary": {"percent": 100.0, "passed_tests": 10, "failed_tests": 0, "skipped_tests": 0, "total_tests": 10, "executed_tests": 10},
            "tests": [],
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        benchmark_runner,
        "run_benchmarks",
        lambda **kwargs: {
            "summary": {
                "score_by_difficulty": {
                    "extremely_easy": 100.0,
                    "easy": 90.0,
                    "normal": 80.0,
                    "hard": 70.0,
                    "extremely_hard": 60.0,
                }
            },
            "tasks": [{"task_id": "demo", "difficulty": "easy", "score_percent": 90.0}],
        },
    )

    payload = run_full_evaluation(output_dir=tmp_path / "eval", clean=True, benchmark_task_ids=["demo"])

    assert payload["group_scores"] == {
        "code_correctness": 100.0,
        "extremely_easy": 100.0,
        "easy": 90.0,
        "normal": 80.0,
        "hard": 70.0,
        "extremely_hard": 60.0,
    }
    assert payload["overall_percent"] == 83.33
    assert (tmp_path / "eval" / "evaluation_results.json").exists()
    assert (tmp_path / "eval" / "evaluation_report.md").exists()
