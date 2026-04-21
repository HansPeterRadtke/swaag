from __future__ import annotations

import subprocess
from pathlib import Path

from swaag.benchmark.agent_support import run_agent_behavior_support_checks
from swaag.benchmark import benchmark_runner
from swaag.benchmark.evaluation_runner import (
    run_agent_behavior_tests,
    run_full_evaluation,
    run_functional_correctness_lane,
    run_agent_behavior_validation,
    run_test_category_evaluation,
    run_combined_test_evaluation,
)


def test_run_functional_correctness_lane_writes_reports(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, **kwargs):
        junit_arg = next(item for item in command if item.startswith("--junitxml="))
        junit_path = Path(junit_arg.split("=", 1)[1])
        junit_path.write_text(
            """<testsuite tests="2" failures="1" errors="0" skipped="0">
<testcase classname="tests.test_demo" name="test_ok" time="0.1" />
<testcase classname="tests.test_demo" name="test_fail" time="0.2"><failure message="boom">boom</failure></testcase>
</testsuite>""",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1, stdout="1 failed, 1 passed", stderr="")

    monkeypatch.setattr("swaag.benchmark.evaluation_runner.subprocess.run", fake_run)

    payload = run_functional_correctness_lane(output_dir=tmp_path / "functional")

    assert payload["summary"]["total_tests"] == 2
    assert payload["summary"]["passed_tests"] == 1
    assert payload["summary"]["failed_tests"] == 1
    assert payload["summary"]["percent"] == 50.0
    assert (tmp_path / "functional" / "functional_correctness_results.json").exists()
    assert (tmp_path / "functional" / "functional_correctness_report.md").exists()


def test_run_full_evaluation_combines_functional_and_agent_scores(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "swaag.benchmark.evaluation_runner.run_functional_correctness_lane",
        lambda *, output_dir, pytest_args=None, env=None: {
            "summary": {"percent": 100.0, "passed_tests": 10, "failed_tests": 0, "skipped_tests": 0, "total_tests": 10, "executed_tests": 10},
            "tests": [],
            "command": ["python3", "-m", "pytest", "-q"],
            "exit_code": 0,
            "stdout_path": str(output_dir / "stdout.txt"),
            "stderr_path": str(output_dir / "stderr.txt"),
            "junit_path": str(output_dir / "junit.xml"),
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
        "functional_correctness": 100.0,
        "extremely_easy": 100.0,
        "easy": 90.0,
        "normal": 80.0,
        "hard": 70.0,
        "extremely_hard": 60.0,
    }
    assert payload["overall_percent"] == 83.33
    assert (tmp_path / "eval" / "evaluation_results.json").exists()
    assert (tmp_path / "eval" / "evaluation_report.md").exists()


def test_run_agent_behavior_support_checks_writes_percent_scored_reports(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, **kwargs):
        junit_arg = next(item for item in command if item.startswith("--junitxml="))
        junit_path = Path(junit_arg.split("=", 1)[1])
        junit_path.write_text(
            """<testsuite tests="3" failures="1" errors="0" skipped="0">
<testcase classname="tests.test_demo" name="test_ok_a" time="0.1" />
<testcase classname="tests.test_demo" name="test_ok_b" time="0.1" />
<testcase classname="tests.test_demo" name="test_fail" time="0.1"><failure message="boom">boom</failure></testcase>
</testsuite>""",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1, stdout="1 failed, 2 passed", stderr="")

    monkeypatch.setattr("swaag.benchmark.agent_support.subprocess.run", fake_run)

    payload = run_agent_behavior_support_checks(
        output_dir=tmp_path / "agent_support",
        family_ids=["record_replay_runtime"],
        clean=True,
    )

    assert payload["summary"]["total_families"] == 1
    assert payload["summary"]["passed_tests"] == 2
    assert payload["summary"]["failed_tests"] == 1
    assert payload["summary"]["percent"] == 66.67
    assert (tmp_path / "agent_support" / "agent_behavior_cached_support_results.json").exists()
    assert (tmp_path / "agent_support" / "agent_behavior_cached_support_report.md").exists()


def test_run_agent_behavior_validation_writes_results(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        benchmark_runner,
        "run_benchmarks",
        lambda **kwargs: {
            "summary": {
                "average_task_score_percent": 91.5,
                "score_by_difficulty": {
                    "extremely_easy": 100.0,
                    "easy": 95.0,
                    "normal": 92.0,
                    "hard": 88.0,
                    "extremely_hard": 82.5,
                },
            },
            "tasks": [{"task_id": "live-demo", "difficulty": "normal", "score_percent": 92.0}],
        },
    )

    payload = run_agent_behavior_validation(
        output_dir=tmp_path / "live",
        benchmark_task_ids=["live-demo"],
    )

    assert payload["percent"] == 91.5
    assert payload["difficulty_tier_scores"]["hard"] == 88.0
    assert (tmp_path / "live" / "agent_behavior_validation_results.json").exists()


def test_run_agent_behavior_tests_writes_mode_specific_reports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        benchmark_runner,
        "run_benchmarks",
        lambda **kwargs: {
            "summary": {
                "average_task_score_percent": 88.0,
                "total_tasks": 1,
                "successful_tasks": 1,
                "failed_tasks": 0,
                "false_positives": 0,
                "score_by_difficulty": {
                    "extremely_easy": 100.0,
                    "easy": 92.0,
                    "normal": 88.0,
                    "hard": 80.0,
                    "extremely_hard": 70.0,
                },
            },
            "tasks": [{"task_id": "demo", "difficulty": "normal", "score_percent": 88.0}],
            "run_metadata": {"replay_cache_root": str(tmp_path / "agent_behavior" / "agent_behavior_cached" / "replay_cache")},
        },
    )

    payload = run_agent_behavior_tests(
        output_dir=tmp_path / "agent_behavior",
        mode="cached",
        benchmark_task_ids=["demo"],
    )

    assert payload["mode"] == "cached"
    assert payload["percent"] == 88.0
    assert (tmp_path / "agent_behavior" / "agent_behavior_cached_results.json").exists()
    assert (tmp_path / "agent_behavior" / "agent_behavior_cached_report.md").exists()


def test_run_test_category_evaluation_writes_category_reports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "swaag.benchmark.evaluation_runner.run_functional_correctness_lane",
        lambda *, output_dir, pytest_args=None, env=None: {
            "summary": {"percent": 100.0, "passed_tests": 5, "failed_tests": 0, "skipped_tests": 0, "total_tests": 5, "executed_tests": 5},
            "tests": [],
        },
    )
    monkeypatch.setattr(
        "swaag.benchmark.evaluation_runner.run_agent_behavior_tests",
        lambda *, mode, **kwargs: {
            "mode": mode,
            "percent": 90.0 if mode == "cached" else 80.0,
            "summary": {"successful_tasks": 1, "failed_tasks": 0, "total_tasks": 1},
            "difficulty_tier_scores": {
                "extremely_easy": 100.0,
                "easy": 95.0,
                "normal": 85.0,
                "hard": 75.0,
                "extremely_hard": 65.0,
            },
            "tasks": [{"task_id": f"{mode}-demo", "difficulty": "normal", "score_percent": 85.0}],
        },
    )

    payload = run_test_category_evaluation(output_dir=tmp_path / "categories", clean=True)

    assert payload["category_scores"] == {
        "deterministic_correctness": 100.0,
        "agent_behavior_cached": 90.0,
        "agent_behavior_validation": 80.0,
    }
    assert payload["overall_percent"] == 90.0
    assert (tmp_path / "categories" / "test_categories_results.json").exists()
    assert (tmp_path / "categories" / "test_categories_report.md").exists()


def test_run_test_category_evaluation_stops_when_deterministic_correctness_fails(monkeypatch, tmp_path: Path) -> None:
    behavior_calls: list[str] = []
    monkeypatch.setattr(
        "swaag.benchmark.evaluation_runner.run_functional_correctness_lane",
        lambda *, output_dir, pytest_args=None, env=None: {
            "summary": {"percent": 80.0, "passed_tests": 4, "failed_tests": 1, "skipped_tests": 0, "total_tests": 5, "executed_tests": 5},
            "tests": [],
            "exit_code": 1,
        },
    )

    def fail_if_called(*, mode, **kwargs):
        behavior_calls.append(mode)
        raise AssertionError("agent behavior tests must not run after deterministic correctness failure")

    monkeypatch.setattr("swaag.benchmark.evaluation_runner.run_agent_behavior_tests", fail_if_called)

    payload = run_test_category_evaluation(output_dir=tmp_path / "categories", clean=True)

    assert behavior_calls == []
    assert payload["status"] == "deterministic_correctness_failed"
    assert payload["skipped_categories"] == ["agent_behavior_cached", "agent_behavior_validation"]
    assert payload["agent_behavior_cached"] is None
    assert payload["agent_behavior_validation"] is None
    assert (tmp_path / "categories" / "test_categories_results.json").exists()
    assert "not run because deterministic correctness failed" in (tmp_path / "categories" / "test_categories_report.md").read_text(encoding="utf-8")


def test_run_combined_evaluation_combines_explicit_category_scores(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "swaag.benchmark.evaluation_runner.run_functional_correctness_lane",
        lambda *, output_dir, pytest_args=None, env=None: {
            "summary": {"percent": 100.0, "passed_tests": 10, "failed_tests": 0, "skipped_tests": 0, "total_tests": 10, "executed_tests": 10},
            "tests": [],
            "command": ["python3", "-m", "pytest", "-q"],
            "exit_code": 0,
            "stdout_path": str(output_dir / "stdout.txt"),
            "stderr_path": str(output_dir / "stderr.txt"),
            "junit_path": str(output_dir / "junit.xml"),
        },
    )
    monkeypatch.setattr(
        "swaag.benchmark.evaluation_runner.run_agent_behavior_validation",
        lambda **kwargs: {
            "percent": 85.0,
            "difficulty_tier_scores": {
                "extremely_easy": 100.0,
                "easy": 90.0,
                "normal": 85.0,
                "hard": 80.0,
                "extremely_hard": 70.0,
            },
            "tasks": [{"task_id": "live-demo", "difficulty": "normal", "score_percent": 85.0}],
            "summary": {"average_task_score_percent": 85.0},
        },
    )
    monkeypatch.setattr(
        "swaag.benchmark.agent_support.run_agent_behavior_support_checks",
        lambda **kwargs: {
            "summary": {"percent": 90.0, "passed_tests": 9, "failed_tests": 1, "total_tests": 10},
            "results": [],
        },
    )

    payload = run_combined_test_evaluation(output_dir=tmp_path / "combined", clean=True)

    assert payload["category_scores"] == {
        "deterministic_correctness": 100.0,
        "agent_behavior_cached_support": 90.0,
        "agent_behavior_validation": 85.0,
    }
    assert payload["overall_percent"] == 91.67
    assert (tmp_path / "combined" / "full_evaluation_results.json").exists()
    assert (tmp_path / "combined" / "full_evaluation_report.md").exists()
