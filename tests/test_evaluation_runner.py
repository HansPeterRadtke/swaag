from __future__ import annotations

import subprocess
from pathlib import Path

from swaag.benchmark.agent_regression import run_agent_loop_regression_lane
from swaag.benchmark import benchmark_runner
from swaag.benchmark.evaluation_runner import (
    run_full_evaluation,
    run_functional_correctness_lane,
    run_live_agent_evaluation_lane,
    run_three_lane_evaluation,
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


def test_run_agent_loop_regression_lane_writes_percent_scored_reports(monkeypatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr("swaag.benchmark.agent_regression.subprocess.run", fake_run)

    payload = run_agent_loop_regression_lane(
        output_dir=tmp_path / "regression",
        family_ids=["record_replay_runtime"],
        clean=True,
    )

    assert payload["summary"]["total_families"] == 1
    assert payload["summary"]["passed_tests"] == 2
    assert payload["summary"]["failed_tests"] == 1
    assert payload["summary"]["percent"] == 66.67
    assert (tmp_path / "regression" / "agent_loop_regression_results.json").exists()
    assert (tmp_path / "regression" / "agent_loop_regression_report.md").exists()


def test_run_live_agent_evaluation_lane_writes_lane_results(monkeypatch, tmp_path: Path) -> None:
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

    payload = run_live_agent_evaluation_lane(
        output_dir=tmp_path / "live",
        benchmark_task_ids=["live-demo"],
    )

    assert payload["percent"] == 91.5
    assert payload["difficulty_tier_scores"]["hard"] == 88.0
    assert (tmp_path / "live" / "live_agent_evaluation_results.json").exists()


def test_run_three_lane_evaluation_combines_explicit_lane_scores(monkeypatch, tmp_path: Path) -> None:
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
        "swaag.benchmark.evaluation_runner.run_live_agent_evaluation_lane",
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
        "swaag.benchmark.agent_regression.run_agent_loop_regression_lane",
        lambda **kwargs: {
            "summary": {"percent": 90.0, "passed_tests": 9, "failed_tests": 1, "total_tests": 10},
            "results": [],
        },
    )

    payload = run_three_lane_evaluation(output_dir=tmp_path / "three", clean=True)

    assert payload["lane_scores"] == {
        "deterministic_correctness": 100.0,
        "agent_loop_regression": 90.0,
        "live_agent_evaluation": 85.0,
    }
    assert payload["overall_percent"] == 91.67
    assert (tmp_path / "three" / "three_lane_evaluation_results.json").exists()
    assert (tmp_path / "three" / "three_lane_evaluation_report.md").exists()
