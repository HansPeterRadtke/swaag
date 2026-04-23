from __future__ import annotations

import subprocess
from pathlib import Path

from swaag.benchmark import benchmark_runner
from swaag.benchmark.evaluation_runner import (
    run_agent_test_category,
    run_code_correctness_category,
    run_full_evaluation,
    run_test_category_evaluation,
)
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


def test_run_agent_test_category_writes_reports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("swaag.benchmark.evaluation_runner.subprocess.run", lambda command, **kwargs: _fake_junit(command, failures=0))

    payload = run_agent_test_category(output_dir=tmp_path / "agent")

    assert payload["category"] == "agent_test"
    assert payload["summary"]["percent"] == 100.0
    assert payload["full_cached_benchmark_catalog"]["total_tasks"] >= 190
    assert payload["full_cached_benchmark_catalog"]["includes_extremely_hard"] is True
    assert set(payload["full_cached_benchmark_catalog"]["counts_by_difficulty"]) == {
        "easy",
        "extremely_easy",
        "extremely_hard",
        "hard",
        "normal",
    }
    assert (tmp_path / "agent" / "agent_test_results.json").exists()
    assert (tmp_path / "agent" / "agent_test_report.md").exists()
    assert "Full Cached Benchmark Catalog" in (tmp_path / "agent" / "agent_test_report.md").read_text(encoding="utf-8")


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
        lambda *, output_dir, pytest_args=None, env=None: {
            "summary": {"percent": 90.0, "passed_tests": 9, "failed_tests": 1, "skipped_tests": 0, "total_tests": 10, "executed_tests": 10},
            "tests": [],
            "exit_code": 1,
        },
    )

    payload = run_test_category_evaluation(output_dir=tmp_path / "categories", clean=True)

    assert payload["category_scores"] == {"code_correctness": 100.0, "agent_test": 90.0}
    assert payload["overall_percent"] == 95.0
    assert set(payload) >= {"code_correctness", "agent_test", "category_scores"}
    assert "agent_behavior_validation" not in payload
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
    assert payload["skipped_categories"] == ["agent_test"]
    assert payload["agent_test"] is None
    assert payload["category_scores"] == {"code_correctness": 80.0, "agent_test": 0.0}
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
