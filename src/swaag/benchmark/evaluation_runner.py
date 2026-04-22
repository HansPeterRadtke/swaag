from __future__ import annotations

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

from swaag.benchmark.task_definitions import BENCHMARK_DIFFICULTY_ORDER
from swaag.fsops import ensure_dir, write_text
from swaag.utils import stable_json_dumps


def _parse_junit_results(xml_path: Path) -> dict[str, Any]:
    if not xml_path.exists():
        return {
            "summary": {
                "total_tests": 0,
                "executed_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
                "percent": 0.0,
            },
            "tests": [],
        }
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    tests: list[dict[str, Any]] = []
    for testcase in root.iter("testcase"):
        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "")
        nodeid = f"{classname}::{name}" if classname else name
        status = "passed"
        detail = ""
        score_percent: float | None = 100.0
        if testcase.find("failure") is not None:
            status = "failed"
            detail = testcase.find("failure").attrib.get("message", "") or (testcase.find("failure").text or "").strip()
            score_percent = 0.0
        elif testcase.find("error") is not None:
            status = "error"
            detail = testcase.find("error").attrib.get("message", "") or (testcase.find("error").text or "").strip()
            score_percent = 0.0
        elif testcase.find("skipped") is not None:
            status = "skipped"
            detail = testcase.find("skipped").attrib.get("message", "") or (testcase.find("skipped").text or "").strip()
            score_percent = None
        tests.append(
            {
                "nodeid": nodeid,
                "classname": classname,
                "name": name,
                "status": status,
                "score_percent": score_percent,
                "duration_seconds": float(testcase.attrib.get("time", "0") or 0.0),
                "detail": detail,
            }
        )
    executed = [item for item in tests if item["status"] != "skipped"]
    passed = [item for item in executed if item["status"] == "passed"]
    failed = [item for item in executed if item["status"] in {"failed", "error"}]
    skipped = [item for item in tests if item["status"] == "skipped"]
    return {
        "summary": {
            "total_tests": len(tests),
            "executed_tests": len(executed),
            "passed_tests": len(passed),
            "failed_tests": len(failed),
            "skipped_tests": len(skipped),
            "percent": round((len(passed) / len(executed) * 100.0), 2) if executed else 0.0,
        },
        "tests": tests,
    }


def _run_pytest_category(*, output_dir: Path, category_name: str, pytest_args: Sequence[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    ensure_dir(output_dir)
    junit_path = output_dir / f"{category_name}.junit.xml"
    stdout_path = output_dir / f"{category_name}.stdout.txt"
    stderr_path = output_dir / f"{category_name}.stderr.txt"
    command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}", *pytest_args]
    completed = subprocess.run(command, check=False, text=True, capture_output=True, env=env or os.environ.copy())
    write_text(stdout_path, completed.stdout, encoding="utf-8")
    write_text(stderr_path, completed.stderr, encoding="utf-8")
    parsed = _parse_junit_results(junit_path)
    payload = {
        "category": category_name,
        "command": command,
        "exit_code": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "junit_path": str(junit_path),
        **parsed,
    }
    write_text(output_dir / f"{category_name}_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text(output_dir / f"{category_name}_report.md", _render_pytest_category_report(payload), encoding="utf-8")
    return payload


def _render_pytest_category_report(payload: dict[str, Any]) -> str:
    title = "Deterministic code-correctness tests" if payload["category"] == "code_correctness" else "Cached agent tests"
    lines = [
        f"# {title}",
        "",
        f"- percent: `{payload['summary']['percent']:.2f}%`",
        f"- passed: `{payload['summary']['passed_tests']}`",
        f"- failed: `{payload['summary']['failed_tests']}`",
        f"- skipped: `{payload['summary']['skipped_tests']}`",
        "",
        "## Per-test results",
        "",
    ]
    for test in payload["tests"]:
        score_text = "n/a" if test["score_percent"] is None else f"{float(test['score_percent']):.2f}%"
        lines.append(f"- `{test['nodeid']}`: `{test['status']}` / `{score_text}`")
    return "\n".join(lines) + "\n"


def run_code_correctness_category(*, output_dir: Path, pytest_args: Sequence[str] | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    if pytest_args is None:
        from swaag.test_categories import CODE_CORRECTNESS_TEST_FILES, project_root as _project_root

        base = _project_root()
        pytest_args = sorted(str(base / f) for f in CODE_CORRECTNESS_TEST_FILES)
    return _run_pytest_category(output_dir=output_dir, category_name="code_correctness", pytest_args=pytest_args, env=env)


def run_agent_test_category(*, output_dir: Path, pytest_args: Sequence[str] | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    if pytest_args is None:
        from swaag.test_categories import AGENT_TEST_FILES, project_root as _project_root

        base = _project_root()
        pytest_args = sorted(str(base / f) for f in AGENT_TEST_FILES)
    return _run_pytest_category(output_dir=output_dir, category_name="agent_test", pytest_args=pytest_args, env=env)


def _deterministic_failed(payload: dict[str, Any]) -> bool:
    return (
        float(payload["summary"].get("percent", 0.0)) < 100.0
        or int(payload["summary"].get("failed_tests", 0)) > 0
        or int(payload.get("exit_code", 0)) != 0
    )


def render_test_category_report(payload: dict[str, Any]) -> str:
    code = payload["code_correctness"]
    agent = payload.get("agent_test")
    lines = [
        "# SWAAG test-category report",
        "",
        f"- overall_percent: `{payload['overall_percent']:.2f}%`",
        f"- code_correctness_percent: `{code['summary']['percent']:.2f}%`",
        f"- agent_test_percent: `{float(agent['summary']['percent']) if isinstance(agent, dict) else 0.0:.2f}%`",
        f"- status: `{payload.get('status', 'complete')}`",
        f"- skip_reason: `{payload.get('skip_reason', '')}`" if payload.get("skip_reason") else "",
        "",
        "## Category summaries",
        "",
        f"- code_correctness: `{code['summary']['passed_tests']}` passed / `{code['summary']['failed_tests']}` failed / `{code['summary']['skipped_tests']}` skipped",
        (
            f"- agent_test: `{agent['summary']['passed_tests']}` passed / `{agent['summary']['failed_tests']}` failed / `{agent['summary']['skipped_tests']}` skipped"
            if isinstance(agent, dict)
            else "- agent_test: not run because code_correctness was not 100% green"
        ),
        "",
        "## Artifact paths",
        "",
        "- code_correctness:",
        "  - `code_correctness/code_correctness_results.json`",
        "  - `code_correctness/code_correctness_report.md`",
        "- agent_test:",
        "  - `agent_test/agent_test_results.json`",
        "  - `agent_test/agent_test_report.md`",
    ]
    return "\n".join(lines) + "\n"


def run_test_category_evaluation(
    *,
    output_dir: Path,
    clean: bool = False,
    functional_pytest_args: Sequence[str] | None = None,
    agent_pytest_args: Sequence[str] | None = None,
    **_: Any,
) -> dict[str, Any]:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    code = run_code_correctness_category(output_dir=output_dir / "code_correctness", pytest_args=functional_pytest_args)
    if _deterministic_failed(code):
        code_percent = float(code["summary"].get("percent", 0.0))
        payload = {
            "code_correctness": code,
            "agent_test": None,
            "category_scores": {"code_correctness": code_percent, "agent_test": 0.0},
            "overall_percent": round(code_percent / 2.0, 2),
            "status": "code_correctness_failed",
            "skipped_categories": ["agent_test"],
            "skip_reason": "code_correctness must be 100% before agent_test can run",
        }
        write_text(output_dir / "test_categories_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
        write_text(output_dir / "test_categories_report.md", render_test_category_report(payload), encoding="utf-8")
        return payload
    agent = run_agent_test_category(output_dir=output_dir / "agent_test", pytest_args=agent_pytest_args)
    category_scores = {"code_correctness": float(code["summary"]["percent"]), "agent_test": float(agent["summary"]["percent"])}
    overall_percent = round(sum(category_scores.values()) / len(category_scores), 2)
    payload = {
        "code_correctness": code,
        "agent_test": agent,
        "category_scores": category_scores,
        "overall_percent": overall_percent,
        "status": "complete",
        "skipped_categories": [],
    }
    write_text(output_dir / "test_categories_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text(output_dir / "test_categories_report.md", render_test_category_report(payload), encoding="utf-8")
    return payload


def run_full_evaluation(
    *,
    output_dir: Path,
    clean: bool = False,
    functional_pytest_args: Sequence[str] | None = None,
    benchmark_task_ids: list[str] | None = None,
    live_subset: bool = False,
    use_live_model: bool = False,
    model_base_url: str | None = None,
    timeout_seconds: int | None = None,
    connect_timeout_seconds: int | None = None,
    model_profile: str | None = None,
    structured_output_mode: str | None = None,
    progress_poll_seconds: float | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Legacy benchmark evaluator retained for explicit benchmark runs, not test categories."""
    from swaag.benchmark.benchmark_runner import run_benchmarks

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    code = run_code_correctness_category(output_dir=output_dir / "code_correctness", pytest_args=functional_pytest_args)
    agent_report = run_benchmarks(
        output_dir=output_dir / "agent_evaluation",
        task_ids=benchmark_task_ids,
        clean=True,
        live_subset=live_subset,
        use_live_model=use_live_model,
        model_base_url=model_base_url,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        model_profile=model_profile,
        structured_output_mode=structured_output_mode,
        progress_poll_seconds=progress_poll_seconds,
        seeds=seeds,
    )
    difficulty_scores = {
        difficulty: float(agent_report["summary"].get("score_by_difficulty", {}).get(difficulty, 0.0))
        for difficulty in BENCHMARK_DIFFICULTY_ORDER
    }
    group_scores = {"code_correctness": float(code["summary"]["percent"]), **difficulty_scores}
    overall_percent = round(sum(group_scores.values()) / len(group_scores), 2) if group_scores else 0.0
    payload = {"code_correctness": code, "agent_evaluation": {**agent_report, "difficulty_tier_scores": difficulty_scores}, "group_scores": group_scores, "overall_percent": overall_percent}
    write_text(output_dir / "evaluation_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text(output_dir / "evaluation_report.md", render_evaluation_report(payload), encoding="utf-8")
    return payload


def render_evaluation_report(payload: dict[str, Any]) -> str:
    lines = [
        "# SWAAG benchmark evaluation report",
        "",
        f"- overall_percent: `{payload['overall_percent']:.2f}%`",
        f"- code_correctness_percent: `{payload['code_correctness']['summary']['percent']:.2f}%`",
        "",
        "## Difficulty tier scores",
        "",
    ]
    for difficulty in BENCHMARK_DIFFICULTY_ORDER:
        score = float(payload["agent_evaluation"].get("difficulty_tier_scores", {}).get(difficulty, 0.0))
        lines.append(f"- `{difficulty}`: `{score:.2f}%`")
    return "\n".join(lines) + "\n"
