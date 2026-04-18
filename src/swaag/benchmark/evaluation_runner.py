from __future__ import annotations

import os
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


def run_functional_correctness_lane(
    *,
    output_dir: Path,
    pytest_args: Sequence[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    junit_path = output_dir / "functional_correctness.junit.xml"
    stdout_path = output_dir / "functional_correctness.stdout.txt"
    stderr_path = output_dir / "functional_correctness.stderr.txt"
    command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}", *(pytest_args or [])]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=env or os.environ.copy(),
    )
    write_text(stdout_path, completed.stdout, encoding="utf-8")
    write_text(stderr_path, completed.stderr, encoding="utf-8")
    parsed = _parse_junit_results(junit_path)
    payload = {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "junit_path": str(junit_path),
        **parsed,
    }
    write_text(output_dir / "functional_correctness_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# Functional correctness lane",
        "",
        f"- Percent: `{payload['summary']['percent']:.2f}%`",
        f"- Passed: `{payload['summary']['passed_tests']}`",
        f"- Failed: `{payload['summary']['failed_tests']}`",
        f"- Skipped: `{payload['summary']['skipped_tests']}`",
        "",
    ]
    for test in payload["tests"]:
        score_text = "n/a" if test["score_percent"] is None else f"{float(test['score_percent']):.2f}%"
        report_lines.append(f"- `{test['nodeid']}`: `{test['status']}` / `{score_text}`")
    write_text(output_dir / "functional_correctness_report.md", "\n".join(report_lines) + "\n", encoding="utf-8")
    return payload


def render_evaluation_report(payload: dict[str, Any]) -> str:
    lines = [
        "# SWAAG evaluation report",
        "",
        f"- overall_percent: `{payload['overall_percent']:.2f}%`",
        f"- functional_correctness_percent: `{payload['functional_correctness']['summary']['percent']:.2f}%`",
        "",
        "## Difficulty tier scores",
        "",
    ]
    for difficulty in BENCHMARK_DIFFICULTY_ORDER:
        score = float(payload["agent_evaluation"]["difficulty_tier_scores"].get(difficulty, 0.0))
        lines.append(f"- `{difficulty}`: `{score:.2f}%`")
    lines.extend(["", "## Group scores", ""])
    for group_name, score in payload["group_scores"].items():
        lines.append(f"- `{group_name}`: `{float(score):.2f}%`")
    lines.extend(["", "## Lowest-scoring agent tasks", ""])
    tasks = sorted(payload["agent_evaluation"]["tasks"], key=lambda item: (float(item.get("score_percent", 0.0)), item["task_id"]))
    for item in tasks[:10]:
        lines.append(f"- `{item['task_id']}` / `{item['difficulty']}`: `{float(item.get('score_percent', 0.0)):.2f}%`")
    return "\n".join(lines) + "\n"


def run_live_agent_evaluation_lane(
    *,
    output_dir: Path,
    benchmark_task_ids: list[str] | None = None,
    live_subset: bool = True,
    model_base_url: str | None = None,
    timeout_seconds: int | None = None,
    connect_timeout_seconds: int | None = None,
    model_profile: str | None = None,
    structured_output_mode: str | None = None,
    progress_poll_seconds: float | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    from swaag.benchmark.benchmark_runner import run_benchmarks

    ensure_dir(output_dir)
    report = run_benchmarks(
        output_dir=output_dir,
        task_ids=benchmark_task_ids,
        clean=True,
        live_subset=live_subset,
        use_live_model=True,
        model_base_url=model_base_url,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        model_profile=model_profile,
        structured_output_mode=structured_output_mode,
        progress_poll_seconds=progress_poll_seconds,
        seeds=seeds,
    )
    difficulty_scores = {
        difficulty: float(report["summary"]["score_by_difficulty"].get(difficulty, 0.0))
        for difficulty in BENCHMARK_DIFFICULTY_ORDER
    }
    payload = {
        **report,
        "difficulty_tier_scores": difficulty_scores,
        "percent": float(report["summary"].get("average_task_score_percent", 0.0)),
    }
    write_text(output_dir / "live_agent_evaluation_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def render_three_lane_evaluation_report(payload: dict[str, Any]) -> str:
    deterministic_lane = payload["deterministic_correctness"]
    regression_lane = payload["agent_loop_regression"]
    live_lane = payload["live_agent_evaluation"]
    lines = [
        "# SWAAG three-lane evaluation report",
        "",
        f"- final_overall_percent: `{payload['overall_percent']:.2f}%`",
        f"- deterministic_correctness_percent: `{deterministic_lane['summary']['percent']:.2f}%`",
        f"- agent_loop_regression_percent: `{regression_lane['summary']['percent']:.2f}%`",
        f"- live_agent_evaluation_percent: `{float(live_lane['percent']):.2f}%`",
        "",
        "## Lane summaries",
        "",
        f"- deterministic_correctness: `{deterministic_lane['summary']['passed_tests']}` passed / `{deterministic_lane['summary']['failed_tests']}` failed / `{deterministic_lane['summary']['skipped_tests']}` skipped",
        f"- agent_loop_regression: `{regression_lane['summary'].get('passed_tests', 0)}` passed / `{regression_lane['summary'].get('failed_tests', 0)}` failed across `{regression_lane['summary'].get('total_families', len(regression_lane.get('results', [])))}` families",
        f"- live_agent_evaluation: `{len(live_lane.get('tasks', []))}` tasks / `{float(live_lane.get('summary', {}).get('average_task_score_percent', live_lane['percent'])):.2f}%` average",
        "",
        "## Live difficulty tiers",
        "",
    ]
    for difficulty in BENCHMARK_DIFFICULTY_ORDER:
        lines.append(f"- `{difficulty}`: `{float(live_lane['difficulty_tier_scores'].get(difficulty, 0.0)):.2f}%`")
    lines.extend(["", "## Agent-loop regression families", ""])
    for family in regression_lane.get("results", []):
        lines.append(
            f"- `{family['family_id']}`: `{float(family['score_percent']):.2f}%` / `{family['status']}` / `{family['passed_tests']}` passed / `{family['failed_tests']}` failed"
        )
    lines.extend(["", "## Lowest-scoring live tasks", ""])
    tasks = sorted(live_lane.get("tasks", []), key=lambda item: (float(item.get("score_percent", 0.0)), item["task_id"]))
    for item in tasks[:10]:
        lines.extend(
            [
                f"- `{item['task_id']}` / `{item['difficulty']}` / `{item.get('task_type', 'unknown')}`: `{float(item.get('score_percent', 0.0)):.2f}%`",
                f"  - success: `{item.get('success', False)}` / failure_category: `{item.get('failure_category', '')}` / reason: `{item.get('failure_reason', '')}`",
            ]
        )
        rubric = item.get("rubric_breakdown", {})
        if isinstance(rubric, dict) and rubric:
            weakest = sorted(
                rubric.items(),
                key=lambda kv: float(kv[1].get("percent", 0.0)),
            )[:3]
            for rubric_name, rubric_payload in weakest:
                lines.append(
                    f"  - rubric `{rubric_name}`: `{float(rubric_payload.get('earned', 0.0)):.2f}/{float(rubric_payload.get('weight', 0.0)):.2f}` (`{float(rubric_payload.get('percent', 0.0)):.2f}%`)"
                )
    lines.extend(
        [
            "",
            "## Artifact paths",
            "",
            "- deterministic_correctness:",
            "  - `deterministic_correctness/functional_correctness_results.json`",
            "  - `deterministic_correctness/functional_correctness_report.md`",
            "- agent_loop_regression:",
            "  - `agent_loop_regression/agent_loop_regression_results.json`",
            "  - `agent_loop_regression/agent_loop_regression_report.md`",
            "- live_agent_evaluation:",
            "  - `live_agent_evaluation/live_agent_evaluation_results.json`",
            "  - `live_agent_evaluation/benchmark_results.json`",
            "  - `live_agent_evaluation/benchmark_report.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_three_lane_evaluation(
    *,
    output_dir: Path,
    clean: bool = False,
    functional_pytest_args: Sequence[str] | None = None,
    regression_family_ids: Sequence[str] | None = None,
    benchmark_task_ids: list[str] | None = None,
    live_subset: bool = True,
    model_base_url: str | None = None,
    timeout_seconds: int | None = None,
    connect_timeout_seconds: int | None = None,
    model_profile: str | None = None,
    structured_output_mode: str | None = None,
    progress_poll_seconds: float | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    from swaag.benchmark.agent_regression import run_agent_loop_regression_lane

    if clean and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    deterministic = run_functional_correctness_lane(
        output_dir=output_dir / "deterministic_correctness",
        pytest_args=functional_pytest_args,
    )
    regression = run_agent_loop_regression_lane(
        output_dir=output_dir / "agent_loop_regression",
        family_ids=regression_family_ids,
        clean=True,
    )
    live = run_live_agent_evaluation_lane(
        output_dir=output_dir / "live_agent_evaluation",
        benchmark_task_ids=benchmark_task_ids,
        live_subset=live_subset,
        model_base_url=model_base_url,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        model_profile=model_profile,
        structured_output_mode=structured_output_mode,
        progress_poll_seconds=progress_poll_seconds,
        seeds=seeds,
    )
    lane_scores = {
        "deterministic_correctness": float(deterministic["summary"]["percent"]),
        "agent_loop_regression": float(regression["summary"]["percent"]),
        "live_agent_evaluation": float(live["percent"]),
    }
    overall_percent = round(sum(lane_scores.values()) / len(lane_scores), 2) if lane_scores else 0.0
    payload = {
        "deterministic_correctness": deterministic,
        "agent_loop_regression": regression,
        "live_agent_evaluation": live,
        "lane_scores": lane_scores,
        "overall_percent": overall_percent,
    }
    write_text(output_dir / "three_lane_evaluation_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text(output_dir / "three_lane_evaluation_report.md", render_three_lane_evaluation_report(payload), encoding="utf-8")
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
    from swaag.benchmark.benchmark_runner import run_benchmarks

    if clean and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    functional = run_functional_correctness_lane(
        output_dir=output_dir / "functional_correctness",
        pytest_args=functional_pytest_args,
    )
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
        difficulty: float(agent_report["summary"]["score_by_difficulty"].get(difficulty, 0.0))
        for difficulty in BENCHMARK_DIFFICULTY_ORDER
    }
    group_scores = {
        "functional_correctness": float(functional["summary"]["percent"]),
        **difficulty_scores,
    }
    overall_percent = round(sum(group_scores.values()) / len(group_scores), 2) if group_scores else 0.0
    payload = {
        "functional_correctness": functional,
        "agent_evaluation": {
            **agent_report,
            "difficulty_tier_scores": difficulty_scores,
        },
        "group_scores": group_scores,
        "overall_percent": overall_percent,
    }
    write_text(output_dir / "evaluation_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text(output_dir / "evaluation_report.md", render_evaluation_report(payload), encoding="utf-8")
    return payload
