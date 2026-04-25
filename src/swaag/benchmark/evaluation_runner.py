from __future__ import annotations

import hashlib
import json
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
    lines = [
        "# Code Correctness Report",
        "",
        f"- percent: `{payload['summary']['percent']:.2f}%`",
        f"- total_checks: `{payload['summary']['executed_tests']}`",
        f"- passed: `{payload['summary']['passed_tests']}`",
        f"- failed: `{payload['summary']['failed_tests']}`",
        f"- skipped: `{payload['summary']['skipped_tests']}`",
        f"- binary_result: `{'passed' if payload['exit_code'] == 0 and payload['summary']['failed_tests'] == 0 else 'failed'}`",
        "",
        "## Per-test results",
        "",
    ]
    for test in payload["tests"]:
        score_text = "n/a" if test["score_percent"] is None else f"{float(test['score_percent']):.2f}%"
        lines.append(f"- `{test['nodeid']}`: `{test['status']}` / `{score_text}`")
    return "\n".join(lines) + "\n"


def _deterministic_failed(payload: dict[str, Any]) -> bool:
    return (
        float(payload["summary"].get("percent", 0.0)) < 100.0
        or int(payload["summary"].get("failed_tests", 0)) > 0
        or int(payload.get("exit_code", 0)) != 0
    )


def _group_average(mapping: dict[str, float]) -> float:
    return round(sum(float(value) for value in mapping.values()) / len(mapping), 2) if mapping else 0.0


def _score_mapping_from_tasks(tasks: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for task in tasks:
        group = str(task.get(key, "")).strip()
        if not group:
            continue
        score = float(task.get("score_percent", 0.0))
        buckets.setdefault(group, []).append(score)
    return {
        group: round(sum(scores) / len(scores), 2)
        for group, scores in sorted(buckets.items())
        if scores
    }


def _build_agent_test_score_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    tasks = list(report.get("tasks", []))
    difficulty_scores = {key: float(value) for key, value in dict(summary.get("score_by_difficulty", {})).items()}
    family_scores = {key: float(value) for key, value in dict(summary.get("score_by_family", {})).items()}
    if not difficulty_scores and tasks:
        difficulty_scores = _score_mapping_from_tasks(tasks, "difficulty")
    if not family_scores and tasks:
        family_scores = _score_mapping_from_tasks(tasks, "task_type")
    difficulty_group_average = float(summary.get("difficulty_group_average_percent", _group_average(difficulty_scores)))
    family_group_average = float(summary.get("family_group_average_percent", _group_average(family_scores)))
    group_average_sources = [value for value in (difficulty_group_average, family_group_average) if value > 0.0]
    group_average = round(sum(group_average_sources) / len(group_average_sources), 2) if group_average_sources else 0.0
    total_tasks = int(summary.get("total_tasks", 0))
    successful_tasks = int(summary.get("successful_tasks", 0))
    full_task_success_percent = float(
        summary.get(
            "full_task_success_percent",
            round(successful_tasks / total_tasks * 100.0, 2) if total_tasks else 0.0,
        )
    )
    return {
        "group_average_percent": group_average,
        "difficulty_group_average_percent": difficulty_group_average,
        "family_group_average_percent": family_group_average,
        "full_task_success_percent": full_task_success_percent,
        "average_task_score_percent": float(summary.get("average_task_score_percent", 0.0)),
        "detailed_substep_score_percent": None,
        "detailed_substep_score_note": "Intentionally omitted because reliable detailed sub-step scoring is not yet implemented.",
        "group_scores_by_difficulty": difficulty_scores,
        "group_scores_by_family": family_scores,
    }


def _full_catalog_cache_key(tasks: Sequence[Any]) -> str:
    payload = [
        {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "difficulty": task.difficulty,
            "tags": list(task.tags),
            "description": task.description,
            "setup_instructions": list(task.setup_instructions),
        }
        for task in tasks
    ]
    raw = json.dumps({"version": 7, "tasks": payload}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _valid_full_catalog_report(report_path: Path, tasks: Sequence[Any]) -> bool:
    if not report_path.exists():
        return False
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_ids = {task.task_id for task in tasks}
    actual_ids = {str(item.get("task_id", "")) for item in payload.get("tasks", [])}
    metadata = payload.get("run_metadata", {})
    seed_results = [
        seed_result
        for task in payload.get("tasks", [])
        for seed_result in task.get("metrics", {}).get("seed_results", [])
        if isinstance(seed_result, dict)
    ]
    return (
        payload.get("summary", {}).get("total_tasks") == len(tasks)
        and actual_ids == expected_ids
        and metadata.get("agent_behavior_mode") == "cached"
        and metadata.get("replay_cache_enabled") is True
        and bool(seed_results)
        and all(seed.get("replay_cache", {}).get("cassette_path") for seed in seed_results)
    )


def _copy_cached_full_catalog(cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(cache_dir, output_dir)
    old_root = str(cache_dir)
    new_root = str(output_dir)
    results_path = output_dir / "agent_test_cached_results.json"
    report_path = output_dir / "agent_test_cached_report.md"
    results_text = results_path.read_text(encoding="utf-8").replace(old_root, new_root)
    write_text(results_path, results_text, encoding="utf-8")
    if report_path.exists():
        write_text(report_path, report_path.read_text(encoding="utf-8").replace(old_root, new_root), encoding="utf-8")
    payload = json.loads(results_text)
    payload.setdefault("run_metadata", {})["artifact_reused_from"] = str(cache_dir)
    return payload


def _reuse_full_catalog_benchmark_artifact(output_dir: Path, benchmark_task_ids: Sequence[str] | None, clean: bool) -> dict[str, Any] | None:
    if benchmark_task_ids:
        return None
    from swaag.benchmark.task_definitions import get_benchmark_tasks

    tasks = get_benchmark_tasks()
    artifact_root = Path(os.environ.get("SWAAG_FULL_CACHED_BENCHMARK_ARTIFACT_ROOT", "/tmp/swaag-full-cached-benchmark-catalog"))
    cache_dir = artifact_root / _full_catalog_cache_key(tasks)
    report_path = cache_dir / "agent_test_cached_results.json"
    if not _valid_full_catalog_report(report_path, tasks):
        return None
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    return _copy_cached_full_catalog(cache_dir, output_dir)


def _seed_full_catalog_replay_cache(output_dir: Path, benchmark_task_ids: Sequence[str] | None) -> None:
    if benchmark_task_ids:
        return
    from swaag.benchmark.task_definitions import get_benchmark_tasks

    target = output_dir / "replay_cache"
    if target.exists():
        return
    artifact_root = Path(os.environ.get("SWAAG_FULL_CACHED_BENCHMARK_ARTIFACT_ROOT", "/tmp/swaag-full-cached-benchmark-catalog"))
    source = artifact_root / _full_catalog_cache_key(get_benchmark_tasks()) / "replay_cache"
    if source.exists():
        shutil.copytree(source, target)


def _render_agent_test_category_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    scores = payload["score_summary"]
    lines = [
        "# Agent Test Benchmark Report",
        "",
        f"- execution_mode: `{payload.get('execution_mode', 'executed_cached_benchmark')}`",
        f"- total_tasks: `{summary['total_tasks']}`",
        f"- successful_tasks: `{summary['successful_tasks']}`",
        f"- failed_tasks: `{summary['failed_tasks']}`",
        f"- false_positives: `{summary['false_positives']}`",
        f"- full_task_success_percent: `{scores['full_task_success_percent']:.2f}%`",
        f"- group_average_percent: `{scores['group_average_percent']:.2f}%`",
        f"- difficulty_group_average_percent: `{scores['difficulty_group_average_percent']:.2f}%`",
        f"- family_group_average_percent: `{scores['family_group_average_percent']:.2f}%`",
        f"- average_task_score_percent: `{scores['average_task_score_percent']:.2f}%`",
        f"- detailed_substep_score: `{scores['detailed_substep_score_note']}`",
        "",
        "## Group Scores By Difficulty",
        "",
    ]
    for difficulty in BENCHMARK_DIFFICULTY_ORDER:
        if difficulty in scores["group_scores_by_difficulty"]:
            lines.append(f"- `{difficulty}`: `{scores['group_scores_by_difficulty'][difficulty]:.2f}%`")
    lines.extend(["", "## Group Scores By Family", ""])
    for family, percent in sorted(scores["group_scores_by_family"].items()):
        lines.append(f"- `{family}`: `{percent:.2f}%`")
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            f"- detailed_results: `{payload['cached_benchmark_results_path']}`",
            f"- detailed_report: `{payload['cached_benchmark_report_path']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_code_correctness_category(*, output_dir: Path, pytest_args: Sequence[str] | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    if pytest_args is None:
        from swaag.test_categories import CODE_CORRECTNESS_TEST_FILES, project_root as _project_root

        base = _project_root()
        pytest_args = sorted(str(base / f) for f in CODE_CORRECTNESS_TEST_FILES)
    return _run_pytest_category(output_dir=output_dir, category_name="code_correctness", pytest_args=pytest_args, env=env)


def run_agent_test_category(
    *,
    output_dir: Path,
    benchmark_task_ids: Sequence[str] | None = None,
    clean: bool = False,
    pytest_args: Sequence[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    del env
    if pytest_args:
        raise ValueError("agent_test no longer accepts pytest_args because it runs the real cached benchmark, not pytest benchmark wrappers")
    from swaag.benchmark.benchmark_runner import run_benchmarks

    reused_report = _reuse_full_catalog_benchmark_artifact(output_dir, benchmark_task_ids, clean)
    if reused_report is not None:
        benchmark_report = reused_report
        execution_mode = "reused_cached_artifact"
    else:
        ensure_dir(output_dir)
        _seed_full_catalog_replay_cache(output_dir, benchmark_task_ids)
        benchmark_report = run_benchmarks(
            output_dir=output_dir,
            task_ids=list(benchmark_task_ids) if benchmark_task_ids is not None else None,
            clean=clean,
            live_subset=False,
            use_live_model=False,
            agent_behavior_mode="cached",
        )
        execution_mode = "executed_cached_benchmark"
    payload = {
        "category": "agent_test",
        "status": "complete",
        "execution_mode": execution_mode,
        "summary": benchmark_report["summary"],
        "aggregate_metrics": benchmark_report["aggregate_metrics"],
        "run_metadata": benchmark_report.get("run_metadata", {}),
        "tasks": benchmark_report.get("tasks", []),
        "score_summary": _build_agent_test_score_summary(benchmark_report),
        "cached_benchmark_results_path": str(output_dir / "agent_test_cached_results.json"),
        "cached_benchmark_report_path": str(output_dir / "agent_test_cached_report.md"),
    }
    write_text(output_dir / "agent_test_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text(output_dir / "agent_test_report.md", _render_agent_test_category_report(payload), encoding="utf-8")
    return payload


def render_test_category_report(payload: dict[str, Any]) -> str:
    code = payload["code_correctness"]
    agent = payload.get("agent_test")
    lines = [
        "# SWAAG Test Category Report",
        "",
        f"- status: `{payload.get('status', 'complete')}`",
        f"- code_correctness_binary_result: `{'passed' if payload.get('code_correctness_binary_passed') else 'failed'}`",
        f"- agent_test_ran: `{payload.get('agent_test_ran', False)}`",
    ]
    if payload.get("skip_reason"):
        lines.append(f"- skip_reason: `{payload['skip_reason']}`")
    lines.extend(
        [
            "",
            "## Code Correctness",
            "",
            f"- total_checks: `{code['summary']['executed_tests']}`",
            f"- passed: `{code['summary']['passed_tests']}`",
            f"- failed: `{code['summary']['failed_tests']}`",
            f"- skipped: `{code['summary']['skipped_tests']}`",
            f"- percent: `{code['summary']['percent']:.2f}%`",
            "",
            "## Agent Test",
            "",
        ]
    )
    if isinstance(agent, dict):
        score_summary = agent["score_summary"]
        lines.extend(
            [
                f"- execution_mode: `{agent.get('execution_mode', 'executed_cached_benchmark')}`",
                f"- total_tasks: `{agent['summary']['total_tasks']}`",
                f"- successful_tasks: `{agent['summary']['successful_tasks']}`",
                f"- failed_tasks: `{agent['summary']['failed_tasks']}`",
                f"- false_positives: `{agent['summary']['false_positives']}`",
                f"- full_task_success_percent: `{score_summary['full_task_success_percent']:.2f}%`",
                f"- group_average_percent: `{score_summary['group_average_percent']:.2f}%`",
                f"- difficulty_group_average_percent: `{score_summary['difficulty_group_average_percent']:.2f}%`",
                f"- family_group_average_percent: `{score_summary['family_group_average_percent']:.2f}%`",
                f"- average_task_score_percent: `{score_summary['average_task_score_percent']:.2f}%`",
                f"- detailed_substep_score: `{score_summary['detailed_substep_score_note']}`",
            ]
        )
    else:
        lines.append("- not run because code_correctness was not 100% green")
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            "- code_correctness:",
            "  - `code_correctness/code_correctness_results.json`",
            "  - `code_correctness/code_correctness_report.md`",
            "- agent_test:",
            "  - `agent_test/agent_test_results.json`",
            "  - `agent_test/agent_test_report.md`",
            "  - `agent_test/agent_test_cached_results.json`",
            "  - `agent_test/agent_test_cached_report.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_test_category_evaluation(
    *,
    output_dir: Path,
    clean: bool = False,
    functional_pytest_args: Sequence[str] | None = None,
    benchmark_task_ids: Sequence[str] | None = None,
    **_: Any,
) -> dict[str, Any]:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    code = run_code_correctness_category(output_dir=output_dir / "code_correctness", pytest_args=functional_pytest_args)
    if _deterministic_failed(code):
        payload = {
            "status": "code_correctness_failed",
            "code_correctness_binary_passed": False,
            "agent_test_ran": False,
            "skip_reason": "code_correctness must be 100% before agent_test can run",
            "code_correctness": code,
            "agent_test": None,
        }
        write_text(output_dir / "test_categories_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
        write_text(output_dir / "test_categories_report.md", render_test_category_report(payload), encoding="utf-8")
        return payload
    agent = run_agent_test_category(output_dir=output_dir / "agent_test", benchmark_task_ids=benchmark_task_ids, clean=False)
    payload = {
        "status": "complete",
        "code_correctness_binary_passed": True,
        "agent_test_ran": True,
        "code_correctness": code,
        "agent_test": agent,
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
    payload = {
        "code_correctness": code,
        "agent_evaluation": {**agent_report, "difficulty_tier_scores": difficulty_scores},
        "group_scores": group_scores,
        "overall_percent": overall_percent,
    }
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
