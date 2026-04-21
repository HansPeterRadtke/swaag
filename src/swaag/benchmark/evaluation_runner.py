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


def _agent_behavior_mode_suffix(mode: str) -> str:
    if mode == "cached":
        return "cached"
    if mode == "no-cache-validation":
        return "validation"
    raise ValueError(f"Unsupported agent behavior mode: {mode}")


def _agent_behavior_mode_label(mode: str) -> str:
    if mode == "cached":
        return "agent behavior tests (cached mode)"
    if mode == "no-cache-validation":
        return "agent behavior tests (no-cache validation mode)"
    raise ValueError(f"Unsupported agent behavior mode: {mode}")


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
    if pytest_args is None:
        from swaag.test_categories import CODE_CORRECTNESS_TEST_FILES, project_root as _project_root

        base = _project_root()
        pytest_args = sorted(str(base / f) for f in CODE_CORRECTNESS_TEST_FILES)
    command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}", *pytest_args]
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
        "# Deterministic correctness tests",
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


def render_agent_behavior_tests_report(payload: dict[str, Any]) -> str:
    mode = str(payload.get("mode", "cached"))
    label = _agent_behavior_mode_label(mode)
    summary = payload.get("summary", {})
    lines = [
        f"# {label[0].upper()}{label[1:]}",
        "",
        f"- percent: `{float(payload.get('percent', 0.0)):.2f}%`",
        f"- task_count: `{int(summary.get('total_tasks', len(payload.get('tasks', []))))}`",
        f"- successful_tasks: `{int(summary.get('successful_tasks', 0))}`",
        f"- failed_tasks: `{int(summary.get('failed_tasks', 0))}`",
        "",
        "## Difficulty tier scores",
        "",
    ]
    for difficulty in BENCHMARK_DIFFICULTY_ORDER:
        lines.append(f"- `{difficulty}`: `{float(payload['difficulty_tier_scores'].get(difficulty, 0.0)):.2f}%`")
    lines.extend(["", "## Lowest-scoring tasks", ""])
    tasks = sorted(payload.get("tasks", []), key=lambda item: (float(item.get("score_percent", 0.0)), item["task_id"]))
    for item in tasks[:10]:
        lines.append(
            f"- `{item['task_id']}` / `{item['difficulty']}` / `{item.get('task_type', 'unknown')}`: `{float(item.get('score_percent', 0.0)):.2f}%`"
        )
        rubric = item.get("rubric_breakdown", {})
        if isinstance(rubric, dict) and rubric:
            weakest = sorted(
                rubric.items(),
                key=lambda kv: float(kv[1].get("percent", 0.0)),
            )[:2]
            for rubric_name, rubric_payload in weakest:
                lines.append(
                    f"  - rubric `{rubric_name}`: `{float(rubric_payload.get('earned', 0.0)):.2f}/{float(rubric_payload.get('weight', 0.0)):.2f}` (`{float(rubric_payload.get('percent', 0.0)):.2f}%`)"
                )
    lines.extend(["", "## Artifact paths", ""])
    suffix = _agent_behavior_mode_suffix(mode)
    lines.extend(
        [
            f"- `{suffix}` results JSON: `agent_behavior_{suffix}_results.json`",
            f"- `{suffix}` report markdown: `agent_behavior_{suffix}_report.md`",
            f"- benchmark task JSON: `agent_behavior_{suffix}/agent_behavior_{suffix}_results.json`",
            f"- benchmark task report: `agent_behavior_{suffix}/agent_behavior_{suffix}_report.md`",
        ]
    )
    if mode == "cached":
        lines.append("- replay cache root: `agent_behavior_cached/replay_cache/`")
    else:
        lines.append("- compatibility JSON alias: `agent_behavior_validation/benchmark_results.json`")
        lines.append("- compatibility report alias: `agent_behavior_validation/benchmark_report.md`")
    return "\n".join(lines) + "\n"


def run_agent_behavior_tests(
    *,
    output_dir: Path,
    mode: str,
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
    normalized_mode = str(mode).strip().lower()
    suffix = _agent_behavior_mode_suffix(normalized_mode)
    report = run_benchmarks(
        output_dir=output_dir / f"agent_behavior_{suffix}",
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
        agent_behavior_mode=normalized_mode,
    )
    difficulty_scores = {
        difficulty: float(report["summary"]["score_by_difficulty"].get(difficulty, 0.0))
        for difficulty in BENCHMARK_DIFFICULTY_ORDER
    }
    payload = {
        **report,
        "mode": normalized_mode,
        "mode_label": _agent_behavior_mode_label(normalized_mode),
        "difficulty_tier_scores": difficulty_scores,
        "percent": float(report["summary"].get("average_task_score_percent", 0.0)),
    }
    write_text(output_dir / f"agent_behavior_{suffix}_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text(output_dir / f"agent_behavior_{suffix}_report.md", render_agent_behavior_tests_report(payload), encoding="utf-8")
    if normalized_mode == "no-cache-validation":
        benchmark_dir = output_dir / "agent_behavior_validation"
        write_text(benchmark_dir / "benchmark_results.json", stable_json_dumps(report, indent=2) + "\n", encoding="utf-8")
        benchmark_report = benchmark_dir / "agent_behavior_validation_report.md"
        if benchmark_report.exists():
            write_text(benchmark_dir / "benchmark_report.md", benchmark_report.read_text(encoding="utf-8"), encoding="utf-8")
    return payload


def run_agent_behavior_validation(
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
    payload = run_agent_behavior_tests(
        output_dir=output_dir,
        mode="no-cache-validation",
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
    write_text(output_dir / "agent_behavior_validation_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def render_combined_test_evaluation_report(payload: dict[str, Any]) -> str:
    deterministic_results = payload["deterministic_correctness"]
    cached_support = payload.get("agent_behavior_cached_support")
    validation_results = payload.get("agent_behavior_validation")
    if cached_support is None or validation_results is None:
        raise KeyError("combined evaluation payload is missing agent behavior results")
    lines = [
        "# SWAAG evaluation report",
        "",
        f"- final_overall_percent: `{payload['overall_percent']:.2f}%`",
        f"- deterministic_correctness_percent: `{deterministic_results['summary']['percent']:.2f}%`",
        f"- agent_behavior_cached_support_percent: `{cached_support['summary']['percent']:.2f}%`",
        f"- agent_behavior_validation_percent: `{float(validation_results['percent']):.2f}%`",
        "",
        "## Test category summaries",
        "",
        f"- deterministic correctness tests: `{deterministic_results['summary']['passed_tests']}` passed / `{deterministic_results['summary']['failed_tests']}` failed / `{deterministic_results['summary']['skipped_tests']}` skipped",
        f"- agent behavior tests (cached mode): `{cached_support['summary'].get('passed_tests', 0)}` passed / `{cached_support['summary'].get('failed_tests', 0)}` failed across `{cached_support['summary'].get('total_families', len(cached_support.get('results', [])))}` focused support-check families",
        f"- agent behavior tests (no-cache validation mode): `{len(validation_results.get('tasks', []))}` tasks / `{float(validation_results.get('summary', {}).get('average_task_score_percent', validation_results['percent'])):.2f}%` average",
        "",
        "## Validation difficulty tiers",
        "",
    ]
    for difficulty in BENCHMARK_DIFFICULTY_ORDER:
        lines.append(f"- `{difficulty}`: `{float(validation_results['difficulty_tier_scores'].get(difficulty, 0.0)):.2f}%`")
    lines.extend(["", "## Agent behavior support checks (cached mode)", ""])
    for family in cached_support.get("results", []):
        lines.append(
            f"- `{family['family_id']}`: `{float(family['score_percent']):.2f}%` / `{family['status']}` / `{family['passed_tests']}` passed / `{family['failed_tests']}` failed"
        )
    lines.extend(["", "## Lowest-scoring agent behavior tasks (no-cache validation mode)", ""])
    tasks = sorted(validation_results.get("tasks", []), key=lambda item: (float(item.get("score_percent", 0.0)), item["task_id"]))
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
            "- agent_behavior_cached_support:",
            "  - `agent_behavior_cached_support/agent_behavior_cached_support_results.json`",
            "  - `agent_behavior_cached_support/agent_behavior_cached_support_report.md`",
            "- agent_behavior_validation:",
            "  - `agent_behavior_validation/agent_behavior_validation_results.json`",
            "  - `agent_behavior_validation/benchmark_results.json`",
            "  - `agent_behavior_validation/benchmark_report.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_combined_test_evaluation(
    *,
    output_dir: Path,
    clean: bool = False,
    functional_pytest_args: Sequence[str] | None = None,
    support_family_ids: Sequence[str] | None = None,
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
    from swaag.benchmark.agent_support import run_agent_behavior_support_checks

    if clean and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    deterministic = run_functional_correctness_lane(
        output_dir=output_dir / "deterministic_correctness",
        pytest_args=functional_pytest_args,
    )
    cached_support = run_agent_behavior_support_checks(
        output_dir=output_dir / "agent_behavior_cached_support",
        family_ids=support_family_ids,
        clean=True,
    )
    live = run_agent_behavior_validation(
        output_dir=output_dir / "agent_behavior_validation",
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
    category_scores = {
        "deterministic_correctness": float(deterministic["summary"]["percent"]),
        "agent_behavior_cached_support": float(cached_support["summary"]["percent"]),
        "agent_behavior_validation": float(live["percent"]),
    }
    overall_percent = round(sum(category_scores.values()) / len(category_scores), 2) if category_scores else 0.0
    payload = {
        "deterministic_correctness": deterministic,
        "agent_behavior_cached_support": cached_support,
        "agent_behavior_validation": live,
        "category_scores": category_scores,
        "overall_percent": overall_percent,
    }
    write_text(output_dir / "full_evaluation_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text(output_dir / "full_evaluation_report.md", render_combined_test_evaluation_report(payload), encoding="utf-8")
    return payload


def render_test_category_report(payload: dict[str, Any]) -> str:
    deterministic = payload["deterministic_correctness"]
    cached = payload.get("agent_behavior_cached")
    validation = payload.get("agent_behavior_validation")
    lines = [
        "# SWAAG test category report",
        "",
        f"- overall_percent: `{payload['overall_percent']:.2f}%`",
        f"- deterministic_correctness_percent: `{deterministic['summary']['percent']:.2f}%`",
        f"- agent_behavior_cached_percent: `{float(cached['percent']) if isinstance(cached, dict) else 0.0:.2f}%`",
        f"- agent_behavior_validation_percent: `{float(validation['percent']) if isinstance(validation, dict) else 0.0:.2f}%`",
        "",
        "## Category summaries",
        "",
        f"- deterministic correctness tests: `{deterministic['summary']['passed_tests']}` passed / `{deterministic['summary']['failed_tests']}` failed / `{deterministic['summary']['skipped_tests']}` skipped",
        (
            f"- agent behavior tests (cached mode): `{cached['summary'].get('successful_tasks', 0)}` succeeded / `{cached['summary'].get('failed_tasks', 0)}` failed / replay cache at `agent_behavior_cached/replay_cache/`"
            if isinstance(cached, dict)
            else "- agent behavior tests (cached mode): not run because deterministic correctness failed"
        ),
        (
            f"- agent behavior tests (no-cache validation mode): `{validation['summary'].get('successful_tasks', 0)}` succeeded / `{validation['summary'].get('failed_tasks', 0)}` failed"
            if isinstance(validation, dict)
            else "- agent behavior tests (no-cache validation mode): not run because deterministic correctness failed"
        ),
        "",
        "## Validation difficulty tiers",
        "",
    ]
    for difficulty in BENCHMARK_DIFFICULTY_ORDER:
        score = float(validation["difficulty_tier_scores"].get(difficulty, 0.0)) if isinstance(validation, dict) else 0.0
        lines.append(f"- `{difficulty}`: `{score:.2f}%`")
    lines.extend(["", "## Lowest-scoring validation tasks", ""])
    tasks = sorted(validation.get("tasks", []) if isinstance(validation, dict) else [], key=lambda item: (float(item.get("score_percent", 0.0)), item["task_id"]))
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
            "- deterministic correctness tests:",
            "  - `deterministic_correctness/functional_correctness_results.json`",
            "  - `deterministic_correctness/functional_correctness_report.md`",
            "- agent behavior tests (cached mode):",
            "  - `agent_behavior_cached_results.json`",
            "  - `agent_behavior_cached_report.md`",
            "  - `agent_behavior_cached/agent_behavior_cached_results.json`",
            "  - `agent_behavior_cached/agent_behavior_cached_report.md`",
            "  - `agent_behavior_cached/replay_cache/`",
            "- agent behavior tests (no-cache validation mode):",
            "  - `agent_behavior_validation_results.json`",
            "  - `agent_behavior_validation_report.md`",
            "  - `agent_behavior_validation/agent_behavior_validation_results.json`",
            "  - `agent_behavior_validation/agent_behavior_validation_report.md`",
            "  - `agent_behavior_validation/benchmark_results.json`",
            "  - `agent_behavior_validation/benchmark_report.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_test_category_evaluation(
    *,
    output_dir: Path,
    clean: bool = False,
    functional_pytest_args: Sequence[str] | None = None,
    benchmark_task_ids: list[str] | None = None,
    validation_subset: bool = True,
    model_base_url: str | None = None,
    timeout_seconds: int | None = None,
    connect_timeout_seconds: int | None = None,
    model_profile: str | None = None,
    structured_output_mode: str | None = None,
    progress_poll_seconds: float | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    if clean and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    deterministic = run_functional_correctness_lane(
        output_dir=output_dir / "deterministic_correctness",
        pytest_args=functional_pytest_args,
    )
    deterministic_failed = (
        float(deterministic["summary"].get("percent", 0.0)) < 100.0
        or int(deterministic["summary"].get("failed_tests", 0)) > 0
        or int(deterministic.get("exit_code", 0)) != 0
    )
    if deterministic_failed:
        payload = {
            "deterministic_correctness": deterministic,
            "agent_behavior_cached": None,
            "agent_behavior_validation": None,
            "category_scores": {
                "deterministic_correctness": float(deterministic["summary"].get("percent", 0.0)),
                "agent_behavior_cached": 0.0,
                "agent_behavior_validation": 0.0,
            },
            "overall_percent": round(float(deterministic["summary"].get("percent", 0.0)) / 3.0, 2),
            "status": "deterministic_correctness_failed",
            "skipped_categories": ["agent_behavior_cached", "agent_behavior_validation"],
            "skip_reason": "deterministic correctness must be 100% before agent behavior tests can run",
        }
        write_text(output_dir / "test_categories_results.json", stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
        write_text(output_dir / "test_categories_report.md", render_test_category_report(payload), encoding="utf-8")
        return payload
    cached = run_agent_behavior_tests(
        output_dir=output_dir,
        mode="cached",
        benchmark_task_ids=benchmark_task_ids,
        live_subset=validation_subset,
        model_base_url=model_base_url,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        model_profile=model_profile,
        structured_output_mode=structured_output_mode,
        progress_poll_seconds=progress_poll_seconds,
        seeds=seeds,
    )
    validation = run_agent_behavior_tests(
        output_dir=output_dir,
        mode="no-cache-validation",
        benchmark_task_ids=benchmark_task_ids,
        live_subset=validation_subset,
        model_base_url=model_base_url,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        model_profile=model_profile,
        structured_output_mode=structured_output_mode,
        progress_poll_seconds=progress_poll_seconds,
        seeds=seeds,
    )
    category_scores = {
        "deterministic_correctness": float(deterministic["summary"]["percent"]),
        "agent_behavior_cached": float(cached["percent"]),
        "agent_behavior_validation": float(validation["percent"]),
    }
    overall_percent = round(sum(category_scores.values()) / len(category_scores), 2) if category_scores else 0.0
    payload = {
        "deterministic_correctness": deterministic,
        "agent_behavior_cached": cached,
        "agent_behavior_validation": validation,
        "category_scores": category_scores,
        "overall_percent": overall_percent,
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
