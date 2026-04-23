from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import swaag.benchmark.benchmark_runner as benchmark_runner
from swaag.benchmark.benchmark_runner import _resolve_live_model_settings, run_benchmarks
from swaag.live_runtime_profiles import get_documented_final_live_benchmark_recommendation
from swaag.benchmark.task_definitions import (
    BenchmarkTaskDefinition,
    BenchmarkVerificationContract,
    ScriptedBenchmarkClient,
    TaskScenario,
    _plan_response,
    _plan_step,
    get_benchmark_tasks,
)


def _full_catalog_cache_key(tasks: list[BenchmarkTaskDefinition]) -> str:
    payload = [
        {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "difficulty": task.difficulty,
            "tags": list(task.tags),
        }
        for task in tasks
    ]
    raw = json.dumps({"version": 2, "tasks": payload}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _valid_full_catalog_report(report_path: Path, tasks: list[BenchmarkTaskDefinition]) -> bool:
    if not report_path.exists():
        return False
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_ids = {task.task_id for task in tasks}
    actual_ids = {str(item.get("task_id", "")) for item in payload.get("tasks", [])}
    return (
        payload.get("summary", {}).get("total_tasks") == len(tasks)
        and payload.get("summary", {}).get("failed_tasks") == 0
        and payload.get("summary", {}).get("false_positives") == 0
        and actual_ids == expected_ids
    )


def _copy_cached_full_catalog(cache_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(cache_dir, output_dir)
    old_root = str(cache_dir)
    new_root = str(output_dir)
    results_path = output_dir / "benchmark_results.json"
    report_path = output_dir / "benchmark_report.md"
    results_text = results_path.read_text(encoding="utf-8").replace(old_root, new_root)
    results_path.write_text(results_text, encoding="utf-8")
    if report_path.exists():
        report_path.write_text(report_path.read_text(encoding="utf-8").replace(old_root, new_root), encoding="utf-8")
    return json.loads(results_text)


def _run_full_catalog_with_artifact_reuse(output_dir: Path, tasks: list[BenchmarkTaskDefinition]) -> dict:
    """Exercise the full cached catalog, reusing full-catalog artifacts when valid."""
    cache_dir = Path("/tmp/swaag-full-cached-benchmark-catalog") / _full_catalog_cache_key(tasks)
    if _valid_full_catalog_report(cache_dir / "benchmark_results.json", tasks):
        return _copy_cached_full_catalog(cache_dir, output_dir)
    report = run_benchmarks(output_dir=cache_dir, clean=True)
    if not _valid_full_catalog_report(cache_dir / "benchmark_results.json", tasks):
        raise AssertionError("full cached benchmark catalog did not produce a valid full-catalog report")
    return _copy_cached_full_catalog(cache_dir, output_dir)


def test_benchmark_runner_executes_full_cached_catalog_and_writes_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    all_tasks = get_benchmark_tasks()

    report = _run_full_catalog_with_artifact_reuse(output_dir, all_tasks)

    assert report["summary"]["total_tasks"] == len(all_tasks)
    assert report["summary"]["failed_tasks"] == 0
    assert report["summary"]["false_positives"] == 0
    assert report["aggregate_metrics"]["primary"]["false_positive_rate"] == 0.0
    assert report["aggregate_metrics"]["primary"]["task_success_rate"] == 1.0
    assert report["aggregate_metrics"]["benchmark_specific"]["environment_usage_correctness"] == 1.0
    assert (output_dir / "benchmark_results.json").exists()
    assert (output_dir / "benchmark_report.md").exists()

    persisted = json.loads((output_dir / "benchmark_results.json").read_text(encoding="utf-8"))
    assert persisted["summary"]["failed_tasks"] == 0
    assert persisted["summary"]["total_tasks"] == len(all_tasks)
    expected_coverage = {}
    expected_difficulties = {}
    for task in all_tasks:
        expected_coverage[task.task_type] = expected_coverage.get(task.task_type, 0) + 1
        expected_difficulties[task.difficulty] = expected_difficulties.get(task.difficulty, 0) + 1
    assert persisted["aggregate_metrics"]["coverage_by_type"] == dict(sorted(expected_coverage.items()))
    assert persisted["aggregate_metrics"]["coverage_by_difficulty"] == dict(sorted(expected_difficulties.items()))
    assert set(persisted["aggregate_metrics"]["coverage_by_difficulty"]) == {
        "easy",
        "extremely_easy",
        "extremely_hard",
        "hard",
        "normal",
    }
    assert persisted["aggregate_metrics"]["coverage_by_difficulty"]["extremely_hard"] > 0
    assert all(item["history_path"] for item in persisted["tasks"])
    assert all(Path(item["history_path"]).exists() for item in persisted["tasks"])
    report_text = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "False Positive Analysis" in report_text
    assert "Prompt Understanding Metrics" in report_text
    assert "Benchmark-Specific Metrics" in report_text
    assert "Run Metadata" in report_text


def test_benchmark_runner_helper_classifies_selected_expected_failure_tasks(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    report = run_benchmarks(
        output_dir=output_dir,
        clean=True,
        task_ids=[
            "failure_wrong_tool_usage",
            "failure_bad_planning",
            "failure_generated_bad_plan_11",
        ],
    )

    categories = {item["task_id"]: item["failure_category"] for item in report["tasks"]}
    assert categories["failure_wrong_tool_usage"] == "wrong_tool_usage"
    assert categories["failure_bad_planning"] == "bad_planning"
    assert categories["failure_generated_bad_plan_11"] == "bad_planning"
    assert all(item["success"] is True for item in report["tasks"])


def test_benchmark_runner_helper_keeps_selected_project_level_multifile_task_consistent(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    report = run_benchmarks(output_dir=output_dir, clean=True, task_ids=["coding_multifile_fix"])
    task = report["tasks"][0]

    assert task["success"] is True
    workspace = Path(task["workspace"])
    assert (workspace / "core.py").read_text(encoding="utf-8") == "def base_value() -> int:\n    return 41\n"
    assert "return f'total={total()}'" in (workspace / "service.py").read_text(encoding="utf-8")


def test_benchmark_runner_helper_executes_selected_environment_tasks_with_persistent_state(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    report = run_benchmarks(
        output_dir=output_dir,
        clean=True,
        task_ids=[
            "multi_step_environment_shell_persistence",
            "multi_step_environment_list_read_write",
            "coding_run_tests_environment",
            "multi_step_iterative_write_refinement",
        ],
    )

    assert report["summary"]["failed_tasks"] == 0
    assert report["aggregate_metrics"]["benchmark_specific"]["tasks_with_environment_usage"] >= 3
    assert report["aggregate_metrics"]["benchmark_specific"]["iteration_improvement_rate"] == 1.0


def test_benchmark_runner_helper_writes_selected_manual_validation_artifacts_in_scripted_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "live_subset"
    report = run_benchmarks(
        output_dir=output_dir,
        clean=True,
        live_subset=True,
        use_live_model=False,
        task_ids=["live_coding_fix_01", "live_file_edit_01", "live_reading_01", "live_multi_step_01", "live_failure_01", "live_quality_01"],
    )

    assert report["summary"]["failed_tasks"] == 0
    assert report["summary"]["false_positives"] == 0
    assert report["run_metadata"]["mode"] == "live_subset"
    assert report["run_metadata"]["use_live_model"] is False
    assert report["run_metadata"]["request_observability_mode"] == "scripted_immediate"
    assert "profile_use_case" in report["run_metadata"]
    assert (output_dir / "benchmark_results.json").exists()
    assert (output_dir / "benchmark_report.md").exists()


def test_benchmark_runner_uses_live_environment_overrides_for_runtime_profile(monkeypatch) -> None:
    monkeypatch.setenv("SWAAG_LIVE_BASE_URL", "http://127.0.0.1:19999")
    monkeypatch.setenv("SWAAG_LIVE_TIMEOUT_SECONDS", "321")
    monkeypatch.setenv("SWAAG_LIVE_CONNECT_TIMEOUT_SECONDS", "11")
    monkeypatch.setenv("SWAAG_LIVE_MODEL_PROFILE", "mid_context")
    monkeypatch.setenv("SWAAG_LIVE_STRUCTURED_OUTPUT_MODE", "post_validate")
    monkeypatch.setenv("SWAAG_LIVE_PROGRESS_POLL_SECONDS", "2.25")
    monkeypatch.setenv("SWAAG_LIVE_SEEDS", "7,13,29")

    settings = _resolve_live_model_settings(
        use_live_model=True,
        model_base_url=None,
        timeout_seconds=None,
        connect_timeout_seconds=None,
        model_profile=None,
        structured_output_mode=None,
        progress_poll_seconds=None,
    )

    assert settings["base_url"] == "http://127.0.0.1:19999"
    assert settings["timeout_seconds"] == 321
    assert settings["connect_timeout_seconds"] == 11
    assert settings["model_profile"] == "mid_context"
    assert settings["structured_output_mode"] == "post_validate"
    assert settings["progress_poll_seconds"] == 2.25
    assert settings["seeds"] == [7, 13, 29]


def test_benchmark_runner_live_defaults_match_the_documented_final_recommendation(monkeypatch) -> None:
    monkeypatch.delenv("SWAAG_LIVE_BASE_URL", raising=False)
    monkeypatch.delenv("SWAAG_LIVE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SWAAG_LIVE_CONNECT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SWAAG_LIVE_MODEL_PROFILE", raising=False)
    monkeypatch.delenv("SWAAG_LIVE_STRUCTURED_OUTPUT_MODE", raising=False)
    monkeypatch.delenv("SWAAG_LIVE_PROGRESS_POLL_SECONDS", raising=False)
    recommendation = get_documented_final_live_benchmark_recommendation()

    settings = _resolve_live_model_settings(
        use_live_model=True,
        model_base_url=None,
        timeout_seconds=None,
        connect_timeout_seconds=None,
        model_profile=None,
        structured_output_mode=None,
        progress_poll_seconds=None,
    )

    assert settings["model_profile"] == recommendation.model_profile
    assert settings["structured_output_mode"] == recommendation.structured_output_mode
    assert settings["seeds"] == list(recommendation.seeds)
    assert settings["timeout_seconds"] == recommendation.timeout_seconds
    assert settings["connect_timeout_seconds"] == recommendation.connect_timeout_seconds
    assert settings["progress_poll_seconds"] == recommendation.progress_poll_seconds


def test_benchmark_runner_helper_records_per_seed_manual_validation_results(monkeypatch, tmp_path: Path) -> None:
    task = BenchmarkTaskDefinition(
        task_id="live_seed_probe",
        task_type="quality",
        description="Probe live seed propagation with a scripted live task.",
        build=lambda workspace: TaskScenario(
            prompt="Reply exactly ok.",
            workspace=workspace,
            model_client=ScriptedBenchmarkClient(
                responses=[
                    _plan_response(
                        goal="Reply exactly ok.",
                        steps=[
                            _plan_step(
                                "step_answer",
                                "Answer",
                                "respond",
                                expected_output="ok",
                                success_criteria="reply exact answer",
                            )
                        ],
                    ),
                    "ok",
                ]
            ),
            verification_contract=BenchmarkVerificationContract(
                task_type="quality",
                expected_answer="ok",
                required_history_events=["verification_passed"],
            ),
        ),
        build_live=lambda workspace: TaskScenario(
            prompt="Reply exactly ok.",
            workspace=workspace,
            model_client=ScriptedBenchmarkClient(
                responses=[
                    _plan_response(
                        goal="Reply exactly ok.",
                        steps=[
                            _plan_step(
                                "step_answer",
                                "Answer",
                                "respond",
                                expected_output="ok",
                                success_criteria="reply exact answer",
                            )
                        ],
                    ),
                    "ok",
                ]
            ),
            verification_contract=BenchmarkVerificationContract(
                task_type="quality",
                expected_answer="ok",
                required_history_events=["verification_passed"],
            ),
        ),
        difficulty="easy",
        tags=["live-subset", "quality"],
    )
    monkeypatch.setattr(benchmark_runner, "_load_tasks", lambda task_ids, live_subset: [task])

    report = run_benchmarks(
        output_dir=tmp_path / "seeded_live",
        clean=True,
        live_subset=True,
        use_live_model=True,
        model_base_url="http://127.0.0.1:19999",
        model_profile="small_fast",
        structured_output_mode="post_validate",
        timeout_seconds=12,
        connect_timeout_seconds=3,
        progress_poll_seconds=0.5,
        seeds=[11, 23, 37],
    )

    assert report["run_metadata"]["seeds"] == [11, 23, 37]
    task_payload = report["tasks"][0]
    assert task_payload["success"] is True
    assert task_payload["metrics"]["seed_success_count"] == 3
    assert [item["seed"] for item in task_payload["metrics"]["seed_results"]] == [11, 23, 37]
    assert "guidance_sources" in task_payload["metrics"]
    assert "selected_skill_ids" in task_payload["metrics"]
    assert "exposed_tool_names" in task_payload["metrics"]
    assert "retrieval_trace_sample" in task_payload["metrics"]
    assert task_payload["metrics"]["verification_trace"]["verification_type_used"] == "composite"


def test_benchmark_runner_helper_keeps_one_fixed_profile_across_seeded_manual_validation_runs(monkeypatch, tmp_path: Path) -> None:
    task = BenchmarkTaskDefinition(
        task_id="live_profile_probe",
        task_type="quality",
        description="Probe profile reuse across seeded live runs.",
        build=lambda workspace: TaskScenario(
            prompt="Reply exactly ok.",
            workspace=workspace,
            model_client=ScriptedBenchmarkClient(
                responses=[
                    _plan_response(
                        goal="Reply exactly ok.",
                        steps=[_plan_step("step_answer", "Answer", "respond", expected_output="ok", success_criteria="reply exact answer")],
                    ),
                    "ok",
                ]
            ),
            verification_contract=BenchmarkVerificationContract(task_type="quality", expected_answer="ok"),
        ),
        build_live=lambda workspace: TaskScenario(
            prompt="Reply exactly ok.",
            workspace=workspace,
            model_client=ScriptedBenchmarkClient(
                responses=[
                    _plan_response(
                        goal="Reply exactly ok.",
                        steps=[_plan_step("step_answer", "Answer", "respond", expected_output="ok", success_criteria="reply exact answer")],
                    ),
                    "ok",
                ]
            ),
            verification_contract=BenchmarkVerificationContract(task_type="quality", expected_answer="ok"),
        ),
        difficulty="easy",
        tags=["live-subset", "quality"],
    )
    monkeypatch.setattr(benchmark_runner, "_load_tasks", lambda task_ids, live_subset: [task])
    seen_profiles: list[tuple[str, str]] = []
    original_build_config = benchmark_runner._build_config

    def _recording_build_config(*args, **kwargs):
        seen_profiles.append((str(kwargs.get("profile_name")), str(kwargs.get("structured_output_mode"))))
        return original_build_config(*args, **kwargs)

    monkeypatch.setattr(benchmark_runner, "_build_config", _recording_build_config)

    report = run_benchmarks(
        output_dir=tmp_path / "fixed_profile_live",
        clean=True,
        live_subset=True,
        use_live_model=True,
        model_base_url="http://127.0.0.1:19999",
        model_profile="small_fast",
        structured_output_mode="post_validate",
        timeout_seconds=12,
        connect_timeout_seconds=3,
        progress_poll_seconds=0.5,
        seeds=[11, 23, 37],
    )

    assert report["summary"]["failed_tasks"] == 0
    assert seen_profiles == [("small_fast", "post_validate")] * 3


def test_scripted_benchmark_client_adapts_tool_decision_payloads_for_tool_input_calls() -> None:
    client = ScriptedBenchmarkClient(
        responses=[
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "edit_text",
                    "tool_input": {
                        "path": "/tmp/demo.txt",
                        "operation": "replace_pattern_once",
                        "pattern": "old",
                        "replacement": "new",
                    },
                }
            )
        ]
    )

    completion = client.send_completion({"contract": "tool_input:edit_text", "prompt": "tool input"}, timeout_seconds=5)
    payload = json.loads(completion.text)

    assert payload == {
        "path": "/tmp/demo.txt",
        "operation": "replace_pattern_once",
        "pattern": "old",
        "replacement": "new",
    }


def test_scripted_benchmark_client_skips_tool_decision_payload_for_plain_text_answer() -> None:
    client = ScriptedBenchmarkClient(
        responses=[
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "calculator",
                    "tool_input": {"expression": "2 + 2"},
                }
            ),
            "4",
        ]
    )

    completion = client.send_completion({"contract": "plain_text", "prompt": "final answer"}, timeout_seconds=5)

    assert completion.text == "4"


def test_scripted_benchmark_client_auto_verification_uses_criterion_names() -> None:
    client = ScriptedBenchmarkClient()

    completion = client.send_completion(
        {
            "contract": "verification",
            "prompt": (
                "Candidate result:\nfixed-01\n\n"
                "Criteria:\n"
                '[{"name":"satisfies_done_condition","criterion":"assistant_response_nonempty"},'
                '{"name":"meets_success_criteria","criterion":"The assistant replies fixed-01."}]\n\n'
            ),
        },
        timeout_seconds=5,
    )

    payload = json.loads(completion.text)

    assert payload == {
        "criteria": [
            {
                "name": "satisfies_done_condition",
                "passed": True,
                "evidence": "candidate result is non-empty",
            },
            {
                "name": "meets_success_criteria",
                "passed": True,
                "evidence": "candidate result is non-empty",
            },
        ]
    }


def test_scripted_benchmark_helper_run_does_not_use_profile_optimized_tool_input_calls(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    report = run_benchmarks(
        output_dir=output_dir,
        clean=True,
        task_ids=["file_edit_generated_exact_01"],
    )

    assert report["summary"]["failed_tasks"] == 0
    history_text = Path(report["tasks"][0]["history_path"]).read_text(encoding="utf-8")
    assert '"kind":"tool_input"' not in history_text
