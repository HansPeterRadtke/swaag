from __future__ import annotations

import json
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


@pytest.mark.benchmark_heavy
def test_benchmark_runner_executes_all_real_tasks_and_writes_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    report = run_benchmarks(output_dir=output_dir, clean=True)

    assert report["summary"]["total_tasks"] == len(get_benchmark_tasks())
    assert report["summary"]["failed_tasks"] == 0
    assert report["summary"]["false_positives"] == 0
    assert report["aggregate_metrics"]["primary"]["false_positive_rate"] == 0.0
    assert report["aggregate_metrics"]["primary"]["task_success_rate"] == 1.0
    assert report["aggregate_metrics"]["benchmark_specific"]["environment_usage_correctness"] == 1.0
    assert (output_dir / "benchmark_results.json").exists()
    assert (output_dir / "benchmark_report.md").exists()

    persisted = json.loads((output_dir / "benchmark_results.json").read_text(encoding="utf-8"))
    assert persisted["summary"]["failed_tasks"] == 0
    assert persisted["summary"]["total_tasks"] >= 170
    assert persisted["aggregate_metrics"]["coverage_by_type"]["coding"] >= 40
    assert persisted["aggregate_metrics"]["coverage_by_type"]["file_edit"] >= 25
    assert persisted["aggregate_metrics"]["coverage_by_type"]["reading"] >= 25
    assert persisted["aggregate_metrics"]["coverage_by_type"]["multi_step"] >= 30
    assert persisted["aggregate_metrics"]["coverage_by_type"]["failure"] >= 30
    assert persisted["aggregate_metrics"]["coverage_by_type"]["quality"] >= 20
    assert all(item["history_path"] for item in persisted["tasks"])
    assert all(Path(item["history_path"]).exists() for item in persisted["tasks"])
    report_text = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "False Positive Analysis" in report_text
    assert "Prompt Understanding Metrics" in report_text
    assert "Benchmark-Specific Metrics" in report_text
    assert "Run Metadata" in report_text


def test_benchmark_runner_classifies_expected_failure_tasks(tmp_path: Path) -> None:
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


def test_benchmark_runner_keeps_project_level_multifile_task_consistent(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    report = run_benchmarks(output_dir=output_dir, clean=True, task_ids=["coding_multifile_fix"])
    task = report["tasks"][0]

    assert task["success"] is True
    workspace = Path(task["workspace"])
    assert (workspace / "core.py").read_text(encoding="utf-8") == "def base_value() -> int:\n    return 41\n"
    assert "return f'total={total()}'" in (workspace / "service.py").read_text(encoding="utf-8")


def test_benchmark_runner_executes_environment_tasks_with_persistent_state(tmp_path: Path) -> None:
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


def test_benchmark_runner_writes_live_subset_artifacts_in_scripted_mode(tmp_path: Path) -> None:
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


def test_benchmark_runner_records_per_seed_live_results(monkeypatch, tmp_path: Path) -> None:
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
    assert task_payload["metrics"]["verification_trace"]["verification_type_used"] == "llm_fallback"


def test_benchmark_runner_keeps_one_fixed_profile_across_seeded_live_runs(monkeypatch, tmp_path: Path) -> None:
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


def test_scripted_benchmark_runs_do_not_use_profile_optimized_tool_input_calls(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    report = run_benchmarks(
        output_dir=output_dir,
        clean=True,
        task_ids=["file_edit_generated_exact_01"],
    )

    assert report["summary"]["failed_tasks"] == 0
    history_text = Path(report["tasks"][0]["history_path"]).read_text(encoding="utf-8")
    assert '"kind":"tool_input"' not in history_text
