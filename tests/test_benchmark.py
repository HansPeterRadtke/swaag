from __future__ import annotations

import json
import os
from pathlib import Path

from swaag.benchmark.benchmark_runner import _build_config, _resolve_live_model_settings, run_benchmarks
from swaag.benchmark.task_definitions import get_benchmark_tasks
from swaag.live_runtime_profiles import get_documented_final_live_benchmark_recommendation



def _run_full_catalog_without_artifact_reuse(output_dir: Path) -> dict:
    return run_benchmarks(
        output_dir=output_dir,
        clean=True,
        agent_behavior_mode="cached",
        model_base_url=os.environ.get("SWAAG_LIVE_BASE_URL", "http://127.0.0.1:14829"),
        model_profile="small_fast",
        structured_output_mode="post_validate",
        connect_timeout_seconds=5,
        timeout_seconds=15,
        progress_poll_seconds=1.0,
    )


def test_benchmark_runner_executes_full_llm_response_cache_catalog_and_writes_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    all_tasks = get_benchmark_tasks()

    report = _run_full_catalog_without_artifact_reuse(output_dir)

    assert report["summary"]["total_tasks"] == len(all_tasks)
    assert 0.0 <= report["summary"]["average_task_score_percent"] <= 100.0
    assert report["summary"]["successful_tasks"] + report["summary"]["failed_tasks"] == len(all_tasks)
    assert report["run_metadata"]["agent_behavior_mode"] == "cached"
    assert report["run_metadata"]["replay_cache_enabled"] is True
    assert 0.0 <= report["aggregate_metrics"]["primary"]["false_positive_rate"] <= 1.0
    assert 0.0 <= report["aggregate_metrics"]["primary"]["task_success_rate"] <= 1.0
    assert (output_dir / "agent_test_run_results.json").exists()
    assert (output_dir / "agent_test_run_report.md").exists()

    persisted = json.loads((output_dir / "agent_test_run_results.json").read_text(encoding="utf-8"))
    assert persisted["summary"]["total_tasks"] == len(all_tasks)
    expected_coverage: dict[str, int] = {}
    expected_difficulties: dict[str, int] = {}
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
    assert all(
        seed.get("replay_cache", {}).get("cassette_path")
        for item in persisted["tasks"]
        for seed in item.get("metrics", {}).get("seed_results", [])
    )
    report_text = (output_dir / "agent_test_run_report.md").read_text(encoding="utf-8")
    assert "False Positive Analysis" in report_text
    assert "Prompt Understanding Metrics" in report_text
    assert "Benchmark-Specific Metrics" in report_text
    assert "Run Metadata" in report_text


def test_benchmark_runner_timeout_override_caps_all_benchmark_model_timeouts(tmp_path: Path) -> None:
    config = _build_config(
        sessions_root=tmp_path / "sessions",
        workspace=tmp_path / "workspace",
        overrides={},
        base_url="http://127.0.0.1:14829",
        connect_timeout_seconds=5,
        timeout_seconds=15,
        profile_name="small_fast",
        structured_output_mode="post_validate",
        progress_poll_seconds=1.0,
        seed=42,
    )

    assert config.model.timeout_seconds == 15
    assert config.model.simple_timeout_seconds == 15
    assert config.model.structured_timeout_seconds == 15
    assert config.model.verification_timeout_seconds == 15
    assert config.model.benchmark_timeout_seconds == 15


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
        seeds=None,
    )

    assert settings["base_url"] == "http://127.0.0.1:19999"
    assert settings["timeout_seconds"] == 321
    assert settings["connect_timeout_seconds"] == 11
    assert settings["model_profile"] == "mid_context"
    assert settings["structured_output_mode"] == "post_validate"
    assert settings["progress_poll_seconds"] == 2.25
    assert settings["seeds"] == [7, 13, 29]


def test_benchmark_runner_live_defaults_match_the_documented_final_recommendation(monkeypatch) -> None:
    for key in [
        "SWAAG_LIVE_BASE_URL",
        "SWAAG_LIVE_TIMEOUT_SECONDS",
        "SWAAG_LIVE_CONNECT_TIMEOUT_SECONDS",
        "SWAAG_LIVE_MODEL_PROFILE",
        "SWAAG_LIVE_STRUCTURED_OUTPUT_MODE",
        "SWAAG_LIVE_PROGRESS_POLL_SECONDS",
        "SWAAG_LIVE_SEEDS",
    ]:
        monkeypatch.delenv(key, raising=False)
    recommendation = get_documented_final_live_benchmark_recommendation()

    settings = _resolve_live_model_settings(
        use_live_model=True,
        model_base_url=None,
        timeout_seconds=None,
        connect_timeout_seconds=None,
        model_profile=None,
        structured_output_mode=None,
        progress_poll_seconds=None,
        seeds=None,
    )

    assert settings["timeout_seconds"] == recommendation.timeout_seconds
    assert settings["connect_timeout_seconds"] == recommendation.connect_timeout_seconds
    assert settings["model_profile"] == recommendation.model_profile
    assert settings["structured_output_mode"] == recommendation.structured_output_mode
    assert settings["seeds"] == list(recommendation.seeds)
    assert settings["progress_poll_seconds"] == recommendation.progress_poll_seconds
