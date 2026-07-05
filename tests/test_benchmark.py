from __future__ import annotations

import hashlib
import json
import os
import shutil

import pytest
from pathlib import Path

from swaag.benchmark.benchmark_runner import _build_agent_behavior_model_client, _resolve_live_model_settings, run_benchmarks
from swaag.benchmark.task_definitions import BenchmarkTaskDefinition, get_benchmark_tasks, make_benchmark_task
from swaag.config import load_config
from swaag.live_runtime_profiles import get_documented_final_live_benchmark_recommendation
from swaag.testing.llm_record_replay import MissingReplayEntryError


def _full_catalog_cache_key(tasks: list[BenchmarkTaskDefinition]) -> str:
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
    raw = json.dumps({"version": 8, "tasks": payload}, sort_keys=True).encode("utf-8")
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


def _copy_cached_full_catalog(cache_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(cache_dir, output_dir)
    old_root = str(cache_dir)
    new_root = str(output_dir)
    results_path = output_dir / "agent_test_cached_results.json"
    report_path = output_dir / "agent_test_cached_report.md"
    results_text = results_path.read_text(encoding="utf-8").replace(old_root, new_root)
    results_path.write_text(results_text, encoding="utf-8")
    if report_path.exists():
        report_path.write_text(report_path.read_text(encoding="utf-8").replace(old_root, new_root), encoding="utf-8")
    return json.loads(results_text)


def _seed_partial_replay_cache(cache_dir: Path) -> None:
    target = cache_dir / "replay_cache"
    if target.exists():
        return
    artifact_root = Path(os.environ.get("SWAAG_FULL_CACHED_BENCHMARK_ARTIFACT_ROOT", "/tmp/swaag-full-cached-benchmark-catalog"))
    candidates = sorted(
        (
            path / "replay_cache"
            for path in artifact_root.iterdir()
            if path.is_dir() and path != cache_dir and (path / "replay_cache").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if artifact_root.exists() else []
    for source in candidates:
        shutil.copytree(source, target)
        return


def _run_full_catalog_with_artifact_reuse(output_dir: Path, tasks: list[BenchmarkTaskDefinition]) -> dict:
    """Run the full cached catalog, reusing only valid full real-response cache artifacts."""
    cache_dir = Path(os.environ.get("SWAAG_FULL_CACHED_BENCHMARK_ARTIFACT_ROOT", "/tmp/swaag-full-cached-benchmark-catalog")) / _full_catalog_cache_key(tasks)
    if _valid_full_catalog_report(cache_dir / "agent_test_cached_results.json", tasks):
        return _copy_cached_full_catalog(cache_dir, output_dir)
    if os.environ.get("SWAAG_BENCHMARK_CACHED_REPLAY_POLICY", "replay_only").strip().lower() not in {
        "record",
        "record_if_missing",
        "replay_if_present_record_if_missing",
    }:
        pytest.skip("no valid full cached benchmark artifact is available; refusing to live-record in cached mode")
    _seed_partial_replay_cache(cache_dir)
    report = run_benchmarks(
        output_dir=cache_dir,
        clean=not cache_dir.exists(),
        agent_behavior_mode="cached",
        model_base_url=os.environ.get("SWAAG_LIVE_BASE_URL", "http://127.0.0.1:14829"),
        model_profile="small_fast",
        structured_output_mode="post_validate",
        connect_timeout_seconds=int(os.environ.get("SWAAG_BENCHMARK_CACHED_CONNECT_TIMEOUT_SECONDS", "5")),
        timeout_seconds=int(os.environ.get("SWAAG_BENCHMARK_CACHED_TIMEOUT_SECONDS", "15")),
        progress_poll_seconds=float(os.environ.get("SWAAG_BENCHMARK_CACHED_PROGRESS_POLL_SECONDS", "1.0")),
    )
    if not _valid_full_catalog_report(cache_dir / "agent_test_cached_results.json", tasks):
        raise AssertionError("full cached benchmark catalog did not produce a valid real-response full-catalog report")
    return _copy_cached_full_catalog(cache_dir, output_dir)


def test_full_catalog_helper_seeds_partial_replay_cache_from_previous_artifacts(monkeypatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    old_cache = artifact_root / "older-cache" / "replay_cache" / "demo_task"
    old_cache.mkdir(parents=True)
    (old_cache / "seed_42.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SWAAG_FULL_CACHED_BENCHMARK_ARTIFACT_ROOT", str(artifact_root))

    cache_dir = artifact_root / "new-cache"
    _seed_partial_replay_cache(cache_dir)

    assert (cache_dir / "replay_cache" / "demo_task" / "seed_42.json").exists()


def test_benchmark_runner_executes_full_cached_catalog_and_writes_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    all_tasks = get_benchmark_tasks()

    report = _run_full_catalog_with_artifact_reuse(output_dir, all_tasks)

    assert report["summary"]["total_tasks"] == len(all_tasks)
    assert 0.0 <= report["summary"]["average_task_score_percent"] <= 100.0
    assert report["summary"]["successful_tasks"] + report["summary"]["failed_tasks"] == len(all_tasks)
    assert report["run_metadata"]["agent_behavior_mode"] == "cached"
    assert report["run_metadata"]["replay_cache_enabled"] is True
    assert 0.0 <= report["aggregate_metrics"]["primary"]["false_positive_rate"] <= 1.0
    assert 0.0 <= report["aggregate_metrics"]["primary"]["task_success_rate"] <= 1.0
    assert (output_dir / "agent_test_cached_results.json").exists()
    assert (output_dir / "agent_test_cached_report.md").exists()

    persisted = json.loads((output_dir / "agent_test_cached_results.json").read_text(encoding="utf-8"))
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
    report_text = (output_dir / "agent_test_cached_report.md").read_text(encoding="utf-8")
    assert "False Positive Analysis" in report_text
    assert "Prompt Understanding Metrics" in report_text
    assert "Benchmark-Specific Metrics" in report_text
    assert "Run Metadata" in report_text


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


def test_generated_live_task_defaults_allow_minimal_four_step_agent_flow() -> None:
    task = make_benchmark_task(
        task_id="unit_live_file_edit",
        task_type="file_edit",
        difficulty="easy",
        tags=("file_edit",),
        description="Edit one file after reading it.",
    )

    assert task.config_overrides["planner_max_plan_steps"] == 4
    assert task.config_overrides["runtime_max_total_actions"] >= 4
    assert task.config_overrides["runtime_max_tool_steps"] >= 4
    assert task.config_overrides["runtime_tool_call_budget"] >= 4


def test_cached_agent_behavior_is_replay_only_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SWAAG_BENCHMARK_CACHED_REPLAY_POLICY", raising=False)
    task = make_benchmark_task(
        task_id="unit_missing_replay",
        task_type="reading",
        difficulty="easy",
        tags=("reading",),
        description="Replay-only cache behavior check.",
    )
    config = load_config()

    client, info = _build_agent_behavior_model_client(
        config=config,
        scenario=None,
        task=task,
        output_dir=tmp_path,
        seed=42,
        agent_behavior_mode="cached",
        live_subset=False,
    )

    assert info["replay_policy"] == "replay_only"
    assert info["cache_mode"] == "replay"
    assert client.mode == "replay"
    with pytest.raises(MissingReplayEntryError):
        client.send_completion({"prompt": "missing", "n_predict": 1})


def test_cached_agent_behavior_can_record_when_explicitly_requested(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWAAG_BENCHMARK_CACHED_REPLAY_POLICY", "record_if_missing")
    task = make_benchmark_task(
        task_id="unit_record_replay",
        task_type="reading",
        difficulty="easy",
        tags=("reading",),
        description="Record-if-missing cache behavior check.",
    )
    config = load_config()

    client, info = _build_agent_behavior_model_client(
        config=config,
        scenario=None,
        task=task,
        output_dir=tmp_path,
        seed=42,
        agent_behavior_mode="cached",
        live_subset=False,
    )

    assert info["replay_policy"] == "record_if_missing"
    assert info["cache_mode"] == "record"
    assert client.mode == "record"


def test_default_prompt_understanding_oracle_uses_prompt_schema_vocabulary() -> None:
    from swaag.benchmark.task_definitions import _default_oracle

    oracle = _default_oracle(
        task_id="reading_extract_structured",
        task_type="reading",
        difficulty="extremely_easy",
        tags=["reading", "json"],
    )

    assert oracle is not None
    assert oracle.task_type == "structured"
    assert oracle.strategy_profile == "reading"

    failure = _default_oracle(
        task_id="failure_generated_quiet_policy_bypass",
        task_type="failure",
        difficulty="hard",
        tags=["failure", "policy-bypass"],
    )

    assert failure is not None and failure.strategy_profile == "generic"
    assert oracle.detected_goals_contains == []
    assert oracle.detected_entities_contains == []


def test_quality_oracle_maps_vague_and_decomposed_to_analysis_types() -> None:
    from swaag.benchmark.task_definitions import _default_oracle

    vague = _default_oracle(
        task_id="quality_vague_expansion",
        task_type="quality",
        difficulty="extremely_easy",
        tags=["quality", "vague"],
    )
    decomposed = _default_oracle(
        task_id="quality_already_decomposed_prompt",
        task_type="quality",
        difficulty="normal",
        tags=["quality", "decomposition"],
    )

    assert vague is not None and vague.task_type == "vague" and vague.strategy_profile == "generic"
    assert decomposed is not None and decomposed.task_type == "already_decomposed" and decomposed.strategy_profile == "generic"


def test_benchmark_config_uses_discovered_server_context_limit(monkeypatch, tmp_path: Path) -> None:
    from swaag.benchmark import benchmark_runner

    monkeypatch.setattr(benchmark_runner, "_discover_server_context_limit", lambda base_url, timeout_seconds: 32000)
    config = benchmark_runner._build_config(
        sessions_root=tmp_path / "sessions",
        workspace=tmp_path / "workspace",
        overrides={},
        base_url="http://127.0.0.1:14829",
    )

    assert config.model.context_limit == 32000
