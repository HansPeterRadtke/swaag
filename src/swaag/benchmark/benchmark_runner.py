from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from swaag.benchmark.failure_analyzer import FailureAnalyzer
from swaag.benchmark import external as external_benchmarks
from swaag.benchmark.result_collector import BenchmarkTaskResult, ResultCollector
from swaag.benchmark.scaled_catalog import generated_live_subset_tasks, validate_live_subset_catalog
from swaag.benchmark.system_suite import get_system_benchmark_families, run_system_benchmarks
from swaag.benchmark.task_definitions import (
    BenchmarkTaskDefinition,
    PromptUnderstandingOracle,
    get_benchmark_tasks,
    validate_benchmark_catalog,
)
from swaag.config import AgentConfig, load_config
from swaag.live_runtime_profiles import get_documented_final_live_benchmark_recommendation
from swaag.model import LlamaCppClient
from swaag.runtime import AgentRuntime
from swaag.testing.llm_record_replay import RecordReplayModelClient
from swaag.utils import stable_json_dumps
from swaag.verification import verify_benchmark_contract


def _parse_seed_list(raw: str | None, *, default: tuple[int, int, int]) -> list[int]:
    if raw is None or not raw.strip():
        return list(default)
    seeds: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        seeds.append(int(value))
    return seeds or list(default)


def _build_config(
    *,
    sessions_root: Path,
    workspace: Path,
    overrides: dict[str, object],
    base_url: str,
    connect_timeout_seconds: int | None = None,
    timeout_seconds: int | None = None,
    profile_name: str | None = None,
    structured_output_mode: str | None = None,
    progress_poll_seconds: float | None = None,
    seed: int | None = None,
    retrieval_backend: str | None = None,
) -> AgentConfig:
    config = load_config(
        env={
            "SWAAG__SESSIONS__ROOT": str(sessions_root),
            "SWAAG__TOOLS__READ_ROOTS": json.dumps([str(workspace)]),
            "SWAAG__MODEL__BASE_URL": base_url,
        }
    )
    if connect_timeout_seconds is not None:
        config.model.connect_timeout_seconds = int(connect_timeout_seconds)
    if timeout_seconds is not None:
        config.model.timeout_seconds = int(timeout_seconds)
    if profile_name is not None:
        config.model.profile_name = str(profile_name)
    if structured_output_mode is not None:
        config.model.structured_output_mode = str(structured_output_mode)
    if progress_poll_seconds is not None:
        config.model.progress_poll_seconds = float(progress_poll_seconds)
    if seed is not None:
        config.model.seed = int(seed)
    if retrieval_backend is not None:
        config.retrieval.backend = str(retrieval_backend)
    config.tools.allow_side_effect_tools = bool(overrides.get("tools_allow_side_effect_tools", config.tools.allow_side_effect_tools))
    config.tools.allow_stateful_tools = bool(overrides.get("tools_allow_stateful_tools", config.tools.allow_stateful_tools))
    config.planner.max_replans = int(overrides.get("planner_max_replans", config.planner.max_replans))
    config.planner.max_plan_steps = int(overrides.get("planner_max_plan_steps", config.planner.max_plan_steps))
    config.runtime.max_reasoning_steps = int(overrides.get("runtime_max_reasoning_steps", config.runtime.max_reasoning_steps))
    config.runtime.max_total_actions = int(overrides.get("runtime_max_total_actions", config.runtime.max_total_actions))
    config.runtime.max_tool_steps = int(overrides.get("runtime_max_tool_steps", config.runtime.max_tool_steps))
    config.runtime.tool_call_budget = int(overrides.get("runtime_tool_call_budget", config.runtime.tool_call_budget))
    return config


def _latest_event_payload(events: Sequence[Any], event_type: str) -> dict[str, Any]:
    for event in reversed(events):
        if getattr(event, "event_type", None) == event_type:
            payload = getattr(event, "payload", {})
            return payload if isinstance(payload, dict) else {}
    return {}


def _task_trace_metrics(events: Sequence[Any]) -> dict[str, Any]:
    context_payload = _latest_event_payload(events, "context_built")
    verification_payload = _latest_event_payload(events, "verification_completed")
    selected_trace = [
        item
        for item in context_payload.get("selection_trace", [])
        if isinstance(item, dict) and item.get("selected")
    ]
    dropped_trace = [
        item
        for item in context_payload.get("selection_trace", [])
        if isinstance(item, dict) and not item.get("selected")
    ]
    subagent_types = sorted(
        {
            str(getattr(event, "payload", {}).get("subagent_type"))
            for event in events
            if getattr(event, "event_type", None) in {"subagent_spawned", "subagent_reported"}
            and isinstance(getattr(event, "payload", {}), dict)
            and getattr(event, "payload", {}).get("subagent_type")
        }
    )
    environment_event_types = {
        "shell_command_started",
        "shell_command_completed",
        "file_read",
        "file_write_applied",
        "edit_applied",
        "filesystem_search",
        "repository_searched",
        "workspace_snapshot_inspected",
        "changes_listed",
        "diff_inspected",
        "tool_called",
    }
    environment_counts: dict[str, int] = {}
    for event in events:
        event_type = getattr(event, "event_type", "")
        if event_type not in environment_event_types:
            continue
        environment_counts[event_type] = environment_counts.get(event_type, 0) + 1
    return {
        "retrieval_trace_sample": selected_trace[:5],
        "dropped_trace_sample": dropped_trace[:5],
        "guidance_sources": list(context_payload.get("guidance_sources", [])),
        "selected_skill_ids": list(context_payload.get("selected_skill_ids", [])),
        "exposed_tool_names": list(context_payload.get("exposed_tool_names", [])),
        "relevant_environment_files": list(context_payload.get("relevant_environment_files", [])),
        "retrieval_mode": context_payload.get("retrieval_mode", ""),
        "retrieval_degraded": bool(context_payload.get("retrieval_degraded", False)),
        "subagent_usage": subagent_types,
        "verification_trace": {
            "verification_type_used": verification_payload.get("verification_type_used", ""),
            "conditions_met": list(verification_payload.get("conditions_met", [])),
            "conditions_failed": list(verification_payload.get("conditions_failed", [])),
            "reason": verification_payload.get("reason", ""),
            "confidence": verification_payload.get("confidence", 0.0),
        },
        "environment_operations_summary": environment_counts,
    }


def _validate_live_subset(tasks: list[BenchmarkTaskDefinition]) -> None:
    validate_live_subset_catalog(tasks)


def _load_tasks(task_ids: list[str] | None, *, live_subset: bool) -> list[BenchmarkTaskDefinition]:
    tasks = generated_live_subset_tasks() if live_subset else get_benchmark_tasks()
    if live_subset:
        _validate_live_subset(tasks)
    else:
        validate_benchmark_catalog(tasks)
    if not task_ids:
        return tasks
    by_id = {task.task_id: task for task in tasks}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise SystemExit(f"Unknown benchmark task ids: {', '.join(missing)}")
    return [by_id[task_id] for task_id in task_ids]


def _resolve_live_model_settings(
    *,
    use_live_model: bool,
    model_base_url: str | None,
    timeout_seconds: int | None,
    connect_timeout_seconds: int | None,
    model_profile: str | None,
    structured_output_mode: str | None,
    progress_poll_seconds: float | None,
    seeds: list[int] | None = None,
) -> dict[str, str | int | float | None]:
    recommendation = get_documented_final_live_benchmark_recommendation()
    return {
        "base_url": model_base_url or ("http://benchmark.local" if not use_live_model else os.environ.get("SWAAG_LIVE_BASE_URL", "http://127.0.0.1:14829")),
        "timeout_seconds": timeout_seconds if timeout_seconds is not None else (int(os.environ.get("SWAAG_LIVE_TIMEOUT_SECONDS", str(recommendation.timeout_seconds))) if use_live_model else None),
        "connect_timeout_seconds": connect_timeout_seconds if connect_timeout_seconds is not None else (int(os.environ.get("SWAAG_LIVE_CONNECT_TIMEOUT_SECONDS", str(recommendation.connect_timeout_seconds))) if use_live_model else None),
        "model_profile": model_profile or (os.environ.get("SWAAG_LIVE_MODEL_PROFILE", recommendation.model_profile) if use_live_model else None),
        "structured_output_mode": structured_output_mode or (os.environ.get("SWAAG_LIVE_STRUCTURED_OUTPUT_MODE", recommendation.structured_output_mode) if use_live_model else None),
        "progress_poll_seconds": progress_poll_seconds if progress_poll_seconds is not None else (float(os.environ.get("SWAAG_LIVE_PROGRESS_POLL_SECONDS", str(recommendation.progress_poll_seconds))) if use_live_model else None),
        "seeds": seeds if seeds is not None else (_parse_seed_list(os.environ.get("SWAAG_LIVE_SEEDS"), default=recommendation.seeds) if use_live_model else [42]),
        "recommendation_use_case": recommendation.use_case if use_live_model else None,
        "recommendation_rationale": recommendation.rationale if use_live_model else None,
    }


def _normalize_agent_behavior_mode(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if not value:
        return None
    if value != "cached":
        raise SystemExit(f"Unsupported agent behavior test mode: {raw}. The test system only supports cached mode.")
    return value

def _build_agent_behavior_model_client(
    *,
    config: AgentConfig,
    scenario,
    task: BenchmarkTaskDefinition,
    output_dir: Path,
    seed: int,
    agent_behavior_mode: str | None,
    live_subset: bool,
):
    delegate = scenario.model_client
    if agent_behavior_mode != "cached":
        if delegate is None and live_subset:
            return LlamaCppClient(config), None
        return delegate, None
    if delegate is not None and getattr(delegate, "is_record_replay_client", False):
        return delegate, None
    # Cached agent tests use real model responses behind the replay cache.
    delegate = LlamaCppClient(config)
    replay_cache_root = output_dir / "replay_cache" / task.task_id
    os.makedirs(replay_cache_root, exist_ok=True)
    cassette_path = replay_cache_root / f"seed_{seed}.json"
    replay_mode = "replay" if cassette_path.exists() else "record"
    wrapped = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode=replay_mode,
        delegate=delegate,
        request_metadata={
            "agent_behavior_mode": "cached",
            "benchmark_scope": "validation_subset" if live_subset else "benchmark_catalog",
            "task_id": task.task_id,
            "task_type": task.task_type,
            "difficulty": task.difficulty,
        },
    )
    return wrapped, {"cassette_path": str(cassette_path), "cache_mode": replay_mode}


def _snapshot_workspace(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not root.exists():
        return snapshot
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts:
            continue
        raw = path.read_bytes()
        try:
            snapshot[str(path.relative_to(root))] = raw.decode("utf-8")
        except UnicodeDecodeError:
            snapshot[str(path.relative_to(root))] = "hex:" + raw.hex()
    return snapshot


def _last_payload(events, event_type: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.event_type == event_type:
            return event.payload
    return None


def _evaluate_quality(oracle: PromptUnderstandingOracle | None, state, events) -> dict[str, Any]:
    if oracle is None:
        return {"passed": True, "checks": {}, "evidence": {}, "oracle": {}}
    checks: dict[str, bool] = {}
    evidence: dict[str, Any] = {}
    analysis = state.prompt_analysis
    decision = state.latest_decision
    strategy = state.active_strategy
    expanded_task = state.expanded_task
    plan = state.active_plan

    def _check(name: str, actual: Any, expected: Any) -> None:
        checks[name] = actual == expected
        evidence[name] = {"actual": actual, "expected": expected}

    if oracle.task_type is not None:
        _check("task_type", analysis.task_type if analysis else None, oracle.task_type)
    if oracle.completeness is not None:
        _check("completeness", analysis.completeness if analysis else None, oracle.completeness)
    if oracle.requires_expansion is not None:
        _check("requires_expansion", analysis.requires_expansion if analysis else None, oracle.requires_expansion)
    if oracle.requires_decomposition is not None:
        _check("requires_decomposition", analysis.requires_decomposition if analysis else None, oracle.requires_decomposition)
    if oracle.expand_task is not None:
        _check("expand_task", decision.expand_task if decision else None, oracle.expand_task)
    if oracle.split_task is not None:
        _check("split_task", decision.split_task if decision else None, oracle.split_task)
    if oracle.ask_user is not None:
        _check("ask_user", decision.ask_user if decision else None, oracle.ask_user)
    if oracle.assume_missing is not None:
        _check("assume_missing", decision.assume_missing if decision else None, oracle.assume_missing)
    if oracle.generate_ideas is not None:
        _check("generate_ideas", decision.generate_ideas if decision else None, oracle.generate_ideas)
    if oracle.strategy_profile is not None:
        _check("strategy_profile", strategy.task_profile if strategy else None, oracle.strategy_profile)
    for goal in oracle.detected_goals_contains:
        actual_goals = [] if analysis is None else list(analysis.detected_goals)
        checks[f"goal:{goal}"] = any(goal.lower() in candidate.lower() for candidate in actual_goals)
        evidence[f"goal:{goal}"] = {"actual": actual_goals, "expected_contains": goal}
    for entity in oracle.detected_entities_contains:
        checks[f"entity:{entity}"] = analysis is not None and any(entity in candidate for candidate in analysis.detected_entities)
        evidence[f"entity:{entity}"] = {"actual": [] if analysis is None else list(analysis.detected_entities), "expected_contains": entity}
    if oracle.expand_task is True:
        checks["expanded_task_present"] = expanded_task is not None
        evidence["expanded_task_present"] = {"present": expanded_task is not None}
    if plan is not None:
        evidence["plan_step_kinds"] = [step.kind for step in plan.steps]
    payload = _last_payload(events, "strategy_selected")
    if payload is not None:
        evidence["strategy_event"] = payload.get("strategy")
    return {
        "passed": all(checks.values()) if checks else True,
        "checks": checks,
        "evidence": evidence,
        "oracle": asdict(oracle),
    }


def _print_benchmark_progress(*, current: int, total: int, task: BenchmarkTaskDefinition, status: str) -> None:
    percent = 0.0 if total == 0 else (current / total) * 100.0
    print(
        f"[agent_test {current}/{total} {percent:5.1f}%] {status} {task.task_id} "
        f"({task.task_type}/{task.difficulty})",
        flush=True,
    )


def _print_benchmark_summary(report: dict[str, Any]) -> None:
    summary = report.get("summary", {})
    aggregate_metrics = report.get("aggregate_metrics", {})
    family_scores = dict(summary.get("score_by_family", {}))
    difficulty_scores = dict(summary.get("score_by_difficulty", {}))
    print("agent_test_summary", flush=True)
    print(f"  total_tasks={summary.get('total_tasks', 0)}", flush=True)
    print(f"  successful_tasks={summary.get('successful_tasks', 0)}", flush=True)
    print(f"  failed_tasks={summary.get('failed_tasks', 0)}", flush=True)
    print(f"  false_positives={summary.get('false_positives', 0)}", flush=True)
    print(f"  full_task_success_percent={float(summary.get('full_task_success_percent', 0.0)):.2f}", flush=True)
    print(f"  group_average_percent={float(summary.get('group_average_percent', 0.0)):.2f}", flush=True)
    print(
        f"  difficulty_group_average_percent={float(summary.get('difficulty_group_average_percent', 0.0)):.2f}",
        flush=True,
    )
    print(
        f"  family_group_average_percent={float(summary.get('family_group_average_percent', 0.0)):.2f}",
        flush=True,
    )
    print(f"  average_task_score_percent={float(summary.get('average_task_score_percent', 0.0)):.2f}", flush=True)
    print("  detailed_substep_score=omitted_unreliable", flush=True)
    if difficulty_scores:
        print(f"  group_scores_by_difficulty={stable_json_dumps(dict(sorted(difficulty_scores.items())))}", flush=True)
    if family_scores:
        print(f"  group_scores_by_family={stable_json_dumps(dict(sorted(family_scores.items())))}", flush=True)
    primary = aggregate_metrics.get("primary", {})
    if primary:
        print(
            f"  task_success_rate={float(primary.get('task_success_rate', 0.0)) * 100.0:.2f}",
            flush=True,
        )


def run_benchmarks(
    *,
    output_dir: Path,
    task_ids: list[str] | None = None,
    clean: bool = False,
    live_subset: bool = False,
    use_live_model: bool = False,
    model_base_url: str | None = None,
    timeout_seconds: int | None = None,
    connect_timeout_seconds: int | None = None,
    model_profile: str | None = None,
    structured_output_mode: str | None = None,
    progress_poll_seconds: float | None = None,
    seeds: list[int] | None = None,
    agent_behavior_mode: str | None = None,
) -> dict[str, object]:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    workspaces_root = output_dir / "workspaces"
    sessions_root = output_dir / "sessions"
    os.makedirs(workspaces_root, exist_ok=True)
    os.makedirs(sessions_root, exist_ok=True)

    analyzer = FailureAnalyzer()
    collector = ResultCollector()
    selected_tasks = _load_tasks(task_ids, live_subset=live_subset)
    resolved_agent_behavior_mode = _normalize_agent_behavior_mode(agent_behavior_mode)
    benchmark_started = time.monotonic()
    live_settings = _resolve_live_model_settings(
        use_live_model=use_live_model,
        model_base_url=model_base_url,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        model_profile=model_profile,
        structured_output_mode=structured_output_mode,
        progress_poll_seconds=progress_poll_seconds,
        seeds=seeds,
    )
    effective_base_url = str(live_settings["base_url"])
    effective_timeout = live_settings["timeout_seconds"]
    effective_connect_timeout = live_settings["connect_timeout_seconds"]
    effective_profile = live_settings["model_profile"]
    effective_structured_output_mode = live_settings["structured_output_mode"]
    effective_progress_poll = live_settings["progress_poll_seconds"]
    effective_seeds = [int(seed) for seed in live_settings.get("seeds", [42]) or [42]]
    effective_profile_use_case = live_settings.get("recommendation_use_case")
    effective_profile_rationale = live_settings.get("recommendation_rationale")
    effective_retrieval_backend = (
        os.environ.get("SWAAG_BENCHMARK_TEST_RETRIEVAL_BACKEND", "degraded_lexical")
        if not use_live_model
        else None
    )

    total_tasks = len(selected_tasks)
    for task_index, task in enumerate(selected_tasks, start=1):
        _print_benchmark_progress(current=task_index - 1, total=total_tasks, task=task, status="running")
        seed_results: list[dict[str, Any]] = []
        aggregate_success = True
        aggregate_false_positive = False
        aggregate_verification_passed = True
        aggregate_assistant_text = ""
        aggregate_failure = None
        aggregate_history_paths: list[str] = []
        aggregate_workspace_paths: list[str] = []
        aggregate_metrics: list[dict[str, Any]] = []
        aggregate_verification_summary: dict[str, Any] | None = None
        aggregate_quality_summary: dict[str, Any] | None = None
        aggregate_wall_clock = 0.0
        aggregate_session_id: str | None = None
        for seed in effective_seeds:
            scenario_root = workspaces_root if not use_live_model else workspaces_root / f"seed_{seed}"
            scenario = task.create(scenario_root, live_mode=use_live_model)
            before_snapshot = _snapshot_workspace(scenario.workspace)
            scenario_retrieval_backend = effective_retrieval_backend
            config = _build_config(
                sessions_root=sessions_root,
                workspace=scenario.workspace,
                overrides=task.config_overrides,
                base_url=effective_base_url,
                connect_timeout_seconds=effective_connect_timeout,
                timeout_seconds=effective_timeout,
                profile_name=effective_profile,
                structured_output_mode=effective_structured_output_mode,
                progress_poll_seconds=effective_progress_poll,
                seed=seed,
                retrieval_backend=scenario_retrieval_backend,
            )
            runtime_model_client, replay_cache_info = _build_agent_behavior_model_client(
                config=config,
                scenario=scenario,
                task=task,
                output_dir=output_dir,
                seed=seed,
                agent_behavior_mode=resolved_agent_behavior_mode,
                live_subset=live_subset,
            )
            runtime = AgentRuntime(config, model_client=runtime_model_client)
            state = runtime.create_or_load_session()
            runtime_error: Exception | None = None
            assistant_text = ""
            task_started = time.monotonic()
            try:
                turn = runtime.run_turn_in_session(state, scenario.prompt)
                assistant_text = turn.assistant_text
            except Exception as exc:
                runtime_error = exc
            task_elapsed = round(time.monotonic() - task_started, 3)
            aggregate_wall_clock += task_elapsed
            after_snapshot = _snapshot_workspace(scenario.workspace)
            rebuilt = runtime.history.rebuild_from_history(state.session_id)
            events = runtime.history.read_history(state.session_id)
            trace_metrics = _task_trace_metrics(events)
            final_text = assistant_text or (rebuilt.messages[-1].content if rebuilt.messages and rebuilt.messages[-1].role == "assistant" else "")
            verification = verify_benchmark_contract(
                scenario.verification_contract,
                assistant_text=final_text,
                state=rebuilt,
                events=events,
                workspace_before=before_snapshot,
                workspace_after=after_snapshot,
            )
            quality = _evaluate_quality(scenario.oracle, rebuilt, events)
            failure = None
            if not verification.passed or scenario.expected_outcome == "expected_failure" or not quality["passed"]:
                failure = analyzer.analyze(
                    state=rebuilt,
                    events=events,
                    deterministic_verification_passed=verification.passed,
                    runtime_error=runtime_error,
                )
            if scenario.expected_outcome == "success":
                success = verification.passed and runtime_error is None and bool(quality["passed"])
                false_positive = runtime_error is None and bool(final_text.strip()) and (not verification.passed or not bool(quality["passed"]))
            else:
                success = verification.passed and failure is not None and failure.category == scenario.expected_failure_category
                false_positive = verification.passed and failure is not None and failure.category != scenario.expected_failure_category
            seed_results.append(
                {
                    "seed": seed,
                    "success": success,
                    "false_positive": false_positive,
                    "deterministic_verification_passed": verification.passed,
                    "assistant_text": final_text,
                    "session_id": state.session_id,
                    "history_path": str(runtime.history.history_path(state.session_id)),
                    "workspace": str(scenario.workspace),
                    "failure_category": failure.category if failure is not None else None,
                    "failure_reason": failure.reason if failure is not None else None,
                    "failure_subsystem": failure.subsystem if failure is not None else None,
                    "wall_clock_seconds": task_elapsed,
                    "replay_cache": replay_cache_info or {},
                }
            )
            aggregate_success = aggregate_success and success
            aggregate_false_positive = aggregate_false_positive or false_positive
            aggregate_verification_passed = aggregate_verification_passed and verification.passed
            aggregate_assistant_text = final_text
            aggregate_failure = aggregate_failure or failure
            aggregate_history_paths.append(str(runtime.history.history_path(state.session_id)))
            aggregate_workspace_paths.append(str(scenario.workspace))
            aggregate_metrics.append(asdict(rebuilt.metrics))
            aggregate_verification_summary = {"checks": verification.checks, "evidence": verification.evidence, "reason": verification.reason}
            aggregate_quality_summary = quality
            aggregate_session_id = aggregate_session_id or state.session_id
        collector.add(
            BenchmarkTaskResult(
                task_id=task.task_id,
                task_type=task.task_type,
                difficulty=task.difficulty,
                tags=list(task.tags),
                description=task.description,
                expected_outcome=scenario.expected_outcome,
                success=aggregate_success,
                false_positive=aggregate_false_positive,
                session_id=aggregate_session_id,
                assistant_text=aggregate_assistant_text,
                deterministic_verification_passed=aggregate_verification_passed,
                verification_summary=aggregate_verification_summary or {"checks": {}, "evidence": {}, "reason": ""},
                quality_summary=aggregate_quality_summary or {"passed": True, "checks": {}, "evidence": {}, "oracle": {}},
                metrics={
                    **(aggregate_metrics[-1] if aggregate_metrics else {}),
                    **trace_metrics,
                    "seed_results": seed_results,
                    "seed_success_count": sum(1 for item in seed_results if item["success"]),
                    "seed_false_positive_count": sum(1 for item in seed_results if item["false_positive"]),
                    "seed_variation": len({item["success"] for item in seed_results}) > 1 or len({item["assistant_text"] for item in seed_results}) > 1,
                },
                failure_category=aggregate_failure.category if aggregate_failure is not None else None,
                failure_reason=aggregate_failure.reason if aggregate_failure is not None else None,
                failure_subsystem=aggregate_failure.subsystem if aggregate_failure is not None else None,
                wall_clock_seconds=round(aggregate_wall_clock, 3),
                improvement_hints=[] if aggregate_failure is None or not aggregate_failure.improvement_hints else list(aggregate_failure.improvement_hints),
                history_path=aggregate_history_paths[0] if aggregate_history_paths else None,
                workspace=aggregate_workspace_paths[0] if aggregate_workspace_paths else "",
            )
        )
        _print_benchmark_progress(
            current=task_index,
            total=total_tasks,
            task=task,
            status=f"finished success={aggregate_success} false_positive={aggregate_false_positive}",
        )
    if resolved_agent_behavior_mode == "cached":
        artifact_prefix = "agent_test_cached"
    else:
        artifact_prefix = "manual_validation" if live_subset and use_live_model else "benchmark"
    report = collector.write(
        output_dir,
        prefix=artifact_prefix,
        run_metadata={
            "mode": "live_subset" if live_subset else "full",
            "use_live_model": use_live_model,
            "agent_behavior_mode": resolved_agent_behavior_mode or "",
            "replay_cache_enabled": resolved_agent_behavior_mode == "cached",
            "replay_cache_root": str(output_dir / "replay_cache") if resolved_agent_behavior_mode == "cached" else "",
            "replay_cache_policy": "replay_if_present_record_if_missing" if resolved_agent_behavior_mode == "cached" else "",
            "request_observability_mode": (
                "replay_cache_or_progress_polling"
                if resolved_agent_behavior_mode == "cached"
                else "progress_polling"
            ),
            "model_base_url": effective_base_url,
            "model_profile": effective_profile or "",
            "profile_use_case": effective_profile_use_case or "",
            "profile_rationale": effective_profile_rationale or "",
            "structured_output_mode": effective_structured_output_mode or "",
            "timeout_seconds": effective_timeout or "",
            "connect_timeout_seconds": effective_connect_timeout or "",
            "progress_poll_seconds": effective_progress_poll or "",
            "seeds": effective_seeds,
            "wall_clock_seconds": round(time.monotonic() - benchmark_started, 3),
            "task_count": len(selected_tasks),
        },
    )
    payload = asdict(report)
    _print_benchmark_summary(payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m swaag.benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the built-in benchmark suite.")
    run_parser.add_argument("--output", default="benchmark_output", help="Output directory for benchmark results and session histories.")
    run_parser.add_argument("--task", action="append", default=[], help="Run only the named task id. Can be passed multiple times.")
    run_parser.add_argument("--clean", action="store_true", help="Delete the output directory before running.")
    run_parser.add_argument("--json", action="store_true", help="Print the full result JSON.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Run code-correctness plus the benchmark evaluator.")
    evaluate_parser.add_argument("--output", default="evaluation_output", help="Output directory for evaluation results.")
    evaluate_parser.add_argument("--clean", action="store_true", help="Delete the output directory before running.")
    evaluate_parser.add_argument("--task", action="append", default=[], help="Run only the named benchmark task id. Can be passed multiple times.")
    evaluate_parser.add_argument("--pytest-arg", action="append", default=[], help="Additional argument forwarded to the code-correctness pytest command.")
    evaluate_parser.add_argument("--json", action="store_true", help="Print the full evaluation JSON.")

    agent_tests_parser = subparsers.add_parser("agent-tests", help="Run the real cached benchmark for agent_test.")
    agent_tests_parser.add_argument("--output", default="agent_test_output", help="Output directory for cached agent-test results.")
    agent_tests_parser.add_argument("--clean", action="store_true", help="Delete the output directory before running.")
    agent_tests_parser.add_argument("--json", action="store_true", help="Print the full agent-test JSON.")

    test_categories_parser = subparsers.add_parser("test-categories", help="Run code_correctness, then the real cached benchmark for agent_test only if code_correctness is 100%% green.")
    test_categories_parser.add_argument("--output", default="test_categories_output", help="Output directory for category results and reports.")
    test_categories_parser.add_argument("--clean", action="store_true", help="Delete the output directory before running.")
    test_categories_parser.add_argument("--pytest-arg", action="append", default=[], help="Additional argument forwarded to the code-correctness pytest command.")
    test_categories_parser.add_argument("--json", action="store_true", help="Print the full category JSON.")

    manual_parser = subparsers.add_parser("manual-validation", help="Run explicit real-model validation. This is not a test category.")
    manual_parser.set_defaults(validation_subset=True)
    manual_parser.add_argument("--output", default="manual_validation_output", help="Output directory for manual validation artifacts.")
    manual_parser.add_argument("--clean", action="store_true", help="Delete the output directory before running.")
    manual_parser.add_argument("--task", action="append", default=[], help="Run only the named validation task id. Can be passed multiple times.")
    manual_parser.add_argument("--validation-subset", dest="validation_subset", action="store_true", help="Run the curated manual-validation subset.")
    manual_parser.add_argument("--full-catalog", dest="validation_subset", action="store_false", help="Run the full benchmark catalog instead of the curated manual-validation subset.")
    manual_parser.add_argument("--model-base-url", help="Override the llama.cpp base URL for manual validation.")
    manual_parser.add_argument("--timeout-seconds", type=int, help="Override the model read timeout.")
    manual_parser.add_argument("--connect-timeout-seconds", type=int, help="Override the model connect timeout.")
    manual_parser.add_argument("--model-profile", help="Record the llama.cpp profile used for manual validation.")
    manual_parser.add_argument("--structured-output-mode", choices=["server_schema", "post_validate", "auto"], help="Override structured output mode.")
    manual_parser.add_argument("--progress-poll-seconds", type=float, help="Override model progress polling interval.")
    manual_parser.add_argument("--seeds", help="Comma-separated fixed seeds for model-backed manual validation.")
    manual_parser.add_argument("--json", action="store_true", help="Print the full manual-validation JSON.")

    subparsers.add_parser("list", help="List available benchmark task ids.")
    system_parser = subparsers.add_parser("system", help="Run deterministic agent-system benchmark families.")
    system_parser.add_argument("--family", action="append", default=[], help="Benchmark family id to run. Repeat to narrow the run.")
    system_parser.add_argument("--all", action="store_true", help="Run every configured family.")
    system_parser.add_argument("--output", default="system_benchmark_output", help="Output directory for system benchmark reports.")
    system_parser.add_argument("--clean", action="store_true", help="Delete the output directory before running.")
    system_parser.add_argument("--json", action="store_true", help="Print the full result JSON.")
    external_parser = subparsers.add_parser("external", help="Run optional official external benchmark harness integrations.")
    external_parser.add_argument("external_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the external benchmark runner.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        tasks = _load_tasks(None, live_subset=False)
        for task in tasks:
            print(f"{task.task_id}: {task.description}")
        return 0
    if args.command == "run":
        report = run_benchmarks(
            output_dir=Path(args.output),
            task_ids=list(args.task),
            clean=bool(args.clean),
            live_subset=False,
            use_live_model=False,
        )
        if args.json:
            print(stable_json_dumps(report, indent=2))
        else:
            summary = report["summary"]
            print(f"total_tasks={summary['total_tasks']}")
            print(f"successful_tasks={summary['successful_tasks']}")
            print(f"failed_tasks={summary['failed_tasks']}")
            print(f"false_positives={summary['false_positives']}")
        summary = report["summary"]
        return 0 if summary["failed_tasks"] == 0 and summary["false_positives"] == 0 else 1
    if args.command == "evaluate":
        from swaag.benchmark.evaluation_runner import run_full_evaluation

        report = run_full_evaluation(
            output_dir=Path(args.output),
            clean=bool(args.clean),
            functional_pytest_args=list(args.pytest_arg),
            benchmark_task_ids=list(args.task),
            live_subset=False,
            use_live_model=False,
        )
        if args.json:
            print(stable_json_dumps(report, indent=2))
        else:
            print(f"overall_percent={report['overall_percent']}")
            print(f"code_correctness_percent={report['code_correctness']['summary']['percent']}")
        return 0 if report["overall_percent"] == 100.0 else 1
    if args.command == "agent-tests":
        from swaag.benchmark.evaluation_runner import run_agent_test_category

        report = run_agent_test_category(output_dir=Path(args.output), clean=bool(args.clean))
        if args.json:
            print(stable_json_dumps(report, indent=2))
        else:
            score_summary = report["score_summary"]
            print(f"group_average_percent={score_summary['group_average_percent']}")
            print(f"full_task_success_percent={score_summary['full_task_success_percent']}")
            print(f"average_task_score_percent={score_summary['average_task_score_percent']}")
            print(f"cached_benchmark_results_path={report['cached_benchmark_results_path']}")
        return 0
    if args.command == "test-categories":
        from swaag.benchmark.evaluation_runner import run_test_category_evaluation

        report = run_test_category_evaluation(
            output_dir=Path(args.output),
            clean=bool(args.clean),
            functional_pytest_args=list(args.pytest_arg) or None,
        )
        if args.json:
            print(stable_json_dumps(report, indent=2))
        else:
            print(f"code_correctness_binary_result={'passed' if report['code_correctness_binary_passed'] else 'failed'}")
            print(f"code_correctness_percent={report['code_correctness']['summary']['percent']}")
            print(f"agent_test_ran={report['agent_test_ran']}")
            if report.get("agent_test"):
                score_summary = report["agent_test"]["score_summary"]
                print(f"agent_test_group_average_percent={score_summary['group_average_percent']}")
                print(f"agent_test_full_task_success_percent={score_summary['full_task_success_percent']}")
                print(f"agent_test_average_task_score_percent={score_summary['average_task_score_percent']}")
            if report.get("skip_reason"):
                print(f"skip_reason={report['skip_reason']}")
        return 0 if report["code_correctness_binary_passed"] else 1
    if args.command == "manual-validation":
        from swaag.manual_validation.runner import run_manual_validation

        report = run_manual_validation(
            output_dir=Path(args.output),
            clean=bool(args.clean),
            benchmark_task_ids=list(args.task) or None,
            validation_subset=bool(args.validation_subset),
            model_base_url=args.model_base_url,
            timeout_seconds=args.timeout_seconds,
            connect_timeout_seconds=args.connect_timeout_seconds,
            model_profile=args.model_profile,
            structured_output_mode=args.structured_output_mode,
            progress_poll_seconds=args.progress_poll_seconds,
            seeds=_parse_seed_list(args.seeds, default=get_documented_final_live_benchmark_recommendation().seeds) if args.seeds else None,
        )
        if args.json:
            print(stable_json_dumps(report, indent=2))
        else:
            print("manual_validation_not_test_category=true")
            print(f"percent={report['percent']}")
            print(f"task_count={report['summary']['total_tasks']}")
        return 0 if report["summary"].get("failed_tasks", 0) == 0 and report["summary"].get("false_positives", 0) == 0 else 1
    if args.command == "system":
        selected_families = list(args.family)
        if args.all:
            selected_families = [family.family_id for family in get_system_benchmark_families()]
        report = run_system_benchmarks(
            output_dir=Path(args.output),
            family_ids=selected_families or None,
            clean=bool(args.clean),
        )
        if args.json:
            print(stable_json_dumps(report, indent=2))
        else:
            summary = report["summary"]
            print(f"total_families={summary['total_families']}")
            print(f"passed_families={summary['passed_families']}")
            print(f"failed_families={summary['failed_families']}")
        return 0 if report["summary"]["failed_families"] == 0 else 1
    if args.command == "external":
        forwarded = list(args.external_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return external_benchmarks.main(forwarded)
    raise SystemExit(f"Unhandled command: {args.command}")
