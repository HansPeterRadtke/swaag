from __future__ import annotations

import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from swaag.fsops import ensure_dir, write_text
from swaag.utils import stable_json_dumps


@dataclass(frozen=True, slots=True)
class AgentSupportFamily:
    family_id: str
    description: str
    nodeids: tuple[str, ...]


@dataclass(slots=True)
class AgentSupportFamilyResult:
    family_id: str
    description: str
    command: list[str]
    status: str
    score_percent: float
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    junit_path: str
    nodeids: list[str]
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_tests: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
            }
        }
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    tests = list(root.iter("testcase"))
    passed = 0
    failed = 0
    skipped = 0
    for testcase in tests:
        if testcase.find("failure") is not None or testcase.find("error") is not None:
            failed += 1
        elif testcase.find("skipped") is not None:
            skipped += 1
        else:
            passed += 1
    executed = passed + failed
    return {
        "summary": {
            "total_tests": len(tests),
            "executed_tests": executed,
            "passed_tests": passed,
            "failed_tests": failed,
            "skipped_tests": skipped,
            "percent": round((passed / executed) * 100.0, 2) if executed else 0.0,
        }
    }


def get_agent_support_families() -> list[AgentSupportFamily]:
    return [
        AgentSupportFamily(
            family_id="continuation_background",
            description="Background job overlap, wait/resume, and recovered finalization.",
            nodeids=(
                "tests/test_runtime.py::test_runtime_continues_other_ready_work_while_background_process_runs",
                "tests/test_runtime.py::test_runtime_enters_wait_state_when_only_background_work_remains",
                "tests/test_history.py::test_history_rebuild_tracks_wait_events_and_background_process_completion",
            ),
        ),
        AgentSupportFamily(
            family_id="session_control",
            description="Active-session control queuing, non-destructive continuation, and deferred work.",
            nodeids=(
                "tests/test_session_control.py::test_runtime_processes_continue_control_without_stopping_current_work",
                "tests/test_session_control.py::test_process_pending_control_messages_returns_non_destructive_control_responses",
                "tests/test_session_control.py::test_process_pending_control_messages_queues_follow_up_task",
                "tests/test_session_control.py::test_continue_with_note_adds_note_without_forcing_replan",
            ),
        ),
        AgentSupportFamily(
            family_id="session_history_checkpoint",
            description="Human-readable session UX, exact-detail history retrieval, and checkpoints.",
            nodeids=(
                "tests/test_sessions.py::test_runtime_create_or_load_user_session_defaults_to_latest",
                "tests/test_session_control.py::test_cli_sessions_list_and_rename_use_human_names",
                "tests/test_session_control.py::test_history_detail_query_returns_exact_old_command",
                "tests/test_session_control.py::test_history_detail_query_handles_quoted_exact_terms",
                "tests/test_session_control.py::test_code_checkpoint_create_and_restore",
            ),
        ),
        AgentSupportFamily(
            family_id="artifact_recovery",
            description="Artifact-first state tracking and bounded recovery after replans.",
            nodeids=(
                "tests/test_project_state.py::test_artifact_tracking_pending_steps_appear_in_expected_and_pending",
                "tests/test_project_state.py::test_artifact_tracking_completed_steps_move_to_completed",
                "tests/test_project_state.py::test_artifact_state_is_rendered_in_context",
                "tests/test_long_running_tasks.py::test_runtime_recovers_after_multiple_replans_and_remains_bounded",
            ),
        ),
        AgentSupportFamily(
            family_id="runtime_recovery_contracts",
            description="Runtime recovery, retry, and contract repair paths stay stable under replay.",
            nodeids=(
                "tests/test_runtime.py::test_runtime_retries_failed_model_request",
                "tests/test_runtime.py::test_runtime_retries_after_verification_failure",
                "tests/test_runtime.py::test_runtime_recovers_malformed_coding_plan_with_shell_recovery_plan",
                "tests/test_runtime.py::test_runtime_recovers_strategy_incompatible_coding_plan_with_shell_recovery_plan",
            ),
        ),
        AgentSupportFamily(
            family_id="direct_response_guardrails",
            description="Direct-answer shortcuts stay bounded by explicit-tool and exact-reply guardrails.",
            nodeids=(
                "tests/test_runtime.py::test_extract_unconditional_exact_reply_ignores_conditional_reply_clauses",
                "tests/test_runtime.py::test_runtime_finalizes_unconditional_exact_reply_without_final_model_call",
                "tests/test_runtime.py::test_runtime_bypasses_llm_plan_generation_for_semantic_direct_response",
                "tests/test_runtime.py::test_runtime_blocks_direct_response_when_prompt_explicitly_requires_named_tool",
                "tests/test_runtime.py::test_runtime_blocks_direct_response_when_strategy_requires_write_steps",
            ),
        ),
        AgentSupportFamily(
            family_id="tool_routing_context_focus",
            description="Tool routing, targeted shell recovery, and edit-context focus remain anchored in real evidence.",
            nodeids=(
                "tests/test_runtime.py::test_runtime_uses_expected_tool_input_contract_for_shell_command_steps",
                "tests/test_runtime.py::test_runtime_normalizes_trivial_shell_command_into_repo_search",
                "tests/test_runtime.py::test_runtime_prefers_recent_source_hint_for_edit_text_step",
                "tests/test_runtime.py::test_runtime_edit_text_prompt_includes_recent_inspection_evidence",
                "tests/test_runtime.py::test_runtime_uses_expected_tool_input_contract_for_profile_optimized_edit_steps",
            ),
        ),
        AgentSupportFamily(
            family_id="subagent_traceability",
            description="Retriever, reviewer, and planner subagents emit auditable runtime artifacts and events.",
            nodeids=(
                "tests/test_subagents.py::test_runtime_records_subagent_events_during_review",
                "tests/test_subagents.py::test_runtime_records_retriever_subagent_events_during_context_build",
                "tests/test_subagents.py::test_retriever_subagent_produces_focused_artifact",
                "tests/test_subagents.py::test_planner_subagent_replan_artifact_is_explicit",
            ),
        ),
        AgentSupportFamily(
            family_id="record_replay_runtime",
            description="Replay-backed runtime behavior check using recorded full-request payload keys.",
            nodeids=("tests/test_agent_loop_replay.py::test_record_replay_client_replays_runtime_tool_flow",),
        ),
        AgentSupportFamily(
            family_id="scripted_benchmark_runtime",
            description="Scripted benchmark tasks still exercise the real agent loop and reject false positives.",
            nodeids=(
                "tests/test_false_positive_killers.py::test_false_positive_killer_tasks_never_report_success_incorrectly",
                "tests/test_scaled_catalog.py::test_scaled_catalog_tasks_run_through_benchmark_runner",
            ),
        ),
    ]


def run_agent_behavior_support_checks(
    *,
    output_dir: Path,
    family_ids: Sequence[str] | None = None,
    clean: bool = False,
) -> dict[str, Any]:
    """Run focused cached-mode agent behavior support-check families.

    Each test family runs the real agent runtime/orchestrator/tool loop without
    no-cache validation traffic. Model calls are handled by one of these
    mechanisms, depending on the family:

    - FakeModelClient: scripted test fixture returning exact pre-specified
      responses in sequence. Used for runtime-behavior and recovery families
      where precise control over model output is required.
    - ScriptedBenchmarkClient: scripted fixture for full benchmark pipeline
      tests. Covers task-catalog and false-positive-killer families.
    - RecordReplayModelClient: replay-cached responses keyed by normalized
      full request payload hash. Used for the ``record_replay_runtime`` family
      to demonstrate that the replay mechanism itself works correctly.

    All three are "cached mode" support mechanisms in the sense that none make
    no-cache validation model calls. The RecordReplayModelClient is the
    authoritative cassette-backed replay path.
    FakeModelClient and ScriptedBenchmarkClient are deterministic scripted fixtures
    appropriate for tests that need precise control over the agent's decision path.
    """
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    families = get_agent_support_families()
    if family_ids:
        selected_map = {item.family_id: item for item in families}
        missing = [family_id for family_id in family_ids if family_id not in selected_map]
        if missing:
            raise SystemExit(f"Unknown agent behavior support families: {', '.join(missing)}")
        families = [selected_map[item] for item in family_ids]
    results: list[AgentSupportFamilyResult] = []
    started = time.monotonic()
    repo_root = _repo_root()
    for family in families:
        family_dir = output_dir / family.family_id
        ensure_dir(family_dir)
        junit_path = family_dir / "junit.xml"
        stdout_path = family_dir / "stdout.txt"
        stderr_path = family_dir / "stderr.txt"
        command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}", *family.nodeids, "-x"]
        run_started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
        )
        write_text(stdout_path, completed.stdout, encoding="utf-8")
        write_text(stderr_path, completed.stderr, encoding="utf-8")
        parsed = _parse_junit_results(junit_path)["summary"]
        results.append(
            AgentSupportFamilyResult(
                family_id=family.family_id,
                description=family.description,
                command=command,
                status="passed" if completed.returncode == 0 else "failed",
                score_percent=float(parsed["percent"]),
                duration_seconds=round(time.monotonic() - run_started, 3),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                junit_path=str(junit_path),
                nodeids=list(family.nodeids),
                passed_tests=int(parsed["passed_tests"]),
                failed_tests=int(parsed["failed_tests"]),
                skipped_tests=int(parsed["skipped_tests"]),
                total_tests=int(parsed["total_tests"]),
            )
        )
    total_tests = sum(item.total_tests for item in results)
    passed_tests = sum(item.passed_tests for item in results)
    failed_tests = sum(item.failed_tests for item in results)
    executed_tests = passed_tests + failed_tests
    payload = {
        "summary": {
            "total_families": len(results),
            "passed_families": sum(1 for item in results if item.status == "passed"),
            "failed_families": sum(1 for item in results if item.status != "passed"),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "percent": round((passed_tests / executed_tests) * 100.0, 2) if executed_tests else 0.0,
            "wall_clock_seconds": round(time.monotonic() - started, 3),
        },
        "results": [asdict(item) for item in results],
    }
    results_json = stable_json_dumps(payload, indent=2) + "\n"
    write_text(output_dir / "agent_behavior_cached_support_results.json", results_json, encoding="utf-8")
    report_lines = [
        "# Agent behavior tests (cached mode)",
        "",
        f"- Percent: `{payload['summary']['percent']:.2f}%`",
        f"- Passed families: `{payload['summary']['passed_families']}`",
        f"- Failed families: `{payload['summary']['failed_families']}`",
        f"- Passed tests: `{payload['summary']['passed_tests']}`",
        f"- Failed tests: `{payload['summary']['failed_tests']}`",
        "",
        "| Family | Score | Status | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        report_lines.append(
            f"| {result.family_id} | {result.score_percent:.2f}% | {result.status} | `{result.stdout_path}` |"
        )
    report_lines.extend(["", "## Family details", ""])
    for result in results:
        report_lines.extend(
            [
                f"### {result.family_id}",
                f"- Description: {next(item.description for item in families if item.family_id == result.family_id)}",
                f"- Score: `{result.score_percent:.2f}%`",
                f"- Status: `{result.status}`",
                f"- Tests: `{result.passed_tests}` passed / `{result.failed_tests}` failed / `{result.skipped_tests}` skipped / `{result.total_tests}` total",
                f"- Command: `{' '.join(result.command)}`",
                f"- Stdout: `{result.stdout_path}`",
                f"- Stderr: `{result.stderr_path}`",
                f"- JUnit: `{result.junit_path}`",
                "- Nodeids:",
            ]
        )
        report_lines.extend(f"  - `{nodeid}`" for nodeid in result.nodeids)
        report_lines.append("")
    report_text = "\n".join(report_lines) + "\n"
    write_text(output_dir / "agent_behavior_cached_support_report.md", report_text, encoding="utf-8")
    return payload
