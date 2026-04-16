from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from swaag.cli import main
from swaag.context_builder import build_context
from swaag.history import HistoryStore
from swaag.project_state import build_project_state
from swaag.runtime import AgentRuntime, TurnResult
from swaag.tokens import ConservativeEstimator
from swaag.types import DeferredTask, Plan, PlanStep
from tests.helpers import FakeModelClient, plan_response, plan_step


def test_runtime_recovered_answer_is_not_blocked_by_stale_failed_turn_state(make_config) -> None:
    failing_config = make_config(runtime__tool_call_budget=0, planner__max_replans=0)
    success_config = make_config(planner__max_replans=0)
    goal = "Use the calculator tool to compute 2 + 2."

    failing_runtime = AgentRuntime(
        failing_config,
        model_client=FakeModelClient(
            responses=[
                plan_response(
                    goal=goal,
                    steps=[
                        plan_step("step_calc", "Compute", "tool", expected_tool="calculator", expected_output="value", success_criteria="calculator returns a value"),
                        plan_step("step_answer", "Answer", "respond", expected_output="answer", success_criteria="answer returned", depends_on=["step_calc"]),
                    ],
                )
            ]
        ),
    )
    first = failing_runtime.run_turn(goal)
    assert first.assistant_text == "not done"

    success_runtime = AgentRuntime(
        success_config,
        history_store=HistoryStore(failing_config.sessions.root),
        model_client=FakeModelClient(
            contract_responses={
                "task_decision": [
                    json.dumps(
                        {
                            "split_task": False,
                            "expand_task": False,
                            "ask_user": False,
                            "assume_missing": False,
                            "generate_ideas": False,
                            "direct_response": False,
                            "execution_mode": "single_tool",
                            "preferred_tool_name": "calculator",
                            "confidence": 1.0,
                            "reason": "one calculator call is sufficient",
                        }
                    )
                ]
            },
            responses=[json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "2 + 2"}})],
        ),
    )
    second = success_runtime.run_turn(goal, session_id=first.session_id)

    assert second.assistant_text == "4"


def test_runtime_processes_continue_control_without_stopping_current_work(make_config) -> None:
    config = make_config(planner__max_replans=0)
    no_spawn = json.dumps({"spawn": False, "subagent_type": "none", "reason": "not needed", "focus": ""})
    runtime = AgentRuntime(
        config,
        model_client=FakeModelClient(
            contract_responses={
                "task_decision": [
                    json.dumps(
                        {
                            "split_task": False,
                            "expand_task": False,
                            "ask_user": False,
                            "assume_missing": False,
                            "generate_ideas": False,
                            "direct_response": False,
                            "execution_mode": "single_tool",
                            "preferred_tool_name": "calculator",
                            "confidence": 1.0,
                            "reason": "one calculator call is sufficient",
                        }
                    )
                ],
                "active_session_control": [
                    json.dumps(
                        {
                            "action": "continue_with_note",
                            "reason": "extra formatting constraint",
                            "response_text": "",
                            "added_context": "Return plain digits only.",
                            "replacement_goal": "",
                            "queued_task": "",
                            "clarification_question": "",
                        }
                    )
                ],
                "subagent_selection": [no_spawn] * 4,
            },
            responses=[None],
        ),
    )
    state = runtime.create_or_load_user_session("control-test")

    def _tool_decision(_payload=None, **_kwargs):
        runtime.history.enqueue_control_message(state.session_id, "Also make the answer digits only.", source="test")
        return json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "6 * 7"}})

    runtime.client._responses = [_tool_decision]
    result = runtime.run_turn_in_session(state, "Use the calculator tool to compute 6 * 7.")
    rebuilt = runtime.history.rebuild_from_history(state.session_id, write_projections=False)
    events = runtime.history.read_history(state.session_id)

    assert result.assistant_text == "42"
    assert any(note.content == "Return plain digits only." for note in rebuilt.notes)
    assert any(event.event_type == "control_message_processed" for event in events)
    assert any(event.event_type == "control_action_applied" for event in events)


def test_process_pending_control_messages_can_request_replacement(make_config, monkeypatch) -> None:
    runtime = AgentRuntime(make_config())
    state = runtime.create_or_load_user_session("replace-me")
    runtime.history.enqueue_control_message(state.session_id, "Replace this task with compute 3 + 3.", source="test")

    monkeypatch.setattr(
        runtime,
        "_classify_control_message_frontend",
        lambda *args, **kwargs: {
            "action": "replace_task",
            "reason": "explicit replacement",
            "response_text": "",
            "added_context": "",
            "replacement_goal": "Use the calculator tool to compute 3 + 3.",
            "queued_task": "",
            "clarification_question": "",
        },
    )

    result = runtime._process_pending_control_messages(state, effective_goal="old goal")
    events = runtime.history.read_history(state.session_id)

    assert result.replacement_goal == "Use the calculator tool to compute 3 + 3."
    assert any(event.event_type == "control_message_processed" for event in events)
    assert any(event.event_type == "control_action_applied" for event in events)
    assert runtime.history.list_pending_control_messages(state.session_id) == []


@pytest.mark.parametrize(
    ("action", "field", "expected_text"),
    [
        ("status", "response_text", "Still running step 1."),
        ("session_summary", "response_text", "Current session summary."),
        ("clarify_conflict", "clarification_question", "Should I replace the current task?"),
    ],
)
def test_process_pending_control_messages_returns_non_destructive_control_responses(
    make_config,
    monkeypatch,
    action: str,
    field: str,
    expected_text: str,
) -> None:
    runtime = AgentRuntime(make_config())
    state = runtime.create_or_load_user_session(f"control-{action}")
    runtime.history.enqueue_control_message(state.session_id, f"{action} request", source="test")
    monkeypatch.setattr(
        runtime,
        "_classify_control_message_frontend",
        lambda *args, **kwargs: {
            "action": action,
            "reason": f"{action} requested",
            "response_text": expected_text if field == "response_text" else "",
            "added_context": "",
            "replacement_goal": "",
            "queued_task": "",
            "clarification_question": expected_text if field == "clarification_question" else "",
        },
    )

    result = runtime._process_pending_control_messages(state, effective_goal="goal")

    assert result.assistant_messages == [expected_text]
    assert not result.stop_requested
    assert not result.replan_requested
    assert result.replacement_goal is None
    assert runtime.history.list_pending_control_messages(state.session_id) == []
    processed_files = list(runtime.history.control_processed_dir(state.session_id).glob("*.json"))
    assert len(processed_files) == 1


@pytest.mark.parametrize("action", ["stop", "cancel"])
def test_process_pending_control_messages_stops_only_on_explicit_stop_actions(make_config, monkeypatch, action: str) -> None:
    runtime = AgentRuntime(make_config())
    state = runtime.create_or_load_user_session(f"stop-{action}")
    runtime.history.enqueue_control_message(state.session_id, f"{action} now", source="test")
    monkeypatch.setattr(
        runtime,
        "_classify_control_message_frontend",
        lambda *args, **kwargs: {
            "action": action,
            "reason": f"{action} requested",
            "response_text": "stopping now",
            "added_context": "",
            "replacement_goal": "",
            "queued_task": "",
            "clarification_question": "",
        },
    )

    result = runtime._process_pending_control_messages(state, effective_goal="goal")

    assert result.stop_requested
    assert result.assistant_messages == ["stopping now"]


def test_process_pending_control_messages_queues_follow_up_task(make_config, monkeypatch) -> None:
    runtime = AgentRuntime(make_config())
    state = runtime.create_or_load_user_session("queue-task")
    runtime.history.enqueue_control_message(state.session_id, "After this, write a summary.", source="test")
    monkeypatch.setattr(
        runtime,
        "_classify_control_message_frontend",
        lambda *args, **kwargs: {
            "action": "queue_after_current",
            "reason": "follow-up task",
            "response_text": "queued for later",
            "added_context": "",
            "replacement_goal": "",
            "queued_task": "Write a summary after the current task.",
            "clarification_question": "",
        },
    )

    result = runtime._process_pending_control_messages(state, effective_goal="goal")
    rebuilt = runtime.history.rebuild_from_history(state.session_id, write_projections=False)

    assert result.assistant_messages == ["queued for later"]
    assert not result.stop_requested
    assert rebuilt.deferred_tasks
    assert rebuilt.deferred_tasks[0].text == "Write a summary after the current task."


def test_continue_with_note_adds_note_without_forcing_replan(make_config, monkeypatch) -> None:
    runtime = AgentRuntime(make_config())
    state = runtime.create_or_load_user_session("continue-note")
    runtime.history.enqueue_control_message(state.session_id, "Also keep the output terse.", source="test")
    monkeypatch.setattr(
        runtime,
        "_classify_control_message_frontend",
        lambda *args, **kwargs: {
            "action": "continue_with_note",
            "reason": "add constraint",
            "response_text": "constraint saved",
            "added_context": "Keep the output terse.",
            "replacement_goal": "",
            "queued_task": "",
            "clarification_question": "",
        },
    )

    result = runtime._process_pending_control_messages(state, effective_goal="goal")
    rebuilt = runtime.history.rebuild_from_history(state.session_id, write_projections=False)

    assert result.assistant_messages == ["constraint saved"]
    assert not result.stop_requested
    assert not result.replan_requested
    assert any(note.content == "Keep the output terse." for note in rebuilt.notes)


def test_history_detail_query_returns_exact_old_command(make_config) -> None:
    runtime = AgentRuntime(make_config())
    state = runtime.create_or_load_user_session("history-query")
    runtime.history.record_event(
        state,
        "shell_command_completed",
        {
            "command": "cp src.txt dst.txt",
            "cwd_before": "/tmp/work",
            "cwd_after": "/tmp/work",
            "env_overrides": {},
            "unset_vars": [],
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        },
    )

    result = runtime.query_history_details(
        session_ref="history-query",
        query_text="What exact command copied src.txt to dst.txt?",
        topic_hint="copy command",
    )

    assert result["matches"]
    assert result["matches"][0]["event_type"] == "shell_command_completed"
    assert "cp src.txt dst.txt" in json.dumps(result["matches"][0]["payload"])


def test_history_detail_query_handles_quoted_exact_terms(make_config) -> None:
    runtime = AgentRuntime(make_config())
    state = runtime.create_or_load_user_session("quoted-query")
    runtime.history.record_event(
        state,
        "file_write_applied",
        {
            "path": "logs/build-output.txt",
            "source": "test",
            "line_count": 1,
            "size_chars": 17,
            "backup_path": "",
            "cause_event": "test_setup",
        },
    )

    result = runtime.query_history_details(
        session_ref="quoted-query",
        query_text='Where did you write "logs/build-output.txt"?',
    )

    assert result["matches"]
    assert "logs/build-output.txt" in json.dumps(result["matches"][0]["payload"])


def test_history_detail_query_defaults_to_latest_session_and_uses_topic_hint(make_config) -> None:
    runtime = AgentRuntime(make_config())
    older = runtime.create_or_load_user_session("older-query")
    runtime.history.record_event(
        older,
        "shell_command_completed",
        {
            "command": "echo older",
            "cwd_before": "/tmp/work",
            "cwd_after": "/tmp/work",
            "env_overrides": {},
            "unset_vars": [],
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    latest = runtime.create_or_load_user_session("latest-query")
    runtime.history.record_event(
        latest,
        "shell_command_completed",
        {
            "command": "pytest -q tests/test_runtime.py",
            "cwd_before": "/tmp/work",
            "cwd_after": "/tmp/work",
            "env_overrides": {},
            "unset_vars": [],
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        },
    )

    result = runtime.query_history_details(
        session_ref=None,
        query_text="What exact command did you run?",
        topic_hint="pytest runtime test command",
    )

    assert result["session_name"] == "latest-query"
    assert result["matches"]
    assert "pytest -q tests/test_runtime.py" in json.dumps(result["matches"][0]["payload"])


def test_code_checkpoint_create_and_restore(make_config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("one", encoding="utf-8")
    (workspace / "b.txt").write_text("two", encoding="utf-8")
    config = make_config(tools__read_roots=[workspace])
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_user_session("checkpoints")

    checkpoint = runtime.create_code_checkpoint(state, label="before-edit")
    (workspace / "a.txt").write_text("changed", encoding="utf-8")
    (workspace / "c.txt").write_text("extra", encoding="utf-8")

    restored = runtime.restore_code_checkpoint(state, checkpoint_ref=checkpoint["checkpoint_id"])
    rebuilt = runtime.history.rebuild_from_history(state.session_id, write_projections=False)

    assert restored["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "one"
    assert (workspace / "b.txt").read_text(encoding="utf-8") == "two"
    assert not (workspace / "c.txt").exists()
    assert rebuilt.code_checkpoints


def test_restore_code_checkpoint_fails_cleanly_for_missing_checkpoint(make_config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("one", encoding="utf-8")
    config = make_config(tools__read_roots=[workspace])
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_user_session("checkpoint-errors")

    with pytest.raises(RuntimeError, match="No code checkpoints are available"):
        runtime.restore_code_checkpoint(state)

    runtime.create_code_checkpoint(state, label="before-edit")
    with pytest.raises(FileNotFoundError, match="Unknown checkpoint: missing"):
        runtime.restore_code_checkpoint(state, checkpoint_ref="missing")


def test_cli_ask_without_prompt_uses_deferred_task(monkeypatch, capsys) -> None:
    class _FakeRuntime:
        def __init__(self):
            self.used_prompt = ""
            self.state = type(
                "State",
                (),
                {
                    "session_id": "session_1",
                    "session_name": "queued",
                    "deferred_tasks": [DeferredTask(task_id="task_1", text="Say hello.", queued_at="t0")],
                },
            )()

        def create_or_load_user_session(self, _session_ref=None):
            return self.state

        def pop_next_deferred_task(self, _state, *, reason):
            assert reason == "ask_without_prompt"
            return self.state.deferred_tasks[0]

        def run_turn_in_session(self, _state, prompt):
            self.used_prompt = prompt
            return TurnResult(session_id="session_1", assistant_text="hello", tool_results=[], budget_reports=[])

    fake_runtime = _FakeRuntime()
    monkeypatch.setattr("swaag.cli.AgentRuntime", lambda _config: fake_runtime)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    rc = main(["ask"])
    captured = capsys.readouterr()

    assert rc == 0
    assert fake_runtime.used_prompt == "Say hello."
    assert "hello" in captured.out


def test_cli_sessions_list_and_rename_use_human_names(tmp_path: Path, capsys, monkeypatch) -> None:
    sessions_root = tmp_path / "sessions"
    monkeypatch.setenv("SWAAG__SESSIONS__ROOT", str(sessions_root))
    monkeypatch.setenv("SWAAG__TOOLS__READ_ROOTS", json.dumps([str(tmp_path)]))
    monkeypatch.setenv("SWAAG__MODEL__BASE_URL", "http://127.0.0.1:9999")
    store = HistoryStore(sessions_root)
    state = store.create(config_fingerprint="cfg", model_base_url="http://example.test", session_name="alpha", session_name_source="explicit")

    assert main(["sessions"]) == 0
    listed = capsys.readouterr().out
    assert "alpha" in listed

    assert main(["rename", state.session_id, "beta"]) == 0
    renamed = capsys.readouterr().out
    assert "beta" in renamed

    assert main(["sessions"]) == 0
    relisted = capsys.readouterr().out
    assert "beta" in relisted


def test_project_state_tracks_expected_pending_and_completed_artifacts(make_config) -> None:
    runtime = AgentRuntime(make_config())
    state = runtime.create_or_load_user_session("artifacts")
    state.active_plan = Plan(
        plan_id="plan_1",
        goal="create artifact outputs",
        steps=[
            PlanStep(
                step_id="step_build",
                title="Build report",
                kind="write",
                goal="write report",
                expected_tool="",
                input_text="",
                expected_output="report file",
                done_condition="report exists",
                success_criteria="report saved",
                status="completed",
                output_refs=["report.md"],
            ),
            PlanStep(
                step_id="step_test",
                title="Run tests",
                kind="tool",
                goal="run tests",
                expected_tool="shell_command",
                input_text="pytest -q",
                expected_output="test log",
                done_condition="tests finish",
                success_criteria="pytest log captured",
                status="running",
                expected_outputs=["pytest.log"],
            ),
        ],
        success_criteria="all artifacts produced",
        fallback_strategy="replan",
        status="running",
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        current_step_id="step_test",
    )

    project_state = build_project_state(state)

    assert "step_build:report.md" in project_state.expected_artifacts
    assert "step_build:report.md" in project_state.completed_artifacts
    assert "step_test:pytest.log" in project_state.expected_artifacts
    assert "step_test:pytest.log" in project_state.pending_artifacts


def test_artifact_state_is_visible_in_built_context(make_config) -> None:
    config = make_config()
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_user_session("artifact-context")
    state.active_plan = Plan(
        plan_id="plan_2",
        goal="create artifact outputs",
        steps=[
            PlanStep(
                step_id="step_write",
                title="Write report",
                kind="write",
                goal="write report",
                expected_tool="",
                input_text="",
                expected_output="report file",
                done_condition="report exists",
                success_criteria="report saved",
                status="completed",
                output_refs=["report.md"],
            ),
            PlanStep(
                step_id="step_verify",
                title="Verify report",
                kind="tool",
                goal="verify report",
                expected_tool="shell_command",
                input_text="test -f report.md",
                expected_output="verification result",
                done_condition="verification finished",
                success_criteria="verification succeeds",
                status="running",
                expected_outputs=["verify.log"],
            ),
        ],
        success_criteria="artifacts verified",
        fallback_strategy="replan",
        status="running",
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        current_step_id="step_verify",
    )
    runtime._refresh_project_state(state, reason="artifact_context_test")

    bundle = build_context(config, state, ConservativeEstimator(), goal="finish the artifact flow", call_kind="analysis")

    assert "Expected artifacts:" in bundle.project_state_text
    assert "step_write:report.md" in bundle.project_state_text
    assert "step_verify:verify.log" in bundle.project_state_text
