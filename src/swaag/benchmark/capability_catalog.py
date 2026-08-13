from __future__ import annotations

import json
from pathlib import Path

from swaag.benchmark.task_definitions import BenchmarkTaskDefinition, BenchmarkVerificationContract, TaskScenario
from swaag.fsops import write_text
from swaag.types import Message


def _write(path: Path, text: str) -> Path:
    write_text(path, text, encoding="utf-8")
    return path


def _verification_script(workspace: Path, expected_file: str, expected_text: str) -> str:
    script = workspace / "verify_capability.py"
    script.write_text(
        "from pathlib import Path\n"
        f"actual = Path({expected_file!r}).read_text(encoding='utf-8')\n"
        f"expected = {expected_text!r}\n"
        "if actual != expected:\n"
        "    raise SystemExit(f'mismatch: {actual!r} != {expected!r}')\n",
        encoding="utf-8",
    )
    return script.name


def _history_retrieval(workspace: Path) -> TaskScenario:
    target = _write(workspace / "recovered_history.txt", "pending\n")
    marker = "amber-quartz-742"
    script = _verification_script(workspace, target.name, marker + "\n")
    filler = [
        Message(role="user" if index % 2 == 0 else "assistant", content=f"Unrelated historical note {index}: value-{index:02d}.", created_at=f"2026-01-01T00:{index:02d}:00+00:00")
        for index in range(1, 15)
    ]
    history = [
        Message(role="user", content=f"Store this exact recovery marker for later: {marker}.", created_at="2026-01-01T00:00:00+00:00"),
        Message(role="assistant", content=f"Recorded recovery marker {marker}.", created_at="2026-01-01T00:00:30+00:00"),
        *filler,
    ]
    return TaskScenario(
        prompt=(
            "Recover the exact recovery marker from durable conversation history using history_search with session_ref=null for the active session, then history_window with the exact returned session_id/sequence, "
            "then write only that marker plus a trailing newline to recovered_history.txt. Do not infer it from workspace files. "
            f"Run python3 {script} before answering."
        ),
        workspace=workspace,
        model_client=None,
        history_messages=history,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_files={str(target): marker + "\n"},
            command=["python3", script],
            command_cwd=str(workspace),
            required_history_events=["agent_action_selected", "history_retrieved", "history_window_read"],
            required_tools_used=["history_search", "history_window"],
            min_tool_calls=3,
            allowed_modified_files=[str(target)],
            forbid_unexpected_workspace_changes=True,
        ),
    )


def _large_output_artifact(workspace: Path) -> TaskScenario:
    target = _write(workspace / "large_output_marker.txt", "pending\n")
    marker = "deep-output-marker-91827"
    generator = workspace / "emit_large_output.py"
    generator.write_text(
        "print('A' * 6000)\n"
        f"print({marker!r})\n"
        "print('B' * 6000)\n",
        encoding="utf-8",
    )
    script = _verification_script(workspace, target.name, marker + "\n")
    return TaskScenario(
        prompt=(
            "Run python3 emit_large_output.py with shell_command. Its important marker is beyond the bounded immediate shell output, "
            "so use the returned output artifact and read_artifact to recover the exact marker. A read_artifact result with finished=false is incomplete; follow next_offset through additional slices until the marker is unambiguous. Write only that marker plus a trailing newline "
            f"to large_output_marker.txt, then run python3 {script}."
        ),
        workspace=workspace,
        model_client=None,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_files={str(target): marker + "\n"},
            command=["python3", script],
            command_cwd=str(workspace),
            required_history_events=["agent_action_selected", "artifact_created", "artifact_read"],
            required_tools_used=["shell_command", "read_artifact"],
            min_tool_calls=3,
            allowed_modified_files=[str(target)],
            forbid_unexpected_workspace_changes=True,
        ),
    )


def _persistent_terminal(workspace: Path) -> TaskScenario:
    target = _write(workspace / "terminal_result.txt", "pending\n")
    expected = "state=violet\nstdin=hello-pty\n"
    script = _verification_script(workspace, target.name, expected)
    return TaskScenario(
        prompt=(
            "Use the persistent terminal tool, not shell_command, for this task. Create one terminal, set shell variable COLOR=violet in it, "
            "then in a later terminal send prove the variable persisted. In the same terminal run an interactive Python input() program and send "
            "hello-pty as later stdin. Use the observed terminal output to write terminal_result.txt exactly as two lines: state=violet and stdin=hello-pty, with a trailing newline after the second line. "
            f"Close the terminal and run python3 {script} before answering."
        ),
        workspace=workspace,
        model_client=None,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_files={str(target): expected},
            command=["python3", script],
            command_cwd=str(workspace),
            required_history_events=["agent_action_selected", "terminal_create", "terminal_send", "terminal_read", "terminal_close"],
            required_tools_used=["terminal"],
            min_tool_calls=6,
            allowed_modified_files=[str(target)],
            forbid_unexpected_workspace_changes=True,
        ),
    )


def _human_duration_wait(workspace: Path) -> TaskScenario:
    target = _write(workspace / "wait_result.txt", "pending\n")
    expected = "waited=25 ms\n"
    script = _verification_script(workspace, target.name, expected)
    return TaskScenario(
        prompt=(
            "Use wait_seconds with the human-readable duration '25 ms'. After it returns, write exactly waited=25 ms plus a trailing newline "
            f"to wait_result.txt and run python3 {script} before answering."
        ),
        workspace=workspace,
        model_client=None,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_files={str(target): expected},
            command=["python3", script],
            command_cwd=str(workspace),
            required_history_events=["agent_action_selected", "wait_completed"],
            required_tools_used=["wait_seconds"],
            min_tool_calls=2,
            allowed_modified_files=[str(target)],
            forbid_unexpected_workspace_changes=True,
        ),
    )



def _sqlite_fts_history(workspace: Path) -> TaskScenario:
    target = _write(workspace / "sqlite_history_result.txt", "pending\n")
    marker = "cobalt-history-fts-531"
    expected = f"backend=sqlite_fts5\nmarker={marker}\n"
    script = _verification_script(workspace, target.name, expected)
    history = [
        Message(role="user", content="Older unrelated note about project delta.", created_at="2026-01-01T00:00:00+00:00"),
        Message(role="assistant", content=f"Durable indexed-history marker: {marker}.", created_at="2026-01-01T00:01:00+00:00"),
        Message(role="user", content="Much later unrelated note about deployment checks.", created_at="2026-01-01T00:02:00+00:00"),
    ]
    return TaskScenario(
        prompt=(
            "Use history_search with session_ref=null for the active session to recover the exact durable indexed-history marker from prior conversation history. "
            "Also inspect the history_search tool result and use its reported search_backend value. Write sqlite_history_result.txt exactly as two lines: "
            "backend=<reported backend> and marker=<exact recovered marker>, each with a trailing newline. "
            f"Run python3 {script} before answering."
        ),
        workspace=workspace,
        model_client=None,
        history_messages=history,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_files={str(target): expected},
            command=["python3", script],
            command_cwd=str(workspace),
            required_history_events=["agent_action_selected", "history_retrieved"],
            required_tools_used=["history_search"],
            min_tool_calls=2,
            allowed_modified_files=[str(target)],
            forbid_unexpected_workspace_changes=True,
        ),
    )


def _structured_status(workspace: Path) -> TaskScenario:
    target = _write(workspace / "status_result.txt", "pending\n")
    expected = "status-persisted\n"
    script = _verification_script(workspace, target.name, expected)
    return TaskScenario(
        prompt=(
            "Write exactly status-persisted plus a trailing newline to status_result.txt, then run "
            f"python3 {script}. Complete the task normally; the runtime must durably persist the structured status that accompanies your action."
        ),
        workspace=workspace,
        model_client=None,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_files={str(target): expected},
            command=["python3", script],
            command_cwd=str(workspace),
            required_history_events=["agent_action_selected", "agent_status"],
            min_tool_calls=2,
            allowed_modified_files=[str(target)],
            forbid_unexpected_workspace_changes=True,
        ),
    )


def _grounded_history_analysis(workspace: Path) -> TaskScenario:
    target = _write(workspace / "analysis_result.txt", "pending\n")
    marker = "root-cause-cache-endpoint-417"
    script = _verification_script(workspace, target.name, marker + "\n")
    history = [
        Message(role="user", content="The repeated failure started after changing the model endpoint.", created_at="2026-01-01T00:00:00+00:00"),
        Message(role="assistant", content=f"Root-cause evidence marker recorded for diagnosis: {marker}.", created_at="2026-01-01T00:01:00+00:00"),
        Message(role="user", content="The old strategy kept retrying without inspecting durable history.", created_at="2026-01-01T00:02:00+00:00"),
    ]
    return TaskScenario(
        prompt=(
            "Use history_analyze with session_ref=null for the active session to diagnose why the repeated failure was mishandled. Then use the analyzer's exact returned session_id/source_sequences with history_window, "
            "to inspect the analyzer's supporting source evidence and recover the exact root-cause evidence marker. Write only that exact marker plus a trailing newline "
            f"to analysis_result.txt and run python3 {script} before answering."
        ),
        workspace=workspace,
        model_client=None,
        history_messages=history,
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            expected_files={str(target): marker + "\n"},
            command=["python3", script],
            command_cwd=str(workspace),
            required_history_events=["agent_action_selected", "history_analyzed", "history_window_read"],
            required_tools_used=["history_analyze", "history_window"],
            min_tool_calls=3,
            allowed_modified_files=[str(target)],
            forbid_unexpected_workspace_changes=True,
        ),
    )

def capability_benchmark_tasks() -> list[BenchmarkTaskDefinition]:
    common = {
        "task_type": "multi_step",
        "difficulty": "hard",
        "setup_instructions": [
            "Create a deterministic capability fixture with no model-response fixture.",
            "Require the capability through observable tool/history evidence and a deterministic output verifier.",
            "Do not add capability-specific logic to the runtime.",
        ],
        "config_overrides": {
            "tools_allow_side_effect_tools": True,
            "runtime_max_total_actions": 18,
            "runtime_tool_call_budget": 20,
        },
    }
    return [
        BenchmarkTaskDefinition(
            task_id="capability_history_exact_retrieval",
            description="Recover an exact older conversation detail through the model-facing history retrieval tools.",
            build=_history_retrieval,
            build_live=_history_retrieval,
            tags=["multi-step", "history", "retrieval", "authoritative-source"],
            **common,
        ),
        BenchmarkTaskDefinition(
            task_id="capability_large_output_artifact_recovery",
            description="Recover an exact marker from durable raw shell output after immediate output is mechanically bounded.",
            build=_large_output_artifact,
            build_live=_large_output_artifact,
            tags=["multi-step", "shell", "large-output", "artifact", "environment"],
            config_overrides={**common["config_overrides"], "environment_max_capture_chars": 512},
            **{k: v for k, v in common.items() if k != "config_overrides"},
        ),
        BenchmarkTaskDefinition(
            task_id="capability_persistent_interactive_terminal",
            description="Use one persistent PTY terminal across calls and provide later stdin to an interactive child process.",
            build=_persistent_terminal,
            build_live=_persistent_terminal,
            tags=["multi-step", "terminal", "interactive", "environment", "long-run"],
            **common,
        ),
        BenchmarkTaskDefinition(
            task_id="capability_human_duration_wait",
            description="Use a human-readable subsecond duration through the shared duration parser.",
            build=_human_duration_wait,
            build_live=_human_duration_wait,
            tags=["multi-step", "wait", "duration", "scheduler"],
            **common,
        ),
        BenchmarkTaskDefinition(
            task_id="capability_sqlite_fts_history",
            description="Recover exact durable history through SQLite FTS5 and use the reported indexed-search backend.",
            build=_sqlite_fts_history,
            build_live=_sqlite_fts_history,
            tags=["multi-step", "history", "sqlite", "fts5", "durability"],
            **common,
        ),
        BenchmarkTaskDefinition(
            task_id="capability_structured_status",
            description="Persist the structured situation/action/reason/importance status accompanying an agent action.",
            build=_structured_status,
            build_live=_structured_status,
            tags=["multi-step", "status", "observability", "history"],
            **common,
        ),
        BenchmarkTaskDefinition(
            task_id="capability_grounded_history_analysis",
            description="Use model-backed root-cause analysis grounded in exact durable history sequence evidence.",
            build=_grounded_history_analysis,
            build_live=_grounded_history_analysis,
            tags=["multi-step", "history", "root-cause", "analysis", "grounding"],
            **common,
        ),
    ]
