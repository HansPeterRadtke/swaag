from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from swaag.benchmark.process_restart_multisource import ATTACHMENT_FACT, TOOL_FACT, USER_FACT
from swaag.config import AgentConfig
from swaag.runtime import AgentRuntime
from swaag.types import Message
from swaag.utils import sha256_text, stable_json_dumps, utc_now_iso


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _spawn(
    *,
    phase: str,
    query: bool,
    output_dir: Path,
    sessions_root: Path,
    session_id: str,
    attachment_id: str,
    model_base_url: str,
    context_limit: int,
    runner: Any,
) -> tuple[subprocess.CompletedProcess, dict[str, Any] | None, list[str]]:
    child_output = output_dir / f"{phase}_report.json"
    command = [
        sys.executable,
        "-m",
        "swaag.benchmark.process_restart_multisource_probe",
        "--sessions-root", str(sessions_root),
        "--session-id", session_id,
        "--user-fact", USER_FACT,
        "--tool-fact", TOOL_FACT,
        "--attachment-fact", ATTACHMENT_FACT,
        "--attachment-id", attachment_id,
        "--model-base-url", model_base_url,
        "--context-limit", str(context_limit),
        "--phase-label", phase,
        "--output", str(child_output),
    ]
    if not query:
        command.append("--skip-query")
    env = dict(os.environ)
    env["SWAAG__MODEL__CACHE_ENABLED"] = "false"
    completed = runner(command, cwd=str(Path.cwd()), env=env, text=True, capture_output=True, check=False)
    _write(output_dir / f"{phase}_stdout.txt", completed.stdout or "")
    _write(output_dir / f"{phase}_stderr.txt", completed.stderr or "")
    report = json.loads(child_output.read_text()) if child_output.exists() else None
    return completed, report, command


def run_process_restart_chain_benchmark(
    *,
    output_dir: Path,
    config: AgentConfig,
    model_client: object | None = None,
    subprocess_runner: Any = subprocess.run,
    clean: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial = deepcopy(config)
    trial.sessions.root = output_dir / "sessions"
    runtime = AgentRuntime(trial, model_client=model_client)
    state = runtime.create_or_load_session()
    runtime._record_message(state, Message(role="user", content=f"Authoritative user constraint: {USER_FACT}", created_at=utc_now_iso()))
    tool_event = runtime.history.record_event(state, "tool_result", {
        "tool_name":"chain_probe","raw_input":{"source":"early"},"validated_input":{"source":"early"},"output":{"code":TOOL_FACT}
    })
    runtime._record_message(state, Message(role="tool", name="chain_probe", content=f"Authoritative tool result: {TOOL_FACT}", created_at=utc_now_iso(), metadata={
        "source_event_sequence":tool_event.sequence,"source_event_hash":tool_event.hash,"source_event_type":tool_event.event_type,"source_event_references":[]
    }))
    attachment = runtime.add_attachment(
        f"Authoritative attachment. attachment_value={ATTACHMENT_FACT}\n".encode(),
        original_name="chain-evidence.txt", media_type="text/plain", source="process_restart_chain_benchmark", session_id=state.session_id,
    )
    state = runtime.create_or_load_session(state.session_id)
    for index in range(16):
        runtime._record_message(state, Message(role="assistant" if index % 2 else "user", content=f"Long unrelated parent progress {index+1}: housekeeping only.", created_at=utc_now_iso()))
    first_compaction = runtime._compact_once(state)
    context_limit, context_source = runtime._resolve_context_limit()
    parent_pid = os.getpid()

    c1, r1, cmd1 = _spawn(
        phase="phase-two", query=False, output_dir=output_dir, sessions_root=trial.sessions.root,
        session_id=state.session_id, attachment_id=attachment.attachment_id, model_base_url=trial.model.base_url,
        context_limit=context_limit, runner=subprocess_runner,
    )
    c2, r2, cmd2 = _spawn(
        phase="phase-three", query=True, output_dir=output_dir, sessions_root=trial.sessions.root,
        session_id=state.session_id, attachment_id=attachment.attachment_id, model_base_url=trial.model.base_url,
        context_limit=context_limit, runner=subprocess_runner,
    )
    pids = [parent_pid] + [int(r["child_pid"]) for r in (r1, r2) if r and r.get("child_pid") is not None]
    distinct_processes = len(pids) == 3 and len(set(pids)) == 3
    passed = bool(
        first_compaction and c1.returncode == 0 and c2.returncode == 0 and r1 and r2
        and r1.get("passed") and r2.get("passed") and r1.get("query_performed") is False
        and r2.get("query_performed") is True and r2.get("exact_delayed_retrieval") is True
        and distinct_processes
    )
    report = {
        "passed": passed,
        "session_id": state.session_id,
        "first_compaction": first_compaction,
        "parent_pid": parent_pid,
        "phase_two_pid": None if not r1 else r1.get("child_pid"),
        "phase_three_pid": None if not r2 else r2.get("child_pid"),
        "distinct_processes": distinct_processes,
        "phase_two_returncode": c1.returncode,
        "phase_three_returncode": c2.returncode,
        "phase_two_report": r1,
        "phase_three_report": r2,
        "phase_two_command": cmd1,
        "phase_three_command": cmd2,
        "context_limit": context_limit,
        "context_limit_source": context_source,
        "parent_model_identity": getattr(runtime.client, "cache_identity", lambda: type(runtime.client).__name__)(),
        "phase_two_stdout_sha256": sha256_text(c1.stdout or ""),
        "phase_three_stdout_sha256": sha256_text(c2.stdout or ""),
    }
    path = output_dir / "process_restart_chain.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(stable_json_dumps(report, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return report
