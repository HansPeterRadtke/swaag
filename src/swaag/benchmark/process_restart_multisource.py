from __future__ import annotations

import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from swaag.config import AgentConfig
from swaag.runtime import AgentRuntime
from swaag.types import Message
from swaag.utils import sha256_text, stable_json_dumps, utc_now_iso

USER_FACT = "constraint-user-C4-55109"
TOOL_FACT = "tool-result-T8-66317"
ATTACHMENT_FACT = "attachment-value-A2-77403"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_process_restart_multisource_benchmark(
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
    trial_config = deepcopy(config)
    trial_config.sessions.root = output_dir / "sessions"
    runtime = AgentRuntime(trial_config, model_client=model_client)
    state = runtime.create_or_load_session()
    runtime._record_message(
        state,
        Message(
            role="user",
            content=f"Authoritative user constraint for later retrieval: {USER_FACT}",
            created_at=utc_now_iso(),
        ),
    )
    tool_event = runtime.history.record_event(
        state,
        "tool_result",
        {
            "tool_name": "multisource_probe",
            "raw_input": {"source": "early"},
            "validated_input": {"source": "early"},
            "output": {"code": TOOL_FACT},
        },
    )
    runtime._record_message(
        state,
        Message(
            role="tool",
            name="multisource_probe",
            content=f"Authoritative early tool result: {TOOL_FACT}",
            created_at=utc_now_iso(),
            metadata={
                "source_event_sequence": tool_event.sequence,
                "source_event_hash": tool_event.hash,
                "source_event_type": tool_event.event_type,
                "source_event_references": [],
            },
        ),
    )
    attachment = runtime.add_attachment(
        f"Authoritative attachment record. attachment_value={ATTACHMENT_FACT}\n".encode(),
        original_name="multisource-evidence.txt",
        media_type="text/plain",
        source="process_restart_multisource_benchmark",
        session_id=state.session_id,
    )
    state = runtime.create_or_load_session(state.session_id)
    for index in range(16):
        runtime._record_message(
            state,
            Message(
                role="assistant" if index % 2 else "user",
                content=f"Long unrelated subprocess phase-one progress {index + 1}: housekeeping only.",
                created_at=utc_now_iso(),
            ),
        )
    first_compaction = runtime._compact_once(state)
    context_limit, context_limit_source = runtime._resolve_context_limit()
    parent_pid = os.getpid()
    child_output = output_dir / "child_report.json"
    stdout_path = output_dir / "child_stdout.txt"
    stderr_path = output_dir / "child_stderr.txt"
    command = [
        sys.executable,
        "-m",
        "swaag.benchmark.process_restart_multisource_probe",
        "--sessions-root",
        str(trial_config.sessions.root),
        "--session-id",
        state.session_id,
        "--user-fact",
        USER_FACT,
        "--tool-fact",
        TOOL_FACT,
        "--attachment-fact",
        ATTACHMENT_FACT,
        "--attachment-id",
        attachment.attachment_id,
        "--model-base-url",
        trial_config.model.base_url,
        "--context-limit",
        str(context_limit),
        "--output",
        str(child_output),
    ]
    env = dict(os.environ)
    env["SWAAG__MODEL__CACHE_ENABLED"] = "false"
    completed = subprocess_runner(
        command,
        cwd=str(Path.cwd()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    _write_text(stdout_path, stdout)
    _write_text(stderr_path, stderr)
    child_report: dict[str, Any] | None = None
    if child_output.exists():
        import json

        child_report = json.loads(child_output.read_text(encoding="utf-8"))
    different_pid = bool(child_report and int(child_report.get("child_pid", parent_pid)) != parent_pid)
    passed = bool(
        first_compaction
        and completed.returncode == 0
        and child_report
        and child_report.get("passed")
        and different_pid
    )
    report = {
        "passed": passed,
        "session_id": state.session_id,
        "first_compaction": first_compaction,
        "parent_pid": parent_pid,
        "child_pid": None if child_report is None else child_report.get("child_pid"),
        "different_process": different_pid,
        "subprocess_returncode": completed.returncode,
        "attachment_id": attachment.attachment_id,
        "attachment_sha256": attachment.sha256,
        "command": command,
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_text(stdout),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_text(stderr),
        "child_report": child_report,
        "parent_model_identity": getattr(runtime.client, "cache_identity", lambda: type(runtime.client).__name__)(),
        "context_limit": context_limit,
        "context_limit_source": context_limit_source,
    }
    report_path = output_dir / "process_restart_multisource.json"
    tmp = report_path.with_name(report_path.name + ".tmp")
    tmp.write_text(stable_json_dumps(report, indent=2) + "\n", encoding="utf-8")
    tmp.replace(report_path)
    return report
