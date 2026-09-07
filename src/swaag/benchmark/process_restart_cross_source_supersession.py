from __future__ import annotations

import json, os, shutil, subprocess, sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from swaag.config import AgentConfig
from swaag.runtime import AgentRuntime
from swaag.types import Message
from swaag.utils import sha256_text, stable_json_dumps, utc_now_iso

OLD_VALUE = "tool-owner-historical-T4-310"
NEW_VALUE = "user-owner-current-U9-720"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_process_restart_cross_source_supersession_benchmark(
    *, output_dir: Path, config: AgentConfig, model_client: object | None = None,
    subprocess_runner: Any = subprocess.run, clean: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if clean and output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial = deepcopy(config); trial.sessions.root = output_dir / "sessions"
    runtime = AgentRuntime(trial, model_client=model_client)
    state = runtime.create_or_load_session()
    tool_event = runtime.history.record_event(state, "tool_result", {
        "tool_name":"ownership_lookup","raw_input":{"release":"Falcon"},
        "validated_input":{"release":"Falcon"},"output":{"backup_owner":OLD_VALUE},
    })
    runtime._record_message(state, Message(
        role="tool", name="ownership_lookup",
        content=f"Authoritative tool result for Falcon backup owner: {OLD_VALUE}",
        created_at=utc_now_iso(), metadata={
            "source_event_sequence":tool_event.sequence,"source_event_hash":tool_event.hash,
            "source_event_type":tool_event.event_type,"source_event_references":[],
        },
    ))
    old_message_sequence = state.event_count
    for i in range(12):
        runtime._record_message(state, Message(role="assistant" if i%2 else "user", content=f"Pre-update cross-source work {i+1}: housekeeping only.", created_at=utc_now_iso()))
    first_compaction = runtime._compact_once(state)
    runtime._record_message(state, Message(
        role="user",
        content=(f"Authoritative user update: the Falcon backup owner is now {NEW_VALUE}. "
                 f"The earlier tool result {OLD_VALUE} remains historical evidence but is superseded for current state."),
        created_at=utc_now_iso(),
    ))
    update_sequence = state.event_count
    for i in range(12):
        runtime._record_message(state, Message(role="assistant" if i%2 else "user", content=f"Pre-restart cross-source work {i+1}: housekeeping only.", created_at=utc_now_iso()))
    context_limit, context_source = runtime._resolve_context_limit(); parent_pid = os.getpid()
    child_output = output_dir / "child_report.json"
    command=[sys.executable,"-m","swaag.benchmark.process_restart_supersession_probe",
        "--sessions-root",str(trial.sessions.root),"--session-id",state.session_id,
        "--old-value",OLD_VALUE,"--new-value",NEW_VALUE,
        "--old-sequence",str(old_message_sequence),"--update-sequence",str(update_sequence),
        "--old-tool-event-sequence",str(tool_event.sequence),"--model-base-url",trial.model.base_url,
        "--context-limit",str(context_limit),"--output",str(child_output)]
    env=dict(os.environ); env["SWAAG__MODEL__CACHE_ENABLED"]="false"
    completed=subprocess_runner(command,cwd=str(Path.cwd()),env=env,text=True,capture_output=True,check=False)
    _write(output_dir/"child_stdout.txt",completed.stdout or ""); _write(output_dir/"child_stderr.txt",completed.stderr or "")
    child=json.loads(child_output.read_text()) if child_output.exists() else None
    different=bool(child and int(child.get("child_pid",parent_pid))!=parent_pid)
    passed=bool(first_compaction and completed.returncode==0 and child and child.get("passed") and different and child.get("old_tool_event_present"))
    report={"passed":passed,"session_id":state.session_id,"first_compaction":first_compaction,
        "old_tool_event_sequence":tool_event.sequence,"old_message_sequence":old_message_sequence,
        "update_sequence":update_sequence,"parent_pid":parent_pid,"child_pid":None if not child else child.get("child_pid"),
        "different_process":different,"subprocess_returncode":completed.returncode,"child_report":child,
        "stdout_sha256":sha256_text(completed.stdout or ""),"stderr_sha256":sha256_text(completed.stderr or ""),
        "context_limit":context_limit,"context_limit_source":context_source,
        "parent_model_identity":getattr(runtime.client,"cache_identity",lambda:type(runtime.client).__name__)(),"command":command}
    path=output_dir/"process_restart_cross_source_supersession.json"; tmp=path.with_name(path.name+".tmp")
    tmp.write_text(stable_json_dumps(report,indent=2)+"\n",encoding="utf-8"); tmp.replace(path); return report
