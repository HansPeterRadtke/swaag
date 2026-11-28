#!/usr/bin/env python3
"""Prepare or run one immutable, read-only SWAAG core-agent demonstration.

Fresh inputs differ from previous SWAAG agent demonstrations.  No production
session, browser, credential, external publisher or voice project is touched.
The active broker reservation is checked before any model request.  The
first attempt irreversibly consumes the run permission, even if it fails.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

MODEL = "qwen3-coder-30b-a3b-instruct-q4-k-m"
# The actual private model address belongs ONLY in process-local configuration.
# Never hard-code the owner's overlay network details in public Git source.
MODEL_URL = os.environ.get("SWAAG_CORE_DEMO_MODEL_URL", "http://127.0.0.1:14833").rstrip("/")
BROKER_OWNER = "owner-20261128-swaag-core"
SOURCES = {
    "tests/test_action_native_roles.py": "test_action_history_uses_native_user_and_assistant_roles_without_forged_tools",
    "tests/test_completion_evidence_expansion_bounds.py": "test_completion_evidence_requests_stop_at_independent_bounds",
}
MISSING = "tests/test_native_role_history_legacy_20261127.py"
ALLOWED_TOOLS = ["read_file", "list_files", "calculator"]
RUN_OPT_IN = "SWAAG_NOV28_CORE_AGENT_NEW_ONCE"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new(path: Path, content: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())


def json_new(path: Path, content: dict) -> None:
    write_new(path, (json.dumps(content, indent=2, sort_keys=True) + "\n").encode())


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], timeout=12, text=True).strip()


def safe_source(repo: Path) -> dict:
    if not repo.is_absolute() or repo.is_symlink() or not repo.is_dir():
        raise ValueError("Absolute, existing non-symlink source checkout required")
    if git(repo, "status", "--porcelain"):
        raise ValueError("Checkout has other uncommitted changes; do not race source edits")
    result = {}
    for name, required in SOURCES.items():
        path = repo / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Pinned source file is missing or symlinked: " + name)
        raw = path.read_bytes()
        functions = [node.name for node in ast.parse(raw).body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith("test_")]
        if required not in functions:
            raise ValueError("Source is missing the required function " + required)
        result[name] = {"sha256": sha(raw), "test_functions": len(functions), "required": required}
    if (repo / MISSING).exists():
        raise ValueError("Controlled nonexistent source is unexpectedly present")
    return result


def config_text(repo: Path, folder: Path) -> str:
    return f'''# One-use read-only SWAAG core-agent developer acceptance.
[model]
base_url = "{MODEL_URL}"
completion_endpoint = "/chat/completions"
profile_name = "{MODEL}"
context_limit = 262144
remote_context_limit_fallback = 262144
cache_enabled = false
max_retries = 0
seed = 20261128
max_generation_wall_seconds = 160
chat_template_kwargs = {{ enable_thinking = false }}

[tools]
enabled = ["read_file", "list_files", "calculator"]
read_roots = ["{repo}"]
allow_stateful_tools = true
allow_side_effect_tools = false
staged_discovery = false

[runtime]
completion_evaluation_enabled = true
max_completion_rejections = 2
max_turn_actions = 11
max_identical_tool_failures = 2
max_validation_recovery_cycles = 1

[sessions]
root = "{folder / 'sessions'}"
write_projections = true

[agent_data]
root = "{folder / 'agent-data'}"
max_list_files = 1000

[logging]
file_path = "{folder / 'operations.jsonl'}"
'''


def task_text(sources: dict) -> str:
    first, second = SOURCES
    count_a, count_b = [sources[name]["test_functions"] for name in SOURCES]
    return f'''Inspect the current SWAAG core agent's genuine source-backed tests. This is a
READ-ONLY developer verification task. You may use ONLY the already enabled
read_file, list_files and calculator tools. Do not write, publish, or run shell.

As a controlled failure check, first call read_file EXACTLY ONCE on
{MISSING}. This file is intentionally missing. Observe and report the real
FilesystemError; do not invent contents or retry it. Once that actual failure
has occurred, next call list_files on the tests directory, then inspect the
current replacements with separate read_file calls:
{first}
{second}

Name the exact functions that test the native assistant/user message roles and
the bounded completion evidence retrieval. From the genuine source content,
count top-level Python `def test_...` declarations in EACH file (parameterized
functions still count once). Call the real calculator to add those two counts.
Do not substitute a mental sum for a real tool receipt. Finish with paths,
function names, original missing-path error and directory recovery, both
counts, calculator input and observed result. Be candid if the task fails.
The original developer task has fixed source counts independently measured
by the verifier, but the counts were deliberately not disclosed to the model.
'''


def prepare(repo: Path, folder: Path) -> dict:
    if (not folder.is_absolute() or folder.is_symlink() or folder.exists()
            or repo == folder or repo in folder.parents):
        raise ValueError("Fresh absolute private evidence path outside repository required")
    from urllib.parse import urlsplit
    import ipaddress
    parsed = urlsplit(MODEL_URL)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
    except ValueError as exc:
        raise ValueError("Demo endpoint must use an explicit private numeric IP") from exc
    if (parsed.scheme != "http" or not (address.is_private or address.is_loopback)
            or parsed.port != 14833 or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path):
        raise ValueError("Demo endpoint must use the configured private resident model port")
    source = safe_source(repo)
    source_revision = git(repo, "rev-parse", "HEAD")
    cfg, task = config_text(repo, folder), task_text(source)
    folder.mkdir(mode=0o700, parents=False)
    write_new(folder / "fixed-task.txt", task.encode())
    write_new(folder / "read-only-config.toml", cfg.encode())
    manifest = {
        "schema": "swaag-core-native-roles-live-v1", "prepared_at": now(),
        "source_revision": source_revision, "source_repo": str(repo),
        "source_checks": source, "model": MODEL, "model_base_url": MODEL_URL,
        "native_transport": "/chat/completions", "read_roots": [str(repo)],
        "tool_names": ALLOWED_TOOLS, "allow_side_effect_tools": False,
        "expected_missing": MISSING, "task_sha256": sha(task.encode()),
        "config_sha256": sha(cfg.encode()), "max_model_calls": 14,
        "controlled_error_count": 1, "no_model_calls_during_prepare": True,
    }
    json_new(folder / "manifest.json", manifest)
    return manifest


def check_preflight(repo: Path, folder: Path) -> dict:
    if not folder.is_dir() or folder.is_symlink() or not (folder / "manifest.json").is_file():
        raise ValueError("No untouched prepared private evidence folder")
    manifest = json.loads((folder / "manifest.json").read_text())
    if (manifest.get("schema") != "swaag-core-native-roles-live-v1"
            or manifest.get("source_revision") != git(repo, "rev-parse", "HEAD")
            or manifest.get("source_repo") != str(repo)
            or manifest.get("model_base_url") != MODEL_URL
            or manifest.get("source_checks") != safe_source(repo)
            or manifest.get("expected_missing") != MISSING
            or manifest.get("tool_names") != ALLOWED_TOOLS
            or manifest.get("allow_side_effect_tools") is not False):
        raise ValueError("Source, tool scope or immutable one-use manifest changed")
    if (sha((folder / "read-only-config.toml").read_bytes()) != manifest["config_sha256"]
            or sha((folder / "fixed-task.txt").read_bytes()) != manifest["task_sha256"]
            or (folder / ".one_use_started.json").exists()
            or (folder / "real-cli-console.log").exists()
            or (folder / "result.json").exists()
            or (folder / "sessions").exists()):
        raise ValueError("A prepared file changed or a model run may have occurred: refuse replay")
    from swaag.config import load_config
    loaded = load_config([str(folder / "read-only-config.toml")])
    if (loaded.model.profile_name != MODEL
            or loaded.tools.enabled != ALLOWED_TOOLS
            or loaded.tools.allow_side_effect_tools
            or loaded.model.max_retries != 0
            or loaded.model.completion_endpoint != "/chat/completions"
            or loaded.tools.read_roots != [repo]):
        raise ValueError("Installed runtime config does not preserve expected authority")
    return manifest


def broker_admitted(lease_id: str) -> dict:
    if not isinstance(lease_id, str) or not lease_id.startswith("rb_"):
        raise ValueError("A verified live broker reservation id is mandatory")
    data = subprocess.check_output(["/data/bin/resource-broker", "status"], timeout=9, text=True)
    status = json.loads(data)
    records = status.get("machines", {}).get("thor", {}).get("requests", [])
    for item in records:
        # The real Raspi resource broker uses `request_id` (not `id`).
        # Do not accept alternate guessed identities or stale authorizations.
        if item.get("request_id") == lease_id:
            if item.get("owner") != BROKER_OWNER or item.get("state") != "active":
                raise ValueError("Broker lease belongs to a different owner or is not active")
            expires = item.get("expires_at")
            if type(expires) not in (int, float) or expires < time.time() + 60:
                raise ValueError("Broker lease has expired or lacks safe launch headroom")
            return {"lease_id": lease_id, "owner": BROKER_OWNER, "state": "active", "verified_at": now()}
    raise ValueError("Active broker lease cannot be verified; no model request permitted")


def stats(folder: Path) -> dict:
    events = []
    files = list((folder / "sessions").glob("session_*/complete_history.jsonl"))
    if len(files) > 1:
        raise ValueError("More than one actual session in the one-use folder")
    if files:
        journal = files[0]
        if journal.is_symlink() or journal.stat().st_size > 90_000_000:
            raise ValueError("Unbounded or symlinked active history")
        for line in journal.open():
            try: events.append(json.loads(line))
            except json.JSONDecodeError: break  # One in-flight trailing append.
    types = Counter(event.get("type") for event in events)
    failures = [event for event in events if event.get("type") == "tool_error"]
    repeats = sum((event.get("payload") or {}).get("error_type") == "FilesystemError"
                  and (event.get("payload") or {}).get("tool_input", {}).get("path") == MISSING
                  for event in failures)
    return {"model_requests": types["model_request_sent"],
            "model_responses": types["model_response_received"],
            "tool_calls": types["tool_called"], "tool_results": types["tool_result"],
            "tool_errors": types["tool_error"], "repeated_missing_file_errors": repeats,
            "terminal_turns": types["turn_finished"] + types["turn_completed"],
            "private_journal": str(files[0]) if files else ""}


def run_once(repo: Path, folder: Path, lease_id: str, max_wall: float) -> int:
    manifest = check_preflight(repo, folder)
    if os.environ.get(RUN_OPT_IN) != "1":
        raise ValueError("Explicit one-use source agent model opt-in missing")
    admission = broker_admitted(lease_id)
    json_new(folder / ".one_use_started.json", {"time": now(), "source_revision": manifest["source_revision"], **admission})
    receipt = {"started": now(), "lease_id": lease_id, "status": "not_completed",
               "source_revision": manifest["source_revision"], "model": MODEL,
               "real_model_result_not_assumed": True, "tool_writes_allowed": False}
    child = None
    start = time.monotonic()
    out_fd = -1
    try:
        with urllib.request.urlopen(MODEL_URL + "/v1/models", timeout=4) as response:
            catalog = json.load(response)
        if [item.get("id") for item in catalog.get("data", [])] != [MODEL]:
            raise ValueError("Connected model alias changed: no request may proceed")
        receipt["model_alias_verified"] = True
        command = [sys.executable, "-m", "swaag", "--config", str(folder / "read-only-config.toml"),
                   "ask", "--session", "swaag-core-native-roles-oneuse-nov28"]
        out_fd = os.open(folder / "real-cli-console.log", os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with (folder / "fixed-task.txt").open("rb") as input_file, os.fdopen(out_fd, "wb") as output:
            out_fd = -1
            child = subprocess.Popen(command, cwd=repo, stdin=input_file, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            receipt["child_pid"] = child.pid
            json_new(folder / "running-child-pid.json", {"pid": child.pid, "launched_at": now(), "lease_id": lease_id})
            stop_reason = None
            while child.poll() is None:
                observed = stats(folder)
                if observed["repeated_missing_file_errors"] >= 2:
                    stop_reason = "second_unnecessary_identical_filesystem_error"
                elif observed["model_requests"] > 14 or observed["tool_calls"] > 14:
                    stop_reason = "bounded_request_or_tool_count"
                elif time.monotonic() - start > max_wall:
                    stop_reason = "bounded_one_use_developer_wall_duration"
                if stop_reason:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=7)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=7)
                    break
                time.sleep(1.0)
            receipt["child_returncode"] = child.returncode
            receipt["bounded_stop_reason"] = stop_reason
    except Exception as error:
        receipt["setup_error_type"] = type(error).__name__
        receipt["setup_error"] = str(error)[:320]
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=7)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=7)
            receipt["child_returncode"] = child.returncode
    finally:
        if out_fd >= 0: os.close(out_fd)
        receipt["elapsed_seconds"] = round(time.monotonic() - start, 3)
        receipt.update(stats(folder))
        receipt["console_sha256"] = sha((folder / "real-cli-console.log").read_bytes()) if (folder / "real-cli-console.log").is_file() else None
        receipt["completed_at"] = now()
        receipt["status"] = "process_zero_pending_independent_oracle" if receipt.get("child_returncode") == 0 else "failed_or_unverified"
        json_new(folder / "result.json", receipt)
    print("SWAAG_CORE_ONE_USE_REAL_RECEIPT", json.dumps({key: receipt.get(key) for key in
          ("status", "child_returncode", "bounded_stop_reason", "elapsed_seconds",
           "model_requests", "model_responses", "tool_calls", "tool_results", "tool_errors")}))
    return 0 if receipt["status"] == "process_zero_pending_independent_oracle" else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--lease-id")
    parser.add_argument("--max-wall-seconds", type=float, default=720.0)
    args = parser.parse_args(argv)
    if args.prepare == args.run:
        parser.error("Choose exactly one of --prepare or --run")
    if args.prepare:
        record = prepare(args.repo, args.evidence)
        print("SWAAG_CORE_PREPARED_NO_MODEL_CALL", json.dumps({
            "rev": record["source_revision"], "model": record["model"],
            "counts": [v["test_functions"] for v in record["source_checks"].values()],
            "evidence_root": str(args.evidence)}))
        return 0
    if not 60 <= args.max_wall_seconds <= 1800:
        parser.error("Bound one-use wall seconds between 60 and 1800")
    return run_once(args.repo, args.evidence, args.lease_id or "", args.max_wall_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
