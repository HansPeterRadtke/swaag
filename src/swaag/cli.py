from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from swaag.config import load_config
from swaag.runtime import AgentRuntime, BudgetExceededError
from swaag.utils import stable_json_dumps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swaag")
    parser.add_argument("--config", action="append", default=[], help="Path to a TOML config file. Can be passed multiple times.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Run one agent turn and print the final answer.")
    ask_parser.add_argument("prompt", nargs="?", help="User prompt. If omitted, stdin is read.")
    ask_parser.add_argument("--session", help="Session name or id to resume. Defaults to latest session.")

    chat_parser = subparsers.add_parser("chat", help="Interactive chat shell.")
    chat_parser.add_argument("--session", help="Session name or id to resume. Defaults to latest session.")

    doctor_parser = subparsers.add_parser("doctor", help="Check local llama.cpp connectivity and structured output paths.")
    doctor_parser.add_argument("--session", help="Optional session name or id.")
    doctor_parser.add_argument("--json", action="store_true", help="Print the result as JSON.")

    budget_parser = subparsers.add_parser("budget-demo", help="Show a budget report for a sample user input.")
    budget_parser.add_argument("prompt", nargs="?", help="User prompt. If omitted, stdin is read.")
    budget_parser.add_argument("--prompt-mode", choices=["standard", "lean"], default="standard")

    subparsers.add_parser("tools", help="List enabled tools.")

    sessions_parser = subparsers.add_parser("sessions", help="List stored sessions with names and metadata.")
    sessions_parser.add_argument("--json", action="store_true", help="Print as JSON.")

    rename_parser = subparsers.add_parser("rename", help="Rename a session.")
    rename_parser.add_argument("session", help="Session name, id, or 'latest'.")
    rename_parser.add_argument("new_name", help="New human-readable name.")

    control_parser = subparsers.add_parser("control", help="Send a control message to the active session.")
    control_parser.add_argument("message", help="Control message text.")
    control_parser.add_argument("--session", help="Session name or id. Defaults to latest active session.")
    control_parser.add_argument("--source", default="cli", help="Source tag for the message.")

    state_parser = subparsers.add_parser("state", help="Print reconstructed session state.")
    state_parser.add_argument("session", nargs="?", help="Session name or id. Defaults to latest.")

    history_parser = subparsers.add_parser("history", help="Inspect session history.")
    history_parser.add_argument("history_args", nargs="+",
        help="One of: <session>, show <session>, replay <session>, diff <session>, detail <session> <query>.")
    history_parser.add_argument("--tail", type=int, default=0, help="Only print the last N events in show mode.")
    history_parser.add_argument("--event-type", action="append", default=[], help="Filter show output to these event types.")
    history_parser.add_argument("--topic", default="", help="Topic hint for detail mode.")
    history_parser.add_argument("--max-results", type=int, default=8, help="Max results in detail mode.")

    notes_parser = subparsers.add_parser("notes", help="Print notes from reconstructed session state.")
    notes_parser.add_argument("session", nargs="?", help="Session name or id. Defaults to latest.")

    readers_parser = subparsers.add_parser("readers", help="Print reader states from reconstructed session state.")
    readers_parser.add_argument("session", nargs="?", help="Session name or id. Defaults to latest.")

    checkpoint_parser = subparsers.add_parser("checkpoint", help="Create or restore a code checkpoint.")
    checkpoint_sub = checkpoint_parser.add_subparsers(dest="checkpoint_command", required=True)
    ckpt_create = checkpoint_sub.add_parser("create", help="Create a checkpoint of the current workspace.")
    ckpt_create.add_argument("--session", help="Session name or id. Defaults to latest.")
    ckpt_create.add_argument("--label", default="", help="Human-readable label.")
    ckpt_create.add_argument("--workspace", default="", help="Workspace root directory. Defaults to cwd.")
    ckpt_restore = checkpoint_sub.add_parser("restore", help="Restore a code checkpoint.")
    ckpt_restore.add_argument("checkpoint_id", nargs="?", help="Checkpoint id to restore. Defaults to latest.")
    ckpt_restore.add_argument("--session", help="Session name or id. Defaults to latest.")
    ckpt_restore.add_argument("--workspace", default="", help="Workspace root to restore into. Defaults to checkpoint's original root.")
    checkpoint_sub.add_parser("list", help="List checkpoints for a session.").add_argument(
        "--session", help="Session name or id. Defaults to latest."
    )

    edit_parser = subparsers.add_parser("edit-dry-run", help="Preview a text edit without writing the file.")
    edit_parser.add_argument("--session", help="Session name or id.")
    edit_parser.add_argument("path")
    edit_parser.add_argument("operation", choices=["replace_range", "insert_at", "delete_range", "replace_pattern_once", "replace_pattern_all"])
    edit_parser.add_argument("--start", type=int)
    edit_parser.add_argument("--end", type=int)
    edit_parser.add_argument("--position", type=int)
    edit_parser.add_argument("--replacement")
    edit_parser.add_argument("--insertion")
    edit_parser.add_argument("--pattern")
    return parser


def _read_prompt_value(value: str | None) -> str:
    if value is not None:
        return value
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No prompt provided")
    return data


def _read_optional_prompt_value(value: str | None) -> str:
    if value is not None:
        return value
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read()


def _print_budget_table(report: dict) -> None:
    for name, payload in report.items():
        budget = payload["budget"]
        print(f"[{name}] prompt_mode={payload['prompt_mode']}")
        print(
            "  "
            f"input={budget['input_tokens']} reserved={budget['reserved_response_tokens']} "
            f"margin={budget['safety_margin_tokens']} required={budget['required_tokens']} "
            f"limit={budget['context_limit']} fits={budget['fits']} exact={budget['exact']}"
        )
        for item in budget["breakdown"]:
            print(
                "    "
                f"{item['name']}: tokens={item['tokens']} category={item['category']} "
                f"in_context={item['include_in_context']} exact={item['exact']}"
            )


def _print_tools(runtime: AgentRuntime) -> None:
    for tool in runtime.tools.enabled_tools(runtime.config):
        print(f"- {tool.name} [{tool.kind}]: {tool.description}")
        print(f"  schema={stable_json_dumps(tool.input_schema)}")


def _resolve_session_id(runtime: AgentRuntime, session_ref: str | None, *, latest_if_none: bool = True) -> str | None:
    """Resolve a session ref (name/id/None) to an id, optionally defaulting to latest."""
    return runtime.resolve_session_ref(session_ref, latest_if_none=latest_if_none)


def _run_ask(runtime: AgentRuntime, prompt: str, session_ref: str | None) -> int:
    state = runtime.create_or_load_user_session(session_ref)
    effective_prompt = prompt.strip()
    if not effective_prompt:
        deferred = runtime.pop_next_deferred_task(state, reason="ask_without_prompt")
        if deferred is None:
            raise SystemExit("No prompt provided and no deferred task queued")
        effective_prompt = deferred.text
    result = runtime.run_turn_in_session(state, effective_prompt)
    print(result.assistant_text)
    print(f"\n[session={result.session_id} name={state.session_name}]", file=sys.stderr)
    return 0


def _run_chat(runtime: AgentRuntime, session_ref: str | None) -> int:
    state = runtime.create_or_load_user_session(session_ref)
    print(f"session={state.session_id} name={state.session_name}")
    print("Enter /exit or /quit to stop. /session shows current id. /deferred shows queued tasks.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            break
        if not line.strip():
            continue
        if line.strip() in {"/exit", "/quit"}:
            break
        if line.strip() == "/session":
            print(f"{state.session_id} ({state.session_name})")
            continue
        if line.strip() == "/deferred":
            for task in state.deferred_tasks:
                print(f"  [{task.task_id}] {task.text}")
            continue
        result = runtime.run_turn_in_session(state, line)
        print(result.assistant_text)
    return 0


def _run_doctor(runtime: AgentRuntime, emit_json: bool, session_ref: str | None) -> int:
    session_id = _resolve_session_id(runtime, session_ref, latest_if_none=True)
    report = runtime.doctor(session_id=session_id)
    if emit_json:
        print(stable_json_dumps(report, indent=2))
    else:
        print(f"session_id={report['session_id']}")
        print(f"health={stable_json_dumps(report['health'])}")
        print(f"tokenize_probe_tokens={report['tokenize_probe_tokens']}")
        print(f"grammar_probe={report['grammar_probe']}")
        print(f"schema_probe={stable_json_dumps(report['schema_probe'])}")
    return 0


def _run_sessions(runtime: AgentRuntime, emit_json: bool) -> int:
    entries = runtime.history.list_session_entries()
    if emit_json:
        print(stable_json_dumps(entries, indent=2))
        return 0
    if not entries:
        print("(no sessions)")
        return 0
    for entry in entries:
        session_id = entry.get("session_id", "")
        name = entry.get("session_name") or session_id
        updated = str(entry.get("updated_at", ""))[:16]
        turns = entry.get("turn_count", 0)
        active = " [active]" if entry.get("active") else ""
        checkpoints = entry.get("code_checkpoint_count", 0)
        ckpt_info = f" ckpts={checkpoints}" if checkpoints else ""
        print(f"{name:<30} {updated}  turns={turns}{ckpt_info}{active}  id={session_id}")
    return 0


def _run_rename(runtime: AgentRuntime, session_ref: str, new_name: str) -> int:
    state = runtime.history.rename_session(session_ref, new_name)
    print(f"Renamed to: {state.session_name}  (id={state.session_id})")
    return 0


def _run_control(runtime: AgentRuntime, message: str, session_ref: str | None, source: str) -> int:
    payload = runtime.queue_control_message(session_ref, message, source=source)
    active_info = f"active={payload['active']}"
    print(f"Queued control message {payload['control_id']} for session {payload['status']['session_name']} [{payload['status']['session_id']}] {active_info}")
    print(stable_json_dumps(payload["status"], indent=2))
    return 0


def _parse_history_args(values: list[str]) -> tuple[str, str, str]:
    """Return (mode, session_ref, extra) where extra is the detail query if mode=='detail'."""
    if not values:
        raise SystemExit("history requires a session id or subcommand")
    if values[0] in {"show", "replay", "diff"}:
        if len(values) != 2:
            raise SystemExit(f"history {values[0]} requires exactly one session id")
        return values[0], values[1], ""
    if values[0] == "detail":
        if len(values) < 3:
            raise SystemExit("history detail requires: <session> <query>")
        return "detail", values[1], " ".join(values[2:])
    if len(values) != 1:
        raise SystemExit("history show mode takes exactly one session id or 'latest'")
    return "show", values[0], ""


def _run_history(
    runtime: AgentRuntime,
    history_args: list[str],
    tail: int,
    event_types: list[str],
    topic: str,
    max_results: int,
) -> int:
    mode, session_ref, extra = _parse_history_args(history_args)
    session_id = _resolve_session_id(runtime, session_ref, latest_if_none=(session_ref in {"", "latest"}))
    if session_id is None:
        print(f"ERROR: Session not found: {session_ref!r}", file=sys.stderr)
        return 1

    if mode == "show":
        events = runtime.history.read_history(session_id)
        if event_types:
            allowed = set(event_types)
            events = [event for event in events if event.event_type in allowed]
        if tail > 0:
            events = events[-tail:]
        for event in events:
            print(stable_json_dumps(asdict(event)))
        return 0
    if mode == "replay":
        state = runtime.history.rebuild_from_history(session_id)
        print(stable_json_dumps(asdict(state), indent=2))
        return 0
    if mode == "diff":
        replayed = asdict(runtime.history.rebuild_from_history(session_id))
        projection_path = runtime.history.current_state_path(session_id)
        projection = None
        if projection_path.exists():
            projection = json.loads(projection_path.read_text(encoding="utf-8"))
        result = {
            "session_id": session_id,
            "projection_exists": projection is not None,
            "match": projection == replayed if projection is not None else False,
            "replayed_event_count": replayed.get("event_count"),
            "replayed_last_event_hash": replayed.get("last_event_hash"),
        }
        print(stable_json_dumps(result, indent=2))
        return 0
    if mode == "detail":
        result = runtime.query_history_details(
            session_ref=session_id,
            query_text=extra,
            topic_hint=topic,
        )
        if max_results > 0:
            result["matches"] = result.get("matches", [])[:max_results]
            result["match_count"] = len(result["matches"])
        print(stable_json_dumps(result, indent=2))
        return 0
    raise SystemExit(f"Unhandled history mode: {mode}")


def _run_state(runtime: AgentRuntime, session_ref: str | None) -> int:
    session_id = _resolve_session_id(runtime, session_ref, latest_if_none=True)
    if session_id is None:
        print("ERROR: No session found.", file=sys.stderr)
        return 1
    state = runtime.history.rebuild_from_history(session_id)
    print(stable_json_dumps(asdict(state), indent=2))
    return 0


def _run_notes(runtime: AgentRuntime, session_ref: str | None) -> int:
    session_id = _resolve_session_id(runtime, session_ref, latest_if_none=True)
    if session_id is None:
        print("ERROR: No session found.", file=sys.stderr)
        return 1
    state = runtime.history.rebuild_from_history(session_id)
    print(stable_json_dumps([asdict(note) for note in state.notes], indent=2))
    return 0


def _run_readers(runtime: AgentRuntime, session_ref: str | None) -> int:
    session_id = _resolve_session_id(runtime, session_ref, latest_if_none=True)
    if session_id is None:
        print("ERROR: No session found.", file=sys.stderr)
        return 1
    state = runtime.history.rebuild_from_history(session_id)
    print(stable_json_dumps({key: asdict(value) for key, value in state.reader_states.items()}, indent=2))
    return 0


def _run_checkpoint(runtime: AgentRuntime, args) -> int:
    cmd = args.checkpoint_command
    session_ref = getattr(args, "session", None)
    session_id = _resolve_session_id(runtime, session_ref, latest_if_none=True)
    if session_id is None:
        print("ERROR: No session found.", file=sys.stderr)
        return 1

    if cmd == "create":
        state = runtime.history.rebuild_from_history(session_id, write_projections=False)
        ckpt = runtime.create_code_checkpoint(state, label=args.label, workspace_root=args.workspace or None)
        print(f"Checkpoint created: {ckpt['checkpoint_id']}")
        print(f"  label={ckpt['label']}  files={ckpt['file_count']}  root={ckpt['workspace_root']}")
        print(f"  archive={ckpt['storage_path']}")
        return 0

    if cmd == "restore":
        state = runtime.history.rebuild_from_history(session_id, write_projections=False)
        checkpoint_id = getattr(args, "checkpoint_id", None) or "latest"
        result = runtime.restore_code_checkpoint(state, checkpoint_ref=checkpoint_id, workspace_root=args.workspace or None)
        print(f"Restored checkpoint {result['checkpoint_id']} ({result['label']})")
        print(f"  files={result['file_count']}  root={result['workspace_root']}")
        return 0

    if cmd == "list":
        state = runtime.history.rebuild_from_history(session_id)
        if not state.code_checkpoints:
            print("(no checkpoints)")
            return 0
        for ckpt in state.code_checkpoints:
            print(f"{ckpt.checkpoint_id}  {ckpt.label:<24}  files={ckpt.file_count}  {ckpt.created_at[:16]}")
        return 0

    raise SystemExit(f"Unhandled checkpoint command: {cmd}")


def _run_edit_dry_run(runtime: AgentRuntime, args) -> int:
    session_ref = getattr(args, "session", None)
    state = runtime.create_or_load_user_session(session_ref)
    raw_input = {
        "path": args.path,
        "operation": args.operation,
        "dry_run": True,
    }
    for key in ["start", "end", "position", "replacement", "insertion", "pattern"]:
        value = getattr(args, key)
        if value is not None:
            raw_input[key] = value
    result = runtime.execute_tool_once("edit_text", raw_input, session_id=state.session_id)
    if result.tool_result is None:
        print("ERROR: edit_text failed", file=sys.stderr)
        return 1
    print(stable_json_dumps(result.tool_result.output, indent=2))
    print(f"\n[session={result.session_id}]", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    runtime = AgentRuntime(config)

    try:
        if args.command == "edit-dry-run":
            return _run_edit_dry_run(runtime, args)
        if args.command == "ask":
            return _run_ask(runtime, _read_optional_prompt_value(args.prompt), args.session)
        if args.command == "chat":
            return _run_chat(runtime, args.session)
        if args.command == "doctor":
            return _run_doctor(runtime, args.json, args.session)
        if args.command == "budget-demo":
            report = runtime.budget_demo(_read_prompt_value(args.prompt), prompt_mode=args.prompt_mode)
            _print_budget_table(report)
            return 0
        if args.command == "tools":
            _print_tools(runtime)
            return 0
        if args.command == "sessions":
            return _run_sessions(runtime, args.json)
        if args.command == "rename":
            return _run_rename(runtime, args.session, args.new_name)
        if args.command == "control":
            return _run_control(runtime, args.message, args.session, args.source)
        if args.command == "history":
            return _run_history(runtime, args.history_args, args.tail, args.event_type, args.topic, args.max_results)
        if args.command == "state":
            return _run_state(runtime, getattr(args, "session", None))
        if args.command == "notes":
            return _run_notes(runtime, getattr(args, "session", None))
        if args.command == "readers":
            return _run_readers(runtime, getattr(args, "session", None))
        if args.command == "checkpoint":
            return _run_checkpoint(runtime, args)
    except BudgetExceededError as exc:
        print(str(exc), file=sys.stderr)
        if exc.report is not None:
            print(stable_json_dumps(asdict(exc.report), indent=2), file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    raise SystemExit(f"Unhandled command: {args.command}")
