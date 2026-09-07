from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.tools.base import _validate_schema_value
from swaag.types import ContractSpec, Message
from swaag.utils import sha256_text, stable_json_dumps, utc_now_iso


def _contract(old_value: str, new_value: str) -> ContractSpec:
    return ContractSpec(
        name="process_restart_authoritative_supersession",
        mode="json_schema",
        json_schema={
            "type": "object",
            "properties": {
                "historical_value": {"type": "string", "enum": [old_value]},
                "current_value": {"type": "string", "enum": [new_value]},
                "supersession_understood": {"type": "boolean", "enum": [True]},
            },
            "required": ["historical_value", "current_value", "supersession_understood"],
            "additionalProperties": False,
        },
    )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_child_probe(
    *,
    sessions_root: Path,
    session_id: str,
    old_value: str,
    new_value: str,
    old_sequence: int,
    update_sequence: int,
    model_base_url: str,
    context_limit: int,
    output_path: Path,
) -> dict[str, Any]:
    env = dict(os.environ)
    env["SWAAG__SESSIONS__ROOT"] = str(sessions_root)
    env["SWAAG__MODEL__BASE_URL"] = model_base_url.rstrip("/")
    env["SWAAG__MODEL__CONTEXT_LIMIT"] = str(int(context_limit))
    env["SWAAG__MODEL__CACHE_ENABLED"] = "false"
    config = load_config(env=env)
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session(session_id)
    rebuilt = runtime.history.rebuild_from_history(session_id, prefer_checkpoint=False)
    restart_replay_matches = [
        (m.role, m.content, m.metadata) for m in state.messages
    ] == [
        (m.role, m.content, m.metadata) for m in rebuilt.messages
    ]
    restart_event_count_matches = state.event_count == rebuilt.event_count

    later_messages: list[Message] = []
    for index in range(12):
        message = Message(
            role="assistant" if index % 2 else "user",
            content=f"Post-update unrelated work {index + 1}: housekeeping only.",
            created_at=utc_now_iso(),
        )
        later_messages.append(message)
        runtime._record_message(state, message)
    no_recent_answer_leak = all(
        value not in "\n".join(message.content for message in later_messages)
        for value in (old_value, new_value)
    )
    second_compaction = runtime._compact_once(state)

    contract = _contract(old_value, new_value)
    exact = False
    response_sha256: str | None = None
    error: dict[str, str] | None = None
    try:
        retained = runtime.prompts.render_messages(state.messages)
        completion = runtime.client.complete(
            (
                "An earlier authoritative value was later explicitly superseded by a newer authoritative update. "
                "Return the exact historical value and the exact current value, and confirm the newer update "
                "supersedes the older value. Do not treat the newer update as an untrusted decoy.\n\n" + retained
            ),
            max_tokens=160,
            contract=contract,
            temperature=0.0,
            kind="benchmark_quality_judge",
            live_mode=True,
        )
        payload = json.loads(completion.text)
        _validate_schema_value(payload, contract.json_schema or {}, path=contract.name)
        exact = (
            payload.get("historical_value") == old_value
            and payload.get("current_value") == new_value
            and payload.get("supersession_understood") is True
        )
        response_sha256 = sha256_text(completion.text)
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "reason": str(exc)}

    events = runtime.history.read_history(session_id)
    old_event_present = any(
        event.sequence == old_sequence and event.event_type == "message_added" and old_value in str(event.payload)
        for event in events
    )
    new_event_present = any(
        event.sequence == update_sequence and event.event_type == "message_added" and new_value in str(event.payload)
        for event in events
    )
    compression_refs = [
        ref
        for event in events
        if event.event_type == "history_compressed"
        for ref in event.payload.get("source_event_references", [])
        if isinstance(ref, dict)
    ]
    old_lineage_present = any(
        int(ref.get("sequence", 0)) == old_sequence for ref in compression_refs
    ) or any(old_value in message.content for message in state.messages)
    new_lineage_present = any(
        int(ref.get("sequence", 0)) == update_sequence for ref in compression_refs
    ) or any(new_value in message.content for message in state.messages)
    resolved_limit, resolved_source = runtime._resolve_context_limit()
    report = {
        "passed": all((
            second_compaction,
            restart_replay_matches,
            restart_event_count_matches,
            no_recent_answer_leak,
            old_event_present,
            new_event_present,
            old_lineage_present,
            new_lineage_present,
            exact,
        )),
        "session_id": session_id,
        "child_pid": os.getpid(),
        "second_compaction": second_compaction,
        "restart_replay_matches": restart_replay_matches,
        "restart_event_count_matches": restart_event_count_matches,
        "no_recent_answer_leak": no_recent_answer_leak,
        "old_sequence": old_sequence,
        "update_sequence": update_sequence,
        "old_event_present": old_event_present,
        "new_event_present": new_event_present,
        "old_lineage_present": old_lineage_present,
        "new_lineage_present": new_lineage_present,
        "exact_supersession_retrieval": exact,
        "response_sha256": response_sha256,
        "model_identity": getattr(runtime.client, "cache_identity", lambda: type(runtime.client).__name__)(),
        "context_limit": resolved_limit,
        "context_limit_source": resolved_source,
        "error": error,
    }
    _atomic_write(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-root", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--old-value", required=True)
    parser.add_argument("--new-value", required=True)
    parser.add_argument("--old-sequence", required=True, type=int)
    parser.add_argument("--update-sequence", required=True, type=int)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--context-limit", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = run_child_probe(
        sessions_root=Path(args.sessions_root),
        session_id=args.session_id,
        old_value=args.old_value,
        new_value=args.new_value,
        old_sequence=args.old_sequence,
        update_sequence=args.update_sequence,
        model_base_url=args.model_base_url,
        context_limit=args.context_limit,
        output_path=Path(args.output),
    )
    print(stable_json_dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
