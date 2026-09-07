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


def _contract(expected_token: str) -> ContractSpec:
    return ContractSpec(
        name="process_restart_delayed_relevance",
        mode="json_schema",
        json_schema={
            "type": "object",
            "properties": {
                "handoff_token": {"type": "string", "enum": [expected_token]}
            },
            "required": ["handoff_token"],
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
    expected_token: str,
    early_event_sequence: int,
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

    phase_two_messages: list[Message] = []
    for index in range(8):
        message = Message(
            role="assistant" if index % 2 else "user",
            content=f"Unrelated subprocess phase-two progress {index + 1}: routine housekeeping only.",
            created_at=utc_now_iso(),
        )
        phase_two_messages.append(message)
        runtime._record_message(state, message)
    no_recent_answer_leak = expected_token not in "\n".join(
        message.content for message in phase_two_messages
    )
    second_compaction = runtime._compact_once(state)

    retained_text = runtime.prompts.render_messages(state.messages)
    contract = _contract(expected_token)
    exact = False
    response_sha256: str | None = None
    error: dict[str, str] | None = None
    try:
        completion = runtime.client.complete(
            (
                "This is the first query for an early authoritative fact after unrelated work, "
                "a complete Python-process restart, and more unrelated work. Recover the exact "
                "handoff token from the retained state.\n\n"
                + retained_text
            ),
            max_tokens=128,
            contract=contract,
            temperature=0.0,
            kind="benchmark_quality_judge",
            live_mode=True,
        )
        payload = json.loads(completion.text)
        _validate_schema_value(payload, contract.json_schema or {}, path=contract.name)
        exact = payload.get("handoff_token") == expected_token
        response_sha256 = sha256_text(completion.text)
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "reason": str(exc)}

    final_events = runtime.history.read_history(session_id)
    early_source_event_present = any(
        event.sequence == early_event_sequence
        and event.event_type == "message_added"
        and expected_token in str(event.payload)
        for event in final_events
    )
    compression_refs = [
        ref
        for event in final_events
        if event.event_type == "history_compressed"
        for ref in event.payload.get("source_event_references", [])
        if isinstance(ref, dict)
    ]
    early_lineage_present = any(
        int(ref.get("sequence", 0)) == early_event_sequence for ref in compression_refs
    ) or any(expected_token in message.content for message in state.messages)
    resolved_limit, resolved_source = runtime._resolve_context_limit()
    report = {
        "passed": all(
            (
                second_compaction,
                restart_replay_matches,
                restart_event_count_matches,
                no_recent_answer_leak,
                early_source_event_present,
                early_lineage_present,
                exact,
            )
        ),
        "session_id": session_id,
        "child_pid": os.getpid(),
        "second_compaction": second_compaction,
        "restart_replay_matches": restart_replay_matches,
        "restart_event_count_matches": restart_event_count_matches,
        "no_recent_answer_leak": no_recent_answer_leak,
        "early_source_event_sequence": early_event_sequence,
        "early_source_event_present": early_source_event_present,
        "early_lineage_present": early_lineage_present,
        "exact_delayed_retrieval": exact,
        "response_sha256": response_sha256,
        "model_identity": getattr(
            runtime.client, "cache_identity", lambda: type(runtime.client).__name__
        )(),
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
    parser.add_argument("--expected-token", required=True)
    parser.add_argument("--early-event-sequence", required=True, type=int)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--context-limit", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = run_child_probe(
        sessions_root=Path(args.sessions_root),
        session_id=args.session_id,
        expected_token=args.expected_token,
        early_event_sequence=args.early_event_sequence,
        model_base_url=args.model_base_url,
        context_limit=args.context_limit,
        output_path=Path(args.output),
    )
    print(stable_json_dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
