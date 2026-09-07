from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from swaag.attachments import AttachmentStore
from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.tools.base import _validate_schema_value
from swaag.types import ContractSpec, Message
from swaag.utils import sha256_text, stable_json_dumps, utc_now_iso


def _contract(user_fact: str, tool_fact: str, attachment_fact: str) -> ContractSpec:
    return ContractSpec(
        name="process_restart_multisource_delayed_relevance",
        mode="json_schema",
        json_schema={
            "type": "object",
            "properties": {
                "user_constraint": {"type": "string", "enum": [user_fact]},
                "tool_result": {"type": "string", "enum": [tool_fact]},
                "attachment_value": {"type": "string", "enum": [attachment_fact]},
            },
            "required": ["user_constraint", "tool_result", "attachment_value"],
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
    user_fact: str,
    tool_fact: str,
    attachment_fact: str,
    attachment_id: str,
    model_base_url: str,
    context_limit: int,
    output_path: Path,
    phase_label: str = "phase-two",
    query: bool = True,
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
        (m.role, m.content, m.name, m.metadata) for m in state.messages
    ] == [
        (m.role, m.content, m.name, m.metadata) for m in rebuilt.messages
    ]
    restart_event_count_matches = state.event_count == rebuilt.event_count
    attachment_reference = next(item for item in state.attachments if item.attachment_id == attachment_id)
    attachment_bytes = AttachmentStore(
        config.sessions.root, max_upload_bytes=config.attachments.max_upload_bytes
    ).read_bytes(attachment_reference)
    attachment_text = attachment_bytes.decode("utf-8")
    attachment_exact = attachment_fact in attachment_text

    phase_two_messages: list[Message] = []
    for index in range(16):
        message = Message(
            role="assistant" if index % 2 else "user",
            content=f"Long unrelated subprocess {phase_label} progress {index + 1}: housekeeping only.",
            created_at=utc_now_iso(),
        )
        phase_two_messages.append(message)
        runtime._record_message(state, message)
    no_recent_answer_leak = all(
        value not in "\n".join(message.content for message in phase_two_messages)
        for value in (user_fact, tool_fact, attachment_fact)
    )
    second_compaction = runtime._compact_once(state)

    retained_text = runtime.prompts.render_messages(state.messages)
    exact = not query
    response_sha256: str | None = None
    error: dict[str, str] | None = None
    if query:
        contract = _contract(user_fact, tool_fact, attachment_fact)
        try:
            prompt = (
                "This is the first query for three early authoritative values after long unrelated "
                "work, multiple complete Python-process restarts, and more unrelated work. Recover all three "
                "exact values. The attachment is authoritative raw evidence and is included below.\n\n"
                + retained_text
                + "\n\nAuthoritative raw attachment evidence:\n"
                + attachment_text
            )
            completion = runtime.client.complete(
                prompt,
                max_tokens=192,
                contract=contract,
                temperature=0.0,
                kind="benchmark_quality_judge",
                live_mode=True,
            )
            payload = json.loads(completion.text)
            _validate_schema_value(payload, contract.json_schema or {}, path=contract.name)
            exact = (
                payload.get("user_constraint") == user_fact
                and payload.get("tool_result") == tool_fact
                and payload.get("attachment_value") == attachment_fact
            )
            response_sha256 = sha256_text(completion.text)
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "reason": str(exc)}

    events = runtime.history.read_history(session_id)
    tool_event_present = any(
        event.event_type == "tool_result" and tool_fact in str(event.payload) for event in events
    )
    user_event_present = any(
        event.event_type == "message_added" and user_fact in str(event.payload) for event in events
    )
    attachment_event_present = any(
        event.event_type == "attachment_added"
        and event.payload.get("attachment", {}).get("attachment_id") == attachment_id
        for event in events
    )
    resolved_limit, resolved_source = runtime._resolve_context_limit()
    report = {
        "passed": all(
            (
                second_compaction,
                restart_replay_matches,
                restart_event_count_matches,
                attachment_exact,
                no_recent_answer_leak,
                tool_event_present,
                user_event_present,
                attachment_event_present,
                exact,
            )
        ),
        "session_id": session_id,
        "child_pid": os.getpid(),
        "second_compaction": second_compaction,
        "restart_replay_matches": restart_replay_matches,
        "restart_event_count_matches": restart_event_count_matches,
        "attachment_exact": attachment_exact,
        "no_recent_answer_leak": no_recent_answer_leak,
        "tool_event_present": tool_event_present,
        "user_event_present": user_event_present,
        "attachment_event_present": attachment_event_present,
        "query_performed": query,
        "phase_label": phase_label,
        "exact_delayed_retrieval": exact if query else None,
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
    parser.add_argument("--user-fact", required=True)
    parser.add_argument("--tool-fact", required=True)
    parser.add_argument("--attachment-fact", required=True)
    parser.add_argument("--attachment-id", required=True)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--context-limit", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-label", default="phase-two")
    parser.add_argument("--skip-query", action="store_true")
    args = parser.parse_args(argv)
    report = run_child_probe(
        sessions_root=Path(args.sessions_root),
        session_id=args.session_id,
        user_fact=args.user_fact,
        tool_fact=args.tool_fact,
        attachment_fact=args.attachment_fact,
        attachment_id=args.attachment_id,
        model_base_url=args.model_base_url,
        context_limit=args.context_limit,
        output_path=Path(args.output),
        phase_label=args.phase_label,
        query=not args.skip_query,
    )
    print(stable_json_dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
