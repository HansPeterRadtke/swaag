from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from swaag.config import AgentConfig, load_config
from swaag.runtime import AgentRuntime
from swaag.types import Message
from swaag.utils import stable_json_dumps, utc_now_iso


PRESERVATION_FACTS = {
    "date": "2041-09-17T14:35:00Z",
    "identifier": "asset-R7K-4419",
    "user_constraint": "deliver one UTF-8 CSV and one Markdown explanation",
    "negative_constraint": "never modify the source ledger",
    "causality": "the checksum failed because the upstream export omitted row 73",
    "unresolved_question": "whether the supplier will restore row 73",
    "promise": "re-run the verifier after the supplier responds",
    "file_path": "/srv/project/evidence/final-ledger.csv",
    "tool_outcome": "verifier exit code 2 with checksum mismatch",
    "completion_state": "the task is not complete until the verifier exits 0",
}


@dataclass(slots=True, frozen=True)
class CompactionPreservationResult:
    cycle: int
    passed: bool
    missing_fact_names: list[str]
    message_count_before: int
    message_count_after: int
    summary_event_sequence: int
    source_reference_count: int
    context_compilation_sequence: int
    context_accounting: dict[str, Any]
    required_recovery_tokens: int = 0
    target_summary_tokens: int = 0
    actual_recovered_tokens: int = 0


def _fact_message() -> str:
    rows = [
        "Authoritative task facts. Preserve every exact value and relationship:",
        *[f"- {name}: {value}" for name, value in PRESERVATION_FACTS.items()],
    ]
    return "\n".join(rows)


def _model_identity(runtime: AgentRuntime) -> Any:
    identity = getattr(runtime.client, "cache_identity", None)
    value = identity() if callable(identity) else type(runtime.client).__name__
    if not isinstance(value, dict):
        return value
    stable_keys = (
        "base_url",
        "completion_endpoint",
        "configured_model_identity",
        "model_alias",
        "model_file",
        "profile_name",
        "server_build_info",
        "local_server_process_sha256",
    )
    return {key: value.get(key) for key in stable_keys}


def _checkpoint(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(stable_json_dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_compaction_preservation_benchmark(
    *,
    config: AgentConfig | None = None,
    cycles: int = 3,
    output_path: Path | None = None,
    model_client: object | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    config = config or load_config()
    runtime = AgentRuntime(config, model_client=model_client)
    identity = _model_identity(runtime)
    context_limit, context_limit_source = runtime._resolve_context_limit()
    previous: dict[str, Any] = {}
    report: dict[str, Any] = {}
    results: list[CompactionPreservationResult] = []
    session_id = ""

    if resume and output_path is not None and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        expected = {
            "benchmark": "history_compaction_preservation",
            "cycles_planned": cycles,
            "facts": PRESERVATION_FACTS,
            "model_identity": identity,
            "context_limit": context_limit,
            "context_limit_source": context_limit_source,
        }
        mismatches = {
            key: {"checkpoint": previous.get(key), "current": value}
            for key, value in expected.items()
            if previous.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Compaction-preservation checkpoint does not match the current run: "
                + stable_json_dumps(mismatches, indent=None)
            )
        session_id = str(previous.get("session_id", ""))
        raw_results = previous.get("results", [])
        if not session_id or not isinstance(raw_results, list):
            raise ValueError("Compaction-preservation checkpoint is incomplete")
        results = [CompactionPreservationResult(**row) for row in raw_results]
        if len(results) > cycles or [result.cycle for result in results] != list(
            range(1, len(results) + 1)
        ):
            raise ValueError(
                "Compaction-preservation checkpoint has invalid cycle results"
            )
        report = dict(previous)

    state = runtime.create_or_load_session(session_id or None)
    if not results:
        runtime._record_message(
            state,
            Message(role="user", content=_fact_message(), created_at=utc_now_iso()),
        )
        runtime._record_message(
            state,
            Message(
                role="assistant",
                content="I will preserve the authoritative facts while investigating.",
                created_at=utc_now_iso(),
            ),
        )
        runtime._record_message(
            state,
            Message(
                role="user",
                content="Continue carefully; ordinary progress text must not displace exact facts.",
                created_at=utc_now_iso(),
            ),
        )
        runtime._record_message(
            state,
            Message(role="assistant", content="Investigation is ongoing.", created_at=utc_now_iso()),
        )

    for cycle in range(len(results) + 1, cycles + 1):
        if cycle > 1:
            runtime._record_message(
                state,
                Message(
                    role="user",
                    content=f"Cycle {cycle} adds routine progress without changing the authoritative facts.",
                    created_at=utc_now_iso(),
                ),
            )
            runtime._record_message(
                state,
                Message(
                    role="assistant",
                    content=f"Routine cycle {cycle} progress acknowledged.",
                    created_at=utc_now_iso(),
                ),
            )
        before = len(state.messages)
        if not runtime._compact_once(state):
            raise RuntimeError(f"history compaction did not run for cycle {cycle}")
        after = len(state.messages)
        retained_text = "\n".join(message.content for message in state.messages)
        missing = [
            name for name, value in PRESERVATION_FACTS.items() if value not in retained_text
        ]
        events = runtime.history.read_history(state.session_id)
        summary_event = next(
            event for event in reversed(events) if event.event_type == "history_compressed"
        )
        compilation_event = next(
            event
            for event in reversed(events)
            if event.event_type == "context_compiled"
            and event.payload.get("kind") == "summary"
        )
        results.append(
            CompactionPreservationResult(
                cycle=cycle,
                passed=not missing,
                missing_fact_names=missing,
                message_count_before=before,
                message_count_after=after,
                summary_event_sequence=summary_event.sequence,
                source_reference_count=len(
                    summary_event.payload.get("source_event_references", [])
                ),
                context_compilation_sequence=compilation_event.sequence,
                context_accounting=dict(
                    compilation_event.payload.get("accounting", {})
                ),
                required_recovery_tokens=int(
                    summary_event.payload.get("required_recovery_tokens", 0)
                ),
                target_summary_tokens=int(
                    summary_event.payload.get("target_summary_tokens", 0)
                ),
                actual_recovered_tokens=int(
                    summary_event.payload.get("actual_recovered_tokens", 0)
                ),
            )
        )
        report = {
            "benchmark": "history_compaction_preservation",
            "model_identity": identity,
            "context_limit": context_limit,
            "context_limit_source": context_limit_source,
            "session_id": state.session_id,
            "facts": PRESERVATION_FACTS,
            "cycles_planned": cycles,
            "cycles_completed": len(results),
            "passed": sum(result.passed for result in results),
            "total": len(results),
            "complete": len(results) == cycles,
            "results": [asdict(result) for result in results],
        }
        _checkpoint(output_path, report)

    return report
