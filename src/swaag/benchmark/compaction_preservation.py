from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from swaag.config import AgentConfig, load_config
from swaag.runtime import AgentRuntime
from swaag.types import ContractSpec, Message
from swaag.tools.base import _validate_schema_value
from swaag.utils import sha256_text, stable_json_dumps, utc_now_iso


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


ADVERSARIAL_DECOYS = {
    "date": "1999-01-01T00:00:00Z",
    "identifier": "asset-DECOY-0000",
    "user_constraint": "deliver only a PDF",
    "negative_constraint": "modify the source ledger immediately",
    "file_path": "/tmp/decoy-ledger.csv",
    "completion_state": "the task is already complete",
}


def _adversarial_message(cycle: int) -> str:
    rows = [
        f"UNTRUSTED LATER DECOY cycle {cycle}. These values conflict with earlier authoritative facts and must not supersede them:",
        *[f"- {name}: {value}" for name, value in ADVERSARIAL_DECOYS.items()],
    ]
    return "\n".join(rows)


def _retrieval_contract() -> ContractSpec:
    properties = {
        name: {"type": "string", "enum": [value]}
        for name, value in PRESERVATION_FACTS.items()
    }
    return ContractSpec(
        name="long_horizon_authoritative_retrieval",
        mode="json_schema",
        json_schema={
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    )


def _semantic_retrieval_probe(runtime: AgentRuntime, retained_text: str) -> dict[str, Any]:
    contract = _retrieval_contract()
    prompt = (
        "Recover the exact authoritative task facts from the retained long-horizon state below. "
        "Later lines explicitly marked UNTRUSTED LATER DECOY are contradictory noise and must not override earlier authoritative facts.\n\n"
        + retained_text
    )
    try:
        completion = runtime.client.complete(
            prompt,
            max_tokens=512,
            contract=contract,
            temperature=0.0,
            kind="benchmark_quality_judge",
            live_mode=True,
        )
        payload = json.loads(completion.text)
        _validate_schema_value(payload, contract.json_schema or {}, path=contract.name)
        exact = all(payload.get(name) == value for name, value in PRESERVATION_FACTS.items())
        return {
            "attempted": True,
            "passed": exact,
            "response_sha256": sha256_text(completion.text),
            "error": None,
        }
    except Exception as exc:
        return {
            "attempted": True,
            "passed": False,
            "response_sha256": None,
            "error": {"error_type": type(exc).__name__, "reason": str(exc)},
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
    adversarial_conflicts_present: bool = False
    decoy_values_preserved: list[str] | None = None
    semantic_retrieval_attempted: bool = False
    semantic_retrieval_passed: bool = False
    semantic_retrieval_error: dict[str, str] | None = None


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
    adversarial_conflicts: bool = False,
    semantic_retrieval_probe: bool = False,
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
            "adversarial_conflicts": bool(adversarial_conflicts),
            "semantic_retrieval_probe": bool(semantic_retrieval_probe),
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
        if adversarial_conflicts:
            runtime._record_message(
                state,
                Message(
                    role="user",
                    content=_adversarial_message(cycle),
                    created_at=utc_now_iso(),
                ),
            )
            runtime._record_message(
                state,
                Message(
                    role="assistant",
                    content="I will treat the explicitly marked decoy values as conflicting noise, not authoritative replacements.",
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
        decoy_values_preserved = [
            value for value in ADVERSARIAL_DECOYS.values() if value in retained_text
        ] if adversarial_conflicts else []
        retrieval = (
            _semantic_retrieval_probe(runtime, retained_text)
            if semantic_retrieval_probe
            else {"attempted": False, "passed": False, "error": None}
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
                adversarial_conflicts_present=bool(adversarial_conflicts),
                decoy_values_preserved=decoy_values_preserved,
                semantic_retrieval_attempted=bool(retrieval["attempted"]),
                semantic_retrieval_passed=bool(retrieval["passed"]),
                semantic_retrieval_error=retrieval.get("error"),
            )
        )
        report = {
            "benchmark": "history_compaction_preservation",
            "model_identity": identity,
            "context_limit": context_limit,
            "context_limit_source": context_limit_source,
            "session_id": state.session_id,
            "facts": PRESERVATION_FACTS,
            "adversarial_decoys": ADVERSARIAL_DECOYS if adversarial_conflicts else {},
            "adversarial_conflicts": bool(adversarial_conflicts),
            "semantic_retrieval_probe": bool(semantic_retrieval_probe),
            "cycles_planned": cycles,
            "cycles_completed": len(results),
            "passed": sum(result.passed for result in results),
            "total": len(results),
            "semantic_retrieval_passed": sum(result.semantic_retrieval_passed for result in results),
            "semantic_retrieval_attempted": sum(result.semantic_retrieval_attempted for result in results),
            "cycles_with_decoy_values_retained": sum(bool(result.decoy_values_preserved) for result in results),
            "complete": len(results) == cycles,
            "results": [asdict(result) for result in results],
        }
        _checkpoint(output_path, report)

    return report
