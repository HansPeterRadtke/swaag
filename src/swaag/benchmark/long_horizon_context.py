from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from swaag.benchmark.compaction_preservation import run_compaction_preservation_benchmark
from swaag.benchmark.context_engineering import run_context_engineering_benchmark
from swaag.config import AgentConfig, load_config
from swaag.redaction import configured_secret_values, redact_for_persistence
from swaag.runtime import AgentRuntime
from swaag.tools.base import _validate_schema_value
from swaag.types import ContractSpec, Message
from swaag.utils import sha256_text, stable_json_dumps, utc_now_iso


BENCHMARK_VERSION = 2

DELAYED_RELEVANCE_FACT = "handoff-token-R9-77124"


def _delayed_relevance_contract() -> ContractSpec:
    return ContractSpec(
        name="long_horizon_delayed_relevance",
        mode="json_schema",
        json_schema={
            "type": "object",
            "properties": {
                "handoff_token": {"type": "string", "enum": [DELAYED_RELEVANCE_FACT]}
            },
            "required": ["handoff_token"],
            "additionalProperties": False,
        },
    )


def _run_restart_delayed_relevance_trial(
    *,
    output_dir: Path,
    config: AgentConfig,
    model_client: object | None,
) -> dict[str, Any]:
    trial_config = deepcopy(config)
    trial_config.sessions.root = output_dir / "restart-delayed" / "sessions"
    runtime_a = AgentRuntime(trial_config, model_client=model_client)
    state_a = runtime_a.create_or_load_session()
    runtime_a._record_message(
        state_a,
        Message(
            role="user",
            content=(
                "Authoritative early handoff fact. Do not use it until a later explicit query. "
                f"handoff_token={DELAYED_RELEVANCE_FACT}"
            ),
            created_at=utc_now_iso(),
        ),
    )
    for index in range(8):
        runtime_a._record_message(
            state_a,
            Message(
                role="assistant" if index % 2 else "user",
                content=f"Unrelated phase-one progress {index + 1}: routine housekeeping only.",
                created_at=utc_now_iso(),
            ),
        )
    first_compaction = runtime_a._compact_once(state_a)
    session_id = state_a.session_id
    before_restart_events = runtime_a.history.read_history(session_id)
    early_event = next(
        event
        for event in before_restart_events
        if event.event_type == "message_added"
        and DELAYED_RELEVANCE_FACT in str(event.payload)
    )

    runtime_b = AgentRuntime(trial_config, model_client=model_client)
    state_b = runtime_b.create_or_load_session(session_id)
    explicit_rebuild = runtime_b.history.rebuild_from_history(
        session_id, prefer_checkpoint=False
    )
    restart_replay_matches = [
        (m.role, m.content, m.metadata) for m in state_b.messages
    ] == [
        (m.role, m.content, m.metadata) for m in explicit_rebuild.messages
    ]
    restart_event_count_matches = state_b.event_count == explicit_rebuild.event_count

    phase_two_messages: list[Message] = []
    for index in range(8):
        message = Message(
            role="assistant" if index % 2 else "user",
            content=f"Unrelated phase-two progress {index + 1}: routine housekeeping only.",
            created_at=utc_now_iso(),
        )
        phase_two_messages.append(message)
        runtime_b._record_message(state_b, message)
    second_compaction = runtime_b._compact_once(state_b)
    recent_source_text = "\n".join(message.content for message in phase_two_messages)
    no_recent_answer_leak = DELAYED_RELEVANCE_FACT not in recent_source_text
    retained_text = runtime_b.prompts.render_messages(state_b.messages)
    contract = _delayed_relevance_contract()
    error: dict[str, str] | None = None
    exact = False
    response_sha256: str | None = None
    try:
        completion = runtime_b.client.complete(
            (
                "This is the first query for an early authoritative fact after unrelated work "
                "and a fresh runtime reconstruction. Recover the exact handoff token from the retained state.\n\n"
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
        exact = payload.get("handoff_token") == DELAYED_RELEVANCE_FACT
        response_sha256 = sha256_text(completion.text)
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "reason": str(exc)}

    final_events = runtime_b.history.read_history(session_id)
    early_source_event_present = any(
        event.sequence == early_event.sequence
        and event.event_type == "message_added"
        and DELAYED_RELEVANCE_FACT in str(event.payload)
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
        int(ref.get("sequence", 0)) == early_event.sequence for ref in compression_refs
    ) or any(
        DELAYED_RELEVANCE_FACT in message.content for message in state_b.messages
    )
    passed = all(
        (
            first_compaction,
            second_compaction,
            restart_replay_matches,
            restart_event_count_matches,
            no_recent_answer_leak,
            early_source_event_present,
            early_lineage_present,
            exact,
        )
    )
    context_limit, context_limit_source = runtime_b._resolve_context_limit()
    report = {
        "passed": passed,
        "session_id": session_id,
        "first_compaction": first_compaction,
        "second_compaction": second_compaction,
        "restart_replay_matches": restart_replay_matches,
        "restart_event_count_matches": restart_event_count_matches,
        "no_recent_answer_leak": no_recent_answer_leak,
        "early_source_event_sequence": early_event.sequence,
        "early_source_event_present": early_source_event_present,
        "early_lineage_present": early_lineage_present,
        "model_identity": getattr(
            runtime_b.client, "cache_identity", lambda: type(runtime_b.client).__name__
        )(),
        "context_limit": context_limit,
        "context_limit_source": context_limit_source,
        "exact_delayed_retrieval": exact,
        "response_sha256": response_sha256,
        "error": error,
    }
    _atomic_report(output_dir / "restart_delayed_relevance.json", report, config=trial_config)
    return report



def _atomic_report(path: Path, payload: dict[str, Any], *, config: AgentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    sanitized = redact_for_persistence(payload, secret_values=configured_secret_values(config))
    raw = stable_json_dumps(sanitized, indent=2) + "\n"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, raw.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _overflow_pass(result: dict[str, Any]) -> bool:
    verification = result.get("verification", {})
    checks = verification.get("checks", {}) if isinstance(verification, dict) else {}
    required = (
        "candidate_overflow_measured",
        "semantic_projection_used",
        "projection_lineage_matches_source",
        "raw_source_recoverable",
        "required_facts_preserved",
        "final_request_fits",
    )
    return bool(verification.get("passed")) and all(bool(checks.get(name)) for name in required)


def run_long_horizon_context_benchmark(
    *,
    output_dir: Path,
    config: AgentConfig | None = None,
    cycles: int = 12,
    overflow_trials: int = 3,
    clean: bool = False,
    compaction_model_client: object | None = None,
    context_runtime_factory: Callable[[AgentConfig], AgentRuntime] = AgentRuntime,
    context_model_identity: Any | None = None,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    if overflow_trials <= 0:
        raise ValueError("overflow_trials must be positive")
    base = deepcopy(config or load_config())
    output_dir = output_dir.expanduser().resolve()
    if clean and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "long_horizon_context_results.json"

    signature = {
        "version": BENCHMARK_VERSION,
        "cycles": int(cycles),
        "overflow_trials": int(overflow_trials),
        "context_limit": int(base.model.context_limit),
        "model_base_url": str(base.model.base_url),
        "structured_output_mode": str(base.model.structured_output_mode),
    }
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError(
                "Long-horizon context checkpoint does not match the current benchmark configuration"
            )
        if previous.get("complete") is True:
            return previous

    compaction_path = output_dir / "compaction_preservation.json"
    compaction = run_compaction_preservation_benchmark(
        config=base,
        cycles=cycles,
        output_path=compaction_path,
        model_client=compaction_model_client,
        resume=True,
        adversarial_conflicts=True,
        semantic_retrieval_probe=True,
    )

    restart_delayed_relevance = _run_restart_delayed_relevance_trial(
        output_dir=output_dir,
        config=base,
        model_client=compaction_model_client,
    )

    overflow_reports: list[dict[str, Any]] = []
    for trial in range(1, overflow_trials + 1):
        trial_dir = output_dir / "overflow" / f"trial-{trial:03d}"
        report = run_context_engineering_benchmark(
            output_dir=trial_dir,
            config=base,
            case_ids=["measured_overflow_projection"],
            clean=False,
            runtime_factory=context_runtime_factory,
            model_identity=context_model_identity,
        )
        overflow_reports.append(report)
        partial = {
            "benchmark": "long_horizon_context",
            "signature": signature,
            "complete": False,
            "compaction": compaction,
            "restart_delayed_relevance": restart_delayed_relevance,
            "overflow_trials_completed": len(overflow_reports),
            "overflow_reports": overflow_reports,
        }
        _atomic_report(report_path, partial, config=base)

    compaction_rows = list(compaction.get("results", []))
    exact_preservation_passed = sum(bool(row.get("passed")) for row in compaction_rows)
    provenance_passed = sum(
        int(row.get("source_reference_count", 0)) > 0
        and int(row.get("required_recovery_tokens", 0)) > 0
        and int(row.get("actual_recovered_tokens", 0)) > 0
        for row in compaction_rows
    )
    semantic_retrieval_passed = sum(
        bool(row.get("semantic_retrieval_passed")) for row in compaction_rows
    )
    adversarial_resistance_passed = sum(
        bool(row.get("semantic_retrieval_passed")) for row in compaction_rows
    )
    overflow_rows = [
        result
        for report in overflow_reports
        for result in report.get("results", [])
        if result.get("case_id") == "measured_overflow_projection"
    ]
    overflow_passed = sum(_overflow_pass(row) for row in overflow_rows)
    complete = (
        bool(compaction.get("complete"))
        and bool(restart_delayed_relevance.get("passed"))
        and len(overflow_reports) == overflow_trials
        and all(report.get("complete") is True for report in overflow_reports)
    )
    all_dimensions_passed = bool(
        complete
        and exact_preservation_passed == cycles
        and provenance_passed == cycles
        and semantic_retrieval_passed == cycles
        and adversarial_resistance_passed == cycles
        and bool(restart_delayed_relevance.get("passed"))
        and overflow_passed == overflow_trials
    )
    aggregate = {
        "benchmark": "long_horizon_context",
        "signature": signature,
        "complete": complete,
        "all_dimensions_passed": all_dimensions_passed,
        "measurement_scope": {
            "exact_preservation": "Exact authoritative values survive repeated semantic compaction.",
            "provenance_recoverability": "Compacted state retains source references and enough recovery evidence to reconstruct durable originals.",
            "semantic_retrieval": "A separate constrained model probe recovers the exact authoritative values after every compaction cycle.",
            "adversarial_conflict_resistance": "Later explicitly untrusted contradictory values do not displace authoritative facts in semantic retrieval.",
            "restart_delayed_relevance": "An early fact remains exactly retrievable only after unrelated work, recursive compaction, a fresh runtime reconstruction boundary, and more unrelated work without recent answer leakage.",
            "measured_overflow_projection": "Independent trials prove actual context overflow, semantic projection, exact lineage, raw-source recovery, preserved required facts, and final fit.",
        },
        "dimensions": {
            "exact_preservation": {"passed": exact_preservation_passed, "total": cycles},
            "provenance_recoverability": {"passed": provenance_passed, "total": cycles},
            "semantic_retrieval": {"passed": semantic_retrieval_passed, "total": cycles},
            "adversarial_conflict_resistance": {
                "passed": adversarial_resistance_passed,
                "total": cycles,
            },
            "restart_delayed_relevance": {
                "passed": int(bool(restart_delayed_relevance.get("passed"))),
                "total": 1,
            },
            "measured_overflow_projection": {"passed": overflow_passed, "total": overflow_trials},
        },
        "compaction": compaction,
        "restart_delayed_relevance": restart_delayed_relevance,
        "overflow_trials_completed": len(overflow_reports),
        "overflow_reports": overflow_reports,
    }
    _atomic_report(report_path, aggregate, config=base)
    return aggregate
