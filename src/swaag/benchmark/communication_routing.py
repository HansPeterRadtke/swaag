from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from swaag.communication import CommunicationService
from swaag.config import AgentConfig, load_config
from swaag.runtime import AgentRuntime
from swaag.types import HistoryEvent
from swaag.utils import stable_json_dumps, utc_now_iso


@dataclass(slots=True, frozen=True)
class CommunicationRoutingCase:
    case_id: str
    split: str
    question: str
    evidence: tuple[str, ...]
    expected_escalation: bool
    required_answer_markers: tuple[tuple[str, ...], ...]


CASES = (
    CommunicationRoutingCase(
        case_id="routine_liveness",
        split="baseline",
        question="Is the worker alive, and what is it doing right now?",
        evidence=(
            "The latest deterministic heartbeat is current and the worker is awaiting the next input.",
        ),
        expected_escalation=False,
        required_answer_markers=(("alive", "current", "responsive"), ("await", "input", "idle")),
    ),
    CommunicationRoutingCase(
        case_id="bounded_missing_verification",
        split="baseline",
        question="Have the tests passed? Answer only as far as the recorded evidence permits.",
        evidence=(
            "A test run was started, but no completion result or exit status has been recorded.",
        ),
        expected_escalation=False,
        required_answer_markers=(
            ("not established", "cannot confirm", "not known", "no completion"),
            ("test",),
        ),
    ),
    CommunicationRoutingCase(
        case_id="contradictory_rollout_evidence",
        split="held_out",
        question=(
            "Reconcile the rollout evidence and decide whether deployment can currently be "
            "claimed as complete. Explain the decisive evidence."
        ),
        evidence=(
            "An early operator note says the deployment command returned success.",
            "A later independent check found no listener on the registered localhost port.",
            "The live unit bytes differ from the repository source-of-truth unit.",
            "The latest verification step ended before service health was established.",
        ),
        expected_escalation=True,
        required_answer_markers=(
            (
                "not complete",
                "cannot claim",
                "cannot be claimed",
                "not deployed",
                "incomplete",
            ),
            ("listener", "port"),
            ("unit", "source"),
        ),
    ),
    CommunicationRoutingCase(
        case_id="high_impact_conflicting_safety_evidence",
        split="held_out",
        question=(
            "Does this evidence justify deleting the old production backup now? Explain the "
            "current risk and the next evidence needed."
        ),
        evidence=(
            "The replacement archive passed a checksum comparison against its local manifest.",
            "A restore rehearsal used a different archive identifier than the replacement archive.",
            "The remote copy has no recorded restore verification.",
            "Deletion of the old backup would be irreversible under the current retention policy.",
        ),
        expected_escalation=True,
        required_answer_markers=(
            ("do not delete", "not justify", "unsafe", "retain"),
            ("backup", "archive"),
            ("restore", "verification", "verify"),
        ),
    ),
)


def select_cases(case_ids: Iterable[str] = ()) -> list[CommunicationRoutingCase]:
    requested = list(case_ids)
    by_id = {case.case_id: case for case in CASES}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError("Unknown communication-routing case: " + ", ".join(unknown))
    return [by_id[case_id] for case_id in requested] if requested else list(CASES)


def _case_config(
    base: AgentConfig,
    *,
    sessions_root: Path,
    model_base_url: str,
) -> AgentConfig:
    config = copy.deepcopy(base)
    config.sessions.root = sessions_root
    config.model.base_url = model_base_url.rstrip("/")
    config.model.cache_enabled = False
    config.tools.enabled = []
    config.tools.allow_stateful_tools = False
    config.tools.allow_side_effect_tools = False
    config.runtime.completion_evaluation_enabled = False
    return config


def _model_identity(runtime: AgentRuntime) -> Any:
    provider = getattr(runtime.client, "cache_identity", None)
    return provider() if callable(provider) else type(runtime.client).__name__


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(stable_json_dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _seed_evidence(
    runtime: AgentRuntime,
    state,
    evidence: Iterable[str],
) -> list[HistoryEvent]:
    events: list[HistoryEvent] = []
    for content in evidence:
        events.append(
            runtime.history.record_event(
                state,
                "message_added",
                {
                    "message": {
                        "role": "user",
                        "content": content,
                        "created_at": utc_now_iso(),
                        "name": None,
                        "metadata": {"source": "communication-routing-benchmark"},
                    }
                },
            )
        )
    return events


def _all_history_events(runtime: AgentRuntime) -> list[HistoryEvent]:
    events = [
        event
        for entry in runtime.history.list_session_entries(include_internal=True)
        for event in runtime.history.read_history(str(entry["session_id"]))
    ]
    return sorted(events, key=lambda event: (event.timestamp, event.session_id, event.sequence))


def _status_from_operation(
    events: Iterable[HistoryEvent],
    operation_session_id: str,
) -> dict[str, Any]:
    for event in events:
        if (
            event.session_id == operation_session_id
            and event.event_type == "communication_status_generated"
        ):
            status = event.payload.get("status")
            return dict(status) if isinstance(status, dict) else {}
    return {}


def _verify_case(
    case: CommunicationRoutingCase,
    *,
    answer: str,
    escalated: bool,
    final_status: dict[str, Any],
    seeded_sequences: list[int],
) -> dict[str, Any]:
    lowered = answer.casefold()
    cited = {
        int(sequence) for sequence in final_status.get("evidence_sequences", [])
    }
    available = {
        int(reference["sequence"])
        for reference in final_status.get("source_event_references", [])
        if isinstance(reference, dict) and "sequence" in reference
    }
    seeded = set(seeded_sequences)
    checks = {
        "route_matches_oracle": escalated is case.expected_escalation,
        "answer_preserves_required_facts": all(
            any(marker.casefold() in lowered for marker in alternatives)
            for alternatives in case.required_answer_markers
        ),
        "answer_is_nonempty": bool(answer.strip()),
        "evidence_citation_is_valid": bool(cited & seeded)
        and cited.issubset(available),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_escalation": case.expected_escalation,
        "observed_escalation": escalated,
    }


def _report(
    *,
    model_identities: dict[str, Any] | None,
    selected_ids: list[str],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    positives = [item for item in results if item["expected_escalation"]]
    negatives = [item for item in results if not item["expected_escalation"]]
    total_prompt_tokens = sum(int(item.get("prompt_tokens") or 0) for item in results)
    total_completion_tokens = sum(
        int(item.get("completion_tokens") or 0) for item in results
    )
    return {
        "benchmark": "communication_model_routing",
        "status": "completed" if len(results) == len(selected_ids) else "running",
        "model_identities": model_identities,
        "planned_cases": selected_ids,
        "results": results,
        "passed": sum(bool(item["verification"]["passed"]) for item in results),
        "total": len(results),
        "routing_correct": sum(
            bool(item["verification"]["checks"]["route_matches_oracle"])
            for item in results
        ),
        "escalation_recall": (
            sum(bool(item["observed_escalation"]) for item in positives) / len(positives)
            if positives
            else None
        ),
        "non_escalation_specificity": (
            sum(not bool(item["observed_escalation"]) for item in negatives)
            / len(negatives)
            if negatives
            else None
        ),
        "mean_elapsed_seconds": (
            sum(float(item["elapsed_seconds"]) for item in results) / len(results)
            if results
            else 0.0
        ),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "complete": len(results) == len(selected_ids),
    }


def run_communication_routing_benchmark(
    *,
    output_dir: Path,
    config: AgentConfig | None = None,
    assistant_model_base_url: str | None = None,
    case_ids: Iterable[str] = (),
    clean: bool = False,
    runtime_factory: Callable[[AgentConfig], AgentRuntime] = AgentRuntime,
    model_identities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = config or load_config()
    selected = select_cases(case_ids)
    selected_ids = [case.case_id for case in selected]
    strong_url = base.model.base_url.rstrip("/")
    assistant_url = (
        assistant_model_base_url
        or base.communication.model_base_url
        or strong_url
    ).rstrip("/")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "communication_routing_results.json"
    results: list[dict[str, Any]] = []
    if report_path.exists():
        checkpoint = json.loads(report_path.read_text(encoding="utf-8"))
        if checkpoint.get("planned_cases") != selected_ids:
            raise ValueError("Communication-routing checkpoint case list changed")
        raw_results = checkpoint.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("Communication-routing checkpoint results are invalid")
        results = [dict(item) for item in raw_results]
        checkpoint_identities = checkpoint.get("model_identities")
        if model_identities is not None and model_identities != checkpoint_identities:
            raise ValueError("Communication-routing checkpoint model identity changed")
        if model_identities is None and len(results) == len(selected):
            probe_root = output_dir / "identity-probe" / "sessions"
            probe_main = runtime_factory(
                _case_config(
                    base,
                    sessions_root=probe_root,
                    model_base_url=strong_url,
                )
            )
            probe_assistant = runtime_factory(
                _case_config(
                    base,
                    sessions_root=probe_root,
                    model_base_url=assistant_url,
                )
            )
            model_identities = {
                "strong": _model_identity(probe_main),
                "assistant": _model_identity(probe_assistant),
                "distinct_endpoints": strong_url != assistant_url,
            }
            if model_identities != checkpoint_identities:
                raise ValueError(
                    "Communication-routing checkpoint model identity changed"
                )
        elif model_identities is None:
            model_identities = checkpoint_identities

    for index, case in enumerate(selected, start=1):
        if any(item.get("case_id") == case.case_id for item in results):
            continue
        case_root = output_dir / "runs" / f"{index:02d}-{case.case_id}"
        sessions_root = case_root / "sessions"
        main = runtime_factory(
            _case_config(base, sessions_root=sessions_root, model_base_url=strong_url)
        )
        assistant = runtime_factory(
            _case_config(base, sessions_root=sessions_root, model_base_url=assistant_url)
        )
        current_identities = {
            "strong": _model_identity(main),
            "assistant": _model_identity(assistant),
            "distinct_endpoints": strong_url != assistant_url,
        }
        if model_identities is None:
            model_identities = current_identities
        elif current_identities != model_identities:
            raise ValueError("Communication-routing runtime model identity changed")
        state = main.create_or_load_session()
        seeded_events = _seed_evidence(main, state, case.evidence)
        service = CommunicationService(main, assistant_runtime=assistant)
        started = time.monotonic()
        answer = ""
        error: dict[str, str] | None = None
        try:
            answer = service.answer_status_question(state.session_id, case.question)
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "reason": str(exc)}
        elapsed_seconds = time.monotonic() - started
        events = _all_history_events(main)
        requested = next(
            (
                event
                for event in events
                if event.event_type == "communication_status_escalation_requested"
            ),
            None,
        )
        resolved = next(
            (
                event
                for event in events
                if event.event_type == "communication_status_escalation_resolved"
            ),
            None,
        )
        assistant_generated = next(
            (
                event
                for event in events
                if event.event_type == "communication_status_generated"
            ),
            None,
        )
        assistant_operation_id = (
            str(requested.payload["status_operation_session_id"])
            if requested is not None
            else (assistant_generated.session_id if assistant_generated is not None else "")
        )
        final_operation_id = (
            str(resolved.payload["stronger_operation_session_id"])
            if resolved is not None
            else assistant_operation_id
        )
        assistant_status = _status_from_operation(events, assistant_operation_id)
        final_status = _status_from_operation(events, final_operation_id)
        completions = [
            event.payload.get("completion", {})
            for event in events
            if event.event_type == "model_response_received"
            and event.payload.get("kind") == "communication_status"
        ]
        verification = _verify_case(
            case,
            answer=answer,
            escalated=requested is not None,
            final_status=final_status,
            seeded_sequences=[event.sequence for event in seeded_events],
        )
        if error is not None:
            verification["passed"] = False
            verification["execution_error"] = error
        results.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "expected_escalation": case.expected_escalation,
                "observed_escalation": requested is not None,
                "target_session_id": state.session_id,
                "assistant_operation_session_id": assistant_operation_id,
                "final_operation_session_id": final_operation_id,
                "elapsed_seconds": elapsed_seconds,
                "answer": answer,
                "assistant_status": assistant_status,
                "final_status": final_status,
                "prompt_tokens": sum(
                    int(item.get("prompt_tokens") or 0)
                    for item in completions
                    if isinstance(item, dict)
                ),
                "completion_tokens": sum(
                    int(item.get("completion_tokens") or 0)
                    for item in completions
                    if isinstance(item, dict)
                ),
                "model_call_count": len(completions),
                "verification": verification,
                "error": error,
            }
        )
        _write_report(
            report_path,
            _report(
                model_identities=model_identities,
                selected_ids=selected_ids,
                results=results,
            ),
        )

    return _report(
        model_identities=model_identities,
        selected_ids=selected_ids,
        results=results,
    )
