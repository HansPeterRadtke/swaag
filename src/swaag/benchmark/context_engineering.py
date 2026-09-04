from __future__ import annotations

import copy
import json
import math
import re
import shutil
import time
import urllib.error

import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from swaag.config import AgentConfig, load_config
from swaag.grammar import agent_action_contract
from swaag.model import ModelClientError
from swaag.runtime import AgentRuntime
from swaag.types import Message
from swaag.utils import sha256_text, stable_json_dumps, utc_now_iso


REQUIRED_FACTS = (
    "change ticket CHG-7419-Z",
    "deadline 2042-06-19T15:40:00Z",
    "never delete the source archive",
    "the checksum failed because source row 812 was absent",
)

DISTRACTOR_MARKERS = (
    "DECOY-CERULEAN-114",
    "DECOY-AMBER-227",
    "DECOY-OLIVE-338",
    "DECOY-SCARLET-449",
)


@dataclass(slots=True, frozen=True)
class ContextEngineeringCase:
    case_id: str
    split: str
    force_overflow: bool


CASES = (
    ContextEngineeringCase(
        case_id="full_fidelity_fit",
        split="baseline",
        force_overflow=False,
    ),
    ContextEngineeringCase(
        case_id="measured_overflow_projection",
        split="held_out",
        force_overflow=True,
    ),
)


def select_cases(case_ids: Iterable[str] = ()) -> list[ContextEngineeringCase]:
    requested = list(case_ids)
    by_id = {case.case_id: case for case in CASES}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError("Unknown context-engineering case: " + ", ".join(unknown))
    return [by_id[case_id] for case_id in requested] if requested else list(CASES)


def _case_config(
    base: AgentConfig,
    *,
    sessions_root: Path,
    workspace: Path,
) -> AgentConfig:
    config = copy.deepcopy(base)
    config.sessions.root = sessions_root
    config.tools.read_roots = [workspace]
    config.model.cache_enabled = False
    config.tools.enabled = []
    config.tools.staged_discovery = False
    config.tools.allow_stateful_tools = False
    config.tools.allow_side_effect_tools = False
    config.runtime.completion_evaluation_enabled = False
    config.context.compact_on_overflow = True
    config.context.max_compaction_rounds = max(
        6, int(config.context.max_compaction_rounds)
    )
    return config


def _model_identity(runtime: AgentRuntime) -> Any:
    provider = getattr(runtime.client, "cache_identity", None)
    value = provider() if callable(provider) else type(runtime.client).__name__
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


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        stable_json_dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _objective() -> str:
    return (
        "Prepare the next action using the exact change ticket, deadline, negative "
        "constraint, and checksum causality from the diagnostic evidence. Routine "
        "healthy-record noise is irrelevant and may be reduced only if the measured "
        "request cannot fit."
    )


def _positioned_source(filler_rows: list[str]) -> str:
    boundaries = tuple(
        index * len(filler_rows) // len(REQUIRED_FACTS)
        for index in range(len(REQUIRED_FACTS) + 1)
    )
    rows = ["Complete diagnostic evidence follows."]
    for index, fact in enumerate(REQUIRED_FACTS):
        rows.extend(filler_rows[boundaries[index] : boundaries[index + 1]])
        rows.append(f"OBJECTIVE_RELEVANT_FACT: {fact}")
        rows.append(f"IRRELEVANT_REFERENCE: {DISTRACTOR_MARKERS[index]}")
    return "\n".join(rows)


def _source_text(
    runtime: AgentRuntime,
    state,
    *,
    force_overflow: bool,
    context_limit: int,
) -> str:
    if not force_overflow:
        return _positioned_source(
            [f"routine healthy record {index:03d}" for index in range(12)]
        )

    counter = runtime._counter(state)
    probe_rows = [
        f"routine healthy record {index:05d} contains no requested incident fact"
        for index in range(128)
    ]
    probe = "\n".join(probe_rows)
    probe_tokens = max(1, counter.count_text(probe).tokens)
    target_tokens = max(context_limit + 512, math.ceil(context_limit * 1.25))
    row_count = max(256, math.ceil(target_tokens * len(probe_rows) / probe_tokens))
    filler_rows = [
        f"routine healthy record {index:05d} contains no requested incident fact"
        for index in range(row_count)
    ]
    source = _positioned_source(filler_rows)
    while counter.count_text(source).tokens < target_tokens:
        start = len(filler_rows)
        filler_rows.extend(
            f"routine healthy record {index:05d} contains no requested incident fact"
            for index in range(start, start + max(128, start // 4))
        )
        source = _positioned_source(filler_rows)
    return source


def _seed_source(
    runtime: AgentRuntime,
    state,
    *,
    objective: str,
    source_text: str,
):
    runtime._record_message(
        state,
        Message(role="user", content=objective, created_at=utc_now_iso()),
    )
    source_event = runtime.history.record_event(
        state,
        "tool_result",
        {
            "tool_name": "diagnostic_reader",
            "raw_input": {"source": "complete-diagnostics"},
            "validated_input": {"source": "complete-diagnostics"},
            "output": {"text": source_text},
        },
    )
    runtime._record_message(
        state,
        Message(
            role="tool",
            name="diagnostic_reader",
            content=source_text,
            created_at=utc_now_iso(),
            metadata={
                "source_event_sequence": source_event.sequence,
                "source_event_hash": source_event.hash,
                "source_event_type": source_event.event_type,
                "source_event_references": [],
            },
        ),
    )
    return source_event


def _semantic_normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _fact_is_preserved(fact: str, text: str) -> bool:
    return _semantic_normalize(fact) in _semantic_normalize(text)


def _verify_case(
    case: ContextEngineeringCase,
    *,
    source_text: str,
    source_event,
    final_prompt: str,
    events: list,
) -> dict[str, Any]:
    action_compilations = [
        event
        for event in events
        if event.event_type == "context_compiled"
        and event.payload.get("kind") == "action"
    ]
    projection_events = [
        event for event in events if event.event_type == "tool_result_projected"
    ]
    raw_event = next(
        (event for event in events if event.sequence == source_event.sequence),
        None,
    )
    first_accounting = (
        dict(action_compilations[0].payload.get("accounting", {}))
        if action_compilations
        else {}
    )
    final_accounting = (
        dict(action_compilations[-1].payload.get("accounting", {}))
        if action_compilations
        else {}
    )
    first_components = first_accounting.get("components", [])
    input_component_tokens = sum(
        int(item.get("tokens", 0))
        for item in first_components
        if isinstance(item, dict) and item.get("include_in_context") is True
    )
    relevant_preserved = all(
        _fact_is_preserved(fact, final_prompt) for fact in REQUIRED_FACTS
    )
    distractors_retained = sum(
        marker in final_prompt for marker in DISTRACTOR_MARKERS
    )
    lineage_matches = bool(projection_events) and all(
        event.payload.get("source_event_sequence") == source_event.sequence
        and event.payload.get("source_event_hash") == source_event.hash
        for event in projection_events
    )
    raw_recoverable = (
        raw_event is not None
        and raw_event.payload.get("output", {}).get("text") == source_text
        and sha256_text(source_text)
        == sha256_text(str(raw_event.payload.get("output", {}).get("text", "")))
    )
    common = {
        "required_facts_preserved": relevant_preserved,
        "raw_source_recoverable": raw_recoverable,
        "final_request_fits": bool(final_accounting.get("fits")),
        "component_accounting_reconstructs_input": bool(first_components)
        and input_component_tokens == first_accounting.get("input_tokens"),
        "exact_source_reference_visible": (
            f"SOURCE EVENT sequence={source_event.sequence} hash={source_event.hash}"
            in final_prompt
        ),
    }
    if case.force_overflow:
        projection_payload = projection_events[-1].payload if projection_events else {}
        checks = {
            **common,
            "candidate_overflow_measured": int(first_accounting.get("overflow_tokens", 0)) > 0,
            "semantic_projection_used": bool(projection_events),
            "projection_lineage_matches_source": lineage_matches,
            "projection_actually_reduced_source": (
                int(projection_payload.get("projected_tokens", 0)) > 0
                and int(projection_payload.get("projected_tokens", 0))
                < int(projection_payload.get("original_tokens", 0))
            ),
            "projection_target_is_dynamic_reduction": (
                int(projection_payload.get("target_tokens", 0)) > 0
                and int(projection_payload.get("target_tokens", 0))
                < int(projection_payload.get("original_tokens", 0))
            ),
            "exact_recovery_guidance_visible": "history_window" in final_prompt,
        }
    else:
        checks = {
            **common,
            "candidate_fit_measured": int(first_accounting.get("overflow_tokens", -1)) == 0,
            "no_preemptive_projection": not projection_events,
            "full_source_retained_when_fit": source_text in final_prompt,
            "irrelevant_material_not_preemptively_dropped": (
                distractors_retained == len(DISTRACTOR_MARKERS)
            ),
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "first_action_accounting": first_accounting,
        "final_action_accounting": final_accounting,
        "projection_event_sequences": [event.sequence for event in projection_events],
        "distractor_markers_in_final_prompt": distractors_retained,
        "distractor_markers_removed": len(DISTRACTOR_MARKERS) - distractors_retained,
        "semantic_selectivity": {
            "retained": distractors_retained,
            "removed": len(DISTRACTOR_MARKERS) - distractors_retained,
            "total": len(DISTRACTOR_MARKERS),
        },
    }


def _is_model_unavailable_interruption(exc: BaseException) -> bool:
    if isinstance(exc, ModelClientError) and str(exc) == "model_unavailable":
        return True
    return isinstance(exc, (requests.ConnectionError, requests.Timeout, urllib.error.URLError))


def _build_report(
    *,
    selected_ids: list[str],
    model_identity: Any,
    results: list[dict[str, Any]],
    interrupted_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "benchmark": "context_engineering",
        "planned_cases": selected_ids,
        "model_identity": model_identity,
        "required_facts": list(REQUIRED_FACTS),
        "distractor_markers": list(DISTRACTOR_MARKERS),
        "results": results,
        "interrupted_attempts": interrupted_attempts,
        "total": len(selected_ids),
        "completed": len(results),
        "passed": sum(
            1 for item in results if item["verification"]["passed"]
        ),
        "complete": len(results) == len(selected_ids),
        "verification_scope": (
            "Exact marker, accounting, lineage, and recoverability checks; live semantic "
            "quality still requires model-backed repeated runs. Transient model-unavailable "
            "attempts are checkpointed as interruptions and are not scored as architecture failures."
        ),
    }


def run_context_engineering_benchmark(
    *,
    output_dir: Path,
    config: AgentConfig | None = None,
    case_ids: Iterable[str] = (),
    clean: bool = False,
    runtime_factory: Callable[[AgentConfig], AgentRuntime] = AgentRuntime,
    model_identity: Any | None = None,
) -> dict[str, Any]:
    base = config or load_config()
    selected = select_cases(case_ids)
    selected_ids = [case.case_id for case in selected]
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "context_engineering_results.json"
    results: list[dict[str, Any]] = []
    interrupted_attempts: list[dict[str, Any]] = []

    if report_path.exists():
        checkpoint = json.loads(report_path.read_text(encoding="utf-8"))
        if checkpoint.get("planned_cases") != selected_ids:
            raise ValueError("Context-engineering checkpoint case list does not match this run")
        raw_results = checkpoint.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("Context-engineering checkpoint results are invalid")
        results = [dict(item) for item in raw_results]
        raw_interrupted = checkpoint.get("interrupted_attempts", [])
        if not isinstance(raw_interrupted, list):
            raise ValueError("Context-engineering checkpoint interrupted attempts are invalid")
        interrupted_attempts = [dict(item) for item in raw_interrupted]
        checkpoint_identity = checkpoint.get("model_identity")
        if model_identity is not None and model_identity != checkpoint_identity:
            raise ValueError("Context-engineering checkpoint model identity does not match this run")
        if model_identity is None and len(results) == len(selected):
            probe_workspace = output_dir / "identity-probe" / "workspace"
            probe_workspace.mkdir(parents=True, exist_ok=True)
            probe_config = _case_config(
                base,
                sessions_root=output_dir / "identity-probe" / "sessions",
                workspace=probe_workspace,
            )
            model_identity = _model_identity(runtime_factory(probe_config))
            if model_identity != checkpoint_identity:
                raise ValueError("Context-engineering checkpoint model identity changed")
        elif model_identity is None:
            model_identity = checkpoint_identity

    for index, case in enumerate(selected, start=1):
        if any(item.get("case_id") == case.case_id for item in results):
            continue
        case_root = output_dir / "runs" / f"{index:02d}-{case.case_id}"
        if case_root.exists():
            attempt_number = 1
            while True:
                archived = case_root.with_name(
                    f"{case_root.name}-interrupted-{attempt_number:03d}"
                )
                if not archived.exists():
                    case_root.rename(archived)
                    break
                attempt_number += 1
        workspace = case_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=False)
        case_config = _case_config(
            base,
            sessions_root=case_root / "sessions",
            workspace=workspace,
        )
        runtime = runtime_factory(case_config)
        current_identity = _model_identity(runtime)
        if model_identity is None:
            model_identity = current_identity
        elif current_identity != model_identity:
            raise ValueError("Context-engineering runtime model identity changed")

        state = runtime.create_or_load_session()
        context_limit, context_limit_source = runtime._resolve_context_limit()
        source_text = _source_text(
            runtime,
            state,
            force_overflow=case.force_overflow,
            context_limit=context_limit,
        )
        objective = _objective()
        source_event = _seed_source(
            runtime,
            state,
            objective=objective,
            source_text=source_text,
        )
        started = time.monotonic()
        error: dict[str, str] | None = None
        interrupted_error: dict[str, str] | None = None
        final_prompt = ""
        try:
            prepared = runtime._prepare_action_call(
                state,
                original_request=objective,
                pending_messages=[],
                tool_specs=[],
                capability_index=None,
                contract=agent_action_contract([]),
                validation_feedback="",
                minimum_output_tokens=128,
            )
            final_prompt = prepared.assembly.prompt_text
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "reason": str(exc)}
            if _is_model_unavailable_interruption(exc):
                interrupted_error = error

        events = runtime.history.read_history(state.session_id)
        if interrupted_error is not None:
            interrupted_attempts.append(
                {
                    "case_id": case.case_id,
                    "split": case.split,
                    "session_id": state.session_id,
                    "elapsed_seconds": time.monotonic() - started,
                    "error": interrupted_error,
                    "context_limit": context_limit,
                    "context_limit_source": context_limit_source,
                    "source_event_sequence": source_event.sequence,
                    "source_event_hash": source_event.hash,
                    "last_event_sequence": events[-1].sequence if events else None,
                }
            )
            report = _build_report(
                selected_ids=selected_ids,
                model_identity=model_identity,
                results=results,
                interrupted_attempts=interrupted_attempts,
            )
            _write_report(report_path, report)
            return json.loads(report_path.read_text(encoding="utf-8"))
        verification = _verify_case(
            case,
            source_text=source_text,
            source_event=source_event,
            final_prompt=final_prompt,
            events=events,
        )
        if error is not None:
            verification = {
                **verification,
                "passed": False,
                "execution_error": error,
            }
        results.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "session_id": state.session_id,
                "elapsed_seconds": time.monotonic() - started,
                "context_limit": context_limit,
                "context_limit_source": context_limit_source,
                "source_event_sequence": source_event.sequence,
                "source_event_hash": source_event.hash,
                "source_sha256": sha256_text(source_text),
                "source_chars": len(source_text),
                "source_tokens": runtime._counter(state).count_text(source_text).tokens,
                "final_prompt_sha256": sha256_text(final_prompt),
                "context_compilations": [
                    {"sequence": event.sequence, **dict(event.payload)}
                    for event in events
                    if event.event_type == "context_compiled"
                ],
                "projection_events": [
                    {"sequence": event.sequence, **dict(event.payload)}
                    for event in events
                    if event.event_type == "tool_result_projected"
                ],
                "verification": verification,
            }
        )
        report = _build_report(
            selected_ids=selected_ids,
            model_identity=model_identity,
            results=results,
            interrupted_attempts=interrupted_attempts,
        )
        _write_report(report_path, report)

    return json.loads(report_path.read_text(encoding="utf-8"))
