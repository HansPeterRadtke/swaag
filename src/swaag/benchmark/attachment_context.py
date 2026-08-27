from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import time
from typing import Callable, Iterable

from swaag.config import AgentConfig, load_config
from swaag.model import LlamaCppClient
from swaag.runtime import AgentRuntime
from swaag.utils import stable_json_dumps


@dataclass(slots=True, frozen=True)
class AttachmentContextCase:
    case_id: str
    objective: str
    attachment_text: str
    chunk_chars: int
    expected_answer: str
    expected_read_policy: str
    require_projection: bool = False


def _sequential_attachment() -> str:
    blocks = [
        f"Section {index}: routine evidence line {index}. " * 8
        for index in range(1, 9)
    ]
    return "\n".join(
        [*blocks, "Authoritative final record: attestation-code=K7M-2041-ZETA"]
    )


def _partial_attachment() -> str:
    prefix = "\n".join(
        f"Preliminary record {index}: routine material without the decision code."
        for index in range(1, 9)
    )
    appendix = "\n".join(
        f"Appendix record {index}: unrelated material after the authoritative decision."
        for index in range(1, 25)
    )
    return (
        f"{prefix}\n"
        "Authoritative decision record: decision-code=DEC-482-NORTH\n"
        f"{appendix}\n"
    )


def _overflow_attachment() -> str:
    filler = "Routine telemetry record; no final attestation appears here.\n" * 1400
    return filler + "Authoritative final record: projection-code=PX9-771-OMEGA\n"


CASES = (
    AttachmentContextCase(
        case_id="no_unnecessary_inspection",
        objective=(
            "Reply with exactly READY. The attached reference is unrelated and must not be "
            "inspected because the answer is already fully specified."
        ),
        attachment_text="Unrelated archived material.\n",
        chunk_chars=256,
        expected_answer="READY",
        expected_read_policy="none",
    ),
    AttachmentContextCase(
        case_id="sequential_raw_reexpansion",
        objective=(
            "Inspect the attached UTF-8 evidence and report its exact authoritative final "
            "attestation-code. Do not guess or report a provisional value."
        ),
        attachment_text=_sequential_attachment(),
        chunk_chars=256,
        expected_answer="K7M-2041-ZETA",
        expected_read_policy="complete",
    ),
    AttachmentContextCase(
        case_id="partial_raw_inspection",
        objective=(
            "Inspect the attached UTF-8 evidence until you find the authoritative decision-code, "
            "then report that exact code without reading the unrelated appendix after it."
        ),
        attachment_text=_partial_attachment(),
        chunk_chars=256,
        expected_answer="DEC-482-NORTH",
        expected_read_policy="partial",
    ),
    AttachmentContextCase(
        case_id="oversized_result_projection",
        objective=(
            "Inspect the attached UTF-8 evidence and report its exact authoritative final "
            "projection-code. Preserve the raw source and use context-pressure recovery if needed."
        ),
        attachment_text=_overflow_attachment(),
        chunk_chars=100_000,
        expected_answer="PX9-771-OMEGA",
        expected_read_policy="some",
        require_projection=True,
    ),
)


def select_cases(case_ids: Iterable[str] = ()) -> list[AttachmentContextCase]:
    by_id = {case.case_id: case for case in CASES}
    requested = list(case_ids)
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown attachment-context case: {', '.join(unknown)}")
    return [by_id[case_id] for case_id in requested] if requested else list(CASES)


def verify_attachment_case(
    case: AttachmentContextCase,
    *,
    assistant_text: str,
    read_outputs: list[dict],
    projection_events: list[dict],
) -> dict:
    exact_lineage = all(
        bool(output.get("source_event_references"))
        and all(
            isinstance(reference, dict)
            and bool(reference.get("hash"))
            and reference.get("event_type") == "attachment_added"
            for reference in output.get("source_event_references", [])
        )
        for output in read_outputs
    )
    if case.expected_read_policy == "none":
        read_policy = not read_outputs
    elif case.expected_read_policy == "complete":
        read_policy = bool(read_outputs) and any(
            output.get("finished") is True for output in read_outputs
        )
    elif case.expected_read_policy == "partial":
        read_policy = bool(read_outputs) and not any(
            output.get("finished") is True for output in read_outputs
        )
    else:
        read_policy = bool(read_outputs)
    answer_present = (
        assistant_text.strip() == case.expected_answer
        if case.expected_read_policy == "none"
        else case.expected_answer in assistant_text
    )
    answer_observed_in_evidence = (
        True
        if case.expected_read_policy == "none"
        else any(
            case.expected_answer in str(output.get("text", ""))
            for output in read_outputs
        )
    )
    projection_lineage = all(
        isinstance(event.get("source_event_sequence"), int)
        and bool(event.get("source_event_hash"))
        for event in projection_events
    )
    checks = {
        "answer_present": answer_present,
        "answer_observed_in_evidence": answer_observed_in_evidence,
        "read_policy": read_policy,
        "exact_lineage": exact_lineage,
        "projection_observed": (
            bool(projection_events) if case.require_projection else True
        ),
        "projection_lineage": projection_lineage,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "read_count": len(read_outputs),
        "projection_count": len(projection_events),
    }


def _case_config(
    base: AgentConfig,
    *,
    workspace: Path,
    sessions_root: Path,
    case: AttachmentContextCase,
) -> AgentConfig:
    config = copy.deepcopy(base)
    config.sessions.root = sessions_root
    config.tools.read_roots = [workspace]
    config.tools.enabled = ["list_attachments", "read_attachment"]
    config.tools.allow_stateful_tools = True
    config.tools.allow_side_effect_tools = False
    config.tools.staged_discovery = True
    config.attachments.preview_chars = case.chunk_chars
    config.model.cache_enabled = False
    config.runtime.completion_evaluation_enabled = False
    return config


def _write_report(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        stable_json_dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_attachment_context_benchmark(
    *,
    output_dir: Path,
    config: AgentConfig | None = None,
    case_ids: Iterable[str] = (),
    clean: bool = False,
    runtime_factory: Callable[[AgentConfig], AgentRuntime] = AgentRuntime,
    model_identity: dict | None = None,
) -> dict:
    base = config or load_config()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not clean:
            raise FileExistsError(
                "Attachment-context output already exists; use --clean or a new path: "
                f"{output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    report_path = output_dir / "attachment_context_results.json"
    selected = select_cases(case_ids)
    if model_identity is None:
        model_identity = LlamaCppClient(base).cache_identity()
    results: list[dict] = []
    report = {
        "benchmark": "attachment_context_reexpansion",
        "status": "running",
        "model_identity": model_identity,
        "planned_cases": [case.case_id for case in selected],
        "results": results,
        "passed": 0,
        "total": 0,
    }
    _write_report(report_path, report)

    for index, case in enumerate(selected, start=1):
        case_root = output_dir / "runs" / f"{index:02d}-{case.case_id}"
        workspace = case_root / "workspace"
        workspace.mkdir(parents=True)
        case_config = _case_config(
            base,
            workspace=workspace,
            sessions_root=case_root / "sessions",
            case=case,
        )
        runtime = runtime_factory(case_config)
        state = runtime.create_or_load_session()
        reference = runtime.add_attachment(
            case.attachment_text.encode("utf-8"),
            original_name=f"{case.case_id}.txt",
            source="attachment_context_benchmark",
            session_id=state.session_id,
        )
        started = time.monotonic()
        assistant_text = ""
        error = ""
        try:
            turn = runtime.run_turn_in_session(state, case.objective)
            assistant_text = turn.assistant_text
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        events = runtime.history.read_history(state.session_id)
        read_outputs = [
            dict(event.payload.get("output", {}))
            for event in events
            if event.event_type == "tool_result"
            and event.payload.get("tool_name") == "read_attachment"
            and isinstance(event.payload.get("output"), dict)
        ]
        projection_events = [
            dict(event.payload)
            for event in events
            if event.event_type == "tool_result_projected"
            and event.payload.get("tool_name") == "read_attachment"
        ]
        verification = verify_attachment_case(
            case,
            assistant_text=assistant_text,
            read_outputs=read_outputs,
            projection_events=projection_events,
        )
        rebuilt = runtime.history.rebuild_from_history(
            state.session_id,
            prefer_checkpoint=False,
        )
        results.append(
            {
                "case_id": case.case_id,
                "session_id": state.session_id,
                "attachment_id": reference.attachment_id,
                "attachment_sha256": reference.sha256,
                "assistant_text": assistant_text,
                "error": error,
                "elapsed_seconds": elapsed,
                "verification": verification,
                "metrics": asdict(rebuilt.metrics),
            }
        )
        report = {
            "benchmark": "attachment_context_reexpansion",
            "status": "running" if index < len(selected) else "complete",
            "model_identity": model_identity,
            "planned_cases": [item.case_id for item in selected],
            "results": results,
            "passed": sum(
                1 for result in results if result["verification"]["passed"]
            ),
            "total": len(results),
        }
        _write_report(report_path, report)
    return report
