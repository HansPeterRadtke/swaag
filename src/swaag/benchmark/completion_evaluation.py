from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from swaag.action import AgentAction, AgentStatus
from swaag.config import AgentConfig
from swaag.runtime import AgentRuntime
from swaag.types import Message
from swaag.utils import stable_json_dumps, utc_now_iso


@dataclass(slots=True, frozen=True)
class CompletionEvaluationCase:
    case_id: str
    objective: str
    candidate_message: str
    expected_complete: bool
    case_kind: str


CASES = (
    CompletionEvaluationCase(
        case_id="current_turn_failure_blocks_completion",
        objective=(
            "The task is complete only if the current verification tool succeeded. "
            "A current-turn tool failure means remaining work exists."
        ),
        candidate_message="Everything is verified and complete.",
        expected_complete=False,
        case_kind="tool_failure",
    ),
    CompletionEvaluationCase(
        case_id="exact_attachment_reexpanded_before_completion",
        objective=(
            "Verify completion only from the exact attached evidence. The final answer is valid only "
            "if the exact attachment proves marker ATTACH-LIVE-731."
        ),
        candidate_message="The attachment proves the required marker, so the task is complete.",
        expected_complete=True,
        case_kind="text_attachment",
    ),
    CompletionEvaluationCase(
        case_id="binary_attachment_requires_specialist",
        objective=(
            "Use the exact attached binary evidence before deciding completion. Do not treat an "
            "unreadable binary attachment as verified text evidence."
        ),
        candidate_message="The binary evidence is sufficient and the task is complete.",
        expected_complete=False,
        case_kind="binary_attachment",
    ),
    CompletionEvaluationCase(
        case_id="preemption_replay_without_success_is_incomplete",
        objective=(
            "A preempted request being exactly replayed is not itself proof of successful task "
            "completion. Report complete only if durable evidence also contains a verified final result."
        ),
        candidate_message="The request was replayed, so the task is complete.",
        expected_complete=False,
        case_kind="preemption_history",
    ),
)


def select_cases(case_ids: Iterable[str] = ()) -> list[CompletionEvaluationCase]:
    by_id = {case.case_id: case for case in CASES}
    requested = list(case_ids)
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError("Unknown completion-evaluation case(s): " + ", ".join(unknown))
    return [by_id[item] for item in requested] if requested else list(CASES)


def _candidate(message: str) -> AgentAction:
    return AgentAction(
        assistant_message=message,
        tool_calls=[],
        continue_loop=False,
        silent_completion=False,
        status=AgentStatus(
            "Candidate completion is ready for independent verification.",
            "Verify completion against durable evidence.",
            "The action model claims the task is complete.",
            "normal",
        ),
        questions=[],
    )


def _record_current_user(runtime: AgentRuntime, state, text: str) -> None:
    runtime._record_message(
        state,
        Message(role="user", content=text, created_at=utc_now_iso()),
    )


def _setup_case(runtime: AgentRuntime, state, case: CompletionEvaluationCase) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    if case.case_kind == "tool_failure":
        _record_current_user(runtime, state, case.objective)
        event = runtime.history.record_event(
            state,
            "tool_error",
            {
                "tool_name": "run_tests",
                "tool_input": {"command": ["verify"]},
                "error": "verification failed: assertion mismatch",
                "error_type": "verification_failed",
            },
        )
        runtime._record_message(
            state,
            Message(
                role="tool",
                name="run_tests",
                content="run_tests failed: verification failed: assertion mismatch",
                created_at=utc_now_iso(),
                metadata={
                    "output": {"passed": False, "exit_code": 1},
                    "source_event_sequence": event.sequence,
                    "source_event_hash": event.hash,
                    "source_event_type": event.event_type,
                    "source_event_session_id": state.session_id,
                    "source_event_references": [],
                },
            ),
        )
        evidence["failure_event_sequence"] = event.sequence
    elif case.case_kind == "text_attachment":
        reference = runtime.add_attachment(
            b"Authoritative verifier evidence: ATTACH-LIVE-731\n",
            original_name="completion-proof.txt",
            media_type="text/plain",
            session_id=state.session_id,
        )
        state = runtime.create_or_load_session(state.session_id)
        _record_current_user(runtime, state, case.objective)
        evidence["attachment_id"] = reference.attachment_id
        evidence["marker"] = "ATTACH-LIVE-731"
    elif case.case_kind == "binary_attachment":
        reference = runtime.add_attachment(
            b"\xff\xfe\x00\x89BINARY-LIVE-EVIDENCE-947",
            original_name="completion-proof.bin",
            media_type="application/octet-stream",
            session_id=state.session_id,
        )
        state = runtime.create_or_load_session(state.session_id)
        _record_current_user(runtime, state, case.objective)
        evidence["attachment_id"] = reference.attachment_id
    elif case.case_kind == "preemption_history":
        request_hash = "deadbeef" * 8
        preempted = runtime.history.record_event(
            state,
            "model_call_preempted",
            {
                "kind": "action",
                "prompt_mode": "standard",
                "attempt": 1,
                "call_id": "call-live-preempt",
                "preemption_id": "preempt-live-1",
                "request_sha256": request_hash,
                "usage_evidence": {
                    "backend_prompt_tokens": 321,
                    "backend_completion_tokens": 17,
                    "prompt_tokens_source": "tokens_evaluated",
                    "completion_tokens_source": "tokens_predicted",
                },
            },
        )
        replayed = runtime.history.record_event(
            state,
            "model_call_replayed",
            {
                "kind": "action",
                "call_id": "call-live-preempt",
                "request_sha256": request_hash,
                "preemption_id": "preempt-live-1",
                "request": {"prompt_sha256": request_hash, "contract": "agent_action"},
            },
        )
        _record_current_user(runtime, state, case.objective)
        evidence["preempted_sequence"] = preempted.sequence
        evidence["replayed_sequence"] = replayed.sequence
    else:
        raise AssertionError(case.case_kind)
    return {"state": state, **evidence}


def _score(case: CompletionEvaluationCase, result: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "expected_complete": bool(result.get("complete")) is case.expected_complete,
    }
    if case.case_kind == "tool_failure":
        references = result.get("evidence_source_references", [])
        checks["failure_reference_preserved"] = any(
            item.get("sequence") == evidence["failure_event_sequence"]
            and item.get("event_type") == "tool_error"
            for item in references
            if isinstance(item, dict)
        )
    elif case.case_kind == "text_attachment":
        rows = result.get("reexpanded_evidence_sources", [])
        checks["exact_attachment_reexpanded"] = any(
            row.get("source_id") == evidence["attachment_id"]
            and row.get("integrity_verified") is True
            for row in rows
            if isinstance(row, dict)
        )
        checks["no_specialist_required"] = result.get("specialist_evidence_required") is False
    elif case.case_kind == "binary_attachment":
        rows = result.get("reexpanded_evidence_sources", [])
        checks["specialist_gate_triggered"] = result.get("specialist_evidence_required") is True
        checks["binary_integrity_verified"] = any(
            row.get("source_id") == evidence["attachment_id"]
            and row.get("integrity_verified") is True
            and row.get("requires_specialist_analysis") is True
            and row.get("specialist_reason") == "non_utf8_attachment"
            for row in rows
            if isinstance(row, dict)
        )
    elif case.case_kind == "preemption_history":
        refs = result.get("historical_source_event_references", [])
        sequences = {
            int(item["sequence"])
            for item in refs
            if isinstance(item, dict) and isinstance(item.get("sequence"), int)
        }
        checks["preemption_and_replay_seen"] = {
            evidence["preempted_sequence"],
            evidence["replayed_sequence"],
        }.issubset(sequences)
    return {"passed": all(checks.values()), "checks": checks}


def run_completion_evaluation_benchmark(
    *,
    output_dir: Path,
    config: AgentConfig,
    case_ids: Iterable[str] = (),
    clean: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(case_ids)
    checkpoint = output_dir / "completion_evaluation_results.json"
    results: list[dict[str, Any]] = []
    if checkpoint.exists() and not clean:
        previous = json.loads(checkpoint.read_text(encoding="utf-8"))
        if previous.get("benchmark") != "completion-evaluation":
            raise ValueError("Completion-evaluation checkpoint has the wrong benchmark kind")
        if str(previous.get("model_base_url", "")) != str(config.model.base_url):
            raise ValueError("Completion-evaluation checkpoint model endpoint does not match this run")
        selected_ids = {case.case_id for case in cases}
        results = [
            row
            for row in previous.get("results", [])
            if isinstance(row, dict) and row.get("case_id") in selected_ids
        ]

    for index, case in enumerate(cases, 1):
        if any(row.get("case_id") == case.case_id for row in results):
            continue
        case_root = output_dir / "runs" / f"{index:02d}-{case.case_id}"
        case_config = copy.deepcopy(config)
        case_config.sessions.root = case_root / "sessions"
        case_config.model.cache_enabled = False
        case_config.runtime.completion_evaluation_enabled = True
        case_config.model.max_semantic_responsibilities_per_call = max(
            2, int(case_config.model.max_semantic_responsibilities_per_call)
        )
        workspace = case_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        case_config.tools.read_roots = [workspace]
        runtime = AgentRuntime(case_config)
        state = runtime.create_or_load_session()
        setup = _setup_case(runtime, state, case)
        state = setup.pop("state")
        started = time.time()
        error = ""
        result: dict[str, Any] = {}
        try:
            result = runtime._evaluate_completion(
                state,
                original_request=case.objective,
                selected_action=_candidate(case.candidate_message),
                tool_results=[],
            )
            verification = _score(case, result, setup)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            verification = {"passed": False, "checks": {"execution": False}}
        events = runtime.history.read_history(state.session_id)
        completion_calls = [
            event
            for event in events
            if event.event_type == "model_response_received"
            and event.payload.get("kind") == "completion_evaluation"
        ]
        item = {
            "case_id": case.case_id,
            "case_kind": case.case_kind,
            "expected_complete": case.expected_complete,
            "elapsed_seconds": time.time() - started,
            "result": result,
            "verification": verification,
            "model_calls": len(completion_calls),
            "error": error,
            "passed": bool(verification["passed"]) and not error,
        }
        results.append(item)
        report = {
            "benchmark": "completion-evaluation",
            "generated_at": utc_now_iso(),
            "complete": len(results) == len(cases),
            "total": len(cases),
            "completed": len(results),
            "passed": sum(bool(row["passed"]) for row in results),
            "model_base_url": config.model.base_url,
            "results": results,
        }
        checkpoint.write_text(stable_json_dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
