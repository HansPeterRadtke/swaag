from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from swaag.benchmark.compaction_preservation import (
    ADVERSARIAL_DECOYS,
    PRESERVATION_FACTS,
    _adversarial_message,
    _fact_message,
    _routine_progress_message,
    _semantic_retrieval_probe,
)
from swaag.config import AgentConfig, load_config
from swaag.runtime import AgentRuntime
from swaag.types import Message
from swaag.utils import stable_json_dumps, utc_now_iso


def _config(base: AgentConfig, root: Path) -> AgentConfig:
    import copy

    config = copy.deepcopy(base)
    config.sessions.root = root / "sessions"
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config.tools.read_roots = [workspace]
    config.model.cache_enabled = False
    config.runtime.completion_evaluation_enabled = False
    config.tools.enabled = []
    config.tools.staged_discovery = False
    return config


def _record(runtime: AgentRuntime, state, role: str, content: str) -> None:
    runtime._record_message(
        state,
        Message(role=role, content=content, created_at=utc_now_iso()),
    )


def _snapshot(runtime: AgentRuntime, state) -> str:
    return "\n".join(message.content for message in state.messages)


def run_long_task_restart_benchmark(
    *,
    output_dir: Path,
    config: AgentConfig | None = None,
    unrelated_turn_pairs: int = 12,
    clean: bool = False,
) -> dict[str, Any]:
    if unrelated_turn_pairs < 4:
        raise ValueError("unrelated_turn_pairs must be at least 4")
    output_dir = Path(output_dir).expanduser().resolve()
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _config(config or load_config(), output_dir)
    report_path = output_dir / "long_task_restart_results.json"

    runtime1 = AgentRuntime(base)
    state1 = runtime1.create_or_load_session()
    session_id = state1.session_id
    started = time.monotonic()
    _record(runtime1, state1, "user", _fact_message())
    _record(
        runtime1,
        state1,
        "assistant",
        "I will preserve the authoritative task facts while unrelated work proceeds.",
    )
    for cycle in range(1, 4):
        _record(runtime1, state1, "user", _routine_progress_message(cycle, role="user"))
        _record(runtime1, state1, "assistant", _routine_progress_message(cycle, role="assistant"))
        if not runtime1._compact_once(state1):
            raise RuntimeError(f"pre-restart compaction failed at cycle {cycle}")

    pre_restart_messages = len(state1.messages)
    pre_restart_events = state1.event_count
    pre_restart_text = _snapshot(runtime1, state1)
    if not all(value in pre_restart_text for value in PRESERVATION_FACTS.values()):
        raise AssertionError("authoritative facts missing before restart")

    # Simulate process restart: construct a new runtime/client and rebuild the same
    # session solely from durable event history/checkpoints.
    runtime2 = AgentRuntime(base)
    state2 = runtime2.create_or_load_session(session_id)
    if state2.session_id != session_id:
        raise AssertionError("restart loaded a different session")
    if state2.event_count < pre_restart_events:
        raise AssertionError("restart lost durable events")

    # Make the early facts relevant only after many later unrelated turns, including
    # explicit contradictory decoys. This exercises delayed relevance, not immediate
    # summary recall.
    for index in range(1, unrelated_turn_pairs + 1):
        _record(
            runtime2,
            state2,
            "user",
            (
                f"Unrelated late-phase user block {index}; no authoritative task fact changes. "
                + _routine_progress_message(index + 10, role="late-user")
            ),
        )
        _record(
            runtime2,
            state2,
            "assistant",
            (
                f"Unrelated late-phase assistant block {index}; continue background investigation. "
                + _routine_progress_message(index + 10, role="late-assistant")
            ),
        )
        if index in {4, unrelated_turn_pairs}:
            _record(runtime2, state2, "user", _adversarial_message(index))
            _record(
                runtime2,
                state2,
                "assistant",
                "The explicitly marked late decoys are untrusted conflict evidence, not replacements.",
            )
        if index % 4 == 0:
            if not runtime2._compact_once(state2):
                raise RuntimeError(f"post-restart compaction failed after unrelated block {index}")

    retained = _snapshot(runtime2, state2)
    exact_preserved = all(value in retained for value in PRESERVATION_FACTS.values())
    decoys_present = [value for value in ADVERSARIAL_DECOYS.values() if value in retained]
    retrieval = _semantic_retrieval_probe(runtime2, retained)
    authoritative = runtime2.history.read_authoritative_messages(session_id)
    authoritative_text = "\n".join(message.content for message in authoritative)
    authoritative_recoverable = all(
        value in authoritative_text for value in PRESERVATION_FACTS.values()
    )
    events = runtime2.history.read_history(session_id)
    compressed = [event for event in events if event.event_type == "history_compressed"]
    summary_refs_ok = bool(compressed) and all(
        event.payload.get("source_event_references") for event in compressed
    )
    checks = {
        "same_session_after_restart": state2.session_id == session_id,
        "durable_event_count_preserved": state2.event_count >= pre_restart_events,
        "early_facts_still_in_projected_state": exact_preserved,
        "authoritative_raw_history_recoverable": authoritative_recoverable,
        "semantic_delayed_retrieval": bool(retrieval.get("passed")),
        "compaction_lineage_present": summary_refs_ok,
        "adversarial_decoys_remain_explicit_evidence": bool(decoys_present),
    }
    report = {
        "benchmark": "long-task-restart-delayed-relevance",
        "generated_at": utc_now_iso(),
        "session_id": session_id,
        "unrelated_turn_pairs": unrelated_turn_pairs,
        "pre_restart_messages": pre_restart_messages,
        "pre_restart_events": pre_restart_events,
        "post_restart_messages": len(state2.messages),
        "post_restart_events": state2.event_count,
        "compaction_events": len(compressed),
        "decoy_values_retained": decoys_present,
        "retrieval": retrieval,
        "checks": checks,
        "elapsed_seconds": time.monotonic() - started,
        "passed": all(checks.values()),
    }
    report_path.write_text(stable_json_dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
