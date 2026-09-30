from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from swaag.config import load_config
from swaag.grammar import evidence_projection_contract
from swaag.runtime import AgentRuntime, PreparedCall
from swaag.utils import stable_json_dumps, utc_now_iso


def _backend_progress(base_url: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/slots", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    for slot in payload:
        if not isinstance(slot, dict) or not bool(slot.get("is_processing")):
            continue
        decoded = 0
        next_token = slot.get("next_token")
        if isinstance(next_token, list) and next_token and isinstance(next_token[0], dict):
            decoded = int(next_token[0].get("n_decoded", 0) or 0)
        return {
            "task": slot.get("id_task"),
            "prompt_tokens": int(slot.get("n_prompt_tokens", 0) or 0),
            "prompt_tokens_processed": int(slot.get("n_prompt_tokens_processed", 0) or 0),
            "decoded_tokens": decoded,
        }
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    root = output.parent / (output.stem + "-state")
    if root.exists():
        import shutil
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(
        env={
            "SWAAG__SESSIONS__ROOT": str(root / "sessions"),
            "SWAAG__MODEL__BASE_URL": args.base_url,
            "SWAAG__MODEL__CACHE_ENABLED": "false",
            "SWAAG__RUNTIME__COMPLETION_EVALUATION_ENABLED": "false",
        }
    )
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    contract = evidence_projection_contract()
    source = (
        "Durable scheduling evidence. Keep explaining the same exact operational fact in detail. "
        * 260
    )
    assembly = runtime.prompts.build_evidence_projection_prompt(
        purpose=(
            "Return only a very short projection stating that durable scheduling evidence is present."
        ),
        source_label="live replay verifier source",
        raw_evidence=source,
        target_tokens=32,
    )
    compilation = runtime._compile_context(
        state,
        assembly,
        contract,
        minimum_output_tokens=32,
        desired_output_tokens=64,
    )
    if not compilation.report.fits:
        raise RuntimeError(f"live replay verifier prompt did not fit: {compilation.report}")
    prepared = PreparedCall(assembly, compilation.report, "lean", contract)
    holder: dict[str, Any] = {}

    def worker() -> None:
        try:
            holder["payload"] = runtime._execute_structured_call(state, prepared)
        except BaseException as exc:  # verifier captures the exact worker failure
            holder["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=worker, daemon=True)
    started = time.monotonic()
    thread.start()
    deadline = started + max(10.0, float(args.timeout_seconds))
    active = None
    sent_seen = False
    backend_progress = None
    while time.monotonic() < deadline:
        active = runtime.preemption.active_call(state.session_id)
        events = runtime.history.read_history(state.session_id)
        sent_seen = any(event.event_type == "model_request_sent" for event in events)
        backend_progress = _backend_progress(args.base_url)
        if (
            active is not None
            and sent_seen
            and backend_progress is not None
            and int(backend_progress.get("decoded_tokens", 0)) >= 1
        ):
            break
        if not thread.is_alive():
            break
        time.sleep(0.02)
    if active is None or not sent_seen or backend_progress is None or int(backend_progress.get("decoded_tokens", 0)) < 1:
        raise RuntimeError(
            "live verifier could not observe dispatched backend decode progress before completion"
        )

    preemption = runtime.preemption.request_preemption(
        state.session_id,
        "Live replay verifier communication interruption",
        source="live_replay_verifier",
    )
    if preemption is None:
        raise RuntimeError("failed to create preemption request")
    interrupted = runtime.preemption.wait_for_status(
        preemption.preemption_id,
        {"interrupted", "failed"},
        timeout_seconds=90.0,
        poll_seconds=0.01,
    )
    if interrupted.status != "interrupted":
        raise RuntimeError(f"preemption failed before replay: {interrupted.reply}")
    runtime.preemption.mark_assistant_running(preemption.preemption_id)
    runtime.preemption.complete(
        preemption.preemption_id,
        target_changed=False,
        reply="verifier communication completed",
    )
    thread.join(timeout=max(0.1, deadline - time.monotonic()))
    if thread.is_alive():
        raise TimeoutError("replayed live model call did not complete within verifier deadline")

    events = runtime.history.read_history(state.session_id)
    preempted = [e for e in events if e.event_type == "model_call_preempted"]
    replayed = [e for e in events if e.event_type == "model_call_replayed"]
    responses = [e for e in events if e.event_type == "model_response_received"]
    inference_rows = runtime.inference.list(session_id=state.session_id)
    hashes_match = bool(preempted and replayed) and (
        preempted[-1].payload.get("request_sha256")
        == replayed[-1].payload.get("request_sha256")
        == active.request_sha256
    )
    replay_request_matches = bool(replayed) and (
        stable_json_dumps(replayed[-1].payload.get("request", {}), indent=None)
        == active.request_json
    )
    inference_completed = any(
        item.call_kind == "evidence_projection"
        and item.status == "completed"
        and item.attempt_count >= 2
        for item in inference_rows
    )
    result = {
        "passed": (
            "error" not in holder
            and isinstance(holder.get("payload"), dict)
            and hashes_match
            and replay_request_matches
            and inference_completed
            and bool(responses)
        ),
        "generated_at": utc_now_iso(),
        "base_url": args.base_url,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "session_id": state.session_id,
        "preemption_id": preemption.preemption_id,
        "active_request_sha256": active.request_sha256,
        "preempted_request_sha256": (
            preempted[-1].payload.get("request_sha256") if preempted else None
        ),
        "replayed_request_sha256": (
            replayed[-1].payload.get("request_sha256") if replayed else None
        ),
        "hashes_match": hashes_match,
        "replay_request_matches_frozen_bytes": replay_request_matches,
        "model_request_sent_observed_before_preemption": sent_seen,
        "backend_progress_before_preemption": backend_progress,
        "inference_completed_after_replay": inference_completed,
        "inference_attempts": [
            {
                "request_id": item.request_id,
                "call_kind": item.call_kind,
                "status": item.status,
                "attempt_count": item.attempt_count,
            }
            for item in inference_rows
        ],
        "preempted_usage_evidence": (
            preempted[-1].payload.get("usage_evidence") if preempted else None
        ),
        "response_event_count": len(responses),
        "payload": holder.get("payload"),
        "error": holder.get("error", ""),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(stable_json_dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
