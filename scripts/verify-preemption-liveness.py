from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any

from swaag.config import load_config
from swaag.inference import InferenceRequestCoordinator
from swaag.model import LlamaCppClient
from swaag.preemption import ModelCallPreempted, ModelPreemptionCoordinator
from swaag.utils import stable_json_dumps, utc_now_iso


def _check(rows: list[dict[str, Any]], name: str, passed: bool, **evidence: Any) -> None:
    rows.append({"name": name, "passed": bool(passed), "evidence": evidence})


def _live_cancel(base_url: str, root: Path) -> dict[str, Any]:
    config = load_config(
        env={
            "SWAAG__SESSIONS__ROOT": str(root / "cancel-sessions"),
            "SWAAG__MODEL__BASE_URL": base_url,
            "SWAAG__MODEL__CACHE_ENABLED": "false",
        }
    )
    client = LlamaCppClient(config)
    observed: dict[str, Any] = {
        "progress_events": 0,
        "completion_tokens": 0,
        "backend_prompt_tokens": None,
        "backend_completion_tokens": None,
        "first_token_seconds": None,
    }

    def on_progress(progress: dict[str, Any]) -> None:
        observed["progress_events"] += 1
        for key in (
            "completion_tokens",
            "backend_prompt_tokens",
            "backend_completion_tokens",
            "first_token_seconds",
            "tokens_per_second",
        ):
            if progress.get(key) is not None:
                observed[key] = progress.get(key)

    def cancel_check() -> bool:
        return int(observed.get("completion_tokens") or 0) >= 5

    payload = {
        "prompt": (
            "Write a long continuous technical explanation of durable agent scheduling, "
            "using many complete sentences so streaming continues until cancellation."
        ),
        "n_predict": 512,
        "temperature": 0.0,
    }
    started = time.monotonic()
    interrupted = False
    error = ""
    try:
        client.send_completion(
            payload,
            timeout_seconds=120,
            progress_callback=on_progress,
            cancel_check=cancel_check,
            cancel_poll_seconds=0.01,
        )
    except ModelCallPreempted as exc:
        interrupted = True
        error = str(exc)
    return {
        "interrupted": interrupted,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "error": error,
        **observed,
    }


def _durable_liveness(root: Path) -> dict[str, Any]:
    coord = InferenceRequestCoordinator(
        root / "inference",
        backend_key="verifier-backend",
        capacity_resolver=lambda: (1, "verifier"),
        poll_seconds=0.01,
        aging_seconds_per_priority=1.0,
        max_running_seconds=1.0,
    )
    first = coord.enqueue(
        session_id="stale-session",
        run_id="stale-run",
        call_id="stale-call",
        call_kind="action",
        priority=0,
        source="worker",
    )
    coord.acquire(first.request_id, timeout_seconds=1)
    with coord._connect() as connection:
        connection.execute(
            "UPDATE inference_requests SET started_at=?, updated_at=? WHERE request_id=?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", first.request_id),
        )
    second = coord.enqueue(
        session_id="next-session",
        run_id="next-run",
        call_id="next-call",
        call_kind="action",
        priority=0,
        source="worker",
    )
    admitted = coord.acquire(second.request_id, timeout_seconds=2)
    stale = coord.get(first.request_id)
    coord.complete(second.request_id)

    fresh = coord.enqueue(
        session_id="fresh-session",
        run_id="fresh-run",
        call_id="fresh-call",
        call_kind="action",
        priority=0,
        source="worker",
    )
    coord.acquire(fresh.request_id, timeout_seconds=1)
    with coord._connect() as connection:
        connection.execute(
            "UPDATE inference_requests SET started_at=?, updated_at=? WHERE request_id=?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", fresh.request_id),
        )
    touched = coord.touch_running(fresh.request_id)
    reconciled_fresh = coord.reconcile_orphans()
    fresh_after = coord.get(fresh.request_id)
    coord.complete(fresh.request_id)

    return {
        "stale_status": stale.status if stale else None,
        "stale_error": stale.error if stale else None,
        "next_admitted_status": admitted.status,
        "fresh_touch_status": touched.status if touched else None,
        "fresh_reconciled_ids": [item.request_id for item in reconciled_fresh],
        "fresh_after_status": fresh_after.status if fresh_after else None,
    }


def _target_change(root: Path) -> dict[str, Any]:
    coord = ModelPreemptionCoordinator(root / "preemption")
    request_payload = {"prompt": "exact frozen request", "n_predict": 64, "seed": 7}
    active = coord.register_active(
        "session-target",
        "call-target",
        "action",
        request_payload,
    )
    request = coord.request_preemption(
        "session-target",
        "redirect objective",
        source="verifier",
    )
    if request is None:
        raise RuntimeError("preemption request was not created")
    coord.mark_interrupted(request.preemption_id)
    coord.mark_assistant_running(request.preemption_id)
    coord.complete(
        request.preemption_id,
        target_changed=True,
        reply="target changed",
    )
    resolved = coord.get(request.preemption_id)
    return {
        "status": resolved.status if resolved else None,
        "target_changed": bool(resolved.target_changed) if resolved else None,
        "request_sha256": active.request_sha256,
        "preemption_id": request.preemption_id,
        "reply": resolved.reply if resolved else None,
    }


def _replay_session(root: Path) -> dict[str, Any]:
    root = root.resolve()
    def ro_connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(
            "file:" + urllib.parse.quote(str(path)) + "?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection

    history = ro_connect(root / "history.sqlite3")
    try:
        preempted_rows = history.execute(
            "SELECT sequence, payload_json FROM events WHERE event_type='model_call_preempted' ORDER BY sequence"
        ).fetchall()
        replayed_rows = history.execute(
            "SELECT sequence, payload_json FROM events WHERE event_type='model_call_replayed' ORDER BY sequence"
        ).fetchall()
        invalidated_rows = history.execute(
            "SELECT sequence, payload_json FROM events WHERE event_type='model_call_replay_invalidated' ORDER BY sequence"
        ).fetchall()
    finally:
        history.close()
    preempted = [json.loads(row["payload_json"]) for row in preempted_rows]
    replayed = [json.loads(row["payload_json"]) for row in replayed_rows]
    invalidated = [json.loads(row["payload_json"]) for row in invalidated_rows]

    inference = ro_connect(root / "inference_requests.sqlite3")
    try:
        inference_rows = [
            dict(row)
            for row in inference.execute(
                "SELECT call_kind, status, attempt_count, error FROM inference_requests ORDER BY queued_epoch"
            )
        ]
    finally:
        inference.close()
    preempted_hashes = [str(item.get("request_sha256", "")) for item in preempted]
    replayed_hashes = [str(item.get("request_sha256", "")) for item in replayed]
    completed_replay = any(
        row["call_kind"] == "action"
        and row["status"] == "completed"
        and int(row["attempt_count"]) >= 2
        for row in inference_rows
    )
    return {
        "preempted_hashes": preempted_hashes,
        "replayed_hashes": replayed_hashes,
        "invalidated_count": len(invalidated),
        "completed_replay": completed_replay,
        "inference_rows": inference_rows,
        "preempted_usage_evidence": [item.get("usage_evidence") for item in preempted],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--replay-session-root", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    root = output.parent / (output.stem + "-state")
    root.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    replay = _replay_session(Path(args.replay_session_root))
    _check(
        checks,
        "live_exact_preemption_replay",
        bool(replay["preempted_hashes"])
        and replay["preempted_hashes"] == replay["replayed_hashes"]
        and replay["invalidated_count"] == 0
        and replay["completed_replay"],
        **replay,
    )

    cancellation = _live_cancel(args.base_url, root)
    _check(
        checks,
        "live_cancellation_after_backend_progress",
        cancellation["interrupted"]
        and int(cancellation.get("completion_tokens") or 0) >= 5
        and int(cancellation.get("backend_completion_tokens") or 0) >= 5,
        **cancellation,
    )
    _check(
        checks,
        "live_partial_usage_preserved",
        cancellation.get("backend_prompt_tokens") is not None
        and cancellation.get("backend_completion_tokens") is not None,
        **cancellation,
    )

    durable = _durable_liveness(root)
    _check(
        checks,
        "stale_heartbeat_releases_capacity",
        durable["stale_status"] == "failed"
        and "liveness heartbeat stale" in str(durable["stale_error"])
        and durable["next_admitted_status"] == "running",
        **durable,
    )
    _check(
        checks,
        "fresh_heartbeat_survives_reconciliation",
        durable["fresh_touch_status"] == "running"
        and durable["fresh_reconciled_ids"] == []
        and durable["fresh_after_status"] == "running",
        **durable,
    )

    target = _target_change(root)
    _check(
        checks,
        "target_change_invalidates_frozen_request",
        target["status"] == "completed"
        and target["target_changed"] is True
        and bool(target["request_sha256"]),
        **target,
    )

    result = {
        "passed": all(item["passed"] for item in checks),
        "generated_at": utc_now_iso(),
        "base_url": args.base_url,
        "replay_session_root": str(Path(args.replay_session_root).resolve()),
        "checks": checks,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(stable_json_dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
