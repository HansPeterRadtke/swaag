from __future__ import annotations

import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from swaag.communication import CommunicationService
from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.utils import stable_json_dumps, utc_now_iso


def _normalize_objective(text: str) -> str:
    return " ".join(str(text).strip().rstrip(".?!").split()).casefold()


DEFAULT_PROMPT = """Create and start a durable orchestration plan now. Do not do the workers' work yourself. Use orchestration_control plan.apply to materialize the complete known graph in one tool call. The plan objective must be exactly: Complete ALPHA and then BETA using the durable dependency flow. Use exactly two workers and sequential scheduling. The first worker objective is: Return exactly ALPHA COMPLETE. Its finish criterion is that exact phrase. The second worker must depend on the first worker completing, receive the first worker result through the dependency input mapping, and its objective is: Return exactly BETA RECEIVED ALPHA COMPLETE. Its finish criterion is that exact phrase. Validate the plan before starting it, then start it. Use important reporting. After starting, tell me briefly that the plan has started and how many workers it contains."""


def _worker_records(service: CommunicationService, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    managers = {"default": service.workers, **service.worker_model_managers}
    records: list[dict[str, Any]] = []
    for node in snapshot["nodes"]:
        worker_id = node.get("worker_id")
        if not worker_id:
            continue
        model_key = str(node.get("model_key") or "default")
        manager = managers.get(model_key)
        if manager is None:
            continue
        record = manager.store.get(str(worker_id))
        records.append({"node_id": node["node_id"], **asdict(record)})
    return records


def run_live_orchestration_benchmark(
    *,
    output_dir: Path,
    base_url: str = "http://127.0.0.1:14829",
    timeout_seconds: float = 1800.0,
    prompt: str = DEFAULT_PROMPT,
    clean: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions_root = output_dir / "sessions"
    workspace_root = output_dir / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    config = load_config()
    config.sessions.root = sessions_root
    config.tools.read_roots = [workspace_root]
    config.model.base_url = base_url.rstrip("/")
    config.model.cache_enabled = False
    config.runtime.completion_evaluation_enabled = False
    # Keep live orchestration proof bounded. The default production action reserve is
    # intentionally generous; this benchmark's constrained steps need only a few
    # hundred observed output tokens, so retain several-fold headroom without asking
    # the local model to budget nearly ten thousand tokens per step.
    config.budget_policy.output_ratio_by_kind["action"] = 0.08
    config.budget_policy.output_floor_ratio_by_kind["action"] = 0.05
    config.tools.enabled = []
    config.tools.staged_discovery = False
    config.communication.enabled = True
    config.communication.model_base_url = base_url.rstrip("/")
    config.communication.enabled_tools = ["orchestration_control", "system_resources"]
    config.communication.model_routes = {}

    main = AgentRuntime(config)
    service = CommunicationService.from_runtime(main)
    started_epoch = time.time()
    error = ""
    orchestrator_reply: dict[str, str] | None = None
    final_snapshot: dict[str, Any] | None = None
    worker_records: list[dict[str, Any]] = []
    try:
        orchestrator_reply = service.orchestrator_message(prompt)
        plans = service.orchestration.store.list_plans()
        if len(plans) != 1:
            raise AssertionError(f"expected exactly one orchestration plan, found {len(plans)}")
        plan_id = plans[0].plan_id
        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        while True:
            final_snapshot = service.orchestration.advance(plan_id)
            states = {str(node["state"]) for node in final_snapshot["nodes"]}
            if final_snapshot["plan"].status == "completed":
                break
            if "failed" in states or "blocked" in states:
                raise AssertionError(f"orchestration entered non-success state: {sorted(states)}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"live orchestration did not complete within {timeout_seconds} seconds"
                )
            time.sleep(0.2)
        worker_records = _worker_records(service, final_snapshot)

        nodes = final_snapshot["nodes"]
        edges = final_snapshot["edges"]
        if len(nodes) != 2:
            raise AssertionError(f"expected two nodes, found {len(nodes)}")
        if len(edges) != 1:
            raise AssertionError(f"expected one dependency edge, found {len(edges)}")
        if final_snapshot["plan"].scheduling_mode != "sequential":
            raise AssertionError(
                f"expected sequential scheduling, got {final_snapshot['plan'].scheduling_mode}"
            )
        expected_objective = (
            "Complete ALPHA and then BETA using the durable dependency flow."
        )
        if _normalize_objective(final_snapshot["plan"].objective) != _normalize_objective(
            expected_objective
        ):
            raise AssertionError(
                "plan objective does not represent the requested overall outcome: "
                + repr(final_snapshot["plan"].objective)
            )
        if not all(node["state"] == "completed" for node in nodes):
            raise AssertionError(f"not all nodes completed: {[node['state'] for node in nodes]}")
        if len(worker_records) != 2:
            raise AssertionError(f"expected two durable workers, found {len(worker_records)}")
        first = worker_records[0]
        second = worker_records[1]
        if "ALPHA COMPLETE" not in str(first.get("result") or ""):
            raise AssertionError(f"first worker result missing marker: {first.get('result')!r}")
        if "ALPHA COMPLETE" not in str(second.get("objective") or ""):
            raise AssertionError("second worker objective did not receive first-worker result")
        if "BETA RECEIVED ALPHA COMPLETE" not in str(second.get("result") or ""):
            raise AssertionError(f"second worker result missing marker: {second.get('result')!r}")
        validation_events = [
            event
            for event in final_snapshot["events"]
            if event["event_type"] == "plan_status_changed"
            and event["payload"].get("to") == "active"
        ]
        if not validation_events:
            raise AssertionError("plan never transitioned from validated draft to active")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        service.workers.shutdown(wait=False)
        for manager in service.worker_model_managers.values():
            manager.shutdown(wait=False)

    elapsed = time.time() - started_epoch
    passed = not error
    report = {
        "benchmark": "orchestration-live",
        "generated_at": utc_now_iso(),
        "base_url": base_url.rstrip("/"),
        "workspace_root": str(workspace_root),
        "prompt": prompt,
        "passed": passed,
        "error": error,
        "elapsed_seconds": elapsed,
        "orchestrator_reply": orchestrator_reply,
        "plan": (
            asdict(final_snapshot["plan"])
            if final_snapshot is not None
            else None
        ),
        "nodes": final_snapshot["nodes"] if final_snapshot is not None else [],
        "edges": final_snapshot["edges"] if final_snapshot is not None else [],
        "worker_records": worker_records,
        "event_count": len(final_snapshot["events"]) if final_snapshot is not None else 0,
    }
    (output_dir / "orchestration_live_results.json").write_text(
        stable_json_dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
