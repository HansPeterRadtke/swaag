from __future__ import annotations

import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from swaag.communication import CommunicationService
from swaag.config import AgentConfig, load_config
from swaag.runtime import AgentRuntime
from swaag.utils import stable_json_dumps, utc_now_iso


CASE_IDS = (
    "bounded_parallel_live",
    "semantic_branch_live",
    "dynamic_replan_replace_live",
    "completion_only_notifications_live",
)


def _config(base_url: str, root: Path) -> AgentConfig:
    config = load_config()
    config.sessions.root = root / "sessions"
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config.tools.read_roots = [workspace]
    config.model.base_url = base_url.rstrip("/")
    config.model.cache_enabled = False
    config.runtime.completion_evaluation_enabled = False
    config.tools.enabled = []
    config.tools.staged_discovery = False
    config.communication.enabled = True
    config.communication.model_base_url = base_url.rstrip("/")
    config.communication.enabled_tools = ["orchestration_control", "system_resources"]
    config.communication.model_routes = {}
    # Real worker objectives in this matrix need small final answers, not a huge
    # action reserve that would obscure scheduler behavior on local models.
    config.budget_policy.output_ratio_by_kind["action"] = 0.08
    config.budget_policy.output_floor_ratio_by_kind["action"] = 0.05
    return config


def _service(base_url: str, root: Path) -> CommunicationService:
    return CommunicationService.from_runtime(AgentRuntime(_config(base_url, root)))


def _shutdown(service: CommunicationService) -> None:
    service.workers.shutdown(wait=False)
    for manager in service.worker_model_managers.values():
        manager.shutdown(wait=False)


def _worker_records(service: CommunicationService, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    managers = {"default": service.workers, **service.worker_model_managers}
    rows: list[dict[str, Any]] = []
    for node in snapshot["nodes"]:
        worker_id = node.get("worker_id")
        if not worker_id:
            continue
        manager = managers[str(node.get("model_key") or "default")]
        rows.append({"node_id": node["node_id"], **asdict(manager.store.get(worker_id))})
    return rows


def _advance_until(
    service: CommunicationService,
    plan_id: str,
    *,
    predicate: Callable[[dict[str, Any]], bool],
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    while True:
        snapshot = service.orchestration.advance(plan_id)
        if predicate(snapshot):
            return snapshot
        states = {str(node["state"]) for node in snapshot["nodes"]}
        if "failed" in states or "blocked" in states:
            raise AssertionError(f"plan entered unexpected state(s): {sorted(states)}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"plan {plan_id} did not reach the requested live state")
        time.sleep(0.2)


def _complete(service: CommunicationService, plan_id: str, timeout_seconds: float) -> dict[str, Any]:
    return _advance_until(
        service,
        plan_id,
        predicate=lambda snap: snap["plan"].status == "completed",
        timeout_seconds=timeout_seconds,
    )


def _node(key: str, marker: str, *, priority: float = 1.0) -> dict[str, Any]:
    return {
        "key": key,
        "objective": f"Return a concise final answer containing the exact marker {marker}.",
        "priority": priority,
        "model_key": None,
        "finish_criteria": f"Final result contains {marker}.",
        "abort_criteria": None,
        "resources": {},
    }


def _case_bounded_parallel(service: CommunicationService, timeout_seconds: float) -> dict[str, Any]:
    applied = service.orchestration.apply_plan_spec(
        {
            "objective": "Complete three independent live workers with at most two admitted at once.",
            "scheduling_mode": "parallel",
            "max_parallel": 2,
            "reporting_mode": "important",
            "resource_limits": {},
            "nodes": [
                _node("a", "LIVE-PARALLEL-A"),
                _node("b", "LIVE-PARALLEL-B"),
                _node("c", "LIVE-PARALLEL-C"),
            ],
            "edges": [],
            "start": True,
        }
    )
    plan_id = applied["snapshot"]["plan"].plan_id
    initial = applied["snapshot"]
    initially_bound = [node for node in initial["nodes"] if node.get("worker_id")]
    if len(applied["started_worker_ids"]) != 2 or len(initially_bound) != 2:
        raise AssertionError("bounded-parallel plan did not admit exactly two workers initially")
    final = _complete(service, plan_id, timeout_seconds)
    records = _worker_records(service, final)
    text = "\n".join(str(row.get("result") or "") for row in records)
    for marker in ("LIVE-PARALLEL-A", "LIVE-PARALLEL-B", "LIVE-PARALLEL-C"):
        if marker not in text:
            raise AssertionError(f"missing live parallel worker marker: {marker}")
    return {
        "plan_id": plan_id,
        "initial_started": list(applied["started_worker_ids"]),
        "initial_bound_count": len(initially_bound),
        "final": final,
        "workers": records,
    }


def _case_semantic_branch(service: CommunicationService, timeout_seconds: float) -> dict[str, Any]:
    applied = service.orchestration.apply_plan_spec(
        {
            "objective": "Inspect live state and execute recovery only after an explicit semantic dependency decision.",
            "scheduling_mode": "parallel",
            "max_parallel": 2,
            "reporting_mode": "important",
            "resource_limits": {},
            "nodes": [
                _node("inspect", "LIVE-INSPECT-COMPLETE"),
                _node("repair", "LIVE-SEMANTIC-REPAIR"),
            ],
            "edges": [
                {
                    "source": "inspect",
                    "target": "repair",
                    "condition": {"when": "semantic", "question": "Does this benchmark require the live repair worker?"},
                    "input_mapping": {},
                }
            ],
            "start": True,
        }
    )
    plan_id = applied["snapshot"]["plan"].plan_id
    source_id = applied["node_ids"]["inspect"]
    target_id = applied["node_ids"]["repair"]
    before = _advance_until(
        service,
        plan_id,
        predicate=lambda snap: next(node for node in snap["nodes"] if node["node_id"] == source_id)["state"] == "completed",
        timeout_seconds=timeout_seconds,
    )
    target_before = next(node for node in before["nodes"] if node["node_id"] == target_id)
    if target_before["state"] != "pending" or target_before.get("worker_id"):
        raise AssertionError("semantic target ran before dependency resolution")
    edge = before["edges"][0]
    service.orchestration.store.resolve_dependency(
        plan_id,
        edge["edge_id"],
        satisfied=True,
        decision="Live benchmark explicitly requires the recovery branch.",
    )
    service.orchestration.start_ready(plan_id)
    final = _complete(service, plan_id, timeout_seconds)
    records = _worker_records(service, final)
    repair = next(row for row in records if row["node_id"] == target_id)
    if "LIVE-SEMANTIC-REPAIR" not in str(repair.get("result") or ""):
        raise AssertionError("semantic branch recovery worker did not complete with its marker")
    return {"plan_id": plan_id, "before_resolution": before, "final": final, "workers": records}


def _case_dynamic_replan(service: CommunicationService, timeout_seconds: float) -> dict[str, Any]:
    applied = service.orchestration.apply_plan_spec(
        {
            "objective": "Replace an active live worker and finish only the revised objective.",
            "scheduling_mode": "parallel",
            "max_parallel": 1,
            "reporting_mode": "important",
            "resource_limits": {},
            "nodes": [_node("mutable", "LIVE-OLD-WORKER")],
            "edges": [],
            "start": True,
        }
    )
    plan_id = applied["snapshot"]["plan"].plan_id
    node_id = applied["node_ids"]["mutable"]
    old_worker_id = applied["started_worker_ids"][0]
    service.orchestration.revise_node(
        plan_id,
        node_id,
        objective="Return a concise final answer containing the exact marker LIVE-REVISED-WORKER.",
        priority=4.0,
        finish_criteria="Final result contains LIVE-REVISED-WORKER.",
        replace_worker=True,
    )
    service.orchestration.start_ready(plan_id)
    final = _complete(service, plan_id, timeout_seconds)
    node = next(item for item in final["nodes"] if item["node_id"] == node_id)
    new_worker_id = str(node.get("worker_id") or "")
    if not new_worker_id or new_worker_id == old_worker_id:
        raise AssertionError("dynamic replan did not bind a replacement worker")
    old = service.workers.store.get(old_worker_id)
    new = service.workers.store.get(new_worker_id)
    if old.status != "canceled":
        raise AssertionError(f"replaced worker did not cancel: {old.status}")
    if "LIVE-REVISED-WORKER" not in str(new.result or ""):
        raise AssertionError("replacement worker did not complete revised objective")
    if float(new.inference_weight) != 4.0:
        raise AssertionError("replacement worker did not inherit revised weight")
    return {"plan_id": plan_id, "old_worker": asdict(old), "new_worker": asdict(new), "final": final}


def _case_completion_only(service: CommunicationService, timeout_seconds: float) -> dict[str, Any]:
    applied = service.orchestration.apply_plan_spec(
        {
            "objective": "Complete one live worker while suppressing routine notifications.",
            "scheduling_mode": "parallel",
            "max_parallel": 1,
            "reporting_mode": "completion_only",
            "resource_limits": {},
            "nodes": [_node("only", "LIVE-NOTIFY-COMPLETE")],
            "edges": [],
            "start": True,
        }
    )
    plan_id = applied["snapshot"]["plan"].plan_id
    # Routine/important non-completion messages are retained as events but not delivered.
    service.orchestration.store.record_notification(
        plan_id,
        "worker_progress",
        {"detail": "routine progress should be suppressed"},
        importance="important",
    )
    final = _complete(service, plan_id, timeout_seconds)
    notifications = service.orchestration.store.notifications(plan_id)
    kinds = [item["kind"] for item in notifications]
    if kinds != ["plan_completed"]:
        raise AssertionError(f"completion_only delivered unexpected notification kinds: {kinds}")
    return {"plan_id": plan_id, "notifications": notifications, "final": final}


CASE_RUNNERS: dict[str, Callable[[CommunicationService, float], dict[str, Any]]] = {
    "bounded_parallel_live": _case_bounded_parallel,
    "semantic_branch_live": _case_semantic_branch,
    "dynamic_replan_replace_live": _case_dynamic_replan,
    "completion_only_notifications_live": _case_completion_only,
}


def run_live_orchestration_matrix(
    *,
    output_dir: Path,
    base_url: str = "http://127.0.0.1:14829",
    timeout_seconds: float = 1800.0,
    case_ids: list[str] | None = None,
    clean: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = list(case_ids or CASE_IDS)
    unknown = sorted(set(requested) - set(CASE_RUNNERS))
    if unknown:
        raise ValueError("Unknown orchestration live matrix case: " + ", ".join(unknown))
    report_path = output_dir / "orchestration_live_matrix_results.json"
    results: list[dict[str, Any]] = []
    if report_path.exists() and not clean:
        import json
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if payload.get("planned_cases") != requested or payload.get("base_url") != base_url.rstrip("/"):
            raise ValueError("orchestration live matrix checkpoint does not match this run")
        results = [dict(row) for row in payload.get("results", [])]
    for index, case_id in enumerate(requested, 1):
        if any(row.get("case_id") == case_id for row in results):
            continue
        case_root = output_dir / "runs" / f"{index:02d}-{case_id}"
        service = _service(base_url, case_root)
        started = time.monotonic()
        error = ""
        evidence: dict[str, Any] = {}
        try:
            evidence = CASE_RUNNERS[case_id](service, timeout_seconds)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            _shutdown(service)
        results.append(
            {
                "case_id": case_id,
                "passed": not error,
                "elapsed_seconds": time.monotonic() - started,
                "error": error,
                "evidence": evidence,
            }
        )
        report = {
            "benchmark": "orchestration-live-matrix",
            "generated_at": utc_now_iso(),
            "base_url": base_url.rstrip("/"),
            "planned_cases": requested,
            "complete": len(results) == len(requested),
            "passed": sum(bool(row["passed"]) for row in results),
            "total": len(results),
            "results": results,
        }
        report_path.write_text(stable_json_dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
