from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from swaag.inference import InferenceRequestCoordinator
from swaag.orchestration import OrchestrationManager, OrchestrationStore
from swaag.orchestration_api import OrchestrationApi
from swaag.utils import stable_json_dumps, utc_now_iso


@dataclass(slots=True)
class _Worker:
    worker_id: str
    status: str = "created"
    result: str | None = None
    error: str | None = None
    inference_weight: float = 1.0


class _WorkerStore:
    def __init__(self) -> None:
        self.items: dict[str, _Worker] = {}

    def get(self, worker_id: str) -> _Worker:
        if worker_id not in self.items:
            raise FileNotFoundError(worker_id)
        return self.items[worker_id]

    def set_inference_weight(self, worker_id: str, weight: float) -> _Worker:
        item = replace(self.get(worker_id), inference_weight=float(weight))
        self.items[worker_id] = item
        return item


class _Workers:
    def __init__(self, label: str = "default") -> None:
        self.label = label
        self.store = _WorkerStore()
        self.created: list[dict[str, Any]] = []
        self.canceled: list[str] = []
        self.messages: list[dict[str, Any]] = []

    def create(
        self,
        objective: str,
        *,
        name: str | None = None,
        inference_weight: float = 1.0,
    ) -> _Worker:
        worker = _Worker(
            f"{self.label}-worker-{len(self.store.items) + 1}",
            inference_weight=float(inference_weight),
        )
        self.store.items[worker.worker_id] = worker
        self.created.append(
            {
                "worker_id": worker.worker_id,
                "objective": objective,
                "name": name,
                "inference_weight": worker.inference_weight,
            }
        )
        return worker

    def start(self, worker_id: str) -> _Worker:
        item = replace(self.store.get(worker_id), status="queued")
        self.store.items[worker_id] = item
        return item

    def cancel(self, worker_id: str, *, reason: str) -> _Worker:
        item = replace(self.store.get(worker_id), status="canceled")
        self.store.items[worker_id] = item
        self.canceled.append(worker_id)
        return item

    def message(
        self,
        worker_id: str,
        message: str,
        *,
        source: str,
        resume_if_idle: bool = True,
    ) -> _Worker:
        self.messages.append(
            {
                "worker_id": worker_id,
                "message": message,
                "source": source,
                "resume_if_idle": resume_if_idle,
            }
        )
        return self.store.get(worker_id)


def _manager(root: Path, *, routes: dict[str, _Workers] | None = None):
    default = _Workers()
    manager = OrchestrationManager(
        default,
        store=OrchestrationStore(root),
        worker_managers=routes,
    )
    return default, manager, OrchestrationApi(manager)


def _pass(case_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"case_id": case_id, "passed": True, "evidence": evidence, "error": ""}


def _case_bulk_plan_apply(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    result = api.execute(
        "plan.apply",
        {
            "plan_spec": {
                "objective": "complete alpha then beta",
                "scheduling_mode": "sequential",
                "max_parallel": 1,
                "reporting_mode": "important",
                "nodes": [
                    {
                        "key": "alpha",
                        "objective": "Return exactly ALPHA COMPLETE",
                        "finish_criteria": "ALPHA COMPLETE",
                        "priority": 1,
                    },
                    {
                        "key": "beta",
                        "objective": "Return exactly BETA RECEIVED ALPHA COMPLETE",
                        "finish_criteria": "BETA RECEIVED ALPHA COMPLETE",
                        "priority": 2,
                    },
                ],
                "edges": [
                    {
                        "source": "alpha",
                        "target": "beta",
                        "condition": {"when": "completed"},
                        "input_mapping": {"input": "alpha result"},
                    }
                ],
                "start": True,
            }
        },
    )
    assert result["validation"]["valid"] is True
    assert len(result["started_worker_ids"]) == 1
    assert set(result["node_ids"]) == {"alpha", "beta"}
    assert result["plan"]["status"] == "active"
    assert len(workers.created) == 1
    return _pass(
        "bulk_plan_apply",
        {
            "node_ids": result["node_ids"],
            "started_worker_ids": result["started_worker_ids"],
            "plan_status": result["plan"]["status"],
        },
    )


def _case_dependency_flow(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute("create", {"objective": "inspect then verify"})["plan"]["plan_id"]
    inspect = api.execute("node.add", {"plan_id": plan, "objective": "inspect"})["node_id"]
    verify = api.execute("node.add", {"plan_id": plan, "objective": "verify"})["node_id"]
    api.execute(
        "dependency.add",
        {
            "plan_id": plan,
            "source_node_id": inspect,
            "target_node_id": verify,
            "input_mapping": {"result": "inspection_evidence"},
        },
    )
    first = api.execute("start", {"plan_id": plan})["started_worker_ids"]
    assert len(first) == 1
    workers.store.items[first[0]] = replace(
        workers.store.items[first[0]], status="completed", result="EXACT-EVIDENCE-17"
    )
    snapshot = api.execute("advance", {"plan_id": plan})
    assert len(workers.created) == 2
    assert "EXACT-EVIDENCE-17" in workers.created[1]["objective"]
    assert "inspection_evidence" in workers.created[1]["objective"]
    states = {node["node_id"]: node["state"] for node in snapshot["nodes"]}
    assert states[inspect] == "completed" and states[verify] == "queued"
    return _pass("dependency_output_flow", {"states": states, "created": workers.created})


def _case_sequential(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute(
        "create", {"objective": "ordered", "scheduling_mode": "sequential"}
    )["plan"]["plan_id"]
    for name in ("one", "two", "three"):
        api.execute("node.add", {"plan_id": plan, "objective": name})
    started = api.execute("start", {"plan_id": plan})["started_worker_ids"]
    assert len(started) == 1 and len(workers.created) == 1
    api.execute("advance", {"plan_id": plan})
    assert len(workers.created) == 1
    workers.store.items[started[0]] = replace(
        workers.store.items[started[0]], status="completed", result="one done"
    )
    api.execute("advance", {"plan_id": plan})
    assert len(workers.created) == 2
    return _pass("sequential_exclusive", {"created": workers.created})


def _case_bounded_parallel(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute(
        "create",
        {"objective": "bounded", "scheduling_mode": "parallel", "max_parallel": 2},
    )["plan"]["plan_id"]
    for name in ("a", "b", "c", "d"):
        api.execute("node.add", {"plan_id": plan, "objective": name})
    started = api.execute("start", {"plan_id": plan})["started_worker_ids"]
    assert len(started) == 2
    api.execute("advance", {"plan_id": plan})
    assert len(workers.created) == 2
    workers.store.items[started[0]] = replace(
        workers.store.items[started[0]], status="completed", result="slot released"
    )
    api.execute("advance", {"plan_id": plan})
    assert len(workers.created) == 3
    return _pass("bounded_parallel", {"started": started, "created_count": len(workers.created)})


def _case_semantic_branch(root: Path) -> dict[str, Any]:
    _workers, _manager_obj, api = _manager(root)
    plan = api.execute("create", {"objective": "conditional"})["plan"]["plan_id"]
    source = api.execute("node.add", {"plan_id": plan, "objective": "inspect"})["node_id"]
    repair = api.execute("node.add", {"plan_id": plan, "objective": "repair"})["node_id"]
    edge = api.execute(
        "dependency.add",
        {
            "plan_id": plan,
            "source_node_id": source,
            "target_node_id": repair,
            "condition": {"when": "semantic", "question": "repair needed?"},
        },
    )["edge_id"]
    _manager_obj.store.set_node_state(plan, source, "completed")
    assert repair not in [n["node_id"] for n in _manager_obj.store.runnable_nodes(plan)]
    api.execute(
        "dependency.resolve",
        {
            "plan_id": plan,
            "edge_id": edge,
            "satisfied": True,
            "decision": "semantic inspection found a defect",
        },
    )
    assert repair in [n["node_id"] for n in _manager_obj.store.runnable_nodes(plan)]
    return _pass("semantic_branch_resolution", {"edge_id": edge})


def _case_failure_branch(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute("create", {"objective": "fallback"})["plan"]["plan_id"]
    primary = api.execute("node.add", {"plan_id": plan, "objective": "primary"})["node_id"]
    recovery = api.execute("node.add", {"plan_id": plan, "objective": "recovery"})["node_id"]
    api.execute(
        "dependency.add",
        {
            "plan_id": plan,
            "source_node_id": primary,
            "target_node_id": recovery,
            "condition": {"when": "failed"},
        },
    )
    started = api.execute("start", {"plan_id": plan})["started_worker_ids"]
    workers.store.items[started[0]] = replace(
        workers.store.items[started[0]], status="failed", error="mechanical failure"
    )
    api.execute("advance", {"plan_id": plan})
    assert len(workers.created) == 2 and workers.created[1]["objective"] == "recovery"
    return _pass("failure_branch", {"workers": workers.created})


def _case_dynamic_replan(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute("create", {"objective": "mutable"})["plan"]["plan_id"]
    node = api.execute("node.add", {"plan_id": plan, "objective": "old task"})["node_id"]
    first = api.execute("start", {"plan_id": plan})["started_worker_ids"][0]
    api.execute(
        "node.revise",
        {
            "plan_id": plan,
            "node_id": node,
            "objective": "new exact task",
            "priority": 4.0,
            "replace_worker": True,
        },
    )
    assert first in workers.canceled
    api.execute("advance", {"plan_id": plan})
    assert len(workers.created) == 2
    assert workers.created[-1]["objective"] == "new exact task"
    assert workers.created[-1]["inference_weight"] == 4.0
    return _pass("dynamic_replan_replace", {"canceled": workers.canceled, "created": workers.created})


def _case_model_routing(root: Path) -> dict[str, Any]:
    strong = _Workers("strong")
    default, _manager_obj, api = _manager(root, routes={"strong": strong})
    plan = api.execute("create", {"objective": "route"})["plan"]["plan_id"]
    api.execute(
        "node.add",
        {"plan_id": plan, "objective": "specialist", "model_key": "strong"},
    )
    started = api.execute("start", {"plan_id": plan})["started_worker_ids"]
    assert len(started) == 1 and default.created == [] and len(strong.created) == 1
    return _pass("named_model_routing", {"started": started, "strong": strong.created})


def _case_model_route_replacement(root: Path) -> dict[str, Any]:
    strong = _Workers("strong")
    default, _manager_obj, api = _manager(root, routes={"strong": strong})
    plan = api.execute("create", {"objective": "route change"})["plan"]["plan_id"]
    node = api.execute(
        "node.add", {"plan_id": plan, "objective": "work"}
    )["node_id"]
    api.execute("start", {"plan_id": plan})
    try:
        api.execute(
            "node.revise",
            {"plan_id": plan, "node_id": node, "model_key": "strong"},
        )
    except ValueError as exc:
        assert "requires replace_worker=true" in str(exc)
    else:
        raise AssertionError("model route changed without worker replacement")
    api.execute(
        "node.revise",
        {
            "plan_id": plan,
            "node_id": node,
            "model_key": "strong",
            "replace_worker": True,
        },
    )
    api.execute("advance", {"plan_id": plan})
    assert len(default.canceled) == 1 and len(strong.created) == 1
    return _pass(
        "model_route_replacement",
        {"default_canceled": default.canceled, "strong_created": strong.created},
    )


def _case_missing_model_fails_closed(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute("create", {"objective": "route"})["plan"]["plan_id"]
    api.execute(
        "node.add",
        {"plan_id": plan, "objective": "specialist", "model_key": "missing"},
    )
    validation = api.execute("validate", {"plan_id": plan})["validation"]
    assert validation["valid"] is False
    assert validation["issues"][0]["code"] == "model_assignment_unavailable"
    try:
        api.execute("start", {"plan_id": plan})
    except ValueError as exc:
        assert "model_assignment_unavailable" in str(exc)
    else:
        raise AssertionError("invalid model route plan started")
    assert workers.created == []
    snapshot = api.execute("get", {"plan_id": plan})
    delivered = api.execute("notifications", {"plan_id": plan})["notifications"]
    assert snapshot["plan"]["status"] == "draft"
    assert delivered[-1]["kind"] == "plan_validation_failed"
    return _pass(
        "missing_model_fails_closed",
        {"validation": validation, "status": snapshot["plan"]["status"]},
    )


def _case_empty_plan_validation(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute("create", {"objective": "empty"})["plan"]["plan_id"]
    validation = api.execute("validate", {"plan_id": plan})["validation"]
    assert validation["valid"] is False
    assert validation["issues"][0]["code"] == "empty_plan"
    try:
        api.execute("start", {"plan_id": plan})
    except ValueError as exc:
        assert "empty_plan" in str(exc)
    else:
        raise AssertionError("empty plan started")
    assert workers.created == []
    return _pass("empty_plan_validation", {"validation": validation})


def _case_cycle_guard(root: Path) -> dict[str, Any]:
    _workers, _manager_obj, api = _manager(root)
    plan = api.execute("create", {"objective": "acyclic"})["plan"]["plan_id"]
    ids = [
        api.execute("node.add", {"plan_id": plan, "objective": name})["node_id"]
        for name in ("a", "b", "c")
    ]
    api.execute("dependency.add", {"plan_id": plan, "source_node_id": ids[0], "target_node_id": ids[1]})
    api.execute("dependency.add", {"plan_id": plan, "source_node_id": ids[1], "target_node_id": ids[2]})
    try:
        api.execute("dependency.add", {"plan_id": plan, "source_node_id": ids[2], "target_node_id": ids[0]})
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cyclic orchestration graph was accepted")
    return _pass("cycle_guard", {"node_ids": ids})


def _case_plan_cancel(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute("create", {"objective": "cancel me"})["plan"]["plan_id"]
    api.execute("node.add", {"plan_id": plan, "objective": "active"})
    started = api.execute("start", {"plan_id": plan})["started_worker_ids"]
    result = api.execute("cancel", {"plan_id": plan, "reason": "user changed direction"})
    assert started[0] in workers.canceled and result["plan"]["status"] == "canceled"
    return _pass("plan_cancel", {"canceled": workers.canceled, "status": result["plan"]["status"]})


def _case_plan_completion(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute("create", {"objective": "finish"})["plan"]["plan_id"]
    api.execute("node.add", {"plan_id": plan, "objective": "only"})
    worker = api.execute("start", {"plan_id": plan})["started_worker_ids"][0]
    workers.store.items[worker] = replace(workers.store.items[worker], status="completed", result="done")
    result = api.execute("advance", {"plan_id": plan})
    assert result["plan"]["status"] == "completed"
    assert result["events"][-1]["payload"]["kind"] == "plan_completed"
    return _pass("plan_auto_completion", {"status": result["plan"]["status"]})


def _coordinator(root: Path, *, aging: float = 1000.0) -> InferenceRequestCoordinator:
    return InferenceRequestCoordinator(
        root,
        backend_key="benchmark-backend",
        capacity_resolver=lambda: (1, "benchmark"),
        poll_seconds=0.001,
        aging_seconds_per_priority=aging,
    )


def _enqueue(
    coordinator: InferenceRequestCoordinator,
    call_id: str,
    *,
    source: str,
    priority: int = 0,
    weight: float = 1.0,
):
    return coordinator.enqueue(
        session_id=f"session-{call_id}",
        run_id=f"run-{call_id}",
        call_id=call_id,
        call_kind="agent_action",
        priority=priority,
        source=source,
        fair_weight=weight,
    )


def _next_id(coordinator: InferenceRequestCoordinator) -> str:
    with coordinator._connect() as connection:
        value = coordinator._fair_candidate(connection, now_epoch=time.time())
    if value is None:
        raise AssertionError("scheduler had no candidate")
    return value


def _run_share(root: Path, weights: dict[str, float], turns: int) -> dict[str, int]:
    coordinator = _coordinator(root)
    pending: dict[str, Any] = {}
    sequence = 0
    for source, weight in weights.items():
        sequence += 1
        pending[source] = _enqueue(
            coordinator, f"{source}-{sequence}", source=source, weight=weight
        )
    counts = {source: 0 for source in weights}
    for _ in range(turns):
        request_id = _next_id(coordinator)
        request = coordinator.get(request_id)
        assert request is not None
        acquired = coordinator.acquire(request_id, timeout_seconds=1.0)
        counts[acquired.source] += 1
        coordinator.complete(request_id)
        sequence += 1
        pending[acquired.source] = _enqueue(
            coordinator,
            f"{acquired.source}-{sequence}",
            source=acquired.source,
            weight=weights[acquired.source],
        )
    return counts


def _case_resource_budget(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute(
        "create",
        {
            "objective": "resource bounded",
            "resource_limits": {"ram_gb": 8, "api_slots": 2},
        },
    )["plan"]["plan_id"]
    api.execute(
        "node.add",
        {
            "plan_id": plan,
            "objective": "large",
            "resources": {"ram_gb": 6, "api_slots": 1},
        },
    )
    api.execute(
        "node.add",
        {
            "plan_id": plan,
            "objective": "medium",
            "resources": {"ram_gb": 4, "api_slots": 1},
        },
    )
    started = api.execute("start", {"plan_id": plan})["started_worker_ids"]
    assert len(started) == 1
    assert len(workers.created) == 1
    workers.store.items[started[0]] = replace(
        workers.store.items[started[0]], status="completed", result="released resources"
    )
    api.execute("advance", {"plan_id": plan})
    assert len(workers.created) == 2
    snapshot = api.execute("get", {"plan_id": plan})
    assert snapshot["resource_limits"] == {"api_slots": 2.0, "ram_gb": 8.0}
    return _pass(
        "resource_budget_admission",
        {"resource_limits": snapshot["resource_limits"], "created": workers.created},
    )


def _case_impossible_resource(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute(
        "create",
        {"objective": "resource guard", "resource_limits": {"gpu_gb": 8}},
    )["plan"]["plan_id"]
    api.execute(
        "node.add",
        {
            "plan_id": plan,
            "objective": "too large",
            "resources": {"gpu_gb": 12},
        },
    )
    validation = api.execute("validate", {"plan_id": plan})["validation"]
    assert validation["valid"] is False
    assert validation["issues"][0]["code"] == "resource_limit_exceeded"
    try:
        api.execute("start", {"plan_id": plan})
    except ValueError as exc:
        assert "resource_limit_exceeded" in str(exc)
    else:
        raise AssertionError("resource-invalid plan started")
    assert workers.created == []
    snapshot = api.execute("get", {"plan_id": plan})
    delivered = api.execute("notifications", {"plan_id": plan})["notifications"]
    assert snapshot["plan"]["status"] == "draft"
    assert delivered[-1]["kind"] == "plan_validation_failed"
    return _pass(
        "impossible_resource_fails_closed",
        {"validation": validation, "status": snapshot["plan"]["status"]},
    )


def _case_finish_abort_criteria(root: Path) -> dict[str, Any]:
    workers, _manager_obj, api = _manager(root)
    plan = api.execute("create", {"objective": "criteria"})["plan"]["plan_id"]
    api.execute(
        "node.add",
        {
            "plan_id": plan,
            "objective": "work",
            "finish_criteria": "tests pass and artifact exists",
            "abort_criteria": "required database is unavailable",
        },
    )
    api.execute("start", {"plan_id": plan})
    objective = workers.created[0]["objective"]
    assert "Explicit finish criteria" in objective
    assert "tests pass and artifact exists" in objective
    assert "Explicit abort/block criteria" in objective
    assert "required database is unavailable" in objective
    return _pass("finish_abort_criteria", {"worker_objective": objective})


def _case_reporting_completion_only(root: Path) -> dict[str, Any]:
    _workers, manager, api = _manager(root)
    plan = api.execute(
        "create",
        {"objective": "quiet", "reporting_mode": "completion_only"},
    )["plan"]["plan_id"]
    manager.store.record_notification(
        plan,
        "worker_state",
        {"worker_id": "w", "state": "completed"},
        importance="normal",
    )
    assert api.execute("notifications", {"plan_id": plan})["notifications"] == []
    manager.store.record_notification(
        plan, "plan_completed", {"node_count": 1}, importance="normal"
    )
    delivered = api.execute("notifications", {"plan_id": plan})["notifications"]
    assert [item["kind"] for item in delivered] == ["plan_completed"]
    return _pass(
        "reporting_completion_only",
        {"delivered_kinds": [item["kind"] for item in delivered]},
    )


def _case_critical_reporting_bypass(root: Path) -> dict[str, Any]:
    _workers, manager, api = _manager(root)
    plan = api.execute(
        "create", {"objective": "manual", "reporting_mode": "manual"}
    )["plan"]["plan_id"]
    manager.store.record_notification(
        plan,
        "worker_state",
        {"state": "failed", "error": "database unreachable"},
        importance="critical",
        force_delivery=True,
    )
    result = api.execute(
        "notifications", {"plan_id": plan, "unacknowledged_only": True}
    )
    assert len(result["notifications"]) == 1
    item = result["notifications"][0]
    api.execute(
        "notification.ack",
        {"plan_id": plan, "notification_id": item["notification_id"]},
    )
    assert api.execute(
        "notifications", {"plan_id": plan, "unacknowledged_only": True}
    )["notifications"] == []
    return _pass(
        "critical_reporting_bypass",
        {"importance": item["importance"], "kind": item["kind"]},
    )


def _case_semantic_interesting_reporting(root: Path) -> dict[str, Any]:
    _workers, _manager_obj, api = _manager(root)
    plan = api.execute(
        "create", {"objective": "interesting", "reporting_mode": "important"}
    )["plan"]["plan_id"]
    api.execute(
        "notify",
        {
            "plan_id": plan,
            "kind": "interesting_result",
            "importance": "important",
            "payload": {"summary": "unexpected useful finding"},
        },
    )
    result = api.execute("notifications", {"plan_id": plan})
    assert len(result["notifications"]) == 1
    assert result["notifications"][0]["payload"]["summary"] == "unexpected useful finding"
    return _pass(
        "semantic_interesting_reporting",
        {"notification": result["notifications"][0]},
    )


def _case_equal_share(root: Path) -> dict[str, Any]:
    counts = _run_share(root, {"a": 1.0, "b": 1.0}, 40)
    assert abs(counts["a"] - counts["b"]) <= 1
    return _pass("equal_round_robin_share", {"counts": counts})


def _case_weighted_share(root: Path) -> dict[str, Any]:
    counts = _run_share(root, {"high": 3.0, "low": 1.0}, 80)
    ratio = counts["high"] / max(1, counts["low"])
    assert 2.7 <= ratio <= 3.3
    return _pass("weighted_fair_share", {"counts": counts, "ratio": ratio})


def _case_multi_worker_weighted_share(root: Path) -> dict[str, Any]:
    weights = {"high": 5.0, "medium": 3.0, "low": 1.0}
    counts = _run_share(root, weights, 180)
    normalized = {name: counts[name] / weight for name, weight in weights.items()}
    spread = max(normalized.values()) - min(normalized.values())
    assert spread <= 1.0, (counts, normalized, spread)
    assert all(counts[name] > 0 for name in weights)
    return _pass(
        "multi_worker_weighted_share",
        {"counts": counts, "normalized_service": normalized, "spread": spread},
    )


def _case_dynamic_weight_change(root: Path) -> dict[str, Any]:
    coordinator = _coordinator(root)
    sequence = 0
    weights = {"a": 1.0, "b": 1.0}
    pending: dict[str, Any] = {}
    for source in weights:
        sequence += 1
        pending[source] = _enqueue(
            coordinator,
            f"{source}-{sequence}",
            source=source,
            weight=weights[source],
        )
    phase_one = {"a": 0, "b": 0}
    for _ in range(20):
        request_id = _next_id(coordinator)
        request = coordinator.get(request_id)
        assert request is not None
        acquired = coordinator.acquire(request_id, timeout_seconds=1.0)
        phase_one[acquired.source] += 1
        coordinator.complete(request_id)
        sequence += 1
        pending[acquired.source] = _enqueue(
            coordinator,
            f"{acquired.source}-{sequence}",
            source=acquired.source,
            weight=weights[acquired.source],
        )
    assert abs(phase_one["a"] - phase_one["b"]) <= 1

    weights["a"] = 4.0
    # Replace the currently queued A request so every future A admission carries the
    # revised weight. The durable virtual-service history is deliberately retained.
    queued_a = pending["a"]
    coordinator.cancel(queued_a.request_id, reason="benchmark weight revision")
    sequence += 1
    pending["a"] = _enqueue(
        coordinator, f"a-{sequence}", source="a", weight=weights["a"]
    )
    phase_two = {"a": 0, "b": 0}
    for _ in range(50):
        request_id = _next_id(coordinator)
        request = coordinator.get(request_id)
        assert request is not None
        acquired = coordinator.acquire(request_id, timeout_seconds=1.0)
        phase_two[acquired.source] += 1
        coordinator.complete(request_id)
        sequence += 1
        pending[acquired.source] = _enqueue(
            coordinator,
            f"{acquired.source}-{sequence}",
            source=acquired.source,
            weight=weights[acquired.source],
        )
    ratio = phase_two["a"] / max(1, phase_two["b"])
    assert 3.3 <= ratio <= 5.0, (phase_two, ratio)
    return _pass(
        "dynamic_weight_change",
        {"phase_one": phase_one, "phase_two": phase_two, "phase_two_ratio": ratio},
    )


def _case_multi_slot_admission(root: Path) -> dict[str, Any]:
    coordinator = InferenceRequestCoordinator(
        root,
        backend_key="benchmark-backend",
        capacity_resolver=lambda: (2, "benchmark_two_slots"),
        poll_seconds=0.001,
        aging_seconds_per_priority=1000.0,
    )
    one = _enqueue(coordinator, "slot-one", source="worker:one")
    two = _enqueue(coordinator, "slot-two", source="worker:two")
    three = _enqueue(coordinator, "slot-three", source="worker:three")
    first = coordinator.acquire(_next_id(coordinator), timeout_seconds=1.0)
    second = coordinator.acquire(_next_id(coordinator), timeout_seconds=1.0)
    assert {first.request_id, second.request_id} == {one.request_id, two.request_id}
    running = coordinator.list(statuses={"running"})
    assert len(running) == 2
    assert coordinator.get(three.request_id).status == "queued"
    coordinator.complete(first.request_id)
    admitted = coordinator.acquire(three.request_id, timeout_seconds=1.0)
    assert admitted.status == "running"
    coordinator.complete(second.request_id)
    coordinator.complete(admitted.request_id)
    return _pass(
        "multi_slot_admission",
        {
            "initial_running": [first.request_id, second.request_id],
            "third_admitted_after_release": admitted.request_id,
        },
    )


def _case_control_priority_preemption(root: Path) -> dict[str, Any]:
    coordinator = _coordinator(root)
    worker = _enqueue(coordinator, "worker", source="worker:w1")
    running = coordinator.acquire(worker.request_id)
    assert running.status == "running"
    coordinator.suspend(worker.request_id, reason="user orchestrator request")
    control = _enqueue(
        coordinator,
        "control",
        source="user_orchestrator",
        priority=1000,
        weight=1.0,
    )
    assert _next_id(coordinator) == control.request_id
    coordinator.acquire(control.request_id)
    coordinator.complete(control.request_id)
    coordinator.resume(worker.request_id, reason="orchestrator resolved")
    replay = coordinator.acquire(worker.request_id)
    assert replay.attempt_count == 2
    coordinator.complete(worker.request_id)
    return _pass(
        "control_priority_preemption_replay",
        {"worker_attempt_count": replay.attempt_count, "control_priority": control.priority},
    )


def _case_aging_prevents_starvation(root: Path) -> dict[str, Any]:
    coordinator = _coordinator(root, aging=0.001)
    old = _enqueue(coordinator, "old", source="worker:old", priority=0)
    with coordinator._connect() as connection:
        connection.execute(
            "UPDATE inference_requests SET queued_epoch=? WHERE request_id=?",
            (time.time() - 0.05, old.request_id),
        )
    fresh = _enqueue(coordinator, "fresh", source="control:fresh", priority=10)
    assert _next_id(coordinator) == old.request_id
    coordinator.acquire(old.request_id)
    coordinator.complete(old.request_id)
    coordinator.acquire(fresh.request_id)
    coordinator.complete(fresh.request_id)
    return _pass("aging_prevents_starvation", {"old": old.request_id, "fresh": fresh.request_id})


CASES: tuple[tuple[str, Callable[[Path], dict[str, Any]]], ...] = (
    ("bulk_plan_apply", _case_bulk_plan_apply),
    ("dependency_output_flow", _case_dependency_flow),
    ("sequential_exclusive", _case_sequential),
    ("bounded_parallel", _case_bounded_parallel),
    ("semantic_branch_resolution", _case_semantic_branch),
    ("failure_branch", _case_failure_branch),
    ("dynamic_replan_replace", _case_dynamic_replan),
    ("named_model_routing", _case_model_routing),
    ("model_route_replacement", _case_model_route_replacement),
    ("missing_model_fails_closed", _case_missing_model_fails_closed),
    ("empty_plan_validation", _case_empty_plan_validation),
    ("cycle_guard", _case_cycle_guard),
    ("plan_cancel", _case_plan_cancel),
    ("plan_auto_completion", _case_plan_completion),
    ("resource_budget_admission", _case_resource_budget),
    ("impossible_resource_fails_closed", _case_impossible_resource),
    ("finish_abort_criteria", _case_finish_abort_criteria),
    ("reporting_completion_only", _case_reporting_completion_only),
    ("critical_reporting_bypass", _case_critical_reporting_bypass),
    ("semantic_interesting_reporting", _case_semantic_interesting_reporting),
    ("equal_round_robin_share", _case_equal_share),
    ("weighted_fair_share", _case_weighted_share),
    ("multi_worker_weighted_share", _case_multi_worker_weighted_share),
    ("dynamic_weight_change", _case_dynamic_weight_change),
    ("multi_slot_admission", _case_multi_slot_admission),
    ("control_priority_preemption_replay", _case_control_priority_preemption),
    ("aging_prevents_starvation", _case_aging_prevents_starvation),
)


def run_orchestration_scheduler_benchmark(
    *,
    output_dir: Path,
    case_ids: list[str] | None = None,
    clean: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = dict(CASES)
    requested = list(case_ids or by_id)
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError("Unknown orchestration-scheduler case: " + ", ".join(unknown))
    results: list[dict[str, Any]] = []
    cases_root = output_dir / "cases"
    cases_root.mkdir(exist_ok=True)
    for case_id in requested:
        case_root = cases_root / case_id
        if case_root.exists():
            shutil.rmtree(case_root)
        case_root.mkdir(parents=True)
        try:
            result = by_id[case_id](case_root)
        except Exception as exc:
            result = {
                "case_id": case_id,
                "passed": False,
                "evidence": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
    passed = sum(bool(item["passed"]) for item in results)
    report = {
        "benchmark": "orchestration-scheduler",
        "generated_at": utc_now_iso(),
        "complete": len(results) == len(requested),
        "passed": passed,
        "total": len(results),
        "all_passed": passed == len(results),
        "results": results,
    }
    (output_dir / "orchestration_scheduler_results.json").write_text(
        stable_json_dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
