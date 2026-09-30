from __future__ import annotations

from dataclasses import dataclass, replace

from swaag.orchestration import OrchestrationManager, OrchestrationStore
from swaag.orchestration_api import OrchestrationApi


@dataclass
class _Worker:
    worker_id: str
    status: str = "created"
    result: str | None = None
    error: str | None = None
    inference_weight: float = 1.0


class _Store:
    def __init__(self):
        self.items: dict[str, _Worker] = {}

    def get(self, worker_id: str) -> _Worker:
        if worker_id not in self.items:
            raise FileNotFoundError(worker_id)
        return self.items[worker_id]

    def set_inference_weight(self, worker_id: str, weight: float) -> _Worker:
        current = self.get(worker_id)
        updated = replace(current, inference_weight=float(weight))
        self.items[worker_id] = updated
        return updated


class _Workers:
    def __init__(self):
        self.store = _Store()
        self.created: list[str] = []
        self.canceled: list[str] = []

    def create(
        self,
        objective: str,
        *,
        name: str | None = None,
        inference_weight: float = 1.0,
    ):
        worker = _Worker(
            f"worker-{len(self.store.items)+1}",
            inference_weight=float(inference_weight),
        )
        self.store.items[worker.worker_id] = worker
        self.created.append(objective)
        return worker

    def start(self, worker_id: str):
        current = self.store.get(worker_id)
        updated = replace(current, status="queued")
        self.store.items[worker_id] = updated
        return updated


    def message(self, worker_id: str, message: str, *, source: str, resume_if_idle: bool = True):
        current = self.store.get(worker_id)
        return current

    def cancel(self, worker_id: str, *, reason: str):
        current = self.store.get(worker_id)
        updated = replace(current, status="canceled")
        self.store.items[worker_id] = updated
        self.canceled.append(worker_id)
        return updated


def test_orchestration_api_builds_and_advances_dependency_graph(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    plan_id = api.execute("create", {"objective": "pipeline"})["plan"]["plan_id"]
    first = api.execute("node.add", {"plan_id": plan_id, "objective": "inspect"})["node_id"]
    second = api.execute("node.add", {"plan_id": plan_id, "objective": "verify", "priority": 2})["node_id"]
    api.execute("dependency.add", {"plan_id": plan_id, "source_node_id": first, "target_node_id": second, "input_mapping": {"result": "evidence"}})
    started = api.execute("start", {"plan_id": plan_id})
    assert len(started["started_worker_ids"]) == 1
    first_worker = started["started_worker_ids"][0]
    workers.store.items[first_worker] = replace(workers.store.items[first_worker], status="completed", result="exact evidence")
    advanced = api.execute("advance", {"plan_id": plan_id})
    assert len(workers.created) == 2
    assert "exact evidence" in workers.created[1]
    assert {n["state"] for n in advanced["nodes"]} >= {"completed", "queued"}


def test_nondefault_model_assignment_fails_closed(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    plan_id = api.execute("create", {"objective": "route"})["plan"]["plan_id"]
    api.execute("node.add", {"plan_id": plan_id, "objective": "specialist", "model_key": "strong"})
    validation = api.execute("validate", {"plan_id": plan_id})["validation"]
    assert validation["valid"] is False
    assert validation["issues"][0]["code"] == "model_assignment_unavailable"
    try:
        api.execute("start", {"plan_id": plan_id})
    except ValueError as exc:
        assert "model_assignment_unavailable" in str(exc)
    else:
        raise AssertionError("invalid model route plan started")
    assert workers.created == []
    snap = api.execute("get", {"plan_id": plan_id})
    assert snap["plan"]["status"] == "draft"
    delivered = api.execute("notifications", {"plan_id": plan_id})["notifications"]
    assert delivered[-1]["kind"] == "plan_validation_failed"


def test_sequential_policy_starts_only_one_independent_worker(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    plan_id = api.execute(
        "create",
        {"objective": "ordered", "scheduling_mode": "sequential"},
    )["plan"]["plan_id"]
    api.execute("node.add", {"plan_id": plan_id, "objective": "first"})
    api.execute("node.add", {"plan_id": plan_id, "objective": "second"})
    started = api.execute("start", {"plan_id": plan_id})
    assert len(started["started_worker_ids"]) == 1
    api.execute("advance", {"plan_id": plan_id})
    assert len(workers.created) == 1
    first = started["started_worker_ids"][0]
    workers.store.items[first] = replace(
        workers.store.items[first], status="completed", result="done"
    )
    api.execute("advance", {"plan_id": plan_id})
    assert len(workers.created) == 2


def test_parallel_policy_respects_max_parallel(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    plan_id = api.execute(
        "create",
        {"objective": "bounded", "scheduling_mode": "parallel", "max_parallel": 2},
    )["plan"]["plan_id"]
    for name in ("a", "b", "c"):
        api.execute("node.add", {"plan_id": plan_id, "objective": name})
    started = api.execute("start", {"plan_id": plan_id})
    assert len(started["started_worker_ids"]) == 2
    assert len(workers.created) == 2


def test_named_model_route_uses_selected_worker_manager(tmp_path):
    default_workers = _Workers()
    strong_workers = _Workers()
    manager = OrchestrationManager(
        default_workers,
        store=OrchestrationStore(tmp_path),
        worker_managers={"strong": strong_workers},
    )
    api = OrchestrationApi(manager)
    plan_id = api.execute("create", {"objective": "route"})["plan"]["plan_id"]
    api.execute(
        "node.add",
        {"plan_id": plan_id, "objective": "hard task", "model_key": "strong"},
    )
    started = api.execute("start", {"plan_id": plan_id})
    assert len(started["started_worker_ids"]) == 1
    assert default_workers.created == []
    assert strong_workers.created == ["hard task"]
    snap = api.execute("get", {"plan_id": plan_id})
    assigned = [
        event
        for event in snap["events"]
        if event["event_type"] == "notification"
        and event["payload"].get("kind") == "worker_model_assigned"
    ]
    assert assigned[-1]["payload"]["model_key"] == "strong"


def test_resource_budget_admits_only_workers_that_fit(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    plan_id = api.execute(
        "create",
        {
            "objective": "resource bounded",
            "resource_limits": {"ram_gb": 8, "api_slots": 2},
        },
    )["plan"]["plan_id"]
    api.execute(
        "node.add",
        {"plan_id": plan_id, "objective": "large", "resources": {"ram_gb": 6, "api_slots": 1}},
    )
    api.execute(
        "node.add",
        {"plan_id": plan_id, "objective": "medium", "resources": {"ram_gb": 4, "api_slots": 1}},
    )
    started = api.execute("start", {"plan_id": plan_id})["started_worker_ids"]
    assert len(started) == 1
    first = started[0]
    workers.store.items[first] = replace(
        workers.store.items[first], status="completed", result="released resources"
    )
    api.execute("advance", {"plan_id": plan_id})
    assert len(workers.created) == 2


def test_impossible_resource_request_fails_closed(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    plan_id = api.execute(
        "create", {"objective": "bounded", "resource_limits": {"gpu_gb": 8}}
    )["plan"]["plan_id"]
    api.execute(
        "node.add",
        {"plan_id": plan_id, "objective": "too large", "resources": {"gpu_gb": 12}},
    )
    validation = api.execute("validate", {"plan_id": plan_id})["validation"]
    assert validation["valid"] is False
    assert validation["issues"][0]["code"] == "resource_limit_exceeded"
    try:
        api.execute("start", {"plan_id": plan_id})
    except ValueError as exc:
        assert "resource_limit_exceeded" in str(exc)
    else:
        raise AssertionError("resource-invalid plan started")
    assert workers.created == []
    delivered = api.execute("notifications", {"plan_id": plan_id})["notifications"]
    assert delivered[-1]["kind"] == "plan_validation_failed"


def test_finish_and_abort_criteria_enter_worker_objective(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    plan_id = api.execute("create", {"objective": "criteria"})["plan"]["plan_id"]
    api.execute(
        "node.add",
        {
            "plan_id": plan_id,
            "objective": "work",
            "finish_criteria": "tests pass and artifact exists",
            "abort_criteria": "required database is unavailable",
        },
    )
    api.execute("start", {"plan_id": plan_id})
    objective = workers.created[0]
    assert "tests pass and artifact exists" in objective
    assert "required database is unavailable" in objective


def test_changing_bound_model_route_requires_worker_replacement(tmp_path):
    default = _Workers()
    strong = _Workers()
    manager = OrchestrationManager(
        default,
        store=OrchestrationStore(tmp_path),
        worker_managers={"strong": strong},
    )
    api = OrchestrationApi(manager)
    plan_id = api.execute("create", {"objective": "route change"})["plan"]["plan_id"]
    node_id = api.execute(
        "node.add", {"plan_id": plan_id, "objective": "work"}
    )["node_id"]
    api.execute("start", {"plan_id": plan_id})
    try:
        api.execute(
            "node.revise",
            {"plan_id": plan_id, "node_id": node_id, "model_key": "strong"},
        )
    except ValueError as exc:
        assert "requires replace_worker=true" in str(exc)
    else:
        raise AssertionError("bound worker model route changed without replacement")
    api.execute(
        "node.revise",
        {
            "plan_id": plan_id,
            "node_id": node_id,
            "model_key": "strong",
            "replace_worker": True,
        },
    )
    api.execute("advance", {"plan_id": plan_id})
    assert len(default.canceled) == 1
    assert len(strong.created) == 1


def test_completion_only_reporting_suppresses_routine_worker_notification(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    plan_id = api.execute(
        "create",
        {"objective": "quiet", "reporting_mode": "completion_only"},
    )["plan"]["plan_id"]
    manager.store.record_notification(
        plan_id,
        "worker_state",
        {"state": "completed", "worker_id": "worker-x"},
        importance="normal",
    )
    assert api.execute("notifications", {"plan_id": plan_id})["notifications"] == []
    manager.store.record_notification(
        plan_id,
        "plan_completed",
        {"node_count": 1},
        importance="normal",
    )
    delivered = api.execute("notifications", {"plan_id": plan_id})["notifications"]
    assert [item["kind"] for item in delivered] == ["plan_completed"]


def test_critical_notification_bypasses_reporting_suppression_and_can_ack(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    plan_id = api.execute(
        "create", {"objective": "manual only", "reporting_mode": "manual"}
    )["plan"]["plan_id"]
    manager.store.record_notification(
        plan_id,
        "worker_state",
        {"state": "failed", "error": "database unreachable"},
        importance="critical",
        force_delivery=True,
    )
    result = api.execute(
        "notifications",
        {"plan_id": plan_id, "unacknowledged_only": True},
    )
    assert len(result["notifications"]) == 1
    item = result["notifications"][0]
    assert item["importance"] == "critical"
    api.execute(
        "notification.ack",
        {"plan_id": plan_id, "notification_id": item["notification_id"]},
    )
    assert api.execute(
        "notifications",
        {"plan_id": plan_id, "unacknowledged_only": True},
    )["notifications"] == []


def test_important_reporting_accepts_semantic_notification_and_cursor_wait(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    plan_id = api.execute(
        "create", {"objective": "interesting", "reporting_mode": "important"}
    )["plan"]["plan_id"]
    api.execute(
        "notify",
        {
            "plan_id": plan_id,
            "kind": "interesting_result",
            "importance": "important",
            "payload": {"summary": "unexpected useful finding"},
        },
    )
    first = api.execute("notifications", {"plan_id": plan_id})
    assert len(first["notifications"]) == 1
    cursor = first["next_sequence"]
    waited = api.execute(
        "notifications.wait",
        {"plan_id": plan_id, "after_sequence": cursor, "timeout_seconds": 0.01},
    )
    assert waited["notifications"] == [] and waited["timed_out"] is True


def test_empty_plan_validation_prevents_start(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    plan_id = api.execute("create", {"objective": "empty"})["plan"]["plan_id"]
    validation = api.execute("validate", {"plan_id": plan_id})["validation"]
    assert validation["valid"] is False
    assert validation["issues"] == [
        {"code": "empty_plan", "message": "orchestration plan has no worker nodes"}
    ]
    try:
        api.execute("start", {"plan_id": plan_id})
    except ValueError as exc:
        assert "empty_plan" in str(exc)
    else:
        raise AssertionError("empty plan started")


def test_bulk_plan_apply_materializes_validates_and_starts_graph(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
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
    assert set(result["node_ids"]) == {"alpha", "beta"}
    assert len(result["started_worker_ids"]) == 1
    assert result["plan"]["status"] == "active"
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1
    assert len(workers.created) == 1
    assert workers.created[0].startswith("Return exactly ALPHA COMPLETE")


def test_bulk_plan_apply_rejects_cycle_before_persisting_any_plan(tmp_path):
    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    api = OrchestrationApi(manager)
    try:
        api.execute(
            "plan.apply",
            {
                "plan_spec": {
                    "objective": "bad cycle",
                    "nodes": [
                        {"key": "a", "objective": "a"},
                        {"key": "b", "objective": "b"},
                    ],
                    "edges": [
                        {"source": "a", "target": "b"},
                        {"source": "b", "target": "a"},
                    ],
                    "start": False,
                }
            },
        )
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cyclic bulk plan was accepted")
    assert manager.store.list_plans() == []


def test_orchestration_tool_bulk_apply_is_side_effect_only_when_starting():
    from swaag.tools.orchestration import OrchestrationControlTool

    tool = OrchestrationControlTool()
    base = {key: None for key in tool.input_schema["required"]}
    base["operation"] = "plan.apply"
    spec = {
        "objective": "x",
        "scheduling_mode": "parallel",
        "max_parallel": 0,
        "reporting_mode": "important",
        "resource_limits_json": None,
        "nodes": [
            {
                "key": "a",
                "objective": "a",
                "priority": 1.0,
                "model_key": None,
                "finish_criteria": None,
                "abort_criteria": None,
                "resources_json": None,
            }
        ],
        "edges": [],
        "start": False,
    }
    base["plan_spec"] = spec
    validated = tool.validate(base)
    assert tool.effective_kind(validated) == "stateful"
    base["plan_spec"] = {**spec, "start": True}
    validated = tool.validate(base)
    assert tool.effective_kind(validated) == "side_effect"


def test_named_route_capability_failure_blocks_plan_before_worker_creation(tmp_path):
    default = _Workers()
    strong = _Workers()
    calls = []

    def validator(model_key, manager):
        calls.append((model_key, manager))
        return False, "server_schema probe failed"

    manager = OrchestrationManager(
        default,
        store=OrchestrationStore(tmp_path),
        worker_managers={"strong": strong},
        route_capability_validator=validator,
    )
    api = OrchestrationApi(manager)
    plan_id = api.execute("create", {"objective": "route guard"})["plan"]["plan_id"]
    api.execute(
        "node.add",
        {"plan_id": plan_id, "objective": "specialist", "model_key": "strong"},
    )

    validation = api.execute("validate", {"plan_id": plan_id})["validation"]

    assert validation["valid"] is False
    assert validation["issues"][0]["code"] == "model_route_capability_failed"
    assert "server_schema probe failed" in validation["issues"][0]["message"]
    assert default.created == []
    assert strong.created == []
    try:
        api.execute("start", {"plan_id": plan_id})
    except ValueError as exc:
        assert "model_route_capability_failed" in str(exc)
    else:
        raise AssertionError("route with failed capability probe started")
    assert len(calls) >= 1


def test_bulk_plan_apply_rejects_failed_route_probe_before_persisting(tmp_path):
    default = _Workers()
    strong = _Workers()
    manager = OrchestrationManager(
        default,
        store=OrchestrationStore(tmp_path),
        worker_managers={"strong": strong},
        route_capability_validator=lambda _key, _manager: (
            False,
            "structured output unavailable",
        ),
    )
    api = OrchestrationApi(manager)

    try:
        api.execute(
            "plan.apply",
            {
                "plan_spec": {
                    "objective": "specialist plan",
                    "scheduling_mode": "parallel",
                    "max_parallel": 0,
                    "reporting_mode": "important",
                    "resource_limits": {},
                    "nodes": [
                        {
                            "key": "specialist",
                            "objective": "do specialist work",
                            "priority": 1.0,
                            "model_key": "strong",
                            "finish_criteria": None,
                            "abort_criteria": None,
                            "resources": {},
                        }
                    ],
                    "edges": [],
                    "start": True,
                }
            },
        )
    except ValueError as exc:
        assert "structured output unavailable" in str(exc)
    else:
        raise AssertionError("bulk plan persisted despite failed route probe")
    assert manager.store.list_plans() == []
    assert strong.created == []


def test_invalid_revision_cannot_change_or_cancel_bound_worker(tmp_path):
    import pytest

    workers = _Workers()
    manager = OrchestrationManager(workers, store=OrchestrationStore(tmp_path))
    plan = manager.create_plan("preserve the active worker")
    node_id = manager.add_worker(plan.plan_id, "work", priority=2.0)
    manager.start_ready(plan.plan_id)
    before = manager.store.snapshot(plan.plan_id)
    bound = before["nodes"][0]["worker_id"]
    original = workers.store.get(bound)
    for value in [float("nan"), float("inf"), float("-inf"), 0, -1, True, "2"]:
        with pytest.raises(ValueError, match="finite and positive"):
            manager.revise_node(plan.plan_id, node_id, priority=value, replace_worker=True)
        assert manager.store.snapshot(plan.plan_id) == before
        assert workers.store.get(bound) == original
        assert workers.canceled == []
