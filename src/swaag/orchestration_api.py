from __future__ import annotations

import math

import time
from dataclasses import asdict
from typing import Any

from swaag.orchestration import OrchestrationManager
from swaag.questions import question_inventory, validate_revision
from swaag.progress import plan_progress
from swaag.utils import stable_json_dumps


class OrchestrationApi:
    """Transport-neutral control/query API for durable orchestration plans."""

    version = "swaag.orchestration.v1"

    def __init__(self, manager: OrchestrationManager, *, background_work=None, supervisor=None):
        self.manager = manager
        self.background_work = background_work
        self.supervisor = supervisor

    def execute(self, operation: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(payload or {})
        if operation == "questions.list":
            return {"version": self.version, "inventory": question_inventory(self.manager)}
        if operation == "questions.revise":
            worker_id = _required_text(args, "worker_id")
            revision = validate_revision(args.get("revision"))
            owner = None
            for manager in self.manager.worker_managers.values():
                try:
                    worker = manager.store.get(worker_id)
                    owner = self.manager.worker_managers.get(worker.model_key, manager)
                    break
                except FileNotFoundError:
                    continue
            if owner is None:
                raise FileNotFoundError(worker_id)
            control = owner.runtime.history.enqueue_control_message(worker.session_id,
                stable_json_dumps({"revision": revision, "actor": args.get("actor") or {"role": "orchestration_api"}}),
                source="question_revision", control_id=args.get("control_id"))
            owner.runtime.apply_question_revisions_if_idle(worker.session_id)
            pending = any(item["control_id"] == control["control_id"]
                          for item in owner.runtime.history.list_pending_control_messages(worker.session_id))
            outcome = next(({"event_type": event.event_type, **event.payload}
                for event in owner.runtime.history.iter_history_reverse(worker.session_id,
                    event_types=("agent_question_revised", "agent_question_revision_rejected"))
                if event.payload.get("control_id") == control["control_id"]), None)
            return {"version": self.version, "control_id": control["control_id"], "pending": pending, "outcome": outcome,
                    "inventory": question_inventory(self.manager)}
        if operation == "supervision":
            return {"version": self.version, "supervision": self.supervisor.snapshot() if self.supervisor else None}
        if operation.startswith("backlog."):
            if self.background_work is None:
                raise ValueError("background-work service is unavailable")
            if operation == "backlog.list":
                return {"version": self.version, "mode": self.background_work.mode,
                        "backlog": self.background_work.list()}
            if operation == "backlog.enqueue":
                item = self.background_work.enqueue(_required_text(args, "plan_id"),
                    authorization_session_id=_required_text(args, "authorization_session_id"),
                    authorization_event_sequence=args.get("authorization_event_sequence"))
                return {"version": self.version, "item": item}
            if operation == "backlog.cancel":
                return {"version": self.version, "canceled": self.background_work.cancel(_required_text(args, "plan_id"))}
            raise ValueError("unknown background-work operation")
        if operation == "plan.apply":
            spec = args.get("plan_spec")
            if not isinstance(spec, dict):
                raise ValueError("plan_spec must be an object")
            applied = self.manager.apply_plan_spec(spec)
            return {
                "version": self.version,
                **self._snapshot_from(applied["snapshot"]),
                "node_ids": applied["node_ids"],
                "validation": applied["validation"],
                "started_worker_ids": applied["started_worker_ids"],
            }
        if operation == "create":
            mode = _optional_text(args, "scheduling_mode") or "parallel"
            maximum = _nonnegative_int(args, "max_parallel", default=0)
            reporting_mode = _optional_text(args, "reporting_mode") or "important"
            plan = self.manager.create_plan(
                _required_text(args, "objective"),
                scheduling_mode=mode,
                max_parallel=maximum,
                reporting_mode=reporting_mode,
            )
            resource_limits = args.get("resource_limits")
            if resource_limits is not None:
                if not isinstance(resource_limits, dict):
                    raise ValueError("resource_limits must be an object or null")
                self.manager.store.set_resource_limits(plan.plan_id, resource_limits)
            return {"version": self.version, **self._snapshot(plan.plan_id)}
        if operation == "configure":
            plan_id = _required_text(args, "plan_id")
            self.manager.configure_plan(
                plan_id,
                scheduling_mode=_required_text(args, "scheduling_mode"),
                max_parallel=_nonnegative_int(args, "max_parallel", default=0),
            )
            resource_limits = args.get("resource_limits")
            if resource_limits is not None:
                if not isinstance(resource_limits, dict):
                    raise ValueError("resource_limits must be an object or null")
                self.manager.store.set_resource_limits(plan_id, resource_limits)
            reporting_mode = _optional_text(args, "reporting_mode")
            if reporting_mode is not None:
                self.manager.store.set_reporting_mode(plan_id, reporting_mode)
            return {"version": self.version, **self._snapshot(plan_id)}
        if operation == "reporting.configure":
            plan_id = _required_text(args, "plan_id")
            self.manager.store.set_reporting_mode(
                plan_id, _required_text(args, "reporting_mode")
            )
            return {"version": self.version, **self._snapshot(plan_id)}
        if operation == "list":
            return {
                "version": self.version,
                "plans": [asdict(item) for item in self.manager.store.list_plans()],
            }
        if operation == "get":
            return {"version": self.version, **self._snapshot(_required_text(args, "plan_id"))}
        if operation == "node.add":
            resources = args.get("resources")
            if resources is not None and not isinstance(resources, dict):
                raise ValueError("resources must be an object or null")
            node_id = self.manager.add_worker(
                _required_text(args, "plan_id"),
                _required_text(args, "objective"),
                priority=_positive_float(args, "priority", default=1.0),
                model_key=_optional_text(args, "model_key"),
                finish_criteria=_optional_text(args, "finish_criteria"),
                abort_criteria=_optional_text(args, "abort_criteria"),
                resources=resources,
                completion_mode=_optional_text(args, "completion_mode") or "natural",
            )
            return {"version": self.version, "node_id": node_id}
        if operation == "node.revise":
            replace_worker = args.get("replace_worker", False)
            if not isinstance(replace_worker, bool):
                raise ValueError("replace_worker must be a boolean")
            self.manager.revise_node(
                _required_text(args, "plan_id"),
                _required_text(args, "node_id"),
                objective=_optional_text(args, "objective"),
                priority=(None if args.get("priority") is None else _positive_float(args, "priority", default=1.0)),
                model_key=_optional_text(args, "model_key"),
                finish_criteria=_optional_text(args, "finish_criteria"),
                abort_criteria=_optional_text(args, "abort_criteria"),
                replace_worker=replace_worker,
            )
            return {"version": self.version, **self._snapshot(_required_text(args, "plan_id"))}
        if operation == "dependency.resolve":
            plan_id = _required_text(args, "plan_id")
            satisfied = args.get("satisfied")
            if not isinstance(satisfied, bool):
                raise ValueError("satisfied must be a boolean")
            self.manager.store.resolve_dependency(
                plan_id,
                _required_text(args, "edge_id"),
                satisfied=satisfied,
                decision=_required_text(args, "decision"),
            )
            return {"version": self.version, **self._snapshot(plan_id)}
        if operation == "dependency.remove":
            plan_id = _required_text(args, "plan_id")
            self.manager.store.remove_dependency(plan_id, _required_text(args, "edge_id"))
            return {"version": self.version, **self._snapshot(plan_id)}
        if operation == "dependency.add":
            condition = args.get("condition")
            input_mapping = args.get("input_mapping")
            if condition is not None and not isinstance(condition, dict):
                raise ValueError("condition must be an object or null")
            if input_mapping is not None and not isinstance(input_mapping, dict):
                raise ValueError("input_mapping must be an object or null")
            edge_id = self.manager.add_dependency(
                _required_text(args, "plan_id"),
                _required_text(args, "source_node_id"),
                _required_text(args, "target_node_id"),
                condition=condition,
                input_mapping=input_mapping,
            )
            return {"version": self.version, "edge_id": edge_id}
        if operation == "validate":
            plan_id = _required_text(args, "plan_id")
            return {
                "version": self.version,
                "plan_id": plan_id,
                "validation": self.manager.validate_plan(plan_id),
            }
        if operation == "start":
            plan_id = _required_text(args, "plan_id")
            started = self.manager.start_ready(plan_id)
            return {"version": self.version, "plan_id": plan_id, "started_worker_ids": started}
        if operation == "advance":
            plan_id = _required_text(args, "plan_id")
            return {"version": self.version, **self._snapshot_from(self.manager.advance(plan_id))}
        if operation == "cancel":
            plan_id = _required_text(args, "plan_id")
            self.manager.cancel_plan(
                plan_id,
                reason=_optional_text(args, "reason") or "orchestration API cancellation",
            )
            return {"version": self.version, **self._snapshot(plan_id)}
        if operation in {"notifications", "notifications.wait"}:
            plan_id = _required_text(args, "plan_id")
            after = _nonnegative_int(args, "after_sequence", default=0)
            unacknowledged_only = args.get("unacknowledged_only", False)
            if not isinstance(unacknowledged_only, bool):
                raise ValueError("unacknowledged_only must be a boolean")
            timeout = 0.0
            if operation == "notifications.wait":
                raw_timeout = args.get("timeout_seconds", 30.0)
                if (
                    isinstance(raw_timeout, bool)
                    or not isinstance(raw_timeout, (int, float))
                    or not 0 <= float(raw_timeout) <= 60
                ):
                    raise ValueError("timeout_seconds must be between 0 and 60")
                timeout = float(raw_timeout)
            deadline = time.monotonic() + timeout
            while True:
                items = self.manager.store.notifications(
                    plan_id,
                    after_sequence=after,
                    unacknowledged_only=unacknowledged_only,
                )
                if items or time.monotonic() >= deadline:
                    return {
                        "version": self.version,
                        "plan_id": plan_id,
                        "notifications": items,
                        "next_sequence": (
                            int(items[-1]["sequence"]) if items else after
                        ),
                        "timed_out": not items and operation == "notifications.wait",
                    }
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if operation == "notification.ack":
            plan_id = _required_text(args, "plan_id")
            self.manager.store.acknowledge_notification(
                plan_id, _required_text(args, "notification_id")
            )
            return {
                "version": self.version,
                "plan_id": plan_id,
                "acknowledged": _required_text(args, "notification_id"),
            }
        if operation == "notify":
            plan_id = _required_text(args, "plan_id")
            kind = _required_text(args, "kind")
            data = args.get("payload") or {}
            if not isinstance(data, dict):
                raise ValueError("payload must be an object")
            importance = _optional_text(args, "importance") or "important"
            self.manager.store.record_notification(
                plan_id, kind, data, importance=importance
            )
            return {"version": self.version, **self._snapshot(plan_id)}
        raise ValueError(f"Unknown orchestration operation: {operation}")

    def _snapshot(self, plan_id: str) -> dict[str, Any]:
        return self._snapshot_from(self.manager.store.snapshot(plan_id))

    @staticmethod
    def _snapshot_from(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "plan": asdict(snapshot["plan"]),
            "resource_limits": snapshot.get("resource_limits", {}),
            "nodes": snapshot["nodes"],
            "edges": snapshot["edges"],
            "events": snapshot["events"],
            "progress": plan_progress(snapshot["nodes"]),
        }


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value.strip() or None


def _positive_float(payload: dict[str, Any], key: str, *, default: float) -> float:
    value = payload.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{key} must be a finite positive number")
    return float(value)


def _nonnegative_int(payload: dict[str, Any], key: str, *, default: int) -> int:
    value = payload.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value
