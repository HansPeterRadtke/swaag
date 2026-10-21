"""Compact, evidence-labeled worker/plan state for the fast spoken interface.

Built during the background world refresh, never by scanning worker histories
in the user-facing inference path. This is observational context, not tool
instructions or proof that model-authored status reflects external effects.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from swaag.progress import plan_progress, step_percentage, validate_progress

_TERMINAL = frozenset({"completed", "failed", "canceled", "archived"})
_MAX_OBJECTIVE_CHARS = 300
_MAX_ACTIVITY_CHARS = 200


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return date.astimezone(timezone.utc) if date.tzinfo is not None else None
    except ValueError:
        return None


def _elapsed(started: Any, now: datetime) -> int | None:
    parsed = _timestamp(started)
    if parsed is None:
        return None
    duration = (now - parsed).total_seconds()
    return max(0, int(duration)) if duration >= -3 else None


def _excerpt(text: Any, limit: int) -> tuple[str, bool]:
    if not isinstance(text, str):
        return "", False
    value = " ".join(text.strip().split())
    if len(value) <= limit:
        return value, False
    # Excerpts are explicitly marked and never passed off as full source.
    return value[:limit].rsplit(" ", 1)[0], True


def _assessment(report: Any, *, completion_mode: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "overall_percent": None, "cycle_percent": None,
        "remaining_seconds_at_assessment": None,
        "assessment_at": None,
        "progress_basis": "no valid explicit worker step assessment",
        "completed_steps": None, "total_steps": None,
    }
    if not isinstance(report, dict):
        return result
    assessment = report.get("latest_progress")
    if not isinstance(assessment, dict) or not isinstance(assessment.get("payload"), dict):
        if report.get("event_type") == "agent_progress":
            assessment = {"payload": report.get("payload"), "timestamp": report.get("timestamp")}
        else:
            return result
    try:
        validated = validate_progress(assessment["payload"])
        percentage = round(step_percentage(validated), 1)
    except (TypeError, KeyError, ValueError, ZeroDivisionError):
        return result
    continuous = completion_mode == "continuous"
    steps = validated["steps"]
    result.update({
        "overall_percent": None if continuous else percentage,
        "cycle_percent": percentage if continuous else None,
        "remaining_seconds_at_assessment": (
            None if continuous else validated["estimated_remaining_seconds"]
        ),
        "assessment_at": assessment.get("timestamp"),
        "progress_basis": "worker model-assessed completed step weights, not elapsed time",
        "completed_steps": sum(item["state"] == "completed" for item in steps),
        "total_steps": len(steps),
        "estimate_conditions": validated.get("estimate_conditions") or None,
    })
    return result


def build_fast_worker_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, compact summary of ALL active workers and plans.

    No invented percentage/ETA. No snippets from tool results. Counts include
    terminal workers, but only active worker details and a few terminal changes.
    """
    if not isinstance(snapshot, dict):
        return {"snapshot_complete": False}
    workers = snapshot.get("active_workers")
    plans = snapshot.get("active_plans")
    terminal = snapshot.get("terminal_workers")
    if not isinstance(workers, list) or not isinstance(plans, list) or not isinstance(terminal, list):
        return {"snapshot_complete": False, "captured_at": snapshot.get("captured_at")}
    if not all(isinstance(item, dict) for item in (*workers, *plans, *terminal)):
        return {"snapshot_complete": False, "captured_at": snapshot.get("captured_at")}
    reports = snapshot.get("active_worker_reports")
    if not isinstance(reports, dict):
        reports = {}
    graphs = snapshot.get("active_plan_graphs")
    if not isinstance(graphs, dict):
        graphs = {}
    supervision = snapshot.get("supervision") or {}
    active_sessions = supervision.get("active_sessions", []) if isinstance(supervision, dict) else []
    if not isinstance(active_sessions, list):
        active_sessions = []
    by_session = {
        item.get("session_id"): item
        for item in active_sessions
        if isinstance(item, dict) and isinstance(item.get("session_id"), str)
    }
    active = []
    for worker in workers:
        ident = str(worker.get("worker_id") or "")
        objective, shortened = _excerpt(worker.get("objective"), _MAX_OBJECTIVE_CHARS)
        report = reports.get(ident)
        if not isinstance(report, dict):
            report = {}
        latest = report.get("latest_status")
        if not isinstance(latest, dict):
            latest = report
        status_payload = latest.get("payload") if isinstance(latest.get("payload"), dict) else {}
        activity = by_session.get(worker.get("session_id"))
        if not isinstance(activity, dict):
            activity = {}
        recent, recent_short = _excerpt(
            status_payload.get("just_happened") or status_payload.get("situation"),
            _MAX_ACTIVITY_CHARS,
        )
        next_action, action_short = _excerpt(status_payload.get("action"), _MAX_ACTIVITY_CHARS)
        row = {
            "worker_id": ident,
            "status": worker.get("status"),
            "task": objective,
            "task_is_excerpt": shortened,
            "model_route": worker.get("model_key"),
            "completion_mode": worker.get("completion_mode", "natural"),
            "started_at": worker.get("started_at"),
            "created_at": worker.get("created_at"),
            "updated_at": worker.get("updated_at"),
            "run_count": worker.get("run_count"),
            "latest_report_at": latest.get("timestamp"),
            "latest_report_kind": latest.get("event_type"),
            "reported_activity": recent or None,
            "reported_activity_is_excerpt": recent_short,
            "reported_next_action": next_action or None,
            "reported_next_action_is_excerpt": action_short,
            "report_scope": "worker-authored, not verified external effects",
            "supervisor_operation": activity.get("operation_kind"),
            "supervisor_phase": activity.get("phase"),
            "supervisor_substate": activity.get("substate"),
            "supervisor_detail": _excerpt(activity.get("detail"), 120)[0] or None,
            "supervisor_heartbeat_age_seconds_at_capture": activity.get("heartbeat_age_seconds"),
            "supervisor_heartbeat_is_computation_evidence": activity.get("heartbeat_is_computation_evidence", False),
            "progress": _assessment(report, completion_mode=worker.get("completion_mode")),
        }
        active.append(row)
    result_plans = []
    for plan in plans:
        ident = str(plan.get("plan_id") or "")
        graph = graphs.get(ident)
        nodes = graph.get("nodes") if isinstance(graph, dict) else None
        if not isinstance(nodes, list) or not all(isinstance(x, dict) for x in nodes):
            progress = {"overall_percent": None, "finite_nodes_percent": None,
                        "work_kind": "unknown", "basis": "node state unavailable"}
            node_details = None
        else:
            progress = plan_progress(nodes)
            node_details = [
                {"node_id": n.get("node_id"), "worker_id": n.get("worker_id"),
                 "state": n.get("state"), "completion_mode": n.get("completion_mode", "natural"),
                 "task": _excerpt(n.get("objective"), 170)[0]}
                for n in nodes
            ]
        objective, shortened = _excerpt(plan.get("objective"), 300)
        result_plans.append({
            "plan_id": ident, "status": plan.get("status"),
            "task": objective, "task_is_excerpt": shortened,
            "created_at": plan.get("created_at"), "updated_at": plan.get("updated_at"),
            "progress": progress, "nodes": node_details,
            "dependency_count": len(graph.get("edges", [])) if isinstance(graph, dict)
                                and isinstance(graph.get("edges"), list) else None,
        })
    terminal_counts = dict(Counter(str(w.get("status")) for w in terminal))
    recent_terminal = sorted(terminal, key=lambda item: str(item.get("updated_at") or ""), reverse=True)[:3]
    questions = snapshot.get("worker_question_inventory")
    pending_questions = questions.get("questions") if isinstance(questions, dict) else None
    question_count = len(pending_questions) if isinstance(pending_questions, list) else None
    question_summaries = []
    if isinstance(pending_questions, list):
        for question in pending_questions[:8]:
            if not isinstance(question, dict):
                continue
            content, shortened = _excerpt(question.get("question"), 170)
            question_summaries.append({
                "worker_id": question.get("worker_id"),
                "criticality": question.get("criticality"),
                "question": content,
                "question_is_excerpt": shortened,
            })
    return {
        "snapshot_complete": True,
        "captured_at": snapshot.get("captured_at"),
        "active_worker_count": len(workers),
        "active_worker_status_counts": dict(Counter(str(w.get("status")) for w in workers)),
        "terminal_worker_status_counts": terminal_counts,
        "active_plan_count": len(plans),
        "workers": active,
        "plans": result_plans,
        "recent_terminal_workers": [
            {"worker_id": w.get("worker_id"), "status": w.get("status"),
             "task": _excerpt(w.get("objective"), 120)[0],
             "updated_at": w.get("updated_at")}
            for w in recent_terminal
        ],
        "open_worker_question_count": question_count,
        "worker_question_inventory_complete": questions.get("complete")
            if isinstance(questions, dict) else None,
        "worker_question_summaries": question_summaries,
        "worker_question_details_not_listed": max(0, question_count - len(question_summaries))
            if question_count is not None else None,
        "orchestrator_active_turn": (
            (snapshot.get("orchestrator") or {}).get("active_turn") is not None
            if isinstance(snapshot.get("orchestrator"), dict) else None
        ),
        "model_routing": snapshot.get("model_routing") or {},
        "host_storage": snapshot.get("host_storage") if isinstance(snapshot.get("host_storage"), dict) else None,
    }


def fast_worker_state_at_prompt(summary: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Derive elapsed time from actual timestamps. Never fabricate ETA/% done."""
    now = now or datetime.now(timezone.utc)
    view = dict(summary)
    view.pop("model_routing", None)
    if summary.get("snapshot_complete") is not True:
        return view
    workers = []
    for item in summary.get("workers", []):
        row = dict(item)
        row["elapsed_seconds"] = _elapsed(item.get("started_at"), now)
        row["report_age_seconds"] = _elapsed(item.get("latest_report_at"), now)
        progress = dict(item.get("progress") or {})
        assessed = progress.get("remaining_seconds_at_assessment")
        age = _elapsed(progress.get("assessment_at"), now)
        progress["estimate_age_seconds"] = age
        progress["estimate_deadline_passed"] = (
            age > assessed if age is not None and isinstance(assessed, (int, float))
            and not isinstance(assessed, bool) and assessed >= 0 else None
        )
        row["progress"] = progress
        workers.append(row)
    view["workers"] = workers
    plans = []
    for item in summary.get("plans", []):
        row = dict(item)
        row["elapsed_seconds_since_creation"] = _elapsed(item.get("created_at"), now)
        assessment = row.pop("progress", {})
        row["plan_node_progress"] = {
            "completed_finite_nodes": assessment.get("completed_finite_nodes"),
            "total_finite_nodes": assessment.get("total_finite_nodes"),
            "finite_nodes_percent": assessment.get("finite_nodes_percent"),
            "work_kind": assessment.get("work_kind"),
            "basis": "completed nodes by count; NOT individual worker task completion",
        }
        plans.append(row)
    view["plans"] = plans
    return view


def compact_worker_prompt_view(full_view: dict[str, Any]) -> dict[str, Any]:
    """One voice-friendly status card per active worker, with explicit unknowns.

    Derived only from the already-prepared state, with no model call, I/O,
    semantic guessing or worker omission. Exact records remain queryable by the
    original orchestrator. This is read-only observation, not authorization.
    """
    if full_view.get("snapshot_complete") is not True:
        return {"snapshot_complete": False, "reason": "complete worker state unavailable"}
    rows = []
    for worker in full_view.get("workers", []):
        assessed = worker.get("progress") or {}
        row = {
            "id": worker.get("worker_id"),
            "status": worker.get("status"),
            "task": worker.get("task"),
            "task_excerpt": bool(worker.get("task_is_excerpt")),
            "work_mode": worker.get("completion_mode"),
            "model_route": worker.get("model_route"),
            "started_at": worker.get("started_at"),
            "running_seconds": worker.get("elapsed_seconds"),
            "last_report_seconds_ago": worker.get("report_age_seconds"),
            "last_reported_work": worker.get("reported_activity"),
            "next_reported_action": worker.get("reported_next_action"),
            "live_model_operation": worker.get("supervisor_operation"),
            "live_model_phase": worker.get("supervisor_phase"),
            "worker_percent": assessed.get("overall_percent"),
            "cycle_percent": assessed.get("cycle_percent"),
            "estimated_seconds_remaining_at_report": assessed.get("remaining_seconds_at_assessment"),
            "estimate_seconds_old": assessed.get("estimate_age_seconds"),
            "estimate_overdue": assessed.get("estimate_deadline_passed"),
            "percent_basis": assessed.get("progress_basis"),
        }
        rows.append({key: val for key, val in row.items() if val is not None
                     or key in {"worker_percent", "estimated_seconds_remaining_at_report"}})
    plans = []
    for plan in full_view.get("plans", []):
        progress = plan.get("plan_node_progress") or {}
        plans.append({
            "id": plan.get("plan_id"), "status": plan.get("status"),
            "task": plan.get("task"),
            "age_seconds": plan.get("elapsed_seconds_since_creation"),
            "plan_node_percent_NOT_worker_percent": progress.get("finite_nodes_percent"),
            "plan_work_kind": progress.get("work_kind"),
            "completed_finite_nodes": progress.get("completed_finite_nodes"),
            "total_finite_nodes": progress.get("total_finite_nodes"),
            "nodes": plan.get("nodes"),
        })
    return {
        "snapshot_complete": True,
        "active_worker_count": full_view.get("active_worker_count"),
        "worker_status_counts": full_view.get("active_worker_status_counts"),
        "terminal_status_counts": full_view.get("terminal_worker_status_counts"),
        "workers": rows,
        "active_plan_count": full_view.get("active_plan_count"),
        "plans": plans,
        "recent_terminal_workers": full_view.get("recent_terminal_workers"),
        "open_questions": full_view.get("open_worker_question_count"),
        "question_inventory_complete": full_view.get("worker_question_inventory_complete"),
        "question_samples": full_view.get("worker_question_summaries"),
        "host_storage": full_view.get("host_storage"),
        "source_limits": "worker reports are model-authored; percent/ETA null means unknown; node percent is not individual task progress",
    }


def meaningful_worker_state_revision(state: dict[str, Any] | None) -> tuple | None:
    """Ignore capture clock/token telemetry, fence substantive job changes."""
    if not isinstance(state, dict) or state.get("snapshot_complete") is not True:
        return None
    workers = tuple(sorted((
        str(w.get("worker_id")), str(w.get("status")), str(w.get("task")),
        str(w.get("latest_report_at")), str((w.get("progress") or {}).get("assessment_at")),
        str((w.get("progress") or {}).get("overall_percent")),
    ) for w in state.get("workers", [])))
    plans = tuple(sorted((
        str(p.get("plan_id")), str(p.get("status")), str(p.get("updated_at")),
        tuple(sorted((str(n.get("node_id")),str(n.get("state")),str(n.get("worker_id")))
                     for n in p.get("nodes") or [])),
    ) for p in state.get("plans", [])))
    terminal = tuple((str(w.get("worker_id")),str(w.get("status")),str(w.get("updated_at")))
                     for w in state.get("recent_terminal_workers", []))
    return workers, plans, terminal, state.get("open_worker_question_count")
