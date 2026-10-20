"""Precompute truthful live worker reports and plan graph, never on the voice call."""
from __future__ import annotations
from copy import deepcopy


def build_active_worker_reports(workers, managers):
    reports = {}
    for worker in workers:
        if worker.status in ('completed', 'failed', 'canceled'):
            continue
        owner = managers.get(worker.model_key, managers['default'])
        # The most recent status is not necessarily the most recent explicit
        # progress assessment. Preserve each independently, so a later agent
        # status update cannot erase a worker's measured step plan and ETA.
        history = owner.runtime.history
        latest = next(history.iter_history_reverse(
            worker.session_id,
            event_types=('agent_status', 'agent_progress', 'assistant_progress', 'worker_input_required'),
        ), None)
        latest_status = next(history.iter_history_reverse(
            worker.session_id, event_types=('agent_status', 'worker_input_required'),
        ), None)
        latest_progress = next(history.iter_history_reverse(
            worker.session_id, event_types=('agent_progress',),
        ), None)
        if latest is not None:
            reports[worker.worker_id] = {
                'event_type': latest.event_type, 'event_sequence': latest.sequence,
                'timestamp': latest.timestamp, 'payload': deepcopy(latest.payload),
                'evidence_scope': 'worker-authored status, not externally verified effects',
                'latest_status': ({
                    'event_type': latest_status.event_type, 'event_sequence': latest_status.sequence,
                    'timestamp': latest_status.timestamp, 'payload': deepcopy(latest_status.payload),
                } if latest_status is not None else None),
                'latest_progress': ({
                    'event_type': latest_progress.event_type, 'event_sequence': latest_progress.sequence,
                    'timestamp': latest_progress.timestamp, 'payload': deepcopy(latest_progress.payload),
                } if latest_progress is not None else None),
            }
    return reports


def build_active_plan_graphs(plans, store):
    graphs = {}
    for plan in plans:
        if plan.status in ('completed', 'canceled'):
            continue
        with store._connect() as connection:
            nodes = [dict(row) for row in connection.execute(
                'SELECT * FROM orchestration_nodes WHERE plan_id=? ORDER BY created_at,node_id', (plan.plan_id,))]
            edges = [dict(row) for row in connection.execute(
                'SELECT * FROM orchestration_edges WHERE plan_id=? ORDER BY created_at,edge_id', (plan.plan_id,))]
        graphs[plan.plan_id] = {'nodes': nodes, 'edges': edges}
    return graphs
