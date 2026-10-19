"""Precompute truthful live worker reports and plan graph, never on the voice call."""
from __future__ import annotations
from copy import deepcopy


def build_active_worker_reports(workers, managers):
    reports = {}
    for worker in workers:
        if worker.status in ('completed', 'failed', 'canceled'):
            continue
        owner = managers.get(worker.model_key, managers['default'])
        event = next(owner.runtime.history.iter_history_reverse(
            worker.session_id, event_types=('agent_status','agent_progress','assistant_progress','worker_input_required')), None)
        if event is not None:
            reports[worker.worker_id] = {
                'event_type': event.event_type, 'event_sequence': event.sequence,
                'timestamp': event.timestamp, 'payload': deepcopy(event.payload),
                'evidence_scope': 'worker-authored status, not externally verified effects',
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
