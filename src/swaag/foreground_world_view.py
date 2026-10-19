"""Small authoritative SWAAG world view for immediate user-facing LLM calls.

The full exact snapshot is still durably recorded and recoverable with
runtime_snapshot_lookup. Do not put old task logs, full worker histories or
unselected capability schemas in a routine voice prompt. Live goals, workers,
plans, blockers and their latest reports remain explicit and untruncated.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any


def _fields(record: dict, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: deepcopy(record[name]) for name in names if name in record}


def foreground_world_view(snapshot: dict[str, Any], sequence: int) -> dict[str, Any]:
    """Expose all live state and source pointers without erasing raw evidence.

    Terminal work is aggregated, with a small recent outcome index. Explicit
    detailed/historical requests can open the saved full snapshot by sequence.
    No natural-language goal or worker result is reinterpreted mechanically.
    """
    from swaag.snapshot_view import snapshot_prompt_view

    view = snapshot_prompt_view(snapshot, sequence)
    active_workers = view.get('active_workers', [])
    active_plans = view.get('active_plans', [])
    reports = snapshot.get('active_worker_reports', {})
    graphs = snapshot.get('active_plan_graphs', {})
    workers = []
    for original in active_workers:
        record = _fields(original, (
            'worker_id', 'objective', 'status', 'model_key', 'completion_mode',
            'created_at', 'updated_at', 'started_at', 'result', 'error'))
        report = reports.get(original.get('worker_id'))
        if report is not None:
            record['last_worker_report'] = deepcopy(report)
        workers.append(record)
    plans = []
    for original in active_plans:
        plan = _fields(original, ('plan_id', 'objective', 'status', 'revision',
                                  'scheduling_mode', 'max_parallel', 'reporting_mode', 'updated_at'))
        graph = graphs.get(original.get('plan_id'))
        if graph is not None:
            plan['nodes'] = deepcopy(graph.get('nodes', []))
            plan['edges'] = deepcopy(graph.get('edges', []))
        plans.append(plan)
    recent = sorted(snapshot.get('terminal_workers', []),
                    key=lambda row: (str(row.get('updated_at') or ''),
                                     str(row.get('worker_id') or '')), reverse=True)[:3]
    recent_outcomes = [_fields(row, ('worker_id', 'status', 'objective', 'updated_at',
                                     'result', 'error')) for row in recent]
    # A model may inspect exact details and historic worker output through
    # the full saved snapshot. Do not claim the abbreviated index is complete.
    terminal = deepcopy(view.get('terminal_workers', {}))
    if isinstance(terminal, dict):
        terminal['recent_outcomes'] = recent_outcomes
    output = {
        'captured_at': view.get('captured_at'),
        'consistency': view.get('consistency'),
        'model_routing': deepcopy(view.get('model_routing', {})),
        'orchestrator': deepcopy(view.get('orchestrator', {})),
        'current_task_goals': [
            {'source': 'active_plan', 'plan_id': plan.get('plan_id'), 'objective': plan.get('objective')}
            for plan in plans
        ] + [
            {'source': 'unplanned_worker', 'worker_id': worker.get('worker_id'),
             'objective': worker.get('objective')}
            for worker in workers if not any(
                node.get('worker_id') == worker.get('worker_id')
                for plan in plans for node in plan.get('nodes', []))
        ],
        'active_workers': workers,
        'active_plans': plans,
        'terminal_workers': terminal,
        'terminal_plans': deepcopy(view.get('terminal_plans', {})),
        'worker_controls': deepcopy(view.get('worker_controls', [])),
        'terminal_control_details': deepcopy(view.get('terminal_control_details', {})),
        'supervision': deepcopy(view.get('supervision', {})),
        'detail_access': deepcopy(view['detail_access']),
        'preserved_exact_fields': {
            'full_snapshot': '',
            'worker_questions': '/worker_question_inventory',
            'worker_latest_result': '/terminal_workers',
            'raw_plan_nodes': '/active_plan_graphs',
            'raw_worker_reports': '/active_worker_reports',
            'user_preferences': 'durable conversation and scoped notes; never inferred from worker metadata',
        },
    }
    return output
