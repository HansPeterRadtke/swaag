"""Explicit model-assessed progress; no percentage inferred from token counts."""
from __future__ import annotations

import math


def validate_progress(payload):
    if not isinstance(payload, dict) or set(payload) != {'steps', 'reason', 'estimated_remaining_seconds', 'estimate_conditions'}:
        raise ValueError('Progress requires exactly steps, reason, estimated_remaining_seconds, estimate_conditions')
    if not isinstance(payload['reason'], str) or not payload['reason'].strip():
        raise ValueError('Progress needs a reason supported by current evidence')
    steps = payload['steps']
    if not isinstance(steps, list) or not 1 <= len(steps) <= 128:
        raise ValueError('Progress requires one to 128 explicit steps')
    seen = set()
    for step in steps:
        if not isinstance(step, dict) or set(step) != {'id', 'label', 'state', 'weight'}:
            raise ValueError('Each step requires exactly id, label, state, weight')
        if any(not isinstance(step[key], str) or not step[key].strip() for key in ('id', 'label')):
            raise ValueError('Progress step id and label must be nonempty')
        if step['id'] in seen or step['state'] not in ('pending', 'in_progress', 'completed'):
            raise ValueError('Progress step IDs must be unique and states valid')
        seen.add(step['id'])
        weight = step['weight']
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('Progress weights must be finite positive numbers')
    estimate = payload['estimated_remaining_seconds']
    if estimate is not None and (isinstance(estimate, bool) or not isinstance(estimate, (int, float)) or not math.isfinite(estimate) or estimate < 0):
        raise ValueError('Remaining duration must be null or a finite nonnegative number')
    conditions = payload['estimate_conditions']
    if not isinstance(conditions, str) or (estimate is not None and not conditions.strip()):
        raise ValueError('A time estimate requires conditions describing model, input, machine, and load; otherwise use null')
    return dict(payload)


def step_percentage(progress):
    steps = progress.get('steps', [])
    total = sum(step['weight'] for step in steps)
    return 100 * sum(step['weight'] for step in steps if step['state'] == 'completed') / total if total else None


def worker_progress(record, state):
    endless = record.completion_mode == 'continuous'
    assessment = dict(state.progress)
    return {'work_kind': 'intentionally_endless' if endless else 'finite',
            'overall_percent': None if endless else (100.0 if record.status == 'completed' else step_percentage(assessment)),
            'current_cycle_percent': step_percentage(assessment) if endless else None,
            'basis': 'explicit model-assessed step weights; completed status is runtime evidence',
            'assessment': assessment or None,
            'estimated_remaining_seconds': None if endless else assessment.get('estimated_remaining_seconds'),
            'started_at': record.started_at, 'completed_at': record.completed_at}


def plan_progress(nodes):
    finite = [node for node in nodes if node.get('completion_mode', 'natural') != 'continuous']
    completed = sum(node['state'] == 'completed' for node in finite)
    endless = len(finite) != len(nodes)
    percent = 100 * completed / len(finite) if finite else None
    return {'work_kind': 'contains_intentionally_endless_work' if endless else 'finite',
            'overall_percent': None if endless else percent,
            'finite_nodes_percent': percent, 'completed_finite_nodes': completed,
            'total_finite_nodes': len(finite), 'continuous_nodes': len(nodes) - len(finite),
            'basis': 'completed finite nodes, equally weighted; not elapsed time or estimated effort'}
