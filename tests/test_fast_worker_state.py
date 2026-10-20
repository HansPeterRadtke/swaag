"""Fast conversation must see the meaningful live worker state, not mere IDs."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

from swaag.fast_worker_state import build_fast_worker_state, fast_worker_state_at_prompt
from swaag.foreground_world_builder import build_active_worker_reports
from swaag.foreground_fast import generate_fast_foreground_response
from tests.test_fast_foreground_reply import fast, setup


NOW = datetime(2026, 10, 20, 18, 30, tzinfo=timezone.utc)


def when(seconds):
    return (NOW - timedelta(seconds=seconds)).isoformat()


def snapshot(*, continuous=False, with_progress=True, active_workers=1):
    workers = []
    reports = {}
    for i in range(active_workers):
        ident = f'worker-{i + 1}'
        workers.append({
            'worker_id': ident, 'session_id': f'session-{i + 1}',
            'status': 'working', 'model_key': 'default',
            'completion_mode': 'continuous' if continuous else 'natural',
            'objective': f'Improve warehouse photorealistic rendering {i + 1}',
            'started_at': when(2 * 86400), 'created_at': when(2 * 86400),
            'updated_at': when(2 * 86400), 'run_count': 1,
        })
        reports[ident] = {
            'event_type': 'agent_status', 'timestamp': when(3600),
            'payload': {'just_happened': 'Inspected the renderer lighting configuration',
                        'action': 'Inspect ray tracing materials next'},
            'latest_status': {
                'event_type': 'agent_status', 'timestamp': when(3600),
                'payload': {'just_happened': 'Inspected the renderer lighting configuration',
                            'action': 'Inspect ray tracing materials next'},
            },
        }
        if with_progress:
            reports[ident]['latest_progress'] = {
                'event_type': 'agent_progress', 'timestamp': when(4500),
                'payload': {
                    'steps': [
                        {'id': '1', 'label': 'Inspect', 'state': 'completed', 'weight': 1},
                        {'id': '2', 'label': 'Improve', 'state': 'in_progress', 'weight': 3},
                    ],
                    'reason': 'Verified progress report',
                    'estimated_remaining_seconds': 20,
                    'estimate_conditions': 'Q8, current machine, project input, known load',
                },
            }
    nodes = [
        {'node_id': 'node-a', 'state': 'completed', 'completion_mode': 'natural',
         'worker_id': None, 'objective': 'Inspect existing pipeline'},
        {'node_id': 'node-b', 'state': 'working', 'completion_mode': 'continuous' if continuous else 'natural',
         'worker_id': 'worker-1', 'objective': 'Improve warehouse quality'},
    ]
    return {
        'captured_at': NOW.isoformat(),
        'active_workers': workers,
        'terminal_workers': [{'worker_id': 'finished-worker', 'status': 'completed',
                              'objective': 'Prepare images', 'updated_at': when(200)}],
        'active_worker_reports': reports,
        'active_plans': [{'plan_id': 'plan-live', 'status': 'active',
                          'objective': 'Complete warehouse work',
                          'created_at': when(2 * 86400), 'updated_at': when(30)}],
        'active_plan_graphs': {'plan-live': {'nodes': nodes, 'edges': []}},
        'terminal_plans': [],
        'worker_controls': [],
        'worker_question_inventory': {'questions': [], 'complete': True, 'scope': 'all workers', 'workers': [], 'pending_revisions': []},
        'supervision': {'active_sessions': [{'session_id': 'session-1',
            'phase': 'inference', 'substate': 'streaming', 'operation_kind': 'action',
            'detail': 'receiving model tokens', 'heartbeat_age_seconds': 1,
            'heartbeat_is_computation_evidence': False}]},
        'model_routing': {'default_worker': {'requested_model_id': 'qwen38-27b-q8-full-262k'}},
        'orchestrator': {'active_turn': None},
    }


def test_worker_state_tracks_full_task_age_last_work_progress_and_overdue_estimate():
    raw = build_fast_worker_state(snapshot())
    data = fast_worker_state_at_prompt(raw, now=NOW)
    assert data['snapshot_complete'] and data['active_worker_count'] == 1
    w = data['workers'][0]
    assert w['task'] == 'Improve warehouse photorealistic rendering 1'
    assert w['elapsed_seconds'] == 172800  # two days, not model token count
    assert w['report_age_seconds'] == 3600
    assert w['reported_activity'] == 'Inspected the renderer lighting configuration'
    assert w['reported_next_action'] == 'Inspect ray tracing materials next'
    assert w['supervisor_operation'] == 'action'
    assert w['progress']['overall_percent'] == 25.0
    assert w['progress']['remaining_seconds_at_assessment'] == 20
    assert w['progress']['estimate_age_seconds'] == 4500
    assert w['progress']['estimate_deadline_passed'] is True
    assert data['plans'][0]['plan_node_progress']['finite_nodes_percent'] == 50
    assert data['plans'][0]['elapsed_seconds_since_creation'] == 172800
    assert data['recent_terminal_workers'][0]['status'] == 'completed'


def test_worker_without_step_assessment_has_no_invented_percent_or_eta():
    data = fast_worker_state_at_prompt(build_fast_worker_state(snapshot(with_progress=False)), now=NOW)
    w = data['workers'][0]
    assert w['elapsed_seconds'] == 172800
    assert w['progress']['overall_percent'] is None
    assert w['progress']['remaining_seconds_at_assessment'] is None
    assert w['progress']['estimate_deadline_passed'] is None
    assert 'no valid' in w['progress']['progress_basis']


def test_permanent_background_worker_reports_cycle_not_overall_completion():
    data = fast_worker_state_at_prompt(build_fast_worker_state(snapshot(continuous=True)), now=NOW)
    w = data['workers'][0]
    assert w['completion_mode'] == 'continuous'
    assert w['progress']['overall_percent'] is None
    assert w['progress']['cycle_percent'] == 25.0
    assert w['progress']['remaining_seconds_at_assessment'] is None
    assert data['plans'][0]['plan_node_progress']['work_kind'] == 'contains_intentionally_endless_work'
    assert data['plans'][0]['plan_node_progress']['finite_nodes_percent'] == 100


def test_all_active_worker_identities_and_objectives_are_included():
    raw = build_fast_worker_state(snapshot(active_workers=9, with_progress=False))
    assert raw['active_worker_count'] == len(raw['workers']) == 9
    assert {w['worker_id'] for w in raw['workers']} == {f'worker-{i}' for i in range(1, 10)}
    assert all('photorealistic' in w['task'] for w in raw['workers'])


def test_broken_or_partial_snapshots_do_not_claim_known_worker_status():
    assert build_fast_worker_state({'active_workers': [], 'active_plans': []})['snapshot_complete'] is False
    assert build_fast_worker_state(None)['snapshot_complete'] is False
    x = snapshot()
    x['active_plans'] = 'not a list'
    assert build_fast_worker_state(x)['snapshot_complete'] is False


def test_worker_status_builder_retains_last_explicit_progress_after_newer_status():
    from swaag.types import HistoryEvent
    events = [
        SimpleNamespace(event_type='agent_status', sequence=13, timestamp=when(30), payload={'action':'Continue'}),
        SimpleNamespace(event_type='agent_progress', sequence=12, timestamp=when(200), payload={'steps': []}),
    ]
    class History:
        def iter_history_reverse(self, session_id, *, event_types):
            return (ev for ev in events if ev.event_type in event_types)
    manager = SimpleNamespace(runtime=SimpleNamespace(history=History()))
    worker = SimpleNamespace(worker_id='worker-1', status='working', model_key='default', session_id='session-1')
    rows = build_active_worker_reports([worker], {'default': manager})
    assert rows['worker-1']['event_type'] == 'agent_status'
    assert rows['worker-1']['latest_progress']['event_sequence'] == 12
    assert rows['worker-1']['latest_status']['event_sequence'] == 13


def test_fast_llm_prompt_contains_status_without_additional_tool_discovery(make_config):
    from datetime import datetime, timezone
    from dataclasses import replace
    from unittest.mock import patch
    service, runtime, client = setup(make_config, [fast('answer', 'One warehouse worker is active.')])
    try:
        # Rebase synthetic observation without touching production worker state.
        world = snapshot()
        world['captured_at'] = datetime.now(timezone.utc).isoformat()
        state = service._orchestrator_state()
        result = generate_fast_foreground_response(runtime, state,
            'How many workers are running and what are they doing?', observed_world=world)
        assert result and result.mode == 'answer'
        prompt = client.requests[0]['prompt']
        assert 'Improve warehouse photorealistic rendering 1' in prompt
        assert 'running_seconds' in prompt
        assert 'worker_percent' in prompt
        assert 'estimated_remaining_seconds' not in prompt or 'remaining_seconds_at_assessment' in prompt
        assert [x['contract'] for x in client.requests] == ['fast_foreground_response']
    finally:service.close()


def test_fast_status_reply_saying_worker_running_does_not_trigger_delegation(make_config):
    service, runtime, client = setup(make_config, [
        fast('answer', 'One warehouse worker is running and has no known ETA.'),
    ])
    try:
        state = service._orchestrator_state()
        world = snapshot(with_progress=False)
        world['captured_at'] = datetime.now(timezone.utc).isoformat()
        response = generate_fast_foreground_response(runtime, state,
            'Which workers are currently running?', observed_world=world)
        assert response and response.mode == 'answer'
        assert len(client.requests) == 1
        assert 'One warehouse worker is running' in response.reply
    finally:
        service.close()


def test_background_refresh_precomputes_worker_summary_for_quick_turn(make_config):
    service, runtime, client = setup(make_config, [
        fast('answer', 'A warehouse worker is doing rendering improvements.'),
    ])
    try:
        current = snapshot(with_progress=False)
        current['captured_at'] = datetime.now(timezone.utc).isoformat()
        service._build_orchestrator_runtime_snapshot = lambda: current
        service._foreground_task_revision = lambda: ('stable_revision',)
        service._refresh_orchestrator_snapshot()
        assert service._fast_foreground_state['snapshot_complete']
        assert len(service._fast_foreground_state['workers']) == 1
        assert service._fast_foreground_state['workers'][0]['task'].startswith('Improve warehouse')
        state = service._orchestrator_state()
        receipt = service._receive_orchestrator_message('What is the warehouse worker working on?',
                                                       control_id='worker_summary')
        result = service._try_fast_foreground_reply(state, receipt)
        assert result and result['mode'] == 'answer'
        assert len(client.requests) == 1
        assert 'Improve warehouse photorealistic rendering 1' in client.requests[0]['prompt']
    finally:
        service.close()


def test_changed_durable_worker_revision_invalidates_cached_fast_status(make_config):
    service, runtime, client = setup(make_config, [
        fast('delegate', 'I need to check the current worker status.'),
    ])
    try:
        current = snapshot(with_progress=False)
        current['captured_at'] = datetime.now(timezone.utc).isoformat()
        service._build_orchestrator_runtime_snapshot = lambda: current
        service._foreground_task_revision = lambda: (1,)
        service._refresh_orchestrator_snapshot()
        service._foreground_task_revision = lambda: (2,)
        state = service._orchestrator_state()
        receipt = service._receive_orchestrator_message('What is running right now?',
                                                       control_id='changed_revision')
        result = service._try_fast_foreground_reply(state, receipt)
        assert result and result['mode'] == 'delegate'
        assert 'Improve warehouse photorealistic rendering' not in client.requests[0]['prompt']
        assert 'not supplied' in client.requests[0]['prompt']
    finally:
        service.close()


def test_small_prompt_projection_preserves_every_worker_and_explicit_unknown_progress():
    from swaag.fast_worker_state import compact_worker_prompt_view
    raw = build_fast_worker_state(snapshot(active_workers=9, with_progress=False))
    projected = compact_worker_prompt_view(fast_worker_state_at_prompt(raw, now=NOW))
    assert projected['active_worker_count'] == len(projected['workers']) == 9
    assert {w['id'] for w in projected['workers']} == {f'worker-{i}' for i in range(1, 10)}
    for w in projected['workers']:
        assert w['worker_percent'] is None
        assert w['estimated_seconds_remaining_at_report'] is None
        assert w['running_seconds'] == 172800
        assert w['last_report_seconds_ago'] == 3600
        assert 'photorealistic' in w['task']
    assert 'plan_node_percent_NOT_worker_percent' in projected['plans'][0]
    assert projected['plans'][0]['plan_node_percent_NOT_worker_percent'] == 50


def test_small_prompt_projection_retains_expired_and_indefinite_estimate_distinction():
    from swaag.fast_worker_state import compact_worker_prompt_view
    finite = compact_worker_prompt_view(fast_worker_state_at_prompt(
        build_fast_worker_state(snapshot()), now=NOW))
    assert finite['workers'][0]['worker_percent'] == 25.0
    assert finite['workers'][0]['estimated_seconds_remaining_at_report'] == 20
    assert finite['workers'][0]['estimate_overdue'] is True
    continuous = compact_worker_prompt_view(fast_worker_state_at_prompt(
        build_fast_worker_state(snapshot(continuous=True)), now=NOW))
    assert continuous['workers'][0]['work_mode'] == 'continuous'
    assert continuous['workers'][0]['worker_percent'] is None
    assert continuous['workers'][0]['cycle_percent'] == 25.0
    assert continuous['workers'][0]['estimated_seconds_remaining_at_report'] is None


def test_worker_status_change_during_fast_inference_rejects_old_spoken_answer(make_config):
    from tests.test_agent_action_loop import FakeModelClient, _action
    from swaag.communication import CommunicationService
    from swaag.runtime import AgentRuntime

    cfg = make_config(model__context_limit=32000,
        runtime__completion_evaluation_enabled=False,
        communication__fast_initial_reply=True,
        communication__poll_seconds=0.01)
    holder = {}
    def stale_response(request):
        holder['service']._foreground_task_revision = lambda: (2,)
        return fast('answer', 'The worker is still running.')
    client = FakeModelClient([stale_response, _action(message='Worker status changed; checking the fresh state.')])
    runtime = AgentRuntime(cfg, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    holder['service'] = service
    try:
        world = snapshot()
        world['captured_at'] = datetime.now(timezone.utc).isoformat()
        service._build_orchestrator_runtime_snapshot = lambda: world
        service._foreground_task_revision = lambda: (1,)
        service._refresh_orchestrator_snapshot()
        reply = service.orchestrator_message('What are the workers doing?')
        assert reply['answer'] == 'Worker status changed; checking the fresh state.'
        events = runtime.history.read_history(reply['session_id'])
        assert all(e.payload.get('assistant_text') != 'The worker is still running.'
                   for e in events if e.event_type == 'turn_finished')
        assert [r['contract'] for r in client.requests] == ['fast_foreground_response','agent_action']
    finally:
        service.close()


def test_progress_report_change_during_fast_generation_rejects_stale_percentage(make_config):
    from tests.test_agent_action_loop import FakeModelClient, _action
    from swaag.communication import CommunicationService
    from swaag.runtime import AgentRuntime
    cfg = make_config(model__context_limit=32000,
        runtime__completion_evaluation_enabled=False,
        communication__fast_initial_reply=True,
        communication__poll_seconds=0.01)
    holder = {}
    def changed_report(request):
        from copy import deepcopy
        service = holder['service']
        updated = deepcopy(service._fast_foreground_state)
        updated['workers'][0]['progress']['assessment_at'] = when(1)
        with service._foreground_snapshot_lock:
            service._fast_foreground_state = updated
        return fast('answer', 'The worker was twenty-five percent complete.')
    client=FakeModelClient([changed_report, _action(message='Rechecking updated progress.')])
    runtime=AgentRuntime(cfg, model_client=client)
    service=CommunicationService(runtime, orchestrator_runtime=runtime)
    holder['service']=service
    try:
        world=snapshot()
        world['captured_at'] = datetime.now(timezone.utc).isoformat()
        service._build_orchestrator_runtime_snapshot = lambda: world
        service._foreground_task_revision=lambda: (1,)
        service._refresh_orchestrator_snapshot()
        answer=service.orchestrator_message('What percentage is complete?')
        assert answer['answer'] == 'Rechecking updated progress.'
        assert [r['contract'] for r in client.requests] == ['fast_foreground_response','agent_action']
    finally:service.close()
