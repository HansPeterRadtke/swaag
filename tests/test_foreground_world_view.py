"""Visible SWAAG world-state summary is compact, faithful and source-backed."""
from __future__ import annotations

from copy import deepcopy

from swaag.foreground_world_view import foreground_world_view
from swaag.snapshot_view import snapshot_value


def _source():
    return {
        'captured_at': '2026-10-19T10:00:00Z',
        'consistency': 'per-source observations',
        'model_routing': {'foreground': {'requested_model_id': 'coder'},
                          'default_worker': {'requested_model_id': 'q8'},
                          'available_worker_model_keys': ['default']},
        'orchestrator': {'session_id': 'session_1', 'active_turn': {'user_text': 'How is A?'}},
        'active_workers': [
            {'worker_id': 'worker_A', 'objective': 'Do task A; NEVER PUBLISH before tests.',
             'status': 'working', 'model_key': 'default', 'completion_mode': 'natural',
             'started_at': '10:00', 'updated_at': '10:04', 'run_count': 2,
             'inference_weight': 1.0},
            {'worker_id': 'worker_B', 'objective': 'Do task B; do not publish without approval.',
             'status': 'input_required', 'model_key': 'default', 'updated_at': '10:05'},
        ],
        'terminal_workers': [
            {'worker_id': 'old', 'status': 'failed', 'updated_at': '08:00', 'objective': 'old', 'error': 'OLD ERROR'},
            {'worker_id': 'new', 'status': 'completed', 'updated_at': '09:00', 'objective': 'finished', 'result': 'DONE'},
        ],
        'active_plans': [{'plan_id': 'plan_1', 'objective': 'Ship warehouse parts safely',
                          'status': 'running', 'revision': 2, 'updated_at': '10:04'}],
        'active_plan_graphs': {'plan_1': {'nodes': [
            {'node_id': 'node_1', 'worker_id': 'worker_A', 'objective': 'Work A', 'state': 'working'},
            {'node_id': 'node_2', 'worker_id': None, 'objective': 'Wait for A', 'state': 'queued'},
        ], 'edges': [{'source_node_id': 'node_1', 'target_node_id': 'node_2', 'condition': {}}]}},
        'terminal_plans': [{'plan_id': 'old_plan', 'status': 'completed', 'objective': 'old'}],
        'active_worker_reports': {'worker_A': {'event_type': 'assistant_progress',
            'timestamp': '10:03', 'payload': {'assistant_text': 'Verified 3 of 5 files; no writes.'}}},
        'worker_controls': [{'worker_id': 'worker_B', 'pending_controls': [
            {'message': 'Do not publish yet'}]}],
        'worker_question_inventory': {'complete': True, 'questions': [
            {'worker_id': 'worker_B', 'criticality': 'blocking', 'question': 'Approve deployment?'}],
            'workers': []},
        'supervision': {'backends': {'coder': {'status': 'processing'},
                                     'q8': {'status': 'idle'}}},
        'unknown_future_field': {'critical': 'STORE_IN_ORIGINAL'},
    }


def test_compact_world_includes_every_live_worker_goal_report_node_and_dependency():
    source=_source(); original=deepcopy(source)
    view=foreground_world_view(source, 42)
    assert source==original
    assert view['model_routing']==source['model_routing']
    assert [w['worker_id'] for w in view['active_workers']]==['worker_A', 'worker_B']
    assert view['active_workers'][0]['objective']=='Do task A; NEVER PUBLISH before tests.'
    assert view['active_workers'][0]['last_worker_report']==source['active_worker_reports']['worker_A']
    assert view['active_workers'][1]['status']=='input_required'
    assert view['active_plans'][0]['nodes']==source['active_plan_graphs']['plan_1']['nodes']
    assert view['active_plans'][0]['edges']==source['active_plan_graphs']['plan_1']['edges']
    assert view['current_task_goals'][0]['objective']=='Ship warehouse parts safely'
    assert view['current_task_goals'][1]['objective']=='Do task B; do not publish without approval.'
    assert view['worker_controls']==source['worker_controls']
    assert view['supervision']==source['supervision']
    assert view['terminal_workers']['count']==2
    assert view['terminal_workers']['recent_outcomes'][0]['result']=='DONE'
    assert view['detail_access']['snapshot_sequence']==42
    assert 'unknown_future_field' not in view
    assert snapshot_value(source,'/unknown_future_field/critical')=='STORE_IN_ORIGINAL'


def test_empty_world_is_short_and_does_not_invent_workers_or_goals():
    source=_source()
    for field in ('active_workers','active_plans','terminal_workers','terminal_plans','worker_controls'):
        source[field]=[]
    source['active_worker_reports']={};source['active_plan_graphs']={}
    view=foreground_world_view(source,3)
    assert view['current_task_goals']==[]
    assert view['active_workers']==[]
    assert view['active_plans']==[]
    assert view['terminal_workers']['count']==0
    assert view['detail_access']['snapshot_sequence']==3


def test_many_terminal_records_do_not_replace_live_workers_or_hide_source():
    source=_source()
    source['terminal_workers']=[{'worker_id': f'old_{i}', 'status':'completed',
                                 'updated_at': f'2026-10-{i % 28 + 1:02d}',
                                 'result': 'OLD_RESULT_' + str(i)} for i in range(150)]
    view=foreground_world_view(source,7)
    assert len(view['active_workers'])==2
    assert view['terminal_workers']['count']==150
    assert len(view['terminal_workers']['recent_outcomes'])==3
    assert view['terminal_workers']['exact_details_pointer']=='/terminal_workers'
    assert snapshot_value(source, '/terminal_workers/0/result')=='OLD_RESULT_0'


def test_report_builder_reads_last_real_worker_event_once():
    from types import SimpleNamespace
    from swaag.foreground_world_builder import build_active_worker_reports
    class History:
        def __init__(self):self.calls=[]
        def iter_history_reverse(self,session_id,*,event_types):
            self.calls.append((session_id,tuple(event_types)))
            return iter([SimpleNamespace(event_type='agent_progress',sequence=81,timestamp='now',
                                         payload={'status':'Checking exact files','progress':{'completed':3,'total':5}})])
    history=History()
    worker=SimpleNamespace(worker_id='worker_one',session_id='s-one',model_key='default',status='working')
    terminal=SimpleNamespace(worker_id='worker_old',session_id='s-old',model_key='default',status='completed')
    managers={'default': SimpleNamespace(runtime=SimpleNamespace(history=history))}
    result=build_active_worker_reports([worker,terminal],managers)
    assert list(result)==['worker_one']
    assert result['worker_one']['payload']['status']=='Checking exact files'
    assert result['worker_one']['event_sequence']==81
    assert len(history.calls)==1 and history.calls[0][0]=='s-one'
