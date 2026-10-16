"""Foreground history disclosure must retain authority and recover exact effects."""
import json
from dataclasses import asdict, replace

import pytest

from swaag.evidence_sources import (ForegroundHistoryEvidenceProvider,
    RuntimeTelemetryEvidenceProvider, is_foreground_observation_event)
from swaag.history import HistoryInvariantError
from swaag.runtime import AgentRuntime
from swaag.types import Message
from tests.test_agent_action_loop import FakeModelClient
from tests.test_completion_evaluation import _completed_action


def _prior_turn(runtime, state):
    authority = '  Keep the report private.\nDo not publish without approval.  '
    runtime._record_message(state, Message(role='user', content=authority, created_at='earlier'))
    runtime.history.record_event(state, 'turn_started', {'turn_index':1, 'user_text':authority})
    snapshot = runtime.history.record_event(state, 'runtime_system_snapshot', {
        'snapshot': {'historical_worker': 'EXACT_OLD_STATE', 'observations': ['retained measurement'] * 2000}})
    control = runtime.history.record_event(state, 'control_message_processed', {
        'control_id':'prior_control', 'session_id':state.session_id,
        'message':'  Continue the checks, but keep every result private.  ',
        'decision':{'draft':'UNVERIFIED_MODEL_PLAN'}, 'selected_action':{'draft':'UNVERIFIED_MODEL_PLAN'}})
    result = runtime.history.record_event(state, 'tool_result', {
        'tool_name':'run_tests', 'raw_input':{}, 'validated_input':{},
        'output':{'passed':False, 'reason':'EXACT_OLD_FAILURE', 'details':'The check failed.'}})
    runtime._record_message(state, Message(role='assistant', content='The prior check failed.', created_at='earlier'))
    return authority, [snapshot, control, result]


def _current_turn(runtime, state, text, *, snapshot=True):
    runtime._record_message(state, Message(role='user', content=text, created_at='now'))
    runtime.history.record_event(state, 'turn_started', {'turn_index':2, 'user_text':text})
    if snapshot:
        runtime.bind_tool_runtime_capability(state.session_id, 'system_snapshot_provider',
            lambda:{'captured_at':'now', 'active_workers':[], 'fresh_fact':'EXACT_CURRENT_STATE'})


def test_greeting_keeps_authority_and_current_state_without_reducing_old_observations(make_config):
    holder = {}
    def verdict(payload):
        prompt = payload['prompt']
        assert json.dumps(holder['authority'], ensure_ascii=True)[1:-1] in prompt
        assert 'Continue the checks, but keep every result private.' in prompt
        assert 'The prior check failed.' in prompt
        assert 'EXACT_CURRENT_STATE' in prompt
        assert 'EXACT_OLD_STATE' not in prompt and 'EXACT_OLD_FAILURE' not in prompt
        assert 'UNVERIFIED_MODEL_PLAN' not in prompt
        assert 'foreground_history' in prompt and 'run_tests' in prompt
        return json.dumps({'complete':True, 'reason':'The greeting is answered.',
                           'remaining_work':[], 'evidence_requests':[]})
    client = FakeModelClient([verdict])
    runtime = AgentRuntime(make_config(model__context_limit=32000), model_client=client)
    state = runtime.create_or_load_session()
    holder['authority'], events = _prior_turn(runtime, state)
    _current_turn(runtime, state, 'Please say hello.')
    action = replace(_completed_action(), assistant_message='Hello!')
    result = runtime._evaluate_completion(state, original_request='Please say hello.',
                                          selected_action=action, tool_results=[])
    assert result['complete'] is True and len(client.requests) == 1
    restored = runtime.history.read_history(state.session_id)
    for event in events:
        assert next(e for e in restored if e.sequence == event.sequence) == event


def test_claim_about_prior_effect_fetches_exact_failure_before_verdict(make_config):
    holder = {}
    def request_source(payload):
        assert 'EXACT_OLD_FAILURE' not in payload['prompt']
        return json.dumps({'complete':False, 'reason':'Inspect the actual prior test result.',
            'remaining_work':[], 'evidence_requests':[{'source_kind':'foreground_history',
            'source_id':holder['source_id'], 'purpose':'Verify the claimed test success.',
            'literal_query':''}]})
    def reject_claim(payload):
        assert 'EXACT_OLD_FAILURE' in payload['prompt']
        assert '\\"passed\\":false' in payload['prompt']
        result = {'complete':False, 'reason':'The actual prior check failed.',
                  'remaining_work':['Repair and verify the failed check.']}
        if 'evidence_requests' in payload['json_schema']['properties']:
            result['evidence_requests'] = []
        return json.dumps(result)
    client = FakeModelClient([request_source, reject_claim])
    runtime = AgentRuntime(make_config(model__context_limit=32000), model_client=client)
    state = runtime.create_or_load_session()
    _, events = _prior_turn(runtime, state)
    holder['source_id'] = f'{state.session_id}:{events[0].sequence}-{events[-1].sequence}'
    _current_turn(runtime, state, 'Did the earlier check pass?')
    result = runtime._evaluate_completion(state, original_request='Did the earlier check pass?',
                                          selected_action=_completed_action(), tool_results=[])
    assert result['complete'] is False and len(client.requests) == 2
    assert any(e.event_type == 'completion_evidence_reexpanded'
               and e.payload['source_kind'] == 'foreground_history'
               for e in runtime.history.read_history(state.session_id))


def test_history_source_recovers_exact_bytes_and_rejects_tampered_or_cross_session_reference(make_config):
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    _, events = _prior_turn(runtime, state)
    source = runtime._completion_evidence_source_inventory(state, evidence_rows=[],
        historical_events=runtime.history.read_history(state.session_id),
        deferred_observations={e.sequence for e in events})
    source = next(s for s in source if s['source_kind'] == 'foreground_history')
    provider = ForegroundHistoryEvidenceProvider()
    expanded = provider.reexpand(config=runtime.config, state=state, source=source)
    assert json.loads(expanded['text']) == [asdict(e) for e in events]
    for patch in [{'session_id':'another_session'}, {'sha256':'wrong'},
                  {'event_sequences':[True]}, {'event_sequences':list(reversed(source['event_sequences']))},
                  {'event_sequences':source['event_sequences'][:-1]}, {'tool_names':['invented']}]:
        with pytest.raises(HistoryInvariantError):
            provider.reexpand(config=runtime.config, state=state, source={**source, **patch})


def test_worker_checker_without_fresh_snapshot_keeps_complete_historical_view(make_config):
    def verdict(payload):
        assert 'EXACT_OLD_FAILURE' in payload['prompt']
        assert 'EXACT_OLD_STATE' in payload['prompt']
        assert 'foreground_history_disclosure' not in payload['prompt']
        return json.dumps({'complete':False, 'reason':'The previous check failed.',
                          'remaining_work':['Verify the fix.']})
    client = FakeModelClient([verdict])
    runtime = AgentRuntime(make_config(model__context_limit=32000), model_client=client)
    state = runtime.create_or_load_session()
    _prior_turn(runtime, state)
    _current_turn(runtime, state, 'Complete the work.', snapshot=False)
    assert runtime._evaluate_completion(state, original_request='Complete the work.',
        selected_action=_completed_action(), tool_results=[])['complete'] is False


def test_custom_provider_without_history_support_falls_back_to_full_rows():
    rows = [{'sequence':7, 'event_type':'tool_result', 'payload':{'result':'exact'}}]
    assert AgentRuntime._foreground_completion_history_view(rows, {7}, []) is rows


def test_compact_catalog_retains_semantic_metadata_and_exact_backend_inventory():
    source = {'source_kind':'foreground_history', 'source_id':'session:7-8',
        'sha256':'EXACT_INTEGRITY_HASH', 'event_sequences':[7,8],
        'tool_names':['run_tests'], 'first_timestamp':'observed time',
        'custom_semantic_field':{'result':'meaningful'}}
    custom = {'source_kind':'custom_provider', 'source_id':'opaque', 'sha256':'SEMANTIC_METADATA'}
    catalog = AgentRuntime._foreground_evidence_catalog([source, custom])
    assert 'sha256' not in catalog[0] and 'event_sequences' not in catalog[0]
    assert catalog[0]['custom_semantic_field'] == source['custom_semantic_field']
    assert catalog[0]['tool_names'] == ['run_tests']
    assert source['sha256'] == 'EXACT_INTEGRITY_HASH' and source['event_sequences'] == [7,8]
    assert catalog[1] == custom


def test_user_role_cannot_be_deferred_by_internal_action_metadata(make_config):
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    event = runtime.history.record_event(state, 'message_added', {'message':{
        'role':'user', 'content':'Do not publish the report.', 'created_at':'now',
        'metadata':{'internal_action':True}}})
    assert is_foreground_observation_event(event) is False


def test_prior_verdict_reason_remains_exact_and_derived_summary_is_recoverable(make_config):
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    _, observations = _prior_turn(runtime, state)
    verdict = runtime.history.record_event(state, 'completion_evaluated', {
        'complete':False, 'reason':'The worker has not acknowledged the instruction.',
        'remaining_work':['Verify acknowledgement.'],
        'historical_evidence_projection':'EXACT_OLD_DERIVED_SUMMARY',
        'historical_source_event_references':[], 'evidence_source_references':[],
        'historical_projection_budget_report':{}})
    events = runtime.history.read_history(state.session_id)
    deferred = {e.sequence for e in observations}
    inventory = runtime._completion_evidence_source_inventory(state, evidence_rows=[],
        historical_events=events, deferred_observations=deferred)
    rows = runtime._foreground_completion_history_view(
        runtime._completion_historical_evidence_rows(events), deferred, inventory)
    row = next(r for r in rows if r.get('sequence') == verdict.sequence)
    assert row['payload']['reason'] == verdict.payload['reason']
    assert row['payload']['remaining_work'] == verdict.payload['remaining_work']
    assert row['payload']['complete'] is False
    assert 'EXACT_OLD_DERIVED_SUMMARY' not in json.dumps(rows)
    source = next(s for s in inventory if s['source_kind']=='runtime_telemetry')
    exact = RuntimeTelemetryEvidenceProvider().reexpand(config=runtime.config, state=state, source=source)
    assert json.loads(exact['text']) == [asdict(verdict)]


def test_foreground_bookkeeping_is_recoverable_without_repeating_hashes(make_config):
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    authority, observations = _prior_turn(runtime, state)
    accounting = [runtime.history.record_event(state, kind, {'kind':'completion_evaluation',
        'reason':'Exact response reserve accounting', **payload}) for kind, payload in (
        ('semantic_reduction_working_set_exceeded', {'hierarchical_depth':0,
            'input_tokens':9000, 'working_set_limit_tokens':8000}),
        ('model_output_budget_exhausted', {'completion_tokens':819, 'finish_reason':'length',
            'prompt_tokens':3000, 'reserved_response_tokens':819}),
        ('budget_repaired', {'requested_response_tokens':819, 'capped_response_tokens':1229}))]
    failure = runtime.history.record_event(state, 'error',
        {'operation':'external_effect', 'error':'The requested external effect failed.', 'error_type':'RuntimeError'})
    events = runtime.history.read_history(state.session_id)
    deferred = {e.sequence for e in observations}
    inventory = runtime._completion_evidence_source_inventory(state, evidence_rows=[],
        historical_events=events, deferred_observations=deferred)
    original = runtime._completion_historical_evidence_rows(events)
    rows = runtime._foreground_completion_history_view(original, deferred, inventory)
    assert any(row.get('hash') for row in original)
    assert all('hash' not in row for row in rows)
    assert sum(row.get('session_id') == state.session_id for row in rows) == 1
    assert any(row.get('payload', {}).get('message', {}).get('content') == authority for row in rows)
    assert any(row.get('sequence') == failure.sequence and row['payload'] == failure.payload for row in rows)
    assert 'Exact response reserve accounting' not in json.dumps(rows)
    source = next(s for s in inventory if s['source_kind']=='runtime_telemetry')
    expanded = RuntimeTelemetryEvidenceProvider().reexpand(config=runtime.config, state=state, source=source)
    assert json.loads(expanded['text']) == [asdict(e) for e in accounting]


def test_legacy_per_fragment_tokenization_is_exact_recoverable_telemetry(make_config):
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    events = []
    for n in range(120):
        events.append(runtime.history.record_event(state, 'model_tokenize_requested',
            {'text_hash': f'{n:064x}'}))
        events.append(runtime.history.record_event(state, 'token_estimate_used',
            {'text_hash': f'{n:064x}', 'tokens': n + 1, 'strategy': 'chars_per_token'}))
    rows = runtime._completion_historical_evidence_rows(events)
    assert len(rows) == 1 and rows[0]['evidence_view'] == 'runtime_telemetry_reference'
    inventory = runtime._completion_evidence_source_inventory(state, evidence_rows=[], historical_events=events)
    source = next(s for s in inventory if s['source_kind'] == 'runtime_telemetry')
    exact = RuntimeTelemetryEvidenceProvider().reexpand(config=runtime.config, state=state, source=source)
    assert json.loads(exact['text']) == [asdict(e) for e in events]


@pytest.mark.parametrize('turn_first', [False, True])
def test_conversation_view_retains_wording_and_unknown_metadata_once_with_exact_detail_recovery(make_config, turn_first):
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    text = '  Keep this private.\nDo not publish.  '
    def turn():
        runtime.history.record_event(state, 'turn_started', {'turn_index':1, 'user_text':text})
    if turn_first:turn()
    runtime._record_message(state, Message(role='user', content=text, created_at='original-time',
        metadata={'custom_constraint':'PRESERVE_THIS'}))
    if not turn_first:turn()
    events = runtime.history.read_history(state.session_id)
    details = {e.sequence for e in events if e.event_type in ('message_added','turn_started')}
    inventory = runtime._completion_evidence_source_inventory(state, evidence_rows=[],
        historical_events=events, disclosed_conversation=details)
    rows = runtime._foreground_completion_history_view(
        runtime._completion_historical_evidence_rows(events), set(), inventory)
    messages = [row for row in rows if row.get('role') == 'user']
    assert len(messages) == 1 and messages[0]['content'] == text
    assert messages[0]['metadata']['custom_constraint'] == 'PRESERVE_THIS'
    assert not any(row.get('event_type') == 'turn_started' for row in rows)
    source = next(s for s in inventory if s['source_kind'] == 'foreground_history')
    recovered = ForegroundHistoryEvidenceProvider().reexpand(config=runtime.config, state=state, source=source)
    assert json.loads(recovered['text']) == [asdict(e) for e in events if e.sequence in details]
