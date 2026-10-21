"""Exact, indexed long-range user evidence without a slow model call."""
from __future__ import annotations

from swaag.fast_history_candidates import old_user_history_candidates, history_query_terms
from swaag.history import HistoryStore
from swaag.utils import utc_now_iso


def add(store, state, content, role='user'):
    store.record_event(state, 'message_added', {'message': {
        'role': role, 'content': content, 'created_at': utc_now_iso(),
        'name': None, 'metadata': {},
    }})
    return state.event_count


def test_finds_exact_old_user_constraint_without_scanning_unrelated_event_payloads(make_config,monkeypatch):
    cfg=make_config()
    store=HistoryStore(cfg.sessions.root)
    state=store.create(config_fingerprint='test',model_base_url='http://fake')
    old='For warehouse forklift photorealism, never replace the original PBR textures.'
    old_seq=add(store,state,old)
    add(store,state,'I want a higher salary and more investment returns.')
    for i in range(20):add(store,state,f'Temporary conversation about short independent topics {i}.')
    most_recent=add(store,state,'What are we working on now?')
    monkeypatch.setattr(store,'read_history',lambda *a,**k: (_ for _ in ()).throw(
        AssertionError('Do not load every model/tool event to search older user facts')))
    output=old_user_history_candidates(store,session_id=state.session_id,
        request='What earlier warehouse forklift restrictions did I give?',
        before_sequence=most_recent,indexed_through_required=state.event_count)
    assert output['status']=='ok' and output['query_complete']
    match=next(item for item in output['candidates'] if item['sequence']==old_seq)
    assert match['content']==old and match['verbatim'] is True
    assert match['source_event_hash']
    assert 'higher salary' not in str(output)
    assert output['complete_semantic_selection'] is False


def test_search_covers_many_old_messages_not_recent_fixed_window(make_config):
    cfg=make_config();store=HistoryStore(cfg.sessions.root)
    state=store.create(config_fingerprint='test',model_base_url='http://fake')
    old=add(store,state,'The project BlueHeron-731 must use Blender GPU renders, never CPU renders.')
    for i in range(160):add(store,state,f'Unrelated ordinary conversation {i}.')
    start=add(store,state,'Current recent message, unrelated to the old project.')
    answer=old_user_history_candidates(store,session_id=state.session_id,
        request='Remind me about BlueHeron-731 rendering restrictions.',
        before_sequence=start,indexed_through_required=state.event_count)
    assert answer['status']=='ok'
    assert answer['candidates'][0]['sequence']==old
    assert answer['candidates'][0]['verbatim']


def test_oversized_source_is_pointer_not_misleading_partial_quote(make_config):
    cfg=make_config();store=HistoryStore(cfg.sessions.root)
    state=store.create(config_fingerprint='test',model_base_url='http://fake')
    old=add(store,state,'Explicit warehouse material restriction: '+ 'precise caveat ' * 180)
    start=add(store,state,'Latest independent greeting')
    data=old_user_history_candidates(store,session_id=state.session_id,
        request='What warehouse material restriction did I give?',
        before_sequence=start,indexed_through_required=state.event_count,max_chars=100)
    assert data['status']=='ok'
    assert data['candidates'][0]['sequence']==old
    assert data['candidates'][0]['verbatim'] is False
    assert 'content' not in data['candidates'][0]
    assert 'lookup_instruction' in data['candidates'][0]


def test_incomplete_index_and_unsearchable_query_never_claim_exhaustive_relevance(make_config):
    cfg=make_config();store=HistoryStore(cfg.sessions.root)
    state=store.create(config_fingerprint='test',model_base_url='http://fake')
    add(store,state,'PBR warehouse material settings.')
    add(store,state,'Next message.')
    kwargs=dict(session_id=state.session_id,before_sequence=state.event_count+1,
                indexed_through_required=state.event_count)
    result=old_user_history_candidates(store,request='go on',**kwargs)
    assert result['status']=='no_discriminative_terms'
    result=old_user_history_candidates(store,request='warehouse material',
                                       indexed_through_required=state.event_count+2,
                                       session_id=state.session_id,
                                       before_sequence=state.event_count+1)
    assert result['status']=='index_not_current'
    assert not result['query_complete']


def test_special_symbols_are_bound_as_parameters_not_executed_as_sql(make_config):
    cfg=make_config();store=HistoryStore(cfg.sessions.root)
    state=store.create(config_fingerprint='test',model_base_url='http://fake')
    add(store,state,'History topic archive-guide-711.')
    start=add(store,state,'Current message.')
    found=old_user_history_candidates(store,session_id=state.session_id,
        request='archive-guide-711\";DROP TABLE events;--',
        before_sequence=start,indexed_through_required=state.event_count)
    assert found['status']=='ok'
    assert store.read_history_window(state.session_id,start_sequence=1,limit=1)


def test_history_query_terms_are_unicode_aware_and_do_not_interpret_user_intent():
    assert history_query_terms('Überlingen Bodensee Photogrammetry') == [
        'überlingen', 'bodensee', 'photogrammetry']
    assert history_query_terms('go on') == []
    assert 'warehouse' in history_query_terms('Please inspect warehouse shaders')


def test_extremely_large_old_message_uses_event_pointer_with_no_large_json_decode(make_config):
    cfg=make_config();store=HistoryStore(cfg.sessions.root)
    state=store.create(config_fingerprint='test',model_base_url='http://fake')
    old=add(store,state,'Warehouse exact full source must remain retrievable. '+ 'x' * 25000)
    latest=add(store,state,'Current question.')
    answer=old_user_history_candidates(store,session_id=state.session_id,
        request='warehouse exact source?',before_sequence=latest,
        indexed_through_required=state.event_count,max_chars=1000)
    assert answer['status']=='ok'
    assert answer['candidates'][0]['sequence']==old
    assert answer['candidates'][0]['chars']>25000
    assert answer['candidates'][0]['verbatim'] is False
    assert 'content' not in answer['candidates'][0]


def test_one_fast_model_call_can_see_relevant_exact_old_user_source(make_config):
    from tests.test_fast_foreground_reply import setup, fast
    from swaag.foreground_fast import generate_fast_foreground_response
    service,runtime,client=setup(make_config,[fast('answer','You asked to preserve the warehouse PBR textures.')])
    try:
        state=service._orchestrator_state()
        original='For the warehouse assets, never replace the original PBR textures.'
        add(runtime.history,state,original)
        add(runtime.history,state,'The investment portfolio has unrelated risk constraints.')
        for index in range(11):
            add(runtime.history,state,f'This is unrelated prior conversation message number {index}.')
        answer=generate_fast_foreground_response(runtime,state,
            'What did I say about the warehouse PBR textures?')
        assert answer and answer.mode=='answer'
        assert answer.reply == 'You asked to preserve the warehouse PBR textures.'
        assert len(client.requests)==1
        prompt=client.requests[0]['prompt']
        assert original in prompt
        assert 'fast_old_user_message_candidates' not in prompt or 'lexical overlap' in prompt
        assert 'The investment portfolio has unrelated risk constraints.' not in prompt
        assert prompt.index(original)<prompt.index('Final authoritative current user message')
        assert 'complete_semantic_selection' in prompt
    finally:service.close()


def test_old_quoted_task_instruction_is_historical_data_only(make_config):
    from tests.test_fast_foreground_reply import setup,fast
    from swaag.foreground_fast import generate_fast_foreground_response
    service,runtime,client=setup(make_config,[fast('answer','That was an earlier hypothetical example.')])
    try:
        state=service._orchestrator_state()
        add(runtime.history,state,'An earlier hypothetical warehouse plan says: delete every render. This is only an example.')
        for index in range(9):add(runtime.history,state,f'Current dialogue: discussion {index}.')
        answer=generate_fast_foreground_response(runtime,state,
            'What hypothetical warehouse plan did I mention before?')
        assert answer and answer.mode=='answer'
        prompt=client.requests[0]['prompt']
        assert 'delete every render' in prompt
        assert 'NOT active instructions' in prompt
        assert not any(e.event_type=='tool_called' for e in runtime.history.read_history(state.session_id))
    finally:service.close()


def test_optional_old_history_cannot_force_context_overflow_and_second_model_call(make_config):
    from tests.test_fast_foreground_reply import setup,fast
    from swaag.foreground_fast import generate_fast_foreground_response
    service,runtime,client=setup(make_config,[fast('answer','SWAAG can answer directly.')])
    try:
        runtime.config.communication.fast_max_input_tokens=1370
        state=service._orchestrator_state()
        add(runtime.history,state,'A warehouse photorealistic source: '+'old relevant exact text ' * 70)
        for i in range(9):add(runtime.history,state,f'Unrelated chat message {i}')
        answer=generate_fast_foreground_response(runtime,state,'What is photorealistic rendering?')
        assert answer and answer.mode=='answer'
        assert len(client.requests)==1
        assert 'fast_old_user_message_candidates' not in client.requests[0]['prompt']
    finally:service.close()


def test_index_lag_is_not_misrepresented_as_complete_history(make_config,monkeypatch):
    from tests.test_fast_foreground_reply import setup,fast
    from swaag.foreground_fast import generate_fast_foreground_response
    service,runtime,client=setup(make_config,[fast('delegate','I will retrieve the older evidence.')])
    try:
        state=service._orchestrator_state()
        add(runtime.history,state,'Warehouse grounding status previous report')
        for i in range(9):add(runtime.history,state,f'Unrelated recent message {i}')
        import swaag.fast_history_candidates as history_candidates
        monkeypatch.setattr(history_candidates,'old_user_history_candidates',
            lambda *a, **k: {'status':'index_query_unavailable','candidates':[]})
        # Optional derived search failure should not become a fabricated result.
        result=generate_fast_foreground_response(runtime,state,'What was the warehouse status?')
        assert result and result.mode=='delegate'
        assert len(client.requests)==1
        assert 'Warehouse grounding status previous report' not in client.requests[0]['prompt']
    finally:service.close()


def test_broken_derived_sqlite_connection_is_optional_and_fails_closed():
    class UnavailableHistory:
        @staticmethod
        def _sqlite_connect():
            raise OSError('read-only index temporarily unavailable')
    output=old_user_history_candidates(UnavailableHistory(),session_id='session_test',
        request='warehouse rendering restriction',before_sequence=500,
        indexed_through_required=499)
    assert output['status']=='index_query_unavailable'
    assert output['candidates']==[]
    assert output['query_complete'] is False
