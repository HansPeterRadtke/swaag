"""Pure independent receipt falsification checks; ALL fixtures are synthetic.

The actual nine real Jetson source tests were separately executed and
source/disk verified using original SHA-pinned private receipts. These tests
do not run or reproduce any model, voice, SMS, browser, phone, user or tool
operation. Passing here alone is never evidence of the real implementation.
"""
from __future__ import annotations
import importlib.util,hashlib,json,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))
SPEC=importlib.util.spec_from_file_location('independent_w70_nine_receipts',ROOT/'scripts/demos/verify_isolated_voice_persisted_epoch_async_evidence.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def fake_case(tmp_path,name='async-ws-cancel-oneuse',status='no_stale_draft_after_cancel_or_new_manual',rows=None,family='corrected',openings=1):
    if rows is None:rows=[]
    root=tmp_path/name;root.mkdir();(root/'conversations').mkdir()
    src=M.SOURCE_W69 if family=='baseline' else M.PATCHED_SERVER_SHA256
    d={'status':status,'source_sha256':src,'real_provider_model_tools_calls':0,
       'real_model_ASR_TTS_agent_tool_calls':0,'actual_Jetson_production_service_changed':False,
       'input_recording_SHA256':M.VOICE_PCM_SHA,'held_authoritative_ASR_requests':1,
       'saved_history_statuses_after_new_manual':rows,'actual_private_child_pid':999999,
       'elapsed_seconds':1.10}
    if 'eventloop' in name:
        d.update({'max_event_loop_tick_gap_seconds':.718 if family=='baseline' else .016,
         'private_durable_JSON_write_count':1,'new_manual_intent_completion_seconds':.7,
         'no_network_or_provider_calls':True})
    if 'legacy' in name:d['legacy_epochless_derived_fixture_is_NOT_original_source_bytes']=True
    result=root/'result.json';result.write_text(json.dumps(d))
    (root/'conversations'/'test.json').write_text(json.dumps({'dialogue_history':[
        {'status':state,'submitted_text':'','assistant':'','assistant_audio_path':''} for state in rows]}))
    if 'eventloop' not in name:
        opens=[{'event':'server_start'}]+[{'event':'session_open'}]*openings
        (root/'voice-private.jsonl').write_text('\n'.join(json.dumps(x) for x in opens)+'\n')
        config={'server':{'host':'127.0.0.1','auth_token_file':str(root/'private-only-token')},
            'interaction':{'require_verified_speaker_for_auto_send':True},
            'llm':{'url':'http://127.0.0.1:9/run'},
            'asr':{'service_url':'http://127.0.0.1:9/asr'},
            'tts':{'service_url':'http://127.0.0.1:9/tts'}}
        (root/'private-config.json').write_text(json.dumps(config))
    return tmp_path,name,(M.sha(result),status,rows,family)


def test_nine_immutable_true_source_receipt_hashes_are_declared():
    assert len(M.CASES)==9
    assert all(len(v[0])==64 for v in M.CASES.values())
    assert M.CASES['old-w69-restart-new-manual-never-before-tested'][2]==['superseded','draft']
    assert M.CASES['corrected-restart-new-manual-fresh-test'][2]==['superseded','superseded']
    assert M.CASES['corrected-legacy-epochless-manual-restart-oneuse'][2]==['superseded','superseded']
    assert M.EXPECTED_SESSION_OPENS['async-ws-new_manual_disconnect_before_asr-oneuse']==3
    assert M.MANIFEST_SHA=='f6372eb49372fde646dbadbc51bdcede078be98008af14d6489542980fbc33bc'


def test_synthetic_private_case_accepted_but_not_real_process_proof(tmp_path):
    r,name,desc=fake_case(tmp_path)
    j=M.check_case(r,name,desc)
    assert j['recorded_durable_statuses']==[] and j['real_private_source_orchestrator_effects']==0


def test_synthetic_stalled_eventloop_vs_recovered_is_measured_as_separate_kinds(tmp_path):
    r,name,desc=fake_case(tmp_path,'w69-known-eventloop-stall-once',
        'observed_prior_voice_event_loop_0_7sec_storage_stall',['superseded','superseded'],'baseline')
    assert M.check_case(r,name,desc)['event_loop_max_gap_seconds']==.718
    r,name,desc=fake_case(tmp_path,'w70-fixed-eventloop-responsive-once',
        'fixed_atomic_voice_intent_durable_write_no_event_loop_stall',['superseded','superseded'],'corrected')
    assert M.check_case(r,name,desc)['event_loop_max_gap_seconds']==.016


def test_nonresponsive_new_eventloop_fails_even_if_receipt_hash_valid(tmp_path):
    r,name,desc=fake_case(tmp_path,'w70-fixed-eventloop-responsive-once',
        'fixed_atomic_voice_intent_durable_write_no_event_loop_stall',['superseded','superseded'],'corrected')
    p=r/name/'result.json';x=json.loads(p.read_text());x['max_event_loop_tick_gap_seconds']=.72;p.write_text(json.dumps(x))
    desc=(M.sha(p),desc[1],desc[2],desc[3])
    with pytest.raises(ValueError,match='responsiveness not demonstrated'):
        M.check_case(r,name,desc)


def test_fake_success_receipt_append_tamper_denied(tmp_path):
    r,name,desc=fake_case(tmp_path)
    p=r/name/'result.json';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='Immutable one-use receipt modified'):
        M.check_case(r,name,desc)


def test_hidden_submitted_turn_in_history_denied(tmp_path):
    r,name,desc=fake_case(tmp_path,'async-ws-idle_reconnect_pending_asr-oneuse',
        'idle_pre_reconnect_ASR_held_for_manual_review',['draft'],'corrected',2)
    p=r/name/'conversations/test.json';j=json.loads(p.read_text());j['dialogue_history'][0]['submitted_text']='a forged action';p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match='reached real agent'):
        M.check_case(r,name,desc)


def test_remote_model_provider_url_denied(tmp_path):
    r,name,desc=fake_case(tmp_path)
    p=r/name/'private-config.json';d=json.loads(p.read_text());d['llm']['url']='https://offsite.example/model';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='not localhost/manual-review only'):
        M.check_case(r,name,desc)


def test_speaker_guard_disabled_denied(tmp_path):
    r,name,desc=fake_case(tmp_path)
    p=r/name/'private-config.json';d=json.loads(p.read_text());d['interaction']['require_verified_speaker_for_auto_send']=False;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='not localhost/manual-review only'):
        M.check_case(r,name,desc)


def test_wrong_client_replacement_count_denied(tmp_path):
    r,name,desc=fake_case(tmp_path,'async-ws-new_manual_disconnect_before_asr-oneuse',
        'no_stale_draft_after_cancel_or_new_manual',[],'corrected',2)
    with pytest.raises(ValueError,match='client replacement count mismatch'):
        M.check_case(r,name,desc)


def test_emitted_tool_action_logged_and_rejected(tmp_path):
    r,name,desc=fake_case(tmp_path)
    p=r/name/'voice-private.jsonl';p.write_text(p.read_text()+json.dumps({'event':'turn_submitted'})+'\n')
    with pytest.raises(ValueError,match='Unexpected user or tool action'):
        M.check_case(r,name,desc)


def test_unpinned_actual_source_bytes_are_denied(tmp_path,monkeypatch):
    baseline=tmp_path/'original';baseline.mkdir();candidate=tmp_path/'candidate';candidate.mkdir()
    module={n:(n+' safe fake module').encode() for n in M.BASE}
    for n,v in module.items():
        (baseline/n).write_bytes(v)
        (candidate/n).write_bytes(v+(b'one new private source' if n=='cloudflare_ws_voice_server.py' else b''))
    monkeypatch.setattr(M,'BASE',{n:M.sha(baseline/n) for n in module})
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256',M.sha(candidate/'cloudflare_ws_voice_server.py'))
    monkeypatch.setattr(M,'patch',lambda b:b+b'one new private source')
    manifest=candidate/'source-manifest.json';manifest.write_text(json.dumps({'new_server_sha256':M.PATCHED_SERVER_SHA256,'only_changed_module':'cloudflare_ws_voice_server.py'}))
    monkeypatch.setattr(M,'MANIFEST_SHA',M.sha(manifest))
    hist=tmp_path/'original-history.json';hist.write_text(json.dumps({'dialogue_history':[
        {'status':'superseded','asr_intent_epoch':1,'submitted_text':'','assistant':''},
        {'status':'draft','asr_intent_epoch':2,'submitted_text':'','assistant':''}]}))
    monkeypatch.setattr(M,'ORIGINAL_HISTORY_SHA',M.sha(hist))
    assert M.verify_sources(baseline,candidate,hist)['unchanged_six_modules']
    (candidate/'voice_ws_core.py').write_bytes(b'changed after authentication')
    with pytest.raises(ValueError,match='not exact deterministic single-module'):
        M.verify_sources(baseline,candidate,hist)


def test_private_result_oneuse_output_never_overwrites_existing(tmp_path,monkeypatch):
    fake={'status':'independently_verified_real_persisted_epoch_recovery_and_heartbeat_fairness'}
    monkeypatch.setattr(M,'verify',lambda *a:fake)
    out=tmp_path/'existing-verdict.json';out.write_text('immutable')
    args=['--evidence',str(tmp_path),'--baseline',str(tmp_path),
          '--candidate',str(tmp_path),'--original-history',str(out),'--output',str(out)]
    with pytest.raises(FileExistsError,match='Unique private verifier'):
        M.main(args)
    assert out.read_text()=='immutable'
