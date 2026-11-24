"""Source-independent negative tests against synthetic, never-real receipts.

This is NOT proof that the Jetson experiment happened. It tests whether the
independent checker rejects modified persisted histories, source bytes,
invalid speaker authority, fake model calls and fabricated success. The
actual separate Jetson interpreter/process evidence is independently saved.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))
SPEC=importlib.util.spec_from_file_location('independent_atomic_disk_and_WS_verifier',ROOT/'scripts/demos/verify_isolated_voice_atomic_storage_and_ws_results.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def fake_scenario(tmp_path,monkeypatch):
    root=tmp_path/'private-fixture-synthetic-only';root.mkdir()
    before=tmp_path/'before-original-source';before.mkdir()
    after=tmp_path/'after-patched-source';after.mkdir()
    raw={k:('FAKE_TEST_FILE_'+k+'\n').encode() for k in M.BASE}
    for key,val in raw.items():
        (before/key).write_bytes(val)
        (after/key).write_bytes(val+(b'W69-TEST-PATCH' if key=='cloudflare_ws_voice_server.py' else b''))
    monkeypatch.setattr(M,'BASE',{k:M.sha(before/k) for k in raw})
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256',M.sha(after/'cloudflare_ws_voice_server.py'))
    monkeypatch.setattr(M,'modify',lambda payload:payload+b'W69-TEST-PATCH')
    (after/'source-manifest.json').write_text(json.dumps({'new_atomic_server_sha256':M.PATCHED_SERVER_SHA256,'changed_only':['cloudflare_ws_voice_server.py']}))
    prior=tmp_path/'prior-failure.json';prior.write_text(json.dumps({'status':'genuine_private_server_stale_ASR_regression_fixed_in_unreleased_candidate','real_process_trials':6,'old_result_stale_draft_count':1,'corrected_real_speaker_guard_tests_passed':4}))
    monkeypatch.setattr(M,'PREVIOUS_VERDICT_SHA',M.sha(prior))
    case_expect={}
    for name,(sha,turns,status,replaced) in M.CASES.items():
        p=root/('atomic-ws-'+name+'-oneuse');p.mkdir();(p/'conversations').mkdir()
        d={'status':status,'source_sha256':M.PATCHED_SERVER_SHA256,
            'input_recording_SHA256':M.VOICE_SHA,
            'held_authoritative_ASR_requests':1,'last_history_turns':turns,
            'real_model_ASR_TTS_agent_tool_calls':0,
            'actual_owner_or_Android_phone_used':False,
            'actual_Jetson_production_service_changed':False,
            'elapsed_seconds':1.1}
        (p/'result.json').write_text(json.dumps(d))
        conf={'interaction':{'auto_send_default':True,'require_verified_speaker_for_auto_send':True},
            'server':{'host':'127.0.0.1','auth_token_file':str(p/'fake-only-bearer')},
            'asr':{'service_url':'http://127.0.0.1:9/disabled','final_quality_url':'http://127.0.0.1:4567/fake'},
            'llm':{'url':'http://127.0.0.1:9/run'},
            'tts':{'service_url':'http://127.0.0.1:9/disabled'}}
        (p/'private-config.json').write_text(json.dumps(conf))
        logs=[{'event':'server_start'}]+[{'event':'session_replaced'}]*replaced+[{'event':'asr_final'}]*turns
        (p/'voice-private.jsonl').write_text('\n'.join(map(json.dumps,logs))+'\n')
        rows=[{'recognized_text':M.FAKE_TEXT,'status':'draft','requires_review':True,
               'submitted_text':'','assistant':'','assistant_audio_path':''} for _ in range(turns)]
        (p/'conversations'/(name+'.json')).write_text(json.dumps({'dialogue_history':rows}))
        case_expect[name]=(M.sha(p/'result.json'),turns,status,replaced)
    monkeypatch.setattr(M,'CASES',case_expect)
    private=root/'atomic-storage-threaded-oneuse';private.mkdir();(private/'conversations').mkdir()
    result={'status':'genuine_atomic_disk_commit_vs_new_manual_intent_race_passed',
        'source_sha256':M.PATCHED_SERVER_SHA256,
        'old_source_result_preserved_but_superseded':True,
        'old_stale_later_draft_and_update_denied':True,
        'latest_new_audio_draft_recorded_exactly_once':True,
        'other_conversation_progressed_during_held_write':True,
        'real_atomic_write_json_tempfile_fsync_rename':True,
        'live_services_changed':False,'other_conversation_elapsed_seconds':0.01}
    (private/'result.json').write_text(json.dumps(result))
    monkeypatch.setattr(M,'STORAGE_SHA',M.sha(private/'result.json'))
    a=[{'status':'superseded','asr_intent_epoch':1,'superseded_by_intent_epoch':2,
        'turn_id':'turn-old','user_audio_path':'/preserved/test-only.wav','submitted_text':'','assistant':''},
       {'status':'draft','asr_intent_epoch':2,'turn_id':'turn-current','submitted_text':'','assistant':''}]
    b=[{'status':'draft','turn_id':'turn-independent','submitted_text':'','assistant':''}]
    (private/'conversations/private-w69-storage-race-A.json').write_text(json.dumps({'dialogue_history':a}))
    (private/'conversations/private-w69-storage-race-B.json').write_text(json.dumps({'dialogue_history':b}))
    return root,prior,before,after


def test_actual_original_pin_contracts_and_nondeployment_status_are_predeclared():
    assert M.PATCHED_SERVER_SHA256=='3d4e6e029aa4a467684c618dc31cfd60d53d77939843f547d394b04a979bad2f'
    assert M.BASE['cloudflare_ws_voice_server.py']=='e0939fd23aefd11f6d3d8bfd0158508b7b2f0a4f75fb70467e67bced5d06f030'
    assert M.VOICE_SHA=='56e81fe7050641490a634e064705673a1f6c61ee2cf46d6b09f33d8270e7fae7'
    assert len(M.CASES)==4
    assert M.CASES['idle_reconnect_pending_asr'][1]==1
    assert M.CASES['new_manual_on_reconnect'][1]==0


def test_synthetic_fixture_valid_source_and_correct_original_storage_shapes(tmp_path,monkeypatch):
    args=fake_scenario(tmp_path,monkeypatch)
    data=M.verify(*args)
    assert data['status']=='independently_source_and_disk_verified_atomic_voice_storage_and_websocket_races'
    assert len(data['actual_four_distinct_private_WS_processes'])==4
    assert data['actual_forced_atomic_write']['old_speech_status_after_new_manual']=='superseded'
    assert data['actual_forced_atomic_write']['new_speech_epoch']==2
    assert data['actual_forced_atomic_write']['independent_other_conversation_draft_count']==1
    assert data['actual_new_local_Whisper_SWAAG_TTS_or_tool_calls']==0
    assert data['production_Jetson_voice_service_changed'] is False


def test_changed_original_source_byte_denied(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    (original/'voice_ws_core.py').write_bytes(b'changed')
    with pytest.raises(ValueError,match='baseline unexpectedly changed'):
        M.verify(root,prior,original,patched)


def test_wrong_candidate_server_file_denied(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    (patched/'cloudflare_ws_voice_server.py').write_bytes(b'changed')
    with pytest.raises(ValueError,match='not exactly the tested'):
        M.verify(root,prior,original,patched)


def test_wrong_candidate_manifest_denied(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    (patched/'source-manifest.json').write_text(json.dumps({'new_atomic_server_sha256':'bad','changed_only':['all']}))
    with pytest.raises(ValueError,match='manifest inconsistent'):
        M.verify(root,prior,original,patched)


def test_original_prior_failed_process_proof_tampered_is_rejected(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    prior.write_text(prior.read_text()+'\n')
    with pytest.raises(ValueError,match='Prior W68 failure'):
        M.verify(root,prior,original,patched)


def test_recorded_private_WS_result_status_cannot_be_forged(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    p=root/'atomic-ws-cancel-oneuse/result.json';d=json.loads(p.read_text());d['status']='pretend';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Real WS receipt changed'):
        M.verify(root,prior,original,patched)


def test_changed_voice_conversation_status_cannot_be_accepted(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    p=next((root/'atomic-ws-idle_reconnect_pending_asr-oneuse/conversations').glob('*.json'))
    d=json.loads(p.read_text());d['dialogue_history'][0]['status']='received';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Wrong original WS unsubmitted'):
        M.verify(root,prior,original,patched)


def test_late_stale_speech_into_new_user_history_is_rejected(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    p=next((root/'atomic-ws-new_manual_on_reconnect-oneuse/conversations').glob('*.json'))
    p.write_text(json.dumps({'dialogue_history':[{'status':'draft','requires_review':True}]}))
    with pytest.raises(ValueError,match='durable draft count'):
        M.verify(root,prior,original,patched)


def test_tool_effect_even_if_receipt_passes_rejected(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    p=root/'atomic-ws-cancel-oneuse/voice-private.jsonl'
    p.write_text(p.read_text()+json.dumps({'event':'turn_submitted'})+'\n')
    with pytest.raises(ValueError,match='real external tool'):
        M.verify(root,prior,original,patched)


def test_public_remote_model_endpoint_hidden_in_config_detected(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    p=root/'atomic-ws-cancel-oneuse/private-config.json';v=json.loads(p.read_text())
    v['llm']['url']='https://offsite-provider.test/run';p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='remote provider'):
        M.verify(root,prior,original,patched)


def test_unverified_owner_speaker_can_not_disable_manual_review(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    p=root/'atomic-ws-cancel-oneuse/private-config.json';d=json.loads(p.read_text())
    d['interaction']['require_verified_speaker_for_auto_send']=False;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='disabled the active speaker review'):
        M.verify(root,prior,original,patched)


def test_atomic_old_draft_did_not_get_safely_superseded_detected(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    p=root/'atomic-storage-threaded-oneuse/conversations/private-w69-storage-race-A.json';d=json.loads(p.read_text())
    d['dialogue_history'][0]['status']='draft';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Original atomic storage states'):
        M.verify(root,prior,original,patched)


def test_atomic_current_epoch_not_accepted_or_different_user_blocked(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    p=root/'atomic-storage-threaded-oneuse/conversations/private-w69-storage-race-B.json';d=json.loads(p.read_text())
    d['dialogue_history'].clear();p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Original atomic storage states'):
        M.verify(root,prior,original,patched)


def test_existing_private_result_is_never_overwritten(tmp_path,monkeypatch):
    root,prior,original,patched=fake_scenario(tmp_path,monkeypatch)
    out=tmp_path/'first_private_result.json'
    command=['--evidence',str(root),'--previous-verdict',str(prior),'--baseline',str(original),'--candidate',str(patched),'--output',str(out)]
    # The actual generated result location is a synthetic private pytest
    # basetemp under the permitted benchmarks root.
    assert M.main(command)==0
    old=out.read_bytes()
    with pytest.raises(FileExistsError,match='Unique private verdict'):
        M.main(command)
    assert out.read_bytes()==old
