"""No-network falsification tests for the source-backed real WS race oracle.

These fixture receipts are synthetic TEST DATA: they are never passed off as
actual Jetson results. The real six-process adjudication has a distinct
SHA-verified original result retained in Jetson private storage.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))
SPEC=importlib.util.spec_from_file_location('independent_swaag_epoch_evidence',ROOT/'scripts/demos/verify_isolated_voice_epoch_real_results.py')
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fake_complete_evidence(tmp_path,monkeypatch):
    root=tmp_path/'private-six-case-evidence';root.mkdir()
    baseline=tmp_path/'private-source-baseline';baseline.mkdir()
    candidate=tmp_path/'private-source-candidate';candidate.mkdir()
    raw={k:('MOCK_ORIGINAL_'+k).encode() for k in M.BASELINE}
    edited={k:(v+b'EXACT_TEST_PATCH' if k=='cloudflare_ws_voice_server.py' else v) for k,v in raw.items()}
    for name in raw:
        (baseline/name).write_bytes(raw[name]);(candidate/name).write_bytes(edited[name])
    monkeypatch.setattr(M,'BASELINE',{k:M.sha(v) for k,v in raw.items()})
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256',M.sha(edited['cloudflare_ws_voice_server.py']))
    monkeypatch.setattr(M,'rewrite',lambda original:original+b'EXACT_TEST_PATCH')
    fakecases={}
    for name,info in M.SCENARIOS.items():
        case=root/name;case.mkdir();(case/'conversations').mkdir()
        server_sha=(M.BASELINE['cloudflare_ws_voice_server.py'] if info['revision']=='baseline'
                    else M.PATCHED_SERVER_SHA256)
        result={'status':info['status'],'source_sha256':server_sha,
            'input_recording_SHA256':M.VOICE_AUDIO_SHA,
            'real_model_ASR_TTS_agent_tool_calls':0,
            'actual_Jetson_production_service_changed':False,
            'elapsed_seconds':1.32}
        if info['status']=='failed_or_unknown':
            result['failure_type']='AssertionError'
            result['failure_detail_private']='Superseded late ASR inserted a stale draft during newer manual recording'
        else:result['held_authoritative_ASR_requests']=1
        receipt=case/'result.json';receipt.write_text(json.dumps(result))
        config={'server':{'host':'127.0.0.1','auth_token_file':str(case/'fake-only-bearer')},
            'interaction':{'require_verified_speaker_for_auto_send':True,'auto_send_default':True},
            'asr':{'service_url':'http://127.0.0.1:9/unavailable',
                'final_quality_url':'http://127.0.0.1:45678/v1/voice/transcribe-verify'},
            'llm':{'url':'http://127.0.0.1:9/run'},
            'tts':{'service_url':'http://127.0.0.1:9/synthesize'}}
        (case/'private-config.json').write_text(json.dumps(config))
        logged=[{'event':'server_start'}]
        logged += [{'event':'session_replaced'}]*info['replaced']
        logged += [{'event':'asr_final'}]*info['turns']
        (case/'voice-private.jsonl').write_text('\n'.join(json.dumps(x) for x in logged)+'\n')
        turns=[{'status':'draft','requires_review':True,'recognized_text':M.FAKE_TEXT,
            'submitted_text':'','assistant':'','assistant_audio_path':''} for _ in range(info['turns'])]
        (case/'conversations'/(name+'.json')).write_text(json.dumps({'dialogue_history':turns}))
        fakecases[name]={**info,'sha':M.sha(receipt.read_bytes())}
    monkeypatch.setattr(M,'SCENARIOS',fakecases)
    return root,baseline,candidate,fakecases


def test_original_private_real_case_receipts_are_SHA_pinned_in_source():
    assert len(M.SCENARIOS)==6
    assert len([x for x in M.SCENARIOS.values() if x['revision']=='patched'])==4
    assert M.SCENARIOS['new-manual-replacement-held-asr-once']['status']=='failed_or_unknown'
    assert M.SCENARIOS['new-manual-replacement-held-asr-once']['turns']==1
    assert M.SCENARIOS['patched-new_manual_on_reconnect-oneuse']['turns']==0
    assert M.SCENARIOS['patched-idle_reconnect_pending_asr-oneuse']['turns']==1
    assert M.VOICE_AUDIO_SHA=='56e81fe7050641490a634e064705673a1f6c61ee2cf46d6b09f33d8270e7fae7'


def test_synthetic_fixtures_verify_one_real_failure_four_fixed_success_shapes(tmp_path,monkeypatch):
    root,baseline,candidate,_=fake_complete_evidence(tmp_path,monkeypatch)
    original=M.inspect(root,baseline,candidate)
    assert original['real_process_trials']==6
    assert original['baseline_real_failure_reproduced']
    assert original['old_result_stale_draft_count']==1
    assert original['patched_stale_draft_count']==0
    assert original['patched_idle_reconnect_good_review_draft_count']==1
    assert original['corrected_real_speaker_guard_tests_passed']==4
    assert original['no_authorized_turn_or_tool_action_in_any_trial']
    assert original['real_resident_Whisper_model_used'] is False
    assert original['live_Jetson_server_modified_or_restarted'] is False


def test_original_result_tampering_denied_even_if_status_still_looks_good(tmp_path,monkeypatch):
    root,baseline,candidate,_=fake_complete_evidence(tmp_path,monkeypatch)
    receipt=root/'patched-cancel-oneuse/result.json'
    receipt.write_text(receipt.read_text()+'\n')
    with pytest.raises(ValueError,match='Immutable real-process receipt'):
        M.inspect(root,baseline,candidate)


def test_real_original_failure_cannot_be_silently_converted_to_pass(tmp_path,monkeypatch):
    root,baseline,candidate,expect=fake_complete_evidence(tmp_path,monkeypatch)
    name='new-manual-replacement-held-asr-once'
    p=root/name/'result.json';d=json.loads(p.read_text());d['status']='no_stale_draft_after_cancel_or_new_manual';p.write_text(json.dumps(d))
    expect[name]['sha']=M.sha(p.read_bytes())
    with pytest.raises(ValueError,match='source status'):
        M.inspect(root,baseline,candidate)


def test_forged_successful_submission_in_history_denied(tmp_path,monkeypatch):
    root,baseline,candidate,_=fake_complete_evidence(tmp_path,monkeypatch)
    p=next((root/'patched-idle_reconnect_pending_asr-oneuse/conversations').glob('*.json'))
    d=json.loads(p.read_text());d['dialogue_history'][0]['submitted_text']='unexpected command';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Unverified stale/idle'):
        M.inspect(root,baseline,candidate)


def test_late_stale_speech_is_detected_after_the_patched_candidate(tmp_path,monkeypatch):
    root,baseline,candidate,_=fake_complete_evidence(tmp_path,monkeypatch)
    case=root/'patched-new_manual_on_reconnect-oneuse'
    p=next((case/'conversations').glob('*.json'))
    p.write_text(json.dumps({'dialogue_history':[{'status':'draft','requires_review':True}] }))
    with pytest.raises(ValueError,match='Persisted draft mismatch'):
        M.inspect(root,baseline,candidate)


def test_wrong_reconnect_event_history_does_not_count_as_pass(tmp_path,monkeypatch):
    root,baseline,candidate,_=fake_complete_evidence(tmp_path,monkeypatch)
    p=root/'patched-new_manual_on_reconnect-oneuse/voice-private.jsonl'
    p.write_text(json.dumps({'event':'server_start'})+'\n')
    with pytest.raises(ValueError,match='source logged event counts'):
        M.inspect(root,baseline,candidate)


def test_unexpected_remote_provider_url_forbidden_even_if_recorded_result_passes(tmp_path,monkeypatch):
    root,baseline,candidate,_=fake_complete_evidence(tmp_path,monkeypatch)
    p=root/'patched-cancel-oneuse/private-config.json'
    d=json.loads(p.read_text());d['llm']['url']='https://remote-provider.test/send';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='non-private provider'):
        M.inspect(root,baseline,candidate)


def test_changed_speaker_review_authority_forbidden(tmp_path,monkeypatch):
    root,baseline,candidate,_=fake_complete_evidence(tmp_path,monkeypatch)
    p=root/'patched-cancel-oneuse/private-config.json'
    d=json.loads(p.read_text());d['interaction']['require_verified_speaker_for_auto_send']=False;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='speaker protection'):
        M.inspect(root,baseline,candidate)


def test_wrong_patched_module_bytes_refuse_even_with_genuine_receipts(tmp_path,monkeypatch):
    root,baseline,candidate,_=fake_complete_evidence(tmp_path,monkeypatch)
    (candidate/'cloudflare_ws_voice_server.py').write_bytes(b'TAMPERED PRIVATE SOURCE')
    with pytest.raises(ValueError,match='Patched source is not'):
        M.inspect(root,baseline,candidate)


def test_out_of_scope_symlinked_source_or_case_denied(tmp_path,monkeypatch):
    root,baseline,candidate,_=fake_complete_evidence(tmp_path,monkeypatch)
    link=tmp_path/'symlink_private_root';link.symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError,match='symlinked private'):
        M.inspect(link,baseline,candidate)
    result=root/'patched-cancel-oneuse/result.json';raw=result.read_bytes();result.unlink()
    external=tmp_path/'other_original';external.write_bytes(raw)
    result.symlink_to(external)
    with pytest.raises(ValueError,match='Immutable real-process receipt'):
        M.inspect(root,baseline,candidate)


def test_duplicate_or_existing_verdict_is_never_overwritten(tmp_path,monkeypatch):
    root,baseline,candidate,_=fake_complete_evidence(tmp_path,monkeypatch)
    # This tmp_path is under the explicitly approved SWAAG private basetemp.
    out=tmp_path/'independent_verdict.json'
    args=['--evidence',str(root),'--baseline',str(baseline),'--candidate',str(candidate),'--out',str(out)]
    assert M.main(args)==0
    original=out.read_bytes()
    with pytest.raises(FileExistsError,match='Unique private'):
        M.main(args)
    assert out.read_bytes()==original
