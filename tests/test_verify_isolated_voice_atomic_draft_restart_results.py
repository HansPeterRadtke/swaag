"""Independent restart-proof oracle rejects forged evidence (synthetic only).

The fixtures in this file are NOT real server runs. Exact original immutable
Jetson process and source evidence is checked separately by the real verifier
with its independently fixed hashes and retained real history/log files.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))
SPEC=importlib.util.spec_from_file_location('atomic_real_recovery_postmortem_source_oracle',ROOT/'scripts/demos/verify_isolated_voice_atomic_draft_restart_results.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def fake_real_shaped_evidence(tmp_path,monkeypatch):
    root=tmp_path/'synthetic-durable-restart-output';root.mkdir()
    source=tmp_path/'unreleased-source';source.mkdir()
    originals={n:(n+'\nFAKE_BYTES_ONLY\n').encode() for n in M.BASE}
    for n,b in originals.items():(source/n).write_bytes(b)
    monkeypatch.setattr(M,'BASE',{n:M.sha(source/n) for n in originals})
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256',M.sha(source/'cloudflare_ws_voice_server.py'))
    old={'turn_id':'turn-old','status':'superseded','asr_intent_epoch':1,'submitted_text':'',
        'assistant':'','assistant_audio_path':'','recognized_text':'old source only'}
    new={'turn_id':'turn-current','status':'draft','asr_intent_epoch':2,'requires_review':True,
         'submitted_text':'','assistant':'','assistant_audio_path':'','recognized_text':'new user speech'}
    original=tmp_path/'forced-real-shape-source.json';original.write_text(json.dumps({'dialogue_history':[old,new]}))
    monkeypatch.setattr(M,'ORIGINAL_HISTORY_SHA',M.sha(original))
    actual_cases=[];history_sha={};log_sha={}
    for mode in ('latest_new_draft','superseded_only'):
        folder=root/mode;folder.mkdir();(folder/'conversations').mkdir()
        hist=folder/'conversations'/(M.CONVERSATION+'.json')
        hist.write_text(json.dumps({'dialogue_history':[old,new] if mode=='latest_new_draft' else [old]}))
        history_sha[mode]=M.sha(hist)
        logs=[{'event':'server_start'},{'event':'session_open'},{'event':'speaker_identity_unverified_manual_review'}]
        log=folder/'voice-private.jsonl';log.write_text('\n'.join(map(json.dumps,logs))+'\n');log_sha[mode]=M.sha(log)
        cfg={'server':{'host':'127.0.0.1','auth_token_file':str(folder/'fake-only-bearer')},
             'interaction':{'require_verified_speaker_for_auto_send':True,'auto_send_default':True},
             'llm':{'url':'http://127.0.0.1:9/run'}}
        (folder/'private-config.json').write_text(json.dumps(cfg))
        actual_cases.append({'mode':mode,'restored_active_review_count':int(mode=='latest_new_draft'),
             'private_owned_child_pid':999999 if mode=='latest_new_draft' else 999998})
    monkeypatch.setattr(M,'HISTORY_SHA',history_sha)
    monkeypatch.setattr(M,'LOG_SHA',log_sha)
    receipt={'status':'both_real_private_source_restart_draft_recovery_cases_passed',
      'source_sha256':M.PATCHED_SERVER_SHA256,
      'source_durable_history_sha256':M.ORIGINAL_HISTORY_SHA,
      'one_actual_new_speaker_review_draft':True,
      'superseded_old_speech_never_restored':True,
      'no_ASR_LLMTTS_SWAAG_tools_or_external_effects':True,
      'actual_owner_phone_or_live_service_modified':False,
      'cases':actual_cases}
    p=root/'result.json';p.write_text(json.dumps(receipt))
    monkeypatch.setattr(M,'RESULT_SHA',M.sha(p))
    return root,original,source


def test_exact_original_source_and_real_oneuse_observation_hashes():
    assert M.RESULT_SHA=='bf2d0a7b94c59004f3dfa14d1bfc92d193754b72db6d2ddfa8a562b4cf462134'
    assert M.ORIGINAL_HISTORY_SHA=='bdde160a4996a1f38e8a88366bf2bb63e6ad3a65b2ebd26687326a03158102fc'
    assert M.PATCHED_SERVER_SHA256=='3d4e6e029aa4a467684c618dc31cfd60d53d77939843f547d394b04a979bad2f'
    assert set(M.HISTORY_SHA)=={'latest_new_draft','superseded_only'}
    assert set(M.LOG_SHA)==set(M.HISTORY_SHA)


def test_synthetic_shape_can_be_verified_but_is_not_real_proof(tmp_path,monkeypatch):
    args=fake_real_shaped_evidence(tmp_path,monkeypatch)
    d=M.inspect(*args)
    assert d['status']=='independent_real_server_restart_restores_latest_draft_but_not_superseded_old_speech'
    assert d['independent_verified_case_count']==2
    assert [c['restored_active_manual_review_draft_count'] for c in d['cases']]==[1,0]
    assert d['model_or_provider_calls']==0
    assert d['production_Jetson_service_modified'] is False


def test_source_bytes_change_detected_without_rerunning_private_server(tmp_path,monkeypatch):
    root,hist,source=fake_real_shaped_evidence(tmp_path,monkeypatch)
    (source/'voice_ws_core.py').write_text('tampered')
    with pytest.raises(ValueError,match='source changed'):
        M.inspect(root,hist,source)


def test_original_history_reordered_changed_refused(tmp_path,monkeypatch):
    root,hist,source=fake_real_shaped_evidence(tmp_path,monkeypatch)
    hist.write_text(hist.read_text()+'\n')
    with pytest.raises(ValueError,match='history changed'):
        M.inspect(root,hist,source)


def test_forged_original_result_after_the_fact_refused(tmp_path,monkeypatch):
    root,hist,source=fake_real_shaped_evidence(tmp_path,monkeypatch)
    p=root/'result.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='Immutable two-process result changed'):
        M.inspect(root,hist,source)


def test_stale_old_speech_reinserted_as_draft_detected(tmp_path,monkeypatch):
    root,hist,source=fake_real_shaped_evidence(tmp_path,monkeypatch)
    p=root/'latest_new_draft/conversations'/f'{M.CONVERSATION}.json'
    x=json.loads(p.read_text());x['dialogue_history'][0]['status']='draft';p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='persisted restart history changed'):
        M.inspect(root,hist,source)


def test_forged_owner_submission_in_history_denied(tmp_path,monkeypatch):
    root,hist,source=fake_real_shaped_evidence(tmp_path,monkeypatch)
    p=root/'latest_new_draft/conversations'/f'{M.CONVERSATION}.json'
    x=json.loads(p.read_text());x['dialogue_history'][1]['submitted_text']='wrong send';p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='persisted restart history changed'):
        M.inspect(root,hist,source)


def test_wrong_actual_server_log_outcome_denied(tmp_path,monkeypatch):
    root,hist,source=fake_real_shaped_evidence(tmp_path,monkeypatch)
    p=root/'superseded_only/voice-private.jsonl'
    p.write_text(p.read_text()+json.dumps({'event':'turn_submitted'})+'\n')
    with pytest.raises(ValueError,match='child logs changed'):
        M.inspect(root,hist,source)


def test_wrong_speaker_guard_configuration_denied(tmp_path,monkeypatch):
    root,hist,source=fake_real_shaped_evidence(tmp_path,monkeypatch)
    p=root/'latest_new_draft/private-config.json';x=json.loads(p.read_text())
    x['interaction']['require_verified_speaker_for_auto_send']=False;p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='not isolated/fail-closed'):
        M.inspect(root,hist,source)


def test_remote_provider_route_in_fake_private_server_denied(tmp_path,monkeypatch):
    root,hist,source=fake_real_shaped_evidence(tmp_path,monkeypatch)
    p=root/'latest_new_draft/private-config.json';x=json.loads(p.read_text())
    x['llm']['url']='https://external-provider.test/run';p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='not isolated/fail-closed'):
        M.inspect(root,hist,source)


def test_preexisting_signed_result_cannot_be_overwritten(tmp_path,monkeypatch,capsys):
    root,hist,source=fake_real_shaped_evidence(tmp_path,monkeypatch)
    out=tmp_path/'independent-only-verdict.json'
    args=['--evidence',str(root),'--source-history',str(hist),'--candidate-source',str(source),'--output',str(out)]
    assert M.main(args)==0
    original=out.read_bytes()
    with pytest.raises(FileExistsError,match='Unique private result'):
        M.main(args)
    assert out.read_bytes()==original
