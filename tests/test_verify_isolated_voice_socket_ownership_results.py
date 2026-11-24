"""Falsification tests for W71 actual source/WS evidence acceptance.

Everything created in this test file is synthetic, never a Jetson server
run or voice model. Actual eight immutable source test receipts and the
independent disk/log verifier live only in private evidence.
"""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))
SPEC=importlib.util.spec_from_file_location('synthetic_only_W71_evidence_checks',ROOT/'scripts/demos/verify_isolated_voice_socket_ownership_results.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def fixture(tmp_path,monkeypatch):
    evidence=tmp_path/'fake-oneuse-source-result-fixture';evidence.mkdir()
    old=tmp_path/'fake-W70';old.mkdir();new=tmp_path/'fake-W71';new.mkdir()
    vals={name:(name+' synthetic data NOT original voice server').encode() for name in M.BASE}
    for name,blob in vals.items():
        (old/name).write_bytes(blob)
        (new/name).write_bytes(blob+(b'FAKE-SOURCE-FIX' if name=='cloudflare_ws_voice_server.py' else b''))
    monkeypatch.setattr(M,'BASE',{name:M.sha(old/name) for name in vals})
    monkeypatch.setattr(M,'NEW_SHA',M.sha(new/'cloudflare_ws_voice_server.py'))
    monkeypatch.setattr(M,'patch',lambda data:data+b'FAKE-SOURCE-FIX')
    manifest=new/'source-manifest.json';manifest.write_text(json.dumps({
        'new_undeployed_server_sha256':M.NEW_SHA,'only_changed_file':'cloudflare_ws_voice_server.py'}))
    monkeypatch.setattr(M,'MANIFEST_SHA',M.sha(manifest))
    hist=tmp_path/'fake-W69-source-disk-history.json';hist.write_text(json.dumps({'dialogue_history':[
        {'turn_id':'turn-old','status':'superseded','asr_intent_epoch':1,'submitted_text':'','assistant':''},
        {'turn_id':'turn-current','status':'draft','asr_intent_epoch':2,'submitted_text':'','assistant':''}]}))
    monkeypatch.setattr(M,'ORIGINAL_HISTORY_SHA',M.sha(hist))
    scenarios={}
    for name,(_,status,family,source_sha,*metadata) in M.SCENARIOS.items():
        src=M.BASE['cloudflare_ws_voice_server.py'] if family.startswith('old_') else M.NEW_SHA
        # old_control/new_control real own source returns statuses at [2]/[3].
        rows=[]
        if family in ('old_control','new_control'):
            rows=[{'status':'superseded','submitted_text':'','assistant':''} for _ in range(2)]
            if family=='new_control':rows.append({'status':'draft','submitted_text':'','assistant':''})
        elif family in ('old_retry','new_retry'):
            rows=[{'status':'superseded','submitted_text':'','assistant':''},
                  {'status':'draft','submitted_text':'','assistant':''},
                  {'status':'draft','submitted_text':'','assistant':''}]
        elif family=='new_WS':
            rows=[{'status':'draft','requires_review':True,'submitted_text':'','assistant':''} for _ in range(metadata[1])]
        folder=evidence/name;folder.mkdir();(folder/'conversations').mkdir()
        (folder/'conversations'/'one.json').write_text(json.dumps({'dialogue_history':rows}))
        result={'status':status,'source_sha256':src,'failure_type':None,
             'elapsed_seconds':.6,'live_voice_phone_modified':False,'real_tool_calls':0,
             'real_model_ASR_TTS_agent_tool_actions':0}
        if family=='old_control':result.update({'new_manually_captured_user_speech_accepted':False,'epoch_after_delayed_old_control':4})
        if family=='new_control':result.update({'new_manually_captured_user_speech_accepted':True,'epoch_after_delayed_old_control':3})
        if family in ('old_retry','new_retry'):
            result['old_client_submit_boundary_invoked']=int(family=='old_retry')
        if family=='new_WS':
            result.update({'last_history_turns':metadata[1],'held_authoritative_ASR_requests':1,
                'input_recording_SHA256':M.VOICE_SHA,'real_model_ASR_TTS_agent_tool_calls':0,
                'actual_Jetson_production_service_changed':False,'actual_private_child_pid':999999})
            log=[{'event':'server_start'}]+[{'event':'session_open'}]*metadata[0]
            (folder/'voice-private.jsonl').write_text('\n'.join(json.dumps(x) for x in log)+'\n')
            cfg={'server':{'host':'127.0.0.1','auth_token_file':str(folder/'fake-only-bearer')},
                'interaction':{'require_verified_speaker_for_auto_send':True},
                'llm':{'url':'http://127.0.0.1:9/run'},
                'asr':{'service_url':'http://127.0.0.1:9/fake'},
                'tts':{'service_url':'http://127.0.0.1:9/fake'}}
            (folder/'private-config.json').write_text(json.dumps(cfg))
        receipt=folder/'result.json';receipt.write_text(json.dumps(result))
        scenarios[name]=(M.sha(receipt),status,family,src,*metadata)
    monkeypatch.setattr(M,'SCENARIOS',scenarios)
    return evidence,old,new,hist


def test_true_original_immutable_sha_contracts_are_declared():
    assert len(M.SCENARIOS)==8
    assert M.NEW_SHA=='873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf'
    assert M.BASELINE=='f3283c43a196595c4b1fade69e6f2ef0f23943dc6d0b91c5b5ab63f16c17c996'
    assert M.MANIFEST_SHA=='f361faa9283b55e708f4c7f590d3b0e5a90bd1eeb9aa16fe2f88876e8c6df6d0'
    assert M.SCENARIOS['guarded-ws-new_manual_disconnect_before_asr-once'][4]==3


def test_synthetic_w71_comparison_acceptance_does_not_count_as_real_data(tmp_path,monkeypatch):
    a=fixture(tmp_path,monkeypatch)
    d=M.verify(*a)
    assert d['total_independently_checked_new_source_or_process_cases']==8
    assert d['observed_original_two_control_failures_without_actual_tool_effects']==2
    assert d['observed_corrected_source_two_control_acceptances']==2
    assert d['observed_new_real_private_PCM_websocket_server_cases']==4
    assert d['no_new_model_ASR_TTS_or_SWAAG_tool_calls']


def test_tampered_original_private_server_source_rejected(tmp_path,monkeypatch):
    root,old,new,hist=fixture(tmp_path,monkeypatch)
    (old/'voice_ws_config.py').write_bytes(b'changed after expectation')
    with pytest.raises(ValueError,match='Original source byte changed'):
        M.verify(root,old,new,hist)


def test_tampered_new_private_server_source_rejected(tmp_path,monkeypatch):
    root,old,new,hist=fixture(tmp_path,monkeypatch)
    (new/'cloudflare_ws_voice_server.py').write_bytes(b'changed after test')
    with pytest.raises(ValueError,match='Exactly one modified source file'):
        M.verify(root,old,new,hist)


def test_missing_frozen_real_W69_history_denied(tmp_path,monkeypatch):
    root,old,new,hist=fixture(tmp_path,monkeypatch)
    hist.write_bytes(hist.read_bytes()+b' ')
    with pytest.raises(ValueError,match='original private history bytes changed'):
        M.verify(root,old,new,hist)


def test_source_result_hash_changed_no_fabricated_success(tmp_path,monkeypatch):
    root,old,new,hist=fixture(tmp_path,monkeypatch)
    p=root/'corrected-v2-v2-verify_isolated_voice_old_socket_control_epoch_once/result.json'
    p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='Immutable oneuse receipt SHA modified'):
        M.verify(root,old,new,hist)


def test_forged_old_stale_control_after_replacement_is_detected(tmp_path,monkeypatch):
    root,old,new,hist=fixture(tmp_path,monkeypatch)
    name='corrected-v2-v2-verify_isolated_voice_old_socket_control_epoch_once'
    p=root/name/'result.json';j=json.loads(p.read_text());j['epoch_after_delayed_old_control']=4;p.write_text(json.dumps(j))
    t=M.SCENARIOS[name];sc=dict(M.SCENARIOS);sc[name]=(M.sha(p),*t[1:]);monkeypatch.setattr(M,'SCENARIOS',sc)
    with pytest.raises(ValueError,match='source fix'):
        M.verify(root,old,new,hist)


def test_old_ACK_retry_reaches_no_action_stub_but_fixed_must_not(tmp_path,monkeypatch):
    root,old,new,hist=fixture(tmp_path,monkeypatch)
    name='corrected-v2-v2-verify_isolated_voice_stale_socket_ack_retry_once'
    p=root/name/'result.json';j=json.loads(p.read_text());j['old_client_submit_boundary_invoked']=1;p.write_text(json.dumps(j))
    t=M.SCENARIOS[name];sc=dict(M.SCENARIOS);sc[name]=(M.sha(p),*t[1:]);monkeypatch.setattr(M,'SCENARIOS',sc)
    with pytest.raises(ValueError,match='stale socket would-call-tool boundary missing'):
        M.verify(root,old,new,hist)


def test_real_websocket_replacement_log_count_mismatch_denied(tmp_path,monkeypatch):
    root,old,new,hist=fixture(tmp_path,monkeypatch)
    p=root/'guarded-ws-new_manual_disconnect_before_asr-once/voice-private.jsonl';p.write_text(json.dumps({'event':'server_start'})+'\n'+json.dumps({'event':'session_open'})+'\n')
    with pytest.raises(ValueError,match='number of running new WebSocket sessions wrong'):
        M.verify(root,old,new,hist)


def test_hidden_remote_tool_effect_in_ws_log_denied(tmp_path,monkeypatch):
    root,old,new,hist=fixture(tmp_path,monkeypatch)
    p=root/'guarded-ws-cancel-once/voice-private.jsonl';p.write_text(p.read_text()+json.dumps({'event':'turn_submitted'})+'\n')
    with pytest.raises(ValueError,match='tool submission unexpectedly logged'):
        M.verify(root,old,new,hist)


def test_private_source_with_external_model_URL_denied(tmp_path,monkeypatch):
    root,old,new,hist=fixture(tmp_path,monkeypatch)
    p=root/'guarded-ws-cancel-once/private-config.json';cfg=json.loads(p.read_text());cfg['llm']['url']='https://provider.example/remote';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='strictly noaction local'):
        M.verify(root,old,new,hist)


def test_original_noaction_control_was_actually_not_an_external_tool_call(tmp_path,monkeypatch):
    root,old,new,hist=fixture(tmp_path,monkeypatch)
    p=root/'old-source-stale-text-idempotency-once/result.json';j=json.loads(p.read_text());j['real_tool_calls']=1;p.write_text(json.dumps(j))
    t=M.SCENARIOS['old-source-stale-text-idempotency-once'];sc=dict(M.SCENARIOS);sc['old-source-stale-text-idempotency-once']=(M.sha(p),*t[1:]);monkeypatch.setattr(M,'SCENARIOS',sc)
    with pytest.raises(ValueError,match='genuine seeded ACK-lost'):
        M.verify(root,old,new,hist)


def test_existing_private_verdict_must_not_be_overwritten(tmp_path,monkeypatch):
    root,old,new,hist=fixture(tmp_path,monkeypatch)
    out=tmp_path/'existing-verdict.json';out.write_text('do not overwrite')
    args=['--evidence',str(root),'--original-source',str(old),'--patched-source',str(new),'--original-history',str(hist),'--output',str(out)]
    with pytest.raises(FileExistsError,match='unused new private'):
        M.main(args)
    assert out.read_text()=='do not overwrite'
