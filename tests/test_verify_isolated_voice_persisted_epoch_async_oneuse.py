"""Source-pinned W70 one-use consent and no-network preflight contract.

All synthetic fixture documents and responses in these tests are FAKE.
Actual private Jetson one-use messages and no-tool evidence were verified
separately and are immutable; tests here do not re-run them.
"""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))

def module(name):
    p=ROOT/'scripts/demos'/name
    spec=importlib.util.spec_from_file_location('test_W70_'+name.replace('.py',''),p)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

A=module('verify_isolated_voice_persisted_epoch_on_new_manual_once.py')
L=module('verify_isolated_voice_legacy_epochless_restart_once.py')
F=module('verify_isolated_voice_nonblocking_intent_fsync_once.py')


def fixture(tmp_path,monkeypatch):
    s=tmp_path/'W70_synthetic_source';s.mkdir()
    raw={key:(key+' mocked source only').encode() for key in A.SUPPORTING_SOURCE_SHA}
    raw['cloudflare_ws_voice_server.py']=b'TEST ONLY: NO installed Jetson server source.'
    for name,v in raw.items():(s/name).write_bytes(v)
    other={n:hashlib.sha256(v).hexdigest() for n,v in raw.items() if n!='cloudflare_ws_voice_server.py'}
    monkeypatch.setattr(A,'SUPPORTING_SOURCE_SHA',other)
    monkeypatch.setattr(L,'SUPPORTING_SOURCE_SHA',other)
    monkeypatch.setattr(F,'SUPPORT',other)
    row=[{'status':'superseded','turn_id':'turn-old','asr_intent_epoch':1,
          'submitted_text':'','assistant':''},
         {'status':'draft','turn_id':'turn-current','asr_intent_epoch':2,
          'submitted_text':'','assistant':''}]
    hist=tmp_path/'real-shaped-FAKE-history.json';hist.write_text(json.dumps({'dialogue_history':row}))
    h=hashlib.sha256(hist.read_bytes()).hexdigest()
    for m in (A,L,F):monkeypatch.setattr(m,'ORIGINAL_HISTORY_SHA',h)
    cfg=tmp_path/'private-config-FAKE.json';cfg.write_text(json.dumps({'server':{'host':'127.0.0.1'}}))
    return s,hist,cfg,hashlib.sha256(raw['cloudflare_ws_voice_server.py']).hexdigest()


def cmd(m,src,hist,cfg,dest,model):
    return ['--source',str(src),'--original-history',str(hist),'--template',str(cfg),
         '--evidence',str(dest),'--expected-server-sha',model,
         '--mode','baseline_old' if m==A else 'corrected']


def test_actual_frozen_old_and_new_server_source_ids_are_declared():
    assert A.W69_SOURCE_SHA==F.OLD=='3d4e6e029aa4a467684c618dc31cfd60d53d77939843f547d394b04a979bad2f'
    assert F.NEW=='f3283c43a196595c4b1fade69e6f2ef0f23943dc6d0b91c5b5ab63f16c17c996'
    assert A.ORIGINAL_HISTORY_SHA=='bdde160a4996a1f38e8a88366bf2bb63e6ad3a65b2ebd26687326a03158102fc'
    assert F.ORIGINAL_HISTORY_SHA==A.ORIGINAL_HISTORY_SHA
    assert L.OPTIN != A.OPTIN != F.OPTIN


def test_original_and_new_restart_preflight_validates_only_source_not_network(tmp_path,monkeypatch,capsys):
    s,h,c,sha=fixture(tmp_path,monkeypatch)
    out=tmp_path/'must-not-appear'
    assert A.main(cmd(A,s,h,c,out,sha))==0
    d=json.loads(capsys.readouterr().out)
    assert d['status']=='source_pinned_fresh_manual_after_restart_preflight_only'
    assert d['original_persisted_epoch']==[1,2]
    assert not out.exists()


def test_derived_epochless_legacy_history_is_explicitly_not_original(tmp_path,monkeypatch,capsys):
    s,h,c,sha=fixture(tmp_path,monkeypatch)
    d=tmp_path/'not-started'
    assert L.main(cmd(L,s,h,c,d,sha))==0
    output=json.loads(capsys.readouterr().out)
    assert output['original_persisted_epoch']==[None,None]
    assert not d.exists()
    assert json.loads(h.read_text())['dialogue_history'][-1]['asr_intent_epoch']==2


def test_source_legacy_missing_original_history_sha_denied(tmp_path,monkeypatch):
    s,h,c,sha=fixture(tmp_path,monkeypatch)
    h.write_bytes(h.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Original real W69 history SHA'):
        A.main(cmd(A,s,h,c,tmp_path/'out',sha))


def test_refusal_when_original_was_edited_even_if_sha_field_valid(tmp_path,monkeypatch):
    s,h,c,sha=fixture(tmp_path,monkeypatch)
    p=s/'cloudflare_ws_voice_server.py';p.write_bytes(b'source deliberately changed')
    with pytest.raises(ValueError,match='Unknown server private module'):
        A.main(cmd(A,s,h,c,tmp_path/'out',sha))


def test_restart_oneuse_server_requires_optin_without_creating_attempt(tmp_path,monkeypatch):
    s,h,c,sha=fixture(tmp_path,monkeypatch)
    target=tmp_path/'unchanged';monkeypatch.delenv(A.OPTIN,raising=False)
    with pytest.raises(ValueError,match='Explicit single private test opt-in'):
        A.main(cmd(A,s,h,c,target,sha)+['--run'])
    assert not target.exists()


def test_replayed_or_uncertain_private_server_run_denied(tmp_path,monkeypatch):
    s,h,c,sha=fixture(tmp_path,monkeypatch)
    target=tmp_path/'already-attempted';target.mkdir();(target/'attempt.json').write_text('unknown effect')
    monkeypatch.setenv(A.OPTIN,'1')
    with pytest.raises(FileExistsError,match='Refuse any uncertain'):
        A.main(cmd(A,s,h,c,target,sha)+['--run'])
    assert (target/'attempt.json').read_text()=='unknown effect'


def test_event_loop_baseline_and_fixed_source_preflight_no_model_call(tmp_path,monkeypatch,capsys):
    s,h,c,sha=fixture(tmp_path,monkeypatch)
    target=tmp_path/'do-not-run'
    monkeypatch.setattr(F,'OLD',sha)
    args=['--source',str(s),'--original-history',str(h),'--template',str(c),
       '--evidence',str(target),'--expected-server-sha',sha,'--mode','old_blocking']
    assert F.main(args)==0
    assert json.loads(capsys.readouterr().out)['status']=='source_pinned_event_loop_stall_test_preflight_only'
    monkeypatch.setattr(F,'OLD','0'*64)
    # The harness must never start with an arbitrary source hash, even if the
    # synthetic fixture otherwise resembles a valid private conversation.
    with pytest.raises(ValueError,match='Unsupported frozen'):
        F.main(['--source',str(s),'--original-history',str(h),'--template',str(c),
           '--evidence',str(target),'--expected-server-sha',sha,'--mode','old_blocking'])
    assert not target.exists()


def test_event_loop_new_fairness_script_embeds_no_unapproved_provider_calls():
    source=(ROOT/'scripts/demos/verify_isolated_voice_nonblocking_intent_fsync_once.py').read_text()
    assert 'time.sleep(.70)' in source
    assert 'async def heartbeat()' in source
    assert 'await s.cancel_response' in source
    assert 'fixed_atomic_voice_intent_durable_write_no_event_loop_stall' in source
    assert 'systemctl restart' not in source
    assert 'HF_TOKEN' not in source


def test_new_real_websocket_acceptance_has_four_distinct_local_mode_scenarios():
    source=(ROOT/'scripts/demos/verify_isolated_voice_async_epoch_ws_once.py').read_text()
    assert source.count('SWAAG_W70_ASYNC_EPOCH_NOACTION_WEBSOCKET')==1
    for name in ('new_manual_on_reconnect','new_manual_disconnect_before_asr',
                 'cancel','idle_reconnect_pending_asr'):
        assert name in source
    assert 'new_server_sha256' in source
    assert 'SWAAG_W69_NEW_ATOMIC_WS_NOACTION' not in source
