"""No-action one-use test admission and freeze rules for W71 source trials.

These synthetic tests do not run any model or replace phone/WS behavior.
Actual eight private receipts are retained separately, with original
source-side failures and genuine separate private WebSocket server runs.
"""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))

def load(file):
    path=ROOT/'scripts/demos'/file
    spec=importlib.util.spec_from_file_location('w71_'+file[:-3],path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

S=load('verify_isolated_voice_old_socket_control_epoch_once.py')
T=load('verify_isolated_voice_stale_socket_ack_retry_once.py')
W=load('verify_isolated_voice_owned_socket_ws_once.py')


def fake_source(tmp_path,monkeypatch,module):
    src=tmp_path/'synthetic-inert-source';src.mkdir()
    raw={n:(n+' only synthetic fake model-free source').encode() for n in S.BASE}
    for n,body in raw.items():(src/n).write_bytes(body)
    digest={n:hashlib.sha256(blob).hexdigest() for n,blob in raw.items()}
    monkeypatch.setattr(S,'BASE',digest)
    monkeypatch.setattr(T,'SUPPORT',{n:hash for n,hash in digest.items() if n!='cloudflare_ws_voice_server.py'})
    original=tmp_path/'synthetic-history.json'
    original.write_text(json.dumps({'dialogue_history':[
        {'turn_id':'turn-old','status':'superseded','asr_intent_epoch':1},
        {'turn_id':'turn-current','status':'draft','asr_intent_epoch':2}]}))
    historical=hashlib.sha256(original.read_bytes()).hexdigest()
    monkeypatch.setattr(S,'ORIGINAL_HISTORY_SHA',historical)
    monkeypatch.setattr(T,'ORIGINAL_HISTORY_SHA',historical)
    cfg=tmp_path/'synthetic-config.json';cfg.write_text('{}')
    return src,original,cfg,digest['cloudflare_ws_voice_server.py']


def args(m,src,hist,cfg,source_hash,out):
    return ['--source',str(src),'--original-history',str(hist),'--template',str(cfg),
        '--evidence',str(out),'--source-sha',source_hash,'--mode','old_source']


def test_true_frozen_public_source_sha_and_four_private_real_ws_cases():
    assert S.NEW_SERVER_SHA=='873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf'
    assert T.W71_NEW_SHA==S.NEW_SERVER_SHA
    assert S.BASE['cloudflare_ws_voice_server.py']=='f3283c43a196595c4b1fade69e6f2ef0f23943dc6d0b91c5b5ab63f16c17c996'
    assert W.OPT_IN=='SWAAG_W71_OWNED_SESSION_GUARD_NOACTION_WS'
    assert W.PATCHED_SERVER_SHA256==S.NEW_SERVER_SHA


def test_old_control_frozen_source_input_preflight_never_starts_a_process(tmp_path,monkeypatch,capsys):
    src,hist,cfg,h=fake_source(tmp_path,monkeypatch,S)
    old=tmp_path/'not-created'
    monkeypatch.setattr(S,'NEW_SERVER_SHA',h)
    assert S.main(args(S,src,hist,cfg,h,old))==0
    report=json.loads(capsys.readouterr().out)
    assert report['status']=='private_inflight_socket_source_test_preflight_only'
    assert report['no_live_or_model_actions']
    assert not old.exists()


def test_ack_retry_frozen_source_preflight_does_not_do_tools(tmp_path,monkeypatch,capsys):
    src,hist,cfg,h=fake_source(tmp_path,monkeypatch,T)
    monkeypatch.setattr(T,'W71_NEW_SHA',h)
    out=tmp_path/'never-created'
    assert T.main(args(T,src,hist,cfg,h,out))==0
    d=json.loads(capsys.readouterr().out)
    assert d['status']=='original_new_source_ack_retry_preflight_noaction'
    assert d['real_provider_tool_calls']==0
    assert not out.exists()


def test_source_changed_after_original_sha_pin_refused_before_start(tmp_path,monkeypatch):
    src,hist,cfg,h=fake_source(tmp_path,monkeypatch,T)
    monkeypatch.setattr(T,'W71_NEW_SHA',h)
    (src/'voice_ws_core.py').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Unqualified model-free source module'):
        T.main(args(T,src,hist,cfg,h,tmp_path/'out'))
    assert not (tmp_path/'out').exists()


def test_original_history_changed_after_sha_pin_refused(tmp_path,monkeypatch):
    src,hist,cfg,h=fake_source(tmp_path,monkeypatch,S)
    hist.write_bytes(hist.read_bytes()+b' ')
    monkeypatch.setattr(S,'NEW_SERVER_SHA',h)
    with pytest.raises(ValueError,match='Historic real durable source SHA mismatch'):
        S.main(args(S,src,hist,cfg,h,tmp_path/'out'))


def test_no_first_effect_without_explicit_oneuse_optin(tmp_path,monkeypatch):
    src,hist,cfg,h=fake_source(tmp_path,monkeypatch,S)
    monkeypatch.setattr(S,'NEW_SERVER_SHA',h)
    monkeypatch.delenv(S.OPTIN,raising=False)
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='One-use noaction test requires explicit opt-in'):
        S.main(args(S,src,hist,cfg,h,out)+['--run'])
    assert not out.exists()


def test_uncertain_attempt_or_previously_finished_oneuse_never_replayed(tmp_path,monkeypatch):
    src,hist,cfg,h=fake_source(tmp_path,monkeypatch,S)
    monkeypatch.setattr(S,'NEW_SERVER_SHA',h)
    monkeypatch.setenv(S.OPTIN,'1')
    out=tmp_path/'already_started';out.mkdir();(out/'attempt.json').write_text('unknown')
    with pytest.raises(FileExistsError,match='Prior/unknown'):
        S.main(args(S,src,hist,cfg,h,out)+['--run'])
    assert (out/'attempt.json').read_text()=='unknown'


def test_source_result_never_installs_to_owner_or_external_destination():
    body=(ROOT/'scripts/demos/verify_isolated_voice_stale_socket_ack_retry_once.py').read_text()
    assert 'systemctl restart' not in body
    assert 'subprocess.Popen' not in body
    assert 'real_tool_calls' in body
    assert 'old.submit_turn=NO_ACTION_submission' in body
    assert 'one-use' in body.lower()


def test_test_source_code_fences_prior_attempt_before_any_work():
    for name in ('verify_isolated_voice_old_socket_control_epoch_once.py',
      'verify_isolated_voice_stale_socket_ack_retry_once.py',
      'verify_isolated_voice_owned_socket_ws_once.py'):
        code=(ROOT/'scripts/demos'/name).read_text()
        assert 'root.exists()' in code or 'x.evidence.exists()' in code
        assert any(marker in code for marker in ('if os.environ.get(OPTIN)', 'if os.getenv(OPTIN)', 'if os.getenv(OPT_IN)'))
        assert 'NO_ACTION' in code or 'no-action' in code or 'noaction' in code
