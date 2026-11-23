"""No-effect regression for pinned REAL voice-process+Whisper admission."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))
P=ROOT/'scripts/demos/verify_isolated_voice_real_audio_reconnect.py'
SPEC=importlib.util.spec_from_file_location('private_real_asr_process',P)
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_only_precise_real_private_asr_relay_is_allowlisted(tmp_path):
    original={'server':{'auth_token_file':'/home/owner/real'},
              'storage':{},'asr':{'final_quality_enabled':True,'service_url':'http://thor:13484/old',
              'secondary_service_url':'http://thor:13484/secondary','reconcile_url':'http://thor:13484/reconcile',
              'final_quality_url':'http://thor:13484/v1/voice/transcribe-verify',
              'health_url':'http://thor:13484/health'},
              'llm':{'url':'http://thor/agent','health_url':'http://thor/health'},
              'tts':{'service_url':'http://nitro/phonon','service_health_url':'http://nitro/health'},
              'worker_events':{},'interaction':{'auto_send_default':True},
              'logging':{'include_content':True}}
    d=M.private_config(original,tmp_path,19345,20982)
    assert d['asr']['final_quality_url']=='http://127.0.0.1:20982/v1/voice/transcribe-verify'
    assert d['asr']['final_quality_enabled'] is True
    assert d['server']['host']=='127.0.0.1' and d['server']['port']==19345
    assert d['server']['auth_token_file'].startswith(str(tmp_path))
    assert d['interaction']['require_verified_speaker_for_auto_send'] is True
    assert d['interaction']['auto_send_default'] is True
    assert d['interaction']['require_client_commit'] is True
    for group in ('asr','llm','tts'):
        for key,val in d[group].items():
            if key.endswith('_url') or key=='url':
                if key=='final_quality_url' and group=='asr':continue
                assert str(val).startswith('http://127.0.0.1:9/')
    assert 'thor' not in json.dumps(d) and 'nitro' not in json.dumps(d)
    assert original['asr']['service_url']=='http://thor:13484/old'


def test_real_relay_rejects_wrong_path_duplicate_and_wrong_audio():
    # Invalid local requests are allowed to reach the PRIVATE listening
    # socket, but MUST NOT reach the real resident ASR model.
    import urllib.error,urllib.request
    pcm=b'\x22\x00'*26000
    relay=M.ASRAuthoritativeRelay(pcm)
    port=relay.start()
    try:
        def attempt(path,raw):
            try:
                with urllib.request.urlopen(urllib.request.Request(
                      f'http://127.0.0.1:{port}{path}',data=raw,
                      method='POST'),timeout=2) as resp:return resp.status
            except urllib.error.HTTPError as exc:return exc.code
        assert attempt('/invalid',b'RIFF'+pcm)==409
        assert attempt('/v1/voice/transcribe-verify',b'RIFF'+pcm)==409
        assert len(relay.calls)==2
    finally:relay.stop()


def test_real_relay_refuses_audio_that_is_not_original_without_forwarding(monkeypatch):
    import urllib.request,urllib.error
    pcm=b'\x20\x00'*32000
    relay=M.ASRAuthoritativeRelay(pcm);port=relay.start()
    original=urllib.request.urlopen
    calls=[]
    # Any attempt to call the remote verified model would mean our negative
    # test failed. Only local test requests are allowed here.
    def stop_remote(request,*a,**kw):
        if getattr(request,'full_url','').startswith('http://thor:'):
            calls.append(1);raise AssertionError('Remote ASR must not be called')
        return original(request,*a,**kw)
    monkeypatch.setattr(M.urllib.request,'urlopen',stop_remote)
    try:
        bad=b'RIFF'+b'\x00'*50000
        try:urllib.request.urlopen(urllib.request.Request(
             f'http://127.0.0.1:{port}/v1/voice/transcribe-verify',data=bad),timeout=3)
        except urllib.error.HTTPError as exc:assert exc.code==502
        else:pytest.fail('Unrelated audio was unexpectedly forwarded')
        assert len(relay.calls)==1 and relay.calls[0]['failure_type']=='ValueError'
        assert calls==[]
    finally:relay.stop()


def test_live_optin_is_required_and_existing_receipt_never_replayed(monkeypatch,tmp_path,capsys):
    # Test *only* admission, not original real-source hashes, which are
    # covered by existing verify_voice_tts_replay_noaction / release tests.
    monkeypatch.setattr(M,'evidence_guard',lambda *a:({},b'\x00\x01'*22000,'The source contains this complete spoken answer for testing.'))
    monkeypatch.delenv(M.OPT_IN,raising=False)
    dirs=[tmp_path/x for x in ('src','helper','template','receipt','audio')]
    cmd=sum((['--'+name,str(d)] for name,d in zip(
        ('source','helpers','template','receipt','audio'),dirs)),[])
    dest=tmp_path/'result-unique'
    cmd+=['--evidence',str(dest)]
    assert M.main(cmd)==0
    assert not dest.exists()
    assert json.loads(capsys.readouterr().out)['status']=='no_network_source_and_audio_preflight'
    with pytest.raises(ValueError,match='explicit environment'):
        M.main(cmd+['--run'])
    assert not dest.exists()
    called=[]
    def pretend(*args):
        called.append(1)
        return {'status':'real_process_real_WAV_ASR_reconnect_review_verified',
           'automatic_submissions':0}
    monkeypatch.setattr(M,'run_private_child',pretend)
    monkeypatch.setenv(M.OPT_IN,'1')
    assert M.main(cmd+['--run'])==0
    assert len(called)==1
    assert (dest/'attempt.json').is_file() and (dest/'result.json').is_file()
    with pytest.raises(FileExistsError):M.main(cmd+['--run'])
    assert len(called)==1


def test_any_failure_preserved_and_uncertain_external_attempt_never_replayed(monkeypatch,tmp_path):
    monkeypatch.setattr(M,'evidence_guard',lambda *a:({},b'\x00\x01'*22000,'The source contains this complete spoken answer for testing.'))
    monkeypatch.setenv(M.OPT_IN,'1')
    attempted=[]
    def fail(*args):attempted.append(1);raise TimeoutError('Resident ASR stalled')
    monkeypatch.setattr(M,'run_private_child',fail)
    cmd=[]
    for k in ('source','helpers','template','receipt','audio','evidence'):
        cmd+=['--'+k,str(tmp_path/k)]
    assert M.main(cmd+['--run'])==2
    d=json.loads((tmp_path/'evidence/result.json').read_text())
    assert d['status']=='failed_or_unknown' and d['failure_type']=='TimeoutError'
    with pytest.raises(FileExistsError):M.main(cmd+['--run'])
    assert attempted==[1]


def test_same_server_process_and_audit_event_requirements_are_in_implementation():
    text=P.read_text()
    for snippet in ('child=subprocess.Popen(', 'same_actual_child_stayed_active_across_socket_reconnect',
                    'real_ws_scenario(port,pcm,reply,relay)', 'relay.calls',
                    "'automatic_submissions':0", "'real_human_owner_identity_verified':False",
                    "x.get('type') in ('turn_proposal','audio')", 'speaker_identity_unverified',
                    'forged-privileged-owner-turn','validate_inputs(src,helper,receipt,audio)'):
        assert snippet in text, snippet
