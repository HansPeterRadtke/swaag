"""Source-only negative tests. Real live end-to-end acceptance is explicit opt-in."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import wave

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/demos/verify_real_voice_phonon_reconnect.py'
SPEC=importlib.util.spec_from_file_location('voice_phonon_acceptance',SCRIPT)
DEMO=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


def proof(tmp_path, **overrides):
    text='Is this a passage you want me to identify or verify, or are you sharing it for another purpose?'
    obj={'canonical_reply':text,'reply_sha256':DEMO.CANONICAL_REPLY_SHA256,
         'source_journal_sha256':DEMO.CANONICAL_JOURNAL_SHA256,
         'source_verified_no_tools':True,'source_finished':True,
         'not_fresh_voice_response':True,'not_authorized_for_external_action':True}
    obj.update(overrides)
    p=tmp_path/'proof.json';p.write_text(json.dumps(obj))
    return p


def wav_bytes(rate=16000,frames=32000,pcm_value=3):
    raw=io.BytesIO()
    with wave.open(raw,'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate)
        f.writeframes(pcm_value.to_bytes(2,'little',signed=True)*frames)
    return raw.getvalue()


def test_correct_durable_reply_is_accepted_without_model_request(tmp_path):
    p=proof(tmp_path)
    assert DEMO.load_verified_source(p).startswith('Is this a passage')
    with pytest.raises(ValueError,match='verified canonical'):
        DEMO.load_verified_source(proof(tmp_path,canonical_reply='I completed the action.'))


@pytest.mark.parametrize('key,value',[
  ('source_verified_no_tools',False),('source_finished',False),
  ('not_fresh_voice_response',False),('not_authorized_for_external_action',False),
  ('source_journal_sha256','unsupported source'),('reply_sha256','wrong')])
def test_source_claims_are_not_trusted_without_exact_sha(tmp_path,key,value):
    with pytest.raises(ValueError,match='Missing durable'):
        DEMO.load_verified_source(proof(tmp_path,**{key:value}))


@pytest.mark.parametrize('endpoint',[
  'https://127.0.0.1:15305/synthesize',
  'http://192.168.1.10:15305/synthesize',
  'http://0.0.0.0:15305/synthesize',
  'http://127.0.0.1:15304/synthesize',
  'http://127.0.0.1:15305/health',
  'http://user@127.0.0.1:15305/synthesize',
  'http://127.0.0.1:15305/synthesize?reply=1'])
def test_only_fixed_local_tts_endpoint_is_admitted(endpoint):
    with pytest.raises(ValueError,match='existing Jetson-local'):
        DEMO.safe_loopback_tts_url(endpoint)


def test_valid_wav_checks_format_duration_and_nonzero_pcm():
    raw=wav_bytes()
    result=DEMO.inspect_wav(raw)
    assert result['duration_seconds']==2
    assert result['pcm_bytes']==64000
    assert result['wav_sha256']==hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('broken',[
    b'not a wav',wav_bytes(rate=22050),wav_bytes(frames=800),
    wav_bytes(frames=16000*31),wav_bytes(pcm_value=0),
    wav_bytes()[:64]])
def test_corrupt_or_unbounded_tts_cannot_become_spoken_reply(broken):
    with pytest.raises(ValueError):
        DEMO.inspect_wav(broken)


def test_live_synthesis_checks_actual_http_media_type_before_voice(monkeypatch):
    class FakeHttp:
        status=200
        headers={'Content-Type':'audio/wav'}
        def __init__(self,data):self.data=data
        def __enter__(self):return self
        def __exit__(self,*args):return None
        def read(self,count):return self.data
    requests=[]
    class FakeOpener:
        def open(self,value,timeout):
            requests.append((value.full_url if hasattr(value,'full_url') else value,
                             value.method if hasattr(value,'method') else 'GET',timeout))
            if len(requests)==1:
                return FakeHttp(json.dumps({'ok':True,'service':'phonon_easton_voice',
                    'engine':'phonon_31m_easton_cpu','speaker':'Easton'}).encode())
            return FakeHttp(wav_bytes())
    monkeypatch.setattr(DEMO,'build_opener',lambda *a:FakeOpener())
    output,info=DEMO.synthesize('Verified source','http://127.0.0.1:15305/synthesize')
    assert info['pcm_bytes']==64000
    assert requests==[('http://127.0.0.1:15305/health','GET',5),
                      ('http://127.0.0.1:15305/synthesize','POST',65)]
    assert output==wav_bytes()


def test_no_action_preflight_makes_no_service_request_or_writes(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(DEMO,'check_voice_source',lambda p:{'verified_source':True})
    def unsafe(*args):raise AssertionError('Service/model contacted before explicit opt-in')
    monkeypatch.setattr(DEMO,'synthesize',unsafe)
    destination=tmp_path/'evidence'
    assert DEMO.main(['--reply-proof',str(proof(tmp_path)),'--voice-source',str(tmp_path),
                      '--evidence',str(destination)])==0
    assert not destination.exists()
    assert json.loads(capsys.readouterr().out)['no_live_model_called']


def test_run_requires_both_explicit_flag_and_environment(tmp_path,monkeypatch):
    monkeypatch.delenv('SWAAG_PHONON_VOICE_NOACTION',raising=False)
    monkeypatch.setattr(DEMO,'check_voice_source',lambda p:{'verified_source':True})
    destination=tmp_path/'evidence'
    with pytest.raises(SystemExit):
        DEMO.main(['--reply-proof',str(proof(tmp_path)),'--voice-source',str(tmp_path),
                   '--evidence',str(destination),'--run'])
    assert not destination.exists()


def test_changed_voice_source_revision_is_rejected_without_effects(tmp_path):
    with pytest.raises(ValueError,match='revision mismatch'):
        DEMO.check_voice_source(tmp_path)


def test_http_redirect_is_not_followed_to_any_provider():
    with pytest.raises(RuntimeError,match='redirect forbidden'):
        DEMO.RejectHttpRedirects().redirect_request(None,None,302,'redirect',{},
            'https://external.example.test/collect-speech')


def test_wrong_local_service_identity_fails_before_synthesis(monkeypatch):
    class FakeHealth:
        def __enter__(self):return self
        def __exit__(self,*args):return None
        def read(self,n):return json.dumps({'ok':True,'service':'other_service'}).encode()
    class FakeOpener:
        def open(self,target,timeout):
            assert isinstance(target,str),'A POST must not follow an unverified health response'
            return FakeHealth()
    monkeypatch.setattr(DEMO,'build_opener',lambda *a:FakeOpener())
    with pytest.raises(RuntimeError,match='identity was not verified'):
        DEMO.synthesize('No action','http://127.0.0.1:15305/synthesize')


def test_fresh_voice_clarification_gate_accepts_one_natural_question():
    good='Would you like me to identify where that passage comes from?'
    assert DEMO.validate_short_voice_clarification(good)==good


@pytest.mark.parametrize('candidate',[
    '', 'The passage was read.',
    'What would you like? What would you like?',
    'I have no tools or permission to act; what would you like me to do?',
    'As an AI assistant, what would you like to know?',
    'I cannot help with tools; what do you mean?',
    '```Here is the result.``` What now?',
    'Would you like me to identify this?\nPlease say yes.',
    'What do you want me to do with this particularly long passage now that I have understood all of the things you explained about it previously and have thought through all of the possible options, including historical and literary references and quotation sources?'])
def test_unacceptable_fresh_speech_answers_fail_closed(candidate):
    with pytest.raises(ValueError):
        DEMO.validate_short_voice_clarification(candidate)


def test_observed_first_fresh_model_reply_did_not_meet_quality_gate():
    observed=("I have no tools or permission to act, and I won't claim any action. "
      "The recognized speech isn't an explicit question, so here is one short clarification: "
      "what would you like me to do with this line? What would you like me to do with this line?")
    with pytest.raises(ValueError):
        DEMO.validate_short_voice_clarification(observed)


def test_fresh_same_turn_acceptance_uses_bounded_remote_checkout_and_session():
    path=ROOT/'scripts/demos/verify_real_voice_fresh_canonical_phonon.py'
    raw=path.read_text(encoding='utf8')
    assert "validate_pinned_swaag_paths(" in raw
    assert "--canonical-config" in raw and "--expected-config-sha256" in raw
    assert "--expected-source-commit" in raw and "--canonical-source" in raw
    assert "checked_remote(cmd)" in raw  # actual zero-tool inventory is inspected
    assert "scripts/demos/claim_fresh_voice_noaction_session.py" in raw
    assert "claim.get('admitted') is not True" in raw
    assert "prior_session_directories') != []" in raw
    assert raw.index("checked_remote(cmd)") < raw.index("claim=json.loads(checked_remote(claim_cmd))")
    assert raw.index("if args.evidence.exists():") < raw.index("claim=json.loads(checked_remote(claim_cmd))")
    assert 'reply=validate_short_voice_clarification(completed.stdout.strip())' in raw
    assert raw.index('reply=validate_short_voice_clarification(completed.stdout.strip())') < raw.index('canonical_source_only.append')
    assert 'input=prompt,text=True,capture_output=True' in raw


def test_pinned_remote_canonical_source_and_new_session_paths():
    ok=DEMO.validate_pinned_swaag_paths(
        '/data/src/worktrees/swaag-voice-phonon-integration-20261121',
        '/data/var/swaag-benchmarks/fresh-reviewed-voice-canonical-20261121-quality3/noaction-config.toml',
        '5'*40,'a'*64)
    assert ok[0].endswith('integration-20261121')
    assert 'quality3' in ok[1]


@pytest.mark.parametrize('bad_source,bad_config',[
    ('/tmp/untrusted',None),
    ('/data/src/worktrees/../../etc',None),
    ('/data/src/worktrees/id;curl',None),
    ('/data/src/worktrees/a$(touch /tmp/x)',None),
    (None,'/etc/voice-agent/ws.json'),
    (None,'/data/var/swaag-benchmarks/../other/noaction-config.toml'),
    (None,'/data/var/swaag-benchmarks/id;rm/noaction-config.toml'),
])
def test_untrusted_remote_paths_never_become_shell_commands(bad_source,bad_config):
    source=bad_source or '/data/src/worktrees/swaag-voice-phonon-integration-20261121'
    cfg=bad_config or '/data/var/swaag-benchmarks/fresh-reviewed-voice-canonical-20261121-quality3/noaction-config.toml'
    with pytest.raises(ValueError):
        DEMO.validate_pinned_swaag_paths(source,cfg,'5'*40,'a'*64)


@pytest.mark.parametrize('bad_revision,bad_hash',[
    ('not-commit',None),('x'*40,None),('5'*41,None),
    (None,'nonhex'),(None,'z'*64),(None,'a'*63)
])
def test_wrong_remote_git_or_config_fingerprints_are_rejected(bad_revision,bad_hash):
    with pytest.raises(ValueError):
        DEMO.validate_pinned_swaag_paths(
            '/data/src/worktrees/swaag-voice-phonon-integration-20261121',
            '/data/var/swaag-benchmarks/fresh-reviewed-voice-canonical-20261121-quality3/noaction-config.toml',
            bad_revision or '5'*40,bad_hash or 'a'*64)
