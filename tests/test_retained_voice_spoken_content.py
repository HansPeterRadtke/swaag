"""Offline regression for exact retained real audio and no-action admission."""
from __future__ import annotations
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import wave

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/demos/verify_retained_voice_spoken_content.py'
SPEC = importlib.util.spec_from_file_location('retained_voice_content', SOURCE)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
REPLY='What would you like to know about the passage?'


def audio_bytes(rate=16000,channels=1,sample_width=2,frames=32000,payload=42):
    b=io.BytesIO()
    with wave.open(b,'wb') as wav:
        wav.setframerate(rate)
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.writeframes((payload.to_bytes(sample_width,'little',signed=True)
                         *channels)*frames)
    return b.getvalue()


def inputs(root: Path, wav=None, **changes):
    content=audio_bytes() if wav is None else wav
    r=root/'receipt.json';a=root/'spoken.wav'
    a.write_bytes(content)
    d={'status':'real_fast_swaag_direct_reply_phonon_reconnect_noaction',
       'canonical_answer_not_fresh':False,'fast_foreground_direct_answer':True,
       'fresh_canonical_response_calls':1,'model_action_calls':0,
       'played_pcm_recovery_exact':True,'normal_ack':True,
       'no_replay_after_ack':True,
       'tts_wav_sha256':VERIFIER.sha(content),
       'preserved_generated_audio':{'retained_wav_sha256':VERIFIER.sha(content),
                                   'retained_wav_bytes':len(content)},
       'canonical_fresh_reply':{'reply':REPLY,'stdout_sha256':VERIFIER.sha(REPLY.encode())}}
    d.update(changes);r.write_text(json.dumps(d))
    return r,a


def args(receipt,audio,evidence,run=False):
    cmd=['--receipt',str(receipt),'--audio',str(audio),'--evidence',str(evidence)]
    if run:cmd.append('--run')
    return cmd


def test_preflight_exercises_real_receipt_and_wav_without_network_or_evidence(tmp_path,monkeypatch,capsys):
    receipt,wav=inputs(tmp_path)
    evidence=tmp_path/'unique'
    monkeypatch.setattr(VERIFIER,'actual_resident_asr',lambda _:pytest.fail('No network in preflight'))
    assert VERIFIER.main(args(receipt,wav,evidence))==0
    assert not evidence.exists()
    observed=json.loads(capsys.readouterr().out)
    assert observed['status']=='preflight_only_no_service_contact'
    assert observed['expected_words']==9
    assert observed['audio_sha256']==VERIFIER.sha(wav.read_bytes())


@pytest.mark.parametrize('change',[{'normal_ack':False},{'fresh_canonical_response_calls':2},
                                   {'model_action_calls':1},{'no_replay_after_ack':False},
                                   {'canonical_answer_not_fresh':True}])
def test_unverified_original_receipt_is_rejected(tmp_path,change):
    p,w=inputs(tmp_path,**change)
    with pytest.raises(ValueError,match='original no-action'):
        VERIFIER.inspect_completed_turn(p,w)


def test_manipulated_answer_and_audio_are_rejected(tmp_path):
    r,a=inputs(tmp_path)
    a.write_bytes(audio_bytes(payload=19))
    with pytest.raises(ValueError,match='Retained WAV differs'):
        VERIFIER.inspect_completed_turn(r,a)
    r,a=inputs(tmp_path)
    d=json.loads(r.read_text());d['canonical_fresh_reply']['reply']='Different words'
    r.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='original no-action'):
        VERIFIER.inspect_completed_turn(r,a)


@pytest.mark.parametrize('wav',[audio_bytes(rate=22050),audio_bytes(channels=2),
                                  audio_bytes(payload=0),audio_bytes(frames=160),
                                  b'not-wav'])
def test_invalid_silent_or_wrong_format_reply_cannot_pass(tmp_path,wav):
    r,a=inputs(tmp_path,wav)
    with pytest.raises(ValueError,match='audio|WAV'):
        VERIFIER.inspect_completed_turn(r,a)


def test_actual_run_requires_explicit_optin_and_no_existing_output(tmp_path,monkeypatch,capsys):
    r,a=inputs(tmp_path);destination=tmp_path/'new'
    monkeypatch.delenv(VERIFIER.OPT_IN,raising=False)
    with pytest.raises(ValueError,match='explicit environment'):
        VERIFIER.main(args(r,a,destination,True))
    assert not destination.exists()
    monkeypatch.setenv(VERIFIER.OPT_IN,'1')
    calls=[]
    def fake_actual(wav):
        calls.append(VERIFIER.sha(wav))
        return REPLY
    monkeypatch.setattr(VERIFIER,'actual_resident_asr',fake_actual)
    assert VERIFIER.main(args(r,a,destination,True))==0
    record=json.loads((destination/'result.json').read_text())
    assert record['status']=='verified' and record['word_error_rate']==0
    assert record['actual_asr_requests_attempted']==1
    assert len(calls)==1
    assert (destination/'attempt.json').exists()
    with pytest.raises(FileExistsError,match='uncertain replay'):
        VERIFIER.main(args(r,a,destination,True))
    assert len(calls)==1


def test_real_language_disagreement_is_preserved_as_a_failure(tmp_path,monkeypatch):
    r,a=inputs(tmp_path);out=tmp_path/'mismatch'
    monkeypatch.setenv(VERIFIER.OPT_IN,'1')
    monkeypatch.setattr(VERIFIER,'actual_resident_asr',lambda wav:'What is the time?')
    assert VERIFIER.main(args(r,a,out,True))==2
    result=json.loads((out/'result.json').read_text())
    assert result['status']=='content_mismatch'
    assert result['word_error_rate']>0
    assert result['word_edit_distance']>0


def test_asr_fault_persists_private_failure_without_retry(tmp_path,monkeypatch):
    r,a=inputs(tmp_path);out=tmp_path/'network-failure'
    monkeypatch.setenv(VERIFIER.OPT_IN,'1')
    counter=[]
    def fail(wav):
        counter.append(1)
        raise TimeoutError('local ASR was busy')
    monkeypatch.setattr(VERIFIER,'actual_resident_asr',fail)
    assert VERIFIER.main(args(r,a,out,True))==2
    d=json.loads((out/'result.json').read_text())
    assert d['status']=='asr_error' and d['error_type']=='TimeoutError'
    with pytest.raises(FileExistsError):VERIFIER.main(args(r,a,out,True))
    assert len(counter)==1


def test_normalized_word_edit_distance_is_exact_and_deterministic():
    assert VERIFIER.word_edits(VERIFIER.words('Hello, there.'),VERIFIER.words('hello there'))==0
    assert VERIFIER.word_edits(['a','b','c'],['a','c'])==1
    assert VERIFIER.word_edits(['a','b'],['b','a'])==2
    assert VERIFIER.word_edits([],['a','b'])==2


def test_whisper_http_rejects_redirects_and_unbounded_results(monkeypatch):
    with pytest.raises(ValueError,match='No redirects'):
        VERIFIER.RejectRedirect().redirect_request(None,None,302,'moved',{},'https://example.org')
    class Response:
        status=200
        def __init__(self,body):self.body=body
        def __enter__(self):return self
        def __exit__(self,*args):return None
        def read(self,limit):return self.body
    class FakeOpener:
        def __init__(self,body):self.body=body
        def open(self,request,timeout):
            assert request.full_url==VERIFIER.ASR_URL
            assert request.get_method()=='POST' and timeout==35
            return Response(self.body)
    monkeypatch.setattr(VERIFIER,'build_opener',lambda *a:FakeOpener(b'{"ok":false,"text":"bad"}'))
    with pytest.raises(ValueError):VERIFIER.actual_resident_asr(audio_bytes())
    monkeypatch.setattr(VERIFIER,'build_opener',lambda *a:FakeOpener(b'{"ok":true,"text":" Hello! "}'))
    assert VERIFIER.actual_resident_asr(audio_bytes())=='Hello!'
