"""Strict offline admission for real retained-voice/replayed-owner negative test."""
from __future__ import annotations
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import wave
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('voice_replay_gate',ROOT/'scripts/demos/verify_voice_tts_replay_noaction.py')
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def make_fixture(tmp_path,monkeypatch):
    candidate=tmp_path/'candidate';candidate.mkdir()
    helper=tmp_path/'helper';helper.mkdir()
    sources={'cloudflare_ws_voice_server.py':b'candidate test server\n',
             'voice_ws_config.py':b'candidate test config\n'}
    for name,content in sources.items(): (candidate/name).write_bytes(content)
    fixture=helper/'test_cloudflare_ws_voice_server.py';fixture.write_bytes(b'fake fixture source')
    monkeypatch.setattr(M,'EXPECTED_SOURCE_HASHES',{k:M.digest(v) for k,v in sources.items()})
    monkeypatch.setattr(M,'EXPECTED_FIXTURE_SHA',M.digest(fixture.read_bytes()))
    sound=io.BytesIO()
    with wave.open(sound,'wb') as w:
        w.setframerate(16000);w.setnchannels(1);w.setsampwidth(2)
        w.writeframes(b'\x21\x00'*32000)
    audio=tmp_path/'original.wav';audio.write_bytes(sound.getvalue())
    monkeypatch.setattr(M,'EXPECTED_AUDIO_SHA',M.digest(audio.read_bytes()))
    original={'status':'real_fast_swaag_direct_reply_phonon_reconnect_noaction',
       'canonical_answer_not_fresh':False,'fresh_canonical_response_calls':1,
       'model_action_calls':0,'played_pcm_recovery_exact':True,
       'normal_ack':True,'no_replay_after_ack':True,
       'tts_wav_sha256':M.EXPECTED_AUDIO_SHA,
       'preserved_generated_audio':{'retained_wav_sha256':M.EXPECTED_AUDIO_SHA},
       'canonical_fresh_reply':{'reply':'Please tell me which section you would like to review.'}}
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps(original))
    monkeypatch.setattr(M,'EXPECTED_RECEIPT_SHA',M.digest(receipt.read_bytes()))
    return candidate,helper,receipt,audio


def cli(files,dest,run=False):
    args=[]
    for flag,path in zip(('--candidate','--helpers','--receipt','--audio'),files):
        args.extend((flag,str(path)))
    args.extend(('--evidence',str(dest)))
    if run:args.append('--run')
    return args


def test_source_pinned_genuine_wav_receipt_check_is_pure(tmp_path,monkeypatch,capsys):
    files=make_fixture(tmp_path,monkeypatch)
    pcm,text,_=M.validate_inputs(*files)
    assert len(pcm)==64000 and len(M.tokens(text))>=5
    async def fail_if_called(*a):pytest.fail('No ASR/WS in preflight')
    monkeypatch.setattr(M,'actual_isolated_replay',fail_if_called)
    dest=tmp_path/'none'
    assert M.main(cli(files,dest))==0
    assert not dest.exists()
    assert json.loads(capsys.readouterr().out)['status']=='source_only_no_network_or_write'


@pytest.mark.parametrize('index',range(4))
def test_every_modified_source_or_original_input_is_rejected(tmp_path,monkeypatch,index):
    paths=make_fixture(tmp_path,monkeypatch)
    target=(paths[0]/'cloudflare_ws_voice_server.py',paths[1]/'test_cloudflare_ws_voice_server.py',
            paths[2],paths[3])[index]
    target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(ValueError):M.validate_inputs(*paths)


@pytest.mark.parametrize('change',[{'canonical_answer_not_fresh':True},
     {'model_action_calls':1},{'fresh_canonical_response_calls':2},
     {'normal_ack':False},{'no_replay_after_ack':False},
     {'played_pcm_recovery_exact':False}])
def test_prior_completed_noaction_receipt_is_required(tmp_path,monkeypatch,change):
    paths=make_fixture(tmp_path,monkeypatch)
    d=json.loads(paths[2].read_text());d.update(change)
    paths[2].write_text(json.dumps(d))
    monkeypatch.setattr(M,'EXPECTED_RECEIPT_SHA',M.digest(paths[2].read_bytes()))
    with pytest.raises(ValueError):M.validate_inputs(*paths)


def test_suspicious_symlink_and_empty_audio_fail_closed(tmp_path,monkeypatch):
    paths=make_fixture(tmp_path,monkeypatch)
    paths[3].unlink()
    paths[3].symlink_to(tmp_path/'external')
    with pytest.raises(ValueError,match='symlinked'):M.validate_inputs(*paths)
    paths[3].unlink()
    paths[3].write_bytes(b'')
    monkeypatch.setattr(M,'EXPECTED_AUDIO_SHA',M.digest(b''))
    with pytest.raises(ValueError):M.validate_inputs(*paths)


def test_real_test_requires_double_optin_and_exclusive_destination(tmp_path,monkeypatch,capsys):
    files=make_fixture(tmp_path,monkeypatch)
    dest=tmp_path/'exclusive'
    monkeypatch.delenv(M.OPT_IN,raising=False)
    with pytest.raises(ValueError,match='environment opt-in'):M.main(cli(files,dest,True))
    assert not dest.exists()
    monkeypatch.setenv(M.OPT_IN,'1')
    seen=[]
    async def fake_isolated(candidate,helper,pcm,reply):
        seen.append(M.digest(pcm))
        return {'status':'verified_replayed_assistant_voice_requires_review',
                'source_ASR_words':len(M.tokens(reply)),
                'automatic_turn_submissions':0,
                'same_reviewable_draft_after_reconnect':True}
    monkeypatch.setattr(M,'actual_isolated_replay',fake_isolated)
    assert M.main(cli(files,dest,True))==0
    result=json.loads((dest/'result.json').read_text())
    assert result['automatic_turn_submissions']==0
    assert result['status']=='verified_replayed_assistant_voice_requires_review'
    assert len(seen)==1 and (dest/'attempt.json').is_file()
    assert stat.S_IMODE((dest/'attempt.json').stat().st_mode)==0o600
    with pytest.raises(FileExistsError):M.main(cli(files,dest,True))
    assert len(seen)==1


def test_failed_live_gate_preserves_unique_failure_without_retry(tmp_path,monkeypatch):
    files=make_fixture(tmp_path,monkeypatch);dest=tmp_path/'failed'
    monkeypatch.setenv(M.OPT_IN,'1')
    calls=[]
    async def fake_fail(*args):
        calls.append(1)
        raise TimeoutError('ASR unavailable')
    monkeypatch.setattr(M,'actual_isolated_replay',fake_fail)
    assert M.main(cli(files,dest,True))==2
    d=json.loads((dest/'result.json').read_text())
    assert d['status']=='failed_or_uncertain' and d['error_type']=='TimeoutError'
    with pytest.raises(FileExistsError):M.main(cli(files,dest,True))
    assert calls==[1]


def test_no_untrusted_external_forwarding_in_staged_source():
    s=(ROOT/'scripts/demos/verify_voice_tts_replay_noaction.py').read_text()
    assert "serve(server.handler,'127.0.0.1',0" in s
    assert 'real_asr(*args,**kwargs)' in s
    assert "patch.object(server.VoiceSession,'submit_turn',forbidden_action)" in s
    assert "patch.object(server,'receive_voice_run',forbidden_action)" in s
    assert "patch.object(server,'cancel_voice_run',forbidden_action)" in s
    assert 'speaker_identity_unverified' in s
