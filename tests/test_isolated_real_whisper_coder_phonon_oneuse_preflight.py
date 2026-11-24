"""No-network synthetic checks for the W73 source-pinned no-action runners.

Never invoke real models, TTS, a WebSocket, user recording or phone. Source
code and one-use admission are separate from actual private source receipts.
"""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))

def mod(name):
    s=importlib.util.spec_from_file_location('W73_'+name.replace('.py',''),ROOT/'scripts/demos'/name)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

W=mod('verify_isolated_voice_real_whisper_realtime_paced_once.py')
C=mod('run_isolated_real_whisper_text_to_resident_coder_once.py')
P=mod('run_isolated_coder_reply_to_phonon_easton_once.py')


def test_exact_as_executed_original_source_material_and_no_external_tools():
    assert W.INPUT_SHA=='03cedb42d9dabeab954793edea1b823f2bd9687df6af683a262cb09816c30d25'
    assert W.PATCHED_SERVER_SHA256=='873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf'
    assert W.MODEL_ORIGIN=='http://127.0.0.1:15300/inference'
    assert C.TRANSCRIPT_SHA=='9249355bc1af6b313f1bbff87bcb834efbcfe52ec396f847c6df6c550b5bad2a'
    assert C.MODEL_BASE=='http://10.8.0.7:14833'
    assert P.PHONON_SYNTH=='http://127.0.0.1:15305/synthesize'
    assert W.OPTIN!=C.OPTIN!=P.OPTIN
    assert P.CODER_RESULT_SHA=='05b1181c76c8c3e1986a3138b34fbc4ca415b282246be4656fdc89d117b5c54b'


def test_true_separate_one_use_limits_present_in_public_runner_sources():
    paths=['verify_isolated_voice_real_whisper_realtime_paced_once.py',
           'run_isolated_real_whisper_text_to_resident_coder_once.py',
           'run_isolated_coder_reply_to_phonon_easton_once.py']
    for name in paths:
        s=(ROOT/'scripts/demos'/name).read_text()
        assert 'one' in s[:300].lower() or 'once' in s[:300].lower()
        assert 'OPTIN=' in s
        assert 'FileExistsError' in s
        assert 'O_EXCL' in s or '_write_unique' in s
        assert '--run' in s
        assert 'systemctl restart' not in s
        assert 'os.system(' not in s
    src=(ROOT/'scripts/demos/run_isolated_real_whisper_text_to_resident_coder_once.py').read_text()
    assert "'actual_tool_calls':0" in src
    assert 'no_tool_schema' in src
    assert "MODEL_BASE+'/v1/chat/completions'" in src
    tts=(ROOT/'scripts/demos/run_isolated_coder_reply_to_phonon_easton_once.py').read_text()
    assert 'physical_owner_phone_audio_playback' in tts
    assert "'speaker')!='Easton'" in tts


def test_coder_existing_or_uncertain_oneuse_result_never_replayed(tmp_path,monkeypatch):
    path=tmp_path/'previous-model-effect';path.mkdir()
    (path/'attempt.json').write_text('unknown and not necessarily successful')
    monkeypatch.setattr(C,'check_orig',lambda *x:'Fake public utterance only.')
    monkeypatch.setattr(C,'fresh_lease_and_local_catalogue',lambda:None)
    monkeypatch.setenv(C.OPTIN,'1')
    files=[tmp_path/name for name in ('source','model','relay')]
    with pytest.raises(FileExistsError,match='uncertain original local model'):
        C.main(['--asr-history',str(files[0]),'--asr-result',str(files[1]),
                '--asr-relay',str(files[2]),'--private-output',str(path),'--run'])
    assert (path/'attempt.json').read_text()=='unknown and not necessarily successful'


def test_coder_denied_model_broker_no_attempt_marker_or_model_call(tmp_path,monkeypatch):
    monkeypatch.setattr(C,'check_orig',lambda *x:'Fake public utterance only.')
    monkeypatch.setattr(C,'fresh_lease_and_local_catalogue',lambda:(_ for _ in ()).throw(RuntimeError('broker denied')))
    monkeypatch.setenv(C.OPTIN,'1')
    f=tmp_path/'placeholder';out=tmp_path/'untouched'
    with pytest.raises(RuntimeError,match='broker denied'):
        C.main(['--asr-history',str(f),'--asr-result',str(f),
                '--asr-relay',str(f),'--private-output',str(out),'--run'])
    assert not out.exists()


def test_tts_no_effect_without_optin_and_source_check(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(P,'check',lambda *x:'Pure fake synthetic sample message for test only.')
    monkeypatch.delenv(P.OPTIN,raising=False)
    r=tmp_path/'a_dummy_private_model_result';out=tmp_path/'speech_not_generated'
    assert P.main(['--real-coder-result',str(r),'--private-output',str(out)])==0
    j=json.loads(capsys.readouterr().out)
    assert j['new_TTS_requests']==0 and j['not_an_actual_owner_voice_action']
    with pytest.raises(ValueError,match='opt-in required'):
        P.main(['--real-coder-result',str(r),'--private-output',str(out),'--run'])
    assert not out.exists()


def test_real_Whisper_WS_runner_never_bypasses_optin_on_safe_fake_input(tmp_path,monkeypatch,capsys):
    import asyncio
    fake_pcm=(b'\0\0'*16000)
    monkeypatch.setattr(W,'preflight',lambda *args:({},fake_pcm,1.0))
    monkeypatch.delenv(W.OPTIN,raising=False)
    root=tmp_path/'new_fake_result';f=tmp_path/'fake.wav'
    args=['--source',str(tmp_path),'--baseline',str(tmp_path),
      '--template',str(f),'--public-speech-wav',str(f),'--private-result',str(root)]
    assert W.main(args)==0
    assert json.loads(capsys.readouterr().out)['model_calls']==0
    with pytest.raises(ValueError,match='Explicit unique real Whisper one-use opt-in'):
        W.main(args+['--run'])
    assert not root.exists()
