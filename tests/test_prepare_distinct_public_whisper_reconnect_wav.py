"""Pure synthetic fixed-original audio derivation/no-replay preflight tests.

No real user recorded speech, local model, connection or phone runs here.
"""
from __future__ import annotations
import hashlib,importlib.util,io,os,wave
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('w73_public_wav',ROOT/'scripts/demos/prepare_distinct_public_whisper_reconnect_wav.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)

def fake(tmp_path,monkeypatch):
    root=tmp_path/'fake-public-input.wav'
    buf=io.BytesIO()
    with wave.open(buf,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes((b'\x90\x10'*int(4.815*16000)))
    root.write_bytes(buf.getvalue())
    monkeypatch.setattr(M,'ORIGINAL_SOURCE_SHA',hashlib.sha256(root.read_bytes()).hexdigest())
    b=io.BytesIO()
    with wave.open(b,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes((b'\x90\x10'*int(4.815*16000))+(b'\0\0'*9600))
    monkeypatch.setattr(M,'EXPECTED_NEW_AUDIO_SHA',hashlib.sha256(b.getvalue()).hexdigest())
    return root,b.getvalue()


def test_real_expected_public_recording_and_derived_hash_are_immutable_constants():
    assert M.ORIGINAL_SOURCE_SHA=='473b1bb573e8f7cdcd639c6e318036236190107b66bfe6892286121a92d464c1'
    assert M.EXPECTED_NEW_AUDIO_SHA=='03cedb42d9dabeab954793edea1b823f2bd9687df6af683a262cb09816c30d25'
    assert M.SILENCE_SAMPLES==9600 and M.DEST_BASENAME=='w73-new-third-public-recital-padded.wav'


def test_synthetic_unedited_source_only_reproduces_exact_16khz_audio_without_writes(tmp_path,monkeypatch,capsys):
    original,expect=fake(tmp_path,monkeypatch)
    out=tmp_path/M.DEST_BASENAME
    assert M.make(original)==expect
    assert M.main(['--public-wav',str(original),'--private-output',str(out)])==0
    assert 'DRYRUN_NO_WRITE' in capsys.readouterr().out
    assert not out.exists()


def test_synthetic_audio_tamper_or_symlink_denied(tmp_path,monkeypatch):
    original,expect=fake(tmp_path,monkeypatch)
    original.write_bytes(original.read_bytes()+b'X')
    with pytest.raises(ValueError,match='reviewed PUBLIC source'):
        M.make(original)
    original.unlink()
    source=tmp_path/'replacement';source.write_bytes(b'unsafe')
    original.symlink_to(source)
    with pytest.raises(ValueError,match='Missing/gone'):
        M.make(original)


def test_original_audio_never_overwritten_after_oneuse_attempt(tmp_path,monkeypatch):
    original,expect=fake(tmp_path,monkeypatch)
    out=tmp_path/M.DEST_BASENAME
    out.write_text('previous effect must never be repeated')
    with pytest.raises(FileExistsError,match='Never overwrite or replay'):
        M.main(['--public-wav',str(original),'--private-output',str(out),'--run'])
    assert out.read_text()=='previous effect must never be repeated'
