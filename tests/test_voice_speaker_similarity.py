"""No-model, no-voice-data speaker similarity evidence and replay safety tests."""
from __future__ import annotations

from array import array
import hashlib
import math
from pathlib import Path
import wave
import pytest

from swaag.voice_speaker_similarity import (
    OfflineSpeakerSimilarity, SpeakerSimilarityUnavailable, SpeakerComparison,
    WESPEAKER_CAMPPLUS_LM_SHA256, WESPEAKER_CAMPPLUS_LM_SIZE,
    cosine_similarity, compare_voice_embeddings, sha256_regular_file,
)


def model_fixture(tmp_path):
    root=tmp_path/'private-audio';root.mkdir()
    file=tmp_path/'static-model.onnx';file.write_bytes(b'fake limited test source only')
    scorer=OfflineSpeakerSimilarity(
        model_path=file,audio_root=root,
        model_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        model_bytes=file.stat().st_size,cpu_threads=1)
    return scorer,root,file


def write_wav(path, *, sr=16000,channels=1,width=2,seconds=2.5,amplitude=900):
    sample_count=int(sr*seconds)
    signal=array('h',(int(amplitude*math.sin(2*math.pi*220*i/sr)) for i in range(sample_count)))
    with wave.open(str(path),'wb') as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(sr)
        f.writeframes(signal.tobytes())
    return path


def test_current_public_model_identification_is_immutable_and_sha_sized():
    assert WESPEAKER_CAMPPLUS_LM_SHA256=='1068e4ac3a76bb9c769e6816ef30bf89363f6e966f1d938210cb8ed4038f8e93'
    assert WESPEAKER_CAMPPLUS_LM_SIZE==29292449


def test_self_credential_audio_replay_may_match_but_never_authorize():
    evidence=compare_voice_embeddings([1,2,3],[1,2,3])
    assert evidence.cosine_similarity==pytest.approx(1.0)
    assert isinstance(evidence,SpeakerComparison)
    assert evidence.identity_claim_is_verified is False
    assert evidence.live_human_proven is False
    assert evidence.recording_replay_excluded is False
    assert evidence.can_auto_submit is False and evidence.can_execute_tools is False
    assert 'no_authority' in evidence.reason
    with pytest.raises(TypeError,match='MUST NOT'):
        if evidence:pass


@pytest.mark.parametrize('first,second,want',[
    ([1,0],[1,0],1.0),
    ([1,0],[-1,0],-1.0),
    ([1,0],[0,1],0.0),
    ([3,4],[3,4],1.0),
    ([3,4],[-4,3],0.0),
])
def test_correct_bounded_cosine_with_no_auth(first,second,want):
    assert cosine_similarity(first,second)==pytest.approx(want,abs=1e-12)
    evidence=compare_voice_embeddings(first,second)
    assert evidence.cosine_similarity==pytest.approx(want,abs=1e-12)
    assert not evidence.can_execute_tools


@pytest.mark.parametrize('a,b',[
    ([],[]),([1],[1,2]),([0,0],[0,0]),
    ([math.nan,1],[1,2]),([math.inf,1],[1,2]),
    ([1,2],[math.nan,2]),([1,2],['bad',2]),
    ([True,0],[0]),([1]*(8193),[1]*(8193)),
])
def test_corrupted_nonfinite_or_oversized_embedding_fails_closed(a,b):
    with pytest.raises(SpeakerSimilarityUnavailable):cosine_similarity(a,b)


def test_explicit_model_hash_and_size_success_without_dependencies(tmp_path):
    evaluator,root,model=model_fixture(tmp_path)
    assert evaluator._model_path==model
    assert evaluator._audio_root==root.resolve()
    assert evaluator._session is None
    assert sha256_regular_file(model,maximum_bytes=100000)==hashlib.sha256(model.read_bytes()).hexdigest()


def test_mutated_model_fails_before_numpy_or_torch_is_imported(tmp_path):
    _,root,model=model_fixture(tmp_path)
    with pytest.raises(SpeakerSimilarityUnavailable,match='expected digest'):
        OfflineSpeakerSimilarity(model_path=model,audio_root=root,model_bytes=model.stat().st_size,
            model_sha256='0'*64)
    with pytest.raises(SpeakerSimilarityUnavailable,match='expected digest'):
        OfflineSpeakerSimilarity(model_path=model,audio_root=root,model_bytes=model.stat().st_size+1,
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())


def test_symlink_model_and_audio_are_rejected(tmp_path):
    source,root,model=model_fixture(tmp_path)
    linked=tmp_path/'model-link.onnx';linked.symlink_to(model)
    with pytest.raises(SpeakerSimilarityUnavailable,match='absolute pinned'):
        OfflineSpeakerSimilarity(model_path=linked,audio_root=root,model_bytes=model.stat().st_size,
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    original=write_wav(root/'good.wav')
    link=root/'alias.wav';link.symlink_to(original)
    with pytest.raises(SpeakerSimilarityUnavailable,match='Invalid private audio'):
        source._validated_pcm16(link)


def test_audio_must_be_under_private_root(tmp_path):
    scorer,root,_=model_fixture(tmp_path)
    external=write_wav(tmp_path/'external.wav')
    with pytest.raises(SpeakerSimilarityUnavailable,match='escaped'):
        scorer._validated_pcm16(external)
    relative=Path('untrusted.wav')
    with pytest.raises(SpeakerSimilarityUnavailable,match='Invalid private'):
        scorer._validated_pcm16(relative)


def test_pcm16_mono_16k_duration_required_and_bytes_preserved(tmp_path):
    scorer,root,_=model_fixture(tmp_path)
    wav=write_wav(root/'valid.wav')
    original=wav.read_bytes()
    body,frames=scorer._validated_pcm16(wav)
    assert frames==40000 and len(body)==80000
    assert wav.read_bytes()==original


@pytest.mark.parametrize('sample_rate,channels,width',[(8000,1,2),(16000,2,2),(16000,1,1)])
def test_incorrect_sample_rate_channels_or_width_rejected(tmp_path,sample_rate,channels,width):
    scorer,root,_=model_fixture(tmp_path)
    wav=write_wav(root/'bad.wav',sr=sample_rate,channels=channels,width=width)
    with pytest.raises(SpeakerSimilarityUnavailable,match='Audio must be'):
        scorer._validated_pcm16(wav)


@pytest.mark.parametrize('duration', [0.4,1.9,15.1,23.0])
def test_short_and_unbounded_audio_rejected(tmp_path,duration):
    scorer,root,_=model_fixture(tmp_path)
    wav=write_wav(root/'duration.wav',seconds=duration)
    with pytest.raises(SpeakerSimilarityUnavailable):scorer._validated_pcm16(wav)


def test_corrupted_audio_never_enters_model(tmp_path):
    scorer,root,_=model_fixture(tmp_path)
    bad=root/'corrupt.wav';bad.write_bytes(b'NOT A WAVE\0\0\0')
    with pytest.raises(SpeakerSimilarityUnavailable):scorer._validated_pcm16(bad)


@pytest.mark.parametrize('cpu_threads',[0,-1,5,100])
def test_cpu_thread_overcommit_denied_before_model_load(tmp_path,cpu_threads):
    _,root,model=model_fixture(tmp_path)
    with pytest.raises(SpeakerSimilarityUnavailable,match='CPU threading'):
        OfflineSpeakerSimilarity(model_path=model,audio_root=root,cpu_threads=cpu_threads,
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),model_bytes=model.stat().st_size)


def test_relative_model_or_private_root_is_rejected(tmp_path):
    _,root,model=model_fixture(tmp_path)
    relative_model=Path('fake.onnx')
    with pytest.raises(SpeakerSimilarityUnavailable):
        OfflineSpeakerSimilarity(model_path=relative_model,audio_root=root,
            model_sha256='0'*64,model_bytes=30)
    with pytest.raises(SpeakerSimilarityUnavailable):
        OfflineSpeakerSimilarity(model_path=model,audio_root=Path('relative'),
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),model_bytes=model.stat().st_size)


def test_no_network_tool_authorization_or_disk_persistence_api():
    assert not hasattr(OfflineSpeakerSimilarity,'submit_turn')
    assert not hasattr(OfflineSpeakerSimilarity,'authorize_owner')
    assert not hasattr(OfflineSpeakerSimilarity,'persist_embedding')
    assert not hasattr(OfflineSpeakerSimilarity,'download_model')
    assert not hasattr(SpeakerComparison,'authorized')
    src=Path(__file__).parents[1]/'src/swaag/voice_speaker_similarity.py'
    code=src.read_text()
    for forbidden in ('urllib.request', 'requests.post(', 'subprocess.Popen(', 'shutil.copy(', 'os.replace('):
        assert forbidden not in code


def test_silent_audio_rejected_before_optional_model_imports(tmp_path, monkeypatch):
    scorer,root,_=model_fixture(tmp_path)
    p=root/'silent.wav'
    with wave.open(str(p),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
        f.writeframes(b'\x00\x00'*40000)
    monkeypatch.setattr(scorer,'_load_private_cpu_model',lambda:pytest.fail('Silence must not load model'))
    with pytest.raises(SpeakerSimilarityUnavailable,match='Silent'):
        scorer.embed_voice(p)


def test_model_modified_after_constructor_rejected_before_optional_imports(tmp_path):
    scorer,root,model=model_fixture(tmp_path)
    before=model.read_bytes()
    model.write_bytes(b'corrupt model bytes with altered size')
    with pytest.raises(SpeakerSimilarityUnavailable,match='Pinned model changed'):
        scorer._load_private_cpu_model()
    model.write_bytes(before)
    assert scorer._session is None
