"""Fast live voice source boundary and one-use admission, without any model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest

HERE=Path(__file__).resolve().parents[1]/'scripts/demos'
sys.path.insert(0,str(HERE))
SPEC=importlib.util.spec_from_file_location('one_fast_noaction_once',HERE/'run_fast_noaction_voice_once.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'fast-private';root.mkdir()
    sessions=root/'sessions';sessions.mkdir()
    agent=root/'agent_data';agent.mkdir()
    config=root/'noaction-config.toml'
    config.write_text('''[model]
base_url="http://10.8.0.7:14835"
profile_name="qwen38-27b-q8-full-262k"
cache_enabled=false
max_retries=0
chat_template_kwargs={enable_thinking=false}
[tools]
enabled=[]
staged_discovery=false
allow_stateful_tools=false
allow_side_effect_tools=false
[communication]
fast_initial_reply=true
[sessions]
root="'''+str(sessions)+'''"
[agent_data]
root="'''+str(agent)+'''"
[logging]
file_path="'''+str(root/'operations.jsonl')+'''"
''')
    sha=hashlib.sha256(config.read_bytes()).hexdigest()
    claim={'admitted':True,'one_use_only':True,'prior_session_directories':[],
           'configuration_sha256':sha,'private_session_root':str(sessions)}
    (sessions/M.CLAIM_NAME).write_text(json.dumps(claim))
    source=tmp_path/'source';source.mkdir()
    monkeypatch.setattr(M,'validate_pinned_swaag_paths',lambda *args:(str(source),str(config)))
    monkeypatch.setattr(M.subprocess,'check_output',lambda *args,**kwargs:'f'*40+'\n')
    return source,config,sha,sessions


def test_exact_tool_free_source_and_claim_are_admitted(tmp_path,monkeypatch):
    source,config,sha,sessions=fixture(tmp_path,monkeypatch)
    c,claim=M.check_admitted_source(config,sha,'f'*40,source)
    assert c.tools.enabled==[] and not c.tools.allow_side_effect_tools
    assert c.communication.fast_initial_reply
    assert claim==sessions/M.CLAIM_NAME


def test_repeated_same_fast_inference_is_refused(tmp_path,monkeypatch):
    _,config,sha,sessions=fixture(tmp_path,monkeypatch)
    marker=M.claim_one_inference(M.load_config([config],env={}), 'a'*64)
    assert marker.stat().st_mode & 0o777 == 0o600
    assert json.loads(marker.read_text())['one_model_request_maximum']
    with pytest.raises(FileExistsError):
        M.claim_one_inference(M.load_config([config],env={}), 'a'*64)


def test_existing_session_refused_even_with_good_durable_claim(tmp_path,monkeypatch):
    source,config,sha,sessions=fixture(tmp_path,monkeypatch)
    (sessions/'session_existing').mkdir()
    with pytest.raises(ValueError,match='reused'):
        M.check_admitted_source(config,sha,'f'*40,source)


def test_wrong_config_bytes_refused(tmp_path,monkeypatch):
    source,config,sha,_=fixture(tmp_path,monkeypatch)
    config.write_text(config.read_text()+'\n# changed after approval')
    with pytest.raises(ValueError,match='does not match'):
        M.check_admitted_source(config,sha,'f'*40,source)


def test_wrong_git_source_refused(tmp_path,monkeypatch):
    source,config,sha,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Git source changed'):
        M.check_admitted_source(config,sha,'e'*40,source)


@pytest.mark.parametrize('section,bad_value',[
    ('tools__enabled','["calculator"]'),('tools__staged_discovery','true'),
    ('tools__allow_stateful_tools','true'),('tools__allow_side_effect_tools','true'),
    ('communication__fast_initial_reply','false'),('model__cache_enabled','true'),
    ('model__max_retries','1'),('model__base_url','"http://attacker.test:14835"')
])
def test_wrong_model_or_tool_policy_refused(tmp_path,monkeypatch,section,bad_value):
    source,config,_,sessions=fixture(tmp_path,monkeypatch)
    name,key=section.split('__')
    original=config.read_text()
    expression={'enabled':'[]','staged_discovery':'false','allow_stateful_tools':'false',
                'allow_side_effect_tools':'false','fast_initial_reply':'true',
                'cache_enabled':'false','max_retries':'0','base_url':'"http://10.8.0.7:14835"'}[key]
    old=key+'='+expression
    assert old in original
    config.write_text(original.replace(old,key+'='+bad_value,1))
    sha=hashlib.sha256(config.read_bytes()).hexdigest()
    claim=json.loads((sessions/M.CLAIM_NAME).read_text());claim['configuration_sha256']=sha
    (sessions/M.CLAIM_NAME).write_text(json.dumps(claim))
    with pytest.raises(ValueError,match='Unapproved'):
        M.check_admitted_source(config,sha,'f'*40,source)


def test_no_prior_oneuse_claim_is_not_authorization(tmp_path,monkeypatch):
    source,config,sha,sessions=fixture(tmp_path,monkeypatch)
    (sessions/M.CLAIM_NAME).unlink()
    with pytest.raises(ValueError,match='prior single-use'):
        M.check_admitted_source(config,sha,'f'*40,source)


def test_altered_claim_is_not_authorization(tmp_path,monkeypatch):
    source,config,sha,sessions=fixture(tmp_path,monkeypatch)
    p=sessions/M.CLAIM_NAME;d=json.loads(p.read_text());d['admitted']=False;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='not independently claimed'):
        M.check_admitted_source(config,sha,'f'*40,source)


def test_fast_voice_script_uses_exact_asr_stdin_and_never_calls_full_controller():
    p=HERE/'verify_real_voice_fast_foreground_reconnect.py'
    text=p.read_text()
    assert 'RECORDED_ASR_SHA256' in text
    assert "--transcript-sha256 '+RECORDED_ASR_SHA256" in text
    assert 'input=source_text,text=True,capture_output=True' in text
    assert "output.get('full_orchestrator_called') is not False" in text
    assert "output.get('model_requests')!=1" in text
    assert "output.get('fast_decision_mode')!='answer'" in text
    assert "STREAMED_PHONON_NOACTION_VERIFIED" in text
    assert 'real_fast_swaag_direct_reply_phonon_reconnect_noaction' in text


def test_fast_nitro_only_emits_json_after_verifying_full_event_receipts():
    p=HERE/'run_fast_noaction_voice_once.py'
    text=p.read_text()
    assert "counts['tool_called']!=0" in text
    assert "counts['turn_finished']!=1" in text
    assert "validate_short_voice_clarification(str(decision['reply']))" in text
    assert "root/'.fast_oneuse" not in text  # marker rooted only in session namespace
    assert "claim_one_inference" in text
    assert 'journal_sha256' in text


def test_preserve_generated_wav_is_exact_private_and_nonreplayable(tmp_path):
    import wave
    from fast_voice_audio_evidence import preserve_new_generated_audio
    source=tmp_path/'temporary_stream.wav';target=tmp_path/'preserved.wav'
    with wave.open(str(source),'wb') as wf:
        wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(16000)
        wf.writeframes((b'\x10\x00')*32000)
    data=source.read_bytes()
    result=preserve_new_generated_audio(source,target)
    assert result['retained_wav_sha256']==hashlib.sha256(data).hexdigest()
    assert result['retained_pcm_bytes']==64000
    assert target.read_bytes()==data
    assert target.stat().st_mode&0o777==0o600
    with pytest.raises(FileExistsError):
        preserve_new_generated_audio(source,target)
    assert target.read_bytes()==data


@pytest.mark.parametrize('bad',[
    b'',b'not_wav',b'RIFF'+b'\0'*45,
])
def test_preserve_generated_wav_rejects_invalid_pcm_before_copy(tmp_path,bad):
    from fast_voice_audio_evidence import preserve_new_generated_audio
    src=tmp_path/'bad.wav';src.write_bytes(bad)
    dst=tmp_path/'preserved.wav'
    with pytest.raises((ValueError,EOFError)):
        preserve_new_generated_audio(src,dst)
    assert not dst.exists()


def test_fast_voice_acceptance_imports_real_wav_preserver():
    source=(HERE/'verify_real_voice_fast_foreground_reconnect.py').read_text()
    assert 'from fast_voice_audio_evidence import preserve_new_generated_audio' in source
    assert "OUTPUT.parent/'fast-generated-reply.wav'" in source
    assert "'preserved_generated_audio':retained" in source
