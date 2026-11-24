"""Synthetic-only falsification of immutable genuine W73 ASR/LLM/TTS proof.

The original real resident Whisper, Coder and Phonon results are NOT rerun
by these tests. Each fixture is newly made fake JSON/PCM; it tests whether
the actual independent verifier detects forgery, replaced source, auto-send,
unsafe subscriber routes, incorrectly described failed trials, missing phone
review and mismatched model/TTS response. Never use a green unit test as
proof of real live microphone or provider/agent tool execution.
"""
from __future__ import annotations
import hashlib,importlib.util,io,json,math,struct,sys,wave
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('W73_independent_real_component_receipt_checker',
    ROOT/'scripts/demos/verify_isolated_real_asr_coder_phonon_chain_evidence.py')
V=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(V)


def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()

def audio_wav():
    samples=bytearray()
    for k in range(16000):samples.extend(struct.pack('<h',int(1800*math.sin(2*math.pi*240*k/16000))))
    out=io.BytesIO()
    with wave.open(out,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes(bytes(samples))
    return out.getvalue()


def fixture(tmp_path,monkeypatch):
    r=tmp_path/'fake-private-evidence';r.mkdir()
    src=tmp_path/'fake-unpublished-scripts';src.mkdir()
    sources={}
    for name in V.SCRIPT_SHA:
        file=src/name;file.write_text('# This is synthetic toy bytes only, never executed\n')
        sources[name]=V.sha(file)
    monkeypatch.setattr(V,'SCRIPT_SHA',sources)
    transcript='A fictional public audio reading that was heard.'
    reply='I acknowledge receipt of this fictional speech for manual review.'
    transhash=sha(transcript.encode());replyhash=sha(reply.encode())
    monkeypatch.setattr(V,'TRANSCRIPT_SHA',transhash)
    monkeypatch.setattr(V,'ANSWER_SHA',replyhash)
    jsons={
       'real_public_asr_local_Whisper_prior_SWAAG_draft.json':{'dialogue_history':[
           {'status':'draft','requires_review':True,'recognized_text':transcript,'submitted_text':'','assistant':''}]},
       'real_public_asr_prior_live_process_result.json':{
           'status':'real_local_Whisper_ASR_complete_and_two_reconnects_review_only','asr_text_sha256':transhash,
           'actual_local_Whisper_model_calls':1,'private_server_WS_connections':3,
           'actual_durable_history_turn_count':1,'manual_speaker_review_required':True,
           'actual_external_provider_SWAAG_tool_or_TTS_calls':0,
           'actual_Android_owner_phone_or_production_server_changed':False},
       'real_public_asr_prior_relay_evidence.json':{
           'requests_seen':1,'error_type':None,'actual_local_Whisper_text_SHA256':transhash,
           'original_HTTP_attempts':[{'real_Whisper_HTTP_status':200,
             'real_recorded_source_PCM_verified':True,
             'real_source_audio_high_energy_chunk_present':True,
             'same_real_ASR_delivered_after_client_reconnect':True,
             'real_Whisper_response_seconds':.25,'WAV_bytes_to_real_Whisper':30000}]},
       'prior_real_Whisper_failure_one_result.json':{'status':'failed_or_unknown',
            'error_type':'AssertionError','actual_Android_owner_phone_or_production_server_changed':False},
       'prior_real_Whisper_failure_two_result.json':{'status':'failed_or_unknown',
            'error_type':'AssertionError','actual_Android_owner_phone_or_production_server_changed':False},
    }
    events=[{'event':'server_start'}]+[{'event':'session_open'}]*3+[{'event':'session_replaced'}]*2
    events += [{'event':'asr_final'},{'event':'speaker_identity_unverified_manual_review'}]+[{'event':'session_close'}]*3
    (r/'copied_actual_guarded_ASR_private_ws_log.jsonl').write_text('\n'.join(json.dumps(q) for q in events)+'\n')
    (r/'copied_prior_public_voice_W73_derived.wav').write_bytes(b'unused synthetic public-audio input bytes')
    folder=r/'one_unique_actual_Qwen_local_noaction_spoken_confirmation';folder.mkdir()
    prompt={'model':'qwen3-coder-30b-a3b-instruct-q4-k-m', 'stream':False,
            'messages':[{'role':'system','content':'toy noaction'},
                        {'role':'user','content':'Recorded public speech sample: '+transcript}]}
    (folder/'private-prompt.json').write_text(json.dumps(prompt))
    (folder/'attempt.json').write_text(json.dumps({'one_new_unique_model_request':True,
        'no_tool_schema':True,'speech_SHA256':transhash,
        'fixed_prompt_sha256':sha(json.dumps(prompt,sort_keys=True).encode())}))
    model_result={'status':'real_resident_Coder_text_response_to_prior_actual_Whisper_noaction',
        'model':'qwen3-coder-30b-a3b-instruct-q4-k-m',
        'original_ASR_transcript_SHA256':transhash,'actual_new_model_request_count':1,
        'actual_tool_calls':0,'actual_live_SWAAG_agent_turn_or_external_action':False,
        'response_text':reply,'response_text_sha256':replyhash,
        'real_completed_model_response':True,'actual_model_reply_HTTP_status':200,
        'model_request_elapsed_seconds':2.2,'model_usage':{'completion_tokens':12}}
    (folder/'result.json').write_text(json.dumps(model_result))
    audio=audio_wav()
    (r/'copied_original_Phonon_noaction_audio.wav').write_bytes(audio)
    model_source_sha=V.sha(folder/'result.json')
    tts_result={'status':'real_Phonon_Easton_synthesized_prior_resident_Coder_response_noaction',
       'prior_Coder_result_sha256':model_source_sha,'prior_real_Whisper_transcript_sha256':transhash,
       'actual_noaction_Coder_response_sha256':replyhash,'real_phonon_tts_attempts':1,
       'TTS_speaker':'Easton','actual_TTS_engine':'phonon_31m_easton_cpu',
       'actual_Phonon_HTTP_status':200,'real_external_tool_provider_calls':0,
       'physical_owner_phone_audio_playback':False,'real_SWAAG_agent_turn_executed':False,
       'actual_audio_SHA256':sha(audio),'actual_audio_byte_size':len(audio),
       'actual_audio_seconds':1.0,'wall_seconds':1.4}
    (r/'copied_original_Phonon_noaction_result.json').write_text(json.dumps(tts_result))
    for n,v in jsons.items(): (r/n).write_text(json.dumps(v))
    originals={n:V.sha(r/n) for n in V.ORIGINAL}
    monkeypatch.setattr(V,'ORIGINAL',originals)
    return r,src


def test_public_original_SHA_and_actual_source_refs_not_fabricated():
    assert V.ORIGINAL['real_public_asr_prior_live_process_result.json']=='35d451e2e5b8d6246f62fdaa374fab404051aa50f3a111978cb953aa26bb5611'
    assert V.ORIGINAL['copied_original_Phonon_noaction_audio.wav']=='3bf61b25c27bccbd333148ba69ed86e686b626150a338924e3faea2f2cbb5e92'
    assert V.SCRIPT_SHA['verify_isolated_voice_real_whisper_realtime_paced_once.py']=='45215b1b3b7b5b47c7fd367301a08598c883fbf38945cf27e1c452ffa5150f05'
    assert V.SCRIPT_SHA['run_isolated_real_whisper_text_to_resident_coder_once.py']=='850344116f70a5232b8e4c3e58c18493df9a5a14702d2934053a805baae7e02b'
    assert V.SCRIPT_SHA['run_isolated_coder_reply_to_phonon_easton_once.py']=='23c3fc4bc8d30f8d4dea7ff449eeec82bef825e009ffcfc37bb3508067e90dde'


def test_synthetic_only_full_source_chain_has_no_model_invocations(tmp_path,monkeypatch):
    r,s=fixture(tmp_path,monkeypatch)
    result=V.loadroot(r,s)
    assert result['status']=='verified_real_LOCAL_separately_linked_Whisper_Coder_Phonon_and_reconnect_without_user_action'
    assert result['actual_guarded_W71_speech_reconnects']['websockets']==3
    assert result['actual_guarded_W71_speech_reconnects']['real_Whisper_requests']==1
    assert result['actual_resident_local_Phonon_speech_only']['real_synthesis_count']==1
    assert result['actual_user_phone_playback_or_live_SWAAG_agent_tools'] is False


def test_tampered_original_Whisper_HTTP_receipt_rejected(tmp_path,monkeypatch):
    r,s=fixture(tmp_path,monkeypatch)
    p=r/'real_public_asr_prior_relay_evidence.json'
    d=json.loads(p.read_text());d['original_HTTP_attempts'][0]['real_Whisper_HTTP_status']=503
    p.write_text(json.dumps(d));monkeypatch.setitem(V.ORIGINAL,p.name,V.sha(p))
    with pytest.raises(ValueError,match='source-verified local Whisper'):
        V.loadroot(r,s)


def test_attempted_tool_or_auto_committed_speech_rejected(tmp_path,monkeypatch):
    r,s=fixture(tmp_path,monkeypatch)
    p=r/'copied_actual_guarded_ASR_private_ws_log.jsonl'
    p.write_text(p.read_text()+json.dumps({'event':'turn_submitted'})+'\n')
    monkeypatch.setitem(V.ORIGINAL,p.name,V.sha(p))
    with pytest.raises(ValueError,match='Unexpected SWAAG tool'):
        V.loadroot(r,s)


def test_prior_historical_failed_trials_cannot_be_relabelled_success(tmp_path,monkeypatch):
    r,s=fixture(tmp_path,monkeypatch)
    p=r/'prior_real_Whisper_failure_one_result.json'
    a=json.loads(p.read_text());a['status']='real_local_Whisper_ASR_complete_and_two_reconnects_review_only'
    p.write_text(json.dumps(a));monkeypatch.setitem(V.ORIGINAL,p.name,V.sha(p))
    with pytest.raises(ValueError,match='failed initial audio admission'):
        V.loadroot(r,s)


def test_forged_conversational_owner_authority_fails(tmp_path,monkeypatch):
    r,s=fixture(tmp_path,monkeypatch)
    p=r/'real_public_asr_local_Whisper_prior_SWAAG_draft.json'
    a=json.loads(p.read_text());a['dialogue_history'][0]['requires_review']=False
    p.write_text(json.dumps(a));monkeypatch.setitem(V.ORIGINAL,p.name,V.sha(p))
    with pytest.raises(ValueError,match='manual review'):
        V.loadroot(r,s)


def test_model_response_must_be_grounded_in_real_original_transcript(tmp_path,monkeypatch):
    r,s=fixture(tmp_path,monkeypatch)
    p=r/'one_unique_actual_Qwen_local_noaction_spoken_confirmation/private-prompt.json'
    d=json.loads(p.read_text());d['messages'][-1]['content']='Recorded public speech sample: wrong utterance.';p.write_text(json.dumps(d))
    a=r/'one_unique_actual_Qwen_local_noaction_spoken_confirmation/attempt.json'
    j=json.loads(a.read_text());j['fixed_prompt_sha256']=sha(json.dumps(d,sort_keys=True).encode());a.write_text(json.dumps(j))
    with pytest.raises(ValueError,match='originally fixed real Whisper text'):
        V.loadroot(r,s)


def test_model_result_having_actual_tool_action_rejected(tmp_path,monkeypatch):
    r,s=fixture(tmp_path,monkeypatch)
    p=r/'one_unique_actual_Qwen_local_noaction_spoken_confirmation/result.json'
    d=json.loads(p.read_text());d['actual_tool_calls']=1;p.write_text(json.dumps(d))
    monkeypatch.setitem(V.ORIGINAL,'one_unique_actual_Qwen_local_noaction_spoken_confirmation/result.json',V.sha(p))
    t=r/'copied_original_Phonon_noaction_result.json'
    x=json.loads(t.read_text());x['prior_Coder_result_sha256']=V.sha(p);t.write_text(json.dumps(x))
    monkeypatch.setitem(V.ORIGINAL,t.name,V.sha(t))
    with pytest.raises(ValueError,match='source-backed single real local Qwen'):
        V.loadroot(r,s)


def test_corrupt_PCM_or_fake_silent_audio_fails_with_exact_result_sha(tmp_path,monkeypatch):
    r,s=fixture(tmp_path,monkeypatch)
    p=r/'copied_original_Phonon_noaction_audio.wav'
    p.write_bytes(b'corrupt WAV')
    monkeypatch.setitem(V.ORIGINAL,p.name,V.sha(p))
    with pytest.raises((ValueError,wave.Error)):
        V.loadroot(r,s)


def test_tts_audio_for_different_generated_text_rejected(tmp_path,monkeypatch):
    r,s=fixture(tmp_path,monkeypatch)
    p=r/'copied_original_Phonon_noaction_result.json'
    d=json.loads(p.read_text());d['actual_noaction_Coder_response_sha256']='a'*64
    p.write_text(json.dumps(d));monkeypatch.setitem(V.ORIGINAL,p.name,V.sha(p))
    with pytest.raises(ValueError,match='not the one real source-linked Coder answer'):
        V.loadroot(r,s)


def test_private_verdict_cannot_overwrite_prior_or_uncertain_result(tmp_path,monkeypatch):
    r,s=fixture(tmp_path,monkeypatch)
    out=tmp_path/'already-existing-verdict.json';out.write_text('exact prior receipt')
    with pytest.raises(FileExistsError,match='never overwrites'):
        V.main(['--private-root',str(r),'--source-dir',str(s),'--out',str(out)])
    assert out.read_text()=='exact prior receipt'
