#!/usr/bin/env python3
"""Independently validate real local Whisper -> Coder -> Phonon component proof.

Zero model, TTS, phone, WebSocket, user Send, or agent tool execution. Read the
ORIGINAL completed, one-use private ASR/LLM/TTS traces copied byte-for-byte
from Jetson/Nitro. Fail closed on forged text, incomplete speech, duplicate
provider calls, missing session reconnects, edited source model metadata,
invalid audio, permission bypass, or any unexpected completed SWAAG action.
This validates three separately coupled real LOCAL components; it does NOT
pretend the phone physically played audio or the live service was upgraded.
"""
from __future__ import annotations
import argparse,audioop,collections,datetime,hashlib,json,math,os,wave
from pathlib import Path

ORIGINAL={
 'real_public_asr_local_Whisper_prior_SWAAG_draft.json':'48e0d9b2ff665cded82e4a40c376d9cce112bdbc0d1882c50b066e4c2da24cd7',
 'real_public_asr_prior_live_process_result.json':'35d451e2e5b8d6246f62fdaa374fab404051aa50f3a111978cb953aa26bb5611',
 'real_public_asr_prior_relay_evidence.json':'6e856e69d1b5c393f71bdbed7311c31915beb9bb680bf97642bdbe235d01db6f',
 'copied_prior_public_voice_W73_derived.wav':'03cedb42d9dabeab954793edea1b823f2bd9687df6af683a262cb09816c30d25',
 'copied_actual_guarded_ASR_private_ws_log.jsonl':'166bb4fd25bfa63d79260062a0699fcb3a4ead0ee5edead53e119835de063de4',
 'one_unique_actual_Qwen_local_noaction_spoken_confirmation/result.json':'05b1181c76c8c3e1986a3138b34fbc4ca415b282246be4656fdc89d117b5c54b',
 'copied_original_Phonon_noaction_result.json':'64b55f42b9536f9acf47197cd42d3dd15571fa3af28772a0f5f7016af936365b',
 'copied_original_Phonon_noaction_audio.wav':'3bf61b25c27bccbd333148ba69ed86e686b626150a338924e3faea2f2cbb5e92',
 'prior_real_Whisper_failure_one_result.json':'2a56610dafdb560c36578ab71f6f7190b565d30cef0f037dd96c482051079c81',
 'prior_real_Whisper_failure_two_result.json':'b278a88133770127ef17f57dab7f4d6b351e4365b55ecf0eef3dc1870b7a74a9',
}
SCRIPT_SHA={
 'verify_isolated_voice_real_whisper_realtime_paced_once.py':'45215b1b3b7b5b47c7fd367301a08598c883fbf38945cf27e1c452ffa5150f05',
 'run_isolated_real_whisper_text_to_resident_coder_once.py':'850344116f70a5232b8e4c3e58c18493df9a5a14702d2934053a805baae7e02b',
 'run_isolated_coder_reply_to_phonon_easton_once.py':'23c3fc4bc8d30f8d4dea7ff449eeec82bef825e009ffcfc37bb3508067e90dde',
}
TRANSCRIPT_SHA='9249355bc1af6b313f1bbff87bcb834efbcfe52ec396f847c6df6c550b5bad2a'
ANSWER_SHA='35f5a465cfba2db4378b12e4f0049313a8946bbab1024e0fe532c022d1c708af'
REAL_ASR_MODEL='http://127.0.0.1:15300/inference'
REAL_TTS_ENGINE='phonon_31m_easton_cpu'


def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def must_regular(p:Path):
    if not p.is_file() or p.is_symlink():raise ValueError('Required one-use source missing or symlinked: '+str(p))

def loadroot(root:Path,scripts:Path)->dict:
    if not root.is_dir() or root.is_symlink() or not scripts.is_dir() or scripts.is_symlink():
        raise ValueError('Wrong original root or public source checkout')
    for name,want in ORIGINAL.items():
        path=root/name;must_regular(path)
        if sha(path)!=want:raise ValueError('Original source evidence altered: '+name)
    for name,want in SCRIPT_SHA.items():
        p=scripts/name;must_regular(p)
        if sha(p)!=want:raise ValueError('As-executed public no-action script changed: '+name)
    reader=lambda n:json.loads((root/n).read_text())
    history=reader('real_public_asr_local_Whisper_prior_SWAAG_draft.json')['dialogue_history']
    ASR=reader('real_public_asr_prior_live_process_result.json')
    relay=reader('real_public_asr_prior_relay_evidence.json')
    model=reader('one_unique_actual_Qwen_local_noaction_spoken_confirmation/result.json')
    tts=reader('copied_original_Phonon_noaction_result.json')
    if len(history)!=1 or history[0].get('status')!='draft' or history[0].get('requires_review') is not True or history[0].get('submitted_text') or history[0].get('assistant'):
        raise ValueError('Genuine local speech not retained for manual review')
    words=' '.join(str(history[0].get('recognized_text') or '').split())
    if not 12<=len(words)<=2000 or hashlib.sha256(words.encode()).hexdigest()!=TRANSCRIPT_SHA:
        raise ValueError('ASR words changed or missing from durable source')
    if (ASR.get('status')!='real_local_Whisper_ASR_complete_and_two_reconnects_review_only'
        or ASR.get('asr_text_sha256')!=TRANSCRIPT_SHA
        or ASR.get('actual_local_Whisper_model_calls')!=1
        or ASR.get('private_server_WS_connections')!=3
        or ASR.get('actual_durable_history_turn_count')!=1
        or ASR.get('manual_speaker_review_required') is not True
        or ASR.get('actual_external_provider_SWAAG_tool_or_TTS_calls')!=0
        or ASR.get('actual_Android_owner_phone_or_production_server_changed') is not False):
        raise ValueError('Actual model-backed private socket reconnect did not succeed safely')
    actual=relay.get('original_HTTP_attempts')
    if (relay.get('requests_seen')!=1 or relay.get('error_type') is not None
        or relay.get('actual_local_Whisper_text_SHA256')!=TRANSCRIPT_SHA
        or not isinstance(actual,list) or len(actual)!=1
        or actual[0].get('real_Whisper_HTTP_status')!=200
        or actual[0].get('real_recorded_source_PCM_verified') is not True
        or actual[0].get('real_source_audio_high_energy_chunk_present') is not True
        or actual[0].get('same_real_ASR_delivered_after_client_reconnect') is not True
        or not 0<actual[0].get('real_Whisper_response_seconds',0)<20
        or not 12000<actual[0].get('WAV_bytes_to_real_Whisper',0)<250000):
        raise ValueError('No single real successful source-verified local Whisper inference')
    log=root/'copied_actual_guarded_ASR_private_ws_log.jsonl'
    event=[json.loads(s) for s in log.read_text().splitlines() if s.strip()]
    counts=collections.Counter(item.get('event') for item in event)
    for typ,count in (('server_start',1),('session_open',3),('session_replaced',2),
                      ('asr_final',1),('speaker_identity_unverified_manual_review',1),('session_close',3)):
        if counts[typ]!=count:raise ValueError('Actual private server did not reconnect and fail closed: '+typ)
    if any(counts[typ] for typ in ('turn_submitted','asr_auto_submitted','answer_start','worker_started','turn_proposed')):
        raise ValueError('Unexpected SWAAG tool or user action during real speech review')
    # The model response is real but made in an independently admitted
    # separate request, NOT a running SWAAG agent or native streaming turn.
    reply=model.get('response_text')
    if (model.get('status')!='real_resident_Coder_text_response_to_prior_actual_Whisper_noaction'
        or model.get('model')!='qwen3-coder-30b-a3b-instruct-q4-k-m'
        or model.get('original_ASR_transcript_SHA256')!=TRANSCRIPT_SHA
        or model.get('actual_new_model_request_count')!=1 or model.get('actual_tool_calls')!=0
        or model.get('actual_live_SWAAG_agent_turn_or_external_action') is not False
        or not isinstance(reply,str) or len(reply)>160 or not reply.strip()
        or hashlib.sha256(reply.encode()).hexdigest()!=ANSWER_SHA
        or model.get('response_text_sha256')!=ANSWER_SHA
        or model.get('real_completed_model_response') is not True
        or model.get('actual_model_reply_HTTP_status')!=200
        or not 0<model.get('model_request_elapsed_seconds',0)<120
        or model.get('model_usage',{}).get('completion_tokens',0)<1):
        raise ValueError('No source-backed single real local Qwen Coder no-action response')
    orig=root/'one_unique_actual_Qwen_local_noaction_spoken_confirmation'
    attempt=orig/'attempt.json';prompt=orig/'private-prompt.json'
    must_regular(attempt);must_regular(prompt)
    frozen=json.loads(attempt.read_text());p=json.loads(prompt.read_text())
    if (frozen.get('one_new_unique_model_request') is not True
        or frozen.get('no_tool_schema') is not True
        or frozen.get('speech_SHA256')!=TRANSCRIPT_SHA
        or frozen.get('fixed_prompt_sha256')!=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
        or p.get('model')!=model['model'] or p.get('stream') is not False
        or p.get('messages',[])[-1].get('content')!='Recorded public speech sample: '+words
        or p.get('tools') or p.get('tool_choice')):
        raise ValueError('Generated no-action response not caused by originally fixed real Whisper text')
    if (tts.get('status')!='real_Phonon_Easton_synthesized_prior_resident_Coder_response_noaction'
        or tts.get('prior_Coder_result_sha256')!=ORIGINAL['one_unique_actual_Qwen_local_noaction_spoken_confirmation/result.json']
        or tts.get('prior_real_Whisper_transcript_sha256')!=TRANSCRIPT_SHA
        or tts.get('actual_noaction_Coder_response_sha256')!=ANSWER_SHA
        or tts.get('real_phonon_tts_attempts')!=1
        or tts.get('TTS_speaker')!='Easton' or tts.get('actual_TTS_engine')!=REAL_TTS_ENGINE
        or tts.get('actual_Phonon_HTTP_status')!=200
        or tts.get('real_external_tool_provider_calls')!=0
        or tts.get('physical_owner_phone_audio_playback') is not False
        or tts.get('real_SWAAG_agent_turn_executed') is not False):
        raise ValueError('Actual local Phonon speech was not the one real source-linked Coder answer')
    file=root/'copied_original_Phonon_noaction_audio.wav'
    with wave.open(str(file),'rb') as audio:
        rate=audio.getframerate();channels=audio.getnchannels();width=audio.getsampwidth()
        frames=audio.getnframes();pcm=audio.readframes(frames)
    duration=frames/rate
    rms=audioop.rms(pcm,2)
    if ((channels,rate,width)!=(1,16000,2) or not 0.3<duration<24 or rms<60
        or hashlib.sha256(file.read_bytes()).hexdigest()!=tts.get('actual_audio_SHA256')
        or file.stat().st_size!=tts.get('actual_audio_byte_size')
        or abs(duration-tts.get('actual_audio_seconds',0))>.005):
        raise ValueError('Actual Phonon file was not a valid complete non-silent WAV')
    for failure_name in ('prior_real_Whisper_failure_one_result.json','prior_real_Whisper_failure_two_result.json'):
        original=reader(failure_name)
        if (original.get('status')!='failed_or_unknown' or original.get('error_type')!='AssertionError'
            or original.get('actual_Android_owner_phone_or_production_server_changed') is not False):
            raise ValueError('Prior two real failed initial audio admission trials misrepresented as successes')
    return {'status':'verified_real_LOCAL_separately_linked_Whisper_Coder_Phonon_and_reconnect_without_user_action',
      'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'actual_previously_recorded_public_speech_SHA256':ORIGINAL['copied_prior_public_voice_W73_derived.wav'],
      'actual_guarded_W71_speech_reconnects':{'websockets':ASR['private_server_WS_connections'],
          'durable_manual_review_only_turns':1,'real_Whisper_requests':1,
          'real_Whisper_HTTP_200_latency_seconds':actual[0]['real_Whisper_response_seconds'],
          'real_ASR_text_SHA256':TRANSCRIPT_SHA,'events':dict(counts)},
      'actual_resident_local_Coder_text_only':{'single_real_model_response':True,
          'response_SHA256':ANSWER_SHA,'latency_seconds':model['model_request_elapsed_seconds'],
          'real_model_tokens':model['model_usage']},
      'actual_resident_local_Phonon_speech_only':{'real_synthesis_count':1,
          'source_model_response_sha256':ANSWER_SHA,'generated_WAV_SHA256':tts['actual_audio_SHA256'],
          'duration_seconds':round(duration,3),'sample_rate':rate,'RMS':rms,
          'TTS_request_wall_seconds':tts['wall_seconds']},
      'two_prior_distinct_real_initial_ASR_source_trials_failed_and_retained':True,
      'no_original_Whisper_Coder_Phonon_requests_replayed':True,
      'model_process_or_tool_provider_calls_during_this_independent_verifier':0,
      'actual_user_phone_playback_or_live_SWAAG_agent_tools':False,
      'actual_Th​or_classifier_or_live_voice_agent_updated':False,
      'private_original_receipt_SHA256':ORIGINAL,
      'source_exact_as_executed_script_SHA256':SCRIPT_SHA,
      'limitations':['This is three separately connected local components, not one live native SWAAG agent session.',
          'Real public recorded speech, not physical user microphone or owner speaker authentication.',
          'Coder used as no-action text-only, not a demonstrated SWAAG autonomous multi-tool agent in this run.',
          'Phonon output was stored as WAV only; no mobile playback or live continuous voice streaming.',
          'Thor dedicated turn/interruption classifier still absent, gateway HTTP503.']}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-root',required=True,type=Path)
    p.add_argument('--source-dir',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args(argv)
    verdict=loadroot(args.private_root,args.source_dir)
    if args.out.exists() or args.out.is_symlink() or not args.out.is_absolute() or not str(args.out).startswith('/data/var/swaag-benchmarks/'):
        raise FileExistsError('Independent model-free chain audit never overwrites existing/uncertain proof')
    fd=os.open(args.out,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(verdict,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':verdict['status'],
      'real_ASR_requests':verdict['actual_guarded_W71_speech_reconnects']['real_Whisper_requests'],
      'websocket_connections':verdict['actual_guarded_W71_speech_reconnects']['websockets'],
      'real_Coder_responses':1,'real_Phonon_synthesis_count':1,'physical_phone_playback':False,
      'verifier_new_model_or_tool_calls':0},sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
