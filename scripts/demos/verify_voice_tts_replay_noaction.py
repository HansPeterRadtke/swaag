#!/usr/bin/env python3
"""Real retained SWAAG/Phonon reply -> isolated voice WebSocket -> resident Whisper.

Security-negative acceptance only. The source is a previous completed and
unambiguously no-action turn, not synthesized again. A fake mobile client
claims it heard the owner; the candidate must treat an external recorded
speaker as UNVERIFIED, preserve an unsubmitted draft, and reject forged
completion after reconnect. No live server, device, Qwen, tool, or TTS call.
The run needs explicit CLI and environment opt-in and claims a unique private
output before the single real resident-ASR session. Never replay an uncertain
attempt. This does not verify true speaker identity or acoustic anti-replay.
"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch
import wave

EXPECTED_SOURCE_HASHES={
    'cloudflare_ws_voice_server.py':'cbcc77fb36ce8cf72cc6442ec0f35d84526aec4868a27b810b2c38bde52bb302',
    'voice_ws_config.py':'b3052253b3d3a22f0f3f69bf6cd264768d6dc0bab99fcd94fbb874b1a98c69ba',
}
EXPECTED_FIXTURE_SHA='dfa7e77a4825639a9b6de17d74dc5167b30d3daa20eccc0af31192bf701bf448'
EXPECTED_AUDIO_SHA='56e81fe7050641490a634e064705673a1f6c61ee2cf46d6b09f33d8270e7fae7'
EXPECTED_RECEIPT_SHA='67e3d1ba1601b1649b65ef594fe7c218cd105ca5123b1b38c3a1dd61941c4367'
OPT_IN='SWAAG_TTS_REPLAY_SPEAKER_NOACTION'
ENDPOINT='http://thor:13484/v1/voice/transcribe-verify'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokens(s: str) -> list[str]:
    return re.findall(r'[a-z0-9]+',s.casefold())


def validate_inputs(candidate: Path, helpers: Path, receipt: Path, audio: Path) -> tuple[bytes,str,dict]:
    """Entire source/voice gate is side-effect-free; never trusts an unpinned path."""
    for p in (candidate,helpers,receipt,audio):
        if p.is_symlink() or not p.exists():
            raise ValueError('Missing or symlinked input')
    for filename,expected in EXPECTED_SOURCE_HASHES.items():
        path=candidate/filename
        if not path.is_file() or path.is_symlink() or digest(path.read_bytes())!=expected:
            raise ValueError('Candidate voice source hash mismatch: '+filename)
    fixture=helpers/'test_cloudflare_ws_voice_server.py'
    if (not fixture.is_file() or fixture.is_symlink()
            or digest(fixture.read_bytes())!=EXPECTED_FIXTURE_SHA):
        raise ValueError('Isolated fixture source hash mismatch')
    if receipt.stat().st_size>200000 or audio.stat().st_size>2000000:
        raise ValueError('Unbounded evidence input')
    original=receipt.read_bytes()
    if digest(original)!=EXPECTED_RECEIPT_SHA:
        raise ValueError('The stored no-action model/source receipt changed')
    prior=json.loads(original)
    reply=(prior.get('canonical_fresh_reply') or {}).get('reply')
    if (prior.get('status')!='real_fast_swaag_direct_reply_phonon_reconnect_noaction'
            or prior.get('canonical_answer_not_fresh') is not False
            or prior.get('fresh_canonical_response_calls')!=1
            or prior.get('model_action_calls')!=0
            or prior.get('played_pcm_recovery_exact') is not True
            or prior.get('normal_ack') is not True
            or prior.get('no_replay_after_ack') is not True
            or not isinstance(reply,str) or len(tokens(reply))<5):
        raise ValueError('No-action original receipt fails acceptance contract')
    raw=audio.read_bytes()
    if (digest(raw)!=EXPECTED_AUDIO_SHA or prior.get('tts_wav_sha256')!=EXPECTED_AUDIO_SHA
            or (prior.get('preserved_generated_audio') or {}).get('retained_wav_sha256')!=EXPECTED_AUDIO_SHA):
        raise ValueError('Retained actual Phonon WAV does not match original model reply')
    with wave.open(str(audio),'rb') as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(16000,1,2):
            raise ValueError('Audio is not the original PCM16 mono 16 kHz format')
        if not 16000<w.getnframes()<16000*30:
            raise ValueError('Recorded voice duration invalid')
        pcm=w.readframes(w.getnframes())
    if len(pcm)<32000 or not any(pcm):
        raise ValueError('Recorded speech is silent or truncated')
    return pcm,reply,prior


async def actual_isolated_replay(candidate: Path, helpers: Path, pcm: bytes,
                                 canonical_reply: str) -> dict:
    # Imports of the candidate happen ONLY after the exact source and opt-in
    # gates. This isolated harness uses its own ephemeral VoiceCore database.
    sys.path.insert(0,str(helpers))
    sys.path.insert(0,str(candidate))
    from websockets.legacy.client import connect
    from websockets.legacy.server import serve
    import cloudflare_ws_voice_server as server
    import test_cloudflare_ws_voice_server as fixture
    test=fixture.ServerTests('test_websocket_bearer_authentication_is_required')
    test.setUp()
    calls=[]; classifier_calls=[]; prohibited_calls=[]; events=[]
    real_asr=server.verify_transcript_large_v3_http

    def measured_real_asr(*args,**kwargs):
        if str(args[0])!=ENDPOINT:
            raise AssertionError('Real resident ASR did not use the fixed private endpoint')
        inp=bytes(args[1])
        if len(inp)<48000 or pcm[24000:36000] not in inp:
            raise AssertionError('The exact recorded assistant voice was not sent to ASR')
        started=time.monotonic()
        text=real_asr(*args,**kwargs)
        calls.append({'seconds':round(time.monotonic()-started,3),
                      'asr_words':len(tokens(text)),'sha256':digest(text.encode()),
                      'real_retained_input_pcm':True})
        return text

    def invented_positive_semantics(*args,**kwargs):
        classifier_calls.append(True)
        return {'complete':True,'kind':'finished','confidence':0.999}

    async def forbidden_action(*args,**kwargs):
        prohibited_calls.append('action')
        raise AssertionError('No SWAAG, TTS, submission or cancellation action allowed')

    async def idle(*args,**kwargs):
        await asyncio.Future()

    async def expect(phone,predicate,seconds=12):
        deadline=asyncio.get_running_loop().time()+seconds
        while True:
            remaining=deadline-asyncio.get_running_loop().time()
            if remaining<=0:raise AssertionError('Missing voice review/reconnect event')
            msg=await asyncio.wait_for(phone.recv(),remaining)
            if not isinstance(msg,str):
                raise AssertionError('A no-action playback-source test emitted binary assistant output')
            obj=json.loads(msg)
            if (obj.get('type')=='turn_proposal' or
                (obj.get('type')=='turn' and obj.get('event')=='submitted') or
                obj.get('type')=='audio'):
                raise AssertionError('Unverified speaker triggered proposed or committed action')
            events.append({k:obj.get(k) for k in ('type','event','state','reason') if k in obj})
            if predicate(obj):return obj

    async def scenario():
        server.CONFIG=replace(server.CONFIG,
            asr=replace(server.CONFIG.asr,final_quality_enabled=True,
                 final_quality_url=ENDPOINT,final_quality_timeout_seconds=44.0),
            interaction=replace(server.CONFIG.interaction,
                auto_send_default=True,require_client_commit=True,
                require_verified_speaker_for_auto_send=True))
        name='isolated-replayed-assistant-speech-unverified-owner'
        headers={'Authorization':'Bearer test-secret-token'}
        with patch.object(server.VoiceSession,'submit_turn',forbidden_action), \
             patch.object(server.VoiceSession,'answer',forbidden_action), \
             patch.object(server,'receive_voice_run',forbidden_action), \
             patch.object(server,'cancel_voice_run',forbidden_action), \
             patch.object(server.VoiceSession,'worker_events_loop',idle), \
             patch.object(server.VoiceSession,'heartbeat_loop',idle), \
             patch.object(server,'dependency_status',return_value={}), \
             patch.object(server,'verify_transcript_large_v3_http',measured_real_asr), \
             patch.object(server,'turn_completion_http',invented_positive_semantics):
            async with serve(server.handler,'127.0.0.1',0,
                             process_request=server.process_request,
                             close_timeout=1,max_queue=256) as listener:
                port=listener.sockets[0].getsockname()[1]
                base='ws://127.0.0.1:'+str(port)+'/ws?conversation='+name+'&generation='
                async with connect(base+'1',extra_headers=headers,close_timeout=1,max_size=2**20) as first:
                    hello=await expect(first,lambda x:x.get('type')=='hello',8)
                    if hello.get('connection_generation')!=1:
                        raise AssertionError('Wrong first generation')
                    # Worst-case old client and forged owner identity, with
                    # automatic submission enabled. Neither is owner authority.
                    await first.send(json.dumps({'type':'client_state',
                        'build':'Voice Agent 1.7.16','auto_send':True,
                        'auto_transcribe':True,'auto_commit_supported':False,
                        'speaker_verified':True,'speaker_identity':'owner',
                        'owner_speaker_identity':'trusted','media_protocol':0}))
                    size=server.cfg().audio.frame_ms*32
                    for n in range(0,len(pcm),size):
                        chunk=pcm[n:n+size]
                        await first.send(chunk+b'\x00'*(size-len(chunk)))
                    for _ in range(31):await first.send(b'\x00'*size)
                    await asyncio.wait_for(first.ping(),5)
                    final=await expect(first,lambda x:x.get('type')=='asr' and
                                       x.get('event')=='final',75)
                    review=await expect(first,lambda x:x.get('type')=='state' and
                        x.get('state')=='awaiting_send' and
                        x.get('reason')=='speaker_identity_unverified',12)
                    if final.get('turn_id')!=review.get('turn_id'):
                        raise AssertionError('Reviewed turn mismatch')
                    if not final.get('auto_send'):
                        raise AssertionError('Automatic scenario was silently disabled')
                    real_words=tokens(final.get('text') or '')
                    exp_words=tokens(canonical_reply)
                    if real_words!=exp_words:
                        raise AssertionError('The resident ASR did not recognize the retained assistant voice exactly')
                    history=server._history_read(name)['dialogue_history']
                    if len(history)!=1 or history[0].get('status')!='draft' or \
                       history[0].get('submitted_text') or not history[0].get('requires_review'):
                        raise AssertionError('Unknown speaker not preserved as one unsubmitted draft')
                    if not history[0].get('user_audio_path'):
                        raise AssertionError('Source recording not preserved for manual review')
                async with connect(base+'2',extra_headers=headers,close_timeout=1,max_size=2**20) as second:
                    await expect(second,lambda x:x.get('type')=='hello',8)
                    await expect(second,lambda x:x.get('type')=='history',8)
                    restored=await expect(second,lambda x:x.get('type')=='asr' and
                        x.get('event')=='final' and x.get('restored') is True,9)
                    await expect(second,lambda x:x.get('type')=='state' and
                        x.get('reason')=='restored_draft',9)
                    if restored.get('turn_id')!=final['turn_id'] or \
                       restored.get('text')!=final['text']:
                        raise AssertionError('Reconnect did not restore same reviewable draft')
                    await second.send(json.dumps({'type':'client_state','auto_send':True,
                       'auto_commit_supported':True,'speaker_verified':True,
                       'speaker_identity':'owner'}))
                    await second.send(json.dumps({'type':'commit_turn',
                       'proposal_id':'forged-owner-approved-result',
                       'text':restored['text'],'speaker_verified':True}))
                    rejected=await expect(second,lambda x:x.get('type')=='state' and
                        x.get('reason')=='speaker_identity_unverified',10)
                    if rejected.get('turn_id')!=final['turn_id']:
                        raise AssertionError('Forged commit changed review authority')
                    after=server._history_read(name)['dialogue_history']
                    if len(after)!=1 or after[0].get('submitted_text') or \
                       after[0].get('status')!='draft':
                        raise AssertionError('Forged reconnect committed unverified speech')
        if len(calls)!=1 or classifier_calls or prohibited_calls:
            raise AssertionError('No-action ASR/classifier/tool count contract failed')
        return {'status':'verified_replayed_assistant_voice_requires_review',
                'recorded_output_WAV_sha256':EXPECTED_AUDIO_SHA,
                'source_ASR_words':len(real_words),
                'canonical_text_words':len(exp_words),'exact_normalized_words':True,
                'resident_large_v3_ASR_requests':len(calls),
                'observed_ASR_seconds':calls[0]['seconds'],
                'positive_semantic_classifier_requests':len(classifier_calls),
                'automatic_turn_proposals':0,'automatic_turn_submissions':0,
                'forbidden_real_SWAAG_tools_TTS_cancel_calls':len(prohibited_calls),
                'legacy_and_new_client_spoof_claims_denied':True,
                'same_reviewable_draft_after_reconnect':True,
                'speaker_verified':False,'live_service_modified':False,
                'real_owner_voice_identity_or_replay_prevention_proven':False,
                'scope':'isolated candidate, test-only client, actual recorded Phonon + resident Whisper'}

    try:
        return await asyncio.wait_for(scenario(),100)
    finally:
        test.tearDown()


def _write_unique(path: Path, payload: dict) -> None:
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as out:
        json.dump(payload,out,ensure_ascii=False,indent=2)
        out.flush();os.fsync(out.fileno())


def main(argv=None) -> int:
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--candidate',type=Path,required=True)
    cli.add_argument('--helpers',type=Path,required=True)
    cli.add_argument('--receipt',type=Path,required=True)
    cli.add_argument('--audio',type=Path,required=True)
    cli.add_argument('--evidence',type=Path,required=True)
    cli.add_argument('--run',action='store_true')
    args=cli.parse_args(argv)
    pcm,reply,_=validate_inputs(args.candidate,args.helpers,args.receipt,args.audio)
    hashes={'candidate_source_sha256':EXPECTED_SOURCE_HASHES['cloudflare_ws_voice_server.py'],
        'original_recorded_assistant_audio_sha256':EXPECTED_AUDIO_SHA,
        'original_result_receipt_sha256':EXPECTED_RECEIPT_SHA,
        'original_pcm_bytes':len(pcm),'expected_reply_words':len(tokens(reply))}
    if not args.run:
        print(json.dumps({'status':'source_only_no_network_or_write',**hashes},sort_keys=True))
        return 0
    if os.environ.get(OPT_IN)!='1':
        raise ValueError('Explicit environment opt-in required for actual resident ASR')
    if args.evidence.is_symlink() or args.evidence.exists():
        raise FileExistsError('Existing evidence/unknown result must never be overwritten or retried')
    args.evidence.mkdir(parents=True,mode=0o700,exist_ok=False)
    os.chmod(args.evidence,0o700)
    _write_unique(args.evidence/'attempt.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'source':hashes,'ASR_requests_permitted':1,'no_retry_of_uncertain_attempt':True})
    t0=time.monotonic()
    try:
        observed=asyncio.run(actual_isolated_replay(args.candidate,args.helpers,pcm,reply))
        observed.update(hashes)
    except BaseException as exc:
        observed={'status':'failed_or_uncertain','error_type':type(exc).__name__,
                  'error_detail_internal':str(exc)[:350],**hashes}
    observed['elapsed_seconds']=round(time.monotonic()-t0,3)
    _write_unique(args.evidence/'result.json',observed)
    # Avoid leaking original owner or candidate model speech to public logs.
    print(json.dumps({k:observed.get(k) for k in ('status','error_type','elapsed_seconds',
         'source_ASR_words','resident_large_v3_ASR_requests',
         'automatic_turn_submissions','same_reviewable_draft_after_reconnect')}))
    return 0 if observed['status']=='verified_replayed_assistant_voice_requires_review' else 2


if __name__=='__main__':
    sys.exit(main())
