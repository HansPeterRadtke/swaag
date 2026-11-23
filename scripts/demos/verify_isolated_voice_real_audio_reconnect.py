#!/usr/bin/env python3
"""One-use real audio -> independent private voice-server process -> live ASR -> reconnect.

Uses an exact source-pinned, UNDEPLOYED patched server and an old real no-tool
SWAAG/Phonon reply recording. An exclusive localhost ASR relay forwards ONE
copy of original captured speech to the existing Thor resident Whisper; no
other service endpoint is reachable through the test configuration. The
private WebSocket client claims to be the owner and requests auto-send.
Expected: recognized words, durable review only, no classified or submitted
turn, same draft after reconnect, and forged owner/commit rejection.
Never deploy, restart, authenticate to the real voice service or rerun the
same uncertain attempt. This is NOT real Android microphone acceptance or
actual human speaker recognition.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import wave

from verify_isolated_voice_release_server import (
    EXPECTED_SOURCES, EXPECTED_TEMPLATE_SHA, inspect, scrub_config,
    runtime_python, _write_unique,
)
from verify_voice_tts_replay_noaction import (
    EXPECTED_AUDIO_SHA, EXPECTED_RECEIPT_SHA, validate_inputs, tokens,
)

OPT_IN = 'SWAAG_REAL_PROCESS_ASR_REVIEW_NOACTION'
ASR_ENDPOINT = 'http://thor:13484/v1/voice/transcribe-verify'
STANDBY_LOCALHOST = 'http://127.0.0.1:9/'


def evidence_guard(src: Path, helper: Path, template: Path,
                   receipt: Path, audio: Path) -> tuple[dict,bytes,str]:
    original=inspect(src,template)
    pcm,reply,_=validate_inputs(src,helper,receipt,audio)
    if len(pcm)<40000 or len(tokens(reply))<5:
        raise ValueError('Insufficient prior genuine speech evidence')
    return original,pcm,reply


class ASRAuthoritativeRelay:
    """Exact one-use forwarded request, with source recording proof."""
    def __init__(self, original_pcm: bytes):
        self.original_pcm=original_pcm
        self.calls=[]
        self.httpd=None
        self.thread=None

    def start(self):
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args,**kwargs): pass

            def do_POST(self):
                note={'path':self.path,'utc':datetime.now(timezone.utc).isoformat()}
                owner.calls.append(note)
                if self.path!='/v1/voice/transcribe-verify' or len(owner.calls)!=1:
                    self.send_error(409,'Unsupported/duplicate ASR request');return
                try:
                    length=int(self.headers.get('Content-Length') or '0')
                    if not 1000<length<250000:
                        raise ValueError('Unbounded ASR input')
                    body=self.rfile.read(length)
                    if len(body)!=length or not body.startswith(b'RIFF'):
                        raise ValueError('Invalid real source PCM WAV')
                    # The actual incoming audio includes a silence tail from
                    # acoustic end-of-speech segmentation. Verify that the
                    # original *interior* generated Phonon PCM is present.
                    if owner.original_pcm[14000:29000] not in body:
                        raise ValueError('Real retained speech was not forwarded')
                    note['input_bytes']=len(body)
                    note['contains_original_recorded_speech']=True
                    headers={'Content-Type':'audio/wav','X-Voice-Language':'en'}
                    prompt=self.headers.get('X-Voice-Prompt')
                    if prompt:headers['X-Voice-Prompt']=prompt[:6000]
                    request=urllib.request.Request(
                        ASR_ENDPOINT,data=body,method='POST',headers=headers)
                    started=time.monotonic()
                    with urllib.request.urlopen(request,timeout=40) as response:
                        raw=response.read(100001)
                        if response.status!=200 or len(raw)>100000:
                            raise ValueError('Unexpected resident Whisper response')
                    value=json.loads(raw)
                    if (not isinstance(value,dict) or value.get('ok') is False
                            or not isinstance(value.get('text'),str)):
                        raise ValueError('Invalid resident Whisper transcription')
                    note['real_Whisper_text_words']=len(tokens(value['text']))
                    note['ASR_seconds']=round(time.monotonic()-started,3)
                    note['real_Whisper_transcript_sha256']=hashlib.sha256(value['text'].encode()).hexdigest()
                    self.send_response(200)
                    self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(raw)))
                    self.end_headers();self.wfile.write(raw)
                except Exception as exc:
                    note['failure_type']=type(exc).__name__
                    try:self.send_error(502,'Isolated real-ASR test failure')
                    except OSError:pass
        self.httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.httpd.daemon_threads=True
        self.thread=threading.Thread(target=self.httpd.serve_forever,
                                     name='single-real-asr-test-relay',daemon=True)
        self.thread.start()
        return self.httpd.server_address[1]

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread:self.thread.join(timeout=2)


def private_config(original:dict,root:Path,port:int,relay_port:int)->dict:
    x=scrub_config(original,root,port)
    x['asr']['final_quality_enabled']=True
    x['asr']['final_quality_url']=f'http://127.0.0.1:{relay_port}/v1/voice/transcribe-verify'
    # No other configured path may reach any active model/agent service.
    for area in ('asr','llm','tts'):
        for k,v in x[area].items():
            if k.endswith('_url') or k=='url':
                if area=='asr' and k=='final_quality_url':continue
                if v and not str(v).startswith(STANDBY_LOCALHOST):
                    raise ValueError('Unexpected external model route: '+area+'.'+k)
    if x['interaction'].get('require_verified_speaker_for_auto_send') is not True:
        raise ValueError('Candidate configuration is not fail-closed')
    return x


async def real_ws_scenario(port:int,pcm:bytes,reply:str,relay:ASRAuthoritativeRelay)->dict:
    from websockets.legacy.client import connect
    base=f'ws://127.0.0.1:{port}/ws?conversation=private-actual-wav-review-w61&generation='
    headers={'Authorization':'Bearer test-only-private-voice-key'}
    events=[]
    async def receive(ws,predicate,seconds=15):
        end=asyncio.get_running_loop().time()+seconds
        while True:
            left=end-asyncio.get_running_loop().time()
            if left<=0:raise TimeoutError('Expected private voice event not received')
            value=await asyncio.wait_for(ws.recv(),left)
            if not isinstance(value,str):
                raise AssertionError('Unexpected generated audio from private no-action server')
            x=json.loads(value)
            if x.get('type') in ('turn_proposal','audio') or (
                    x.get('type')=='turn' and x.get('event')=='submitted'):
                raise AssertionError('Unverified actual speech was submitted or spoken as agent')
            if x.get('type')=='state' and x.get('state')=='checking_turn':
                raise AssertionError('Semantic auto-completion was invoked for unknown speaker')
            events.append({k:x.get(k) for k in ('type','event','state','reason') if k in x})
            if predicate(x):return x

    async with connect(base+'1',extra_headers=headers,open_timeout=5,
                       close_timeout=1,max_size=2**20) as first:
        hello=await receive(first,lambda x:x.get('type')=='hello',6)
        if hello.get('unverified_speaker_requires_review') is not True:
            raise AssertionError('Patched speaker-review safeguard not advertised')
        await first.send(json.dumps({'type':'client_state','build':'Voice Agent 1.7.16',
             'auto_send':True,'auto_transcribe':True,'media_protocol':0,
             'auto_commit_supported':False,'speaker_verified':True,
             'speaker_identity':'owner','owner_speaker_identity':'trusted'}))
        # Old-Android-like binary microphone PCM. Silences force a real
        # acoustic utterance boundary rather than synthetic text submission.
        size=640
        for i in range(0,len(pcm),size):
            frame=pcm[i:i+size]
            await first.send(frame+b'\0'*(size-len(frame)))
        for i in range(35):await first.send(b'\0'*size)
        await asyncio.wait_for(first.ping(),6)
        final=await receive(first,lambda x:x.get('type')=='asr' and
                            x.get('event')=='final',85)
        review=await receive(first,lambda x:x.get('type')=='state' and
                      x.get('state')=='awaiting_send' and
                      x.get('reason')=='speaker_identity_unverified',12)
        if (final.get('turn_id')!=review.get('turn_id') or
            final.get('auto_send') is not True):
            raise AssertionError('Actual auto-send scenario was not reviewed')
        if tokens(final.get('text') or '')!=tokens(reply):
            raise AssertionError('Real audio/transcription differs from original reply')
        turn=final['turn_id']

    async with connect(base+'2',extra_headers=headers,open_timeout=5,
                       close_timeout=1,max_size=2**20) as new:
        await receive(new,lambda x:x.get('type')=='hello',6)
        history=await receive(new,lambda x:x.get('type')=='history',8)
        drafts=[r for r in history.get('turns',[]) if r.get('status')=='draft']
        if (len(drafts)!=1 or drafts[0].get('turn_id')!=turn or
            drafts[0].get('submitted_text') or not drafts[0].get('requires_review')):
            raise AssertionError('Unverified speaker was not stored as one durable review draft')
        restored=await receive(new,lambda x:x.get('type')=='asr' and
                               x.get('event')=='final' and x.get('restored') is True,9)
        await receive(new,lambda x:x.get('type')=='state' and
                      x.get('reason')=='restored_draft',9)
        if restored.get('turn_id')!=turn or restored.get('text')!=final.get('text'):
            raise AssertionError('Reconnect did not replay the exact same review draft')
        await new.send(json.dumps({'type':'client_state','auto_send':True,
             'auto_commit_supported':True,'speaker_verified':True,
             'owner_speaker_identity':'trusted'}))
        await new.send(json.dumps({'type':'commit_turn',
             'proposal_id':'forged-privileged-owner-turn',
             'text':final['text'],'speaker_verified':True}))
        rejected=await receive(new,lambda x:x.get('type')=='state' and
                   x.get('reason')=='speaker_identity_unverified',9)
        if rejected.get('turn_id')!=turn:
            raise AssertionError('Forged owner commit did not preserve review turn')
        await new.send(json.dumps({'type':'history_get'}))
        state=await receive(new,lambda x:x.get('type')=='history',9)
        still=[r for r in state.get('turns',[]) if r.get('status')=='draft']
        if len(still)!=1 or still[0].get('submitted_text'):
            raise AssertionError('Forged commit changed unsubmitted durable record')
    if len(relay.calls)!=1 or relay.calls[0].get('real_Whisper_text_words') is None:
        raise AssertionError('Exactly one successful resident Whisper request required')
    return {'status':'real_process_real_WAV_ASR_reconnect_review_verified',
        'source_recording_sha256':EXPECTED_AUDIO_SHA,
        'recognized_word_count':len(tokens(reply)),
        'one_actual_resident_Whisper_request':True,
        'ASR_seconds':relay.calls[0]['ASR_seconds'],
        'ASR_result_sha256':relay.calls[0]['real_Whisper_transcript_sha256'],
        'real_original_pcm_inside_ASR_request':True,
        'original_auto_send_true':True,
        'forged_legacy_owner_claim_did_not_authorize':True,
        'forged_new_protocol_commit_did_not_authorize':True,
        'durable_review_restored_after_socket_reconnect':True,
        'automatic_submissions':0,'semantic_classifier_requests':0,
        'actual_SWAAG_LLM_TTS_calls':0,
        'real_human_owner_identity_verified':False,
        'physical_Android_device_used':False,
        'events_seen':len(events),
        'same_actual_child_stayed_active_across_socket_reconnect':True}


def run_private_child(source:Path,original:dict,pcm:bytes,reply:str,out:Path)->dict:
    import socket
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    relay=ASRAuthoritativeRelay(pcm)
    relay_port=relay.start()
    child=None
    try:
        cfg=private_config(original,out,port,relay_port)
        _write_unique(out/'fake-only-bearer',b'test-only-private-voice-key\n')
        _write_unique(out/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
        _write_unique(out/'inert-piper.onnx',b'no-real-model')
        _write_unique(out/'inert-piper.onnx.json',b'{}')
        _write_unique(out/'config.json',json.dumps(cfg,sort_keys=True).encode())
        # First prove an actual identical original source release starts;
        # no service worker restarts, shared token, or external agent login.
        logfile=out/'private-child-console.log'
        with logfile.open('xb') as sink:
            child=subprocess.Popen([runtime_python(),'-u',str(source/'cloudflare_ws_voice_server.py'),
                    '--config',str(out/'config.json')],cwd=str(source),
                    stdin=subprocess.DEVNULL,stdout=sink,stderr=subprocess.STDOUT,
                    close_fds=True,start_new_session=True,
                    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(source)})
            # Reuse only the original process, never resubmit to SWAAG.
            for i in range(40):
                if child.poll() is not None:
                    raise RuntimeError('Isolated real-source voice server exited before bind')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.2):
                        break
                except OSError:time.sleep(.1)
            else:raise RuntimeError('Isolated real-source server never bound')
            return asyncio.run(asyncio.wait_for(
                real_ws_scenario(port,pcm,reply,relay),timeout=100))
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait(timeout=4)
            if child.poll() is None:raise RuntimeError('Private child could not be stopped')
        relay.stop()
        if (out/'private-child-console.log').exists():
            os.chmod(out/'private-child-console.log',0o600)


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','helpers','template','receipt','audio','evidence'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    original,pcm,reply=evidence_guard(a.source,a.helpers,a.template,a.receipt,a.audio)
    if not a.run:
        print(json.dumps({'status':'no_network_source_and_audio_preflight',
            'source_sha256':EXPECTED_SOURCES['cloudflare_ws_voice_server.py'],
            'audio_sha256':EXPECTED_AUDIO_SHA,'expected_words':len(tokens(reply))}))
        return 0
    if os.environ.get(OPT_IN)!='1':
        raise ValueError('One-use real resident ASR requires explicit environment opt-in')
    if a.evidence.is_symlink() or a.evidence.exists():
        raise FileExistsError('Refuse to replay or overwrite any earlier/uncertain attempt')
    a.evidence.mkdir(parents=True,mode=0o700,exist_ok=False)
    os.chmod(a.evidence,0o700)
    _write_unique(a.evidence/'attempt.json',json.dumps({
        'at':datetime.now(timezone.utc).isoformat(),
        'server_source_sha256':EXPECTED_SOURCES['cloudflare_ws_voice_server.py'],
        'input_audio_sha256':EXPECTED_AUDIO_SHA,'one_real_ASR_call':True,
        'no_retry_after_uncertain_attempt':True}).encode())
    start=time.monotonic()
    try:result=run_private_child(a.source,original,pcm,reply,a.evidence)
    except BaseException as exc:
        result={'status':'failed_or_unknown','failure_type':type(exc).__name__,
                'failure_detail_private':str(exc)[:300]}
    result.update({'elapsed_seconds':round(time.monotonic()-start,3),
        'source_sha256':EXPECTED_SOURCES['cloudflare_ws_voice_server.py'],
        'test_was_only_private_ephemeral_child':True,
        'live_Jetson_service_never_restarted_or_modified':True})
    _write_unique(a.evidence/'result.json',json.dumps(result,indent=2).encode())
    print(json.dumps({k:result.get(k) for k in ('status','failure_type','elapsed_seconds',
        'recognized_word_count','one_actual_resident_Whisper_request',
        'durable_review_restored_after_socket_reconnect','automatic_submissions')}))
    return 0 if result['status']=='real_process_real_WAV_ASR_reconnect_review_verified' else 2

if __name__=='__main__':raise SystemExit(main())
