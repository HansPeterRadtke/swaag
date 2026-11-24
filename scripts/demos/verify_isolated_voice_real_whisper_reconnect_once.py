#!/usr/bin/env python3
"""One-use genuine *local Whisper* ASR through W71 guarded SWAAG WebSocket.

Source-backed real 16 kHz public LibriSpeech WAV is streamed to a separate
private localhost SWAAG server child. Its source native quality verification
WAV is forwarded exactly ONCE via an explicit local-only adaptor to the
already running Jetson Whisper `/inference`. With the genuine ASR result
held, a replacement WebSocket reconnects, then receives exactly one
unsubmitted review-only voice draft. Another reconnect tests durable replay.
This tests real recorded speech + real shared local ASR + real server/WebSocket
and process shutdown; it does NOT test physical phone microphone, owner voice
speaker liveness, Qwen dialogue, Phonon spoken replies or any agent tool.
Model/LLM/TTS and action routes are 127.0.0.1:9 (unreachable), and the server
requires manual speaker review even when an untrusted client claims verified.
No production systemd/restart/phone/provider/owner account writes.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime,timezone
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import wave

from verify_isolated_voice_owned_socket_ws_once import (
    BASE, PATCHED_SERVER_SHA256, FAKE_BEARER,
    private_config, sha, runtime_python, _write_unique,
)
from verify_isolated_voice_release_server import inspect,EXPECTED_TEMPLATE_SHA

SOURCE_MANIFEST_SHA='f361faa9283b55e708f4c7f590d3b0e5a90bd1eeb9aa16fe2f88876e8c6df6d0'
INPUT_SHA='799f78ed4beb4de7ceae3a809262d4ce242394342ccd1d58cef7d49dbc2def46'
MODEL_ORIGIN='http://127.0.0.1:15300/inference'
BASENAME='public_speech_fixture_1272_0000.wav'
OPTIN='SWAAG_W73_REAL_WHISPER_RECONNECT_MANUAL_REVIEW_ONCE'


def preflight(source:Path,template:Path,baseline:Path,audio:Path):
    original=inspect(baseline,template)
    if source.is_symlink() or not source.is_dir():raise ValueError('Unknown private W71 source')
    for name,expected in BASE.items():
        file=source/name
        want=PATCHED_SERVER_SHA256 if name=='cloudflare_ws_voice_server.py' else expected
        if file.is_symlink() or not file.is_file() or sha(file)!=want:
            raise ValueError('Unpinned previously tested SWAAG source: '+name)
    manifest=source/'source-manifest.json'
    if manifest.is_symlink() or not manifest.is_file() or sha(manifest)!=SOURCE_MANIFEST_SHA:
        raise ValueError('Missing immutable one-module source manifest')
    if json.loads(manifest.read_text()).get('new_undeployed_server_sha256')!=PATCHED_SERVER_SHA256:
        raise ValueError('Not the previously separately tested W71 source release candidate')
    if audio.is_symlink() or not audio.is_file() or audio.name!=BASENAME or sha(audio)!=INPUT_SHA:
        raise ValueError('Missing/changed genuine public LibriSpeech recorded voice')
    with wave.open(str(audio),'rb') as src:
        if (src.getnchannels(),src.getsampwidth(),src.getframerate())!=(1,2,16000):
            raise ValueError('Expected actual mono 16kHz 16-bit public speech')
        pcm=src.readframes(src.getnframes())
        seconds=src.getnframes()/src.getframerate()
    if not 5<=seconds<=7 or len(pcm)!=audio.stat().st_size-44:
        raise ValueError('Unexpected actual source recording duration or header')
    if (os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')) and '127.0.0.1' not in os.getenv('NO_PROXY',''):
        raise ValueError('Private local model route would pass through configured proxy')
    return original,pcm,round(seconds,3)


class ExactOneLocalWhisper:
    def __init__(self,pcm:bytes):
        self.pcm=pcm
        self.received=threading.Event()
        self.forwarded=threading.Event()
        self.release=threading.Event()
        self.done=threading.Event()
        self.httpd=None;self.thread=None
        self.records=[]
        self.transcript=None
        self.error_type=None

    def start(self)->int:
        outer=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                entry={'time_utc':datetime.now(timezone.utc).isoformat(),'path':self.path}
                outer.records.append(entry)
                try:
                    if self.path!='/v1/voice/transcribe-verify' or len(outer.records)!=1:
                        raise ValueError('Real local ASR relay accepts only one unique private WAV')
                    n=int(self.headers.get('Content-Length') or 0)
                    if not 180000<n<250000:raise ValueError('Unexpected actual server WAV bytes')
                    data=self.rfile.read(n)
                    if data[:4]!=b'RIFF' or outer.pcm[14000:29000] not in data:
                        raise ValueError('Server audio does not contain exact public recorded PCM')
                    with wave.open(io.BytesIO(data),'rb') as wav:
                        if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())!=(1,2,16000):
                            raise ValueError('Not actual 16kHz mono PCM from guarded SWAAG')
                    entry['real_server_WAV_sha256']=hashlib.sha256(data).hexdigest()
                    entry['real_recorded_source_PCM_verified']=True
                    entry['WAV_bytes_to_real_Whisper']=len(data)
                    outer.received.set()
                    # Locally installed resident Whisper.cpp. Exactly one
                    # network request to literal private LOOPBACK, zero
                    # external providers or user accounts. No retry on 429/5xx.
                    import requests
                    start=time.perf_counter()
                    with requests.Session() as session:
                        session.trust_env=False
                        response=session.post(MODEL_ORIGIN,
                            files={'file':('one-public-test.wav',data,'audio/wav')},
                            data={'response_format':'json','language':'en'},
                            timeout=33)
                    entry['real_Whisper_HTTP_status']=response.status_code
                    entry['real_Whisper_response_seconds']=round(time.perf_counter()-start,3)
                    response.raise_for_status()
                    parsed=response.json()
                    transcript=' '.join(str(parsed.get('text') or '').split())
                    if not transcript or len(transcript)<12 or len(transcript)>2000:
                        raise ValueError('No genuine usable recognized words, fail closed')
                    outer.transcript=transcript
                    entry['actual_Whisper_text_sha256']=hashlib.sha256(transcript.encode()).hexdigest()
                    entry['actual_Whisper_text_length']=len(transcript)
                    outer.forwarded.set()
                    if not outer.release.wait(timeout=19):
                        raise TimeoutError('New socket did not reconnect before delivering already computed result')
                    body=json.dumps({'ok':True,'text':transcript,'language':'en',
                        'time_sec':entry['real_Whisper_response_seconds']}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    entry['same_real_ASR_delivered_after_client_reconnect']=True
                except BaseException as exc:
                    outer.error_type=type(exc).__name__
                    entry['error_type']=type(exc).__name__
                    entry['error_private']=str(exc)[:240]
                    outer.forwarded.set()
                    try:self.send_error(502,'Local ASR private relay rejected')
                    except OSError:pass
                finally:outer.done.set()
        self.httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.httpd.daemon_threads=True
        self.thread=threading.Thread(target=self.httpd.serve_forever,daemon=True,name='one-genuine-whisper-private-adaptor')
        self.thread.start()
        return self.httpd.server_address[1]

    def stop(self):
        self.release.set()
        if self.httpd:self.httpd.shutdown();self.httpd.server_close()
        if self.thread:self.thread.join(timeout=3)


async def real_speech_reconnect_scenario(port:int,pcm:bytes,asr:ExactOneLocalWhisper)->dict:
    from websockets.legacy.client import connect
    url=f'ws://127.0.0.1:{port}/ws?conversation=w73-one-public-speech-real-whisper&generation='
    headers={'Authorization':'Bearer '+FAKE_BEARER}
    events=[]
    async def until(ws,predicate,seconds=9):
        deadline=asyncio.get_running_loop().time()+seconds
        while True:
            remaining=deadline-asyncio.get_running_loop().time()
            if remaining<=0:raise TimeoutError('Missing real private WebSocket progress event')
            message=await asyncio.wait_for(ws.recv(),remaining)
            if not isinstance(message,str):raise AssertionError('Unverified speaker triggered binary TTS/playback')
            data=json.loads(message)
            if (data.get('type') in ('audio','turn_proposal')
                or (data.get('type')=='turn' and data.get('event')=='submitted')
                or (data.get('type')=='state' and data.get('state')=='checking_turn')):
                raise AssertionError('Untrusted real transcription tried voice tool, LLM, semantic auto-send or playback')
            events.append({k:data.get(k) for k in ('type','event','state','reason') if k in data})
            if predicate(data):return data
    async with connect(url+'1',extra_headers=headers,close_timeout=1,max_size=2**20) as first:
        first_hello=await until(first,lambda d:d.get('type')=='hello')
        if first_hello.get('unverified_speaker_requires_review') is not True:
            raise AssertionError('Missing manual speaker-review safeguard')
        await first.send(json.dumps({'type':'client_state','build':'W73 private real ASR recorded test',
            'auto_send':True,'auto_transcribe':True,'auto_commit_supported':False,
            'speaker_verified':True,'owner_speaker_identity':'owner_verified_UNTRUSTED_CLIENT_CLAIM'}))
        frame=640
        for pos in range(0,len(pcm),frame):
            part=pcm[pos:pos+frame]
            await first.send(part+b'\0'*(frame-len(part)))
        for _ in range(75):await first.send(b'\0'*frame)
        await asyncio.wait_for(first.ping(),4)
        await until(first,lambda d:d.get('type')=='state' and d.get('state')=='transcribing',10)
        observed=await asyncio.get_running_loop().run_in_executor(None,asr.received.wait,10)
        if not observed:raise AssertionError('Actual guarded server never submitted genuine recorded PCM to ASR relay')
        recognized=await asyncio.get_running_loop().run_in_executor(None,asr.forwarded.wait,42)
        if not recognized or asr.error_type or not asr.transcript:
            raise RuntimeError('Actual local Whisper did not recognize recorded words: '+str(asr.error_type))
        # The actual Whisper result is HELD, no fake transcript; idle socket
        # replacement must retain a real source speech draft for review.
        async with connect(url+'2',extra_headers=headers,close_timeout=1,max_size=2**20) as replacement:
            h=await until(replacement,lambda d:d.get('type')=='hello')
            if h.get('unverified_speaker_requires_review') is not True:
                raise AssertionError('Replacement socket speaker review disabled')
            initial=await until(replacement,lambda d:d.get('type')=='history')
            if initial.get('turns')!=[]:raise AssertionError('Draft created before actual ASR result was released')
            asr.release.set()
            finished=await asyncio.get_running_loop().run_in_executor(None,asr.done.wait,10)
            if not finished or asr.error_type:raise RuntimeError('Real Whisper result delivery failed')
            actual=await until(replacement,lambda d:d.get('type')=='asr' and d.get('event')=='final',12)
            if actual.get('text')!=asr.transcript:
                raise AssertionError('Actual model transcript differed from guarded socket output')
            state=await until(replacement,lambda d:d.get('type')=='state' and
                d.get('state')=='awaiting_send' and d.get('reason')=='speaker_identity_unverified',10)
            await replacement.send(json.dumps({'type':'history_get'}))
            history=await until(replacement,lambda d:d.get('type')=='history')
            turns=history.get('turns') or []
            if (len(turns)!=1 or turns[0].get('status')!='draft'
                or turns[0].get('recognized_text')!=asr.transcript
                or turns[0].get('requires_review') is not True
                or turns[0].get('submitted_text') or turns[0].get('assistant')):
                raise AssertionError('Actual resident Whisper source did not persist precisely one review-only draft')
        # Third actual reconnect after model completion verifies no duplicate.
        async with connect(url+'3',extra_headers=headers,close_timeout=1,max_size=2**20) as third:
            hello3=await until(third,lambda d:d.get('type')=='hello')
            if hello3.get('unverified_speaker_requires_review') is not True:
                raise AssertionError('Third connection speaker protection lost')
            history3=await until(third,lambda d:d.get('type')=='history')
            rows=history3.get('turns') or []
            if (len(rows)!=1 or rows[0].get('recognized_text')!=asr.transcript or
                rows[0].get('status')!='draft' or rows[0].get('submitted_text') or
                rows[0].get('requires_review') is not True):
                raise AssertionError('Reconnecting after real ASR did not replay one exact review-only draft')
    return {'status':'real_local_Whisper_ASR_complete_and_two_reconnects_review_only',
        'actual_local_Whisper_model_calls':len(asr.records),
        'asr_text_sha256':hashlib.sha256(asr.transcript.encode()).hexdigest(),
        'asr_text_chars':len(asr.transcript),
        'private_server_WS_connections':3,
        'model_generated_text_fully_matches_actual_guarded_speech_draft':True,
        'actual_durable_history_turn_count':1,
        'actual_durable_history_status':'draft',
        'manual_speaker_review_required':True,
        'actual_external_provider_SWAAG_tool_or_TTS_calls':0,
        'recorded_safety_WebSocket_event_count':len(events)}


def private_runtime_once(source:Path,original:dict,pcm:bytes,root:Path)->dict:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    asr=ExactOneLocalWhisper(pcm)
    relay_port=asr.start();child=None
    try:
        c=private_config(original,root,port,relay_port)
        c['asr']['final_quality_timeout_seconds']=51
        # A single explicit real local ASR call is allowed from ONLY the
        # relay; any other model/LLM/TTS calls must remain unroutable.
        for group in ('asr','llm','tts'):
            for key,value in c[group].items():
                if (key.endswith('_url') or key=='url') and value:
                    if group=='asr' and key=='final_quality_url':
                        if value!=f'http://127.0.0.1:{relay_port}/v1/voice/transcribe-verify':
                            raise ValueError('ASR relay url unexpected')
                    elif not str(value).startswith('http://127.0.0.1:9/'):
                        raise ValueError('Any hidden nonlocal or real agent action/model route forbidden')
        if c['interaction']['require_verified_speaker_for_auto_send'] is not True:
            raise ValueError('Manual speaker review safety guard must stay ON')
        _write_unique(root/'fake-only-bearer',(FAKE_BEARER+'\n').encode())
        _write_unique(root/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
        _write_unique(root/'inert-piper.onnx',b'NO_TTS_IN_THIS_TEST')
        _write_unique(root/'inert-piper.onnx.json',b'{}')
        _write_unique(root/'private-config.json',json.dumps(c,sort_keys=True).encode())
        with (root/'private-child-console.log').open('xb') as out:
            child=subprocess.Popen([runtime_python(),'-u',str(source/'cloudflare_ws_voice_server.py'),
                '--config',str(root/'private-config.json')],cwd=source,
                stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(source)})
            for _ in range(75):
                if child.poll() is not None:raise RuntimeError('Private source child exited before ready')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.15):break
                except OSError:time.sleep(.1)
            else:raise TimeoutError('Private source child never bound localhost')
            result=asyncio.run(asyncio.wait_for(real_speech_reconnect_scenario(port,pcm,asr),timeout=97))
            result['private_server_pid']=child.pid
            result['actual_Whisper_adaptor_records']=asr.records
            return result
    finally:
        asr.stop()
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=6)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            if child.poll() is None:raise RuntimeError('Owned private source server process not stopped')
        for f in (root/'private-child-console.log',root/'private-config.json'):
            if f.exists():os.chmod(f,0o600)


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','baseline','template','public-speech-wav','private-result'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    original,pcm,seconds=preflight(a.source,a.template,a.baseline,a.public_speech_wav)
    if not a.run:
        print(json.dumps({'status':'source_pinned_real_Whisper_ASR_input_preflight_ONLY',
            'audio_sha256':INPUT_SHA,'recorded_public_speech_seconds':seconds,
            'guarded_server_SHA256':PATCHED_SERVER_SHA256,'model_calls':0,
            'no_live_service_or_phone_changed':True}))
        return 0
    if os.getenv(OPTIN)!='1':raise ValueError('Explicit unique real Whisper one-use opt-in missing')
    root=a.private_result
    if (root.exists() or root.is_symlink() or not root.is_absolute()
        or not str(root).startswith('/data/var/swaag-benchmarks/')):
        raise FileExistsError('Unknown or prior real ASR effect; never repeat')
    root.mkdir(parents=True,mode=0o700,exist_ok=False);os.chmod(root,0o700)
    _write_unique(root/'attempt.json',json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_SHA256':PATCHED_SERVER_SHA256,'public_audio_SHA256':INPUT_SHA,
        'unique_real_local_Whisper_model_requests_permitted':1,
        'external_provider_phone_tool_llm_or_tts_calls_permitted':0,
        'no_replay_if_started_or_unknown':True},indent=2).encode())
    start=time.perf_counter()
    try:r=private_runtime_once(a.source,original,pcm,root)
    except BaseException as e:r={'status':'failed_or_unknown','error_type':type(e).__name__,
        'private_error_detail':str(e)[:300]}
    r.update({'elapsed_seconds':round(time.perf_counter()-start,3),
        'source_SHA256':PATCHED_SERVER_SHA256,'public_recorded_audio_SHA256':INPUT_SHA,
        'real_Whisper_endpoint':MODEL_ORIGIN,'actual_Android_owner_phone_or_production_server_changed':False,
        'original_public_recorded_input_duration_sec':seconds,
        'no_external_account_or_tool_effects':True,
        'real_agent_response_generation_or_TTS_used':False})
    _write_unique(root/'result.json',json.dumps(r,indent=2,ensure_ascii=False).encode())
    print(json.dumps({k:r.get(k) for k in ('status','error_type','elapsed_seconds','actual_local_Whisper_model_calls',
        'actual_durable_history_turn_count','manual_speaker_review_required','private_server_pid','asr_text_chars')},sort_keys=True))
    return 0 if r['status']=='real_local_Whisper_ASR_complete_and_two_reconnects_review_only' else 3

if __name__=='__main__':raise SystemExit(main())
