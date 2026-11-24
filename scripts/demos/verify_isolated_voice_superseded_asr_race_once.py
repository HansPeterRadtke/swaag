#!/usr/bin/env python3
"""One-use real private Jetson WS server: ASR late after user cancels/replaces.

This sends a SHA-pinned prior real Phonon recorded PCM to a real separate
server process with all model/TTS/agent URLs inert. A local *fake* ASR relay
intentionally holds a single source audio transcription response until a
concurrent WebSocket cancellation or a replacement client's new manual
recording starts. The real server must not persist stale speech or propose
an action. No real transcription is requested: this is a race test, not an
ASR-accuracy test or a microphone/Android acceptance test.

Each scenario is one unique new conversation with its own private source
files, faked bearer and closed process; any failure is preserved, not
replayed. Original live Jetson, Thor LLM, prior audio and user sessions are
never modified, and no original Send is repeated.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime,timezone
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from verify_isolated_voice_release_server import (
    EXPECTED_SOURCES,inspect,scrub_config,runtime_python,_write_unique,
)
from verify_voice_tts_replay_noaction import validate_inputs, EXPECTED_AUDIO_SHA

OPT_IN='SWAAG_W68_NEW_ISOLATED_SUPERSEDED_ASR_NOACTION'
FAKE_TEXT='This is a local controlled delayed speech result with no authorization.'
FAKE_BEARER='test-only-private-voice-key'


def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()


class HeldASR:
    """One-use local-only endpoint. NEVER connects to Thor or providers."""
    def __init__(self,original_pcm:bytes):
        self.original_pcm=original_pcm
        self.received=threading.Event()
        self.release=threading.Event()
        self.done=threading.Event()
        self.records=[]
        self.httpd=None;self.thread=None

    def start(self)->int:
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_POST(self):
                note={'path':self.path,'time_utc':datetime.now(timezone.utc).isoformat()}
                owner.records.append(note)
                if self.path!='/v1/voice/transcribe-verify' or len(owner.records)!=1:
                    self.send_error(409,'private only one unique delayed ASR');return
                try:
                    n=int(self.headers.get('Content-Length') or '0')
                    if not 48000<n<250000:raise ValueError('Unexpected WAV length')
                    data=self.rfile.read(n)
                    if not data.startswith(b'RIFF') or owner.original_pcm[14000:29000] not in data:
                        raise ValueError('Original SHA-pinned voice not received')
                    note['wav_audio_bytes']=len(data)
                    note['real_recorded_pcm_verified']=True
                    owner.received.set()
                    if not owner.release.wait(timeout=14):raise TimeoutError('Held test response not released')
                    response=json.dumps({'ok':True,'text':FAKE_TEXT,'language':'en',
                        'time_sec':0.4},sort_keys=True).encode()
                    self.send_response(200)
                    self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(response)))
                    self.end_headers();self.wfile.write(response)
                    note['delivered_mock_text_sha256']=hashlib.sha256(FAKE_TEXT.encode()).hexdigest()
                except Exception as e:
                    note['failure']=type(e).__name__
                    try:self.send_error(502,'private held ASR failed')
                    except OSError:pass
                finally:owner.done.set()
        self.httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.httpd.daemon_threads=True
        self.thread=threading.Thread(target=self.httpd.serve_forever,name='single-held-ASR-fixture',daemon=True)
        self.thread.start();return self.httpd.server_address[1]

    def stop(self)->None:
        self.release.set()
        if self.httpd:
            self.httpd.shutdown();self.httpd.server_close()
        if self.thread:self.thread.join(timeout=4)


def private_config(original:dict,root:Path,port:int,relayport:int)->dict:
    c=scrub_config(original,root,port)
    c['asr']['final_quality_enabled']=True
    c['asr']['final_quality_url']='http://127.0.0.1:'+str(relayport)+'/v1/voice/transcribe-verify'
    c['asr']['final_quality_timeout_seconds']=16
    c['interaction']['auto_send_default']=True
    c['interaction']['require_verified_speaker_for_auto_send']=True
    for group in ('asr','llm','tts'):
        for k,v in c[group].items():
            if k.endswith('_url') or k=='url':
                expected=('http://127.0.0.1:'+str(relayport)+'/') if (group=='asr' and k=='final_quality_url') else 'http://127.0.0.1:9/'
                if v and not str(v).startswith(expected):
                    raise AssertionError('Unexpected nonloopback or live model endpoint: '+group+'.'+k)
    return c


async def scenario(port:int,pcm:bytes,relay:HeldASR,conversation:str,kind:str)->dict:
    from websockets.legacy.client import connect
    from websockets.exceptions import ConnectionClosed
    events=[]
    url='ws://127.0.0.1:'+str(port)+'/ws?conversation='+conversation+'&generation='
    headers={'Authorization':'Bearer '+FAKE_BEARER}
    async def until(ws,pred,seconds=7):
        deadline=asyncio.get_running_loop().time()+seconds
        while True:
            left=deadline-asyncio.get_running_loop().time()
            if left<=0:raise TimeoutError('No expected independent server event')
            m=await asyncio.wait_for(ws.recv(),left)
            if not isinstance(m,str):raise AssertionError('Private no-tool WS emitted binary media')
            row=json.loads(m)
            if (row.get('type')=='turn_proposal' or row.get('type')=='audio' or
                (row.get('type')=='turn' and row.get('event')=='submitted')):
                raise AssertionError('Untrusted delayed audio triggered an action/proposal/playback')
            if row.get('type')=='state' and row.get('state')=='checking_turn':
                raise AssertionError('Unknown speaker reached semantic classifier')
            events.append({k:row.get(k) for k in ('type','event','state','reason','turn_id') if k in row})
            if pred(row):return row
    async def survey(ws,duration=1.2):
        done=asyncio.get_running_loop().time()+duration
        while done>asyncio.get_running_loop().time():
            try:await until(ws,lambda _x:False,seconds=min(.2,done-asyncio.get_running_loop().time()))
            except (TimeoutError,asyncio.TimeoutError):continue
            except ConnectionClosed:break
    async with connect(url+'1',extra_headers=headers,close_timeout=1,max_size=2**20) as first:
        hello=await until(first,lambda m:m.get('type')=='hello')
        if hello.get('unverified_speaker_requires_review') is not True:
            raise AssertionError('Private patched candidate did not advertise guard')
        await first.send(json.dumps({'type':'client_state','build':'Voice Agent 1.7.16',
            'auto_send':True,'auto_transcribe':True,'auto_commit_supported':False,
            'speaker_verified':True,'owner_speaker_identity':'trusted'}))
        frame_size=640
        for i in range(0,len(pcm),frame_size):
            part=pcm[i:i+frame_size]
            await first.send(part+b'\0'*(frame_size-len(part)))
        for _ in range(35):await first.send(b'\0'*frame_size)
        await asyncio.wait_for(first.ping(),4)
        await until(first,lambda m:m.get('type')=='state' and m.get('state')=='transcribing',7)
        is_active=await asyncio.get_running_loop().run_in_executor(None,relay.received.wait,8)
        if not is_active:raise AssertionError('Real server never issued held local ASR request')
        if kind=='cancel':
            await first.send(json.dumps({'type':'cancel','reason':'wave68_user_cancel_during_asr'}))
            stopped=await until(first,lambda m:m.get('type')=='state' and m.get('reason')=='wave68_user_cancel_during_asr',8)
            relay.release.set()
            finished=await asyncio.get_running_loop().run_in_executor(None,relay.done.wait,8)
            if not finished:raise AssertionError('Local held ASR did not return')
            await survey(first,1.5)
            await first.send(json.dumps({'type':'history_get'}))
            history=await until(first,lambda m:m.get('type')=='history',8)
            if history.get('turns')!=[]:
                raise AssertionError('Cancel during held ASR left stale turn in history')
        elif kind=='new_manual_on_reconnect':
            # Replacing the websocket intentionally supersedes the first
            # session while its ASR is pending in another thread.
            async with connect(url+'2',extra_headers=headers,close_timeout=1,max_size=2**20) as newer:
                await until(newer,lambda m:m.get('type')=='hello',7)
                await until(newer,lambda m:m.get('type')=='history',7)
                await newer.send(json.dumps({'type':'begin_manual_utterance'}))
                await until(newer,lambda m:m.get('type')=='state' and
                   m.get('state')=='transcribing' and m.get('source')=='manual_recording',7)
                relay.release.set()
                finished=await asyncio.get_running_loop().run_in_executor(None,relay.done.wait,8)
                if not finished:raise AssertionError('Local held ASR did not return')
                await survey(newer,1.4)
                await newer.send(json.dumps({'type':'history_get'}))
                history=await until(newer,lambda m:m.get('type')=='history',8)
                if history.get('turns')!=[]:
                    # Real candidate bug: old superseded ASR clobbers new
                    # manual recording or leaves an unwanted old turn.
                    raise AssertionError('Superseded late ASR inserted a stale draft during newer manual recording')
        else:raise ValueError('Invalid isolated race scenario')
    return {'kind':kind,'status':'no_stale_draft_after_cancel_or_new_manual',
        'held_authoritative_ASR_requests':len(relay.records),
        'saw_original_PCM_in_real_server_request':relay.records[0].get('real_recorded_pcm_verified') is True,
        'new_real_ASR_model_requests':0,'manual_tool_or_provider_actions':0,
        'events_count':len(events),'last_history_turns':len(history['turns'])}


def run_private_once(source:Path,original:dict,pcm:bytes,out:Path,kind:str)->dict:
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    relay=HeldASR(pcm);rport=relay.start();child=None
    try:
        c=private_config(original,out,port,rport)
        _write_unique(out/'fake-only-bearer',(FAKE_BEARER+'\n').encode())
        _write_unique(out/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
        _write_unique(out/'inert-piper.onnx',b'no-model')
        _write_unique(out/'inert-piper.onnx.json',b'{}')
        _write_unique(out/'private-config.json',json.dumps(c,sort_keys=True).encode())
        with (out/'private-child-console.log').open('xb') as sink:
            child=subprocess.Popen([runtime_python(),'-u',str(source/'cloudflare_ws_voice_server.py'),
             '--config',str(out/'private-config.json')],cwd=source,
             stdin=subprocess.DEVNULL,stdout=sink,stderr=subprocess.STDOUT,
             start_new_session=True,close_fds=True,
             env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(source)})
            for _ in range(55):
                if child.poll() is not None:
                    raise RuntimeError('Private child exited prematurely')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.2):break
                except OSError:time.sleep(.1)
            else:raise TimeoutError('Private source server never bound')
            answer=asyncio.run(asyncio.wait_for(scenario(port,pcm,relay,
                 'w68-'+kind+'-controlled',kind),timeout=46))
            answer['actual_private_child_pid']=child.pid
            return answer
    finally:
        relay.stop()
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait(timeout=4)
            if child.poll() is None:raise RuntimeError('Own private server child did not stop')
        for p in (out/'private-child-console.log',out/'private-config.json'):
            if p.exists():os.chmod(p,0o600)


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','helpers','template','receipt','audio','evidence'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scenario',choices=['cancel','new_manual_on_reconnect'],required=True)
    p.add_argument('--run',action='store_true')
    x=p.parse_args(argv)
    orig=inspect(x.source,x.template)
    pcm,reply,_=validate_inputs(x.source,x.helpers,x.receipt,x.audio)
    if len(pcm)<40000 or len(reply)<20:raise ValueError('Original real sampled input unavailable')
    if not x.run:
        print(json.dumps({'status':'source_pinned_delayed_ASR_race_preflight_no_actions',
          'source_sha256':EXPECTED_SOURCES['cloudflare_ws_voice_server.py'],
          'audio_sha256':EXPECTED_AUDIO_SHA,'input_bytes':len(pcm),
          'scenario':x.scenario,'model_calls':0}))
        return 0
    if os.getenv(OPT_IN)!='1':raise ValueError('Explicit one-use scenario opt-in required')
    if x.evidence.is_symlink() or x.evidence.exists():
        raise FileExistsError('Private one-use evidence already exists, never retry')
    x.evidence.mkdir(parents=True,mode=0o700,exist_ok=False)
    os.chmod(x.evidence,0o700)
    _write_unique(x.evidence/'attempt.json',json.dumps({
      'utc':datetime.now(timezone.utc).isoformat(),
      'scenario':x.scenario,'source_sha256':EXPECTED_SOURCES['cloudflare_ws_voice_server.py'],
      'audio_sha256':EXPECTED_AUDIO_SHA,'one_use_only':True,
      'real_ASR_model_calls':0,'live_server_or_phone_untouched':True}).encode())
    t=time.monotonic()
    try:r=run_private_once(x.source,orig,pcm,x.evidence,x.scenario)
    except BaseException as e:
        r={'status':'failed_or_unknown','failure_type':type(e).__name__,
            'failure_detail_private':str(e)[:300]}
    r.update({'elapsed_seconds':round(time.monotonic()-t,3),
      'source_sha256':EXPECTED_SOURCES['cloudflare_ws_voice_server.py'],
      'input_recording_SHA256':EXPECTED_AUDIO_SHA,
      'actual_Jetson_production_service_changed':False,
      'private_dummy_bearer_only':True,'real_model_ASR_TTS_agent_tool_calls':0,
      'actual_owner_or_Android_phone_used':False})
    _write_unique(x.evidence/'result.json',json.dumps(r,indent=2).encode())
    print(json.dumps({k:r.get(k) for k in ('status','failure_type','elapsed_seconds',
          'last_history_turns','held_authoritative_ASR_requests','actual_private_child_pid')},
          sort_keys=True))
    return 0 if r['status']=='no_stale_draft_after_cancel_or_new_manual' else 3

if __name__=='__main__':raise SystemExit(main())
