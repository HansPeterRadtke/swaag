#!/usr/bin/env python3
"""Real private voice server + delayed original framed media fixture.

Exercises stop during an unfinished TTS transport, not new speech synthesis.
Both HTTP fixtures use exact earlier public-test model output; no real model
or agent endpoints are available. Stop precedes stream end and no playback ACK.
"""
from __future__ import annotations
import argparse
import asyncio
import http.server
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

from verify_private_voice_stop_reconnect import (
    inspect, sha, json_bytes, ROOT_PREFIX, BEARER, _write_unique, runtime_python,
)
from verify_private_voice_durable_stop_followup import CANDIDATE
from verify_isolated_voice_release_server import scrub_config
from verify_integrated_voice_chain_manual_once import framed_pcm

CONV='w100-private-stop-pending-media'
INPUT='Private fixture playback, no action.'
CODER_SHA='a33a6523a019550520a54579cd858c523d99d7dbe43113044bb2fb4f51c3bb4d'
TTS_SHA='519a46e88e5d1290feb28db83b69cee27557427ef6b08e60b5d6870e4e2c1e22'


def require(ok,msg):
    if not ok:raise AssertionError(msg)


class HeldMedia:
    def __init__(self,root,reply,framed):
        self.root=root;self.reply=reply;self.framed=framed
        self.release=threading.Event();self.sent_first=threading.Event();self.done=threading.Event()
        self.count={'reply_fixture':0,'media_fixture':0};self.errors=[];self.turn=None;self.server=None
        self.lock=threading.Lock()

    def start(self):
        outer=self
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):self.send_error(404)
            def do_POST(self):
                kind={'/v1/voice/run':'reply_fixture','/synthesize-stream':'media_fixture'}.get(self.path)
                if not kind:
                    outer.errors.append('unexpected request '+self.path);self.send_error(403);return
                try:
                    n=int(self.headers.get('Content-Length','0'))
                    require(0<n<10000,'unexpected fixture body')
                    body=json.loads(self.rfile.read(n))
                    with outer.lock:
                        require(outer.count[kind]==0,'duplicate fixture request')
                        outer.count[kind]+=1
                    _write_unique(outer.root/(kind+'-request.json'),json_bytes(body))
                    if kind=='reply_fixture':
                        require(body['conversation_id']==CONV and body['text']==INPUT,'wrong private test turn')
                        outer.turn=body['turn_id']
                        op='voice-'+sha((CONV+'\0'+outer.turn).encode())
                        events=[{'type':'RUN_STARTED','operation_id':op},
                           {'type':'TEXT_MESSAGE_START','messageId':'fixture-reply','role':'assistant'},
                           {'type':'TEXT_MESSAGE_CONTENT','messageId':'fixture-reply','delta':outer.reply},
                           {'type':'TEXT_MESSAGE_END','messageId':'fixture-reply'},
                           {'type':'RUN_FINISHED','result':outer.reply}]
                        raw=b''.join(('data: '+json.dumps(e)+'\n\n').encode() for e in events)
                        self.send_response(200);self.send_header('Content-Type','text/event-stream')
                        self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
                    else:
                        require(body=={'text':outer.reply,'sample_rate':16000},'not original retained public response')
                        split=4+int.from_bytes(outer.framed[:4],'big')
                        self.send_response(200);self.send_header('Content-Type','application/octet-stream')
                        self.send_header('Content-Length',str(len(outer.framed)));self.end_headers()
                        self.wfile.write(outer.framed[:split]);self.wfile.flush()
                        outer.sent_first.set()
                        require(outer.release.wait(20),'test never released remaining media')
                        self.wfile.write(outer.framed[split:]);self.wfile.flush();outer.done.set()
                except Exception as e:
                    outer.errors.append(type(e).__name__+': '+str(e)[:160])
                    try:self.send_error(502)
                    except OSError:pass
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=False
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        return self.server.server_address[1]

    def close(self):
        self.release.set()
        if self.server:
            self.server.shutdown();self.server.server_close();self.thread.join(5)


async def scenario(port,relay,root,pcm):
    from websockets.legacy.client import connect
    url='ws://127.0.0.1:'+str(port)+'/ws?conversation='+CONV+'&generation='
    headers={'Authorization':'Bearer '+BEARER}
    async def recv(ws,timeout):
        x=await asyncio.wait_for(ws.recv(),timeout)
        if isinstance(x,bytes):return x
        d=json.loads(x)
        with (root/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
        require(d.get('type') not in ('error','turn_proposal','asr'),'unexpected error/action proposal')
        return d
    async def until(ws,typ,event=None,timeout=15):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            x=await recv(ws,max(.01,end-time.monotonic()))
            if isinstance(x,dict) and x.get('type')==typ and (event is None or x.get('event')==event):return x
        raise TimeoutError(typ)
    async with connect(url+'1',extra_headers=headers,close_timeout=1,max_size=2**20) as ws:
        await until(ws,'hello');h=await until(ws,'history');require(not h['turns'],'not fresh private test')
        await ws.send(json.dumps({'type':'text_turn','text':INPUT,'client_request_id':'a'*32}))
        sub=await until(ws,'turn','submitted')
        require(sub['automatic'] is False,'not explicit private turn')
        start=await until(ws,'audio','start')
        require(start.get('streaming') is True and start['turn_id']==relay.turn,'not live media path')
        while True:
            x=await recv(ws,8)
            if isinstance(x,bytes):
                require(pcm[:len(x)]==x,'first PCM differs');initial=len(x);break
            require(not (x.get('type')=='audio' and x.get('event')=='end'),'premature stream end')
        require(relay.sent_first.is_set() and not relay.done.is_set(),'media was not held')
        await ws.send(json.dumps({'type':'stop_audio'}))
        stop=await until(ws,'audio','cancel')
        require(stop['reason']=='playback_stopped' and stop['stream_id']==start['stream_id'],'wrong stop acknowledgement')
        require(not relay.done.is_set(),'media completed before stop')
        async with connect(url+'2',extra_headers=headers,close_timeout=1,max_size=2**20) as replacement:
            await until(replacement,'hello')
            h=await until(replacement,'history')
            require(len(h['turns'])==1 and h['turns'][0].get('audio_stopped') is True,'stop not durable during unfinished stream')
            relay.release.set()
            path=root/'conversations'/(CONV+'.json')
            deadline=time.monotonic()+12;row=None
            while time.monotonic()<deadline:
                try:
                    row=json.loads(path.read_text())['dialogue_history'][0]
                except (OSError,ValueError,IndexError):row=None
                if row and row.get('assistant_audio_path') and row.get('status')=='completed':break
                try:
                    x=await recv(replacement,.2)
                    require(not isinstance(x,bytes) and not (x.get('type')=='audio' and x.get('event')=='start'),
                            'stopped pending stream restarted')
                except asyncio.TimeoutError:pass
            require(row and row.get('status')=='completed' and row['audio_stopped'] is True
                    and row['audio_completed'] is False and row['audio_transmitted'] is False,
                    'late generated answer did not preserve durable stop')
            path=Path(row['assistant_audio_path'])
            require(path.resolve().parent.parent== (root/'reply-audio').resolve(),'saved media outside test root')
            import wave
            with wave.open(str(path),'rb') as f:actual=f.readframes(f.getnframes())
            require(actual==pcm,'late final WAV does not match original media')
            end=time.monotonic()+1
            while time.monotonic()<end:
                try:x=await recv(replacement,max(.01,end-time.monotonic()))
                except asyncio.TimeoutError:break
                require(not isinstance(x,bytes) and not (x.get('type')=='audio' and x.get('event')=='start'),
                        'late automatic media restart')
    require(relay.done.is_set() and relay.count=={'reply_fixture':1,'media_fixture':1} and not relay.errors,
            'fixture delivery incomplete or repeated')
    return {'stop_ack_before_final_frame':True,'initial_pcm_bytes':initial,
            'full_retained_pcm_bytes':len(pcm),'original_media_saved_after_stop':True,
            'no_reconnect_restart':True,'durable_playback_stopped_not_completed':True,
            'real_upstream_model_or_agent_calls':0,'local_fixture_requests':relay.count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','template','audio','prior','evidence'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    orig,_,pcm=inspect(a.source,a.template,a.audio,CANDIDATE)
    model=a.prior/'coder-raw.json';framed=a.prior/'tts-framed.bin'
    require(not model.is_symlink() and not framed.is_symlink(),'unsafe fixture')
    raw=model.read_bytes();frames=framed.read_bytes()
    require(sha(raw)==CODER_SHA and sha(frames)==TTS_SHA and framed_pcm(frames)==pcm,'original model/media proof changed')
    reply=json.loads(raw)['choices'][0]['message']['content']
    if not a.run:print(json.dumps({'preflight':True,'original_media_bytes':len(pcm)}));return 0
    if os.getenv('SWAAG_W100_PENDING_MEDIA_STOP')!='1':raise ValueError('explicit opt-in')
    if a.evidence.exists() or a.evidence.is_symlink() or not str(a.evidence.resolve()).startswith(ROOT_PREFIX):
        raise ValueError('unique private result root')
    root=a.evidence;root.mkdir(mode=0o700,parents=False)
    _write_unique(root/'attempt.json',json_bytes({'source_sha':CANDIDATE,'original_coder_sha':CODER_SHA,
        'original_framed_media_sha':TTS_SHA,'real_model_or_agent_requests':0,'no_lost_ACK_test':True}))
    relay=HeldMedia(root,reply,frames);child=None;t=time.monotonic();r={'status':'failed_or_unknown'}
    try:
        rp=relay.start()
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        c=scrub_config(orig,root,port);base='http://127.0.0.1:'+str(rp)
        c['llm']['url']=base+'/v1/voice/run'
        c['tts'].update(service_url=base+'/synthesize',retry_count=1)
        for group in ('asr','llm','tts'):
            for k,v in c[group].items():
                if (k.endswith('_url') or k=='url') and v:
                    require(str(v).startswith(('http://127.0.0.1:9/',base+'/')),'nonprivate route')
        _write_unique(root/'fake-only-bearer',(BEARER+'\n').encode())
        _write_unique(root/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
        _write_unique(root/'inert-piper.onnx',b'no-test-model')
        _write_unique(root/'inert-piper.onnx.json',b'{}')
        _write_unique(root/'private-config.json',json_bytes(c))
        with (root/'child-console.log').open('xb') as out:
            child=subprocess.Popen([runtime_python(),'-u',str(a.source/'cloudflare_ws_voice_server.py'),
                '--config',str(root/'private-config.json')],cwd=str(a.source),stdin=subprocess.DEVNULL,
                stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(a.source)})
            for _ in range(80):
                if child.poll() is not None:raise RuntimeError('private child exited')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.1):break
                except OSError:time.sleep(.1)
            else:raise RuntimeError('private child failed to listen')
            r.update(asyncio.run(asyncio.wait_for(scenario(port,relay,root,pcm),55)))
        r['status']='verified_private_stop_while_pending_media_transport'
    except BaseException as e:r.update(error_type=type(e).__name__,error=str(e)[:240])
    finally:
        relay.release.set()
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(5)
            except subprocess.TimeoutExpired:child.kill();child.wait(5)
        relay.close()
        r.update(child_stopped=child is None or child.poll() is not None,
                 elapsed_seconds=round(time.monotonic()-t,3),fixture_errors=relay.errors,
                 real_model_agent_phone_calls=0)
        _write_unique(root/'result.json',json_bytes(r));print(json.dumps(r))
    return 0 if r['status'].startswith('verified_private') else 3

if __name__=='__main__':raise SystemExit(main())
