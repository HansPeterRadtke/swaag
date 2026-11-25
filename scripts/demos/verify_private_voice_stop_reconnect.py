#!/usr/bin/env python3
"""Private real-process stop-before-stream-end test; NO model or agent calls.

Uses a new synthetic conversation containing a checksum-verified previously
generated public-test reply WAV. Tests explicit playback stop, not lost ACK.
No microphone, production endpoints, live conversations or external writes.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import wave

from verify_isolated_voice_release_server import (
    EXPECTED_SOURCES, EXPECTED_TEMPLATE_SHA, scrub_config, _write_unique, runtime_python,
)

BASE = '873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf'
WAV_SHA = 'f8da0c638b6771faac445d831cdae16cd4315ecd86db0a63fa22e1290666dc4f'
ROOT_PREFIX = '/data/var/swaag-benchmarks/owner-serial-000100-20261125/'
BEARER = 'test-only-private-voice-key'
TURN = 'turn-private-explicit-stop'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def json_bytes(value):
    return json.dumps(value, indent=2, ensure_ascii=False).encode()


def inspect(source, template, audio, expected):
    if source.is_symlink() or template.is_symlink() or audio.is_symlink():
        raise ValueError('symlinked source')
    for name,h in EXPECTED_SOURCES.items():
        p = source/name
        want = expected if name == 'cloudflare_ws_voice_server.py' else h
        if p.is_symlink() or not p.is_file() or sha(p.read_bytes()) != want:
            raise ValueError('source drift: '+name)
    if sha(template.read_bytes()) != EXPECTED_TEMPLATE_SHA:
        raise ValueError('template drift')
    raw = audio.read_bytes()
    if sha(raw) != WAV_SHA:
        raise ValueError('source WAV changed')
    with wave.open(str(audio), 'rb') as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()) != (1,2,16000,87040):
            raise ValueError('source WAV format changed')
        pcm = f.readframes(f.getnframes())
    return json.loads(template.read_text()), raw, pcm


def make_private_config(orig, root, port):
    c = scrub_config(orig,root,port)
    for group in ('asr','llm','tts'):
        for k,v in c[group].items():
            if (k.endswith('_url') or k=='url') and v and not str(v).startswith('http://127.0.0.1:9/'):
                raise ValueError('unexpected model or agent route')
    if c['worker_events']['host']!='127.0.0.1' or c['worker_events']['port']!=9:
        raise ValueError('nonprivate worker event route')
    return c


def seed(root, name, raw):
    (root/'conversations').mkdir(mode=0o700)
    (root/'reply-audio').mkdir(mode=0o700)
    path = root/'reply-audio'/'previous-public-test-reply.wav'
    _write_unique(path,raw)
    row = {'turn_id': TURN, 'user':'Synthetic private playback fixture',
           'recognized_text':'Synthetic private playback fixture', 'submitted_text':'Synthetic private playback fixture',
           'assistant':'The prior public recording is retained for this private media test.',
           'status':'ready','input_source':'typed','assistant_audio_path':str(path),
           'audio_started':False,'audio_completed':False,'audio_transmitted':False,'t':time.time()}
    _write_unique(root/'conversations'/(name+'.json'),json_bytes({'dialogue_history':[row]}))


async def scenario(port, name, root, pcm):
    from websockets.legacy.client import connect
    url = 'ws://127.0.0.1:'+str(port)+'/ws?conversation='+name+'&generation='
    headers = {'Authorization':'Bearer '+BEARER}
    events = []
    stop_ack = None
    before_bytes = 0
    reconnect_bytes = 0

    async def recv(ws, timeout):
        data = await asyncio.wait_for(ws.recv(),timeout)
        if isinstance(data,bytes):
            return data
        obj=json.loads(data)
        events.append(obj)
        with (root/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(obj,ensure_ascii=False)+'\n')
        if obj.get('type') in ('turn_proposal','turn') or obj.get('type')=='asr':
            raise AssertionError('unexpected voice/agent submission')
        return obj

    async def until(ws, typ, event=None, timeout=8):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            d=await recv(ws,max(.01,end-time.monotonic()))
            if isinstance(d,dict) and d.get('type')==typ and (event is None or d.get('event')==event):
                return d
        raise TimeoutError(typ)

    async with connect(url+'1',extra_headers=headers,close_timeout=1,max_size=2**20) as first:
        h=await until(first,'hello')
        if not h.get('unverified_speaker_requires_review'):
            raise AssertionError('speaker gate absent')
        await until(first,'history')
        start=await until(first,'audio','start')
        if start.get('turn_id')!=TURN:
            raise AssertionError('wrong private replay')
        while not before_bytes:
            d=await recv(first,8)
            if isinstance(d,bytes):
                before_bytes=len(d)
                if pcm[:before_bytes]!=d:
                    raise AssertionError('input PCM mismatch')
            elif d.get('type')=='audio' and d.get('event')=='end':
                raise AssertionError('stream ended before stop')
        # No playback_completed acknowledgement is ever sent. The user
        # explicitly stops a still-in-progress stream; this is not lost ACK.
        await first.send(json.dumps({'type':'stop_audio'}))
        stop_ack=await until(first,'audio','cancel')
        if stop_ack.get('reason')!='playback_stopped':
            raise AssertionError('stop not acknowledged')
        # Replace immediately after receipt of stop, while the old playback
        # coroutine may still be unwinding. This is the real reconnect race.
        async with connect(url+'2',extra_headers=headers,close_timeout=1,max_size=2**20) as second:
            await until(second,'hello')
            await until(second,'history')
            deadline=time.monotonic()+1.4
            replay_start=None
            while time.monotonic()<deadline:
                try:d=await recv(second,max(.01,deadline-time.monotonic()))
                except asyncio.TimeoutError:break
                if isinstance(d,bytes):
                    reconnect_bytes+=len(d)
                elif d.get('type')=='audio' and d.get('event')=='start':
                    replay_start=d
                    # Do not let an unexpected resumed stream finish.
                    break
            await second.send(json.dumps({'type':'history_get'}))
            await until(second,'history')
    raw=json.loads((root/'conversations'/(name+'.json')).read_text())
    row=raw['dialogue_history'][0]
    return {'stop_acknowledged_before_disconnect':True,
            'initial_media_bytes':before_bytes,'reconnected_media_bytes':reconnect_bytes,
            'automatic_restart_observed':replay_start is not None or reconnect_bytes>0,
            'stop_stream_id':stop_ack.get('stream_id'),
            'resumed_stream_id':(replay_start or {}).get('stream_id'),
            'durable_audio_stopped':row.get('audio_stopped',False),
            'durable_audio_completed':row.get('audio_completed'),
            'durable_turn_status':row.get('status'),
            'original_audio_unchanged':sha((root/'reply-audio'/'previous-public-test-reply.wav').read_bytes())==WAV_SHA,
            'new_agent_or_model_calls':0,'playback_completed_ACK_sent':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','template','audio','evidence'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--source-sha',default=BASE)
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    orig,raw,pcm=inspect(a.source,a.template,a.audio,a.source_sha)
    if not a.run:
        print(json.dumps({'preflight':True,'source':a.source_sha,'source_audio':WAV_SHA}))
        return 0
    if os.getenv('SWAAG_W100_PRIVATE_STOP_TEST')!='1':
        raise ValueError('explicit one-use opt-in needed')
    if not str(a.evidence.resolve()).startswith(ROOT_PREFIX) or a.evidence.exists() or a.evidence.is_symlink():
        raise ValueError('unique private output root required')
    root=a.evidence;root.mkdir(mode=0o700,parents=False)
    name='w100-private-stop-'+root.name
    _write_unique(root/'attempt.json',json_bytes({'source_sha':a.source_sha,'audio_sha':WAV_SHA,
        'one_private_process':True,'no_real_agent_or_model_calls':True,'no_lost_ACK_test':True}))
    seed(root,name,raw)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    c=make_private_config(orig,root,port)
    _write_unique(root/'fake-only-bearer',(BEARER+'\n').encode())
    _write_unique(root/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
    _write_unique(root/'inert-piper.onnx',b'no-test-model')
    _write_unique(root/'inert-piper.onnx.json',b'{}')
    _write_unique(root/'private-config.json',json_bytes(c))
    child=None;result={'status':'failed_or_unknown'};t=time.monotonic()
    try:
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
            result.update(asyncio.run(asyncio.wait_for(scenario(port,name,root,pcm),30)))
            result['status']='observed_private_stop_reconnect'
    except BaseException as e:
        result.update(error_type=type(e).__name__,error=str(e)[:240])
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(5)
            except subprocess.TimeoutExpired:child.kill();child.wait(5)
        result.update(child_stopped=child is None or child.poll() is not None,
                      elapsed_seconds=round(time.monotonic()-t,3))
        _write_unique(root/'result.json',json_bytes(result))
        print(json.dumps(result))
    return 0 if result['status']=='observed_private_stop_reconnect' else 3

if __name__=='__main__':raise SystemExit(main())
