#!/usr/bin/env python3
"""Private candidate: restart with a stopped reply, explicit replay, normal recovery.

No upstream requests, physical playback, production user state or lost-ACK test.
The stored source is the new W100 stopped-before-end private fixture only.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace

from verify_private_voice_stop_reconnect import (
    inspect, make_private_config, seed, sha, json_bytes, ROOT_PREFIX,
    WAV_SHA, TURN, BEARER, _write_unique, runtime_python,
)

CANDIDATE='cd7ccfa7019e4dd63a2ed25e01ba257f405ebaffe3a415364f1724e931b134ea'
STOP_RESULT='6520fe818e8672eb5e0fdcba7c16d50b36d547ea90ff2dc97c562a13fb273e0f'
OLD_CONV='w100-private-stop-corrected-stop-before-end'
STOPPED='w100-restarted-stopped-private'
NORMAL='w100-normal-uninterrupted-recovery-private'


def require(ok, why):
    if not ok: raise AssertionError(why)


def read_prior(prior):
    p=prior/'result.json'
    if p.is_symlink() or sha(p.read_bytes())!=STOP_RESULT:
        raise ValueError('original corrected stop result changed')
    r=json.loads(p.read_bytes())
    require(r['stop_acknowledged_before_disconnect'] is True
            and r['durable_audio_stopped'] is True
            and r['automatic_restart_observed'] is False
            and r['playback_completed_ACK_sent'] is False,'not original stopped fixture')
    h=prior/'conversations'/(OLD_CONV+'.json')
    if h.is_symlink(): raise ValueError('unsafe original history')
    rows=json.loads(h.read_text())['dialogue_history']
    require(len(rows)==1,'original stopped row count changed')
    row=rows[0]
    require(row['turn_id']==TURN and row['status']=='completed'
            and row['audio_stopped'] is True and row['audio_completed'] is False
            and row['audio_stopped_stream_id']==r['stop_stream_id']
            and row['audio_stream_id']==r['stop_stream_id'],'original stop identity changed')
    return row


def configure(root, orig, port, raw, prior):
    cfg=make_private_config(orig,root,port)
    seed(root,NORMAL,raw)
    row=dict(read_prior(prior))
    row['assistant_audio_path']=str(root/'reply-audio'/'previous-public-test-reply.wav')
    _write_unique(root/'conversations'/(STOPPED+'.json'),json_bytes({'dialogue_history':[row]}))
    _write_unique(root/'fake-only-bearer',(BEARER+'\n').encode())
    _write_unique(root/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
    _write_unique(root/'inert-piper.onnx',b'no-test-model')
    _write_unique(root/'inert-piper.onnx.json',b'{}')
    _write_unique(root/'private-config.json',json_bytes(cfg))


def source_storage_checks(source, root):
    """Use real source functions and durable storage, without any sockets/model.

    Separate test conversation only. A stale socket, mismatched stream and
    completed playback must not be overwritten by a stop. Late transport
    bookkeeping may not resurrect a stopped reply.
    """
    import sys
    sys.path.insert(0,str(source))
    import cloudflare_ws_voice_server as server
    require(sha(Path(server.__file__).read_bytes())==CANDIDATE,'wrong imported source')
    server.CONFIG=server.load_voice_ws_config(str(root/'private-config.json'))
    server.CORE=server.VoiceCore(server.CONFIG)
    conv='w100-pure-storage-stop-boundary'
    try:
        server.record_draft(conv,TURN,'test','test','','typed')
        server.update_turn(conv,TURN,status='speaking',assistant='test answer',
            assistant_audio_path=str(root/'reply-audio'/'previous-public-test-reply.wav'),
            audio_stream_id='bound-stream',audio_generation=7,audio_transmitted=True,
            audio_completed=False)
        active=SimpleNamespace(closed=False,superseded=False)
        old=SimpleNamespace(closed=False,superseded=False)
        server.ACTIVE_BY_CONVERSATION[conv]=active
        require(not server.persist_audio_stop(conv,TURN,'bound-stream',7,old),'old socket stopped newer media')
        require(not server.persist_audio_stop(conv,TURN,'wrong-stream',7,active),'wrong stream stopped')
        require(not server.persist_audio_stop(conv,TURN,'bound-stream',6,active),'wrong generation stopped')
        require(not server.get_turn(conv,TURN).get('audio_stopped'),'invalid stop modified durable state')
        server.update_turn(conv,TURN,audio_completed=True)
        require(server.persist_audio_stop(conv,TURN,'bound-stream',7,active),'completed receipt check failed')
        require(server.get_turn(conv,TURN)['audio_completed'] is True
                and not server.get_turn(conv,TURN).get('audio_stopped'),'stop erased genuine completion')
        server.update_turn(conv,TURN,audio_completed=False)
        require(server.persist_audio_stop(conv,TURN,'bound-stream',7,active),'valid stop not stored')
        server.update_turn(conv,TURN,status='ready',audio_completed=True,audio_transmitted=True)
        require(not server.update_turn(conv,TURN,status='speaking',_require_unstopped_audio=True),'stopped stream re-admitted')
        r=server.get_turn(conv,TURN)
        require(r['audio_stopped'] is True and r['audio_completed'] is False
                and r['audio_transmitted'] is False and r['status']=='completed'
                and server.latest_ready_turn(conv) is None,'late update resurrected stopped media')
        return {'stale_socket_rejected':True,'wrong_stream_rejected':True,
                'wrong_generation_rejected':True,'genuine_ack_preserved':True,
                'late_ready_and_ack_cannot_resurrect':True}
    finally:
        server.ACTIVE_BY_CONVERSATION.pop(conv,None)
        server.CORE.close()


async def scenario(port, root, pcm):
    from websockets.legacy.client import connect
    headers={'Authorization':'Bearer '+BEARER}
    def url(conv,gen):return 'ws://127.0.0.1:'+str(port)+'/ws?conversation='+conv+'&generation='+str(gen)
    async def recv(ws,timeout):
        d=await asyncio.wait_for(ws.recv(),timeout)
        if isinstance(d,bytes):return d
        row=json.loads(d)
        with (root/'followup-events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        require(row.get('type') not in ('turn','turn_proposal','asr','error'),'unexpected action/error')
        return row
    async def until(ws,typ,event=None,timeout=12):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            x=await recv(ws,max(.01,end-time.monotonic()))
            if isinstance(x,dict) and x.get('type')==typ and (event is None or x.get('event')==event):return x
        raise TimeoutError(typ)
    async def history(ws):
        await ws.send(json.dumps({'type':'history_get'}))
        h=await until(ws,'history')
        require(len(h['turns'])==1,'unexpected conversation count')
        return h['turns'][0]
    async def media(ws,typ):
        end=time.monotonic()+15;chunks=[]
        while time.monotonic()<end:
            x=await recv(ws,max(.01,end-time.monotonic()))
            if isinstance(x,bytes):chunks.append(x)
            elif x.get('type')==typ and x.get('event')=='end':
                return b''.join(chunks),x
        raise TimeoutError('no media end')
    # This really is a new process with the old stopped result copied into
    # private test storage, not just a fresh in-memory output_muted flag.
    async with connect(url(STOPPED,1),extra_headers=headers,close_timeout=1,max_size=2**20) as ws:
        await until(ws,'hello');h=await until(ws,'history');row=h['turns'][0]
        require(row.get('audio_stopped') is True and row['audio_completed'] is False,'stopped state not restored')
        end=time.monotonic()+1.0
        while time.monotonic()<end:
            try:x=await recv(ws,max(.01,end-time.monotonic()))
            except asyncio.TimeoutError:break
            require(not isinstance(x,bytes) and not (x.get('type')=='audio' and x.get('event')=='start'),
                    'stopped audio replayed automatically after process restart')
        await ws.send(json.dumps({'type':'replay_assistant','turn_id':TURN}))
        await until(ws,'replay','start')
        b,end=await media(ws,'replay')
        require(b==pcm,'explicit replay differs from original retained WAV')
        row=await history(ws)
        require(row.get('audio_stopped') is True and row['audio_completed'] is False,
                'explicit historical replay rewrote original delivery state')
    # Separate new ordinary pre-end disconnect: no stop, no prior playback ACK.
    async with connect(url(NORMAL,1),extra_headers=headers,close_timeout=1,max_size=2**20) as ws:
        await until(ws,'hello');await until(ws,'history')
        first=await until(ws,'audio','start')
        while True:
            x=await recv(ws,8)
            if isinstance(x,bytes):
                require(x==pcm[:len(x)],'wrong initial media');break
            require(not (x.get('type')=='audio' and x.get('event')=='end'),'ended before disconnect')
    async with connect(url(NORMAL,2),extra_headers=headers,close_timeout=1,max_size=2**20) as ws:
        await until(ws,'hello');await until(ws,'history')
        start=await until(ws,'audio','start')
        require(start.get('resumed') is True and start['stream_id']!=first['stream_id'],'normal recovery lost')
        b,end=await media(ws,'audio')
        require(b==pcm and start['stream_id']==end['stream_id'],'ordinary recovered media corrupt')
        await ws.send(json.dumps({'type':'playback_completed','turn_id':TURN,
            'generation':end['generation'],'stream_id':end['stream_id']}))
        row=await history(ws)
        require(row['audio_completed'] is True and not row.get('audio_stopped'),'normal ACK not retained')
    return {'process_restart_suppresses_stopped_audio':True,'explicit_replay_bytes':len(pcm),
            'explicit_replay_preserves_original_stop':True,'normal_pre_end_disconnect_recovery_bytes':len(pcm),
            'normal_ack_persisted':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','template','audio','prior','evidence'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    orig,raw,pcm=inspect(a.source,a.template,a.audio,CANDIDATE)
    read_prior(a.prior)
    if not a.run:print(json.dumps({'preflight':True}));return 0
    if os.getenv('SWAAG_W100_PRIVATE_STOP_FOLLOWUP')!='1':raise ValueError('explicit opt-in')
    if a.evidence.exists() or a.evidence.is_symlink() or not str(a.evidence.resolve()).startswith(ROOT_PREFIX):
        raise ValueError('new private root required')
    root=a.evidence;root.mkdir(mode=0o700,parents=False)
    _write_unique(root/'attempt.json',json_bytes({'source_sha':CANDIDATE,'audio_sha':WAV_SHA,
        'prior_stop_result_sha':STOP_RESULT,'no_model_or_agent_routes':True,'no_lost_ACK_test':True}))
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    configure(root,orig,port,raw,a.prior)
    child=None;t=time.monotonic();r={'status':'failed_or_unknown'}
    try:
        r.update(source_storage_checks(a.source,root))
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
            r.update(asyncio.run(asyncio.wait_for(scenario(port,root,pcm),50)))
        r['status']='verified_private_durable_stop_and_unmodified_normal_recovery'
    except BaseException as e:r.update(error_type=type(e).__name__,error=str(e)[:240])
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(5)
            except subprocess.TimeoutExpired:child.kill();child.wait(5)
        r.update(child_stopped=child is None or child.poll() is not None,elapsed_seconds=round(time.monotonic()-t,3),
                 model_agent_or_phone_calls=0)
        _write_unique(root/'result.json',json_bytes(r));print(json.dumps(r))
    return 0 if r['status'].startswith('verified_private') else 3

if __name__=='__main__':raise SystemExit(main())
