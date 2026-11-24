#!/usr/bin/env python3
"""One-use NEW private Jetson server process reloads W69 real saved history.

After an actual forced on-disk ASR/user-intent race, the source history has a
superseded old speech row and a legitimate newer unsubmitted speech draft.
Start a fresh server process against a COPY of those bytes. Confirm only the
newer input is restored for manual review, while old superseded speech is
never resurrected or submitted; verify a second separate process against an
explicitly old-only private copy also does not revive the abandoned speech.
This test never replays any original model/ASR/audio/tool effects or alters
production voice/Android source, tokens, user histories, or external state.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from prepare_isolated_voice_atomic_intent_draft_guard import BASE,PATCHED_SERVER_SHA256
from verify_isolated_voice_release_server import scrub_config,runtime_python,_write_unique

ORIGINAL_DISK_SHA='bdde160a4996a1f38e8a88366bf2bb63e6ad3a65b2ebd26687326a03158102fc'
SOURCE_CID='private-w69-storage-race-A'
FAKE_BEARER='only-this-private-test-bearer'
OPT_IN='SWAAG_W69_DURABLE_RESTART_AFTER_ATOMIC_RACE_NOACTION'


def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()


def safe_source(src:Path,hist:Path,template:Path)->tuple[dict,dict]:
    if any(p.is_symlink() for p in (src,hist,template)) or not src.is_dir() or not hist.is_file() or not template.is_file():
        raise ValueError('Source, history or template missing/symlinked')
    for name,old in BASE.items():
        file=src/name;want=PATCHED_SERVER_SHA256 if name=='cloudflare_ws_voice_server.py' else old
        if not file.is_file() or file.is_symlink() or sha(file)!=want:
            raise ValueError('Actual undeployed source bytes differ: '+name)
    if sha(hist)!=ORIGINAL_DISK_SHA:raise ValueError('Original forced-race durable history changed')
    data=json.loads(hist.read_text());rows=data.get('dialogue_history') or []
    if (len(rows)!=2 or rows[0].get('status')!='superseded' or rows[1].get('status')!='draft'
        or rows[0].get('turn_id')!='turn-old' or rows[1].get('turn_id')!='turn-current'
        or rows[0].get('asr_intent_epoch')!=1 or rows[1].get('asr_intent_epoch')!=2
        or any(v.get('submitted_text') or v.get('assistant') for v in rows)):
        raise ValueError('Original captured old/new durable voice turn shape invalid')
    original=json.loads(template.read_text())
    return original,data


def private_config(original:dict,root:Path,port:int)->dict:
    d=scrub_config(original,root,port)
    if d['interaction']['require_verified_speaker_for_auto_send'] is not True:
        raise ValueError('No source guard in private restart')
    for group in ('asr','llm','tts'):
        for k,v in d[group].items():
            if (k.endswith('_url') or k=='url') and v:
                if not str(v).startswith('http://127.0.0.1:9/'):
                    raise ValueError('Unexpected model/tool route')
    return d


async def scenario(port:int,mode:str,rows:list[dict])->dict:
    # Defer the optional WebSocket import until an explicitly admitted private
    # on-wire run. Source pin and one-use refusal tests need no network dep.
    from websockets.legacy.client import connect
    url=f'ws://127.0.0.1:{port}/ws?conversation={SOURCE_CID}&generation=9'
    headers={'Authorization':'Bearer '+FAKE_BEARER}
    events=[]
    async def receive(sock,predicate,seconds=7):
        end=asyncio.get_running_loop().time()+seconds
        while True:
            left=end-asyncio.get_running_loop().time()
            if left<=0:raise TimeoutError('Expected real source server event not observed')
            msg=await asyncio.wait_for(sock.recv(),left)
            if not isinstance(msg,str):raise AssertionError('Unexpected TTS/media returned')
            d=json.loads(msg)
            if d.get('type') in ('turn_proposal','audio') or (
                d.get('type')=='turn' and d.get('event')=='submitted'):
                raise AssertionError('Unverified voice caused an automatic action or playback')
            events.append({k:d.get(k) for k in ('type','event','state','reason') if k in d})
            if predicate(d):return d
    async with connect(url,extra_headers=headers,open_timeout=5,close_timeout=1,max_size=2**20) as socket:
        hello=await receive(socket,lambda d:d.get('type')=='hello')
        if hello.get('unverified_speaker_requires_review') is not True:
            raise AssertionError('Private server does not advertise speaker review')
        history=await receive(socket,lambda d:d.get('type')=='history')
        expected_status=['superseded','draft'] if mode=='latest_new_draft' else ['superseded']
        if [v.get('status') for v in history.get('turns') or []]!=expected_status:
            raise AssertionError('Real source failed to reload original observed disk status')
        if mode=='latest_new_draft':
            restored=await receive(socket,lambda d:d.get('type')=='asr' and d.get('event')=='final' and d.get('restored') is True)
            if restored.get('turn_id')!='turn-current' or restored.get('text')!=rows[1]['recognized_text']:
                raise AssertionError('Old superseded ASR replaced newest pending user draft')
            await receive(socket,lambda d:d.get('type')=='state' and d.get('reason')=='restored_draft')
            # The client cannot provide any trusted voice identity by claiming
            # it is the owner. Its forged commit cannot submit the text.
            await socket.send(json.dumps({'type':'client_state','auto_send':True,
                'speaker_verified':True,'speaker_identity':'owner',
                'owner_speaker_identity':'trusted'}))
            await socket.send(json.dumps({'type':'commit_turn','proposal_id':'forged-owner-after-atomic-restart',
                'speaker_verified':True,'text':rows[1]['recognized_text']}))
            await receive(socket,lambda d:d.get('type')=='state' and d.get('reason')=='speaker_identity_unverified')
        else:
            # With only an explicitly superseded old row, there is nothing
            # eligible for editor recovery, and no model/remote Send.
            await receive(socket,lambda d:d.get('type')=='state' and d.get('state')=='listening')
            await socket.send(json.dumps({'type':'client_state','speaker_verified':True,
                 'owner_speaker_identity':'trusted','auto_send':True}))
            await socket.send(json.dumps({'type':'commit_turn','speaker_verified':True,
                 'proposal_id':'bad-old-user-commit','text':rows[0]['recognized_text']}))
            await receive(socket,lambda d:d.get('type')=='state' and d.get('reason')=='speaker_identity_unverified')
        await socket.send(json.dumps({'type':'history_get'}))
        persisted=await receive(socket,lambda d:d.get('type')=='history')
        if [v.get('status') for v in persisted.get('turns') or []]!=expected_status:
            raise AssertionError('Forged owner commit changed durable reconstructed history')
        after=[v for v in persisted.get('turns') or [] if v.get('status')=='draft']
        if mode=='latest_new_draft':
            if len(after)!=1 or after[0]['turn_id']!='turn-current' or after[0].get('requires_review') is not True:
                raise AssertionError('Only newest unsubmitted draft must remain active/reviewable')
        elif after:
            raise AssertionError('Superseded-only historic voice draft was resurrected')
    return {'status':'source_pinned_real_server_restart_from_W69_durable_history_verified',
        'mode':mode,'restored_active_review_count':len(after),
        'historical_superseded_count':expected_status.count('superseded'),
        'source_history_expected_statuses':expected_status,
        'fake_owner_claim_and_commit_could_not_execute':True,
        'actual_new_model_or_ASR_TTS_tool_calls':0,
        'actual_private_server_WS_events':len(events)}


def one_child(src:Path,original:dict,history:dict,out:Path,mode:str)->dict:
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    d=private_config(original,out,port)
    _write_unique(out/'fake-only-bearer',(FAKE_BEARER+'\n').encode())
    _write_unique(out/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
    _write_unique(out/'inert-piper.onnx',b'no-real-voice-model')
    _write_unique(out/'inert-piper.onnx.json',b'{}')
    _write_unique(out/'private-config.json',json.dumps(d,sort_keys=True).encode())
    conversations=out/'conversations';conversations.mkdir(mode=0o700,exist_ok=False)
    path=conversations/(SOURCE_CID+'.json')
    _write_unique(path,json.dumps(history,ensure_ascii=False,indent=2).encode())
    child=None
    try:
        with (out/'private-child-console.log').open('xb') as log:
            child=subprocess.Popen([runtime_python(),'-u',str(src/'cloudflare_ws_voice_server.py'),
                        '--config',str(out/'private-config.json')],cwd=str(src),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(src)})
            for _ in range(60):
                if child.poll() is not None:raise RuntimeError('Private restart source child exited')
                try:
                    with socket.socket() as sock:
                        sock.settimeout(.2);sock.connect(('127.0.0.1',port));break
                except OSError:time.sleep(.1)
            else:raise TimeoutError('Private restart source did not bind')
            result=asyncio.run(asyncio.wait_for(scenario(port,mode,history['dialogue_history']),timeout=25))
            result['private_owned_child_pid']=child.pid
            return result
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            if child.poll() is None:raise RuntimeError('Private restart child did not stop')
        for name in ('private-config.json','private-child-console.log'):
            p=out/name
            if p.is_file():p.chmod(0o600)


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','template','original-history','evidence'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    original,hist=safe_source(a.source,a.original_history,a.template)
    if not a.run:
        print(json.dumps({'status':'private_durable_history_restart_preflight_no_network',
            'original_history_sha256':ORIGINAL_DISK_SHA,
            'source_sha256':PATCHED_SERVER_SHA256,
            'distinct_test_server_child_processes_not_started':2}))
        return 0
    if os.environ.get(OPT_IN)!='1':raise ValueError('Explicit unique actual server restart opt-in required')
    if a.evidence.is_symlink() or a.evidence.exists():raise FileExistsError('Never repeat previous real/uncertain restart test')
    a.evidence.mkdir(parents=True,mode=0o700,exist_ok=False)
    os.chmod(a.evidence,0o700)
    _write_unique(a.evidence/'attempt.json',json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':PATCHED_SERVER_SHA256,'history_sha256':ORIGINAL_DISK_SHA,
        'planned_two_new_private_server_processes':True,
        'no_original_voice_audio_or_model_replay':True}).encode())
    start=time.perf_counter()
    results=[]
    try:
        for mode in ('latest_new_draft','superseded_only'):
            path=a.evidence/mode
            path.mkdir(mode=0o700,exist_ok=False)
            rows=hist if mode=='latest_new_draft' else {'dialogue_history':[dict(hist['dialogue_history'][0])]}
            outcome=one_child(a.source,original,rows,path,mode)
            results.append(outcome)
        result={'status':'both_real_private_source_restart_draft_recovery_cases_passed',
            'cases':results,'one_actual_new_speaker_review_draft':True,
            'superseded_old_speech_never_restored':True,
            'no_ASR_LLMTTS_SWAAG_tools_or_external_effects':True}
    except BaseException as e:
        result={'status':'failed_or_unknown','failure_type':type(e).__name__,
           'failure_internal':str(e)[:250],'completed_cases':results}
    result['source_sha256']=PATCHED_SERVER_SHA256
    result['source_durable_history_sha256']=ORIGINAL_DISK_SHA
    result['wall_seconds']=round(time.perf_counter()-start,3)
    result['actual_owner_phone_or_live_service_modified']=False
    _write_unique(a.evidence/'result.json',json.dumps(result,indent=2).encode())
    print(json.dumps({k:result.get(k) for k in ('status','failure_type','wall_seconds')},sort_keys=True))
    return 0 if result['status']=='both_real_private_source_restart_draft_recovery_cases_passed' else 3

if __name__=='__main__':raise SystemExit(main())
