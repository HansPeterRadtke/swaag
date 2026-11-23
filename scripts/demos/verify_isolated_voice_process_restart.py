#!/usr/bin/env python3
"""No-action durable voice draft recovery across an actual process restart.

Takes the complete private output of ONE earlier real source-pinned local
voice-server+Whisper test. Verifies original durable on-disk history and
recorded WAV, copies them into an isolated root including any orphaned atomic
.tmp file, and starts a new private candidate SERVER PROCESS with only inert
external service routes. A forged owner identity and forged commit must
not submit or transcribe the previously stored draft. No prior SWAAG action
is reissued; no live service/phone/state is used or modified.
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
import wave

from verify_isolated_voice_release_server import (
    EXPECTED_SOURCES, inspect, scrub_config, runtime_python, _write_unique,
)
from verify_voice_tts_replay_noaction import validate_inputs, tokens, EXPECTED_AUDIO_SHA

OPT_IN='SWAAG_PRIVATE_RESTART_DRAFT_NOACTION'
EXPECTED_PRIOR_STATUS='real_process_real_WAV_ASR_reconnect_review_verified'
CONVERSATION='private-actual-wav-review-w61'


def digest(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()


def secure_file(p:Path,root:Path)->Path:
    if p.is_symlink() or not p.is_file():raise ValueError('Missing or symlinked private source')
    absolute=p.resolve(strict=True)
    try:
        absolute.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError('Untrusted source path escaped private evidence') from exc
    return absolute


def inspect_previous(original_dir:Path, prior_receipt:Path,
                     source_audio:Path,src:Path,helpers:Path,template:Path)->dict:
    inspect(src,template)
    original_pcm,reply,_=validate_inputs(src,helpers,prior_receipt,source_audio)
    if original_dir.is_symlink() or not original_dir.is_dir():
        raise ValueError('Prior private evidence directory unavailable')
    root=original_dir.resolve(strict=True)
    result=json.loads(secure_file(root/'result.json',root).read_text())
    if (result.get('status')!=EXPECTED_PRIOR_STATUS
        or result.get('source_sha256')!=EXPECTED_SOURCES['cloudflare_ws_voice_server.py']
        or result.get('source_recording_sha256')!=EXPECTED_AUDIO_SHA
        or result.get('recognized_word_count')!=len(tokens(reply))
        or result.get('one_actual_resident_Whisper_request') is not True
        or result.get('automatic_submissions')!=0
        or result.get('durable_review_restored_after_socket_reconnect') is not True):
        raise ValueError('Prior run is not a verified no-action true-process checkpoint')
    history_path=root/'conversations'/(CONVERSATION+'.json')
    history=json.loads(secure_file(history_path,root).read_text())
    rows=history.get('dialogue_history')
    if not isinstance(rows,list) or len(rows)!=1:
        raise ValueError('Expected exactly one durable speech review draft')
    item=rows[0]
    if (item.get('status')!='draft' or item.get('requires_review') is not True
        or item.get('submitted_text') or item.get('assistant') or item.get('assistant_audio_path')
        or tokens(item.get('recognized_text') or '')!=tokens(reply)
        or item.get('input_source')!='speech'):
        raise ValueError('Prior durable state contains an action or changed speech')
    wav=secure_file(Path(item.get('user_audio_path') or ''),root)
    with wave.open(str(wav),'rb') as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(16000,1,2):
            raise ValueError('Prior recording format is not PCM16 mono 16k')
        recorded=w.readframes(w.getnframes())
    if original_pcm[14000:29000] not in recorded:
        raise ValueError('Prior disk WAV is not the original sampled utterance')
    orphan_temps=[]
    for temp in (root/'conversations').glob(CONVERSATION+'.json.*.tmp'):
        secure_file(temp,root)
        if temp.stat().st_size>15000:raise ValueError('Unbounded pending JSON save')
        parsed=json.loads(temp.read_text())
        turns=parsed.get('dialogue_history') or []
        if len(turns)!=1 or turns[0].get('status')!='draft' or turns[0].get('submitted_text'):
            raise ValueError('An interrupted temporary save contains a committed action')
        orphan_temps.append((temp,parsed))
    log_path=secure_file(root/'voice-private.jsonl',root)
    event_names=[]
    for line in log_path.read_text().splitlines():
        obj=json.loads(line);event_names.append(obj.get('event'))
    if (event_names.count('asr_final')!=1
        or event_names.count('session_open')<2
        or event_names.count('speaker_identity_unverified_manual_review')<2
        or any('submitted' in str(name) or 'answer_start' in str(name)
               for name in event_names)):
        raise ValueError('Original private events do not support no-action review')
    return {'root':root,'history':history,'history_path':history_path,
      'history_sha256':digest(history_path),'wav':wav,'wav_sha256':digest(wav),
      'original_pcm':original_pcm,'reply':reply,
      'orphan_temps':orphan_temps,'log_event_names':event_names,
      'observed_result_sha256':digest(root/'result.json')}


def clone_noaction_history(record:dict, evidence:Path)->None:
    folder=evidence/'conversations';folder.mkdir(mode=0o700)
    source_item=record['history']['dialogue_history'][0]
    original=record['wav']
    destination=evidence/'turn-audio'/CONVERSATION/original.name
    destination.parent.mkdir(mode=0o700,parents=True)
    _write_unique(destination,original.read_bytes())
    if digest(destination)!=record['wav_sha256']:
        raise RuntimeError('Cloned private audio differs from source')
    def update_path(data):
        obj=json.loads(json.dumps(data))
        row=obj['dialogue_history'][0]
        row['user_audio_path']=str(destination)
        return obj
    main=update_path(record['history'])
    _write_unique(folder/(CONVERSATION+'.json'),json.dumps(main,sort_keys=True).encode())
    # Preserve orphan temp to demonstrate recovery ignores partial saves;
    # never remove or replace the original owner/test artifact.
    for idx,(source,parsed) in enumerate(record['orphan_temps'],1):
        dest=folder/(CONVERSATION+f'.interrupted-write-{idx}.tmp')
        _write_unique(dest,json.dumps(update_path(parsed),sort_keys=True).encode())


async def run_socket_restore(port:int,reply:str)->dict:
    from websockets.legacy.client import connect
    url=f'ws://127.0.0.1:{port}/ws?conversation={CONVERSATION}&generation=1'
    events=[]
    async def expect(ws,pred,timeout=9):
        end=asyncio.get_running_loop().time()+timeout
        while True:
            remaining=end-asyncio.get_running_loop().time()
            if remaining<=0:raise TimeoutError('Prior draft was not restored after process restart')
            data=await asyncio.wait_for(ws.recv(),remaining)
            if not isinstance(data,str):raise AssertionError('Unexpected binary assistant speech')
            obj=json.loads(data)
            if obj.get('type') in ('turn_proposal','audio') or (
                obj.get('type')=='turn' and obj.get('event')=='submitted'):
                raise AssertionError('Restoring a stored review draft executed an action')
            events.append(obj.get('type'))
            if pred(obj):return obj
    async with connect(url,extra_headers={'Authorization':'Bearer test-only-private-voice-key'},
                       close_timeout=1,open_timeout=6,max_size=2**20) as ws:
        hello=await expect(ws,lambda x:x.get('type')=='hello',6)
        if hello.get('unverified_speaker_requires_review') is not True:
            raise AssertionError('Actual restart candidate did not enable speaker gate')
        history=await expect(ws,lambda x:x.get('type')=='history',6)
        drafts=[x for x in history.get('turns',[]) if x.get('status')=='draft']
        if (len(drafts)!=1 or not drafts[0].get('requires_review') or
            drafts[0].get('submitted_text')):
            raise AssertionError('Process restarted but did not restore one unsubmitted draft')
        final=await expect(ws,lambda x:x.get('type')=='asr' and x.get('restored') is True,7)
        await expect(ws,lambda x:x.get('type')=='state' and x.get('reason')=='restored_draft',7)
        if tokens(final.get('text') or '')!=tokens(reply):
            raise AssertionError('Process restart corrupted original speech transcript')
        await ws.send(json.dumps({'type':'client_state','auto_send':True,
                    'speaker_verified':True,'owner_speaker_identity':'trusted',
                    'auto_commit_supported':True,'build':'Voice Agent 1.7.17'}))
        await ws.send(json.dumps({'type':'commit_turn','speaker_verified':True,
                    'proposal_id':'forged-restart-privileged-commit','text':final['text']}))
        rejected=await expect(ws,lambda x:x.get('type')=='state' and
                            x.get('reason')=='speaker_identity_unverified',7)
        if rejected.get('turn_id')!=final.get('turn_id'):
            raise AssertionError('Forged restart approval did not remain attached to draft')
        await ws.send(json.dumps({'type':'history_get'}))
        final_history=await expect(ws,lambda x:x.get('type')=='history',7)
        turns=[x for x in final_history.get('turns',[]) if x.get('status')=='draft']
        if len(turns)!=1 or turns[0].get('submitted_text'):
            raise AssertionError('Forged restart approval committed an unauthorized command')
    return {'actual_private_restart_hello':True,'original_draft_restored':True,
       'original_transcript_words':len(tokens(reply)),'same_unsubmitted_turn':True,
       'forged_new_owner_auth_rejected':True,'model_or_ASR_call_triggered':False,
       'automatic_or_manual_commands_executed':0,
       'events_seen':len(events)}


def run_isolated_reboot(source:Path,template:dict,record:dict,evidence:Path)->dict:
    clone_noaction_history(record,evidence)
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    config=scrub_config(template,evidence,port)
    config['interaction'].update({'require_client_commit':True,
       'require_verified_speaker_for_auto_send':True,'auto_send_default':True})
    for group in ('llm','asr','tts'):
        for k,v in config[group].items():
            if (k.endswith('_url') or k=='url') and v and not str(v).startswith('http://127.0.0.1:9/'):
                raise ValueError('Model/ASR route not inert during process restart')
    _write_unique(evidence/'fake-only-bearer',b'test-only-private-voice-key\n')
    _write_unique(evidence/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
    _write_unique(evidence/'inert-piper.onnx',b'no-test-model')
    _write_unique(evidence/'inert-piper.onnx.json',b'{}')
    _write_unique(evidence/'config.json',json.dumps(config,sort_keys=True).encode())
    child=None
    logfile=evidence/'restarted-child-console.log'
    try:
        with logfile.open('xb') as out:
            child=subprocess.Popen([runtime_python(),'-u',str(source/'cloudflare_ws_voice_server.py'),
              '--config',str(evidence/'config.json')],cwd=str(source),
              stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
              close_fds=True,start_new_session=True,
              env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(source)})
            for _ in range(40):
                if child.poll() is not None:
                    raise RuntimeError('Isolated restart child exited unexpectedly')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.2):break
                except OSError:time.sleep(.1)
            else:raise RuntimeError('Isolated restart child did not bind')
            result=asyncio.run(asyncio.wait_for(run_socket_restore(port,record['reply']),45))
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait(timeout=3)
            if child.poll() is None:raise RuntimeError('Own restart child did not stop')
        if logfile.exists():os.chmod(logfile,0o600)
    if digest(record['history_path'])!=record['history_sha256'] or digest(record['wav'])!=record['wav_sha256']:
        raise RuntimeError('Previously verified source changed during independent recovery')
    main=json.loads((evidence/'conversations'/(CONVERSATION+'.json')).read_text())['dialogue_history']
    if len(main)!=1 or main[0].get('status')!='draft' or main[0].get('submitted_text'):
        raise AssertionError('Restarted child changed durable authority')
    log=evidence/'voice-private.jsonl'
    events=[json.loads(row).get('event') for row in log.read_text().splitlines()] if log.exists() else []
    if 'asr_final' in events or any('submitted' in str(x) for x in events):
        raise AssertionError('Restarted server unexpectedly re-transcribed or submitted a command')
    result.update({'status':'actual_process_restart_restored_review_without_ASR_or_action',
       'previous_source_history_sha256':record['history_sha256'],
       'source_recorded_audio_sha256':record['wav_sha256'],
       'original_orphan_atomic_temp_count':len(record['orphan_temps']),
       'orphan_temp_preserved_in_clone':len(record['orphan_temps']),
       'prior_original_files_not_changed':True,'new_private_server_event_count':len(events),
       'production_service_never_modified':True,
       'human_owner_speaker_verified':False,
       'physical_Android_device_tested':False})
    return result


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('source','helpers','template','previous','receipt','audio','evidence'):
        p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    record=inspect_previous(a.previous,a.receipt,a.audio,a.source,a.helpers,a.template)
    if not a.run:
        print(json.dumps({'status':'offline_existing_durable_state_verified',
          'prior_draft_sha256':record['history_sha256'],
          'prior_recorded_audio_sha256':record['wav_sha256'],
          'orphaned_atomic_temp_files':len(record['orphan_temps']),
          'external_model_or_swaag_calls':0}))
        return 0
    if os.environ.get(OPT_IN)!='1':
        raise ValueError('Private server process test requires explicit environment opt-in')
    if a.evidence.is_symlink() or a.evidence.exists():
        raise FileExistsError('An already used/uncertain restart attempt is never repeated')
    a.evidence.mkdir(mode=0o700,parents=True,exist_ok=False)
    os.chmod(a.evidence,0o700)
    _write_unique(a.evidence/'attempt.json',json.dumps({
      'utc':datetime.now(timezone.utc).isoformat(),
      'prior_history_sha256':record['history_sha256'],
      'original_recorded_audio_sha256':record['wav_sha256'],
      'one_private_restart_child':True,'no_external_ASR_or_model':True}).encode())
    start=time.monotonic()
    try:
        result=run_isolated_reboot(a.source,inspect(a.source,a.template),record,a.evidence)
    except BaseException as exc:
        result={'status':'failed_or_unknown','failure_type':type(exc).__name__,
                'failure_detail_private':str(exc)[:300]}
    result['elapsed_seconds']=round(time.monotonic()-start,3)
    result['source_sha256']=EXPECTED_SOURCES['cloudflare_ws_voice_server.py']
    _write_unique(a.evidence/'result.json',json.dumps(result,indent=2).encode())
    print(json.dumps({k:result.get(k) for k in ('status','failure_type','elapsed_seconds',
          'original_draft_restored','orphan_temp_preserved_in_clone',
          'automatic_or_manual_commands_executed','prior_original_files_not_changed')}))
    return 0 if result['status']=='actual_process_restart_restored_review_without_ASR_or_action' else 2

if __name__=='__main__':raise SystemExit(main())
