#!/usr/bin/env python3
"""Actual local VoiceCore durable JSON write vs newer user intent.

Runs one unique separate interpreter with the exact undeployed 7-module
Jetson source, private disk, real atomic_write_json/fsync, and controlled
thread barriers. No WebSocket, microphone, LLM, ASR, TTS, tools, production
service, real phone or user auth. This tests the narrow race that previous
real WebSocket sessions did not force: an old ASR writer ALREADY inside
atomic persistence when a new user manual recording starts.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from prepare_isolated_voice_atomic_intent_draft_guard import BASE,PATCHED_SERVER_SHA256,digest,modify
from verify_isolated_voice_release_server import scrub_config

OPTIN='SWAAG_W69_ONEUSE_ATOMIC_DRAFT_DISK_RACE_NOACTION'


def source_guard(src:Path)->None:
    if src.is_symlink() or not src.is_dir():raise ValueError('Missing/symlinked source')
    for name,sha in BASE.items():
        p=src/name
        expected=PATCHED_SERVER_SHA256 if name=='cloudflare_ws_voice_server.py' else sha
        if not p.is_file() or p.is_symlink() or digest(p.read_bytes())!=expected:
            raise ValueError('Not exact isolated patched module bytes '+name)


def unique(path:Path,raw:bytes,mode:int=0o600)->None:
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,mode)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def execute(src:Path,template:Path,out:Path)->dict:
    source_guard(src)
    if template.is_symlink() or not template.is_file():raise ValueError('No fixed source template')
    original=json.loads(template.read_text())
    c=scrub_config(original,out,0)
    c['server']['port']=19359
    c['interaction']['auto_send_default']=True
    c['interaction']['require_verified_speaker_for_auto_send']=True
    if c['server']['host']!='127.0.0.1' or c['llm']['url']!='http://127.0.0.1:9/run':
        raise RuntimeError('Actual source test may not use model or live network')
    unique(out/'fake-only-bearer',b'private-oneuse-test-only\n')
    unique(out/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
    unique(out/'inert-piper.onnx',b'no-TTS-model')
    unique(out/'inert-piper.onnx.json',b'{}')
    unique(out/'config.json',json.dumps(c,sort_keys=True).encode())
    # New subprocess launched with this exact source root. Cannot import the
    # production-installed code, a different source, or an already imported
    # cloudflare module from an unrelated developer context.
    if 'cloudflare_ws_voice_server' in sys.modules:
        raise RuntimeError('Refuse to inspect an already imported global server')
    sys.path.insert(0,str(src))
    from voice_ws_config import load_voice_ws_config
    from voice_ws_core import VoiceCore
    import cloudflare_ws_voice_server as server
    server.CONFIG=load_voice_ws_config(out/'config.json')
    server.CORE=VoiceCore(server.CONFIG)
    t0=time.perf_counter()
    try:
        a='private-w69-storage-race-A';b='private-w69-storage-race-B'
        epoch_a=server.advance_conversation_intent_epoch(a)
        epoch_b=server.advance_conversation_intent_epoch(b)
        assert (epoch_a,epoch_b)==(1,1)
        assert server.CONVERSATION_INTENT_EPOCH=={a:1,b:1}
        arrived=threading.Event();release=threading.Event();intent_started=threading.Event()
        intent_done=threading.Event();state={}
        real_write=server.core().atomic_write_json
        def held_write(path,data):
            # Start real, private persistent storage but pause exactly after
            # entering the writer's protected critical section and before its
            # actual tempfile/fsync/rename. This interleaving was not covered
            # by Wave68's remote-ASR-delay test.
            if (threading.current_thread().name=='test-old-ASR-disk-writer'
                and path==server.core().conversation_path(a)):
                arrived.set()
                if not release.wait(10):raise TimeoutError('Private disk writer barrier not released')
            return real_write(path,data)
        server.core().atomic_write_json=held_write
        def writer():
            try:
                state['old_write_accepted']=server.record_draft(a,'turn-old',
                    'Earlier dictated speech','Earlier dictated speech',
                    str(out/'recording-still-preserved-original.wav'),'speech',
                    expected_intent_epoch=epoch_a)
            except BaseException as e:state['writer_error']=str(e)
        def newer_input():
            intent_started.set()
            try:
                state['new_epoch']=server.advance_conversation_intent_epoch(
                    a,supersede_prior_speech=True)
            except BaseException as e:state['new_intent_error']=str(e)
            finally:intent_done.set()
        thread_writer=threading.Thread(target=writer,name='test-old-ASR-disk-writer',daemon=True)
        thread_writer.start()
        if not arrived.wait(5):raise TimeoutError('Real old ASR never entered protected disk writer')
        thread_intent=threading.Thread(target=newer_input,name='test-new-manual-intent',daemon=True)
        thread_intent.start()
        if not intent_started.wait(3):raise TimeoutError('No newer input request issued')
        # An unrelated conversation MUST NOT be blocked by first
        # conversation's slow fsync/rename. Save a true second JSON row.
        before=time.perf_counter()
        accepted_b=server.record_draft(b,'turn-independent','Other speech',
            'Other speech',str(out/'other-recording-preserved.wav'),'speech',
            expected_intent_epoch=epoch_b)
        unrelated_seconds=round(time.perf_counter()-before,3)
        if not accepted_b or unrelated_seconds>1.0:
            raise AssertionError('Per-conversation serialization blocked unrelated client')
        # The newer manual input has *arrived*, but it cannot mutate the
        # intent version until the already-admitted old disk write finishes.
        if intent_done.wait(.1):
            raise AssertionError('New intent crossed an in-progress atomic draft write')
        release.set()
        thread_writer.join(timeout=8);thread_intent.join(timeout=8)
        if thread_writer.is_alive() or thread_intent.is_alive():
            raise TimeoutError('Private storage write/intent deadlock')
        if any(k in state for k in ('writer_error','new_intent_error')):
            raise RuntimeError('Source thread errors: '+str(state))
        if state.get('old_write_accepted') is not True or state.get('new_epoch')!=2:
            raise AssertionError('Unknown order of serialized storage and user input')
        rows=server._history_read(a)['dialogue_history']
        if (len(rows)!=1 or rows[0]['status']!='superseded' or
            rows[0].get('submitted_text') or rows[0].get('assistant') or
            rows[0].get('asr_intent_epoch')!=1 or
            rows[0].get('superseded_by_intent_epoch')!=2 or
            not rows[0].get('user_audio_path')):
            raise AssertionError('Already-admitted old speech was not preserved but retired safely')
        # A slow old ASR callback attempting another *new* draft after epoch
        # advancement must be rejected with no new durable row.
        stale=server.record_draft(a,'turn-stale-late','Late after new manual',
              'Late after new manual','', 'speech',expected_intent_epoch=epoch_a)
        if stale is not False or len(server._history_read(a)['dialogue_history'])!=1:
            raise AssertionError('Late unverified draft crossed new user intent')
        before_update=server._history_read(a)['dialogue_history'][0].copy()
        modified=server.update_turn(a,'turn-old',status='draft',
                     _expected_intent_epoch=epoch_a)
        if modified is not False or server._history_read(a)['dialogue_history'][0]!=before_update:
            raise AssertionError('Late stale speaker review altered retired durable history')
        # A genuinely NEW current user speech may still produce one draft.
        newly=server.record_draft(a,'turn-current','New currently valid words',
            'New currently valid words',str(out/'new-recording-preserved.wav'),
            'speech',expected_intent_epoch=state['new_epoch'])
        if newly is not True:
            raise AssertionError('User-authorized newer acoustic input was blocked')
        rows=server._history_read(a)['dialogue_history']
        if (len(rows)!=2 or rows[-1]['status']!='draft' or
            rows[-1]['turn_id']!='turn-current' or
            rows[-1]['asr_intent_epoch']!=2 or rows[0]['status']!='superseded'):
            raise AssertionError('Current user recording was overwritten by stale ASR')
        brow=server._history_read(b)['dialogue_history']
        if len(brow)!=1 or brow[0]['status']!='draft' or brow[0]['turn_id']!='turn-independent':
            raise AssertionError('Independent conversation history polluted')
        return {'status':'genuine_atomic_disk_commit_vs_new_manual_intent_race_passed',
            'original_source_server_sha256':PATCHED_SERVER_SHA256,
            'source_version_checked':True,'one_actual_blocked_atomic_json_write':True,
            'actual_new_manual_intent_waited_for_disk_transaction':True,
            'old_source_result_preserved_but_superseded':True,
            'old_stale_later_draft_and_update_denied':True,
            'latest_new_audio_draft_recorded_exactly_once':True,
            'other_conversation_progressed_during_held_write':True,
            'other_conversation_elapsed_seconds':unrelated_seconds,
            'real_atomic_write_json_tempfile_fsync_rename':True,
            'durable_A_statuses':[row['status'] for row in rows],
            'durable_A_turn_count':len(rows),'durable_B_turn_count':len(brow),
            'one_use_wall_seconds':round(time.perf_counter()-t0,3),
            'user_phone_model_provider_tool_calls':0,
            'injected_model_or_audio_accuracy_claims':False}
    finally:
        server.CORE.close();server.CORE=None;server.CONFIG=None


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','template','evidence'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    source_guard(a.source)
    if not a.run:
        print(json.dumps({'status':'actual_source_storage_race_preflight_only',
         'pinned_source_SHA256':PATCHED_SERVER_SHA256,
         'no_source_process_or_result_created':True}))
        return 0
    if os.environ.get(OPTIN)!='1':raise ValueError('One-use private source-disk race opt-in required')
    if a.evidence.is_symlink() or a.evidence.exists():
        raise FileExistsError('Uncertain or prior local writer test must not be replayed')
    a.evidence.mkdir(parents=True,mode=0o700,exist_ok=False)
    os.chmod(a.evidence,0o700)
    unique(a.evidence/'attempt.json',json.dumps({'at':datetime.now(timezone.utc).isoformat(),
        'source_sha256':PATCHED_SERVER_SHA256,'one_use_non_network_real_core_storage':True,
        'no_real_phone_or_model_communication':True}).encode())
    start=time.perf_counter()
    try:
        d=execute(a.source,a.template,a.evidence)
    except BaseException as e:
        d={'status':'failed_or_unknown','failure_type':type(e).__name__,
           'private_reason':str(e)[:250]}
    d['source_sha256']=PATCHED_SERVER_SHA256
    d['total_wall_seconds']=round(time.perf_counter()-start,3)
    d['live_services_changed']=False
    unique(a.evidence/'result.json',json.dumps(d,indent=2).encode())
    print(json.dumps({k:d.get(k) for k in ('status','failure_type','one_use_wall_seconds',
          'durable_A_statuses','durable_B_turn_count','other_conversation_elapsed_seconds')},sort_keys=True))
    return 0 if d['status']=='genuine_atomic_disk_commit_vs_new_manual_intent_race_passed' else 3

if __name__=='__main__':raise SystemExit(main())
