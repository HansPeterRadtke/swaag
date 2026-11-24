#!/usr/bin/env python3
"""New no-action source test: slow fsync must not stall all voice sessions.

This deliberately executes the REAL VoiceSession.cancel_response method and
REAL VoiceCore atomic JSON write against a private on-disk conversation. A
single local 0.70s test-only slow disk-write hook is held while an unrelated
async heartbeat ticks. The old, undeployed source performs the write on the
event loop; the separate new candidate offloads it to a thread while keeping
its conversation-scoped atomic mutation. No real models, TTS, microphone,
WebSocket network, owner credentials, external tools or production service.
"""
from __future__ import annotations
import argparse,asyncio,hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
import verify_isolated_voice_atomic_draft_restart_once as previous
from verify_isolated_voice_release_server import _write_unique,scrub_config

OLD='3d4e6e029aa4a467684c618dc31cfd60d53d77939843f547d394b04a979bad2f'
NEW='f3283c43a196595c4b1fade69e6f2ef0f23943dc6d0b91c5b5ab63f16c17c996'
ORIGINAL_HISTORY_SHA='bdde160a4996a1f38e8a88366bf2bb63e6ad3a65b2ebd26687326a03158102fc'
SUPPORT={k:v for k,v in previous.BASE.items() if k!='cloudflare_ws_voice_server.py'}
CID=previous.SOURCE_CID
OPTIN='SWAAG_W70_NEW_SOURCE_NONBLOCKING_AUDIO_CONTROL_STALL_TEST'


def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()


def pin(source:Path,history:Path,template:Path,chosen_sha:str):
    if any(x.is_symlink() for x in (source,history,template)) or not source.is_dir() or not history.is_file() or not template.is_file():
        raise ValueError('Missing or symlinked source, fixed history, config')
    if sha(history)!=ORIGINAL_HISTORY_SHA:raise ValueError('Real original W69 durable source was changed')
    for n,expected in {**SUPPORT,'cloudflare_ws_voice_server.py':chosen_sha}.items():
        f=source/n
        if not f.is_file() or f.is_symlink() or sha(f)!=expected:
            raise ValueError('No exact previously frozen candidate module '+n)
    d=json.loads(history.read_text());rows=d['dialogue_history']
    if ([r.get('asr_intent_epoch') for r in rows]!=[1,2]
        or [r.get('status') for r in rows]!=['superseded','draft']
        or any(r.get('submitted_text') or r.get('assistant') for r in rows)):
        raise ValueError('Original archived previous user text is not exact')
    return d,json.loads(template.read_text())


def run_one(source:Path,history:dict,original_cfg:dict,out:Path,chosen_sha:str,mode:str):
    config=scrub_config(original_cfg,out,0)
    config['server']['port']=19358
    if config['server']['host']!='127.0.0.1' or config['interaction']['require_verified_speaker_for_auto_send'] is not True:
        raise ValueError('Local-only protected voice runtime required')
    if config['llm']['url']!='http://127.0.0.1:9/run':raise ValueError('Unexpected remote LLM route')
    _write_unique(out/'fake-only-bearer',b'new-unreleased-test-only\n')
    _write_unique(out/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
    _write_unique(out/'inert-piper.onnx',b'not-a-TTS')
    _write_unique(out/'inert-piper.onnx.json',b'{}')
    _write_unique(out/'private-config.json',json.dumps(config,sort_keys=True).encode())
    p=out/'conversations'/(CID+'.json');p.parent.mkdir(mode=0o700,exist_ok=False)
    _write_unique(p,json.dumps(history,ensure_ascii=False,indent=2).encode())
    if 'cloudflare_ws_voice_server' in sys.modules:
        raise ValueError('Already imported other source runtime, refuse')
    sys.path.insert(0,str(source))
    from voice_ws_config import load_voice_ws_config
    from voice_ws_core import VoiceCore
    import cloudflare_ws_voice_server as server
    server.CONFIG=load_voice_ws_config(out/'private-config.json')
    server.CORE=VoiceCore(server.CONFIG)
    try:
        server.CONVERSATION_INTENT_EPOCH[CID]=2
        original_write=server.core().atomic_write_json
        hold={'counts':0}
        def slow_actual_json(path,obj):
            if path==server.core().conversation_path(CID):
                hold['counts']+=1
                time.sleep(.70)  # Controlled measured slow fsync analogy.
            return original_write(path,obj)  # Still executes real fsync/rename.
        server.core().atomic_write_json=slow_actual_json
        events=[]
        class FakePrivateWs:
            async def send(self,message):
                assert isinstance(message,str)
                j=json.loads(message)
                if j.get('type') in ('turn_proposal','audio') or j.get('event')=='submitted':
                    raise AssertionError('Unknown speaker caused a voice tool or playback')
                events.append({k:j.get(k) for k in ('type','event','state','reason') if k in j})
        async def test():
            s=server.VoiceSession(FakePrivateWs(),CID)
            ticks=[]
            async def heartbeat():
                end=time.monotonic()+1.10
                while time.monotonic()<end:
                    ticks.append(time.monotonic())
                    await asyncio.sleep(.012)
                ticks.append(time.monotonic())
            ht=asyncio.create_task(heartbeat())
            await asyncio.sleep(.07)
            started=time.monotonic()
            await s.cancel_response('new_manual_utterance')
            intent_elapsed=time.monotonic()-started
            await ht
            gaps=[b-a for a,b in zip(ticks,ticks[1:])]
            return {'max_event_loop_tick_gap_seconds':round(max(gaps),3),
                'total_heartbeat_samples':len(ticks),
                'new_manual_intent_completion_seconds':round(intent_elapsed,3),
                'async_WS_events':events,'source_epoch_after':s.intent_epoch,
                'pending_untrusted_TTS_model_or_tool_actions':0}
        result=asyncio.run(test())
        current=server._history_read(CID)['dialogue_history']
        status=[t.get('status') for t in current]
        if status!=['superseded','superseded'] or hold['counts']!=1 or result['source_epoch_after']!=3:
            raise ValueError('Private durable storage epoch did not retire prior draft exactly once')
        gap=result['max_event_loop_tick_gap_seconds']
        if mode=='old_blocking':
            if gap<.55:raise AssertionError('Original old source did not reproduce event-loop fsync stall')
            result['status']='observed_prior_voice_event_loop_0_7sec_storage_stall'
        elif mode=='new_offloaded':
            if gap>.28:raise AssertionError('New source still stalls independent event loop ticks')
            result['status']='fixed_atomic_voice_intent_durable_write_no_event_loop_stall'
        else:raise ValueError('Unknown source test mode')
        result['actual_source_history_statuses']=status
        result['private_durable_JSON_write_count']=hold['counts']
        result['original_disk_history_is_source_pinned']=True
        result['source_sha256']=chosen_sha
        result['no_network_or_provider_calls']=True
        return result
    finally:
        server.CORE.close();server.CORE=None;server.CONFIG=None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','original-history','template','evidence','expected-server-sha'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--mode',required=True,choices=['old_blocking','new_offloaded'])
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    if a.expected_server_sha not in (OLD,NEW):raise ValueError('Unsupported frozen model-free server source')
    d,config=pin(Path(a.source),Path(a.original_history),Path(a.template),a.expected_server_sha)
    if not a.run:
        print(json.dumps({'status':'source_pinned_event_loop_stall_test_preflight_only',
           'model_calls':0,'source_sha256':a.expected_server_sha,'original_row_epochs':[r['asr_intent_epoch'] for r in d['dialogue_history']]}))
        return 0
    if os.environ.get(OPTIN)!='1':raise ValueError('Explicit new private no-action test opt-in required')
    root=Path(a.evidence)
    if not root.is_absolute() or not str(root).startswith('/data/var/swaag-benchmarks/') or root.exists() or root.is_symlink():
        raise FileExistsError('Refuse existing or outside private one-use result')
    root.mkdir(mode=0o700,parents=True,exist_ok=False);os.chmod(root,0o700)
    _write_unique(root/'attempt.json',json.dumps({'at':datetime.now(timezone.utc).isoformat(),
      'mode':a.mode,'source_sha256':a.expected_server_sha,'original_history_sha':ORIGINAL_HISTORY_SHA,
      'no_real_models_network_or_phone':True}).encode())
    start=time.monotonic()
    try:r=run_one(Path(a.source),d,config,root,a.expected_server_sha,a.mode)
    except BaseException as e:r={'status':'failed_or_unknown','failure_type':type(e).__name__,'detail_private':str(e)[:250]}
    r.update({'mode':a.mode,'total_wall_seconds':round(time.monotonic()-start,3),
       'no_live_voice_or_phone_changes':True,'external_provider_model_tool_calls':0})
    _write_unique(root/'result.json',json.dumps(r,indent=2).encode())
    print(json.dumps({k:r.get(k) for k in ('status','mode','max_event_loop_tick_gap_seconds','new_manual_intent_completion_seconds','total_heartbeat_samples','failure_type')},sort_keys=True))
    return 0 if r['status'] in ('observed_prior_voice_event_loop_0_7sec_storage_stall',
        'fixed_atomic_voice_intent_durable_write_no_event_loop_stall') else 3

if __name__=='__main__':raise SystemExit(main())
