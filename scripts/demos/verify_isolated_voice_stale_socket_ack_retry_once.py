#!/usr/bin/env python3
"""A replaced old voice socket must not replay a typed turn ACK retry.

A source-grounded no-model test of REAL VoiceSession.control('text_turn') and
its existing-turn lookup: the old socket requested a retry while current,
then blocked at a test-only barrier immediately before database lookup.
A new socket took over. The existing draft means the old code could call
submit_turn despite losing ownership. submit_turn is intentionally replaced
with a hard no-op recorder: NO tool/provider or real SWAAG effect is allowed.
The original previously recorded voice conversation and all other work are
untouched; each test uses one unique private new source process + history.
This is a true *control-path* test, not a genuine model/tool execution.
"""
from __future__ import annotations
import argparse,asyncio,hashlib,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
from verify_isolated_voice_release_server import scrub_config,_write_unique

W70_SHA='f3283c43a196595c4b1fade69e6f2ef0f23943dc6d0b91c5b5ab63f16c17c996'
W71_NEW_SHA='873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf'
SUPPORT={
 'voice_ws_config.py':'b3052253b3d3a22f0f3f69bf6cd264768d6dc0bab99fcd94fbb874b1a98c69ba',
 'voice_ws_core.py':'97e7b1309d675631c3655de1808821b8d11e5b063eed27bd7d4fc981e02d31b9',
 'voice_committed_speech.py':'81ad793b3e138724fe1607489013b2674ec1c2f0745ccdbb2b72a46d3a117c52',
 'voice_speech.py':'85a201a5ac805aba59536b7b527c477315788ef637df76749c1c6d72ed777bfb',
 'voice_swaag_transport.py':'87c6040493e4e6d1c99ef6f2ee1a125337f257f44fd90424d0896d09565dd1f9',
 'voice_log_privacy.py':'907caacd5156d1216c7ef2be4da109e99fb3bf28bb0d41047c46cb42d4342d36',
}
ORIGINAL_HISTORY_SHA='bdde160a4996a1f38e8a88366bf2bb63e6ad3a65b2ebd26687326a03158102fc'
CID='private-w69-storage-race-A'
OPTIN='SWAAG_W71_ONEUSE_DELAYED_OLD_SOCKET_ACK_RETRY_NOACTION'


def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def verify_source(src:Path,original:Path,template:Path,expected_sha:str):
    if src.is_symlink() or not src.is_dir() or original.is_symlink() or not original.is_file() or template.is_symlink() or not template.is_file():
        raise ValueError('Source/history/template not safe private regular files')
    for name,want in {**SUPPORT,'cloudflare_ws_voice_server.py':expected_sha}.items():
        p=src/name
        if p.is_symlink() or not p.is_file() or sha(p)!=want:
            raise ValueError('Unqualified model-free source module: '+name)
    if sha(original)!=ORIGINAL_HISTORY_SHA:
        raise ValueError('Original W69 immutable history has changed')
    data=json.loads(original.read_text());turns=data.get('dialogue_history') or []
    if [x.get('status') for x in turns]!=['superseded','draft']:
        raise ValueError('Original status unexpectedly changed')
    return data,json.loads(template.read_text())


def run_one(source:Path,history:dict,template:dict,root:Path,expected_sha:str,mode:str):
    d=scrub_config(template,root,0)
    d['server']['port']=19356
    if d['server']['host']!='127.0.0.1' or d['llm']['url']!='http://127.0.0.1:9/run' or d['interaction']['require_verified_speaker_for_auto_send'] is not True:
        raise ValueError('Unsafe network/model configuration')
    _write_unique(root/'fake-only-bearer',b'private frozen no-action fixture\n')
    _write_unique(root/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
    _write_unique(root/'inert-piper.onnx',b'no-model')
    _write_unique(root/'inert-piper.onnx.json',b'{}')
    _write_unique(root/'private-config.json',json.dumps(d,sort_keys=True).encode())
    root_hist=root/'conversations';root_hist.mkdir(mode=0o700,exist_ok=False)
    _write_unique(root_hist/(CID+'.json'),json.dumps(history,ensure_ascii=False,indent=2).encode())
    if 'cloudflare_ws_voice_server' in sys.modules:raise ValueError('No imported prior runtime may be mixed')
    sys.path.insert(0,str(source))
    from voice_ws_config import load_voice_ws_config
    from voice_ws_core import VoiceCore
    import cloudflare_ws_voice_server as s
    s.CONFIG=load_voice_ws_config(root/'private-config.json')
    s.CORE=VoiceCore(s.CONFIG)
    class PrivateFakeWs:
        async def send(self,val):
            if not isinstance(val,str):raise ValueError('Unexpected media output')
            j=json.loads(val)
            if j.get('type') in ('audio','turn_proposal') or j.get('event')=='submitted':
                raise AssertionError('Unknown speaker produced external effect')
    async def scenario():
        old=s.VoiceSession(PrivateFakeWs(),CID,client_generation=1)
        s.ACTIVE_BY_CONVERSATION[CID]=old
        text='Private fake typed input with source-backed idempotency'
        request_id='1'*32
        turn_id='turn-client-'+request_id
        # Only private conversation JSON. There is *no* broker nor external
        # system submission. This is the already-received ACK-lost history.
        s.record_draft(CID,turn_id,text,text,'','typed')
        entered=asyncio.Event();release=asyncio.Event();called=[]
        old_task=None;real_block=s.run_blocking
        async def controlled_block(fn,*args,**kw):
            if fn is s.get_turn and asyncio.current_task() is old_task:
                entered.set()
                await asyncio.wait_for(release.wait(),9)
            return await real_block(fn,*args,**kw)
        async def NO_ACTION_submission(*args,**kw):
            # NEVER invoke actual LLM or SWAAG. This proves the original
            # control WOULD have called the tool-submission boundary.
            called.append({'would_submit':True})
            return None
        old.submit_turn=NO_ACTION_submission
        s.run_blocking=controlled_block
        try:
            old_task=asyncio.create_task(old.control(json.dumps({'type':'text_turn',
                'text':text,'client_request_id':request_id})),name='old-delayed-typed-ACK-retry')
            await asyncio.wait_for(entered.wait(),6)
            newer=s.VoiceSession(PrivateFakeWs(),CID,client_generation=2)
            old.superseded=True;old.closed=True
            s.ACTIVE_BY_CONVERSATION[CID]=newer
            release.set()
            await asyncio.wait_for(old_task,6)
            row=s.get_turn(CID,turn_id)
            if not row or row.get('input_source')!='typed' or row.get('recognized_text')!=text:
                raise AssertionError('Private ACK-lost typed source changed')
            if mode=='old_source':
                if len(called)!=1:
                    raise AssertionError('Original stale old source did not attempt a duplicate submission')
                status='old_source_would_repeat_ack_lost_turn_after_socket_replacement'
            elif mode=='corrected':
                if called:raise AssertionError('Old closed socket still attempted a tool submission')
                status='corrected_closed_source_cannot_retry_submitted_text_after_replacement'
            else:raise ValueError('Unsupported source mode')
            return {'status':status,'old_client_submit_boundary_invoked':len(called),
                'already_durable_user_typed_history_preserved':True,
                'new_socket_has_exclusive_active_owner':s.ACTIVE_BY_CONVERSATION.get(CID) is newer,
                'source_history_statuses':[r.get('status') for r in s._history_read(CID)['dialogue_history']],
                'fake_typed_turn_id':turn_id,
                'real_provider_or_tool_calls':0}
        finally:
            release.set();s.run_blocking=real_block;s.ACTIVE_BY_CONVERSATION.pop(CID,None)
    try:return asyncio.run(asyncio.wait_for(scenario(),timeout=18))
    finally:s.CORE.close();s.CORE=None;s.CONFIG=None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','original-history','template','evidence','source-sha'):
        p.add_argument('--'+n,required=True,type=Path if n!='source-sha' else str)
    p.add_argument('--mode',required=True,choices=['old_source','corrected']);p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    if a.source_sha not in (W70_SHA,W71_NEW_SHA):raise ValueError('Source SHA not qualified')
    hist,cfg=verify_source(a.source,a.original_history,a.template,a.source_sha)
    if not a.run:
        print(json.dumps({'status':'original_new_source_ack_retry_preflight_noaction',
            'source_sha256':a.source_sha,'original_history_SHA256':ORIGINAL_HISTORY_SHA,
            'real_provider_tool_calls':0}))
        return 0
    if os.environ.get(OPTIN)!='1':raise ValueError('Explicit unique one-use opt-in required')
    root=a.evidence
    if root.exists() or root.is_symlink() or not root.is_absolute() or not str(root).startswith('/data/var/swaag-benchmarks/'):
        raise FileExistsError('Previously attempted or outside private result must not be replayed')
    root.mkdir(parents=True,mode=0o700,exist_ok=False);os.chmod(root,0o700)
    _write_unique(root/'attempt.json',json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':a.source_sha,'mode':a.mode,'NO_actual_external_actions':True}).encode())
    start=time.monotonic()
    try:result=run_one(a.source,hist,cfg,root,a.source_sha,a.mode)
    except BaseException as e:result={'status':'failed_or_unknown','failure_type':type(e).__name__,'detail':str(e)[:270]}
    result.update({'source_sha256':a.source_sha,'mode':a.mode,'elapsed_seconds':round(time.monotonic()-start,3),
       'real_microphone_websocket_network_or_voice_model_used':False,'real_tool_calls':0,'live_user_voice_service_changed':False})
    _write_unique(root/'result.json',json.dumps(result,indent=2).encode())
    print(json.dumps({k:result.get(k) for k in ('status','mode','old_client_submit_boundary_invoked','failure_type','elapsed_seconds')},sort_keys=True))
    return 0 if result['status'] in ('old_source_would_repeat_ack_lost_turn_after_socket_replacement',
        'corrected_closed_source_cannot_retry_submitted_text_after_replacement') else 3

if __name__=='__main__':raise SystemExit(main())
