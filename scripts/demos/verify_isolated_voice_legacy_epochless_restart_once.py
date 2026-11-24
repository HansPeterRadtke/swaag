#!/usr/bin/env python3
"""New distinct Jetson WS test: legacy history without speech epoch metadata.

Use the immutable W69 actual stored private conversation, which contains an
older superseded speech draft at epoch one and a legitimate newer draft at
epoch two. A fresh server process then receives an actual WebSocket
begin_manual_utterance message. Is the old resumed draft retired? Tests
previously missing restart+fresh manual case, NOT another replay of prior
speech/model/ASR. All endpoints are private localhost dummy, no user token.
"""
from __future__ import annotations
import argparse,asyncio,hashlib,json,os,pathlib,re,sys,time
from datetime import datetime,timezone
from pathlib import Path
import verify_isolated_voice_atomic_draft_restart_once as predecessor
from verify_isolated_voice_release_server import _write_unique

ORIGINAL_HISTORY_SHA='bdde160a4996a1f38e8a88366bf2bb63e6ad3a65b2ebd26687326a03158102fc'
W69_SOURCE_SHA='3d4e6e029aa4a467684c618dc31cfd60d53d77939843f547d394b04a979bad2f'
SUPPORTING_SOURCE_SHA={k:v for k,v in predecessor.BASE.items() if k!='cloudflare_ws_voice_server.py'}
OPTIN='SWAAG_W70_LEGACY_EPOCHLESS_RECOVERY_PRIVATE_ONCE'


def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()


def pin(source:Path,history:Path,template:Path,expected_source_sha:str):
    if any(x.is_symlink() for x in (source,history,template)):
        raise ValueError('Symlinked test source/history/template refused')
    if not source.is_dir() or not history.is_file() or not template.is_file():
        raise ValueError('Missing immutable source/history/config')
    if sha(history)!=ORIGINAL_HISTORY_SHA:raise ValueError('Original real W69 history SHA changed')
    for name,old_sha in {**SUPPORTING_SOURCE_SHA,'cloudflare_ws_voice_server.py':expected_source_sha}.items():
        f=source/name
        if not f.is_file() or f.is_symlink() or sha(f)!=old_sha:
            raise ValueError('Unknown server private module '+name)
    d=json.loads(history.read_text())
    v=d.get('dialogue_history') or []
    if (len(v)!=2 or [x.get('status') for x in v]!=['superseded','draft']
        or [x.get('asr_intent_epoch') for x in v]!=[1,2]
        or [x.get('turn_id') for x in v]!=['turn-old','turn-current']
        or any(x.get('submitted_text') or x.get('assistant') for x in v)):
        raise ValueError('Not authentic unsubmitted W69 durable source')
    return json.loads(template.read_text()),d


async def scenario(port:int,mode:str,rows:list[dict])->dict:
    from websockets.legacy.client import connect
    url=f'ws://127.0.0.1:{port}/ws?conversation={predecessor.SOURCE_CID}&generation=12'
    headers={'Authorization':'Bearer '+predecessor.FAKE_BEARER}
    messages=[]
    async def recv(ws,stop,timeout=6):
        end=asyncio.get_running_loop().time()+timeout
        while True:
            left=end-asyncio.get_running_loop().time()
            if left<=0:raise TimeoutError('Expected protected WS event did not arrive')
            raw=await asyncio.wait_for(ws.recv(),left)
            if not isinstance(raw,str):raise AssertionError('Unexpected binary TTS/playback')
            data=json.loads(raw)
            if (data.get('type') in ('turn_proposal','audio')
                or data.get('event')=='submitted'):
                raise AssertionError('Untrusted voice unexpectedly executed an action')
            messages.append({key:data.get(key) for key in ('type','state','reason','source','event') if key in data})
            if stop(data):return data
    async with connect(url,extra_headers=headers,open_timeout=4,close_timeout=1,max_size=2**20) as ws:
        h=await recv(ws,lambda v:v.get('type')=='hello')
        if h.get('unverified_speaker_requires_review') is not True:
            raise AssertionError('Unverified speech speaker safeguard absent')
        history=await recv(ws,lambda v:v.get('type')=='history')
        if [row.get('status') for row in history.get('turns') or []]!=['superseded','draft']:
            raise AssertionError('Cold restarted server did not load exact old+new source')
        restored=await recv(ws,lambda v:v.get('type')=='asr' and v.get('restored') is True)
        if restored.get('turn_id')!='turn-current':raise AssertionError('Wrong resumed voice draft')
        await recv(ws,lambda v:v.get('type')=='state' and v.get('reason')=='restored_draft')
        # NEW TEST INTENT. No PCM uploaded, no ASR/remote model call.
        await ws.send(json.dumps({'type':'begin_manual_utterance'}))
        await recv(ws,lambda v:v.get('type')=='state' and v.get('state')=='transcribing'
            and v.get('source')=='manual_recording',6)
        await ws.send(json.dumps({'type':'history_get'}))
        updated=await recv(ws,lambda v:v.get('type')=='history',7)
        statuses=[v.get('status') for v in updated.get('turns') or []]
        if mode=='baseline_old':
            if statuses!=['superseded','draft']:
                raise AssertionError('W69 prior-source reset bug DID NOT reproduce: '+str(statuses))
            status='confirmed_unreleased_old_restart_epoch_RESET_bug'
        elif mode=='corrected':
            if statuses!=['superseded','superseded']:
                raise AssertionError('Corrected version left prior editable draft active: '+str(statuses))
            status='corrected_user_intent_retires_pre_restart_draft'
        else:raise ValueError('Unknown test model scenario')
    return {'status':status,'saved_history_statuses_after_new_manual':statuses,
        'old_draft_epoch':rows[-1]['asr_intent_epoch'],
        'new_manual_delivered_by_real_WebSocket':True,
        'recorded_server_events':messages,'real_model_ASR_TTS_tool_calls':0}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','original-history','template','evidence','expected-server-sha'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--mode',required=True,choices=['baseline_old','corrected'])
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    source=Path(a.source);history=Path(a.original_history);template=Path(a.template);root=Path(a.evidence)
    if not re.fullmatch('[0-9a-f]{64}',a.expected_server_sha):raise ValueError('Invalid immutable SHA')
    original,seed=pin(source,history,template,a.expected_server_sha)
    # Construct a deliberately DIFFERENT, private legacy historical input
    # from exact hash-pinned W69 real test history. Neither that historic
    # source nor the original completed W70 experiment is altered/replayed.
    seed=json.loads(json.dumps(seed))
    for item in seed['dialogue_history']:
        item['asr_intent_epoch']=None
        item.pop('superseded_by_intent_epoch',None)
    if seed['dialogue_history'][-1]['status']!='draft':
        raise ValueError('The synthetic legacy fixture must contain an old draft')
    if not a.run:
        print(json.dumps({'status':'source_pinned_fresh_manual_after_restart_preflight_only',
            'original_persisted_epoch':[x.get('asr_intent_epoch') for x in seed['dialogue_history']],
            'source_sha256':a.expected_server_sha,'no_actions':True}))
        return 0
    if os.environ.get(OPTIN)!='1':raise ValueError('Explicit single private test opt-in required')
    if root.exists() or root.is_symlink() or not root.is_absolute() or not str(root).startswith('/data/var/swaag-benchmarks/'):
        raise FileExistsError('Refuse any uncertain or unapproved existing source test')
    root.mkdir(parents=True,mode=0o700,exist_ok=False)
    os.chmod(root,0o700)
    _write_unique(root/'attempt.json',json.dumps({'at':datetime.now(timezone.utc).isoformat(),
       'mode':a.mode,'source':a.expected_server_sha,'history_sha':ORIGINAL_HISTORY_SHA,
       'real_ASR_model_or_tool_calls':0,
       'derived_old_speech_rows_have_no_revision_metadata':True}).encode())
    start=time.monotonic()
    try:
        # Use pinned real server process/harness, only its *new scenario*
        # is changed; seed private history is copied, never modified.
        predecessor.scenario=scenario
        r=predecessor.one_child(source,original,seed,root,a.mode)
    except BaseException as e:
        r={'status':'failed_or_unknown','error_type':type(e).__name__,'error_private':str(e)[:220]}
    r.update({'mode':a.mode,'wall_seconds':round(time.monotonic()-start,3),
        'source_sha256':a.expected_server_sha,'history_sha256':ORIGINAL_HISTORY_SHA,
        'legacy_epochless_derived_fixture_is_NOT_original_source_bytes':True,
        'live_Jetson_or_phone_changed':False,'real_provider_model_tools_calls':0})
    _write_unique(root/'result.json',json.dumps(r,indent=2,ensure_ascii=False).encode())
    print(json.dumps({k:r.get(k) for k in ('status','mode','saved_history_statuses_after_new_manual','error_type','wall_seconds')}))
    return 0 if r['status'] in ('confirmed_unreleased_old_restart_epoch_RESET_bug','corrected_user_intent_retires_pre_restart_draft') else 3

if __name__=='__main__':
    raise SystemExit(main())
