#!/usr/bin/env python3
"""One-use genuine source-coroutine test for delayed superseded client control.

No model, real WS listener, ASR, TTS, user phone, provider or tool call. This
is a **real original VoiceSession.control coroutine and VoiceCore fsync test**,
NOT a physical on-wire/socket test. The predecessor W70 actually finished
its server reconnect tests; DO NOT rerun those. Here a previously accepted
old socket 'draft_claim' is delayed before its executor has taken the lock.
A new mobile socket replaces it and begins manual recording. If the late old
control invalidates the new session's intent, genuine later speech is lost.
"""
from __future__ import annotations
import argparse,asyncio,hashlib,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
from verify_isolated_voice_release_server import scrub_config,_write_unique

BASE={
 'cloudflare_ws_voice_server.py':'f3283c43a196595c4b1fade69e6f2ef0f23943dc6d0b91c5b5ab63f16c17c996',
 'voice_ws_config.py':'b3052253b3d3a22f0f3f69bf6cd264768d6dc0bab99fcd94fbb874b1a98c69ba',
 'voice_ws_core.py':'97e7b1309d675631c3655de1808821b8d11e5b063eed27bd7d4fc981e02d31b9',
 'voice_committed_speech.py':'81ad793b3e138724fe1607489013b2674ec1c2f0745ccdbb2b72a46d3a117c52',
 'voice_speech.py':'85a201a5ac805aba59536b7b527c477315788ef637df76749c1c6d72ed777bfb',
 'voice_swaag_transport.py':'87c6040493e4e6d1c99ef6f2ee1a125337f257f44fd90424d0896d09565dd1f9',
 'voice_log_privacy.py':'907caacd5156d1216c7ef2be4da109e99fb3bf28bb0d41047c46cb42d4342d36',
}
# Freeze alternate fixed source after private deterministic rendering.
NEW_SERVER_SHA='873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf'
ORIGINAL_HISTORY_SHA='bdde160a4996a1f38e8a88366bf2bb63e6ad3a65b2ebd26687326a03158102fc'
OPTIN='SWAAG_W71_ONEUSE_OLD_SOCKET_CONTROL_OUT_OF_ORDER_SOURCE'
CID='private-w69-storage-race-A'


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def check(source:Path,history:Path,template:Path,chosen:str):
    if source.is_symlink() or history.is_symlink() or template.is_symlink():raise ValueError('Symlinked source/history not allowed')
    if not source.is_dir() or not history.is_file() or not template.is_file():raise ValueError('Missing source or durable history')
    if sha(history)!=ORIGINAL_HISTORY_SHA:raise ValueError('Historic real durable source SHA mismatch')
    for name,expected in BASE.items():
        p=source/name
        want=chosen if name=='cloudflare_ws_voice_server.py' else expected
        if not p.is_file() or p.is_symlink() or sha(p)!=want:raise ValueError('Not pinned genuine source '+name)
    d=json.loads(history.read_text());rows=d.get('dialogue_history') or []
    if len(rows)!=2 or [x.get('status') for x in rows]!=['superseded','draft'] or [x.get('asr_intent_epoch') for x in rows]!=[1,2]:
        raise ValueError('Not the previous real voice durable history')
    return d,json.loads(template.read_text())


def run_one(source:Path,history:dict,original_config:dict,root:Path,chosen:str,mode:str):
    c=scrub_config(original_config,root,0)
    c['server']['port']=19357
    if c['server']['host']!='127.0.0.1' or c['llm']['url']!='http://127.0.0.1:9/run' or c['interaction']['require_verified_speaker_for_auto_send'] is not True:
        raise RuntimeError('No real provider or unguarded speaker allowed')
    _write_unique(root/'fake-only-bearer',b'W71 synthetic session authorization only\n')
    _write_unique(root/'inert-piper',b'#!/bin/sh\nexit 127\n',0o700)
    _write_unique(root/'inert-piper.onnx',b'NO TTS MODEL')
    _write_unique(root/'inert-piper.onnx.json',b'{}')
    _write_unique(root/'private-config.json',json.dumps(c,sort_keys=True).encode())
    conv=root/'conversations';conv.mkdir(mode=0o700,exist_ok=False)
    _write_unique(conv/(CID+'.json'),json.dumps(history,ensure_ascii=False,indent=2).encode())
    if 'cloudflare_ws_voice_server' in sys.modules:raise ValueError('Cannot reuse an already imported runtime source')
    sys.path.insert(0,str(source))
    from voice_ws_config import load_voice_ws_config
    from voice_ws_core import VoiceCore
    import cloudflare_ws_voice_server as s
    s.CONFIG=load_voice_ws_config(root/'private-config.json');s.CORE=VoiceCore(s.CONFIG)
    class FakePrivateSessionSocket:
        def __init__(self):self.events=[]
        async def send(self,payload):
            if not isinstance(payload,str):raise AssertionError('Unexpected TTS/media/binary event')
            event=json.loads(payload)
            if event.get('type') in ('audio','turn_proposal') or event.get('event')=='submitted':
                raise AssertionError('Unverified speaker attempted forbidden operation')
            self.events.append({k:event.get(k) for k in ('type','state','event','reason','turn_id') if k in event})
    async def scenario():
        old= s.VoiceSession(FakePrivateSessionSocket(),CID,client_generation=1)
        s.ACTIVE_BY_CONVERSATION[CID]=old
        control_entered=asyncio.Event();release_control=asyncio.Event();state={}
        previous_run=s.run_blocking
        old_task=None
        async def delay_old_advance(fn,*args,**kwargs):
            # Exact asynchronous W70 source accepts old draft_claim and
            # awaits its executor. Test-only barrier stops at the boundary
            # BEFORE execution of the durable epoch mutation.
            if fn is s.advance_conversation_intent_epoch and asyncio.current_task() is old_task:
                control_entered.set()
                await asyncio.wait_for(release_control.wait(),timeout=10)
            return await previous_run(fn,*args,**kwargs)
        s.run_blocking=delay_old_advance
        try:
            old_task=asyncio.create_task(old.control(json.dumps({'type':'draft_claim'})),name='old-source-already-admitted-draft-claim')
            await asyncio.wait_for(control_entered.wait(),6)
            new=s.VoiceSession(FakePrivateSessionSocket(),CID,client_generation=2)
            old.superseded=True;old.closed=True;s.ACTIVE_BY_CONVERSATION[CID]=new
            await new.control(json.dumps({'type':'begin_manual_utterance'}))
            if not new.manual_utterance_upload or new.intent_epoch!=3:
                raise AssertionError('New user manual source did not receive expected durable new input epoch')
            state['new_epoch_before_old_released']=new.intent_epoch
            release_control.set()
            await asyncio.wait_for(old_task,8)
            state['epoch_after_delayed_old_control']=s.CONVERSATION_INTENT_EPOCH[CID]
            # This is the genuine source durable draft operation that follows
            # a recognized new manual user utterance, not fake model output.
            accepted=s.record_draft(CID,'new-manual-spoken-text',
                'Fresh manual utterance','Fresh manual utterance',
                str(root/'private-new-manual-audio-path-placeholder.wav'),'speech',
                expected_intent_epoch=new.intent_epoch)
            state['new_manually_captured_user_speech_accepted']=accepted
            status=[x.get('status') for x in s._history_read(CID)['dialogue_history']]
            state['stored_history_statuses']=status
            state['old_client_fake_events']=old.ws.events
            state['new_client_fake_events']=new.ws.events
            state['real_source_control_methods_invoked']=2
            if mode=='old_source':
                if accepted is not False or state['epoch_after_delayed_old_control']!=4:
                    raise AssertionError('Original W70 source did not reproduce stale-socket invalidation')
                state['status']='confirmed_old_superseded_socket_invalidated_fresh_manual_speech'
            elif mode=='corrected':
                if accepted is not True or state['epoch_after_delayed_old_control']!=3:
                    raise AssertionError('Corrected server let superseded socket invalidate fresh manual audio')
                state['status']='corrected_superseded_old_control_does_not_invalidate_new_speech'
            else:raise ValueError('Unknown distinct one-use test mode')
            if (status[:2]!=['superseded','superseded'] or any(
                item.get('submitted_text') or item.get('assistant') for item in s._history_read(CID)['dialogue_history'])):
                raise AssertionError('New user/speaker turns were submitted or old speech revived')
            return state
        finally:
            release_control.set()
            s.run_blocking=previous_run
            s.ACTIVE_BY_CONVERSATION.pop(CID,None)
    try: result=asyncio.run(asyncio.wait_for(scenario(),timeout=22))
    finally:s.CORE.close();s.CORE=None;s.CONFIG=None
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','original-history','template','evidence','source-sha'):
        p.add_argument('--'+n,required=True,type=Path if n!='source-sha' else str)
    p.add_argument('--mode',required=True,choices=['old_source','corrected'])
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    if a.source_sha not in (BASE['cloudflare_ws_voice_server.py'],NEW_SERVER_SHA):
        raise ValueError('Unqualified/private source SHA not admitted')
    historical,c=check(a.source,a.original_history,a.template,a.source_sha)
    if not a.run:
        print(json.dumps({'status':'private_inflight_socket_source_test_preflight_only',
            'source_sha':a.source_sha,'durable_history_epochs':[x.get('asr_intent_epoch') for x in historical['dialogue_history']],
            'no_live_or_model_actions':True}))
        return 0
    if os.getenv(OPTIN)!='1':raise ValueError('One-use noaction test requires explicit opt-in')
    root=a.evidence
    if root.exists() or root.is_symlink() or not root.is_absolute() or not str(root).startswith('/data/var/swaag-benchmarks/'):
        raise FileExistsError('Prior/unknown test effect must never be replayed')
    root.mkdir(mode=0o700,parents=True,exist_ok=False);os.chmod(root,0o700)
    _write_unique(root/'attempt.json',json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':a.source_sha,'mode':a.mode,'no_audio_model_ASR_TTS_SWAAG_provider_or_tools':True}).encode())
    start=time.monotonic()
    try:result=run_one(a.source,historical,c,root,a.source_sha,a.mode)
    except BaseException as e:
        result={'status':'failed_or_unknown','failure_type':type(e).__name__,
           'failure_detail_private':str(e)[:250]}
    result.update({'source_sha256':a.source_sha,'mode':a.mode,
      'elapsed_seconds':round(time.monotonic()-start,3),
      'only_private_source_coroutines_and_true_durable_JSON':True,
      'real_audio_recording_or_websocket_network_USED':False,
      'real_model_ASR_TTS_agent_tool_actions':0,'live_voice_phone_modified':False})
    _write_unique(root/'result.json',json.dumps(result,indent=2).encode())
    print(json.dumps({k:result.get(k) for k in ('status','mode','failure_type','new_epoch_before_old_released','epoch_after_delayed_old_control','new_manually_captured_user_speech_accepted','elapsed_seconds')},sort_keys=True))
    return 0 if result['status'] in ('confirmed_old_superseded_socket_invalidated_fresh_manual_speech',
      'corrected_superseded_old_control_does_not_invalidate_new_speech') else 3

if __name__=='__main__':raise SystemExit(main())
