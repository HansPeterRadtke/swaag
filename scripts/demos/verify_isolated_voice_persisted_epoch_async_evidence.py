#!/usr/bin/env python3
"""Independently adjudicate actual nine new W70 private voice runtime trials.

Only read protected one-use result receipts, source, real WebSocket/logged
conversation histories and the genuine W69 durable source. Verify the
source-changed baseline restart failure, the cold-restart repair, slow-
filesystem asyncio responsiveness and four fresh PCM/WebSocket regressions.
No model, phone, provider or live server action is invoked or repeated.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path

from prepare_isolated_voice_persisted_epoch_async_guard import BASE,PATCHED_SERVER_SHA256,patch

SOURCE_W69=BASE['cloudflare_ws_voice_server.py']
ORIGINAL_HISTORY_SHA='bdde160a4996a1f38e8a88366bf2bb63e6ad3a65b2ebd26687326a03158102fc'
VOICE_PCM_SHA='56e81fe7050641490a634e064705673a1f6c61ee2cf46d6b09f33d8270e7fae7'
MANIFEST_SHA='f6372eb49372fde646dbadbc51bdcede078be98008af14d6489542980fbc33bc'
# Real reconnect scenarios open a first client, a replacement, and in one
# scenario a third after the newer manual socket disconnects. Do NOT
# assume every real scenario has only one connection to the server.
EXPECTED_SESSION_OPENS={
    'old-w69-restart-new-manual-never-before-tested':1,
    'corrected-restart-new-manual-fresh-test':1,
    'async-ws-new_manual_on_reconnect-oneuse':2,
    'async-ws-new_manual_disconnect_before_asr-oneuse':3,
    'async-ws-cancel-oneuse':1,
    'async-ws-idle_reconnect_pending_asr-oneuse':2,
    'corrected-legacy-epochless-manual-restart-oneuse':1,
}
CASES={
 'old-w69-restart-new-manual-never-before-tested':('76b020c9a008f6b2d98bd31ccf32826fca80cb836f52cd135b6a2032558cc5a2','confirmed_unreleased_old_restart_epoch_RESET_bug',['superseded','draft'],'baseline'),
 'corrected-restart-new-manual-fresh-test':('067b704ef334bd980909ab2db170e126364e83f9b5e2278e8b5030fd337080f2','corrected_user_intent_retires_pre_restart_draft',['superseded','superseded'],'corrected'),
 'w69-known-eventloop-stall-once':('3ebf9463466667b0370085882a520be335f25d604e1b41bfb2052870bd9c32d7','observed_prior_voice_event_loop_0_7sec_storage_stall',['superseded','superseded'],'baseline'),
 'w70-fixed-eventloop-responsive-once':('c3c5f9c8f7d59c6ae09d96149ebfeff6b1f7905e2b867526f8190359a9b966e0','fixed_atomic_voice_intent_durable_write_no_event_loop_stall',['superseded','superseded'],'corrected'),
 'async-ws-new_manual_on_reconnect-oneuse':('ded9994c78f015963bbac34bb1bf222dbb11f94318646a655651d9d1f801aa9d','no_stale_draft_after_cancel_or_new_manual',[],'corrected'),
 'async-ws-new_manual_disconnect_before_asr-oneuse':('8a56e6245e272329e2dbc35c8a5da019fefeb8490fa593c7a0482ec6ede4f35e','no_stale_draft_after_cancel_or_new_manual',[],'corrected'),
 'async-ws-cancel-oneuse':('ccd1141317d108d70de1c6fd6e10d0acdc52ea106fc8b675a6dd733e18b63d40','no_stale_draft_after_cancel_or_new_manual',[],'corrected'),
 'async-ws-idle_reconnect_pending_asr-oneuse':('beeb39e95438f03afcc11de75ae9b028fa0ebb97b7278382f856e68ca9c1901f','idle_pre_reconnect_ASR_held_for_manual_review',['draft'],'corrected'),
 'corrected-legacy-epochless-manual-restart-oneuse':('a3a9b9d9226485b119eca018051caa708cce48d9f1eccda99fe64985a1c607c9','corrected_user_intent_retires_pre_restart_draft',['superseded','superseded'],'corrected'),
}

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def regular(p:Path)->None:
    if p.is_symlink() or not p.is_file():raise ValueError('Required private evidence missing/symlinked: '+str(p))


def verify_sources(baseline:Path,candidate:Path,history:Path)->dict:
    if baseline.is_symlink() or candidate.is_symlink() or not baseline.is_dir() or not candidate.is_dir():
        raise ValueError('No known private baseline/new source directories')
    for n,want in BASE.items():
        a=baseline/n;b=candidate/n;regular(a);regular(b)
        if sha(a)!=want:raise ValueError('Changed original W69 source module: '+n)
        expected=patch(a.read_bytes()) if n=='cloudflare_ws_voice_server.py' else a.read_bytes()
        if b.read_bytes()!=expected:raise ValueError('Candidate not exact deterministic single-module patch: '+n)
    if sha(candidate/'cloudflare_ws_voice_server.py')!=PATCHED_SERVER_SHA256:
        raise ValueError('W70 source final SHA mismatch')
    manifest=candidate/'source-manifest.json';regular(manifest)
    if sha(manifest)!=MANIFEST_SHA:raise ValueError('W70 provenance manifest bytes changed')
    v=json.loads(manifest.read_text())
    if v.get('new_server_sha256')!=PATCHED_SERVER_SHA256 or v.get('only_changed_module')!='cloudflare_ws_voice_server.py':
        raise ValueError('Expected only one modified voice server source module')
    regular(history)
    if sha(history)!=ORIGINAL_HISTORY_SHA:raise ValueError('Old real W69 durable conversation source changed')
    orig=json.loads(history.read_text())['dialogue_history']
    if ([x['status'] for x in orig]!=['superseded','draft']
        or [x.get('asr_intent_epoch') for x in orig]!=[1,2]
        or any(x.get('submitted_text') or x.get('assistant') for x in orig)):
        raise ValueError('Original source conversation not the genuine persisted result')
    return {'W69_original_SHA256':SOURCE_W69,'W70_new_SHA256':PATCHED_SERVER_SHA256,
        'original_history_SHA256':ORIGINAL_HISTORY_SHA,'unchanged_six_modules':True,
        'candidate_manifest_SHA256':MANIFEST_SHA}


def check_case(root:Path,name:str,meta:tuple)->dict:
    want,status,expected_rows,family=meta
    folder=root/name
    if not folder.is_dir() or folder.is_symlink():raise ValueError('No genuine private one-use case: '+name)
    result=folder/'result.json';regular(result)
    if sha(result)!=want:raise ValueError('Immutable one-use receipt modified: '+name)
    d=json.loads(result.read_text())
    expected_source=SOURCE_W69 if family=='baseline' else PATCHED_SERVER_SHA256
    if d.get('status')!=status or d.get('source_sha256')!=expected_source:
        raise ValueError('Case source or result code differs: '+name)
    if d.get('error_type') or d.get('failure_type') or d.get('real_provider_model_tools_calls',0)!=0:
        raise ValueError('Incomplete or unexpected live provider side effect: '+name)
    history=[]
    conv=folder/'conversations'
    if not conv.is_dir() or conv.is_symlink():raise ValueError('Actual private disk history missing: '+name)
    for f in conv.glob('*.json'):
        regular(f)
        if f.stat().st_size>500000:raise ValueError('Unexpected oversized source history')
        history.extend(json.loads(f.read_text()).get('dialogue_history') or [])
    # The stalled heartbeat test and restarter each write one source history.
    rows=[v.get('status') for v in history]
    if rows!=expected_rows:raise ValueError('Real persisted new/old turn state differs: '+name)
    if any(v.get('submitted_text') or v.get('assistant') or v.get('assistant_audio_path') for v in history):
        raise ValueError('User text reached real agent/assistant result: '+name)
    if 'eventloop' in name:
        if d.get('private_durable_JSON_write_count')!=1:
            raise ValueError('The measured stall did not execute one true storage write')
        tick=d.get('max_event_loop_tick_gap_seconds')
        if type(tick) not in (int,float) or not 0<tick<10 or not 0.65<=d.get('new_manual_intent_completion_seconds',0)<=2:
            raise ValueError('No source-measured heartbeat/slow-storage timing')
        if (family=='baseline' and tick<.55) or (family=='corrected' and tick>.28):
            raise ValueError('Old/new actual event-loop responsiveness not demonstrated')
        if d.get('no_network_or_provider_calls') is not True:
            raise ValueError('Event loop input unexpectedly reached a model or network')
    else:
        log=folder/'voice-private.jsonl';regular(log)
        events=[json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        count=Counter(e.get('event') for e in events)
        if (count['server_start']!=1
            or count['session_open']!=EXPECTED_SESSION_OPENS[name]):
            raise ValueError('Real private source process/client replacement count mismatch: '+name)
        if any(count[k] for k in ('turn_submitted','asr_auto_submitted','answer_start','worker_started')):
            raise ValueError('Unexpected user or tool action logged: '+name)
        config=folder/'private-config.json';regular(config)
        cfg=json.loads(config.read_text())
        if (cfg['server']['host']!='127.0.0.1' or cfg['interaction']['require_verified_speaker_for_auto_send'] is not True
            or not cfg['server']['auth_token_file'].startswith(str(folder))
            or cfg['llm']['url']!='http://127.0.0.1:9/run'):
            raise ValueError('Real WebSocket server was not localhost/manual-review only')
        for group in ('asr','llm','tts'):
            for key,value in cfg[group].items():
                if (key.endswith('_url') or key=='url') and value:
                    if not str(value).startswith('http://127.0.0.1:'):
                        raise ValueError('Unexpected provider endpoint in no-action test')
        pid=d.get('private_owned_child_pid') or d.get('actual_private_child_pid')
        if type(pid) is not int or pid<=1:
            raise ValueError('No actual private server child process recorded')
        proc=Path('/proc')/str(pid)/'cmdline'
        if proc.is_file() and b'cloudflare_ws_voice_server.py' in proc.read_bytes():
            raise ValueError('Test-owned private server child still active')
        if name.startswith('async-ws-'):
            if (d.get('held_authoritative_ASR_requests')!=1
                or d.get('input_recording_SHA256')!=VOICE_PCM_SHA
                or d.get('real_model_ASR_TTS_agent_tool_calls')!=0
                or d.get('actual_Jetson_production_service_changed') is not False):
                raise ValueError('Real one-use held ASR replay scope invalid')
        else:
            if d.get('saved_history_statuses_after_new_manual')!=expected_rows:
                raise ValueError('Actual on-wire manual input failed to update history')
        if name=='corrected-legacy-epochless-manual-restart-oneuse':
            if d.get('legacy_epochless_derived_fixture_is_NOT_original_source_bytes') is not True:
                raise ValueError('Derived missing-epoch legacy fixture falsely treated as original')
    return {'test':name,'status':status,'sha256':want,'source_sha256':expected_source,
       'recorded_durable_statuses':rows,
       'event_loop_max_gap_seconds':d.get('max_event_loop_tick_gap_seconds'),
       'total_wall_seconds':d.get('total_wall_seconds') or d.get('elapsed_seconds') or d.get('wall_seconds'),
       'real_private_source_orchestrator_effects':0}


def verify(root:Path,base:Path,new:Path,history:Path)->dict:
    if not root.is_dir() or root.is_symlink():raise ValueError('No private results source root')
    sources=verify_sources(base,new,history)
    cases=[check_case(root,n,req) for n,req in CASES.items()]
    old_fair=next(v for v in cases if v['test']=='w69-known-eventloop-stall-once')['event_loop_max_gap_seconds']
    new_fair=next(v for v in cases if v['test']=='w70-fixed-eventloop-responsive-once')['event_loop_max_gap_seconds']
    if old_fair/new_fair<10:raise ValueError('True event-loop heartbeat improvement below ten-fold')
    return {'status':'independently_verified_real_persisted_epoch_recovery_and_heartbeat_fairness',
       'checked_at_utc':datetime.now(timezone.utc).isoformat(),
       'original_reproduced_restart_draft_bug':True,
       'new_source_retires_stale_draft_after_process_restart':True,
       'new_source_does_not_block_other_voice_heartbeats_during_slow_fs_sync':True,
       'source':sources,'actual_separate_bounded_test_cases':len(cases),
       'prior_old_event_loop_gap_seconds':old_fair,'new_async_gap_seconds':new_fair,
       'observed_event_loop_gap_reduction_factor':round(old_fair/new_fair,2),
       'actual_cases':cases,'new_Whisper_LLM_TTS_or_agent_actions':0,
       'actual_owner_phone_or_live_Jetson_deployment':False,
       'limits':['Fixed 0.70s injected disk delay in a private interpreter, not a host-wide production SLA',
                 'Server source tested in nine discrete non-tool cases, not human live streaming microphone',
                 'Server code not deployed, and owner voice authentication and anti-recording-replay not accepted']}


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','baseline','candidate','original-history','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv)
    j=verify(a.evidence,a.baseline,a.candidate,a.original_history)
    if a.output.exists() or a.output.is_symlink() or not a.output.is_absolute() or not str(a.output).startswith('/data/var/swaag-benchmarks/'):
        raise FileExistsError('Unique private verifier output required, no duplicate evidence overwrite')
    fd=os.open(a.output,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(j,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':j['status'],'genuine_cases':j['actual_separate_bounded_test_cases'],
        'old_gap_s':j['prior_old_event_loop_gap_seconds'],'new_gap_s':j['new_async_gap_seconds'],
        'replayed_owner_or_tool_effects':0}))
    return 0

if __name__=='__main__':raise SystemExit(main())
