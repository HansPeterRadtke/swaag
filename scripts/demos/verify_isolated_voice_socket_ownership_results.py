#!/usr/bin/env python3
"""Read-only independent source/disk adjudication of eight Wave71 new cases.

Two genuine W70 control-coroutine failures were observed (a late predecessor
control invalidated new PCM; a late ACK retry reached a no-op submit recorder).
A separate new source revision blocked both. Four truly separate private
Jetson WebSocket/PCM server processes verified no regressions. This reader
NEVER replays the one-use private source operations or contacts providers.
The first four are real source/disk coroutine tests, not network WS tests.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path

from prepare_isolated_voice_stale_socket_intent_owner_guard import BASE,NEW_SHA,patch

BASELINE=BASE['cloudflare_ws_voice_server.py']
ORIGINAL_HISTORY_SHA='bdde160a4996a1f38e8a88366bf2bb63e6ad3a65b2ebd26687326a03158102fc'
VOICE_SHA='56e81fe7050641490a634e064705673a1f6c61ee2cf46d6b09f33d8270e7fae7'
MANIFEST_SHA='f361faa9283b55e708f4c7f590d3b0e5a90bd1eeb9aa16fe2f88876e8c6df6d0'
SCENARIOS={
 'original-w70-inflight-old-control-oneuse':('ca9d663171a1a0e4c1ce047e40db1ec267394e7df239f6385e28af97b2e3b79a','confirmed_old_superseded_socket_invalidated_fresh_manual_speech','old_control',BASELINE),
 'corrected-v2-v2-verify_isolated_voice_old_socket_control_epoch_once':('31543580fd540b432c36e760fe4de89ae21e0bbf0f5cc3aa02999a4e1072455e','corrected_superseded_old_control_does_not_invalidate_new_speech','new_control',NEW_SHA),
 'old-source-stale-text-idempotency-once':('d6080dcfb72adede6bdcc357959a4d6fc936b5a300e1fbb598cc919f7041a09d','old_source_would_repeat_ack_lost_turn_after_socket_replacement','old_retry',BASELINE),
 'corrected-v2-v2-verify_isolated_voice_stale_socket_ack_retry_once':('6d89895e71e811b70f326529092f625b3c462700ed0ab0e13638243a6e4d28a4','corrected_closed_source_cannot_retry_submitted_text_after_replacement','new_retry',NEW_SHA),
 'guarded-ws-new_manual_on_reconnect-once':('9324dd4e51f965554efb51284f95dd3eec17e108c97ee93006195d00973b5175','no_stale_draft_after_cancel_or_new_manual','new_WS',NEW_SHA,2,0),
 'guarded-ws-new_manual_disconnect_before_asr-once':('25c4b87ae64167c66041612d1f07148af491716c2e0f62fe059cb27831828747','no_stale_draft_after_cancel_or_new_manual','new_WS',NEW_SHA,3,0),
 'guarded-ws-cancel-once':('e0fd5ee783622de11a4d7d51d58bbd105d6bb9e3fd0644fc2964f7e523028d3c','no_stale_draft_after_cancel_or_new_manual','new_WS',NEW_SHA,1,0),
 'guarded-ws-idle_reconnect_pending_asr-once':('58af06a4859578c19364b8c3e36249472bab7137499c80efa12158b858f8c6ba','idle_pre_reconnect_ASR_held_for_manual_review','new_WS',NEW_SHA,2,1),
}

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def regular(p:Path)->None:
    if not p.is_file() or p.is_symlink():raise ValueError('Missing, altered or symlinked private evidence: '+str(p))


def check_source(old:Path,new:Path,history:Path)->dict:
    if not old.is_dir() or old.is_symlink() or not new.is_dir() or new.is_symlink():raise ValueError('Not an actual private source directory')
    for name,expected in BASE.items():
        p=old/name;q=new/name;regular(p);regular(q)
        if sha(p)!=expected:raise ValueError('Original source byte changed: '+name)
        newbytes=patch(p.read_bytes()) if name=='cloudflare_ws_voice_server.py' else p.read_bytes()
        if q.read_bytes()!=newbytes:raise ValueError('Exactly one modified source file was not preserved: '+name)
    if sha(new/'cloudflare_ws_voice_server.py')!=NEW_SHA:raise ValueError('Unqualified candidate server source')
    manifest=new/'source-manifest.json';regular(manifest)
    if sha(manifest)!=MANIFEST_SHA:raise ValueError('Source provenance manifest changed')
    m=json.loads(manifest.read_text())
    if m.get('new_undeployed_server_sha256')!=NEW_SHA or m.get('only_changed_file')!='cloudflare_ws_voice_server.py':
        raise ValueError('Private source evidence does not describe exact one-module patch')
    regular(history)
    if sha(history)!=ORIGINAL_HISTORY_SHA:raise ValueError('Actual W69 original private history bytes changed')
    h=json.loads(history.read_text())['dialogue_history']
    if ([v.get('status') for v in h]!=['superseded','draft']
        or [v.get('asr_intent_epoch') for v in h]!=[1,2]
        or any(v.get('submitted_text') or v.get('assistant') for v in h)):
        raise ValueError('Prior original case not a safe source-backed private history')
    return {'original_W70_source_SHA256':BASELINE,'new_W71_source_SHA256':NEW_SHA,
      'original_history_sha256':ORIGINAL_HISTORY_SHA,'six_supporting_source_modules_unchanged':True,
      'new_source_manifest_sha256':MANIFEST_SHA}


def check_case(root:Path,name:str,expected:tuple)->dict:
    expected_sha,status,family,source_sha,*metadata=expected
    d=root/name
    if d.is_symlink() or not d.is_dir():raise ValueError('No actual prior oneuse receipt '+name)
    p=d/'result.json';regular(p)
    if sha(p)!=expected_sha:raise ValueError('Immutable oneuse receipt SHA modified: '+name)
    result=json.loads(p.read_text())
    # The first four direct coroutine tests and four real WebSocket tests
    # have deliberately different metadata schemas. All must explicitly
    # assert that their actual live owner voice service was NOT changed.
    no_live_change = result.get('live_voice_phone_modified')
    if no_live_change is None:
        no_live_change = result.get('live_user_voice_service_changed')
    if no_live_change is None:
        no_live_change = result.get('actual_Jetson_production_service_changed')
    if (result.get('status')!=status or result.get('source_sha256')!=source_sha or
            result.get('failure_type') is not None or no_live_change is not False):
        raise ValueError('Unverified result state/provider side effect: '+name)
    historic=[]
    path=d/'conversations'
    if not path.is_dir() or path.is_symlink():raise ValueError('No test actual private disk history')
    for f in path.glob('*.json'):
        regular(f)
        historic.extend(json.loads(f.read_text()).get('dialogue_history') or [])
    for row in historic:
        if row.get('submitted_text') or row.get('assistant') or row.get('assistant_audio_path'):
            raise ValueError('Private test unexpectedly executed agent or user request')
    statuses=[r.get('status') for r in historic]
    if family in ('old_control','new_control'):
        expected_statuses=['superseded','superseded']
        if family=='new_control':expected_statuses.append('draft')
        if statuses!=expected_statuses or result.get('real_model_ASR_TTS_agent_tool_actions')!=0:
            raise ValueError('Old/new owner intent real source coroutine result changed')
        assert_status=result.get('new_manually_captured_user_speech_accepted')
        if (family=='old_control' and (assert_status is not False or result.get('epoch_after_delayed_old_control')!=4)) or (
            family=='new_control' and (assert_status is not True or result.get('epoch_after_delayed_old_control')!=3)):
            raise ValueError('Old/new captured speech acceptance was not changed by source fix')
    elif family in ('old_retry','new_retry'):
        if (len(historic)!=3 or len(statuses)!=3 or statuses[-1]!='draft'
            or result.get('real_tool_calls')!=0):
            raise ValueError('No genuine seeded ACK-lost typed turn in own private history')
        expected_calls=1 if family=='old_retry' else 0
        if result.get('old_client_submit_boundary_invoked')!=expected_calls:
            raise ValueError('True before/after stale socket would-call-tool boundary missing')
    elif family=='new_WS':
        expected_connection_count,expected_history_rows=metadata
        if (len(historic)!=expected_history_rows or result.get('last_history_turns')!=expected_history_rows
            or result.get('held_authoritative_ASR_requests')!=1
            or result.get('input_recording_SHA256')!=VOICE_SHA
            or result.get('real_model_ASR_TTS_agent_tool_calls')!=0):
            raise ValueError('Real source-backed recorded voice stream result not matched')
        if historic and (historic[0].get('status')!='draft' or historic[0].get('requires_review') is not True):
            raise ValueError('Unverified owner voice bypassed manual review')
        log=d/'voice-private.jsonl';regular(log)
        counts=Counter(v.get('event') for v in [json.loads(s) for s in log.read_text().splitlines() if s.strip()])
        if counts['server_start']!=1 or counts['session_open']!=expected_connection_count:
            raise ValueError('Actual number of running new WebSocket sessions wrong')
        if any(counts[k] for k in ('turn_submitted','answer_start','worker_started','asr_auto_submitted')):
            raise ValueError('Actual tool submission unexpectedly logged')
        cfg=d/'private-config.json';regular(cfg)
        c=json.loads(cfg.read_text())
        if (c['server']['host']!='127.0.0.1' or c['interaction']['require_verified_speaker_for_auto_send'] is not True
            or not c['server']['auth_token_file'].startswith(str(d))
            or c['llm']['url']!='http://127.0.0.1:9/run'):
            raise ValueError('Actual recorded voice WS not strictly noaction local')
        for group in ('asr','llm','tts'):
            for key,val in c[group].items():
                if (key.endswith('_url') or key=='url') and val and not str(val).startswith('http://127.0.0.1:'):
                    raise ValueError('External provider model URL in test')
        pid=result.get('actual_private_child_pid')
        if type(pid) is not int or pid<=1:raise ValueError('No real private WebSocket server child observed')
        proc=Path('/proc')/str(pid)/'cmdline'
        if proc.is_file() and b'cloudflare_ws_voice_server.py' in proc.read_bytes():
            raise ValueError('Private source server process was not cleaned up')
    else:raise ValueError('Unknown test type')
    return {'case':name,'scenario_type':family,'status':status,
     'source_sha256':source_sha,'original_result_sha256':expected_sha,
     'real_persisted_turn_statuses':statuses,
     'new_user_speech_accepted':result.get('new_manually_captured_user_speech_accepted'),
     'old_retry_noaction_submit_stub_calls':result.get('old_client_submit_boundary_invoked'),
     'real_external_model_tool_or_phone_calls':0}


def verify(root:Path,original:Path,fixed:Path,history:Path)->dict:
    if not root.is_dir() or root.is_symlink():raise ValueError('Missing immutable source evidence directory')
    src=check_source(original,fixed,history)
    cases=[check_case(root,n,t) for n,t in SCENARIOS.items()]
    if (len(cases)!=8 or
        [x['old_retry_noaction_submit_stub_calls'] for x in cases if x['scenario_type'] in ('old_retry','new_retry')]!=[1,0] or
        [x['new_user_speech_accepted'] for x in cases if x['scenario_type'] in ('old_control','new_control')]!=[False,True]):
        raise ValueError('No genuine W70 unsafe-to-W71 safe side-by-side evidence')
    return {'status':'independently_verified_replaced_voice_socket_cannot_overwrite_new_speech_or_retry_typed_turn',
      'checked_utc':datetime.now(timezone.utc).isoformat(),
      'source_pins':src,'observed_original_two_control_failures_without_actual_tool_effects':2,
      'observed_corrected_source_two_control_acceptances':2,
      'observed_new_real_private_PCM_websocket_server_cases':4,
      'total_independently_checked_new_source_or_process_cases':len(cases),
      'cases':cases,'no_new_model_ASR_TTS_or_SWAAG_tool_calls':True,
      'actual_owner_phone_or_production_Jetson_modified':False,
      'limitations':['The two delayed predecessor-control tests exercise real source coroutines and persistent JSON with test-only fake WebSockets, not live physical phone WS packets.',
        'The four other tests are separate real private server/WebSocket/PCM processes with controlled fake local ASR, not actual resident Whisper accuracy.',
        'W71 source remains undeployed, Thor classifier 503, and human voice liveness/anti-recorded-replay not yet accepted']}


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('evidence','original-source','patched-source','original-history','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(argv)
    d=verify(a.evidence,a.original_source,a.patched_source,a.original_history)
    if a.output.exists() or a.output.is_symlink() or not a.output.is_absolute() or not str(a.output).startswith('/data/var/swaag-benchmarks/'):
        raise FileExistsError('Only an unused new private result path may be written')
    fd=os.open(a.output,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':d['status'],'actual_source_or_WS_cases':len(d['cases']),
       'original_unsafe_bounded_source_cases':d['observed_original_two_control_failures_without_actual_tool_effects'],
       'real_PCM_WS_server_cases':d['observed_new_real_private_PCM_websocket_server_cases'],
       'external_tool_model_calls':0},sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
