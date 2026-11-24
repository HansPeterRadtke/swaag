#!/usr/bin/env python3
"""Independent immutable source, durable storage, and genuine WS race oracle.

This tool never replays any voice/audio/model/provider operation. It reads
exactly five prior one-use result receipts from a fresh private Jetson test
and the real disk/log evidence they produced, plus the previous W68 failure
provenance. It fails closed if any result, source module, authorisation gate,
server history, or no-action claim disagrees with the original observed data.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re

from prepare_isolated_voice_atomic_intent_draft_guard import BASE,PATCHED_SERVER_SHA256,modify

VOICE_SHA='56e81fe7050641490a634e064705673a1f6c61ee2cf46d6b09f33d8270e7fae7'
PREVIOUS_VERDICT_SHA='9dfbc536c6634848fdd380e5a434a782782d8238045c67b11d0b9c6c7a7ec988'
STORAGE_SHA='84e57fa00d0a8da10e0874a3644782517b5ecb411b6b72276955ec3471198182'
CASES={
 'new_manual_on_reconnect':('3167e9724096734728101805d4089a5eaab4e83c737ae7630bf272c433824f7b',0,'no_stale_draft_after_cancel_or_new_manual',1),
 'new_manual_disconnect_before_asr':('474b9cf03b82abffca1052f49f8849b51c0ee5790f5a79a529fa075ba652047b',0,'no_stale_draft_after_cancel_or_new_manual',1),
 'cancel':('03f4fb8ceef383eaebb855eb035a1bc78c40f25a07589492d6383305c47406cf',0,'no_stale_draft_after_cancel_or_new_manual',0),
 'idle_reconnect_pending_asr':('8479045a0d0562a44598032ff820d1b4778366351a13995f6372c4dd9cc34391',1,'idle_pre_reconnect_ASR_held_for_manual_review',1),
}
FAKE_TEXT='This is a local controlled delayed speech result with no authorization.'

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def regular(p:Path)->None:
    if p.is_symlink() or not p.is_file():raise ValueError('Missing/symlinked private evidence '+p.name)


def check_source(old:Path,new:Path)->dict:
    if old.is_symlink() or new.is_symlink() or not old.is_dir() or not new.is_dir():
        raise ValueError('Unsafe or missing source directory')
    for name,want in BASE.items():
        previous=old/name;current=new/name
        regular(previous);regular(current)
        if sha(previous)!=want:raise ValueError('Prior source baseline unexpectedly changed '+name)
        expected=modify(previous.read_bytes()) if name=='cloudflare_ws_voice_server.py' else previous.read_bytes()
        if current.read_bytes()!=expected:
            raise ValueError('Patched source is not exactly the tested single-module code '+name)
    if sha(new/'cloudflare_ws_voice_server.py')!=PATCHED_SERVER_SHA256:
        raise ValueError('Patched actual server checksum mismatch')
    manifest=new/'source-manifest.json';regular(manifest)
    d=json.loads(manifest.read_text())
    if d.get('new_atomic_server_sha256')!=PATCHED_SERVER_SHA256 or d.get('changed_only')!=['cloudflare_ws_voice_server.py']:
        raise ValueError('Source provenance manifest inconsistent')
    return {'baseline_sha256':BASE['cloudflare_ws_voice_server.py'],
        'atomic_server_sha256':PATCHED_SERVER_SHA256,
        'manifest_sha256':sha(manifest),'same_six_supporting_modules':True}


def check_ws_case(root:Path,name:str,expected:tuple)->dict:
    result_sha,count,status,replaced=expected
    folder=root/('atomic-ws-'+name+'-oneuse')
    if folder.is_symlink() or not folder.is_dir():raise ValueError('WS evidence folder missing '+name)
    receipt=folder/'result.json';regular(receipt)
    if sha(receipt)!=result_sha:raise ValueError('Real WS receipt changed '+name)
    recorded=json.loads(receipt.read_text())
    if (recorded.get('status')!=status
            or recorded.get('source_sha256')!=PATCHED_SERVER_SHA256
            or recorded.get('input_recording_SHA256')!=VOICE_SHA
            or recorded.get('held_authoritative_ASR_requests')!=1
            or recorded.get('last_history_turns')!=count
            or recorded.get('real_model_ASR_TTS_agent_tool_calls')!=0
            or recorded.get('actual_owner_or_Android_phone_used') is not False
            or recorded.get('actual_Jetson_production_service_changed') is not False):
        raise ValueError('Real WS source/no-action/status not as documented '+name)
    p=folder/'private-config.json';regular(p);cfg=json.loads(p.read_text())
    if cfg['interaction']['auto_send_default'] is not True or cfg['interaction']['require_verified_speaker_for_auto_send'] is not True:
        raise ValueError('Actual WS test disabled the active speaker review guard')
    if cfg['server']['host']!='127.0.0.1' or not cfg['server']['auth_token_file'].startswith(str(folder)):
        raise ValueError('WS test server not private and isolated')
    for part in ('asr','llm','tts'):
        for k,v in cfg[part].items():
            if (k.endswith('_url') or k=='url') and v:
                if not re.match(r'^http://127\.0\.0\.1:\d+/',str(v)):
                    raise ValueError('Unexpected real remote provider URL '+part+'.'+k)
    p=folder/'voice-private.jsonl';regular(p)
    events=[json.loads(row) for row in p.read_text().splitlines() if row.strip()]
    tally=Counter(ev.get('event') for ev in events)
    if (tally['session_replaced']!=replaced or tally['server_start']!=1
        or tally['asr_final']!=count):
        raise ValueError('Actual source logs disagree with persisted WS test result '+name)
    if any(tally[k]>0 for k in ('answer_start','turn_submitted','worker_started','asr_auto_submitted')):
        raise ValueError('A real external tool or submitted turn was logged '+name)
    folder_history=folder/'conversations';history=[]
    if not folder_history.is_dir() or folder_history.is_symlink():
        raise ValueError('Missing original test private conversation root')
    for f in folder_history.glob('*.json'):
        regular(f)
        history.extend(json.loads(f.read_text()).get('dialogue_history') or [])
    if len(history)!=count:raise ValueError('Original WS durable draft count changed')
    for row in history:
        if (row.get('recognized_text')!=FAKE_TEXT or row.get('status')!='draft'
            or row.get('requires_review') is not True or row.get('submitted_text')
            or row.get('assistant') or row.get('assistant_audio_path')):
            raise ValueError('Wrong original WS unsubmitted review draft')
    return {'case':name,'source_SHA256':recorded['source_sha256'],
       'real_WS_process_elapsed_seconds':recorded['elapsed_seconds'],
       'real_local_fake_ASR_callback_count':1,'durable_review_drafts':count,
       'source_logged_final_ASR_events':tally['asr_final'],
       'source_logged_socket_replacements':tally['session_replaced'],
       'no_owner_or_provider_effects':True,'private_result_SHA256':result_sha,
       'private_events_SHA256':sha(p)}


def check_storage(root:Path)->dict:
    folder=root/'atomic-storage-threaded-oneuse'
    if not folder.is_dir() or folder.is_symlink():raise ValueError('Missing actual storage race process')
    p=folder/'result.json';regular(p)
    if sha(p)!=STORAGE_SHA:raise ValueError('Atomic disk race actual result altered')
    d=json.loads(p.read_text())
    if (d.get('status')!='genuine_atomic_disk_commit_vs_new_manual_intent_race_passed'
        or d.get('source_sha256')!=PATCHED_SERVER_SHA256
        or d.get('old_source_result_preserved_but_superseded') is not True
        or d.get('old_stale_later_draft_and_update_denied') is not True
        or d.get('latest_new_audio_draft_recorded_exactly_once') is not True
        or d.get('other_conversation_progressed_during_held_write') is not True
        or d.get('real_atomic_write_json_tempfile_fsync_rename') is not True
        or d.get('live_services_changed') is not False):
        raise ValueError('Forced real disk write/epoch interleaving was not observed')
    names={
      'A':'private-w69-storage-race-A','B':'private-w69-storage-race-B'
    }
    states={}
    for label,cid in names.items():
        f=folder/'conversations'/(cid+'.json');regular(f)
        turns=json.loads(f.read_text()).get('dialogue_history') or []
        states[label]=turns
    A=states['A'];B=states['B']
    if (len(A)!=2 or len(B)!=1
        or [r.get('status') for r in A]!=['superseded','draft']
        or A[0].get('asr_intent_epoch')!=1
        or A[0].get('superseded_by_intent_epoch')!=2
        or A[0].get('turn_id')!='turn-old' or A[1].get('turn_id')!='turn-current'
        or A[1].get('asr_intent_epoch')!=2
        or B[0].get('turn_id')!='turn-independent'
        or not A[0].get('user_audio_path')):
        raise ValueError('Original atomic storage states do not match the forced race')
    if any(r.get('submitted_text') or r.get('assistant') for r in A+B):
        raise ValueError('Stored result unexpectedly claims an agent action')
    return {'status':d['status'],'source_sha256':PATCHED_SERVER_SHA256,
       'real_JSON_fsync_and_rename_interleaving':True,
       'old_speech_status_after_new_manual':A[0]['status'],
       'new_speech_status':A[1]['status'],
       'new_speech_epoch':A[1]['asr_intent_epoch'],
       'independent_other_conversation_draft_count':len(B),
       'nonblocking_other_conversation_seconds':d['other_conversation_elapsed_seconds'],
       'private_result_SHA256':STORAGE_SHA,
       'A_history_SHA256':sha(folder/'conversations'/(names['A']+'.json')),
       'B_history_SHA256':sha(folder/'conversations'/(names['B']+'.json'))}


def verify(root:Path,prior:Path,baseline:Path,candidate:Path)->dict:
    if root.is_symlink() or not root.is_dir():raise ValueError('Not a private result root')
    regular(prior)
    if sha(prior)!=PREVIOUS_VERDICT_SHA:raise ValueError('Prior W68 failure/recovery evidence drift')
    orig=json.loads(prior.read_text())
    if (orig.get('status')!='genuine_private_server_stale_ASR_regression_fixed_in_unreleased_candidate'
        or orig.get('real_process_trials')!=6
        or orig.get('old_result_stale_draft_count')!=1
        or orig.get('corrected_real_speaker_guard_tests_passed')!=4):
        raise ValueError('Prior real baseline failure and epoch guard not grounded')
    return {'status':'independently_source_and_disk_verified_atomic_voice_storage_and_websocket_races',
       'checked_at_utc':datetime.now(timezone.utc).isoformat(),
       'source':check_source(baseline,candidate),
       'prior_source_observed_bug_and_fix_sha256':PREVIOUS_VERDICT_SHA,
       'actual_forced_atomic_write':check_storage(root),
       'actual_four_distinct_private_WS_processes':[check_ws_case(root,name,expect)
          for name,expect in CASES.items()],
       'actual_new_local_Whisper_SWAAG_TTS_or_tool_calls':0,
       'physical_Android_owner_speaker_verification':False,
       'production_Jetson_voice_service_changed':False,
       'note':'This verifies the source-pinned forced disk interleaving and four real separate WS tests. It is not the entire Android/production voice workflow.'}


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('evidence','previous-verdict','baseline','candidate','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(argv)
    d=verify(a.evidence,a.previous_verdict,a.baseline,a.candidate)
    if a.output.exists() or a.output.is_symlink() or not a.output.is_absolute() or not str(a.output).startswith('/data/var/swaag-benchmarks/'):
        raise FileExistsError('Unique private verdict destination required')
    fd=os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':d['status'],
       'atomic_disk_race':d['actual_forced_atomic_write']['status'],
       'new_real_WS_process_cases':len(d['actual_four_distinct_private_WS_processes']),
       'real_model_or_tool_effects':d['actual_new_local_Whisper_SWAAG_TTS_or_tool_calls']}))
    return 0

if __name__=='__main__':raise SystemExit(main())
