#!/usr/bin/env python3
"""Independent disk-level adjudication of real private Jetson ASR races.

Reads original *failed* unmodified real running-server receipts and four
separate real passing fixed-server receipts. A valid historical failure is
not relabelled successful. The source code was explicitly pinned before each
trial. The voice and model endpoints were private inert, with a delayed
local ASR response whose transcript was controlled by the test—not measured
Whisper accuracy. Never reads owner bearer or sends/receives live audio.
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

from prepare_isolated_voice_asr_intent_epoch_guard import BASELINE, PATCHED_SERVER_SHA256,rewrite

SCENARIOS={
 'cancel-held-asr-once':{'revision':'baseline','status':'no_stale_draft_after_cancel_or_new_manual','turns':0,'replaced':0,'sha':'b05a90d5c48833023734c2e7890fb9e25593b5cdd3cbc5ee27e13f86d8d72fb8'},
 'new-manual-replacement-held-asr-once':{'revision':'baseline','status':'failed_or_unknown','turns':1,'replaced':1,'sha':'75c51ce8ac8bc7bb1f2340fa7c3197b81e5d5e039b26f3486ba9c7ee53556fc5'},
 'patched-new_manual_on_reconnect-oneuse':{'revision':'patched','status':'no_stale_draft_after_cancel_or_new_manual','turns':0,'replaced':1,'sha':'22fd43b2ffbbf116e9aa4662bd0c9c6a9a286052095680ce43c4154b2bda92b3'},
 'patched-new_manual_disconnect_before_asr-oneuse':{'revision':'patched','status':'no_stale_draft_after_cancel_or_new_manual','turns':0,'replaced':1,'sha':'993c9f16d9d17872ef81f7ccbced221dcfdab0449979eb283eecba2dfefdbf5b'},
 'patched-cancel-oneuse':{'revision':'patched','status':'no_stale_draft_after_cancel_or_new_manual','turns':0,'replaced':0,'sha':'6078acbbb89ad1d0e2cc760a5aca5268e4229171af7f3111857ed3414b8a0464'},
 'patched-idle_reconnect_pending_asr-oneuse':{'revision':'patched','status':'idle_pre_reconnect_ASR_held_for_manual_review','turns':1,'replaced':1,'sha':'a702d84acd1b483c0394825321606a25a11b307d0dbeae3f6a4c8aa95230daa1'},
}
VOICE_AUDIO_SHA='56e81fe7050641490a634e064705673a1f6c61ee2cf46d6b09f33d8270e7fae7'
FAKE_TEXT='This is a local controlled delayed speech result with no authorization.'

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()


def read_case(folder:Path,expected:dict)->dict:
    if folder.is_symlink() or not folder.is_dir():raise ValueError('Missing or symlinked scenario')
    receipt=folder/'result.json'
    if receipt.is_symlink() or not receipt.is_file() or sha(receipt.read_bytes())!=expected['sha']:
        raise ValueError('Immutable real-process receipt missing or changed: '+folder.name)
    result=json.loads(receipt.read_text())
    expected_source=BASELINE['cloudflare_ws_voice_server.py'] if expected['revision']=='baseline' else PATCHED_SERVER_SHA256
    if (result.get('status')!=expected['status'] or result.get('source_sha256')!=expected_source
        or result.get('input_recording_SHA256')!=VOICE_AUDIO_SHA
        or result.get('real_model_ASR_TTS_agent_tool_calls')!=0
        or result.get('actual_Jetson_production_service_changed') is not False):
        raise ValueError('Real-source status, no-action contract or pin drift: '+folder.name)
    if result.get('status')=='failed_or_unknown':
        if (result.get('failure_type')!='AssertionError' or
            result.get('failure_detail_private')!='Superseded late ASR inserted a stale draft during newer manual recording'):
            raise ValueError('Original failure did not match the reproduced actual bug')
    else:
        if result.get('held_authoritative_ASR_requests')!=1 or result.get('real_model_ASR_TTS_agent_tool_calls')!=0:
            raise ValueError('No real local fake-authoritative ASR socket request measured')
    log=folder/'voice-private.jsonl'
    if log.is_symlink() or not log.is_file():raise ValueError('Private process log missing')
    lines=[json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    counts=Counter(v.get('event') for v in lines)
    if (counts['server_start']!=1 or counts['session_replaced']!=expected['replaced'] or
        counts['asr_final']!=expected['turns']):
        raise ValueError('Actual source logged event counts differ: '+folder.name)
    if any(x in counts for x in ('answer_start','asr_auto_submitted','turn_submitted','worker_started')):
        raise ValueError('Unexpected external agent effect in isolated test')
    configs=folder/'private-config.json'
    if not configs.is_file() or configs.is_symlink():raise ValueError('No private ephemeral token/model config')
    cfg=json.loads(configs.read_text())
    if (cfg['server']['host']!='127.0.0.1' or cfg['interaction']['require_verified_speaker_for_auto_send'] is not True
        or cfg['interaction']['auto_send_default'] is not True
        or not str(cfg['server']['auth_token_file']).startswith(str(folder))):
        raise ValueError('Private model/voice config lacks speaker protection or isolation')
    for group in ('asr','llm','tts'):
        for key,value in cfg[group].items():
            if (key.endswith('_url') or key=='url') and value:
                if not re.match(r'^http://127\.0\.0\.1:(?:9|[1-9][0-9]{3,4})/',str(value)):
                    raise ValueError('Scenario config has a non-private provider route: '+group+'.'+key)
    hist=[]
    conv=folder/'conversations'
    if not conv.is_dir():raise ValueError('No private conversation history directory')
    for p in conv.glob('*.json'):
        if p.is_symlink() or p.stat().st_size>250000:raise ValueError('Unexpected source history file')
        data=json.loads(p.read_text());hist.extend(data.get('dialogue_history') or [])
    if len(hist)!=expected['turns']:
        raise ValueError('Persisted draft mismatch after actual ASR callback: '+folder.name)
    for row in hist:
        if (row.get('status')!='draft' or row.get('requires_review') is not True
            or row.get('recognized_text')!=FAKE_TEXT
            or row.get('submitted_text') or row.get('assistant') or row.get('assistant_audio_path')):
            raise ValueError('Unverified stale/idle session was allowed to submit or produce an answer')
    return {'case':folder.name,'original_or_patched':expected['revision'],
       'observed_status':result['status'],'expected_history_turns':expected['turns'],
       'actual_history_turns':len(hist),'model_and_tool_calls':0,
       'asr_completed_event_count':counts['asr_final'],
       'session_replaced_event_count':counts['session_replaced'],
       'one_real_local_delayed_asr_called':counts['asr_final']==1 or result.get('held_authoritative_ASR_requests')==1,
       'wall_seconds':result['elapsed_seconds'],
       'immutable_result_sha256':expected['sha'],'real_server_log_sha256':sha(log.read_bytes())}


def inspect(root:Path,baseline:Path,candidate:Path)->dict:
    if any(not p.is_dir() or p.is_symlink() for p in (root,baseline,candidate)):
        raise ValueError('Missing or symlinked private original/fixed evidence directory')
    for name,orig_sha in BASELINE.items():
        inp=baseline/name;out=candidate/name
        if any(not x.is_file() or x.is_symlink() for x in (inp,out)):
            raise ValueError('Expected source module missing')
        old=inp.read_bytes();new=out.read_bytes()
        if sha(old)!=orig_sha:raise ValueError('Original source hash drifted')
        if new!=(rewrite(old) if name=='cloudflare_ws_voice_server.py' else old):
            raise ValueError('Patched source is not the exact one-module deterministic revision')
    if sha((candidate/'cloudflare_ws_voice_server.py').read_bytes())!=PATCHED_SERVER_SHA256:
        raise ValueError('Patch pinned SHA not reproduced')
    checked=[]
    for name,expect in SCENARIOS.items():
        checked.append(read_case(root/name,expect))
    original_failed=next(c for c in checked if c['case']=='new-manual-replacement-held-asr-once')
    corrected=next(c for c in checked if c['case']=='patched-new_manual_on_reconnect-oneuse')
    if original_failed['actual_history_turns']!=1 or corrected['actual_history_turns']!=0:
        raise ValueError('No original observed failure and patched success contrast')
    if len([c for c in checked if c['original_or_patched']=='patched'])!=4:
        raise ValueError('All four new patched actual private server cases must be observed')
    return {'status':'genuine_private_server_stale_ASR_regression_fixed_in_unreleased_candidate',
       'verified_at_utc':datetime.now(timezone.utc).isoformat(),
       'baseline_source_sha256':BASELINE['cloudflare_ws_voice_server.py'],
       'patched_server_source_sha256':PATCHED_SERVER_SHA256,
       'all_unchanged_six_module_pins':{k:v for k,v in BASELINE.items() if k!='cloudflare_ws_voice_server.py'},
       'real_process_trials':len(checked),'baseline_real_failure_reproduced':True,
       'corrected_real_speaker_guard_tests_passed':4,
       'old_result_stale_draft_count':1,'patched_stale_draft_count':0,
       'patched_idle_reconnect_good_review_draft_count':1,
       'no_authorized_turn_or_tool_action_in_any_trial':True,
       'real_resident_Whisper_model_used':False,
       'physical_owner_Android_phone_used':False,
       'live_Jetson_server_modified_or_restarted':False,
       'complete_cases':checked,
       'remaining_known_limits':['No actual microphone or human speaker authentication','Only one controlled fake-local-ASR response per case, not model accuracy','ASR durable-record mutation can still race after a pre-write epoch test in an unusual filesystem interleaving','This is private source not installed as a production voice unit']}


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','baseline','candidate','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv)
    d=inspect(a.evidence,a.baseline,a.candidate)
    if a.out.exists() or a.out.is_symlink() or not a.out.is_absolute() or not str(a.out).startswith('/data/var/swaag-benchmarks/'):
        raise FileExistsError('Unique private result evidence required')
    fd=os.open(a.out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':d['status'],'real_process_cases':d['real_process_trials'],
         'original_stale_drafts':d['old_result_stale_draft_count'],
         'patched_stale_drafts':d['patched_stale_draft_count'],
         'patched_guard_passed':d['corrected_real_speaker_guard_tests_passed']}))
    return 0

if __name__=='__main__':raise SystemExit(main())
