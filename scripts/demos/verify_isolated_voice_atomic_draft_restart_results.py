#!/usr/bin/env python3
"""Independent immutable source, WS, and durable restart/orphaned draft oracle.

No voice model/ASR/call/phone/production access is performed. It re-reads
the frozen result, initial real source JSON, original exact source modules,
two different private child process histories, logs and no-model config.
Do not rerun the original successful restart experiment; use the saved data.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

from prepare_isolated_voice_atomic_intent_draft_guard import BASE,PATCHED_SERVER_SHA256

RESULT_SHA='bf2d0a7b94c59004f3dfa14d1bfc92d193754b72db6d2ddfa8a562b4cf462134'
ORIGINAL_HISTORY_SHA='bdde160a4996a1f38e8a88366bf2bb63e6ad3a65b2ebd26687326a03158102fc'
HISTORY_SHA={
 'latest_new_draft':'4b93968156084ad66ca0571a56bd0ca088e8e419f0c5fabc39052b08bd9ba42d',
 'superseded_only':'d2a3469d17fa80db826012465ea19548eaa67cc475300ecf222eb5eb73a645ae',
}
LOG_SHA={
 'latest_new_draft':'33d9d0bf8647b15c4e3dc9b2e0eb032f364fd0a4db91ade4b323f03e852fa062',
 'superseded_only':'8714f1d77e6febb42deddde6255f3b9dfa0b7dfb963df83792132046440aa73e',
}
CONVERSATION='private-w69-storage-race-A'

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def regular(p:Path)->None:
    if p.is_symlink() or not p.is_file():raise ValueError('Missing or symlinked private saved artifact '+p.name)

def inspect(root:Path,original:Path,source:Path)->dict:
    if any(p.is_symlink() or not p.is_dir() for p in (root,source)):
        raise ValueError('Missing private source or case root')
    regular(original)
    if sha(original)!=ORIGINAL_HISTORY_SHA:raise ValueError('Original forced-race history changed')
    base=json.loads(original.read_text())
    rows=base.get('dialogue_history') or []
    if (len(rows)!=2 or [x.get('status') for x in rows]!=['superseded','draft']
        or [x.get('turn_id') for x in rows]!=['turn-old','turn-current']
        or [x.get('asr_intent_epoch') for x in rows]!=[1,2]):
        raise ValueError('Original source is not the exact prior atomic storage result')
    for n,h in BASE.items():
        p=source/n;regular(p)
        if sha(p)!=(PATCHED_SERVER_SHA256 if n=='cloudflare_ws_voice_server.py' else h):
            raise ValueError('Private revised Jetson server source changed '+n)
    receipt=root/'result.json';regular(receipt)
    if sha(receipt)!=RESULT_SHA:raise ValueError('Immutable two-process result changed')
    r=json.loads(receipt.read_text())
    cases=r.get('cases') or []
    if (r.get('status')!='both_real_private_source_restart_draft_recovery_cases_passed'
        or r.get('source_sha256')!=PATCHED_SERVER_SHA256
        or r.get('source_durable_history_sha256')!=ORIGINAL_HISTORY_SHA
        or r.get('one_actual_new_speaker_review_draft') is not True
        or r.get('superseded_old_speech_never_restored') is not True
        or r.get('no_ASR_LLMTTS_SWAAG_tools_or_external_effects') is not True
        or r.get('actual_owner_phone_or_live_service_modified') is not False
        or len(cases)!=2 or [x.get('mode') for x in cases]!=list(HISTORY_SHA)):
        raise ValueError('Source-pinned real restart evidence contradicts saved result')
    out=[]
    for c in cases:
        mode=c['mode'];d=root/mode
        if d.is_symlink() or not d.is_dir():raise ValueError('Original subprocess output missing')
        hist=d/'conversations'/(CONVERSATION+'.json');regular(hist)
        if sha(hist)!=HISTORY_SHA[mode]:raise ValueError('Original persisted restart history changed')
        payload=json.loads(hist.read_text())
        persisted=payload.get('dialogue_history') or []
        expected=['superseded','draft'] if mode=='latest_new_draft' else ['superseded']
        if ([x.get('status') for x in persisted]!=expected
            or [x.get('turn_id') for x in persisted]!=['turn-old','turn-current'][:len(expected)]
            or any(x.get('submitted_text') or x.get('assistant') or x.get('assistant_audio_path') for x in persisted)):
            raise ValueError('Superseded draft was revived or newer draft lost')
        if mode=='latest_new_draft':
            if persisted[-1].get('requires_review') is not True or c.get('restored_active_review_count')!=1:
                raise ValueError('Newer input was not restored solely as manual-review draft')
        elif c.get('restored_active_review_count')!=0:
            raise ValueError('Superseded-only source unexpectedly restored a pending draft')
        log=d/'voice-private.jsonl';regular(log)
        if sha(log)!=LOG_SHA[mode]:raise ValueError('Original actual child logs changed')
        ev=[json.loads(s) for s in log.read_text().splitlines() if s.strip()]
        counts=Counter(x.get('event') for x in ev)
        if counts['server_start']!=1 or counts['session_open']!=1 or counts['speaker_identity_unverified_manual_review']!=1:
            raise ValueError('A real authenticated no-action child did not run')
        if any(counts[k] for k in ('turn_submitted','answer_start','worker_started','asr_final')):
            raise ValueError('Spoken model result was submitted or recreated')
        cfg=d/'private-config.json';regular(cfg);config=json.loads(cfg.read_text())
        if (config['server']['host']!='127.0.0.1'
            or config['interaction']['require_verified_speaker_for_auto_send'] is not True
            or config['interaction']['auto_send_default'] is not True
            or not str(config['server']['auth_token_file']).startswith(str(d))
            or config['llm']['url']!='http://127.0.0.1:9/run'):
            raise ValueError('Actual restart source config not isolated/fail-closed')
        pid=c.get('private_owned_child_pid')
        if type(pid) is not int or pid<=1:raise ValueError('Missing actual child process identity')
        proc=Path('/proc')/str(pid)/'cmdline'
        if proc.is_file():
            args=proc.read_bytes()
            if b'cloudflare_ws_voice_server.py' in args and str(source).encode() in args:
                raise ValueError('Private restart child source process still running')
        out.append({'mode':mode,'source_server_started':True,
          'restored_active_manual_review_draft_count':c['restored_active_review_count'],
          'persisted_record_statuses':expected,'private_process_reaped':True,
          'original_history_SHA256':HISTORY_SHA[mode],
          'original_log_SHA256':LOG_SHA[mode],
          'private_owned_PID_from_evidence':pid})
    return {'status':'independent_real_server_restart_restores_latest_draft_but_not_superseded_old_speech',
       'checked_at_utc':datetime.now(timezone.utc).isoformat(),
       'model_or_provider_calls':0,'real_owner_phone_used':False,
       'production_Jetson_service_modified':False,
       'pinned_server_source_SHA256':PATCHED_SERVER_SHA256,
       'genuine_previous_durable_source_SHA256':ORIGINAL_HISTORY_SHA,
       'immutable_observed_process_result_SHA256':RESULT_SHA,
       'independent_verified_case_count':len(out),'cases':out}


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','source-history','candidate-source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv)
    result=inspect(a.evidence,a.source_history,a.candidate_source)
    if (a.output.exists() or a.output.is_symlink() or not a.output.is_absolute()
        or not str(a.output).startswith('/data/var/swaag-benchmarks/')):
        raise FileExistsError('Unique private result only; no replay or replacement')
    fd=os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':result['status'],'cases':result['independent_verified_case_count'],
       'model_or_provider_calls':0,'live_voice_modified':False}))
    return 0

if __name__=='__main__':raise SystemExit(main())
