#!/usr/bin/env python3
"""Run ONE real read-only SWAAG task; bound repetitive model errors safely.

The real pinned Coder model and actual SWAAG tools are used, never mocks.
Only CLI child launched here can be terminated, not the shared Coder/voice
services. Exclusive marker and durable evidence prevent a later automatic
retry after a transport interruption. The real model cannot access shell,
write files, providers, accounts, device peripherals or outside workspace.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

OPTIN='SWAAG_FRESH_RECONNECT_SOURCE_AGENT_NOACTION_ONCE'
EXPECTED_ALIAS='qwen3-coder-30b-a3b-instruct-q4-k-m'
EXPECTED_MODEL_URL='http://10.8.0.7:14833/v1/models'
EXPECTED_FILES=('tests/test_orchestrator_indexed_receipt_replay.py',
                'tests/test_tool_result_provenance.py')
EXPECTED_MISSING='tests/test_replay_connection_legacy.py'


def digest(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()


def exclusive_json(path:Path,obj:dict)->None:
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())


def scan_journal(folder:Path)->dict:
    events=[]
    found=list((folder/'sessions').glob('session_*/complete_history.jsonl'))
    if len(found)>1:raise ValueError('Mixed multiple sessions detected, refuse result')
    if found:
        p=found[0]
        if p.is_symlink() or p.stat().st_size>30_000_000:
            raise ValueError('Untrusted or unbounded private session journal')
        for line in p.read_text().splitlines():
            if not line.strip():continue
            try:events.append(json.loads(line))
            except json.JSONDecodeError:break
    types=Counter(x.get('type') for x in events)
    failures=[x for x in events if x.get('type')=='tool_error']
    repeats=sum(1 for x in failures if x.get('payload',{}).get('error_type')=='FilesystemError'
        and x.get('payload',{}).get('tool_input',{}).get('path')==EXPECTED_MISSING)
    return {'real_model_request_count':types['model_request_sent'],
         'real_model_response_count':types['model_response_received'],
         'real_tool_calls':types['tool_called'],
         'actual_successful_tool_results':types['tool_result'],
         'real_tool_errors':types['tool_error'],
         'repeated_identical_missing_file_errors':repeats,
         'terminal_turn_events':types['turn_completed']+types['turn_finished'],
         'recoverable_error_feedback_events':types['tool_invocation_failed_recoverably'],
         'journal_paths':[str(x) for x in found]}


def inspect_preflight(repo:Path,folder:Path)->dict:
    if not repo.is_absolute() or not repo.is_dir() or repo.is_symlink():
        raise ValueError('Source must be an existing absolute non-symlink checkout')
    if not folder.is_absolute() or folder.is_symlink() or not folder.is_dir():
        raise ValueError('Private test must already be prepared')
    manifest=json.loads((folder/'manifest.json').read_text())
    cfg=folder/'read-only-config.toml';task=folder/'fixed-new-reconnect-task.txt'
    if (cfg.is_symlink() or task.is_symlink() or digest(cfg)!=manifest['configuration_sha256']
        or digest(task)!=manifest['fixed_task_sha256']):
        raise ValueError('Frozen one-use task and tool config do not match original SHA')
    if (manifest['source_revision']!=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        or list(manifest['original_source_hash_and_oracle'])!=list(EXPECTED_FILES)
        or manifest['controlled_obsolete_missing_file']!=EXPECTED_MISSING
        or manifest['read_only_tools']!=['read_file','list_files','calculator']
        or manifest['no_side_effect_tools'] is not True):
        raise ValueError('Original SHA-pinned tool authority changed')
    if scan_journal(folder)['real_model_request_count'] or (folder/'.one_use_started.json').exists():
        raise ValueError('Prior one-use session activity exists; do not replay any uncertain provider request')
    if (folder/'result.json').exists() or (folder/'real-cli-console.log').exists():
        raise ValueError('Existing effect or console output, refusal to duplicate')
    # Import the real SWAAG configuration reader, not hand-rolled TOML checks.
    from swaag.config import load_config
    source=load_config([str(cfg)])
    if (source.model.profile_name!=EXPECTED_ALIAS
        or source.tools.enabled!=['read_file','list_files','calculator']
        or source.tools.allow_side_effect_tools or source.model.max_retries!=0):
        raise ValueError('Read-only authority violates the checked source configuration')
    return {'source_revision':manifest['source_revision'],'model':source.model.profile_name,
            'model_url':manifest['model_endpoint'],'tool_access':source.tools.enabled,
            'task_sha256':manifest['fixed_task_sha256'],'cfg_sha256':manifest['configuration_sha256']}


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--evidence',type=Path,required=True)
    ap.add_argument('--run',action='store_true')
    args=ap.parse_args(argv)
    evidence=inspect_preflight(args.repo,args.evidence)
    if not args.run:
        print('PREFLIGHT_ONLY_NO_MODEL_CALL',json.dumps(evidence));return 0
    if os.environ.get(OPTIN)!='1':raise ValueError('Explicit source-pinned one-use model run opt-in missing')
    marker=args.evidence/'.one_use_started.json'
    evidence.update({'created_utc':datetime.now(timezone.utc).isoformat(),
       'one_use':True,'no_retries_of_prior_model_provider_effects':True,
       'model_requests_prior_to_marker':0,'tool_actions_allowed':evidence['tool_access'],
       'external_tools_shell_file_writes_provider_effects_denied':True,
       'estimated_max_wall_seconds':500,'stop_on_same_missing_file_error_count':3})
    exclusive_json(marker,evidence)
    result={'started_utc':evidence['created_utc'],'status':'not_run','tool_executions_not_assumed_before_history_verification':True,
       'owner_user_phone_live_services_untouched':True,'scope':'one fresh real model/tool session only'}
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    task=args.evidence/'fixed-new-reconnect-task.txt'
    logfile=args.evidence/'real-cli-console.log'
    stdout_fd=os.open(logfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    start=time.monotonic();proc=None
    try:
        # Verify exact resident model alias via a read-only preflight. Never
        # send user/private source context to an unexpected model endpoint.
        with urllib.request.urlopen(EXPECTED_MODEL_URL,timeout=4) as r:
            if r.status!=200:raise ValueError('Model catalogue unavailable')
            models=json.loads(r.read(32000))
        if [x.get('id') for x in models.get('data',[])]!=[EXPECTED_ALIAS]:
            raise ValueError('Resident model alias no longer matches pinned Qwen Coder')
        result['model_catalogue_verified']=True
        command=[sys.executable,'-m','swaag','--config',str(args.evidence/'read-only-config.toml'),
          'ask','--session','wave66-reconnect-provenance-firstpass']
        with task.open('rb') as inp,os.fdopen(stdout_fd,'wb') as out:
            stdout_fd=-1
            proc=subprocess.Popen(command,cwd=args.repo,env=env,stdin=inp,
                       stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            result['own_child_pid']=proc.pid
            result['own_child_argv']=command
            stop_reason=None
            while proc.poll() is None:
                stats=scan_journal(args.evidence)
                if stats['repeated_identical_missing_file_errors']>=3:
                    stop_reason='three_real_identical_missing_path_failures'
                elif stats['real_tool_calls']>=14:
                    stop_reason='bounded_fresh_developer_tool_count'
                elif stats['real_model_request_count']>=14:
                    stop_reason='bounded_fresh_developer_model_call_count'
                elif time.monotonic()-start>=500:
                    stop_reason='bounded_fresh_developer_elapsed_wall_time'
                if stop_reason:
                    proc.terminate()  # owned exact new source test Python only
                    try:proc.wait(timeout=7)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
                    break
                time.sleep(2.0)
            result['bounded_stop_reason']=stop_reason
            result['exit_code']=proc.returncode
    except Exception as exc:
        result['internal_setup_error']=type(exc).__name__
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=7)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        result['exit_code']=None if proc is None else proc.returncode
    finally:
        if stdout_fd!=-1:os.close(stdout_fd)
        result['elapsed_seconds']=round(time.monotonic()-start,3)
        result.update(scan_journal(args.evidence))
        result['console_sha256']=digest(logfile)
        result['finished_utc']=datetime.now(timezone.utc).isoformat()
        result['status']=('process_exited_zero_pending_independent_oracle' if result.get('exit_code')==0
          else 'real_agent_failed_or_incomplete')
        exclusive_json(args.evidence/'result.json',result)
    print('GENUINE_NEW_SWAAG_REAL_AGENT_RUN_RECEIPT',json.dumps({
       k:result.get(k) for k in ('status','exit_code','bounded_stop_reason','elapsed_seconds',
         'real_model_request_count','real_model_response_count','real_tool_calls',
         'actual_successful_tool_results','real_tool_errors','terminal_turn_events','recoverable_error_feedback_events')}))
    return 0 if result['exit_code']==0 else 3

if __name__=='__main__':raise SystemExit(main())
