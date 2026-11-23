#!/usr/bin/env python3
"""Independently verify a DIFFERENT, real SWAAG reconnect/provenance agent task.

This is an INDEPENDENT post-run oracle, never a model call or simulation.
It requires the whole original hash-linked durable SWAAG journal, matching
source Git blobs pinned before the model ran, a genuinely failed file read,
real directory recovery, two exact file reads and a real calculator receipt.
A plausible model answer alone can never pass. Supervisor correction is
prohibited for this one-turn first-pass demonstration.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

from swaag.events import verify_event_integrity
from swaag.types import HistoryEvent

BASELINE_REVISION='c2a9e55008c8872abed290ac4e34e08c473083dc'
FILES={
 'tests/test_orchestrator_indexed_receipt_replay.py':(
    '675779990610a4b56d581e626ed64ea1a908987eb96d4696bcdbb93f43c1f9ff',
    'test_reconnect_replays_exact_answer_without_materializing_giant_journal',2),
 'tests/test_tool_result_provenance.py':(
    'e44c724b7cf102f1c22a88775e41c6f7e8b1962df2f4ee1c7cbbab8a55b79aed',
    'test_tool_message_is_separate_accounting_component_with_source_event',3),
}
BAD_PATH='tests/test_replay_connection_legacy.py'
ALLOWED={'read_file','list_files','calculator'}


def digest(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()


def git_source(repo:Path,name:str)->str:
    if name not in FILES:raise ValueError('Unrecognized source file for fixed real oracle')
    try:
        res=subprocess.run(['git','-C',str(repo),'cat-file','blob',BASELINE_REVISION+':'+name],
                           capture_output=True,check=True,timeout=9)
    except (OSError,subprocess.SubprocessError) as exc:
        raise ValueError('The exact historical source revision is unavailable') from exc
    if digest(res.stdout)!=FILES[name][0]:
        raise ValueError('Pinned historical Git blob differs from original test source')
    return res.stdout.decode('utf8')


def inspect(repo:Path,journal:Path,manifest:Path,task:Path,process_receipt:Path|None=None)->dict:
    if any(not p.is_file() or p.is_symlink() for p in (journal,manifest,task)):
        raise ValueError('Missing/symlinked private real event evidence')
    m=json.loads(manifest.read_text())
    # The original frozen source-blob revision and the actual executable
    # runtime revision are separate evidence. They matched in the observed
    # first-pass run; future source-only improvements may change HEAD while
    # exact original source bytes and the oracle remain unchanged.
    original_blob_rev=m.get('original_source_blob_revision',m.get('source_revision'))
    runtime_revision=m.get('source_revision')
    if (original_blob_rev!=BASELINE_REVISION
        or not isinstance(runtime_revision,str) or len(runtime_revision)!=40
        or any(c not in '0123456789abcdef' for c in runtime_revision)
        or m.get('read_only_tools')!=['read_file','list_files','calculator']
        or m.get('no_side_effect_tools') is not True):
        raise ValueError('Unexpected model/tools/source configuration in frozen manifest')
    try:
        subprocess.run(['git','-C',str(repo),'cat-file','-e',runtime_revision+'^{commit}'],
                       check=True,capture_output=True,timeout=8)
    except (OSError,subprocess.SubprocessError) as exc:
        raise ValueError('Frozen runtime revision is unavailable; no invented Git source') from exc
    if m.get('fixed_task_sha256')!=digest(task.read_bytes()):
        raise ValueError('The current task bytes differ from frozen original user prompt')
    for name,(want,required,count) in FILES.items():
        meta=(m.get('original_source_hash_and_oracle') or {}).get(name) or {}
        if meta!={'sha256':want,'top_level_test_count':count,'required_exact_name':required}:
            raise ValueError('Frozen original source/oracle manifest mismatched')
    actual_sources={name:git_source(repo,name) for name in FILES}
    oracle={}
    for name,source in actual_sources.items():
        funcs=[n.name for n in ast.parse(source).body
               if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')]
        required=FILES[name][1];expected=FILES[name][2]
        if len(funcs)!=expected or required not in funcs:
            raise ValueError('Exact originally required test source does not meet frozen oracle')
        oracle[name]={'sha256':FILES[name][0],'top_level_test_functions':len(funcs),
                      'required_name':required,'required_name_present':True}
    raw=journal.read_text().splitlines()
    if not raw or len(raw)>5000:raise ValueError('Missing or unbounded genuine SWAAG event journal')
    rows=[json.loads(line) for line in raw if line.strip()]
    if not rows or any(x['sequence']>=y['sequence'] for x,y in zip(rows,rows[1:])):
        raise ValueError('Events out of source sequence')
    if len({x.get('session_id') for x in rows})!=1:
        raise ValueError('Mixed unrelated model sessions in durable journal')
    # Independently verify the SWAAG SHA-256 event chain AND required
    # event-schema payload, rejecting substituted result text, orphan
    # receipts, fabricated model activity and edited timestamps.
    prev=None
    for line in rows:
        try:
            e=HistoryEvent(**line)
            verify_event_integrity(e,prev)
        except Exception as exc:
            raise ValueError('Bad durable source-event hash or schema at sequence '+str(line.get('sequence'))) from exc
        prev=e.hash
    types=Counter(x['type'] for x in rows)
    originals=[(x.get('payload') or {}).get('message') or {}
               for x in rows if x['type']=='message_added'
               and ((x.get('payload') or {}).get('message') or {}).get('role')=='user']
    one_user_prompt=len(originals)==1 and originals[0].get('content')==task.read_text()
    starts={};results={};errors={};integrity_errors=[];durations=[]
    successful=[]
    for row in rows:
        typ=row['type'];p=row.get('payload') or {};cid=p.get('call_id')
        if typ=='tool_called':
            if not isinstance(cid,str) or not cid or cid in starts:
                integrity_errors.append('missing_or_duplicate_call_id')
            else:starts[cid]=row
        if typ not in ('tool_result','tool_error'):
            continue
        src=starts.get(cid)
        if src is None or cid in results or cid in errors:
            integrity_errors.append('unpaired_or_duplicate_receipt');continue
        was=src['payload'];provided=p.get('validated_input') if typ=='tool_result' else p.get('tool_input')
        if (p.get('tool_name')!=was.get('tool_name') or provided!=was.get('tool_input') or row['sequence']<=src['sequence']):
            integrity_errors.append('mismatched_real_tool_result');continue
        t0=datetime.fromisoformat(src['timestamp']);t1=datetime.fromisoformat(row['timestamp'])
        durations.append({'tool':p.get('tool_name'),'seconds':round((t1-t0).total_seconds(),6),'success':typ=='tool_result'})
        if typ=='tool_error':errors[cid]=row
        else:
            results[cid]=row
            out=p.get('output') or {}
            item={'sequence':row['sequence'],'tool':p['tool_name'],'input':provided}
            if p['tool_name']=='read_file' and provided.get('path') in FILES:
                item['exact_pinned_source_match']=out.get('text')==actual_sources[provided['path']]
            elif p['tool_name']=='list_files':
                entries=out.get('entries') or []
                item['both_source_files_listed']=all(name in entries for name in FILES)
            elif p['tool_name']=='calculator':
                item['expression']=out.get('expression')
                item['result']=out.get('result')
            successful.append(item)
    if set(starts)!=set(results)|set(errors):integrity_errors.append('tool_call_without_any_correlated_receipt')
    calls=[{'sequence':x['sequence'],'name':x['payload'].get('tool_name'),'input':x['payload'].get('tool_input')}
           for x in rows if x['type']=='tool_called']
    fail_seq=next((e['sequence'] for e in errors.values()
             if e['payload'].get('tool_name')=='read_file'
             and (e['payload'].get('tool_input') or {}).get('path')==BAD_PATH
             and e['payload'].get('error_type')=='FilesystemError'),None)
    listed_seq=next((e['sequence'] for e in successful
               if e['tool']=='list_files' and e['input']=={'path':'tests'}
               and e.get('both_source_files_listed') is True
               and fail_seq is not None and e['sequence']>fail_seq),None)
    read_map={e['input']['path']:e for e in successful
            if e['tool']=='read_file' and e['input'].get('path') in FILES
            and e.get('exact_pinned_source_match')}
    reads_after_list=bool(listed_seq is not None and len(read_map)==2 and
        all(e['sequence']>listed_seq for e in read_map.values()))
    calculators=[e for e in successful if e['tool']=='calculator']
    correct_calc=any(e.get('result')==5 and str(e.get('expression') or '').replace(' ','') in ('2+3','3+2') and
                     all(e['sequence']>r['sequence'] for r in read_map.values())
                     for e in calculators)
    final_texts=[(r['sequence'],(r.get('payload') or {}).get('message',{}).get('content'))
                 for r in rows if r['type']=='message_added'
                 and (r.get('payload') or {}).get('message',{}).get('role')=='assistant']
    last=final_texts[-1][1] if final_texts else ''
    final_supported=bool(isinstance(last,str) and
         all(name in last for name in FILES) and
         all(item[1] in last for item in FILES.values()) and
         bool(re.search(r'\b5\b|five',last,re.I)) and
         bool(re.search(r'2\s*\+\s*3|3\s*\+\s*2|two\s+(?:and|plus)\s+three',last,re.I)))
    unique_turn=types['turn_started']==1 and types['turn_completed']+types['turn_finished']>=1
    tool_names_safe=bool(calls) and all(x['name'] in ALLOWED for x in calls)
    # Every reported real tool call must have been selected by an actual
    # model action following a completed model response, not injected as
    # standalone fraudulent journal events. An action may legitimately
    # select more than one tool, so align with the immediately preceding
    # selected action and its exact validated arguments.
    model_selected_tool_calls=True
    action_events=[e for e in rows if e['type']=='agent_action_selected']
    model_response_events=[e for e in rows if e['type']=='model_response_received']
    for call in calls:
        action=next((a for a in reversed(action_events)
                     if a['sequence']<call['sequence']),None)
        if action is None:
            model_selected_tool_calls=False
            break
        selected=(action.get('payload') or {}).get('action') or {}
        toolcalls=selected.get('tool_calls') or []
        if (not isinstance(toolcalls,list) or not any(
            isinstance(c,dict) and c.get('tool_name')==call['name']
            and c.get('arguments')==call['input'] for c in toolcalls
        ) or not any(resp['sequence']<action['sequence'] for resp in model_response_events)):
            model_selected_tool_calls=False
            break
    no_model_failures=types['model_call_failed']==0 and types['model_unavailable']==0
    model_request_rows=[e for e in rows if e['type']=='model_request_sent']
    model_response_rows=[e for e in rows if e['type']=='model_response_received']
    model_pending={};model_response_ids=set();model_event_errors=[]
    for request in model_request_rows:
        call_id=(request.get('payload') or {}).get('call_id')
        if not isinstance(call_id,str) or not call_id or call_id in model_pending:
            model_event_errors.append('missing_or_duplicate_model_request_id')
        else:
            model_pending[call_id]=request
    for response in model_response_rows:
        cid=(response.get('payload') or {}).get('call_id')
        src=model_pending.get(cid)
        if src is None or cid in model_response_ids or response['sequence']<=src['sequence']:
            model_event_errors.append('orphan_duplicate_or_ordered_model_response')
        else:
            model_response_ids.add(cid)
    model_requests_complete=(bool(model_pending) and
        types['model_retry_scheduled']==0 and
        not model_event_errors and set(model_pending)==model_response_ids)
    # Recovery is model-driven only if a new actual model request *after* the
    # tool error included that concrete evidence, was answered, and preceded
    # the selected directory listing. A preplanned list in the same action
    # must not masquerade as observed error handling.
    model_observed_error_before_listing=False
    if fail_seq is not None and listed_seq is not None:
        for req in model_request_rows:
            if not fail_seq<req['sequence']<listed_seq:
                continue
            prompt=(req.get('payload') or {}).get('request',{}).get('prompt')
            cid=(req.get('payload') or {}).get('call_id')
            if (isinstance(prompt,str) and 'FilesystemError' in prompt
                and BAD_PATH in prompt
                and cid in model_response_ids
                and any(row['sequence']<listed_seq and
                        (row.get('payload') or {}).get('call_id')==cid
                        for row in model_response_rows)):
                model_observed_error_before_listing=True
                break
    no_supervisor_correction=len(originals)==1 and types['turn_started']==1
    checks={'one_original_exact_source_prompt':one_user_prompt,
            'single_model_user_turn_with_completion':unique_turn,
            'genuine_controlled_missing_file_error':fail_seq is not None,
            'genuine_directory_listing_after_error':listed_seq is not None,
            'model_received_error_before_recovering_with_directory_listing':model_observed_error_before_listing,
            'two_real_source_reads_after_directory_recovery':reads_after_list,
            'real_correct_calculator_after_reads':correct_calc,
            'source_exact_test_names_and_counts_held':True,
            'source_linked_hashed_event_chain_valid':True,
            'tool_call_receipts_exact_and_unique':not integrity_errors,
            'only_readonly_allowlisted_tools_used':tool_names_safe,
            'model_action_selected_exact_real_tool_calls':model_selected_tool_calls,
            'no_model_errors_or_timeouts':no_model_failures,
            'all_model_requests_completed_without_retry':model_requests_complete,
            'no_supervisor_correction_turn':no_supervisor_correction,
            'final_answer_supported_by_actual_tools':final_supported}
    allpass=all(checks.values())
    model_timings=[{'request_call_id':(r.get('payload') or {}).get('call_id'),
       'seconds':(r.get('payload') or {}).get('elapsed_seconds'),
       'completion_tokens':(r.get('payload') or {}).get('completion_tokens'),
       'first_token_seconds':(r.get('payload') or {}).get('first_token_seconds')}
       for r in rows if r['type']=='model_response_received']
    if process_receipt is not None:
        if not process_receipt.is_file():raise ValueError('Real independent process-exit receipt unavailable')
        process=json.loads(process_receipt.read_text())
    else:process={}
    if process:
        code=process.get('one_shot_exit_code',process.get('exit_code'))
        checks['one_use_process_exited_successfully']=(type(code) is int and code==0)
        allpass=allpass and checks['one_use_process_exited_successfully']
    return {'status':'verified_first_pass' if allpass else 'failed_or_incomplete',
       'checks':checks,'source_revision':BASELINE_REVISION,
       'real_runtime_revision_recorded_in_manifest':runtime_revision,
       'pinned_source_oracle':oracle,'session_id':rows[0]['session_id'],
       'event_count':len(rows),'event_counts':dict(types),
       'actual_tool_calls':calls,'actual_successful_tool_results':successful,
       'real_tool_failures':[{'sequence':x['sequence'],'tool':x['payload'].get('tool_name'),
                             'path':(x['payload'].get('tool_input') or {}).get('path'),
                             'type':x['payload'].get('error_type')} for x in errors.values()],
       'tool_call_integrity_errors':integrity_errors,
       'model_request_response_integrity_errors':model_event_errors,
       'model_request_count':types['model_request_sent'],
       'model_completed_response_count':types['model_response_received'],
       'model_timing':model_timings,'tool_timing':durations,
       'single_turn':types['turn_started']==1,
       'final_answer_content':last if isinstance(last,str) else '',
       'saved_process_exit':process,
       'one_model_answer_without_tools_not_accepted':True,
       'no_supervisor_correction_or_external_actions_are_permitted':True}


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--journal',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--process-receipt',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        p.error('Previous model verification is not overwritten')
    verdict=inspect(args.repo,args.journal,args.manifest,args.task,args.process_receipt)
    args.out.write_text(json.dumps(verdict,ensure_ascii=False,indent=2)+'\n')
    args.out.chmod(0o600)
    print('Genuine read-only SWAAG independent oracle',json.dumps({
       'status':verdict['status'],'checks':verdict['checks'],
       'model_requests':verdict['model_request_count'],
       'tool_calls':len(verdict['actual_tool_calls']),
       'tool_failures':len(verdict['real_tool_failures']),
       'source_revision':BASELINE_REVISION}))
    return 0 if verdict['status']=='verified_first_pass' else 3

if __name__=='__main__':raise SystemExit(main())
