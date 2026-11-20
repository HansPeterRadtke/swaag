#!/usr/bin/env python3
"""Independently grade actual SWAAG tool receipts against on-disk source.

A plausible model answer is never proof of tool execution or completion.
No model invocation or filesystem mutation outside --output.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path

FILES = {
  'tests/test_history_replay.py':
     'test_external_history_tampering_is_still_detected_with_file_stat_cache',
  'tests/test_storage_boundaries.py':
     'test_session_storage_rejects_symlink_escape',
}
BAD_PATH='tests/test_history_integrity.py'


def inspect(repo: Path, journal: Path) -> dict:
    rows=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    if not rows or any(a['sequence']>=b['sequence'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Durable journal is missing or out of order')
    types=Counter(x['type'] for x in rows)
    oracle={}
    for rel,expected_name in FILES.items():
        data=(repo/rel).read_text(encoding='utf8')
        tests=[n.name for n in ast.parse(data).body
               if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
               and n.name.startswith('test_')]
        oracle[rel]={'tests':len(tests),'exact_required_name':expected_name,
                     'required_name_present':expected_name in tests,
                     'sha256':hashlib.sha256(data.encode()).hexdigest()}
    expected_total=sum(item['tests'] for item in oracle.values())
    calls=[]
    successes=[]
    failures=[]
    starts={}
    result_call_ids=set()
    receipt_integrity_errors=[]
    durations=[]
    for e in rows:
        kind=e['type'];p=e.get('payload') or {}
        if kind=='tool_called':
            row={'sequence':e['sequence'],'tool':p.get('tool_name'),
                 'input':p.get('tool_input')}
            calls.append(row)
            cid=p.get('call_id')
            if not isinstance(cid,str) or not cid or cid in starts:
                receipt_integrity_errors.append('missing_or_duplicate_tool_call_id')
            else:
                starts[cid]=e
        elif kind in ('tool_error','tool_result'):
            cid=p.get('call_id')
            start=starts.get(cid)
            if start is None:
                receipt_integrity_errors.append('unpaired_tool_receipt')
                continue
            if cid in result_call_ids:
                receipt_integrity_errors.append('duplicate_tool_receipt')
                continue
            original=start.get('payload') or {}
            expected_args=original.get('tool_input')
            reported_args=p.get('validated_input') if kind=='tool_result' else p.get('tool_input')
            if (e['sequence']<=start['sequence']
                    or p.get('tool_name')!=original.get('tool_name')
                    or not isinstance(expected_args,dict)
                    or reported_args!=expected_args):
                receipt_integrity_errors.append('tool_receipt_call_mismatch')
                continue
            result_call_ids.add(cid)
            t0=datetime.fromisoformat(start['timestamp'])
            t1=datetime.fromisoformat(e['timestamp'])
            durations.append({'tool':p.get('tool_name'),
                              'seconds':round((t1-t0).total_seconds(),6),
                              'success':kind=='tool_result'})
            if kind=='tool_error':
                failures.append({'sequence':e['sequence'],'tool':p.get('tool_name'),
                                 'input':p.get('tool_input'),
                                 'type':p.get('error_type'),
                                 'message_present':bool(p.get('error'))})
            else:
                name=p.get('tool_name')
                item={'sequence':e['sequence'],'tool':name,
                      'input':p.get('validated_input')}
                out=p.get('output') or {}
                if name=='read_file':
                    rel=(p.get('validated_input') or {}).get('path')
                    if rel in FILES:
                        expected=(repo/rel).read_text(encoding='utf8')
                        item['exact_source_match']=(out.get('text')==expected)
                        item['source_sha256']=hashlib.sha256(expected.encode()).hexdigest()
                if name=='list_files':
                    item['replay_file_listed']='tests/test_history_replay.py' in out.get('entries',[])
                if name=='calculator':
                    item['expression']=out.get('expression')
                    item['result']=out.get('result')
                successes.append(item)
    if set(starts)!=result_call_ids:
        receipt_integrity_errors.append('tool_calls_without_valid_receipts')
    missing=next((x['sequence'] for x in failures
                  if x['tool']=='read_file' and (x['input'] or {}).get('path')==BAD_PATH
                  and x['type']=='FilesystemError'),None)
    listing=next((x['sequence'] for x in successes
                  if x['tool']=='list_files' and x['input']=={'path':'tests'}
                  and x.get('replay_file_listed') and
                  (missing is None or x['sequence']>missing)),None)
    correct_reads={x['input']['path'] for x in successes
                   if x['tool']=='read_file' and x.get('exact_source_match')}
    calc=[x for x in successes if x['tool']=='calculator']
    correct_calc=any(x['result']==expected_total and
                     x.get('expression','').replace(' ','') in ('7+3','3+7')
                     for x in calc)
    checks={
      'actual_missing_path_error':missing is not None,
      'recovered_by_real_directory_listing':listing is not None,
      'two_distinct_exact_file_reads':correct_reads==set(FILES),
      'actual_correct_calculator_operation':correct_calc,
      'oracle_test_functions_exist':all(x['required_name_present'] for x in oracle.values()),
    }
    final_messages=[(e['sequence'],(e.get('payload') or {}).get('message',{}).get('content',''))
                    for e in rows if e['type']=='message_added'
                    and (e.get('payload') or {}).get('message',{}).get('role')=='assistant']
    final_text=[value for _,value in final_messages]
    last_start=max((r['sequence'] for r in rows if r['type']=='turn_started'),default=None)
    finished=(last_start is not None and any(r['sequence']>last_start
              and r['type'] in ('turn_completed','turn_finished') for r in rows))
    checks['tool_receipts_correlate_with_calls']=not receipt_integrity_errors
    checks['last_turn_has_terminal_event']=finished
    # A correction to a failed model run is an independently successful
    # recovery, never evidence that the autonomous first attempt succeeded.
    initial_model_failure=types['model_call_failed']>0
    correction_turn=types['turn_started']>=2
    answer_follows_last_start=bool(final_messages and last_start is not None
                                   and final_messages[-1][0]>last_start)
    last_answer=final_text[-1] if answer_follows_last_start else ''
    checks['answer_follows_latest_turn']=answer_follows_last_start
    checks['model_failure_has_separate_corrective_turn']=(
        not initial_model_failure or correction_turn)
    checks['final_answer_is_source_grounded']=(
        isinstance(last_answer,str)
        and all(name in last_answer for name in FILES.values())
        and all(path in last_answer for path in FILES)
        and ('Ten tests total' in last_answer or '10 tests total' in last_answer)
        and bool(re.search(r'7\s*\+\s*3|seven (?:and|plus) three',last_answer,re.I))
        and (not initial_model_failure or '0 + 0' in last_answer)
    )
    checks['allowed_tools_only']=all(r['tool'] in ('list_files','read_file','calculator')
                                     for r in calls)
    checks['history_is_single_session']=len({r['session_id'] for r in rows})==1
    verified=finished and all(checks.values())
    if verified and initial_model_failure and correction_turn:
        status='verified_after_supervisor_correction'
    elif verified:
        status='verified_first_pass'
    else:
        status='failed_or_incomplete'
    model_timing=[]
    for e in rows:
        if e['type']!='model_response_received':
            continue
        p=e.get('payload') or {}
        elapsed=p.get('elapsed_seconds')
        first_token=p.get('first_token_seconds')
        tokens=p.get('completion_tokens')
        timing={'call_id':p.get('call_id'),'seconds':elapsed,
                'kind':p.get('kind'),'first_token_seconds':first_token,
                'completion_tokens':tokens}
        if (isinstance(elapsed,(int,float)) and isinstance(first_token,(int,float))
                and isinstance(tokens,int) and 0<=first_token<elapsed):
            timing['generation_tokens_per_second']=round(tokens/(elapsed-first_token),3)
        model_timing.append(timing)
    model_queue_waits=[(e.get('payload') or {}).get('queue_wait_seconds')
                       for e in rows if e['type']=='inference_request_started']
    return {'status':status,'turn_finished':finished,'expected_total':expected_total,
       'initial_model_failure_recorded':initial_model_failure,
       'separate_correction_turn_observed':correction_turn,
       'model_failure_types':[(e.get('payload') or {}).get('error_type')
                              for e in rows if e['type']=='model_call_failed'],
       'completed_model_call_timings':model_timing,
       'model_queue_wait_seconds':model_queue_waits,
       'checks':checks,'receipt_integrity_errors':receipt_integrity_errors,
       'source_oracle':oracle,'calls':calls,
       'successful_tool_results':successes,'tool_errors':failures,
       'tool_durations':durations,'model_request_count':types['model_request_sent'],
       'model_response_count':types['model_response_received'],
       'event_count':len(rows),'event_counts':dict(types),
       'final_assistant_text':final_text[-2:],
       'session_id':rows[0]['session_id'],
       'a_model_claim_alone_is_never_considered_verification':True}


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--repo',type=Path,required=True)
    cli.add_argument('--journal',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    if args.out.exists():cli.error('Destination exists; preserve prior evidence')
    result=inspect(args.repo,args.journal)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('INDEPENDENT_AGENT_ORACLE',
          json.dumps({'status':result['status'],'turn_finished':result['turn_finished'],
          'checks':result['checks'],'tool_calls':len(result['calls']),
          'model_requests':result['model_request_count'],
          'model_responses':result['model_response_count'],
          'calculator_results':[{'expression':x['expression'],'result':x['result']}
                                for x in result['successful_tool_results']
                                if x['tool']=='calculator']},sort_keys=True))
    return 0 if result['status'].startswith('verified_') else 3

if __name__=='__main__':
    raise SystemExit(main())
