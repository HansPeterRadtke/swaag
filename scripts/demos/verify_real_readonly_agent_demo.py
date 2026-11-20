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
    durations=[]
    for e in rows:
        kind=e['type'];p=e.get('payload') or {}
        if kind=='tool_called':
            row={'sequence':e['sequence'],'tool':p.get('tool_name'),
                 'input':p.get('tool_input')}
            calls.append(row)
            starts[p.get('call_id')]=e
        elif kind in ('tool_error','tool_result'):
            start=starts.get(p.get('call_id'))
            if start:
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
    final_text=[e['payload']['message'].get('content','')
                for e in rows if e['type']=='message_added'
                and (e.get('payload') or {}).get('message',{}).get('role')=='assistant']
    finished=bool(types['turn_completed'] or types['turn_finished'])
    # A correction to a failed model run is an independently successful
    # recovery, never evidence that the autonomous first attempt succeeded.
    initial_model_failure=types['model_call_failed']>0
    correction_turn=types['turn_started']>=2
    last_answer=final_text[-1] if final_text else ''
    checks['final_answer_is_source_grounded']=(
        all(name in last_answer for name in FILES.values())
        and ('Ten tests total' in last_answer or '10 tests total' in last_answer)
        and '7 + 3' in last_answer and '0 + 0' in last_answer
    ) if correction_turn else bool(last_answer)
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
    model_timing=[{'call_id':(e.get('payload') or {}).get('call_id'),
                   'seconds':(e.get('payload') or {}).get('elapsed_seconds'),
                   'kind':(e.get('payload') or {}).get('kind')}
                  for e in rows if e['type']=='model_response_received']
    return {'status':status,'turn_finished':finished,'expected_total':expected_total,
       'initial_model_failure_recorded':initial_model_failure,
       'separate_correction_turn_observed':correction_turn,
       'model_failure_types':[(e.get('payload') or {}).get('error_type')
                              for e in rows if e['type']=='model_call_failed'],
       'completed_model_call_timings':model_timing,
       'checks':checks,'source_oracle':oracle,'calls':calls,
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
