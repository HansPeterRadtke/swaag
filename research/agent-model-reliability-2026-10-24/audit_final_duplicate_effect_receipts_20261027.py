#!/usr/bin/env python3
"""Independent read-only accounting of already recorded isolated replay journals.

Checks duplicate/missing case receipts; does not rerun any model, tool, source
comparison, inference request, calendar operation or production side effect.
Only the research artifacts from completed sandbox experiments are examined.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FILES={
    'foreground_guard':'evidence/final-foreground-receipts/isolated_final_receipts_attempt02.jsonl',
    'source_policy':'evidence/source-authority-policy/swaag_final_user_bridge_receipts_v1.jsonl',
    'independent_last_mile':'evidence/independent-last-mile-replication/immutable_trial_stream_v1.jsonl',
}
EXPECTED={
    'foreground_guard':(46,46,'case_begin','case_completed','run_complete'),
    'source_policy':(12,12,'isolated_case_begin','isolated_case_completed','all_complete'),
    'independent_last_mile':(24,24,'isolated_case_begin','isolated_case_completed','all_cases_attempted'),
}


def tracked_original_bytes(path):
    rel=path.relative_to(ROOT).as_posix()
    result=subprocess.run(['git','show','HEAD:'+rel],cwd=ROOT,check=True,capture_output=True)
    return result.stdout


def digest(blob):return hashlib.sha256(blob).hexdigest()


def key(name, event, finished):
    r=event.get('result',{}) if finished else event
    if name=='foreground_guard':return (r['id'],r['variant'])
    if name=='source_policy':
        return ((r['source_claim_id'],r['source_paper_id'],r['policy']) if finished
                else (r['id'],r['paper'],r['policy']))
    return (r['case_id'],r['arm'])


def audit():
    details={}
    for name,rel in FILES.items():
        path=HERE/rel
        original=tracked_original_bytes(path)
        actual=path.read_bytes()
        if actual!=original:
            raise AssertionError(f'Pre-existing tracked journal changed: {rel}')
        rows=[json.loads(line) for line in actual.decode('utf8').splitlines() if line.strip()]
        planned_n,completed_n,begin_name,complete_name,last_name=EXPECTED[name]
        counts=Counter(row.get('event') for row in rows)
        begins=[key(name,row,False) for row in rows if row.get('event')==begin_name]
        completed=[key(name,row,True) for row in rows if row.get('event')==complete_name]
        if len(begins)!=planned_n or len(completed)!=completed_n:
            raise AssertionError(f'Incomplete or repeated sandbox experiment: {name}')
        if len(set(begins))!=planned_n or len(set(completed))!=completed_n or set(begins)!=set(completed):
            raise AssertionError(f'Duplicate/missing case identities: {name}')
        if counts[last_name]!=1 or rows[-1].get('event')!=last_name:
            raise AssertionError(f'Missing completion receipt: {name}')
        if counts.get('case_failed',0) or counts.get('isolated_case_failed',0):
            raise AssertionError(f'A technical failure must remain visible: {name}')
        replay_ok=0
        tools=0
        for row in rows:
            if row.get('event')!=complete_name:continue
            value=row.get('result',{})
            if name=='foreground_guard':
                okay=value['replayed_no_duplicate_model_or_tool_request']
                tools+=value['real_calculator_tool_result_count']
            elif name=='source_policy':
                okay=value['reconnect_idempotent']
                tools+=value['tool_effects_count']
            else:
                okay=value['reopened_session_and_control_replay_without_new_effects']
                tools+=value['actual_read_file_tool_results']
            if not okay:raise AssertionError(f'Replay made duplicate effect: {name}, {key(name,row,True)}')
            replay_ok+=1
        if name=='foreground_guard' and tools!=2:raise AssertionError('real test calculator receipts count drift')
        if name=='source_policy' and tools!=0:raise AssertionError('source-polarity tests may not perform tools')
        if name=='independent_last_mile' and tools!=12:raise AssertionError('original-source read tool count drift')
        details[name]={
            'tracked_at_original_git_head':True,
            'original_journal_sha256':digest(actual),
            'distinct_isolated_case_attempts':len(begins),
            'distinct_finished_final_case_receipts':len(completed),
            'same_control_id_idempotent_replay_successes':replay_ok,
            'technical_failed_case_receipts':0,
            'sandbox_tool_result_receipts':tools,
            'exact_original_event_kinds':dict(counts),
        }
    return {
        'version':'20261027_pre_0600_isolated_receipt_dedup_v1',
        'prior_source_git_revision':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,
                                   check=True,capture_output=True,text=True).stdout.strip(),
        'inspected_studies':details,
        'sum_distinct_final_case_receipts':sum(v['distinct_finished_final_case_receipts'] for v in details.values()),
        'sum_verified_idempotent_replays':sum(v['same_control_id_idempotent_replay_successes'] for v in details.values()),
        'sum_sandbox_tool_results':sum(v['sandbox_tool_result_receipts'] for v in details.values()),
        'limits':[
            'This checks exact immutable test-journal final/control receipts only; the archived journals do not hold every original ephemeral SQLite payload.',
            'These were isolated scripted-model research sessions and not proof of absence of duplicate effects in live owner production services.',
            'Original Qwen stage-2 final responses and GPU queue measurements remain unavailable; no model operation was replayed.',
            'The audit did not read/write production SQLite, inspect user messages, or interact with outside applications.',
        ],
    }


if __name__=='__main__':
    result=audit()
    path=HERE/'evidence/final-qwen-readiness/pre_0600_isolated_duplicate_effect_audit_v1.json'
    if path.exists():raise FileExistsError('Do not overwrite prior final checkpoint')
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('VERIFIED',result['sum_distinct_final_case_receipts'],'distinct final receipts; ',
          result['sum_verified_idempotent_replays'],'deduplicated replays; ',
          result['sum_sandbox_tool_results'],'genuine sandbox tool results')
    print('SAVED',path)
