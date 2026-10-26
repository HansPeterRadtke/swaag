#!/usr/bin/env python3
"""Paired 20-task-cluster uncertainty over immutable source-graded agent outcomes.

This is a read-only supplemental analysis, not a replay of any prior inference,
benchmark task, downloaded source or alternative workflow.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
BASE=HERE/'evidence/oracle-free-agent-escalation'
OUTCOME_SHA='540b1d0cb97118c3766e11752b4f1e6b570db7160e69ee27fc6bc3bb5c882700'
CASE_SHA='b5e00a03f401260eff5659c4f2ca66bd8367ee9f7f5b157dd0fff51769bde189'
SELECTOR_SHA='14d344d8878576a2c948c12c8329874431fe6495a1e369b8fb69dfae1fae1b31'
VERSION='externally_graded_20_task_cluster_escalation_delta_v1_20261026'
ALT=('multi-agent-independent','multi-agent-centralized','multi-agent-decentralized','multi-agent-hybrid')
CMP=('never','no_output','hash_budget_20pct')
TARGET=('cross_benchmark_failure_risk_budget_20pct','no_output_then_hash_budget_20pct','no_output')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source():
    if digest(BASE/'policy_final_outcomes_v1.json')!=OUTCOME_SHA:raise ValueError('original result changed')
    if digest(BASE/'external_task_final_transition_ledger_v1.json')!=CASE_SHA:raise ValueError('original source-keyed task ledger changed')
    if digest(BASE/'observable_selector_receipts_v1.json')!=SELECTOR_SHA:raise ValueError('observable trigger receipt changed')
    report=json.loads((BASE/'policy_final_outcomes_v1.json').read_text())
    ledger=[json.loads(x) for x in (BASE/'external_task_final_transition_ledger_v1.json').read_text().splitlines()]
    assert len(ledger)==2*4*9*140
    indexed={}
    for r in ledger:
        k=(r['domain'],r['alternative_architecture'],r['policy'],r['provider'],r['model'],r['task_id'])
        if k in indexed:raise ValueError('duplicate independent original source match')
        indexed[k]=r
    return report,ledger,indexed


def paired_delta(indexed,domain,alt,candidate,baseline,resamples=1000):
    key=lambda policy, r:(domain,alt,policy,r['provider'],r['model'],r['task_id'])
    sampled=[r for k,r in indexed.items() if k[:3]==(domain,alt,candidate)]
    if len(sampled)!=140:raise ValueError('not exactly 140 model/task source pairings')
    per_task=defaultdict(list)
    for row in sampled:
        ref=indexed[key(baseline,row)]
        if ref['primary_independently_graded_resolved']!=row['primary_independently_graded_resolved']:
            raise ValueError('single-agent source baseline shifted')
        if ref['alternative_independently_graded_resolved']!=row['alternative_independently_graded_resolved']:
            raise ValueError('alternative independently graded original task changed')
        per_task[row['task_id']].append({
           'delta_correct':row['blind_final_independently_graded_resolved']-ref['blind_final_independently_graded_resolved'],
           'delta_additional_workflow':int(row['fallback_invoked_by_observable_policy'])-int(ref['fallback_invoked_by_observable_policy']),
           'delta_primary_error_trigger_false_positive':int(row['fallback_invoked_by_observable_policy'] and row['primary_independently_graded_resolved'])-int(ref['fallback_invoked_by_observable_policy'] and ref['primary_independently_graded_resolved']),
        })
    if len(per_task)!=20 or any(len(v)!=7 for v in per_task.values()):
        raise ValueError('20-task cluster preserving all seven model repeats failed')
    actual=[p for v in per_task.values() for p in v]
    task_ids=sorted(per_task)
    randomer=random.Random('20261026/'+domain+'/'+alt+'/'+candidate+'/'+baseline)
    boots={k:[] for k in actual[0]}
    for _ in range(resamples):
        values=[p for task in (randomer.choice(task_ids) for _ in task_ids) for p in per_task[task]]
        for key in boots:
            boots[key].append(sum(v[key] for v in values)/len(values))
    response={key:{'value_per_140_model_task_units':round(sum(v[key] for v in actual),5),
                   'fraction_of_model_task_units':round(sum(v[key] for v in actual)/140,6),
                   'descriptive_task_cluster_interval_95pct':[round(sorted(vals)[24],6),round(sorted(vals)[975],6)]}
              for key,vals in boots.items()}
    return {'domain':domain,'architecture':alt,'candidate_policy':candidate,'comparison_policy':baseline,
      'number_original_task_clusters':20,'models_repeated_per_task':7,'number_source_task_model_evaluations':140,
      'bootstrap_resamples':resamples,'paired_delays':response,
      'disclaimer':'Intervals cluster only the 20 original tasks, do not incorporate model-family independence, retrospective policy selection or full inference/GPU costs.'}


def main():
    summary,ledger,idx=source()
    delta=[]
    for domain in ('swebench-verified','terminalbench'):
        for alt in ALT:
            for cand in TARGET:
                for baseline in CMP:
                    if cand!=baseline:delta.append(paired_delta(idx,domain,alt,cand,baseline))
    result={'version':VERSION,'source_result_sha256':OUTCOME_SHA,'source_case_ledger_sha256':CASE_SHA,
            'source_observable_decisions_sha256':SELECTOR_SHA,
            'n_paired_policy_effects':len(delta),'original_standalone_task_gold_used_only_in_scoring':True,
            'exact_target_model_outcomes_not_available_to_deployable_selector':True,
            'claim_of_real_local_GPU_cost':False,'bootstrap_comparisons':delta}
    out=BASE/'paired_task_cluster_uncertainty_v1.json'
    if out.exists():raise FileExistsError('Do not overwrite first versioned result')
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    for domain in ('swebench-verified','terminalbench'):
        print('===',domain,'===')
        for alt in ('multi-agent-centralized','multi-agent-decentralized'):
            for baseline in ('never','no_output','hash_budget_20pct'):
                r=next(x for x in delta if x['domain']==domain and x['architecture']==alt
                    and x['candidate_policy']=='cross_benchmark_failure_risk_budget_20pct' and x['comparison_policy']==baseline)
                x=r['paired_delays']['delta_correct'];extra=r['paired_delays']['delta_additional_workflow']
                print(alt.split('-')[-1],'risk20 -',baseline,'net_correct',x['value_per_140_model_task_units'],
                    'CI',x['descriptive_task_cluster_interval_95pct'],'delta_workflows',extra['value_per_140_model_task_units'])
    print('SAVED',out)

if __name__=='__main__':main()
