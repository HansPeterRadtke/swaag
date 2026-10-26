#!/usr/bin/env python3
"""Externally graded same-task software-agent escalation, without online oracle.

This computes offline, full-workflow replacement outcomes from an immutable
published task-instance source, not new SWAAG/Qwen calls or actual live retries.
All four alternatives retained; no favorable-task oracle except labeled bound.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import io
import json
import math
import pathlib
import random
import statistics
import time
import zipfile

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE=pathlib.Path(__file__).resolve().parent
ARCHIVE=pathlib.Path('/data/tmp/agent-model-architecture-source-20261025/agent-scaling-v2.1.3.zip')
FIGURE=HERE/'evidence/matched-compute-architecture/nature-machine-intelligence-2026-source-figs1-2-original.csv'
SOURCE_ZIP_SHA='3b6752af8271db7319b1b23dbbe93733bd13e585c87e6f1abc4a17e15b96dc35'
SOURCE_CSV_SHA='c7bde9eac1f28a29397c10ab10e29f0962595e9dd45f17aedc2d309289b12007'
FIGURE_SHA='402a9fea7430bf4dbf1645c8bafd7850811ffb5bb331568e8088d85c4dc9ffd9'
VERSION='swaag_swe_terminal_oracle_free_transfer_v1_20261026'
ALTS=('multi-agent-independent','multi-agent-centralized','multi-agent-decentralized','multi-agent-hybrid')
ARCHS=('single-agent',*ALTS)
DATASETS={'swebench-verified':'swe-bench-verified','terminalbench':'terminal-bench'}
POLICIES=('never','no_output','no_output_or_at_most_2_steps','no_output_or_at_least_8_steps',
 'hash_budget_20pct','no_output_then_hash_budget_20pct','cross_benchmark_failure_risk_budget_20pct',
 'all','verified_failure_oracle')
BUDGET=28


def sha(raw):return hashlib.sha256(raw).hexdigest()


def source():
    if sha(ARCHIVE.read_bytes())!=SOURCE_ZIP_SHA:raise ValueError('original archive source changed')
    if sha(FIGURE.read_bytes())!=FIGURE_SHA:raise ValueError('publisher source figure changed')
    with zipfile.ZipFile(ARCHIVE) as z:
        matches=[k for k in z.namelist() if k.endswith('/scripts/per_instance_results_swe_tb.csv')]
        if len(matches)!=1:raise ValueError('ambiguous original case-source member')
        raw=z.read(matches[0])
    if sha(raw)!=SOURCE_CSV_SHA:raise ValueError('original task results changed')
    observed=list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    if len(observed)!=1800 or list(observed[0])!=['dataset','agent_type','provider','model','instance_idx','resolved','num_steps','success','has_output']:
        raise ValueError('unexpected original source schema/count')
    grouped=collections.defaultdict(list)
    by_key={}
    for r in observed:
        if r['dataset'] not in DATASETS or r['agent_type'] not in ARCHS or r['resolved'] not in ('0','1'):
            raise ValueError('invalid original benchmark/architecture/graded outcome')
        if r['success']!='0' or r['has_output'] not in ('True','False') or not r['num_steps'].isdigit():
            raise ValueError('ambiguous original outcome or workload feature')
        if r['has_output']=='False' and r['resolved']=='1':raise ValueError('successful result without output')
        if len(r['instance_idx'])!=4 or not r['instance_idx'].isdigit():raise ValueError('source task ID changed')
        key=(r['dataset'],r['provider'],r['model'],r['instance_idx'],r['agent_type'])
        if key in by_key:raise ValueError('duplicate original case outcome')
        item={'domain':r['dataset'],'provider':r['provider'],'model':r['model'],
              'task_id':r['instance_idx'],'architecture':r['agent_type'],
              'resolved':int(r['resolved']),'num_steps':int(r['num_steps']),
              'has_output':r['has_output']=='True'}
        by_key[key]=item
        grouped[(r['dataset'],r['provider'],r['model'],r['agent_type'])].append(item)
    if len(by_key)!=1800 or len(grouped)!=90:raise ValueError('missing original architectures')
    with FIGURE.open('r',newline='',encoding='utf-8-sig') as f:fig=list(csv.DictReader(f))
    if len(fig)!=260:raise ValueError('publisher original fig row count changed')
    figmap={(r['dataset'],r['family'],r['model'],r['architecture']):float(r['performance']) for r in fig}
    compatible=collections.defaultdict(set)
    matched=excluded=0
    for (domain,provider,model,architecture),case_rows in grouped.items():
        if len(case_rows)!=20 or len({r['task_id'] for r in case_rows})!=20:
            raise ValueError('source case count changed per source configuration')
        f=figmap.get((DATASETS[domain],provider,model,architecture))
        if f is None:excluded+=1;continue
        if abs(sum(r['resolved'] for r in case_rows)/20-f)>1e-9:
            raise ValueError('external grader and publisher mean contradicted')
        compatible[domain].add((provider,model));matched+=1
    if (matched,excluded)!=(70,20) or any(len(compatible[d])!=7 for d in DATASETS):
        raise ValueError('publisher version reconciliation changed')
    out={}
    for domain in DATASETS:
        rows=[]
        for provider,model in sorted(compatible[domain]):
            for task_num in range(20):
                task_id=f'{task_num:04d}'
                keys={(domain,provider,model,task_id,arch):by_key.get((domain,provider,model,task_id,arch)) for arch in ARCHS}
                if any(x is None for x in keys.values()):raise ValueError('incomplete same-task source set')
                original=keys[(domain,provider,model,task_id,'single-agent')]
                alternatives={arch:keys[(domain,provider,model,task_id,arch)] for arch in ALTS}
                rows.append({'domain':domain,'provider':provider,'model':model,'task_id':task_id,
                             'single_agent':original,'alts':alternatives})
        if len(rows)!=140:raise ValueError('not 7 original models times 20 case instances')
        out[domain]=rows
    return out,{'original_rows':1800,'matched_publisher_configuration_means':matched,
                'excluded_nonidentical_model_version_configurations':excluded,
                'exact_distinct_task_ids_per_benchmark':20,
                'repeated_model_task_units_per_benchmark':140,
                'original_archive_sha256':SOURCE_ZIP_SHA,
                'original_internal_case_csv_sha256':SOURCE_CSV_SHA,
                'original_figure_csv_sha256':FIGURE_SHA}


def digest_order(row):
    signature=f'20261026|{row["domain"]}|{row["provider"]}|{row["model"]}|{row["task_id"]}'
    return hashlib.sha256(signature.encode()).hexdigest()


def input_features(row):
    s=row['single_agent']
    return [1.0 if not s['has_output'] else 0.0,math.log1p(s['num_steps'])]


def transfer_probs(training,testing):
    # Training uses source independent grader labels for OTHER benchmark only.
    # No fitted metadata from test benchmark except observable input features.
    train_x=np.asarray([input_features(x) for x in training],dtype=float)
    train_y=np.asarray([1-x['single_agent']['resolved'] for x in training],dtype=int)
    if len(set(train_y))!=2:raise ValueError('training outcomes collapsed to one class')
    model=LogisticRegression(C=1.0,class_weight='balanced',solver='liblinear',
                             max_iter=1000,random_state=20261026)
    started=time.perf_counter()
    model.fit(train_x,train_y)
    test_x=np.asarray([input_features(x) for x in testing],dtype=float)
    scores=model.predict_proba(test_x)[:,1]
    measured=time.perf_counter()-started
    return scores.tolist(),{'train_domain':training[0]['domain'],'test_domain':testing[0]['domain'],
       'training_graded_repeated_model_tasks':len(training),'training_distinct_task_ids':20,
       'training_failure_count':int(train_y.sum()),
       'coefficient_features':[float(x) for x in model.coef_[0]],'intercept':float(model.intercept_[0]),
       'training_and_scoring_cpu_wall_seconds':round(measured,6),'threshold_tuning':False,
       'inference_does_not_access_test_resolved_or_any_alt_outcome':True}


def selectors(rows,transferred_failure_risk,train_metadata):
    n=len(rows)
    if n!=140 or BUDGET!=28:raise ValueError('budget/source cardinality changed')
    every=set(range(n))
    no_output={i for i,r in enumerate(rows) if not r['single_agent']['has_output']}
    low_steps={i for i,r in enumerate(rows) if r['single_agent']['num_steps']<=2}
    long_steps={i for i,r in enumerate(rows) if r['single_agent']['num_steps']>=8}
    random_order=sorted(range(n),key=lambda i:digest_order(rows[i]))
    if len(no_output)>BUDGET:raise ValueError('no-output subset exceeds fixed 20% budget')
    random20=set(random_order[:BUDGET])
    fill=set(no_output)
    for i in random_order:
        if len(fill)==BUDGET:break
        fill.add(i)
    risk20=set(sorted(range(n),key=lambda i:(-transferred_failure_risk[i],digest_order(rows[i])))[:BUDGET])
    out={
      'never':set(),
      'no_output':no_output,
      'no_output_or_at_most_2_steps':no_output|low_steps,
      'no_output_or_at_least_8_steps':no_output|long_steps,
      'hash_budget_20pct':random20,
      'no_output_then_hash_budget_20pct':fill,
      'cross_benchmark_failure_risk_budget_20pct':risk20,
      'all':every,
      'verified_failure_oracle':{i for i,r in enumerate(rows) if r['single_agent']['resolved']==0},
    }
    if set(out)!=set(POLICIES) or any(not selected<=every for selected in out.values()):
        raise ValueError('policy definition mismatch')
    if any(len(out[x])!=BUDGET for x in ('hash_budget_20pct','no_output_then_hash_budget_20pct',
                                        'cross_benchmark_failure_risk_budget_20pct')):
        raise ValueError('fixed policy capacity not respected')
    return out


def score_one(rows,selected,domain,arch,policy):
    status=collections.Counter()
    detail=[]
    for i,r in enumerate(rows):
        primary=r['single_agent'];alt=r['alts'][arch]
        started=i in selected
        original=primary['resolved'];alternative=alt['resolved']
        final=alternative if started else original
        # Real external source grader measures independently completed
        # architectures; this is a counterfactual replacement, not a replay.
        status['source_case_units']+=1
        status['primary_correct']+=original
        status['primary_failures']+=1-original
        status['fallback_workflows']+=int(started)
        status['trigger_tp_primary_fail']+=int(started and not original)
        status['trigger_fp_primary_correct']+=int(started and original)
        status['trigger_fn_primary_fail']+=int(not started and not original)
        status['trigger_tn_primary_correct']+=int(not started and original)
        status['final_correct_after_blind_selection']+=final
        status['wrong_to_right_rescues']+=int(started and not original and alternative)
        status['right_to_wrong_regressions']+=int(started and original and not alternative)
        status['right_to_right_redundant']+=int(started and original and alternative)
        status['wrong_to_wrong_no_improvement']+=int(started and not original and not alternative)
        status['original_alt_reported_steps_for_triggered_workflows']+=alt['num_steps'] if started else 0
        detail.append({'domain':domain,'provider':r['provider'],'model':r['model'],'task_id':r['task_id'],
          'policy':policy,'alternative_architecture':arch,
          'primary_has_output':primary['has_output'],
          'primary_original_reported_steps':primary['num_steps'],
          'primary_independently_graded_resolved':original,
          'fallback_invoked_by_observable_policy':started,
          'alternative_original_reported_steps':alt['num_steps'],
          'alternative_has_output':alt['has_output'],
          'alternative_independently_graded_resolved':alternative,
          'blind_final_independently_graded_resolved':final,
          'wrong_to_right_rescue':started and not original and bool(alternative),
          'right_to_wrong_regression':started and bool(original) and not alternative,
          'workflow_is_original_standalone_not_actual_post_primary_retry':True})
    assert status['source_case_units']==140
    assert status['trigger_tp_primary_fail']+status['trigger_fp_primary_correct']==status['fallback_workflows']
    assert status['wrong_to_right_rescues']+status['right_to_wrong_regressions']+status['right_to_right_redundant']+status['wrong_to_wrong_no_improvement']==status['fallback_workflows']
    assert status['final_correct_after_blind_selection']==status['primary_correct']+status['wrong_to_right_rescues']-status['right_to_wrong_regressions']
    return {**status,'domain':domain,'architecture':arch,'policy':policy,
       'primary_success_fraction':round(status['primary_correct']/140,6),
       'final_success_fraction':round(status['final_correct_after_blind_selection']/140,6),
       'net_change_correct_tasks':status['wrong_to_right_rescues']-status['right_to_wrong_regressions'],
       'fallbacks_per_net_added_correct_task':round(status['fallback_workflows']/(status['wrong_to_right_rescues']-status['right_to_wrong_regressions']),4) if status['wrong_to_right_rescues']>status['right_to_wrong_regressions'] else None,
       'cost_units':'complete additional original architecture workflows; original reported steps descriptive only, NOT GPU/LLM tokens or seconds'},detail


def clustered_interval(rows,delta_key,seed=20261026,resamples=1000):
    bytask=collections.defaultdict(list)
    for r in rows:bytask[r['task_id']].append(r)
    ids=sorted(bytask)
    if len(ids)!=20 or any(len(bytask[x])!=7 for x in ids):raise ValueError('source task dependence structure changed')
    randomer=random.Random(seed)
    actual=sum(r[delta_key] for r in rows)/len(rows)
    samples=[]
    for _ in range(resamples):
        draw=[r for sid in (randomer.choice(ids) for _ in ids) for r in bytask[sid]]
        samples.append(sum(r[delta_key] for r in draw)/len(draw))
    samples.sort()
    return {'fraction_of_140_model_task_units':round(actual,6),
       'descriptive_cluster_interval_95pct':[round(samples[24],6),round(samples[975],6)],
       'claim_clusters_are_original_task_ids':20,'repeated_model_runs_per_task':7,
       'resamples':resamples}


def summarize(source_rows):
    result=[];counters={};fitted=[];selection_records=[]
    for domain in DATASETS:
        other=next(s for s in DATASETS if s!=domain)
        rows=source_rows[domain]
        scores,trained=transfer_probs(source_rows[other],rows)
        fitted.append(trained)
        selects=selectors(rows,scores,trained)
        for i,r in enumerate(rows):
            selection_records.append({'domain':domain,'provider':r['provider'],'model':r['model'],
               'task_id':r['task_id'],'observable_has_output':r['single_agent']['has_output'],
               'observable_num_steps':r['single_agent']['num_steps'],
               'cross_domain_risk_without_target_outcome':round(scores[i],8),
               'selected_by':{k:i in v for k,v in selects.items() if k!='verified_failure_oracle'},
               'blind_triggers_only_without_target_or_alt_gold':True})
        for arch in ALTS:
            for policy,chosen in selects.items():
                counts,details=score_one(rows,chosen,domain,arch,policy)
                if policy=='all':
                    if counts['final_correct_after_blind_selection']!=sum(r['alts'][arch]['resolved'] for r in rows):
                        raise ValueError('all-switch does not match external original completed alternative')
                if policy=='never' and (counts['final_correct_after_blind_selection']!=counts['primary_correct'] or counts['fallback_workflows']!=0):
                    raise ValueError('never-switch direct outcome changed')
                if policy=='no_output' and counts['right_to_wrong_regressions']:
                    raise ValueError('no-output blind switch harmed originally successful primary')
                cluster=clustered_interval(details,'blind_final_independently_graded_resolved')
                counts['task_id_cluster_interval_for_final_success']=cluster
                result.append(counts)
                # Avoid repeated unnecessary source-case records identical between alt policies.
                counters[(domain,arch,policy)]=details
    return result,selection_records,fitted,counters


def main():
    started=time.perf_counter()
    input_data,source_evidence=source()
    rows,selections,fitted,case_records=summarize(input_data)
    results={'version':VERSION,'source':source_evidence,
       'method':'Original independently graded completed single-agent and four alternative architectures on same 20 tasks per benchmark; observable post-primary trigger then blind replacement; no online grader',
       'training':'Cross-benchmark source-gold training only; never same target domain labels at trigger-time',
       'policy_definitions':'ORACLE_FREE_AGENT_ESCALATION_PROTOCOL_2026-10-26.md',
       'policy_definitions_sha256':sha((HERE/'ORACLE_FREE_AGENT_ESCALATION_PROTOCOL_2026-10-26.md').read_bytes()),
       'independent_final_task_grader_column':'resolved',
       'incorrect_always_zero_source_column':'success',
       'model_training':fitted,
       'benchmark_arms':rows,
       'research_wall_seconds':round(time.perf_counter()-started,4),
       'real_local_Qwen_calls':0,
       'actual_additional_workflows_reexecuted':0,
       'counterfactual_alternative_source_workflows_counted_not_reexecuted':True,
       'limitations':[
          'Externally graded real benchmark tasks, not actual local SWAAG/Qwen tasks or live agent outputs.',
          'Blind replacement of a different independently completed architecture is offline; in production an extra execution after a prior action could differ and require side-effect checks.',
          'Only 20 distinct original tasks per benchmark; seven model versions reuse each source task; uncertainty is task-ID clustered.',
          'Reasoning steps are NOT wall time, GPU, actual token cost, or comparable resource usage across architectures. No per-task tokens/latency/VRAM exist in this archive.',
          'Cross-benchmark policies are retrospective validation on a source already studied; not untouched prospective field confirmation.',
          'The verified_failure_oracle intentionally uses after-the-fact gold and is not available online; all other selectors use observables only.',
          'This cannot decide local relevance-rating scale or modern Qwen syntax auxiliary training efficacy.',
       ]}
    out=HERE/'evidence/oracle-free-agent-escalation'
    out.mkdir(parents=True,exist_ok=True)
    report_path=out/'policy_final_outcomes_v1.json'
    cases_path=out/'external_task_final_transition_ledger_v1.json'
    selector_path=out/'observable_selector_receipts_v1.json'
    for p in (report_path,cases_path,selector_path):
        if p.exists():raise FileExistsError('An already completed immutable experiment must not be overwritten '+str(p))
    report_path.write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    with cases_path.open('w') as f:
        for k in sorted(case_records):
            for row in case_records[k]:f.write(json.dumps(row,sort_keys=True)+'\n')
    selector_path.write_text(json.dumps(selections,indent=2,sort_keys=True)+'\n')
    print('MAIN RESULTS')
    for domain in DATASETS:
        print('DOMAIN',domain)
        for arch in ALTS:
            print('ARCH',arch)
            for row in rows:
                if row['domain']==domain and row['architecture']==arch:
                    print(row['policy'],'fallbacks',row['fallback_workflows'],
                       'trigger_tp',row['trigger_tp_primary_fail'],'trigger_fp',row['trigger_fp_primary_correct'],
                       'rescue',row['wrong_to_right_rescues'],'harm',row['right_to_wrong_regressions'],
                       'net',row['net_change_correct_tasks'],'final_correct',row['final_correct_after_blind_selection'])
    print('OUTPUT',out)

if __name__=='__main__':main()
