#!/usr/bin/env python3
"""Independently adjudicate actual SWAAG voice-classifier isolated model traces.

Reads only saved model/gateway benchmark evidence. Invalid or interrupted
classifications are FAILURES, never correct negatives. Full local oracle
coverage and real model process lifecycle are mandatory for completion.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from swaag.benchmark.voice_classifier_acceptance import check_oracle_result

ORACLE_SHA='929f4fd01e21a12a2535db824034699fbe6359d9ecf872d978449f291cc9a26e'
MODEL_SHA='d02fe9b69ad8cadbbd228e387667af66612c44bed29ffc8eb1e7caf9ac486c12'
MODEL_ALIAS='swaag-qwen25vl-3b-isolated-20261124'
ORIGINAL_CASES=35
MODEL_CASE_TIMEOUT_SECONDS=8.0
# Conservative, *declared before final trial outcome*, not an owner promise.
ENGINEERING_P95_TARGET_SECONDS=2.0


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values:list[float],pct:float)->float:
    if not values or not 0<pct<=100:raise ValueError('Valid nonempty sample and percentile required')
    return sorted(values)[max(0,math.ceil(len(values)*pct/100)-1)]


def inspect(oracle:Path,benchmark:Path,process:Path)->dict:
    if any(not p.is_file() or p.is_symlink() for p in (oracle,benchmark,process)):
        raise ValueError('All pinned private model and oracle files must be regular files')
    if sha(oracle)!=ORACLE_SHA:raise ValueError('Original 35-case classifier oracle SHA mismatch')
    d=json.loads(oracle.read_text());cases=d['cases']
    if (len(cases)!=ORIGINAL_CASES or len({x['case_id'] for x in cases})!=len(cases)
        or Counter(x['family'] for x in cases)!=Counter({'turn_completion':17,'interruption':18})):
        raise ValueError('Original classifier categories or IDs changed')
    actual=json.loads(benchmark.read_text())
    run=json.loads(process.read_text())
    if run.get('model_alias')!=MODEL_ALIAS or run.get('benchmark_started') is not True:
        raise ValueError('Wrong or absent real isolated local model benchmark')
    if (run.get('model_stopped') is not True or run.get('benchmark_stopped') is not True
        or run.get('no_production_configuration_or_service_change') is not True):
        raise ValueError('Own isolated model and benchmark must be reaped, production untouched')
    if run.get('benchmark_file_sha256')!=sha(benchmark):
        raise ValueError('Changed benchmark evidence after model/process receipt')
    if run.get('actual_tested_cases')!=len(actual.get('cases',[])):
        raise ValueError('Mismatch between actual case receipts and model process receipt')
    recorded=actual.get('cases') or []
    if len({x.get('case_id') for x in recorded})!=len(recorded):
        raise ValueError('Duplicated classifier case receipts')
    if len(recorded)>len(cases):raise ValueError('More purported output cases than pinned oracle')
    evaluated=[]
    for i,case in enumerate(recorded):
        source=cases[i]
        if case.get('case_id')!=source['case_id'] or case.get('expected') is not source['expected']:
            raise ValueError('Out of order, changed expected result or invented case')
        elapsed=case.get('elapsed_seconds')
        if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<=elapsed<120:
            raise ValueError('Bad duration in real model response')
        # Reproduce the trusted-result gate independently from actual source.
        response={'complete' if source['family']=='turn_completion' else 'interrupt':case.get('observed'),
          'kind':case.get('kind'),'confidence':case.get('confidence')}
        checked=check_oracle_result(source['family'],source['expected'],response)
        if 'exception' in case:checked['passed']=False;checked['classifier_valid']=False
        if checked['passed'] is not case.get('passed') or checked['classifier_valid'] is not case.get('classifier_valid'):
            raise ValueError('Model case result inconsistent with pinned strict source comparator')
        evaluated.append({'case_id':source['case_id'],'family':source['family'],'expected':source['expected'],
            'observed':case.get('observed'),'passed':checked['passed'],'classifier_valid':checked['classifier_valid'],
            'kind':case.get('kind'),'elapsed_seconds':float(elapsed),'exception':case.get('exception')})
    overall={}
    for family in ('turn_completion','interruption'):
        subset=[x for x in evaluated if x['family']==family]
        times=[x['elapsed_seconds'] for x in subset]
        overall[family]={'cases':len(subset),'pass_count':sum(x['passed'] for x in subset),
            'invalid_responses':sum(not x['classifier_valid'] for x in subset),
            'true_expected_cases':sum(x['expected'] for x in subset),
            'false_expected_cases':sum(not x['expected'] for x in subset),
            'false_negative_count':sum(x['expected'] and not x['passed'] for x in subset),
            'false_positive_count':sum(not x['expected'] and not x['passed'] and x['classifier_valid'] for x in subset),
            'p50_seconds':round(percentile(times,50),3) if times else None,
            'p95_seconds':round(percentile(times,95),3) if times else None}
    times=[x['elapsed_seconds'] for x in evaluated]
    fully_measured=(len(evaluated)==ORIGINAL_CASES and actual.get('status')=='finished'
       and run.get('status')=='completed_full_35_case_benchmark')
    passed=sum(x['passed'] for x in evaluated)
    invalid=sum(not x['classifier_valid'] for x in evaluated)
    quality_acceptable=(fully_measured and passed==ORIGINAL_CASES and invalid==0
        and percentile(times,95)<=ENGINEERING_P95_TARGET_SECONDS
        and run.get('benchmark_exit_code')==0)
    return {'created_utc':datetime.now(timezone.utc).isoformat(),
      'status':'complete_independent_35_case_review' if fully_measured else 'incomplete_independent_review',
      'model_qualified_for_live_voice':False,
      'isolated_candidate_engineering_acceptance':bool(quality_acceptable),
      'candidate_recommendation':'further_isolated_tests_only' if quality_acceptable else 'do_not_deploy_or_route_live_voice',
      'model_alias':MODEL_ALIAS,'model_sha256':MODEL_SHA,
      'oracle_sha256':ORACLE_SHA,'model_source_case_count':ORIGINAL_CASES,
      'actual_case_count':len(evaluated),'actual_passed_cases':passed,
      'actual_failed_cases':len(evaluated)-passed,'invalid_responses_count':invalid,
      'source_family_metrics':overall,
      'latency_p50_seconds':round(percentile(times,50),3) if times else None,
      'latency_p95_seconds':round(percentile(times,95),3) if times else None,
      'latency_max_seconds':round(max(times),3) if times else None,
      'timing_budget_gateway_timeout_seconds':MODEL_CASE_TIMEOUT_SECONDS,
      'engineering_target_p95_seconds':ENGINEERING_P95_TARGET_SECONDS,
      'warmup_response_seconds':run.get('warmup_model_response_seconds'),
      'model_ready_seconds':run.get('model_ready_elapsed_seconds'),
      'whole_35_test_wall_seconds':run.get('elapsed_seconds'),
      'original_source_model_receipt_sha256':sha(process),
      'original_unmodified_classifier_cases_sha256':sha(benchmark),
      'actual_per_case_measurements':evaluated,
      'external_provider_account_changes':0,'real_tool_actions':0,
      'production_service_or_config_changed':False,
      'why_not_liveness_or_phone_test':'This is only transcript completion/interruption classification against fixed text. No incoming owner mic, speaker enrollment, cancellation, replay defense or Blackview playback tested.',
      'disclaimer':'No case reported successful merely because expected=False. Invalid/timeout always fails; no population acceptance rate or natural conversation usability has been established.'}


def save(result:Path,data:dict)->None:
    fd=os.open(result,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(data,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--process',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    d=inspect(a.oracle,a.benchmark,a.process)
    save(a.out,d)
    print('SOURCE_INDEPENDENT_CLASSIFIER_QUALITY_REVIEW',json.dumps({
      'status':d['status'],'tested':d['actual_case_count'],'passed':d['actual_passed_cases'],
      'invalid':d['invalid_responses_count'],'p95_s':d['latency_p95_seconds'],
      'production_qualified':d['model_qualified_for_live_voice']}))
    return 0 if d['status']=='complete_independent_35_case_review' else 3

if __name__=='__main__':raise SystemExit(main())
