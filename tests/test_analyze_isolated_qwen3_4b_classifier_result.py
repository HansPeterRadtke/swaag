"""Independent local text-only classifier model evaluation; no inference or service."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'scripts/demos/analyze_isolated_qwen3_4b_classifier_result.py'
SPEC=importlib.util.spec_from_file_location('new_source_qwen3_4b_independent_outcome',S)
A=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(A)
ORIGINAL=ROOT/'src/swaag/benchmark/fixtures/voice_classifier_oracle_20261104.json'


def fixture(tmp_path):
    oracle=tmp_path/'real-exact-source-oracle.json';oracle.write_bytes(ORIGINAL.read_bytes())
    source=json.loads(oracle.read_text())['cases'];assert len(source)==35
    outcomes=[]
    for x in source:
        family=x['family'];expected=x['expected'];kind=(
            'finished' if expected else 'continuing') if family=='turn_completion' else (
            'take_floor' if expected else 'keep_listening')
        outcomes.append({'case_id':x['case_id'],'expected':expected,'passed':True,
            'classifier_valid':True,'observed':expected,'kind':kind,
            'confidence':0.97,'elapsed_seconds':0.45})
    bench=tmp_path/'real-recorded-classifier-output.json'
    process=tmp_path/'oneuse-model-and-process-record.json'
    def write():
        bench.write_text(json.dumps({'status':'finished','cases':outcomes}))
        process.write_text(json.dumps({'status':'completed_full_35_case_benchmark',
             'model_alias':A.MODEL_ALIAS,'benchmark_started':True,
             'model_stopped':True,'benchmark_stopped':True,
             'no_production_configuration_or_service_change':True,
             'benchmark_exit_code':0,'benchmark_file_sha256':A.sha(bench),
             'actual_tested_cases':len(outcomes),
             'model_ready_elapsed_seconds':7,'warmup_model_response_seconds':0.8,
             'elapsed_seconds':18}))
    write()
    return oracle,bench,process,outcomes,write


def test_exact_oracle_source_hash_and_hard_limits():
    assert A.ORACLE_SHA=='929f4fd01e21a12a2535db824034699fbe6359d9ecf872d978449f291cc9a26e'
    assert A.sha(ORIGINAL)==A.ORACLE_SHA
    assert A.ORIGINAL_CASES==35
    assert A.MODEL_CASE_TIMEOUT_SECONDS==8.0
    assert A.ENGINEERING_P95_TARGET_SECONDS==2.0
    assert A.MODEL_SHA=='2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e'


def test_distinct_source_model_35_complete_case_types_without_production_claim(tmp_path):
    oracle,bench,process,cases,_=fixture(tmp_path)
    r=A.inspect(oracle,bench,process)
    assert r['status']=='complete_independent_35_case_review'
    assert r['actual_case_count']==r['actual_passed_cases']==35
    assert r['invalid_responses_count']==0
    assert r['source_family_metrics']['turn_completion']['cases']==17
    assert r['source_family_metrics']['interruption']['cases']==18
    assert r['latency_p95_seconds']==0.45
    assert r['isolated_candidate_engineering_acceptance'] is True
    assert r['model_qualified_for_live_voice'] is False
    assert r['candidate_recommendation']=='further_isolated_tests_only'


def test_expected_false_with_timeout_cannot_be_counted_as_correct_negative(tmp_path):
    oracle,bench,process,cases,write=fixture(tmp_path)
    i=next(i for i,x in enumerate(cases) if x['expected'] is False)
    cases[i].update({'passed':False,'classifier_valid':False,
        'observed':False,'kind':'classifier_invalid','confidence':0.0,'elapsed_seconds':8.012})
    write()
    r=A.inspect(oracle,bench,process)
    assert r['actual_failed_cases']==1
    assert r['actual_passed_cases']==34
    assert r['invalid_responses_count']==1
    assert r['isolated_candidate_engineering_acceptance'] is False
    assert r['candidate_recommendation']=='do_not_deploy_or_route_live_voice'
    assert r['latency_max_seconds']==8.012


def test_expected_false_incorrectly_reported_success_fails_integrity_check(tmp_path):
    oracle,bench,process,cases,write=fixture(tmp_path)
    i=next(i for i,x in enumerate(cases) if x['expected'] is False)
    cases[i].update({'passed':True,'classifier_valid':False,
        'observed':False,'kind':'classifier_invalid','confidence':0.0})
    write()
    with pytest.raises(ValueError,match='inconsistent'):
        A.inspect(oracle,bench,process)


def test_original_oracle_source_tamper_denied(tmp_path):
    oracle,bench,process,_,_=fixture(tmp_path)
    oracle.write_bytes(oracle.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='SHA mismatch'):
        A.inspect(oracle,bench,process)


def test_original_model_output_tamper_denied(tmp_path):
    oracle,bench,process,_,_=fixture(tmp_path)
    bench.write_bytes(bench.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Changed benchmark evidence'):
        A.inspect(oracle,bench,process)


def test_two_duplicated_case_receipts_or_changed_expected_denied(tmp_path):
    oracle,bench,process,cases,write=fixture(tmp_path)
    cases[1]['case_id']=cases[0]['case_id']
    write()
    with pytest.raises(ValueError,match='Duplicated'):
        A.inspect(oracle,bench,process)
    cases[1]['case_id']=json.loads(oracle.read_text())['cases'][1]['case_id']
    cases[1]['expected']=not cases[1]['expected']
    write()
    with pytest.raises(ValueError,match='changed expected result'):
        A.inspect(oracle,bench,process)


def test_wrong_alias_or_model_not_stopped_is_refused(tmp_path):
    oracle,bench,process,cases,_=fixture(tmp_path)
    d=json.loads(process.read_text());d['model_stopped']=False
    process.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='reaped'):
        A.inspect(oracle,bench,process)
    d['model_stopped']=True;d['model_alias']='wrong_network_model'
    process.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Wrong or absent'):
        A.inspect(oracle,bench,process)


def test_incomplete_cases_never_full_35_case_quality_success(tmp_path):
    oracle,bench,process,cases,write=fixture(tmp_path)
    cases.pop();write()
    d=json.loads(bench.read_text());d['status']='started';bench.write_text(json.dumps(d))
    p=json.loads(process.read_text());p['benchmark_file_sha256']=A.sha(bench);p['status']='failed';process.write_text(json.dumps(p))
    v=A.inspect(oracle,bench,process)
    assert v['status']=='incomplete_independent_review'
    assert v['actual_case_count']==34
    assert not v['isolated_candidate_engineering_acceptance']
    assert v['model_qualified_for_live_voice'] is False


def test_ordinary_three_second_response_exceeds_predeclared_engineering_target(tmp_path):
    oracle,bench,process,cases,write=fixture(tmp_path)
    for x in cases:x['elapsed_seconds']=3.1
    write()
    d=A.inspect(oracle,bench,process)
    assert d['actual_passed_cases']==35
    assert d['latency_p95_seconds']==3.1
    assert d['isolated_candidate_engineering_acceptance'] is False
    assert d['model_qualified_for_live_voice'] is False


def test_output_refuses_overwrite_of_existing_verdict(tmp_path):
    p=tmp_path/'verdict.json'
    A.save(p,{'status':'already_saved','cannot_rewrite':True})
    with pytest.raises(FileExistsError):A.save(p,{'status':'fabricated_success'})
    assert json.loads(p.read_text())['status']=='already_saved'


def test_p95_nearest_rank_is_bounded_and_unbiased_by_failure():
    assert A.percentile([8.0,0.2,0.5,1,3.0],95)==8.0
    assert A.percentile([8.0,0.2,0.5,1,3.0],50)==1.0
    with pytest.raises(ValueError):A.percentile([],95)
