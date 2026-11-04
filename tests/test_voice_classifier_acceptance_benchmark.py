import json
from pathlib import Path
import pytest
from swaag.benchmark.voice_classifier_acceptance import (
    DEFAULT_ORACLE,check_oracle_result,validate_isolated_endpoint,
)


def test_voice_classifier_benchmark_golden_cases_are_grounded_and_nonempty():
    data=json.loads(DEFAULT_ORACLE.read_text(encoding='utf-8'))
    cases=data['cases']
    assert len(cases)==35
    assert len({case['case_id'] for case in cases})==len(cases)
    assert sum(case['family']=='turn_completion' for case in cases)==17
    assert sum(case['family']=='interruption' for case in cases)==18
    assert all(type(case['expected']) is bool and case['text'].strip() for case in cases)
    assert all(case['source_group'] in ('base','initial','acknowledgment') for case in cases)
    assert not any('token' in case or 'secret' in case for case in cases)


def test_invalid_classifier_output_never_passes_as_negative_oracle():
    negative=check_oracle_result('turn_completion',False,{
        'complete':False,'kind':'classifier_invalid','confidence':0.0})
    assert negative['passed'] is False and negative['classifier_valid'] is False
    negative=check_oracle_result('interruption',False,{
        'interrupt':False,'kind':'classifier_invalid','confidence':0.0})
    assert negative['passed'] is False and negative['classifier_valid'] is False
    valid=check_oracle_result('turn_completion',False,{
        'complete':False,'kind':'continuing','confidence':0.9})
    assert valid['passed'] is True
    assert check_oracle_result('interruption',True,{
        'interrupt':True,'kind':'take_floor','confidence':0.95})['passed'] is True
    assert not check_oracle_result('interruption',False,{
        'interrupt':False,'kind':'continuing','confidence':0.9})['passed']


def test_classifier_benchmark_refuses_live_service_and_remote_urls():
    assert validate_isolated_endpoint('http://127.0.0.1:14921')=='http://127.0.0.1:14921'
    for endpoint in ('http://127.0.0.1:14920','http://127.0.0.1:13484',
                     'http://127.0.0.1:13401','http://thor:14921',
                     'https://127.0.0.1:14921','http://127.0.0.1:14921/health'):
        with pytest.raises(ValueError):validate_isolated_endpoint(endpoint)


def test_classifier_benchmark_default_oracle_is_packaged_with_implementation():
    from swaag.benchmark import voice_classifier_acceptance as acceptance
    assert DEFAULT_ORACLE.parent == Path(acceptance.__file__).resolve().parent / 'fixtures'
    assert DEFAULT_ORACLE.is_file()
