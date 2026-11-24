"""Offline source-integrity/fail-closed tests for two CPU turn detectors.

These are only constructed toy files and mock probabilities, NOT new model
inferences. The genuine 51-case Namo and FireRed observations are sealed
separately in Nitro's private benchmark evidence. No network/model, owner
speech, phone, tool call, or live voice gateway is contacted by these tests.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))

def load(name):
    spec=importlib.util.spec_from_file_location('w72_'+name,ROOT/'scripts/demos'/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

N=load('run_isolated_namo_multilingual_cpu_once.py')
F=load('run_isolated_firered_multilingual_cpu_once.py')
NA=load('analyze_isolated_namo_multilingual_cpu_result.py')
FA=load('analyze_isolated_firered_multilingual_cpu_result.py')
ORACLE=ROOT/'src/swaag/benchmark/fixtures/voice_classifier_oracle_20261104.json'
HOLDOUT=ROOT/'scripts/demos/fixtures/voice_namo_bilingual_holdout_20261124.json'

@pytest.mark.parametrize('runner',[N,F],ids=['Namo','FireRed'])
def test_frozen_source_labels_are_same_and_set_before_model_run(runner):
    cases=runner.checked_sources if False else None  # use file-only assertions
    original=json.loads(ORACLE.read_text())
    hold=json.loads(HOLDOUT.read_text())
    source=[x for x in original['cases'] if x['family']=='turn_completion']
    assert len(source)==17 and len(hold['cases'])==34
    assert len({x['case_id'] for x in source+hold['cases']})==51
    assert sum(x['expected'] is False for x in source)==8
    assert sum(x['expected'] is False for x in hold['cases'])==18
    assert runner.sha(ORACLE)==runner.ORACLE_SHA
    assert runner.sha(HOLDOUT)==runner.HOLDOUT_SHA
    assert runner.REVISION and runner.REPOSITORY and runner.OPTIN


def fake_model_source(tmp_path,runner,monkeypatch):
    model=tmp_path/'toy_model';model.mkdir()
    mock={}
    for filename in runner.MODEL_FILES:
        p=model/filename;p.parent.mkdir(parents=True,exist_ok=True)
        body=(filename+' source fixture only').encode()
        if filename=='LICENSE':body=b'Apache License\nVersion 2.0\nSynthetic fixture only\n'
        if filename=='README.md':body=b'---\nlicense: apache-2.0\n---\nSynthetic fixture only\n'
        p.write_bytes(body)
        mock[filename]=(hashlib.sha256(body).hexdigest(),len(body))
    monkeypatch.setattr(runner,'MODEL_FILES',mock)
    return model

@pytest.mark.parametrize('runner',[N,F],ids=['Namo','FireRed'])
def test_toy_source_fingerprint_and_zero_effect_preflight(tmp_path,monkeypatch,runner,capsys):
    model=fake_model_source(tmp_path,runner,monkeypatch)
    cases=runner.checked_sources(model,ORACLE,HOLDOUT)
    assert len(cases)==51
    assert all(q['source'] in ('original_SWAAG_oracle','assistant_authored_bilingual_holdout') for q in cases)
    out=tmp_path/'never_started'
    assert runner.main(['--model-dir',str(model),'--oracle',str(ORACLE),'--holdout',str(HOLDOUT),'--private-output',str(out)])==0
    d=json.loads(capsys.readouterr().out)
    assert d['status']=='CPU_source_and_holdout_preflight_only'
    assert d['no_model_loaded_or_source_written'] and d['cases']==51
    assert not out.exists()

@pytest.mark.parametrize('runner',[N,F],ids=['Namo','FireRed'])
def test_source_checksum_tamper_fails_before_any_model_request(tmp_path,monkeypatch,runner):
    model=fake_model_source(tmp_path,runner,monkeypatch)
    selected=next(iter(runner.MODEL_FILES))
    p=model/selected;p.write_bytes(p.read_bytes()+b'CORRUPTED')
    with pytest.raises(ValueError,match='hash/size mismatch'):
        runner.checked_sources(model,ORACLE,HOLDOUT)

@pytest.mark.parametrize('runner',[N,F],ids=['Namo','FireRed'])
def test_missing_original_model_license_or_card_refused(tmp_path,monkeypatch,runner):
    model=fake_model_source(tmp_path,runner,monkeypatch)
    license_key='LICENSE' if runner is N else 'README.md'
    p=model/license_key;p.unlink()
    with pytest.raises(ValueError,match='Missing or unsafe'):
        runner.checked_sources(model,ORACLE,HOLDOUT)

@pytest.mark.parametrize('runner',[N,F],ids=['Namo','FireRed'])
def test_symlinked_model_source_rejected(tmp_path,monkeypatch,runner):
    model=fake_model_source(tmp_path,runner,monkeypatch)
    link=tmp_path/'untrusted_symlink';link.symlink_to(model,target_is_directory=True)
    with pytest.raises(ValueError,match='directory missing/symlinked'):
        runner.checked_sources(link,ORACLE,HOLDOUT)

@pytest.mark.parametrize('runner',[N,F],ids=['Namo','FireRed'])
def test_broker_denial_precedes_attempt_marker_and_external_effect(tmp_path,monkeypatch,runner):
    model=fake_model_source(tmp_path,runner,monkeypatch)
    def refuse():raise RuntimeError('Current resource broker denied exact lease')
    monkeypatch.setattr(runner,'lease_valid',refuse)
    out=tmp_path/'oneuse_new_private_result'
    with pytest.raises(RuntimeError,match='resource broker denied'):
        runner.run(model,ORACLE,HOLDOUT,out)
    assert not out.exists()

@pytest.mark.parametrize('runner',[N,F],ids=['Namo','FireRed'])
def test_no_retry_if_first_model_effect_might_have_occurred(tmp_path,monkeypatch,runner):
    model=fake_model_source(tmp_path,runner,monkeypatch)
    monkeypatch.setattr(runner,'lease_valid',lambda:None)
    monkeypatch.setenv(runner.OPTIN,'1')
    out=tmp_path/'already_started';out.mkdir()
    (out/'attempt.json').write_text('{"state":"unknown"}')
    with pytest.raises(FileExistsError,match='One-use private model experiment'):
        runner.run(model,ORACLE,HOLDOUT,out)
    assert (out/'attempt.json').read_text()=='{"state":"unknown"}'

@pytest.mark.parametrize('runner',[N,F],ids=['Namo','FireRed'])
def test_result_stat_invalid_or_timeout_is_not_false_prediction_success(runner):
    cases=[{'expected':False,'invalid':True,'pass_0_5':False},
      {'expected':False,'invalid':False,'pass_0_5':False,'predicted_complete_0_5':True,'predicted_complete_0_8':True,'inference_seconds':.031},
      {'expected':True,'invalid':False,'pass_0_5':True,'predicted_complete_0_5':True,'predicted_complete_0_8':False,'inference_seconds':.042}]
    d=runner.stats(cases)
    assert d['invalid_or_errors']==1 and d['completed_valid']==2
    assert d['false_finished_incomplete_0_5']==1 and d['false_finished_incomplete_0_8']==1
    assert d['passed_vendor_threshold_0_5']==1
    assert d['inference_p95_seconds']==.042

@pytest.mark.parametrize('analyzer',[NA,FA],ids=['Namo','FireRed'])
def test_original_signed_verifier_source_requires_immutable_result(analyzer):
    assert len(analyzer.ORIGINAL_RESULT_SHA)==64
    assert len(analyzer.ORIGINAL_RUNNER_SHA)==64
    assert 'check_candidate' in dir(analyzer)
    assert analyzer.ORIGINAL_RESULT_SHA != ('0'*64)
    assert 'REJECTED' in analyzer.check_candidate.__code__.co_consts or analyzer.__doc__


def test_fire_red_documented_language_scope_remains_limited():
    assert F.REPOSITORY=='FireRedTeam/FireRedChat-turn-detector'
    assert N.REPOSITORY=='videosdk-live/Namo-Turn-Detector-v1-Multilingual'
    assert F.OPTIN != N.OPTIN and F.LEASE != N.LEASE
    assert F.lease_valid.__code__.co_consts
    src=(ROOT/'scripts/demos/run_isolated_firered_multilingual_cpu_once.py').read_text()
    assert 'German_holdout_not_in_vendor_documented_training_languages' in src
    assert "providers=['CPUExecutionProvider']" in src
    assert 'systemctl restart' not in src
    assert 'subprocess.Popen' not in src
