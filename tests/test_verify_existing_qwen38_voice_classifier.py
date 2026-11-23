"""Offline fixed-oracle and fail-closed Qwen3.8 classifier pilot tests."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'scripts/demos/verify_existing_qwen38_voice_classifier.py'
SPEC=importlib.util.spec_from_file_location('oneuse_qwen38_noaction',P)
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
ORIGINAL=ROOT/'src/swaag/benchmark/fixtures/voice_classifier_oracle_20261104.json'
HOLDOUT=ROOT/'scripts/demos/fixtures/voice_classifier_existing_coder_holdout_20261123.json'


def test_fixed_nonoverlapping_pilot_and_holdout_selection(monkeypatch):
    monkeypatch.setattr(M,'ORIGINAL',ORIGINAL)
    assert M.sha(HOLDOUT)==M.EXPECTED_HASHES['nov23_holdout']
    assert M.sha(ORIGINAL)==M.EXPECTED_HASHES['original_oracle']
    pilot=M.cases_for('pilot8',HOLDOUT)
    remainder=M.cases_for('remaining27',HOLDOUT)
    holdout=M.cases_for('fresh_holdout30',HOLDOUT)
    assert (len(pilot),len(remainder),len(holdout))==(8,27,30)
    assert [c['case_id'] for c in pilot]==M.QWEN_PILOT_ORDER
    assert set(c['case_id'] for c in pilot).isdisjoint(c['case_id'] for c in remainder)
    assert set(c['case_id'] for c in holdout).isdisjoint(c['case_id'] for c in pilot+remainder)
    assert 'acknowledgment_interruption_correct' in M.QWEN_PILOT_ORDER
    assert 'base_turn_completion_hold_command' in M.QWEN_PILOT_ORDER


def test_preflight_without_network_or_recorder(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(M,'ORIGINAL',ORIGINAL)
    monkeypatch.setattr(M,'source_gate',lambda p:None)
    monkeypatch.setattr(M,'model_config',lambda:pytest.fail('No model call during preflight'))
    out=tmp_path/'not-created.json'
    assert M.main(['--phase','pilot8','--holdout',str(HOLDOUT),'--evidence',str(out)])==0
    assert not out.exists()
    d=json.loads(capsys.readouterr().out)
    assert d['status']=='read_only_no_model_call' and d['expected_cases']==8


def test_explicit_optin_and_exact_exclusive_output(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'ORIGINAL',ORIGINAL)
    monkeypatch.setattr(M,'source_gate',lambda p:None)
    monkeypatch.setattr(M,'BASE',tmp_path)
    monkeypatch.delenv(M.OPT_IN,raising=False)
    cmd=['--phase','pilot8','--holdout',str(HOLDOUT),'--evidence',str(tmp_path/'case.json'),'--run']
    with pytest.raises(ValueError,match='Explicit'):
        M.main(cmd)
    assert not (tmp_path/'case.json').exists()
    monkeypatch.setenv(M.OPT_IN,'1')
    (tmp_path/'case.json').write_text('{"status":"uncertain"}')
    with pytest.raises(FileExistsError):M.main(cmd)


def test_two_consecutive_invalids_stop_at_two_cases(tmp_path,monkeypatch):
    cls=type('Classifier',(),{'classify_turn_completion':lambda self,s:{'complete':False},
      'classify_interruption':lambda self,s,a:{'interrupt':False}})
    monkeypatch.setattr(M,'model_config',lambda:(cls(),
      lambda *a:{'passed':False,'classifier_valid':False,'kind':'classifier_invalid'},
      {'calls':0}))
    cases=[{'case_id':str(i),'family':'turn_completion','text':'word','expected':True} for i in range(8)]
    receipt=M.execute(cases,tmp_path/'isolated.json','pilot8')
    assert receipt['status']=='aborted_due_to_consecutive_invalid_classifier_responses'
    assert len(receipt['results'])==2 and len(receipt['unattempted_case_ids'])==6
    assert receipt['failed']==2 and receipt['passed']==0


def test_malformed_holdout_never_triggers_model(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'ORIGINAL',ORIGINAL)
    d=json.loads(HOLDOUT.read_text());d['cases'][0]['expected']='true'
    p=tmp_path/'bad.json';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Malformed'):
        M.cases_for('fresh_holdout30',p)


def test_only_expected_existing_model_no_tools_in_implementation():
    s=P.read_text()
    assert 'chat_url=QWEN_URL' in s and 'model=QWEN_MODEL' in s
    assert 'request_timeout_seconds=8.0' in s and 'enable_thinking=False' in s
    assert "'tools' in payload" in s and "count['calls']+=1" in s
    assert "'model')!=QWEN_MODEL" in s
    assert 'subprocess' not in s and 'submit_turn' not in s
    assert M.QWEN_URL=='http://10.8.0.7:14835/v1/chat/completions'
    assert M.QWEN_MODEL=='qwen38-27b-q8-full-262k'


def test_slow_or_incomplete_pilot_refuses_extended_shared_model_use(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'BASE',tmp_path)
    p=tmp_path/'qwen38-pilot8-original-noaction-result.json'
    example={'status':'completed','phase':'pilot8','model':M.QWEN_MODEL,
             'model_endpoint':M.QWEN_URL,'fixed_source_sha256':M.EXPECTED_HASHES,
             'passed':8,'failed':0,'model_requests':8,
             'response_p95_seconds':5.082,
             'results':[{'case_id':i,'state':'finished','passed':True,
               'classifier_valid':True} for i in M.QWEN_PILOT_ORDER]}
    p.write_text(json.dumps(example))
    with pytest.raises(ValueError,match='latency exceeds'):
        M.pilot_permits_extended_inference()
    example['response_p95_seconds']=1.8
    p.write_text(json.dumps(example))
    assert M.pilot_permits_extended_inference()['source_pinned_pilot_passed']==8
    example['passed']=7
    p.write_text(json.dumps(example))
    with pytest.raises(ValueError,match='Pilot result is not'):
        M.pilot_permits_extended_inference()


def test_frozen_pilot_rejection_happens_before_new_receipt_or_model(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'BASE',tmp_path)
    monkeypatch.setattr(M,'ORIGINAL',ORIGINAL)
    monkeypatch.setattr(M,'source_gate',lambda *a:None)
    monkeypatch.setenv(M.OPT_IN,'1')
    monkeypatch.setattr(M,'execute',lambda *a:pytest.fail('No inference after invalid pilot'))
    dest=tmp_path/'never-created.json'
    with pytest.raises(ValueError,match='Missing unique completed'):
        M.main(['--phase','remaining27','--holdout',str(HOLDOUT),
                '--evidence',str(dest),'--run'])
    assert not dest.exists()
