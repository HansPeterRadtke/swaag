"""No-model admission, one-use output, immutable oracle and saturation stop."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import stat
import pytest

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'scripts/demos/verify_existing_coder_voice_classifier.py'
SPEC=importlib.util.spec_from_file_location('oneuse_existing_voice_classifier',P)
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
ORIGINAL=ROOT/'src/swaag/benchmark/fixtures/voice_classifier_oracle_20261104.json'
HOLDOUT=ROOT/'scripts/demos/fixtures/voice_classifier_existing_coder_holdout_20261123.json'


def test_november_23_holdout_was_predeclared_and_has_expected_sha():
    assert M.sha(HOLDOUT)==M.EXPECTED_HASHES['nov23_holdout']
    data=json.loads(HOLDOUT.read_text())
    assert data['not_run_on_coder_before_definition'] is True
    assert len(data['cases'])==30
    assert len({x['case_id'] for x in data['cases']})==30
    assert sum(x['family']=='turn_completion' for x in data['cases'])==20
    assert sum(x['family']=='interruption' for x in data['cases'])==10
    assert all(type(x['expected']) is bool for x in data['cases'])


def test_only_untested_29_original_cases_selected_and_no_six_pilot_duplication(monkeypatch):
    monkeypatch.setattr(M,'ORIGINAL',ORIGINAL)
    assert M.sha(ORIGINAL)==M.EXPECTED_HASHES['original_oracle']
    original=json.loads(ORIGINAL.read_text())['cases']
    assert len(original)==35
    selected=M.cases_for('remaining29',HOLDOUT)
    assert len(selected)==29
    assert not M.PILOT_IDS.intersection({c['case_id'] for c in selected})
    assert {x['case_id'] for x in original}==({x['case_id'] for x in selected}|M.PILOT_IDS)


def test_holdout_is_separate_and_never_fabricated_or_derived_from_pilot():
    h=M.cases_for('fresh_holdout30',HOLDOUT)
    assert len(h)==30
    assert all(x['case_id'].startswith('independent_20261123_') for x in h)
    assert not M.PILOT_IDS.intersection(x['case_id'] for x in h)


def test_any_malformed_or_duplicated_oracle_is_rejected(tmp_path,monkeypatch):
    original=json.loads(HOLDOUT.read_text())
    original['cases'][1]['case_id']=original['cases'][0]['case_id']
    path=tmp_path/'tampered.json';path.write_text(json.dumps(original))
    with pytest.raises(ValueError,match='Repeated case identities'):
        M.cases_for('fresh_holdout30',path)
    original['cases'][1]['case_id']='unique2'
    original['cases'][1]['expected']='true'
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError,match='Malformed'):
        M.cases_for('fresh_holdout30',path)


def test_source_integrity_checks_all_five_pins_without_external_effects(tmp_path,monkeypatch):
    gateway=tmp_path/'gateway';gateway.mkdir()
    monkeypatch.setattr(M,'GATEWAY_DIR',gateway)
    paths={'gateway.py':gateway/'voice_agent_gateway.py',
           'gateway_config.py':gateway/'voice_agent_gateway_config.py',
           'gateway_runtime_toml':tmp_path/'voice-gateway.toml',
           'original_oracle':tmp_path/'original-oracle.json',
           'nov23_holdout':tmp_path/'holdout.json'}
    for name,path in paths.items():path.write_text('private source for '+name)
    monkeypatch.setattr(M,'CONFIG',paths['gateway_runtime_toml'])
    monkeypatch.setattr(M,'ORIGINAL',paths['original_oracle'])
    monkeypatch.setattr(M,'EXPECTED_HASHES',{name:M.sha(path) for name,path in paths.items()})
    M.source_gate(paths['nov23_holdout'])
    for name,path in paths.items():
        content=path.read_text();path.write_text(content+' changed')
        with pytest.raises(ValueError,match='Source or fixed test oracle drift'):
            M.source_gate(paths['nov23_holdout'])
        path.write_text(content)
    q=tmp_path/'symlink.json';q.symlink_to(paths['nov23_holdout'])
    with pytest.raises(ValueError,match='Source or fixed test oracle drift'):
        M.source_gate(q)


def test_read_only_phase_does_not_call_models_or_create_receipts(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(M,'ORIGINAL',ORIGINAL)
    monkeypatch.setattr(M,'source_gate',lambda path:None)
    monkeypatch.setattr(M,'model_config',lambda :pytest.fail('preflight must never contact LLM'))
    out=tmp_path/'not-created.json'
    assert M.main(['--phase','remaining29','--holdout',str(HOLDOUT),'--evidence',str(out)])==0
    assert not out.exists()
    returned=json.loads(capsys.readouterr().out)
    assert returned['status']=='read_only_no_model_call' and returned['expected_cases']==29


def test_inference_requires_explicit_optin_and_exact_exclusive_receipt(monkeypatch,tmp_path):
    monkeypatch.setattr(M,'source_gate',lambda path:None)
    monkeypatch.setattr(M,'BASE',tmp_path)
    monkeypatch.delenv(M.OPT_IN,raising=False)
    monkeypatch.setattr(M,'execute',lambda *args:pytest.fail('model must not be invoked'))
    dest=tmp_path/'owned.json'
    with pytest.raises(ValueError,match='Explicit existing model'):
        M.main(['--phase','fresh_holdout30','--holdout',str(HOLDOUT),'--evidence',str(dest),'--run'])
    assert not dest.exists()
    monkeypatch.setenv(M.OPT_IN,'1')
    dest.write_text('{"status":"uncertain prior request"}')
    with pytest.raises(FileExistsError):
        M.main(['--phase','fresh_holdout30','--holdout',str(HOLDOUT),'--evidence',str(dest),'--run'])
    assert dest.read_text()=='{"status":"uncertain prior request"}'


def test_secretless_receipt_exclusive_and_atomic_update(tmp_path):
    path=tmp_path/'receipt.json'
    M.write_once(path,{'status':'started','no_effects':True})
    assert stat.S_IMODE(path.stat().st_mode)==0o600
    with pytest.raises(FileExistsError):M.write_once(path,{'fake':'overwrite'})
    M.update_owned_receipt(path,{'status':'observed','count':2})
    assert json.loads(path.read_text())['count']==2
    pending=path.with_name(path.name+'.atomic-pending');pending.write_text('unfinished')
    with pytest.raises(FileExistsError):M.update_owned_receipt(path,{'status':'new'})
    assert json.loads(path.read_text())['status']=='observed'


def test_two_consecutive_invalid_responses_stop_shared_model_traffic(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'model_config',lambda:(
       type('Gateway',(),{'classify_turn_completion':lambda self,text:{'complete':False,'kind':'classifier_invalid'},
                           'classify_interruption':lambda self,text,a:{'interrupt':False,'kind':'classifier_invalid'}})(),
       lambda f,expected,value:{'passed':False,'classifier_valid':False,'kind':value['kind']},
       {'calls':0}))
    cases=[{'case_id':'first_unreliable','family':'turn_completion','text':'first','expected':True},
           {'case_id':'second_unreliable','family':'turn_completion','text':'second','expected':True},
           {'case_id':'should_never_be_sent','family':'turn_completion','text':'third','expected':True}]
    target=tmp_path/'breaker.json'
    result=M.execute(cases,target,'fresh_holdout30')
    assert result['status']=='aborted_due_to_consecutive_invalid_classifier_responses'
    assert result['unattempted_case_ids']==['should_never_be_sent']
    assert len(result['results'])==2
    assert result['passed']==0 and result['failed']==2
    assert target.is_file()
    assert json.loads(target.read_text())['status']==result['status']


def test_valid_classifier_result_resets_saturation_counter(tmp_path,monkeypatch):
    def gateway():
        class G:
            calls=0
            def classify_turn_completion(self,text):
                self.calls+=1
                return {'complete':self.calls==2,'kind':'finished' if self.calls==2 else 'classifier_invalid'}
        return G()
    monkeypatch.setattr(M,'model_config',lambda:(gateway(),
           lambda f,expected,value:{'passed':value['kind']=='finished','classifier_valid':value['kind']=='finished'},
           {'calls':0}))
    cases=[{'case_id':str(i),'family':'turn_completion','text':'next','expected':True} for i in range(3)]
    d=M.execute(cases,tmp_path/'observed.json','remaining29')
    assert d['status']=='completed' and len(d['results'])==3
    assert d['results'][1]['classifier_valid'] is True


def test_no_executable_swaag_or_tool_call_is_in_classifier_runner():
    source=P.read_text()
    assert "'tools' in payload" in source
    assert 'model_config()' in source and 'Coder no-tool semantic inference' in source
    assert "model_requests" in source
    assert 'subprocess' not in source
    assert 'SWAAG_CODER_CLASSIFIER_EXTENDED_NOACTION' in source
