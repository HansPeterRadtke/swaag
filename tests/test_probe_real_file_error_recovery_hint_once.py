"""Test one-use diagnostic admission and returned model proposals without any LLM calls."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import hashlib
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/demos/probe_real_file_error_recovery_hint_once.py'
SPEC=importlib.util.spec_from_file_location('fixed_new_model_only_recovery_probe',SCRIPT)
PROBE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def fixture(tmp_path,monkeypatch):
    p=tmp_path/'actual_prior_journal_fixture.jsonl'
    base={
      'prompt':'Actual FilesystemError: File does not exist '+PROBE.ERROR_PATH+'\nAvailable tools: read_file, list_files, calculator',
      'n_predict':78643,'json_schema':{'type':'object'},
      'temperature':0,'top_p':1.0,'seed':45}
    rows=[]
    def add(kind,payload):
        rows.append({'sequence':len(rows)+1,'type':kind,'payload':payload})
    add('model_request_sent',{'request':{**base,'prompt':'First model initial source prompt'}})
    for i in range(4):
        add('tool_error',{'tool_name':'read_file','error_type':'FilesystemError',
          'tool_input':{'path':PROBE.ERROR_PATH},'error':'File does not exist'})
        add('model_request_sent',{'request':{**base,'seed':45+i}})
    p.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    monkeypatch.setattr(PROBE,'JOURNAL_SHA',hashlib.sha256(p.read_bytes()).hexdigest())
    return p


def test_source_immutable_prior_request_selected_only_after_first_real_error(tmp_path,monkeypatch):
    journal=fixture(tmp_path,monkeypatch)
    req,details=PROBE.read_original_request(journal)
    assert details['original_sequence']==3
    assert details['genuine_tool_error_sequence']==2
    assert details['real_file_error_count']==4
    assert PROBE.ERROR_PATH in req['prompt'] and 'FilesystemError' in req['prompt']
    assert 'list_files' in req['prompt']


def test_direct_new_request_cannot_be_identical_to_original(tmp_path,monkeypatch):
    prior=fixture(tmp_path,monkeypatch);old,_=PROBE.read_original_request(prior)
    new=PROBE.modified_request(old)
    assert new['prompt'].startswith(old['prompt'])
    assert new['prompt']!=old['prompt']
    assert new['n_predict']==768 and old['n_predict']==78643
    assert new['seed']!=old['seed']
    assert new['json_schema']==old['json_schema']
    assert not set(new)-set(old)
    assert 'Never repeat an unverified Send' in PROBE.RECOVERY_HINT
    assert 'new diagnostic action choice' in PROBE.RECOVERY_HINT


def test_rewritten_prior_journal_blocked_before_any_model_call(tmp_path,monkeypatch):
    prior=fixture(tmp_path,monkeypatch)
    prior.write_text(prior.read_text()+'\n')
    with pytest.raises(ValueError,match='pinned immutable'):
        PROBE.read_original_request(prior)


def test_fake_no_real_error_is_rejected(tmp_path,monkeypatch):
    prior=fixture(tmp_path,monkeypatch)
    rows=[json.loads(x) for x in prior.read_text().splitlines() if x.strip()]
    rows[1]['payload']['error_type']='PermissionError'
    prior.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    monkeypatch.setattr(PROBE,'JOURNAL_SHA',hashlib.sha256(prior.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='four real failures'):
        PROBE.read_original_request(prior)


def test_proposal_is_only_nonexecuted_action_evidence():
    good=PROBE.evaluate_proposal(json.dumps({'tool_calls':[
        {'tool_name':'list_files','arguments':{'path':'tests'}}]}))
    assert good['valid_json'] and good['has_proposed_listing']
    assert good['proposed_tools_executed']==0
    wrong=PROBE.evaluate_proposal(json.dumps({'tool_calls':[
        {'tool_name':'read_file','arguments':{'path':PROBE.ERROR_PATH}}]}))
    assert wrong['valid_json'] and not wrong['has_proposed_listing']
    assert PROBE.evaluate_proposal('not JSON')['valid_json'] is False


def test_no_action_one_use_distinct_model_response_is_persisted(tmp_path,monkeypatch,capsys):
    p=fixture(tmp_path,monkeypatch)
    monkeypatch.setenv(PROBE.OPTIN,'1')
    # pytest basetemp is an exclusive private /data/var/swaag-benchmarks path.
    assert '/data/var/swaag-benchmarks/' in str(tmp_path)
    remote_calls=[]
    class Response:
        status=200
        def __init__(self,payload):self.payload=payload
        def __enter__(self):return self
        def __exit__(self,*a):return None
        def read(self,limit):return json.dumps(self.payload).encode()
    def no_real_provider_http(req,timeout):
        remote_calls.append(req)
        if isinstance(req,str):
            assert req==PROBE.MODEL_CATALOG
            return Response({'data':[{'id':PROBE.EXPECTED_MODEL_ALIAS}]})
        assert req.full_url==PROBE.MODEL_ENDPOINT
        body=json.loads(req.data)
        assert body['n_predict']==768 and 'NEW_EXPLICIT_MODEL_RECOVERY_GUIDANCE' in body['prompt']
        return Response({'content':json.dumps({'tool_calls':[
          {'tool_name':'list_files','arguments':{'path':'tests'}}]}),
          'tokens_predicted':31})
    monkeypatch.setattr(PROBE.urllib.request,'urlopen',no_real_provider_http)
    output=tmp_path/'private-new-model-only-result'
    assert PROBE.main(['--journal',str(p),'--evidence',str(output),'--run'])==0
    assert len(remote_calls)==2
    result=json.loads((output/'result.json').read_text())
    assert result['status']=='completed_model_only'
    assert result['request_distinct_from_original']
    assert result['proposal']['has_proposed_listing']
    assert result['tools_executed']==0 and result['model_not_authorized_to_execute_any_tools']
    assert result['actual_multistep_agent_success_claimed'] is False
    assert (output/'attempt.json').is_file()
    with pytest.raises(ValueError,match='Destination must be'):
        PROBE.main(['--journal',str(p),'--evidence',str(output),'--run'])
    assert len(remote_calls)==2


def test_untrusted_model_catalogue_refuses_provider_request_and_preserves_unknown_receipt(tmp_path,monkeypatch):
    p=fixture(tmp_path,monkeypatch)
    monkeypatch.setenv(PROBE.OPTIN,'1')
    calls=[]
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*a):return None
        def read(self,limit):return b'{"data":[{"id":"unexpected_model"}]}'
    def fake_http(req,timeout):
        calls.append(req)
        assert isinstance(req,str) and req==PROBE.MODEL_CATALOG
        return Response()
    monkeypatch.setattr(PROBE.urllib.request,'urlopen',fake_http)
    path=tmp_path/'no-model-sent'
    assert PROBE.main(['--journal',str(p),'--evidence',str(path),'--run'])==3
    result=json.loads((path/'result.json').read_text())
    assert result['status']=='failed_or_unknown'
    assert result['error_type']=='ValueError'
    assert result['tools_executed']==0
    assert len(calls)==1
    assert (path/'attempt.json').exists()
    with pytest.raises(ValueError,match='Destination must be'):
        PROBE.main(['--journal',str(p),'--evidence',str(path),'--run'])
    assert len(calls)==1
