"""Only unit tests for the independent verifier: NOT evidence of a model run.

The real Coder/tool attempt is recorded separately in a private one-use
session. Synthetic hashed event fixtures here reject unsupported success.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone,timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

from swaag.events import create_event,compute_event_hash

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/demos/verify_fresh_voice_safety_agent_demo.py'
SPEC=importlib.util.spec_from_file_location('source_only_firstpass_oracle',SCRIPT)
ORACLE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def fixture(tmp_path):
    private=tmp_path/'original_one_use'
    private.mkdir()
    task=private/'task.txt'
    prompt='Read the real test files, observe an error, list directory, calculate 4 + 7 and report.'
    task.write_text(prompt)
    manifest=private/'manifest.json'
    manifest.write_text(json.dumps({
        'source_revision':ORACLE.BASELINE_REVISION,
        'read_only_tools':['read_file','list_files','calculator'],
        'no_side_effect_tools':True,
        'fixed_task_sha256':hashlib.sha256(task.read_bytes()).hexdigest(),
        'original_source_hash_and_oracle':{
           name:{'sha256':hash,'required_exact_name':func,'top_level_test_count':count}
           for name,(hash,func,count) in ORACLE.FILES.items()}}))
    journal=private/'complete_history.jsonl'
    process=private/'original_one_use_exit.json'
    process.write_text(json.dumps({'one_shot_exit_code':0,'completion_claimed':False}))
    rows=[];prev=None
    def event(kind,payload):
        nonlocal prev
        e=create_event(session_id='synthetic_verifier_unit_only',sequence=len(rows)+1,
             event_type=kind,payload=payload,prev_hash=prev,
             timestamp=(datetime(2026,11,23,19,tzinfo=timezone.utc)+timedelta(seconds=len(rows)+1)).isoformat())
        rows.append(asdict(e));prev=e.hash
    def decide_tool(kind,name,args,callid,context):
        event('model_request_sent',{'kind':'action','prompt_mode':'standard','attempt':1,
             'request':{'prompt':context},'budget_report':{},'call_id':'request-'+callid})
        event('model_response_received',{'kind':'action','prompt_mode':'standard','attempt':1,
             'completion':{'text':'model selected '+name},'call_id':'request-'+callid,
             'completion_tokens':20,'elapsed_seconds':1.5,'first_token_seconds':0.2})
        event('agent_action_selected',{'action_index':len([r for r in rows if r['type']=='agent_action_selected'])+1,
             'action':{'tool_calls':[{'tool_name':name,'arguments':args}],'continue_loop':True},'occurrence':1})
        event('tool_called',{'call_id':callid,'tool_name':name,'tool_input':args})
    event('message_added',{'message':{'role':'user','content':prompt,'created_at':'now','metadata':{},'name':None}})
    event('turn_started',{'turn_index':1,'user_text':prompt})
    decide_tool('read','read_file',{'path':ORACLE.BAD_PATH},'bad',prompt)
    event('tool_error',{'call_id':'bad','tool_name':'read_file','tool_input':{'path':ORACLE.BAD_PATH},
       'error_type':'FilesystemError','error':'File does not exist'})
    decide_tool('list','list_files',{'path':'tests'},'list',prompt+'\nFilesystemError: File does not exist: '+ORACLE.BAD_PATH)
    event('tool_result',{'call_id':'list','tool_name':'list_files','raw_input':{'path':'tests'},
       'validated_input':{'path':'tests'},'output':{'entries':list(ORACLE.FILES),'count':2}})
    contents={}
    for i,name in enumerate(ORACLE.FILES,1):
        text=ORACLE.git_source(ROOT,name)
        contents[name]=text
        decide_tool('read','read_file',{'path':name},'source-'+str(i),prompt+'\nfound source after listing')
        event('tool_result',{'call_id':'source-'+str(i),'tool_name':'read_file',
            'raw_input':{'path':name},'validated_input':{'path':name},
            'output':{'text':text,'path':name}})
    decide_tool('calculate','calculator',{'expression':'4 + 7'},'calculator',prompt+'\nobserved test counts 4 and 7')
    event('tool_result',{'call_id':'calculator','tool_name':'calculator',
          'raw_input':{'expression':'4 + 7'},'validated_input':{'expression':'4 + 7'},
          'output':{'expression':'4 + 7','result':11}})
    event('model_request_sent',{'kind':'action','prompt_mode':'standard','attempt':1,
              'request':{'prompt':prompt+'\nactual calculator result 11'},'budget_report':{},
              'call_id':'final'})
    event('model_response_received',{'kind':'action','prompt_mode':'standard','attempt':1,
             'completion':{'text':'completed audit summary'},'call_id':'final',
             'completion_tokens':50,'elapsed_seconds':2.5,'first_token_seconds':0.4})
    event('agent_action_selected',{'action_index':6,
             'action':{'tool_calls':[],'continue_loop':False},'occurrence':1})
    final=('Observed missing file error, recovered with directory listing. '
          'Read '+', '.join(ORACLE.FILES)+'. Exact tests: '
          +', '.join(v[1] for v in ORACLE.FILES.values())+'. '
          'Top level tests 4 + 7 = 11 total.')
    event('message_added',{'message':{'role':'assistant','content':final,'created_at':'now','metadata':{},'name':None}})
    event('turn_finished',{'turn_index':1,'assistant_text':final,'tool_steps':5,'budget_reports':[]})
    save(journal,rows)
    return dict(journal=journal,manifest=manifest,task=task,process=process,rows=rows,repo=ROOT)


def save(journal,rows):
    journal.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows)+'\n')


def rehash(rows):
    prev=None
    for index,row in enumerate(rows,1):
        row['sequence']=index
        row['timestamp']=(datetime(2026,11,23,19,tzinfo=timezone.utc)+timedelta(seconds=index)).isoformat()
        row['prev_hash']=prev
        row['hash']=compute_event_hash(row)
        prev=row['hash']


def evaluate(fx):
    return ORACLE.inspect(fx['repo'],fx['journal'],fx['manifest'],fx['task'],fx['process'])


def test_valid_hash_linked_synthetic_source_oracle_is_consistent(tmp_path):
    fx=fixture(tmp_path)
    verdict=evaluate(fx)
    assert verdict['status']=='verified_first_pass'
    assert all(verdict['checks'].values())
    assert verdict['model_request_count']==6
    assert len(verdict['actual_tool_calls'])==5
    assert len(verdict['real_tool_failures'])==1
    assert len(verdict['actual_successful_tool_results'])==4
    assert verdict['single_turn'] is True


def test_changed_tool_result_with_forged_event_chain_still_fails_source_oracle(tmp_path):
    fx=fixture(tmp_path);rows=fx['rows']
    for r in rows:
        if r['type']=='tool_result' and r['payload'].get('tool_name')=='read_file':
            r['payload']['output']['text']='the source says something else';break
    rehash(rows);save(fx['journal'],rows)
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert verdict['checks']['two_real_source_reads_after_directory_recovery'] is False
    assert verdict['checks']['source_linked_hashed_event_chain_valid'] is True


def test_wrong_real_calculator_result_is_not_repaired_by_model_claim(tmp_path):
    fx=fixture(tmp_path);rows=fx['rows']
    for r in rows:
        if r['type']=='tool_result' and r['payload'].get('tool_name')=='calculator':
            r['payload']['output']['result']=0;break
    rehash(rows);save(fx['journal'],rows)
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert verdict['checks']['real_correct_calculator_after_reads'] is False


def test_single_byte_event_hash_corruption_is_refused_with_error(tmp_path):
    fx=fixture(tmp_path)
    next(r for r in fx['rows'] if r['type']=='tool_called')['payload']['tool_input']['path']='tests/something_else.py'
    save(fx['journal'],fx['rows'])
    with pytest.raises(ValueError,match='hash or schema'):
        evaluate(fx)


def test_failed_process_exit_never_passes_even_when_all_other_checks_pass(tmp_path):
    fx=fixture(tmp_path)
    fx['process'].write_text(json.dumps({'one_shot_exit_code':143}))
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert verdict['checks']['one_use_process_exited_successfully'] is False
    assert all(v for k,v in verdict['checks'].items() if k!='one_use_process_exited_successfully')


def test_unfinished_model_response_cannot_pass_even_with_all_tool_receipts(tmp_path):
    fx=fixture(tmp_path)
    rows=fx['rows']
    idx=next(i for i,r in enumerate(rows) if r['type']=='model_response_received')
    rows.pop(idx)
    rehash(rows);save(fx['journal'],rows)
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert verdict['checks']['all_model_requests_completed_without_retry'] is False


def test_no_real_model_requests_cannot_be_claimed_as_genuine(tmp_path):
    fx=fixture(tmp_path)
    rows=[r for r in fx['rows'] if r['type'] not in ('model_request_sent','model_response_received')]
    rehash(rows);save(fx['journal'],rows)
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert verdict['checks']['all_model_requests_completed_without_retry'] is False


def test_extra_user_turn_is_a_supervisor_correction_not_first_pass(tmp_path):
    fx=fixture(tmp_path)
    rows=fx['rows']
    clone=deepcopy(next(r for r in rows if r['type']=='turn_started'))
    clone['payload']['turn_index']=2
    rows.insert(-1,clone)
    rehash(rows);save(fx['journal'],rows)
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert verdict['checks']['single_model_user_turn_with_completion'] is False


def test_duplicate_tool_receipt_with_separately_valid_hash_chain_is_refused(tmp_path):
    fx=fixture(tmp_path);rows=fx['rows']
    original=deepcopy(next(r for r in rows if r['type']=='tool_result' and r['payload'].get('tool_name')=='calculator'))
    rows.insert(-1,original)
    rehash(rows);save(fx['journal'],rows)
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert 'duplicate' in ' '.join(verdict['tool_call_integrity_errors'])


def test_changing_frozen_task_or_manifest_is_refused_before_journal_acceptance(tmp_path):
    fx=fixture(tmp_path)
    fx['task'].write_text('different task')
    with pytest.raises(ValueError,match='task bytes differ'):
        evaluate(fx)
    fx['task'].write_text('Read the real test files, observe an error, list directory, calculate 4 + 7 and report.')
    d=json.loads(fx['manifest'].read_text());d['original_source_hash_and_oracle'][next(iter(ORACLE.FILES))]['top_level_test_count']=100
    fx['manifest'].write_text(json.dumps(d))
    with pytest.raises(ValueError,match='source/oracle'):
        evaluate(fx)


def test_unrelated_side_effect_tool_is_never_permitted(tmp_path):
    fx=fixture(tmp_path);rows=fx['rows']
    for r in rows:
        if r['type']=='tool_called' and r['payload'].get('tool_name')=='calculator':
            r['payload']['tool_name']='execute_shell';break
    rehash(rows);save(fx['journal'],rows)
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert verdict['checks']['only_readonly_allowlisted_tools_used'] is False


def test_preplanned_directory_listing_without_model_observing_error_is_not_recovery(tmp_path):
    fx=fixture(tmp_path)
    rows=fx['rows']
    request=next(r for r in rows if r['type']=='model_request_sent'
                 and 'FilesystemError' in r['payload']['request'].get('prompt',''))
    request['payload']['request']['prompt']='List files, but ignore tool error evidence.'
    rehash(rows);save(fx['journal'],rows)
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert verdict['checks']['model_received_error_before_recovering_with_directory_listing'] is False
    assert verdict['checks']['genuine_directory_listing_after_error'] is True


def test_forged_model_response_id_with_valid_rehashed_events_is_rejected(tmp_path):
    fx=fixture(tmp_path)
    rows=fx['rows']
    response=next(r for r in rows if r['type']=='model_response_received')
    response['payload']['call_id']='unpaired_fake_model_response'
    rehash(rows);save(fx['journal'],rows)
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert verdict['checks']['all_model_requests_completed_without_retry'] is False
    assert verdict['model_request_response_integrity_errors']


def test_forged_tool_input_not_selected_by_actual_model_action_fails(tmp_path):
    fx=fixture(tmp_path);rows=fx['rows']
    # An adversary can recompute all history hashes after forging a tool call;
    # source-backed model-to-tool causal correlation must still reject it.
    called=next(r for r in rows if r['type']=='tool_called'
                and r['payload'].get('tool_name')=='calculator')
    selected=next(r for r in rows if r['type']=='agent_action_selected'
                  and any(x.get('tool_name')=='calculator'
                          for x in (r.get('payload',{}).get('action',{}).get('tool_calls') or [])))
    selected['payload']['action']['tool_calls'][0]['arguments']={'expression':'0 + 0'}
    rehash(rows);save(fx['journal'],rows)
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert verdict['checks']['model_action_selected_exact_real_tool_calls'] is False
    assert verdict['checks']['source_linked_hashed_event_chain_valid'] is True


def test_forged_tool_call_without_any_model_selected_action_fails(tmp_path):
    fx=fixture(tmp_path);rows=fx['rows']
    rows=[r for r in rows if r['type']!='agent_action_selected']
    rehash(rows);save(fx['journal'],rows)
    verdict=evaluate(fx)
    assert verdict['status']=='failed_or_incomplete'
    assert verdict['checks']['model_action_selected_exact_real_tool_calls'] is False


def test_frozen_original_source_blob_revision_may_differ_from_later_runtime_head(tmp_path):
    fx=fixture(tmp_path)
    d=json.loads(fx['manifest'].read_text())
    d['source_revision']=ORACLE.subprocess.check_output(
        ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    d['original_source_blob_revision']=ORACLE.BASELINE_REVISION
    fx['manifest'].write_text(json.dumps(d))
    verdict=evaluate(fx)
    assert verdict['status']=='verified_first_pass'
    assert verdict['real_runtime_revision_recorded_in_manifest']==d['source_revision']
    d['original_source_blob_revision']='0'*40
    fx['manifest'].write_text(json.dumps(d))
    with pytest.raises(ValueError,match='source configuration'):
        evaluate(fx)
