"""Tests for the independent buyer-demo oracle, NOT a mock agent demo.

The actual multi-step live agent execution and its source tool receipts
are captured in private run logs; these cases only guard the verifier
against falsely marking an unsupported claim as verified.
"""
import importlib.util
import json
from pathlib import Path
from datetime import datetime,timezone,timedelta

import pytest


ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/demos/verify_real_readonly_agent_demo.py'
SPEC=importlib.util.spec_from_file_location('live_agent_demo_oracle',SCRIPT)
ORACLE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def row(index,kind,payload):
    return {'sequence':index,'timestamp':(
        datetime(2026,11,20,tzinfo=timezone.utc)+timedelta(seconds=index)).isoformat(),
        'session_id':'test_fixture_only','type':kind,'payload':payload}


def event_data(tmp_path,*,wrong_calc=False,spoofed_only=False,wrong_read=False,completed=True):
    values=[];seq=0
    def add(kind,payload):
        nonlocal seq
        seq+=1;values.append(row(seq,kind,payload))
    add('turn_started',{'user_text':'fixed source audit'})
    if spoofed_only:
        add('message_added',{'message':{'role':'assistant',
            'content':'I completed all reads and calculated ten.'}})
    else:
        add('tool_called',{'call_id':'bad','tool_name':'read_file',
            'tool_input':{'path':ORACLE.BAD_PATH}})
        add('tool_error',{'call_id':'bad','tool_name':'read_file',
            'tool_input':{'path':ORACLE.BAD_PATH},
            'error_type':'FilesystemError','error':'File missing'})
        add('tool_called',{'call_id':'list','tool_name':'list_files',
            'tool_input':{'path':'tests'}})
        add('tool_result',{'call_id':'list','tool_name':'list_files',
            'validated_input':{'path':'tests'},'output':{
              'path':'tests','entries':list(ORACLE.FILES),'count':2}})
        for i,rel in enumerate(ORACLE.FILES,1):
            content=(ROOT/rel).read_text()
            cid='source-'+str(i)
            add('tool_called',{'call_id':cid,'tool_name':'read_file',
                'tool_input':{'path':rel}})
            add('tool_result',{'call_id':cid,'tool_name':'read_file',
                'validated_input':{'path':rel},
                'output':{'text':'corrupted source text' if wrong_read and i==2 else content,
                          'size_chars':len(content)}})
        cid='calc'
        expr='0 + 0' if wrong_calc else '7 + 3'
        add('tool_called',{'call_id':cid,'tool_name':'calculator',
            'tool_input':{'expression':expr}})
        add('tool_result',{'call_id':cid,'tool_name':'calculator',
            'validated_input':{'expression':expr},
            'output':{'expression':expr,'result':0 if wrong_calc else 10}})
    if completed:
        add('message_added',{'message':{'role':'assistant',
            'content':'Ten tests total, source functions verified, answer supported.'}})
        add('turn_completed',{})
    path=tmp_path/'fake_events_for_verifier_unit_test.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in values))
    return path


def test_good_receipts_pass_only_with_correct_calculation(tmp_path):
    result=ORACLE.inspect(ROOT,event_data(tmp_path))
    assert result['status']=='verified_first_pass'
    assert result['expected_total']==10
    assert len(result['calls'])==5
    assert all(result['checks'].values())


@pytest.mark.parametrize('variation',[
    {'wrong_calc':True},
    {'wrong_read':True},
    {'spoofed_only':True},
    {'completed':False},
])
def test_unsupported_claim_cannot_pass_oracle(tmp_path,variation):
    result=ORACLE.inspect(ROOT,event_data(tmp_path,**variation))
    assert result['status']=='failed_or_incomplete'
    assert not (result['turn_finished'] and all(result['checks'].values()))


def test_real_error_then_second_corrective_turn_is_distinguished(tmp_path):
    p=event_data(tmp_path,wrong_calc=True,completed=False)
    records=[json.loads(line) for line in p.read_text().splitlines()]
    next_seq=records[-1]['sequence']+1
    t=records[-1]['timestamp']
    def add(kind,payload):
        nonlocal next_seq
        records.append(row(next_seq,kind,payload))
        next_seq+=1
    add('model_call_failed',{'error_type':'ReadTimeout','error':'model_unavailable'})
    add('turn_started',{'user_text':'correct arithmetic'})
    add('tool_called',{'call_id':'calc2','tool_name':'calculator',
                       'tool_input':{'expression':'7 + 3'}})
    add('tool_result',{'call_id':'calc2','tool_name':'calculator',
        'validated_input':{'expression':'7 + 3'},
        'output':{'expression':'7 + 3','result':10}})
    add('message_added',{'message':{'role':'assistant',
      'content':'Ten tests total: seven and three. '
       'test_external_history_tampering_is_still_detected_with_file_stat_cache '
       'test_session_storage_rejects_symlink_escape. 0 + 0 was wrong; 7 + 3 corrected.'}})
    add('turn_finished',{})
    p.write_text(''.join(json.dumps(rec)+'\n' for rec in records))
    result=ORACLE.inspect(ROOT,p)
    assert result['status']=='verified_after_supervisor_correction'
    assert result['initial_model_failure_recorded']
    assert result['separate_correction_turn_observed']
    assert result['checks']['final_answer_is_source_grounded']
    assert result['checks']['actual_correct_calculator_operation']
