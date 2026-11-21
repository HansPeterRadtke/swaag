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
            content=ORACLE.pinned_source(ROOT,rel)
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
            'content':('Ten tests total: 7 + 3 = 10 in tests/test_history_replay.py and tests/test_storage_boundaries.py; '
                       'test_external_history_tampering_is_still_detected_with_file_stat_cache '
                       'and test_session_storage_rejects_symlink_escape.')}})
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
      'content':'Ten tests total: seven and three in tests/test_history_replay.py and tests/test_storage_boundaries.py. '
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


def test_unpaired_and_corrupt_tool_receipts_are_rejected(tmp_path):
    """A tool-looking event is not evidence unless its matching call actually ran."""
    import copy
    for scenario in ("orphan", "mismatched_requested_path", "duplicate",
                     "wrong_tool_name", "result_before_call"):
        path=event_data(tmp_path,wrong_calc=scenario=="orphan")
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if scenario=="orphan":
            rows.insert(-2,row(0,'tool_result',{'call_id':'never-called',
                'tool_name':'calculator','validated_input':{'expression':'7 + 3'},
                'output':{'expression':'7 + 3','result':10}}))
        elif scenario=="mismatched_requested_path":
            for entry in rows:
                if entry['type']=='tool_called' and entry['payload'].get('call_id')=='source-2':
                    entry['payload']['tool_input']={'path':'tests/unrelated.py'}
        elif scenario=="duplicate":
            idx=next(i for i,e in enumerate(rows) if e['type']=='tool_result'
                     and e['payload'].get('call_id')=='calc')
            rows.insert(idx+1,copy.deepcopy(rows[idx]))
        elif scenario=="wrong_tool_name":
            for entry in rows:
                if entry['type']=='tool_result' and entry['payload'].get('call_id')=='calc':
                    entry['payload']['tool_name']='read_file'
        elif scenario=="result_before_call":
            called=next(i for i,e in enumerate(rows) if e['type']=='tool_called'
                        and e['payload'].get('call_id')=='calc')
            result=next(i for i,e in enumerate(rows) if e['type']=='tool_result'
                        and e['payload'].get('call_id')=='calc')
            rows.insert(called,rows.pop(result))
        for i,e in enumerate(rows,1):e['sequence']=i
        path.write_text(''.join(json.dumps(e)+'\n' for e in rows))
        result=ORACLE.inspect(ROOT,path)
        assert result['status']=='failed_or_incomplete',scenario
        assert result['checks']['tool_receipts_correlate_with_calls'] is False,scenario


def test_incomplete_later_turn_cannot_inherit_earlier_finished_status(tmp_path):
    p=event_data(tmp_path)
    rows=[json.loads(line) for line in p.read_text().splitlines()]
    rows.append(row(len(rows)+1,'turn_started',{'user_text':'unfinished correction'}))
    p.write_text(''.join(json.dumps(e)+'\n' for e in rows))
    result=ORACLE.inspect(ROOT,p)
    assert result['status']=='failed_or_incomplete'
    assert result['checks']['last_turn_has_terminal_event'] is False


def test_first_pass_answer_needs_actual_source_grounding(tmp_path):
    original=event_data(tmp_path)
    rows=[json.loads(line) for line in original.read_text().splitlines()]
    final=next(e['payload']['message'] for e in rows if e['type']=='message_added'
               and e['payload']['message']['role']=='assistant')
    for missing in ('all', 'required test name', 'arithmetic'):
        if missing=='all':
            final['content']='Finished the audit. Everything looks fine.'
        elif missing=='required test name':
            final['content']='Ten tests total, tests/test_history_replay.py and tests/test_storage_boundaries.py; 7 + 3 = 10.'
        else:
            final['content']=(
                'Ten tests total in tests/test_history_replay.py and tests/test_storage_boundaries.py: '
                'test_external_history_tampering_is_still_detected_with_file_stat_cache '
                'and test_session_storage_rejects_symlink_escape.'
            )
        original.write_text(''.join(json.dumps(e)+'\n' for e in rows))
        assert ORACLE.inspect(ROOT,original)['status']=='failed_or_incomplete',missing


def test_historical_evaluation_is_pinned_to_live_source_revision():
    """A new test in current HEAD must not invalidate the Nov 20 journal."""
    import ast
    from hashlib import sha256
    for rel,digest in ORACLE.BASELINE_SOURCE_SHA256.items():
        original=ORACLE.pinned_source(ROOT,rel)
        assert sha256(original.encode()).hexdigest()==digest
    tests=[n for n in ast.parse(ORACLE.pinned_source(ROOT,'tests/test_history_replay.py')).body
           if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')]
    assert len(tests)==7
    assert ORACLE.BASELINE_REVISION=='0f663eace62d6148f42aa33bdcc02c436e5d95c6'


def test_repointing_history_to_a_new_revision_must_fail_closed(monkeypatch):
    monkeypatch.setattr(ORACLE,'BASELINE_REVISION','HEAD')
    with pytest.raises(ValueError,match='source hash mismatch'):
        ORACLE.pinned_source(ROOT,'tests/test_history_replay.py')
