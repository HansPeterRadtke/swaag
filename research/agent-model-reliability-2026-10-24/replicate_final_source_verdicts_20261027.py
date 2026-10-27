#!/usr/bin/env python3
"""Independent source-version replication, using genuine isolated SWAAG tools.

Six independent existing Git file/source-heading pairs; same-name decoy,
real read_file and real persisted/reopened SQLite/history/control receipts.
The LLM client is SCRIPTED, not a local Qwen model. All source originals read
from immutable Git objects. Never touches the live user/service state.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import time

from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.communication import CommunicationService
from tests.test_agent_action_loop import FakeModelClient,_action
from tests.test_fast_foreground_reply import fast

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
VERSION='independent_original_git_source_verdicts_v1_20261027'
COMMIT='2f7667d71ae9a15bb4dcfe7ddb837aff38b79015'
DOCS={
 'l1':{'path':'docs/fast-long-history-candidates-2026-10-21.md',
       'sha256':'0ca57219ccad1cc76893fe72e2e37561e18ff28bdc8cb70bd9e70607f540e7bf'},
 'l2':{'path':'docs/fast-worker-world-context-2026-10-20.md',
       'sha256':'ecd807546428758fe72fc40489d7eeff8133dabd777d68a6bc2b589edb8edecf'},
 'l3':{'path':'docs/history-action-scope-2026-10-16.md',
       'sha256':'78d72fd8b81d0abb06ce7e8967103c1aa7f2e41133360ad07b3a54804cc155a2'},
}
CASES=[
 {'id':'l1_yes','doc':'l1','heading':'## Problem and measured cause','expected':True,'decoy':'l2'},
 {'id':'l1_no','doc':'l1','heading':'## Implemented data flow','expected':False,'decoy':'l2'},
 {'id':'l2_yes','doc':'l2','heading':'## Implemented data flow','expected':True,'decoy':'l1'},
 {'id':'l2_no','doc':'l2','heading':'# History reuse across ordinary action delivery — 2026-10-16','expected':False,'decoy':'l3'},
 {'id':'l3_yes','doc':'l3','heading':'# History reuse across ordinary action delivery — 2026-10-16','expected':True,'decoy':'l1'},
 {'id':'l3_no','doc':'l3','heading':'## Problem and measured cause','expected':False,'decoy':'l1'},
]
ARMS=('read_accurate','read_inverted','direct_yes_no_tool','cautious_no_tool')
AFFIRMATIVE='Yes. The active original source contains the exact requested heading.'
NEGATIVE='No. The active original source does not contain the exact requested heading.'
ABSTAIN='I have not read the file and cannot verify whether that exact heading is present.'
FROZEN_CASES_SHA256='595f647586fe6bace2121b840d0ad4297bfb9e74b7ef185d37c4a16822a52c3f'  # Exact static input and annotation identities pinned BEFORE any execution.


def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()


def manifest():
    return sha(json.dumps(CASES,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8'))


def independent_source_gold():
    original={}
    for key,doc in DOCS.items():
        raw=subprocess.run(['git','show',COMMIT+':'+doc['path']],cwd=ROOT,
            capture_output=True,check=True).stdout
        if sha(raw)!=doc['sha256']:
            raise ValueError('Git source bytes changed for '+doc['path'])
        original[key]=raw
    if FROZEN_CASES_SHA256 is None or manifest()!=FROZEN_CASES_SHA256:
        raise ValueError('Cases must be explicitly frozen before any test run')
    for case in CASES:
        source=original[case['doc']].decode('utf-8')
        decoy=original[case['decoy']].decode('utf-8')
        if (case['heading'] in source.splitlines())!=case['expected']:
            raise ValueError('Source-derived target gold drift '+case['id'])
        if (case['heading'] in decoy.splitlines())==case['expected']:
            raise ValueError('Decoy must have opposite original source label '+case['id'])
    return original


def config(root:Path):
    env={'SWAAG__SESSIONS__ROOT':str(root/'sessions'),
         'SWAAG__AGENT_DATA__ROOT':str(root/'agent-data'),
         'SWAAG__TOOLS__READ_ROOTS':json.dumps([str(root)]),
         'SWAAG__TOOLS__STAGED_DISCOVERY':'false',
         'SWAAG__RUNTIME__COMPLETION_EVALUATION_ENABLED':'false',
         'SWAAG__MODEL__BASE_URL':'http://127.0.0.1:9999',
         'SWAAG__MODEL__CACHE_ENABLED':'false',
         'SWAAG__RETRIEVAL__BACKEND':'unavailable'}
    cfg=load_config(env=env)
    cfg.model.context_limit=32000
    cfg.communication.fast_initial_reply=True
    cfg.communication.poll_seconds=.01
    cfg.communication.early_tool_acknowledgement=True
    cfg.tools.enabled=['read_file']
    return cfg


def user_request(path,heading):
    return (f'Read the exact current original report at {path} using read_file. '
            f'Is the exact heading {heading!r} present as a complete Markdown heading line? '
            'Answer yes or no based on the current original file, not on a previous quote or a same-named historical file.')


def proposals(arm,case,active,original,user):
    correct=AFFIRMATIVE if case['expected'] else NEGATIVE
    inverted=NEGATIVE if case['expected'] else AFFIRMATIVE
    if arm=='direct_yes_no_tool':return FakeModelClient([fast('answer',AFFIRMATIVE)])
    if arm=='cautious_no_tool':return FakeModelClient([fast('answer',ABSTAIN),_action(message=ABSTAIN)])
    if arm not in ('read_accurate','read_inverted'):raise ValueError('unknown arm')
    expected_final=correct if arm=='read_accurate' else inverted
    body=original.decode('utf-8')
    def check_original_source_before_output(req):
        if req['contract']!='agent_action':raise AssertionError('not an actual action output')
        prompt=req['prompt']
        if body not in prompt and json.dumps(body)[1:-1] not in prompt:
            raise AssertionError('the actual full source was not preserved in the final input')
        if 'SOURCE EVENT sequence=' not in prompt:
            raise AssertionError('tool provenance not present before final user response')
        if user not in prompt and json.dumps(user)[1:-1] not in prompt:
            raise AssertionError('original user task authority not retained')
        if str(active) not in prompt:raise AssertionError('active source identity not retained')
        return _action(message=expected_final)
    return FakeModelClient([
        fast('delegate','I will inspect the active original source file.'),
        _action(message='Reading the exact file.',
                tool_calls=[('read_file',{'path':str(active)})],continue_loop=True),
        check_original_source_before_output,
    ])


def verify_trial(case,arm,source,decoy_source):
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='swaag-independent-source-verdicts-') as directory:
        root=Path(directory)
        active=root/'active'/Path(DOCS[case['doc']]['path']).name
        decoy=root/'historical'/active.name
        active.parent.mkdir(parents=True)
        decoy.parent.mkdir(parents=True)
        active.write_bytes(source)
        decoy.write_bytes(decoy_source)
        assert active.name==decoy.name and sha(active.read_bytes())==DOCS[case['doc']]['sha256']
        assert (case['heading'] in decoy.read_text().splitlines()) != case['expected']
        cfg=config(root)
        user=user_request(active,case['heading'])
        control_id='independent-source-'+case['id']+'-'+arm
        client=proposals(arm,case,active,source,user)
        runtime=AgentRuntime(cfg,model_client=client)
        service=CommunicationService(runtime,orchestrator_runtime=runtime)
        try:
            stream=list(service.iter_orchestrator_message(user,control_id=control_id))
            assert stream[0]['event']=='accepted' and stream[-1]['event']=='completed'
            final=stream[-1]['result']['answer']
            sid=stream[0]['session_id']
            events=runtime.history.read_history(sid)
            selected={name:[e for e in events if e.event_type==name]
                for name in ('control_message_processed','turn_started','file_read_requested','tool_called',
                             'filesystem_read','tool_result','turn_finished','assistant_progress')}
            assert len(selected['control_message_processed'])==len(selected['turn_finished'])==1
            assert selected['control_message_processed'][0].payload.get('message')==user
            assert selected['turn_finished'][0].payload['assistant_text']==final
            model_requests=len(client.requests)
            if arm in ('read_accurate','read_inverted'):
                assert len(selected['tool_called'])==len(selected['tool_result'])==len(selected['filesystem_read'])==1
                assert len(selected['file_read_requested'])==1
                assert len(selected['assistant_progress'])==1
                read=selected['filesystem_read'][0]
                result=selected['tool_result'][0]
                assert read.payload['text'].encode('utf-8')==source
                assert result.payload['output']['text'].encode('utf-8')==source
                assert Path(read.payload['path']).resolve()==active.resolve()
                assert Path(result.payload['output']['path']).resolve()==active.resolve()
                assert selected['tool_called'][0].sequence<read.sequence<result.sequence<selected['turn_finished'][0].sequence
                assert [x['contract'] for x in client.requests]==['fast_foreground_response','agent_action','agent_action']
            else:
                assert not selected['tool_called'] and not selected['tool_result'] and not selected['filesystem_read']
            first_events=len(events)
            last_hash=events[-1].hash
            rebuilt=runtime.history.rebuild_from_history(sid)
            assert rebuilt.event_count==events[-1].sequence and rebuilt.last_event_hash==last_hash
        finally:service.close()
        # Truly reopen existing persistent SQLite session with an empty client.
        empty=FakeModelClient([])
        restored=AgentRuntime(cfg,model_client=empty)
        second=CommunicationService(restored,orchestrator_runtime=restored)
        try:
            replay=list(second.iter_orchestrator_message(user,control_id=control_id))
            assert replay[-1]['event']=='completed' and replay[-1]['result']['answer']==final
            assert replay[0]['session_id']==sid
            assert len(empty.requests)==0
            after=restored.history.read_history(sid)
            assert len(after)==first_events and after[-1].hash==last_hash
            assert sum(e.event_type=='tool_called' for e in after)==len(selected['tool_called'])
            assert sum(e.event_type=='tool_result' for e in after)==len(selected['tool_result'])
        finally:second.close()
        guessed=final==AFFIRMATIVE or final==NEGATIVE
        predicted=final==AFFIRMATIVE if guessed else None
        true_gold=case['expected']
        wrong=guessed and predicted!=true_gold
        grounded=len(selected['tool_result'])==1 and guessed
        return {
            'version':VERSION,'case_id':case['id'],'document_id':case['doc'],
            'original_source_git_commit':COMMIT,'original_source_git_path':DOCS[case['doc']]['path'],
            'original_source_sha256':sha(source),'original_decoy_sha256':sha(decoy_source),
            'active_source_and_historical_decoy_same_basename':True,
            'original_exact_heading':case['heading'],'independent_git_gold_heading_present':true_gold,
            'arm':arm,'final_user_visible_answer':final,
            'final_asserts_source_fact':bool(guessed),'final_correct_against_exact_git_heading':bool(guessed and not wrong),
            'final_wrong_against_exact_git_heading':bool(wrong),
            'unverified_coincidental_right_answer':bool(guessed and not wrong and not grounded),
            'source_grounded_and_final_correct':bool(grounded and not wrong),
            'source_read_success_but_final_wrong':bool(grounded and wrong),
            'cautious_abstention_unresolved':not guessed,
            'original_user_request_sha256':sha(user.encode()),
            'control_id':control_id,'session_id':sid,
            'number_scripted_model_requests':model_requests,
            'model_contracts':[x['contract'] for x in client.requests],
            'input_prompt_whitespace_word_proxy':sum(len(str(x['prompt']).split()) for x in client.requests),
            'full_isolated_harness_wall_seconds':round(time.perf_counter()-started,6),
            'actual_read_file_tool_calls':len(selected['tool_called']),
            'actual_read_file_tool_results':len(selected['tool_result']),
            'actual_filesystem_source_reads':len(selected['filesystem_read']),
            'durable_control_event_sequence':selected['control_message_processed'][0].sequence,
            'durable_control_event_hash':selected['control_message_processed'][0].hash,
            'durable_final_turn_sequence':selected['turn_finished'][0].sequence,
            'durable_final_turn_hash':selected['turn_finished'][0].hash,
            'durable_tool_result_event_sequence':selected['tool_result'][0].sequence if selected['tool_result'] else None,
            'durable_tool_result_event_hash':selected['tool_result'][0].hash if selected['tool_result'] else None,
            'persisted_final_event_count':first_events,
            'persisted_final_history_chain_hash':last_hash,
            'reopened_session_and_control_replay_without_new_effects':True,
            'source_bytes_not_written_to_git_evidence':True,
            'real_qwen_calls_or_actual_gpu_inference':0,
        }


def aggregate(all_rows,failures):
    groups={}
    for arm in ARMS:
        selected=[r for r in all_rows if r['arm']==arm]
        errors=[r for r in failures if r['arm']==arm]
        cnt=Counter()
        for r in selected:
            gold=r['independent_git_gold_heading_present']
            if r['final_asserts_source_fact']:
                correct=r['final_correct_against_exact_git_heading']
                label='TP' if gold and correct else 'TN' if not gold and correct else 'FP' if not gold else 'FN'
            else:label='ABSTAIN_POS' if gold else 'ABSTAIN_NEG'
            cnt[label]+=1
        groups[arm]={
            'planned_cases':len(CASES),'verified_completed_cases':len(selected),'technical_failed_cases':len(errors),
            'TP':cnt['TP'],'TN':cnt['TN'],'FP':cnt['FP'],'FN':cnt['FN'],
            'ABSTAIN_POS':cnt['ABSTAIN_POS'],'ABSTAIN_NEG':cnt['ABSTAIN_NEG'],
            'source_grounded_final_correct_count':sum(r['source_grounded_and_final_correct'] for r in selected),
            'source_read_success_but_wrong_final_count':sum(r['source_read_success_but_final_wrong'] for r in selected),
            'unverified_coincidental_correct_count':sum(r['unverified_coincidental_right_answer'] for r in selected),
            'true_read_file_tool_call_count':sum(r['actual_read_file_tool_calls'] for r in selected),
            'true_read_file_tool_result_count':sum(r['actual_read_file_tool_results'] for r in selected),
            'durable_reopen_replay_successes':sum(r['reopened_session_and_control_replay_without_new_effects'] for r in selected),
            'scripted_model_requests_total':sum(r['number_scripted_model_requests'] for r in selected),
            'prompt_whitespace_word_proxy_total':sum(r['input_prompt_whitespace_word_proxy'] for r in selected),
            'total_harness_wall_seconds':round(sum(r['full_isolated_harness_wall_seconds'] for r in selected),5),
        }
    return groups


def run():
    source=independent_source_gold()
    root=HERE/'evidence/independent-last-mile-replication'
    root.mkdir(parents=True,exist_ok=True)
    journal=root/'immutable_trial_stream_v1.jsonl'
    summary_path=root/'source_verdicts_and_cost_v1.json'
    for p in (journal,summary_path):
        if p.exists():raise FileExistsError('Refuse to replay or overwrite a previous completed source/tool experiment: '+str(p))
    cases=[];failures=[];began=time.perf_counter()
    with journal.open('x',encoding='utf8') as f:
        def record(data):
            f.write(json.dumps(data,sort_keys=True,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())
        record({'event':'immutable_protocol_begin','version':VERSION,'case_manifest_sha256':FROZEN_CASES_SHA256,
                'source_original_git_commit':COMMIT,'four_arms':list(ARMS),
                'source_original_summaries':DOCS,'models_are_scripted':True})
        for i,case in enumerate(CASES):
            # Alternating fully specified method order so a single method is
            # not systematically warmed first on an occupied laptop.
            arms=list(ARMS)
            rotated=arms[i%len(arms):]+arms[:i%len(arms)]
            for arm in rotated:
                print('CASE',case['id'],arm,flush=True)
                record({'event':'isolated_case_begin','case_id':case['id'],'arm':arm,
                        'no_external_owner_effects':True})
                try:
                    r=verify_trial(case,arm,source[case['doc']],source[case['decoy']])
                except Exception as exc:
                    failure={'case_id':case['id'],'arm':arm,'type':type(exc).__name__,
                         'summary':str(exc)[:250],'do_not_score_as_semantic_error':True}
                    failures.append(failure)
                    record({'event':'isolated_case_failed','error':failure})
                    print('CASE FAILURE',case['id'],arm,failure['type'],flush=True)
                    continue
                cases.append(r)
                record({'event':'isolated_case_completed','result':r})
        record({'event':'all_cases_attempted','planned':len(CASES)*len(ARMS),
                'completed':len(cases),'technical_failed':len(failures)})
    report={'version':VERSION,'original_git_commit':COMMIT,'immutable_cases_sha256':FROZEN_CASES_SHA256,
        'original_source_documents':DOCS,'source_labels_basis':'exact Markdown heading line in original SHA-checked Git object',
        'original_files_and_original_model_prompts_not_committed':True,
        'models_scripted_not_qwen':True,'actual_qwen_llm_calls':0,'production_worker_side_effects':0,
        'test_input_cases':[{'id':c['id'],'doc':c['doc'],'heading':c['heading'],'independent_gold':c['expected'],
                             'decoy_doc_id':c['decoy']} for c in CASES],
        'cost_and_final_outcomes_by_arm':aggregate(cases,failures),
        'technical_failures_no_semantic_grade':failures,
        'full_trial_wall_seconds':round(time.perf_counter()-began,5),
        'provenance_limits':[
          'The material consists of actual unmodified Git source documents, but the head-presence questions are six investigator-selected exact-string predicates and do not represent natural semantic relevance.',
          'The actual SWAAG read_file, SQLite, source-event and final-turn paths run; fake model responses are scripted and deliberately correct/inverted by condition.',
          'A model can be wrong despite complete source availability, and a correct looking unsupported answer remains unverified.',
          'Six cases do not support general model error rates, and there are no human-written role-relevance gold labels.',
          'Actual llm token counts, Qwen-GPU usage, complete specialist reviewer cost and training syntax auxiliary evidence remain unavailable.',
        ]}
    summary_path.write_text(json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    print('SUMMARY',json.dumps(report['cost_and_final_outcomes_by_arm'],indent=2))
    print('SAVED',summary_path)
    return report

if __name__=='__main__':run()
