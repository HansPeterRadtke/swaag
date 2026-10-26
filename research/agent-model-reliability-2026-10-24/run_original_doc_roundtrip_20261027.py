#!/usr/bin/env python3
"""SWAAG full source-read/tool/stream/final/restart identity test on real git docs.

Harness uses real tools and persistent per-test SQLite, but scripted fake model
answers. No live Qwen or production service, no private file modifications.
Protocol: ORIGINAL_DOC_ROUNDTRIP_PROTOCOL_2026-10-27.md.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from swaag.config import load_config
from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from tests.test_agent_action_loop import FakeModelClient, _action
from tests.test_fast_foreground_reply import fast

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
VERSION='SWAAG_original_real_git_document_source_roundtrip_v1_20261027'
SOURCE='docs/history-index-cost-2026-10-16.md'
BASENAME='history-index-cost-2026-10-16.md'
SECTION='October 26 follow-up: canonical JSONL append visibility under concurrent readers'
GIT_VERSIONS={
    'old':{'commit':'140f24d478de22e428dabed499c4aa7d3931590d',
           'sha256':'b5fa7ab07881a3c83995b2f672940024b977e2590c5e6b93149a1266b8cc9e00',
           'has_section':False},
    'new':{'commit':'4267dfb52f2abdb04b3793c0069863ce518dbe10',
           'sha256':'ce9d7691f7fdc3f4c0ef019809c945bfd05b9c8a5855c4d916bea7785006ce72',
           'has_section':True},
}
OLD_ANSWER='No. The specified source revision does not contain the October 26 journal-lock follow-up.'
NEW_ANSWER='Yes. The specified source revision contains the October 26 per-session journal-lock follow-up.'
ABSTAIN='I have not read the source file, so I cannot verify which revision it contains.'


def sha(b:bytes):return hashlib.sha256(b).hexdigest()


def originals():
    ret={}
    for version,spec in GIT_VERSIONS.items():
        proc=subprocess.run(['git','show',spec['commit']+':'+SOURCE],cwd=ROOT,
            check=True,capture_output=True)
        raw=proc.stdout
        if sha(raw)!=spec['sha256']:
            raise ValueError('Original committed source version hash changed: '+version)
        text=raw.decode('utf-8')
        if (SECTION in text)!=spec['has_section']:
            raise ValueError('Original source did not contain expected exact section predicate: '+version)
        ret[version]=raw
    assert sha(ret['old'])!=sha(ret['new'])
    return ret


def config(root:Path):
    env={
        'SWAAG__SESSIONS__ROOT':str(root/'sessions'),
        'SWAAG__AGENT_DATA__ROOT':str(root/'agent-data'),
        'SWAAG__TOOLS__READ_ROOTS':json.dumps([str(root)]),
        'SWAAG__TOOLS__STAGED_DISCOVERY':'false',
        'SWAAG__RUNTIME__COMPLETION_EVALUATION_ENABLED':'false',
        'SWAAG__MODEL__BASE_URL':'http://127.0.0.1:9999',
        'SWAAG__MODEL__CACHE_ENABLED':'false',
        'SWAAG__RETRIEVAL__BACKEND':'unavailable',
    }
    out=load_config(env=env)
    out.model.context_limit=32000
    out.communication.fast_initial_reply=True
    out.communication.poll_seconds=0.01
    out.communication.early_tool_acknowledgement=True
    out.tools.enabled=['read_file']
    return out


def user_message(active:Path):
    return ('Read the exact current project report using the read_file tool at '
        +str(active)+'. Determine whether it contains the October 26 per-session '
        +'history journal-lock follow-up section. Answer based on that exact '
        +'file content and the most recent source observation, not an old quote.')


def make_client(active:Path, expected_raw:bytes, version:str):
    expected_text=expected_raw.decode('utf-8')
    expected_answer=NEW_ANSWER if GIT_VERSIONS[version]['has_section'] else OLD_ANSWER
    def source_finalizer(req):
        if req['contract']!='agent_action':
            raise AssertionError('unexpected response contract')
        prompt=req['prompt']
        # This is a deterministic assertion for a scripted fake model, NOT an
        # independently generated language-model response or semantic judgment.
        if expected_text not in prompt and json.dumps(expected_text)[1:-1] not in prompt:
            raise AssertionError('full exact read_file source missing in final model prompt')
        if 'SOURCE EVENT sequence=' not in prompt:
            raise AssertionError('source tool event identity was not forwarded to final model')
        if str(active) not in prompt:
            raise AssertionError('original explicit file identity lost from final model prompt')
        return _action(message=expected_answer)
    return FakeModelClient([
        fast('delegate','I will read the exact current source before answering.'),
        _action(message='Reading the source now.',tool_calls=[('read_file',{'path':str(active)})],continue_loop=True),
        source_finalizer,
    ])


def open_service(cfg,client):
    runtime=AgentRuntime(cfg,model_client=client)
    service=CommunicationService(runtime,orchestrator_runtime=runtime)
    return service,runtime


def verify_source_turn(runtime,stream,client,version,source_raw,active,decoy,user,control):
    assert [x['event'] for x in stream]==['accepted','assistant_progress','completed']
    expected=NEW_ANSWER if GIT_VERSIONS[version]['has_section'] else OLD_ANSWER
    assert stream[-1]['result']['answer']==expected
    sid=stream[0]['session_id']
    events=runtime.history.read_history(sid)
    selected=[]
    for event_type in ['file_read_requested','filesystem_read','tool_called','tool_result',
                       'turn_finished','control_message_processed']:
        selected.append((event_type,[x for x in events if x.event_type==event_type]))
    controls=[x for x in events if x.event_type=='control_message_processed' and x.payload.get('control_id')==control]
    assert len(controls)==1
    idx=controls[0].sequence
    relevant=[x for x in events if x.sequence>=idx]
    calls=[x for x in relevant if x.event_type=='tool_called' and x.payload.get('tool_name')=='read_file']
    results=[x for x in relevant if x.event_type=='tool_result' and x.payload.get('tool_name')=='read_file']
    reads=[x for x in relevant if x.event_type=='filesystem_read']
    turns=[x for x in relevant if x.event_type=='turn_finished' and x.payload.get('assistant_text')==expected]
    assert len(calls)==len(results)==len(reads)==len(turns)==1
    assert reads[0].payload['text'].encode('utf-8')==source_raw
    assert results[0].payload['output']['text'].encode('utf-8')==source_raw
    assert Path(reads[0].payload['path']).resolve()==active.resolve()
    assert Path(results[0].payload['output']['path']).resolve()==active.resolve()
    assert reads[0].payload['path']!=str(decoy)
    assert calls[0].sequence<reads[0].sequence<results[0].sequence<turns[0].sequence
    assert controls[0].payload.get('message')==user
    assert [x['contract'] for x in client.requests]==['fast_foreground_response','agent_action','agent_action']
    assert len(events)>5
    rebuilt=runtime.history.rebuild_from_history(sid)
    assert rebuilt.event_count==events[-1].sequence
    assert rebuilt.last_event_hash==events[-1].hash
    return {
        'version':version,'original_git_commit':GIT_VERSIONS[version]['commit'],
        'original_source_sha256':sha(source_raw),'source_section_present_exact_git_gold':GIT_VERSIONS[version]['has_section'],
        'source_path_alias':'active/'+BASENAME,'decoy_path_alias':'historical/'+BASENAME,
        'decoy_exists_and_same_basename':decoy.exists() and decoy.name==active.name,
        'original_input_user_instruction_sha256':sha(user.encode()),
        'persisted_session_id':sid,'persisted_control_id':control,
        'final_answer':expected,'correct_against_original_git_source':True,
        'verified_true_source_reads':len(reads),
        'verified_exact_byte_source_contents':True,
        'source_full_text_handed_to_model_after_tool':True,
        'actual_read_file_tool_called':len(calls),
        'actual_read_file_tool_result':len(results),
        'model_request_contracts':[x['contract'] for x in client.requests],
        'scripted_model_requests':len(client.requests),
        'scripted_model_prompt_whitespace_words_proxy':sum(len(str(x['prompt']).split()) for x in client.requests),
        'control_receipt_hash':controls[0].hash,'control_receipt_sequence':controls[0].sequence,
        'filesystem_read_receipt_hash':reads[0].hash,'filesystem_read_sequence':reads[0].sequence,
        'tool_called_receipt_hash':calls[0].hash,'tool_called_sequence':calls[0].sequence,
        'tool_result_receipt_hash':results[0].hash,'tool_result_sequence':results[0].sequence,
        'turn_finished_receipt_hash':turns[0].hash,'turn_finished_sequence':turns[0].sequence,
        'hash_chain_verified_at_exact_terminal_event':rebuilt.event_count==events[-1].sequence,
        'stream_event_names':[x['event'] for x in stream],
    }


def two_revision_roundtrip(root:Path,source_data):
    (root/'active').mkdir(parents=True);(root/'historical').mkdir(parents=True)
    active=root/'active'/BASENAME
    decoy=root/'historical'/BASENAME
    active.write_bytes(source_data['old']);decoy.write_bytes(source_data['new'])
    assert sha(active.read_bytes())==GIT_VERSIONS['old']['sha256']
    assert sha(decoy.read_bytes())==GIT_VERSIONS['new']['sha256']
    cfg=config(root)
    msg=user_message(active)
    started=time.perf_counter()
    client=make_client(active,source_data['old'],'old')
    service,runtime=open_service(cfg,client)
    try:
        first=list(service.iter_orchestrator_message(msg,control_id='exact-source-revision-old'))
        old=verify_source_turn(runtime,first,client,'old',source_data['old'],active,decoy,msg,'exact-source-revision-old')
    finally:service.close()
    old['roundtrip_wall_seconds']=round(time.perf_counter()-started,6)
    # True historical version change in the SOURCE, not a user-created fake claim.
    active.write_bytes(source_data['new']);decoy.write_bytes(source_data['old'])
    assert sha(active.read_bytes())==GIT_VERSIONS['new']['sha256']
    assert sha(decoy.read_bytes())==GIT_VERSIONS['old']['sha256']
    client2=make_client(active,source_data['new'],'new')
    service2,runtime2=open_service(cfg,client2)
    try:
        before_calls=len(client2.requests)
        old_replay=list(service2.iter_orchestrator_message(msg,control_id='exact-source-revision-old'))
        assert old_replay[-1]['result']['answer']==OLD_ANSWER
        assert len(client2.requests)==before_calls
        new_begin=time.perf_counter()
        fresh=list(service2.iter_orchestrator_message(msg,control_id='exact-source-revision-new'))
        new=verify_source_turn(runtime2,fresh,client2,'new',source_data['new'],active,decoy,msg,'exact-source-revision-new')
        assert new['persisted_session_id']==old['persisted_session_id']
        assert new['filesystem_read_sequence']>old['filesystem_read_sequence']
        assert new['tool_result_receipt_hash']!=old['tool_result_receipt_hash']
        new['roundtrip_wall_seconds']=round(time.perf_counter()-new_begin,6)
    finally:service2.close()
    # Restart *again* and validate both distinct historical source/answer receipts.
    client3=FakeModelClient([])
    service3,runtime3=open_service(cfg,client3)
    try:
        for control,answer in [('exact-source-revision-old',OLD_ANSWER),('exact-source-revision-new',NEW_ANSWER)]:
            replay=list(service3.iter_orchestrator_message(msg,control_id=control))
            assert replay[-1]['result']['answer']==answer
            assert replay[0]['session_id']==old['persisted_session_id']
        assert len(client3.requests)==0
        before=runtime3.history.read_history(old['persisted_session_id'])
        count_before=Counter(e.event_type for e in before)
        try:
            list(service3.iter_orchestrator_message(msg+' The control identity is reused incorrectly.',control_id='exact-source-revision-old'))
        except ValueError:
            rejected=True
        else:
            rejected=False
        assert rejected,'reused control with conflicting user instruction must be refused'
        after=runtime3.history.read_history(old['persisted_session_id'])
        count_after=Counter(e.event_type for e in after)
        assert len(client3.requests)==0
        assert count_after['tool_called']==count_before['tool_called']==2
        assert count_after['tool_result']==count_before['tool_result']==2
        assert count_after['turn_finished']==count_before['turn_finished']==2
        rebuilt=runtime3.history.rebuild_from_history(old['persisted_session_id'])
        assert rebuilt.event_count==after[-1].sequence
        assert rebuilt.last_event_hash==after[-1].hash
        durable={'two_original_distinct_source_versions':True,
          'reopened_original_persisted_session_twice':True,
          'identical_original_user_message_sha256':sha(msg.encode()),
          'old_control_historical_answer_replayed_after_file_changed':True,
          'both_old_and_new_controls_replayed_after_second_restart':True,
          'new_model_calls_during_both_restarts_replays':0,
          'new_tool_calls_during_both_restarts_replays':0,
          'conflicting_user_instruction_same_control_id_rejected':rejected,
          'source_versioned_receipts_remain_distinct':True,
          'persisted_history_hash_chain_matches_terminal_sequence':True,
          'persisted_final_tool_count':count_before['tool_result'],
          'persisted_final_turn_count':count_before['turn_finished'],
          'persisted_final_filesystem_read_count':count_before['filesystem_read'],
          'last_event_sequence':after[-1].sequence,'last_event_hash':after[-1].hash}
    finally:service3.close()
    return [old,new],durable


def one_no_tool_control(root:Path,source_raw:bytes,version:str,control_kind:str):
    root.mkdir(parents=True)
    active=root/'active'/BASENAME
    active.parent.mkdir(parents=True)
    active.write_bytes(source_raw)
    cfg=config(root)
    user=user_message(active)
    proposal=NEW_ANSWER if control_kind=='unguarded_direct_yes' else ABSTAIN
    deep=_action(message=ABSTAIN)
    fake=FakeModelClient([fast('answer',proposal),deep])
    service,runtime=open_service(cfg,fake)
    start=time.perf_counter()
    try:
        stream=list(service.iter_orchestrator_message(user,control_id=control_kind+'-'+version))
        assert stream[0]['event']=='accepted' and stream[-1]['event']=='completed'
        final=stream[-1]['result']['answer']
        events=runtime.history.read_history(stream[0]['session_id'])
        types=Counter(x.event_type for x in events)
        assert types['tool_result']==types['filesystem_read']==types['tool_called']==0
        assert types['turn_finished']==types['control_message_processed']==1
        expected=GIT_VERSIONS[version]['has_section']
        asserts=final.startswith('Yes.') or final.startswith('No.')
        prediction=(final.startswith('Yes.') if asserts else None)
        wrong_assert=asserts and prediction!=expected
        return {'version':version,'control':control_kind,
          'source_original_commit':GIT_VERSIONS[version]['commit'],
          'original_source_sha256':sha(source_raw),
          'gold_section_present':expected,
          'final_user_answer':final,
          'asserted_unverified_fact':asserts,
          'factual_assertion_wrong_against_original_source':bool(wrong_assert),
          'correct_text_without_source_is_not_accepted_as_evidence':True,
          'source_verified_task_complete':False,
          'safe_unresolved_abstention':not asserts,
          'no_real_file_tool_called':True,
          'model_requests_to_scripted_client':len(fake.requests),
          'model_request_contracts':[x['contract'] for x in fake.requests],
          'model_prompt_whitespace_words_proxy_total':sum(len(str(x['prompt']).split()) for x in fake.requests),
          'end_to_end_wall_seconds':round(time.perf_counter()-start,6),
          'original_user_request_sha256':sha(user.encode()),
          'control_receipt_hash':next(e.hash for e in events if e.event_type=='control_message_processed'),
          'turn_receipt_hash':next(e.hash for e in events if e.event_type=='turn_finished')}
    finally:service.close()


def run():
    source=originals()
    began=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='swaag-authentic-document-roundtrip-') as tmp:
        root=Path(tmp)
        main,durable=two_revision_roundtrip(root/'full_source',source)
        controls=[]
        for version in ('old','new'):
            for control in ('unguarded_direct_yes','cautious_no_tool'):
                controls.append(one_no_tool_control(root/f'negative_{version}_{control}',source[version],version,control))
    result={'version':VERSION,'protocol':'ORIGINAL_DOC_ROUNDTRIP_PROTOCOL_2026-10-27.md',
      'original_repository_source':'docs/history-index-cost-2026-10-16.md',
      'git_snapshot_source_sha256':{k:{'commit':v['commit'],'source_sha256':v['sha256'],
         'header_presence_is_ground_truth':v['has_section']} for k,v in GIT_VERSIONS.items()},
      'authoritative_gold_type':'exact immutable real Git source file bytes + exact section predicate; no human or model vote',
      'model_outputs_are_scripted':True,
      'real_qwen_inference_calls':0,'real_external_services_modified':0,
      'gpu_vram_or_true_llamacpp_tokens_measured':False,
      'source_read_to_persisted_final_and_replay':main,
      'saved_state_and_control_identity_checks':durable,
      'negative_controls_no_real_source_tool':controls,
      'summary':{
        'true_grounded_complete_tasks':sum(x['correct_against_original_git_source'] for x in main),
        'original_source_versions_tested':len(main),
        'original_source_tool_reads_executed':sum(x['actual_read_file_tool_called'] for x in main),
        'exact_original_source_byte_matches':sum(x['verified_exact_byte_source_contents'] for x in main),
        'original_source_claim_errors_in_grounded_arm':sum(not x['correct_against_original_git_source'] for x in main),
        'grounded_fake_model_requests':sum(x['scripted_model_requests'] for x in main),
        'grounded_prompt_whitespace_words_proxy':sum(x['scripted_model_prompt_whitespace_words_proxy'] for x in main),
        'grounded_2_turns_wall_seconds':round(sum(x['roundtrip_wall_seconds'] for x in main),6),
        'no_tool_direct_false_asserts':sum(x['factual_assertion_wrong_against_original_source'] for x in controls if x['control']=='unguarded_direct_yes'),
        'no_tool_direct_source_verified_answers':sum(x['source_verified_task_complete'] for x in controls if x['control']=='unguarded_direct_yes'),
        'no_tool_cautious_safe_unresolved_answers':sum(x['safe_unresolved_abstention'] for x in controls if x['control']=='cautious_no_tool'),
        'no_tool_control_fake_model_requests':sum(x['model_requests_to_scripted_client'] for x in controls),
        'full_experiment_cpu_harness_wall_seconds':round(time.perf_counter()-began,6),
      },
      'limitations':[
        'The saved authoritative source documents are genuine unchanged Git versions, but all model responses are scripted test fixtures, not actually generated by Qwen.',
        'The source presence yes/no is an exact deterministic ground truth; it cannot establish semantic reading quality of a real model.',
        'The direct unsupported answer is a deliberately wrong negative control, not representative model false-positive rate.',
        'Source reads are temporary sandboxed local read_file tool calls, not autonomous original user project changes.',
        'No real model tokens, two-stage specialist/reviewer corrections, GPU queue/VRAM inference attribution or syntax-auxiliary training is measured.',
      ]}
    dest=HERE/'evidence/original-doc-roundtrip'
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/'original_git_doc_source_to_final_state_v1.json'
    if path.exists():raise FileExistsError('Frozen roundtrip observation exists; refuse to replay/overwrite')
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result['summary'],indent=2))
    print('STATE',json.dumps(durable,indent=2))
    print('CONTROLS',[(x['version'],x['control'],x['final_user_answer'],x['model_requests_to_scripted_client']) for x in controls])
    print('SAVED',path)
    return result

if __name__=='__main__':run()
