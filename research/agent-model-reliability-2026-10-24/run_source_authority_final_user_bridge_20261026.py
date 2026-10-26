#!/usr/bin/env python3
"""Evaluate *actual* final SWAAG user answers on 6 original human-gold sources.

CPU classifier decisions are precomputed from the pinned prior research run;
SWAAG uses its genuine communication service and temporary DB, with a fake
model returning only those already observed decisions. No private source text
is committed; no real model, production worker or external effect invoked.
"""
from __future__ import annotations
import hashlib
import gzip
import io
import json
import tarfile
import os
from pathlib import Path
import random
import tempfile
import time

from run_full_foreground_receipt_study_20261026 import isolated_config
from swaag.runtime import AgentRuntime
from swaag.communication import CommunicationService
from tests.test_agent_action_loop import FakeModelClient
from tests.test_fast_foreground_reply import fast

HERE=Path(__file__).resolve().parent
SOURCE_RESULTS=HERE/'evidence/source-authority-policy/final_verdict_and_cost_ledger_v1.json'
SOURCE_RESULTS_SHA='9abb684833e3cd1ea76998c8ae049b7843c07c7025b3c2dd8bf383f9cde279a0'
SCI_ARCHIVE=Path('/data/tmp/agent-model-independent-qrels-20261025/scifact_original.tar.gz')
SCI_ARCHIVE_SHA='11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be'
BM25_ARCHIVE=HERE/'evidence/scifact-offline/source_evidence_case_predictions_v1.json.gz'
BM25_SHA='0e0f797b9cafdd0abb7a15ff06b898b9ad8ef4d15bdf4f22ec609831872a076e'
VERSION='swaag_sci_authority_final_user_bridge_v1_20261026'
SELECTED=((56,'4709641','SUPPORT'),(57,'4709641','CONTRADICT'),
          (1029,'13923140','CONTRADICT'),(1379,'16322674','SUPPORT'),
          (903,'10648422','CONTRADICT'),(179,'17450673','SUPPORT'))
POLICIES=('one_word_top3','two_consensus_top3')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source():
    if sha(SOURCE_RESULTS)!=SOURCE_RESULTS_SHA:raise ValueError('Policy outcomes changed after earlier audit')
    if sha(SCI_ARCHIVE)!=SCI_ARCHIVE_SHA or sha(BM25_ARCHIVE)!=BM25_SHA:
        raise ValueError('Original SciFact source or prior frozen BM25 prediction digest mismatch')
    with tarfile.open(SCI_ARCHIVE,'r:gz') as tar:
        def rows(name):
            with io.TextIOWrapper(tar.extractfile('data/'+name),encoding='utf8') as f:
                return [json.loads(line) for line in f]
        corpus=rows('corpus.jsonl');dev=rows('claims_dev.jsonl')
    assert len(corpus)==5183 and len(dev)==300
    docs={str(d['doc_id']):d for d in corpus}
    selected_ids={(c,d) for c,d,_ in SELECTED}
    lookup={}
    for claim in dev:
        for docid,annotations in (claim.get('evidence') or {}).items():
            key=(int(claim['id']),str(docid))
            if key not in selected_ids:continue
            assert annotations and len({a['label'] for a in annotations})==1
            rationale=sorted({tuple(sorted(set(int(i) for i in a['sentences']))) for a in annotations},key=lambda x:(len(x),x))
            lookup[key]={'claim_id':int(claim['id']),'doc_id':str(docid),'claim':claim['claim'],
                         'label':annotations[0]['label'],'gold_sets':rationale,
                         'original_abstract':docs[str(docid)]['abstract']}
    with gzip.open(BM25_ARCHIVE,'rt',encoding='utf8') as f:
        prior_ranking=json.load(f)
    assert prior_ranking['source_sha256']==SCI_ARCHIVE_SHA
    rankings={(int(r['claim_id']),str(r['doc_id'])):r for r in prior_ranking['cases']}
    assert len(lookup)==len(SELECTED)
    prior=json.loads(SOURCE_RESULTS.read_text())
    scores={(r['case_id'],r['doc_id'],r['policy']):r for r in prior}
    assert len(scores)==209*10
    out=[]
    for cid,docid,gold in SELECTED:
        ref=lookup[(cid,docid)]
        if ref['label']!=gold:raise ValueError('Independent expert label changed')
        saved=rankings[(cid,docid)]
        if saved['label']!=gold or {tuple(x) for x in saved['gold_sets']}!=set(ref['gold_sets']):
            raise ValueError('Original SciFact expert rationale version mismatch')
        original=ref['original_abstract']
        ids=tuple(sorted(saved['methods']['bm25']['top5_sentence_ids'][:3]))
        if not ids or len(set(ids))!=len(ids) or max(ids)>=len(original):
            raise ValueError('BM25 original source sentence index invalid')
        src={'text':' '.join(original[i] for i in ids),'ids':ids,
             'complete_rationale_retained':any(set(r)<=set(ids) for r in ref['gold_sets'])}
        for policy in POLICIES:
            recorded=scores[(cid,docid,policy)]
            assert recorded['original_gold_relation']==gold
            if recorded['final_decision']=='needs_independent_verification':
                raise ValueError('Selected bridge specimen must have decisive final output for exact-text replay')
            assert recorded['final_user_visible_reply'] in (
              'The cited paper supports the specific claim.',
              'The cited paper contradicts the specific claim.',)
            out.append((ref,src,recorded,policy))
    return out


def run_once(ref,src,decision,policy):
    started=time.perf_counter_ns()
    original_user=(f"Does the following *specific cited original paper* support or contradict the claim?\n"
      f"Original scientific claim: {ref['claim']}\nOriginal paper ID: {ref['doc_id']}\n"
      f"Original evidence excerpt (exact BM25-selected source sentences):\n{src['text']}")
    with tempfile.TemporaryDirectory(prefix='swaag-human-gold-fg-') as scratch:
        config=isolated_config(Path(scratch))
        model=FakeModelClient([fast('answer',decision['final_user_visible_reply'])])
        runtime=AgentRuntime(config,model_client=model)
        service=CommunicationService(runtime,orchestrator_runtime=runtime)
        try:
            control_id=f"research-expert-source-{ref['claim_id']}-{ref['doc_id']}-{policy}"
            stream=list(service.iter_orchestrator_message(original_user,control_id=control_id))
            assert [i['event'] for i in stream]==['accepted','completed']
            final=stream[-1]['result']['answer']
            assert final==decision['final_user_visible_reply']
            state=stream[0]['session_id']
            events=runtime.history.read_history(state)
            terminal=[i for i in events if i.event_type=='turn_finished']
            control=[i for i in events if i.event_type=='control_message_processed']
            tool=[i for i in events if i.event_type=='tool_called']
            assert len(terminal)==len(control)==1 and not tool
            assert terminal[0].payload['assistant_text']==final
            assert len(model.requests)==1
            assert model.requests[0]['contract']=='fast_foreground_response'
            assert ref['claim'] in model.requests[0]['prompt']
            assert src['text'] in model.requests[0]['prompt']
            assert ref['doc_id'] in model.requests[0]['prompt']
            before_requests=len(model.requests)
            replay=list(service.iter_orchestrator_message(original_user,control_id=control_id))
            assert replay[-1]['result']['answer']==final and len(model.requests)==before_requests
            assert len([i for i in runtime.history.read_history(state) if i.event_type=='tool_called'])==0
            return {'source_claim_id':ref['claim_id'],'source_paper_id':ref['doc_id'],
                'independent_expert_gold':ref['label'],'policy':policy,
                'source_rationale_complete_in_initial_prompt':src['complete_rationale_retained'],
                'source_excerpt_sha256':hashlib.sha256(src['text'].encode()).hexdigest(),
                'original_user_message_sha256':hashlib.sha256(original_user.encode()).hexdigest(),
                'original_source_text_not_exported':True,
                'final_user_visible_reply':final,'final_source_scoped_verdict':decision['final_decision'],
                'correct_against_external_expert_source':decision['final_decision']==ref['label'],
                'fake_model_contracts':[r['contract'] for r in model.requests],
                'fake_model_requests':len(model.requests),
                'actual_model_qwen_inference_requests':0,
                'full_source_to_final_sqlite_controller_wall_milliseconds':round((time.perf_counter_ns()-started)/1e6,4),
                'prompt_whitespace_words_not_actual_model_tokens':len(model.requests[0]['prompt'].split()),
                'tool_effects_count':len(tool),
                'turn_finished_receipt_seq':terminal[0].sequence,
                'turn_finished_receipt_hash':terminal[0].hash,
                'control_processed_receipt_seq':control[0].sequence,
                'control_processed_receipt_hash':control[0].hash,
                'reconnect_idempotent':True,
            }
        finally:service.close()


def main():
    cases=source()
    out=HERE/'evidence/source-authority-policy'
    journal=out/'swaag_final_user_bridge_receipts_v1.jsonl'
    if journal.exists():raise FileExistsError('Prior final SWAAG bridge test already exists; do not replay')
    rows=[];start=time.perf_counter()
    with journal.open('x',encoding='utf8') as handle:
        def record(e):
            handle.write(json.dumps(e,sort_keys=True)+'\n');handle.flush();os.fsync(handle.fileno())
        record({'event':'frozen_begin','policy_result_sha256':SOURCE_RESULTS_SHA,'version':VERSION,
                'selected_source_pairs':SELECTED,'model_inference':'scripted original CPU-model verdicts'})
        for ref,src,decision,policy in cases:
            record({'event':'isolated_case_begin','id':ref['claim_id'],'paper':ref['doc_id'],'policy':policy})
            print('SRC',ref['claim_id'],ref['doc_id'],policy,flush=True)
            try:r=run_once(ref,src,decision,policy)
            except Exception as exc:
                record({'event':'isolated_case_failed','case_id':ref['claim_id'],'paper':ref['doc_id'],
                        'policy':policy,'error_class':type(exc).__name__,
                        'error_summary':str(exc)[:150],
                        'no_non_isolated_side_effects':True})
                raise
            rows.append(r);record({'event':'isolated_case_completed','result':r})
        record({'event':'all_complete','pairs':len(SELECTED),'final_trajectories':len(rows)})
    result={'version':VERSION,'source_original_label_status':'genuine independently annotated SciFact claim/paper relationships',
      'claim_paper_selected_after_model_outcomes':'intentionally diagnostic, not prevalence',
      'inference_backend':'SWAAG real isolated service with scripted FakeModelClient; no real Qwen',
      'source_score_file_sha256':SOURCE_RESULTS_SHA,
      'paired_source_cases':len(SELECTED),
      'final_user_trajectories':len(rows),
      'true_negative_CONTRADICT_source_cases':sum(x[2]=='CONTRADICT' for x in SELECTED),
      'true_positive_SUPPORT_source_cases':sum(x[2]=='SUPPORT' for x in SELECTED),
      'policy_results':{p:{'n':sum(x['policy']==p for x in rows),
          'wrong_terminal_final_user_replies':sum(x['policy']==p and not x['correct_against_external_expert_source'] for x in rows),
          'right_terminal_final_user_replies':sum(x['policy']==p and x['correct_against_external_expert_source'] for x in rows),
          'model_calls_to_mock':sum(x['policy']==p for x in rows),
          'actual_qwen_calls':0,
          'cpu_swaag_harness_end_to_end_wall_milliseconds':round(sum(x['full_source_to_final_sqlite_controller_wall_milliseconds'] for x in rows if x['policy']==p),4),
          'authentic_tool_receipts':0,
          'source_complete_context_cases':sum(x['policy']==p and x['source_rationale_complete_in_initial_prompt'] for x in rows)}
        for p in POLICIES},
      'limitations':['Six cases intentionally selected from known mistakes, not representative',
                     'Real model verdicts were precomputed CPU models, and then returned by FakeModelClient; no new semantic model inference in SWAAG',
                     'Not a real Qwen deployment or an independent SWAAG worker action',
                     'The service cannot itself verify scientific entailment without independently returned evidence/tool results']}
    path=out/'swaag_final_user_bridge_summary_v1.json'
    if path.exists():raise FileExistsError('No overwrite of prior verified evidence')
    path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
