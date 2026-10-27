#!/usr/bin/env python3
"""Pure-stdlib independent re-grader for two immutable SWAAG research rounds.

Never imports SWAAG, never reruns source tools/model calls, never touches any
service. Re-derives source ground truth from exact original Git bytes, final
prose from archived output, tool receipts from recorded event metadata, and
all costs directly from immutable completed trial rows rather than trusting
summary fields. The source-specific deterministic gate is a POSTHOC diagnostic
for exact Markdown headings, not a deployable general semantic verifier.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EVIDENCE=HERE/'evidence/independent-last-mile-replication'
SOURCE_RECORD=EVIDENCE/'source_verdicts_and_cost_v1.json'
JOURNAL=EVIDENCE/'immutable_trial_stream_v1.jsonl'
ORIGINAL_ROUND9=HERE/'evidence/original-doc-roundtrip/original_git_doc_source_to_final_state_v1.json'
SOURCE_SHA='e3fb78cecb4c1afd457b6553531cb6d6ef6cd05e97f6f6ebc4ac68ee996b7307'
JOURNAL_SHA='6ed306bafa3a8e6a15165eb1bc387ace4b0252dcc22f3c8833a087c70c6bb479'
ROUND9_SHA='c4056dfd950c95a20ad2dda579809deded6d608d230459b73fd3c41846920bb2'
VERSION='source_receipt_independent_regrade_v1_20261027'
YES='Yes. The active original source contains the exact requested heading.'
NO='No. The active original source does not contain the exact requested heading.'
ABSTAIN='I have not read the file and cannot verify whether that exact heading is present.'


def hash_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def git_bytes(commit,filename):
    return subprocess.run(['git','show',commit+':'+filename],cwd=ROOT,
                          capture_output=True,check=True).stdout


def independent_gold_and_receipt_audit():
    start=time.perf_counter()
    for p,h in [(SOURCE_RECORD,SOURCE_SHA),(JOURNAL,JOURNAL_SHA),(ORIGINAL_ROUND9,ROUND9_SHA)]:
        if hash_file(p)!=h:raise ValueError('Recorded experiment changed after completion: '+str(p))
    reported=json.loads(SOURCE_RECORD.read_text())
    raw=[json.loads(s) for s in JOURNAL.read_text().splitlines()]
    stages=Counter(x['event'] for x in raw)
    if stages!={'immutable_protocol_begin':1,'isolated_case_begin':24,
                 'isolated_case_completed':24,'all_cases_attempted':1}:
        raise ValueError('incomplete or duplicate original completed trial journal')
    if raw[-1]['completed']!=24 or raw[-1]['technical_failed']!=0:
        raise ValueError('source experiment completion changed')
    if raw[0]['case_manifest_sha256']!=reported['immutable_cases_sha256']:
        raise ValueError('input manifest changed')
    docs=reported['original_source_documents']
    original={}
    for doc_id,meta in docs.items():
        blob=git_bytes(reported['original_git_commit'],meta['path'])
        if hashlib.sha256(blob).hexdigest()!=meta['sha256']:
            raise ValueError('source-specific gold version is not the original Git object')
        original[doc_id]=blob
    pairs={}
    for case in reported['test_input_cases']:
        cid=case['id']
        if cid in pairs:raise ValueError('duplicate case definition')
        source=original[case['doc']].decode('utf-8').splitlines()
        decoy=original[case['decoy_doc_id']].decode('utf-8').splitlines()
        truth=case['heading'] in source
        if truth != case['independent_gold'] or (case['heading'] in decoy)==truth:
            raise ValueError('source gold changed or non-contradictory decoy: '+cid)
        pairs[cid]={'original_source_is_positive':truth,'source_doc_id':case['doc'],
                    'decoy_doc_id':case['decoy_doc_id'],'heading':case['heading']}
    if len(pairs)!=6 or sum(x['original_source_is_positive'] for x in pairs.values())!=3:
        raise ValueError('original source balanced positive/negative cases changed')
    attempts={}
    completed={}
    for row in raw:
        if row['event']=='isolated_case_begin':
            key=(row['case_id'],row['arm'])
            if key in attempts:raise ValueError('duplicated attempted case')
            attempts[key]=True
        elif row['event']=='isolated_case_completed':
            result=row['result'];key=(result['case_id'],result['arm'])
            if key in completed:raise ValueError('duplicated completed case')
            completed[key]=result
    if attempts.keys()!=completed.keys() or len(completed)!=24:
        raise ValueError('partial or unaccounted case status')
    arms={'read_accurate','read_inverted','direct_yes_no_tool','cautious_no_tool'}
    if {arm for _,arm in completed}!=arms:raise ValueError('four case arms are missing')
    recalculated=defaultdict(Counter)
    individual=[]
    gate=Counter()
    for case_id in sorted(pairs):
        ref=pairs[case_id];truth=ref['original_source_is_positive']
        for arm in sorted(arms):
            r=completed[case_id,arm]
            if r['independent_git_gold_heading_present']!=truth:
                raise ValueError('stored original human/grammatical source claim not equal exact Git truth')
            original_src=original[ref['source_doc_id']]
            original_decoy=original[ref['decoy_doc_id']]
            if r['original_source_sha256']!=hashlib.sha256(original_src).hexdigest():
                raise ValueError('source hash mismatch')
            if r['original_decoy_sha256']!=hashlib.sha256(original_decoy).hexdigest():
                raise ValueError('historical same-named source decoy hash mismatch')
            if r['original_source_git_path']!=docs[ref['source_doc_id']]['path']:
                raise ValueError('original repo source filename mismatch')
            if not r['active_source_and_historical_decoy_same_basename']:
                raise ValueError('decoy design changed')
            if not r['reopened_session_and_control_replay_without_new_effects']:
                raise ValueError('replay persisted a duplicate call/effect')
            if len(r['durable_control_event_hash'])!=64 or len(r['durable_final_turn_hash'])!=64:
                raise ValueError('durable receipt metadata incomplete')
            if r['durable_control_event_sequence']>=r['durable_final_turn_sequence']:
                raise ValueError('control persisted after final answer')
            if len(r['persisted_final_history_chain_hash'])!=64 or r['real_qwen_calls_or_actual_gpu_inference']!=0:
                raise ValueError('source history chain or fake-model assertion mismatch')
            assert r['model_contracts'][0]=='fast_foreground_response'
            final=r['final_user_visible_answer']
            if final not in (YES,NO,ABSTAIN):raise ValueError('unknown final verdict')
            answered=final!=ABSTAIN
            yes=final==YES
            correctness=answered and yes==truth
            wrong=answered and yes!=truth
            label=('TP' if answered and yes and truth else
                   'TN' if answered and not yes and not truth else
                   'FP' if answered and yes and not truth else
                   'FN' if answered and not yes and truth else
                   'ABSTAIN_POS' if truth else 'ABSTAIN_NEG')
            tool_calls=r['actual_read_file_tool_calls'];tool_results=r['actual_read_file_tool_results']
            read_events=r['actual_filesystem_source_reads']
            if (tool_calls,tool_results,read_events) not in ((0,0,0),(1,1,1)):
                raise ValueError('source/receipt count mismatch')
            grounded=answered and tool_results==1
            independently_verified=grounded and correctness
            if label in ('FP','FN') and independently_verified:
                raise ValueError('incorrect final was somehow marked source verified')
            if r['final_correct_against_exact_git_heading']!=correctness or r['final_wrong_against_exact_git_heading']!=wrong:
                raise ValueError('archived grader verdict disagrees with independent original source')
            if r['source_grounded_and_final_correct']!=independently_verified:
                raise ValueError('source read plus right final declared wrongly')
            if r['source_read_success_but_final_wrong']!=(grounded and wrong):
                raise ValueError('original source read followed by wrong final mislabeled')
            if r['unverified_coincidental_right_answer']!=(answered and correctness and not grounded):
                raise ValueError('coincidental right output needs unverified status')
            if r['cautious_abstention_unresolved']!=(not answered):
                raise ValueError('abstention incorrectly marked complete')
            original_counts=reported['cost_and_final_outcomes_by_arm'][arm]
            recalculated[arm][label]+=1
            recalculated[arm]['cases']+=1
            recalculated[arm]['real_source_reads']+=tool_results
            recalculated[arm]['model_requests']+=r['number_scripted_model_requests']
            recalculated[arm]['input_whitespace_words']+=r['input_prompt_whitespace_word_proxy']
            recalculated[arm]['microseconds']+=round(r['full_isolated_harness_wall_seconds']*1e6)
            recalculated[arm]['source_verified_completed']+=int(independently_verified)
            recalculated[arm]['source_revealed_but_false']+=int(grounded and wrong)
            recalculated[arm]['coincidentally_correct_no_tool']+=int(correctness and not grounded)
            recalculated[arm]['safe_unresolved']+=int(not answered)
            # Posthoc EXACT-SCHEMA verifier. It needs *both* a genuine source
            # read and correct statement under the original fixed heading-
            # presence predicate. It is not general natural-language proof.
            if independently_verified:gate['accepted_known_true_with_receipt']+=1
            else:gate['rejected_or_abstained_without_verified_completion']+=1
            if independently_verified and wrong:gate['verified_false_accept']+=1
            if grounded and correctness and not independently_verified:gate['verified_false_reject']+=1
            individual.append({'case_id':case_id,'policy':arm,
              'exact_git_source_truth':truth,'final_source_predicate_outcome':yes if answered else None,
              'final_grade_rederived':label,
              'real_read_file_and_receipt_observed':tool_results==1,
              'source_and_final_correct_and_verified':independently_verified,
              'no_valid_source_support_for_final':not independently_verified,
              'followup_model_calls_with_no_real_qwen_inference':r['number_scripted_model_requests']})
    for arm,score in recalculated.items():
        original_score=reported['cost_and_final_outcomes_by_arm'][arm]
        comparisons={'cases':'verified_completed_cases','real_source_reads':'true_read_file_tool_result_count',
             'model_requests':'scripted_model_requests_total','input_whitespace_words':'prompt_whitespace_word_proxy_total',
             'source_verified_completed':'source_grounded_final_correct_count',
             'source_revealed_but_false':'source_read_success_but_wrong_final_count',
             'coincidentally_correct_no_tool':'unverified_coincidental_correct_count'}
        for k,v in comparisons.items():
            if score[k]!=original_score[v]:
                raise ValueError('independently computed '+k+' disagrees with stored '+v)
        for k in ('TP','TN','FP','FN','ABSTAIN_POS','ABSTAIN_NEG'):
            if score[k]!=original_score[k]:raise ValueError('confusion matrix drift: '+arm+' '+k)
        if abs(score['microseconds']/1e6-original_score['total_harness_wall_seconds'])>0.0001:
            raise ValueError('full source-to-final cost proxy inconsistent')
    if gate['accepted_known_true_with_receipt']!=6 or gate['rejected_or_abstained_without_verified_completion']!=18:
        raise ValueError('posthoc evidence+exact predicate gate missed known source outcomes')
    prior=json.loads(ORIGINAL_ROUND9.read_text())
    older={x['version']:x for x in prior['source_read_to_persisted_final_and_replay']}
    if set(older)!={'old','new'}:raise ValueError('historical source-revision manifest changed')
    for version,row in older.items():
        ref=prior['git_snapshot_source_sha256'][version]
        content=git_bytes(ref['commit'],prior['original_repository_source'])
        if hashlib.sha256(content).hexdigest()!=row['original_source_sha256']:
            raise ValueError('previous source data record is not original committed Git content')
        if row['correct_against_original_git_source'] is not True or row['actual_read_file_tool_result']!=1:
            raise ValueError('previous source experiment positive result incomplete')
        if ('October 26 follow-up: canonical JSONL append visibility under concurrent readers' in content.decode()) != ref['header_presence_is_ground_truth']:
            raise ValueError('historical immutable source gold changed')
    prior_controls=prior['negative_controls_no_real_source_tool']
    if len(prior_controls)!=4 or sum(x['factual_assertion_wrong_against_original_source'] for x in prior_controls)!=1:
        raise ValueError('previous historical no-source false positive changed')
    return {'version':VERSION,
       'original_prior_round9_artifact_sha256':ROUND9_SHA,
       'round10_immutable_ledger_sha256':JOURNAL_SHA,
       'round10_original_result_sha256':SOURCE_SHA,
       'source_labels_independently_derived_from_original_git_bytes':True,
       'labels_are_exact_heading_predicates_not_blinded_human_semantic_relevance':True,
       'previous_round9_original_immutable_versioned_source_cases_regraded':len(older),
       'previous_round9_negative_control_cases_checked':len(prior_controls),
       'total_frozen_original_git_source_cases':len(pairs),
       'total_final_answer_policies_regraded':len(individual),
       'original_case_arms':{k:dict(v) for k,v in recalculated.items()},
       'posthoc_structural_verifier_diagnostic':{
          'accepted_correct_receipted_conclusions':gate['accepted_known_true_with_receipt'],
          'withheld_wrong_or_unsupported_or_abstained':gate['rejected_or_abstained_without_verified_completion'],
          'false_accepts_relative_to_exact_heading_truth':gate['verified_false_accept'],
          'false_rejects_relative_to_exact_heading_truth':gate['verified_false_reject'],
          'is_source_specific_and_was_developed_after_observing_results':True,
          'no_extra_llm_calls_and_cpu_only':True,
          'not_independent_model_reviewer_and_not_deployable_as_general_semantic_truth_verifier':True,
       },
       'independently_rechecked_cases':individual,
       'pure_stdlib_independent_validation_cpu_wall_seconds':round(time.perf_counter()-start,5),
       'notes':[
          'The original saved experiment journal retains source event and terminal hashes but not the full temporary SQLite source-event payloads; verifying a full cryptographic event hash chain independently from the archive alone is impossible. That chain was checked inside the test-isolated SWAAG runtime.',
          'This independent regrader never imported the original experiment implementation or used its original expected labels to decide truth; source truth comes from Git show and exact heading-line tests.',
          'All model outputs are scripted, not local Qwen; one model call count is not a measured Qwen/GPU cost.',
          'Six deliberately selected exact-string cases with real source docs do not establish population error rates or human role-relevance accuracy.',
       ]}


def main():
    result=independent_gold_and_receipt_audit()
    target=EVIDENCE/'independent_source_and_cost_regrade_v1.json'
    if target.exists():raise FileExistsError('Do not overwrite independent evidence audit')
    target.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    print('AUDIT:',json.dumps({
        'regraded_cases':result['total_final_answer_policies_regraded'],
        'arms':result['original_case_arms'],
        'posthoc_gate':result['posthoc_structural_verifier_diagnostic'],
        'limitations':result['notes']},indent=2))
    print('SAVED',target)

if __name__=='__main__':main()
