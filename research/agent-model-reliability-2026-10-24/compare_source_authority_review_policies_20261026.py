#!/usr/bin/env python3
"""Actual CPU classifier calls, final source-grounded verdicts and review costs.

Source and evaluation method are specified in
SOURCE_AUTHORITY_REVIEW_POLICY_PROTOCOL_2026-10-26.md. This is not Qwen.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import resource
import statistics
import tarfile
import time

from evaluate_scifact_hard_polarity_20261026 import (
    HERE, SOURCE, ORIGINAL_SHA256, ARCHIVED_DEV, ARCHIVED_SHA256,
    load, pairs, verify_dev_predictions, content_for, classifier,
    input_text, POSITIVE, NEGATIVE,
)

VERSION='swaag_sci_source_authority_review_policy_v1_20261026'
THRESHOLD_LOW=0.4
THRESHOLD_HIGH=0.6
ABSTAIN='needs_independent_verification'
POLICIES=(
    'zero_call_support',
    'zero_call_abstain',
    'one_word_top3',
    'one_word_full',
    'one_char_top3',
    'two_consensus_top3',
    'selective_review_uncertain',
    'selective_review_support',
    'one_word_abstain_uncertain',
    'two_consensus_full',
)
PROTOCOL=HERE/'SOURCE_AUTHORITY_REVIEW_POLICY_PROTOCOL_2026-10-26.md'
PRIOR_RESULTS=HERE/'evidence/scifact-hard-polarity/case_errors_v1.json.gz'
PRIOR_SUMMARY=HERE/'evidence/scifact-hard-polarity/summary_v1.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs_same_source():
    whole_start=time.perf_counter()
    corpus, train, dev, frozen_prior=load()
    train_pairs=pairs(train,corpus)
    dev_pairs=pairs(dev,corpus)
    if len(train_pairs)!=564 or len(dev_pairs)!=209:raise ValueError('original split drift')
    if len({p['claim_id'] for p in train_pairs})!=505 or len({p['claim_id'] for p in dev_pairs})!=188:
        raise ValueError('source claim groups drift')
    train_texts={p['claim'] for p in train_pairs}
    if any(p['claim'] in train_texts for p in dev_pairs):raise ValueError('exact training claim reappeared')
    ranking=verify_dev_predictions(dev_pairs,frozen_prior)
    scopes={k:[content_for(p,'bm25_top3' if k=='top3' else 'whole_abstract',ranking) for p in dev_pairs]
            for k in ('top3','full')}
    count=Counter()
    polarity=defaultdict(list)
    for split,cs in [('train',train),('dev',dev)]:
        for c in cs:
            for did,rs in (c.get('evidence') or {}).items():
                if rs:polarity[str(did)].append((split,int(c['id']),rs[0]['label']))
    mixed={k:v for k,v in polarity.items() if {x[2] for x in v}=={POSITIVE,NEGATIVE}}
    assert mixed and all(len(v)>=2 for v in mixed.values())
    return train_pairs,dev_pairs,scopes,mixed,time.perf_counter()-whole_start


def output_text(answer):
    return {
        POSITIVE:'The cited paper supports the specific claim.',
        NEGATIVE:'The cited paper contradicts the specific claim.',
        ABSTAIN:'I cannot verify this claim against the cited paper reliably from the available evidence.',
    }[answer]


def model_call(model_key,scope,case,context,models):
    model_name='joint_pair_'+('word' if model_key=='word' else 'char')
    vec,reg=models[model_name]
    original=input_text(case,context['text'],model_name)
    tic=time.perf_counter_ns()
    prob=float(reg.predict_proba(vec.transform([original]))[0,1])
    elapsed_ns=time.perf_counter_ns()-tic
    answer=POSITIVE if prob>=0.5 else NEGATIVE
    return {'model':model_key,'source_scope':scope,'proposal':answer,
            'raw_support_score_not_calibrated':round(prob,6),
            'input_whitespace_word_proxy':len(original.split()),
            'source_char_count':context['source_chars'],
            'source_words':context['source_words_proxy'],
            'full_rationale_was_available':context['complete_rationale_retained'],
            'wall_microseconds':round(elapsed_ns/1e3,2)}


def final_decision(policy,pair,source_cases,trained):
    calls=[]
    def once(which,scope):
        r=model_call(which,scope,pair,source_cases[scope],trained)
        calls.append(r)
        return r
    if policy=='zero_call_support':prediction=POSITIVE
    elif policy=='zero_call_abstain':prediction=ABSTAIN
    elif policy=='one_word_top3':prediction=once('word','top3')['proposal']
    elif policy=='one_word_full':prediction=once('word','full')['proposal']
    elif policy=='one_char_top3':prediction=once('char','top3')['proposal']
    elif policy=='one_word_abstain_uncertain':
        a=once('word','top3')
        p=a['raw_support_score_not_calibrated']
        prediction=ABSTAIN if THRESHOLD_LOW<=p<=THRESHOLD_HIGH else a['proposal']
    elif policy=='two_consensus_top3':
        a=once('word','top3');b=once('char','top3')
        prediction=a['proposal'] if a['proposal']==b['proposal'] else ABSTAIN
    elif policy=='two_consensus_full':
        a=once('word','full');b=once('char','full')
        prediction=a['proposal'] if a['proposal']==b['proposal'] else ABSTAIN
    elif policy=='selective_review_uncertain':
        a=once('word','top3');p=a['raw_support_score_not_calibrated']
        if THRESHOLD_LOW<=p<=THRESHOLD_HIGH:
            b=once('char','full')
            prediction=a['proposal'] if a['proposal']==b['proposal'] else ABSTAIN
        else:prediction=a['proposal']
    elif policy=='selective_review_support':
        a=once('word','top3')
        if a['proposal']==POSITIVE:
            b=once('char','full')
            prediction=POSITIVE if b['proposal']==POSITIVE else ABSTAIN
        else:prediction=NEGATIVE
    else:raise ValueError(f'policy not allowed: {policy}')
    return prediction,calls


def one_case(pair,index,source_scopes,models,policy):
    scope={k:v[index] for k,v in source_scopes.items()}
    before=time.perf_counter_ns()
    outcome,calls=final_decision(policy,pair,scope,models)
    elapsed_ns=time.perf_counter_ns()-before
    if outcome not in (POSITIVE,NEGATIVE,ABSTAIN):raise ValueError('unexpected final decision')
    gold=pair['label']
    assert gold in (POSITIVE,NEGATIVE)
    correct_decisive=outcome==gold
    wrong_decisive=outcome!=ABSTAIN and outcome!=gold
    source_full_available=any(x['full_rationale_was_available'] for x in calls)
    return {
       'case_id':pair['claim_id'],'doc_id':pair['doc_id'],'original_gold_relation':gold,
       'full_original_rationale_alternatives':[list(s) for s in pair['gold_sets']],
       'minimal_rationale_len':len(pair['oracle_indices']),
       'requires_two_or_more_gold_sentences':pair['multi_premise'],
       'multiple_original_gold_alternatives':pair['multiple_alternatives'],
       'policy':policy,'final_user_visible_reply':output_text(outcome),
       'final_decision':outcome,'is_decisive':outcome!=ABSTAIN,
       'agrees_with_human_gold':correct_decisive,
       'incorrect_decisive':wrong_decisive,
       'false_support':outcome==POSITIVE and gold==NEGATIVE,
       'false_contradiction':outcome==NEGATIVE and gold==POSITIVE,
       'complete_source_rationale_was_available_to_any_called_model':source_full_available,
       'correct_final_without_complete_source_rationale':correct_decisive and not source_full_available,
       'model_calls':len(calls),'stages':calls,
       'whole_pipeline_cpu_wall_microseconds':round(elapsed_ns/1e3,2),
       'total_stage_cpu_wall_microseconds':round(sum(x['wall_microseconds'] for x in calls),2),
       'input_whitespace_word_proxy_total':sum(x['input_whitespace_word_proxy'] for x in calls),
       'source_character_count_total':sum(x['source_char_count'] for x in calls),
       'source_word_count_total':sum(x['source_words'] for x in calls),
    }


def eval_stats(rows):
    c=Counter()
    for x in rows:
        gold=x['original_gold_relation'];out=x['final_decision']
        key=('TP' if gold==POSITIVE and out==POSITIVE else
             'TN' if gold==NEGATIVE and out==NEGATIVE else
             'FP' if gold==NEGATIVE and out==POSITIVE else
             'FN' if gold==POSITIVE and out==NEGATIVE else
             'ABSTAIN_POSITIVE' if gold==POSITIVE else 'ABSTAIN_NEGATIVE')
        c[key]+=1
    n=len(rows);n_gold_neg=c['TN']+c['FP']+c['ABSTAIN_NEGATIVE'];n_gold_pos=c['TP']+c['FN']+c['ABSTAIN_POSITIVE']
    decisive=c['TP']+c['TN']+c['FP']+c['FN']
    correct=c['TP']+c['TN']
    all_call_times=[s['wall_microseconds'] for x in rows for s in x['stages']]
    return {
        'n_original_claim_paper_pairs':n,'n_unique_original_claim_ids':len({x['case_id'] for x in rows}),
        **{k:c[k] for k in ('TP','TN','FP','FN','ABSTAIN_POSITIVE','ABSTAIN_NEGATIVE')},
        'source_expert_SUPPORT_pairs':n_gold_pos,'source_expert_CONTRADICT_pairs':n_gold_neg,
        'decisive_total':decisive,'abstained_total':n-decisive,
        'decisive_coverage':round(decisive/n,6) if n else None,
        'correct_decisive_total':correct,'incorrect_decisive_total':c['FP']+c['FN'],
        'correct_fraction_of_all_pairs_not_crediting_abstention':round(correct/n,6) if n else None,
        'error_fraction_of_all_pairs_not_crediting_abstention':round((c['FP']+c['FN'])/n,6) if n else None,
        'correct_fraction_when_decisive':round(correct/decisive,6) if decisive else None,
        'false_support_fraction_of_all_contradictions':round(c['FP']/n_gold_neg,6) if n_gold_neg else None,
        'false_contradiction_fraction_of_all_supports':round(c['FN']/n_gold_pos,6) if n_gold_pos else None,
        'correct_decisive_with_complete_rationale_present':sum(x['agrees_with_human_gold'] and x['complete_source_rationale_was_available_to_any_called_model'] for x in rows),
        'correct_decisive_without_complete_rationale_present':sum(x['correct_final_without_complete_source_rationale'] for x in rows),
        'model_inference_calls_total':sum(x['model_calls'] for x in rows),
        'model_inference_calls_by_type':dict(Counter(s['model'] for x in rows for s in x['stages'])),
        'measured_cpu_model_call_wall_milliseconds_total':round(sum(all_call_times)/1000,4),
        'measured_cpu_policy_wall_milliseconds_total':round(sum(x['whole_pipeline_cpu_wall_microseconds'] for x in rows)/1000,4),
        'median_full_policy_cpu_wall_milliseconds_per_case':round(statistics.median(x['whole_pipeline_cpu_wall_microseconds'] for x in rows)/1000,4) if rows else None,
        'source_characters_reprocessed_across_calls':sum(x['source_character_count_total'] for x in rows),
        'source_whitespace_words_reprocessed_across_calls':sum(x['source_word_count_total'] for x in rows),
        'model_input_whitespace_word_proxy_total_including_claim':sum(x['input_whitespace_word_proxy_total'] for x in rows),
    }


def reviewer_joint_error(rows):
    by=defaultdict(dict)
    for r in rows:
        if r['policy'] not in ('one_word_top3','one_char_top3','one_word_full','two_consensus_full'):continue
        by[(r['case_id'],r['doc_id'])][r['policy']]=r
    joint=Counter()
    detailed=[]
    for case,rr in sorted(by.items()):
        if 'one_word_top3' not in rr or 'one_char_top3' not in rr:raise ValueError('missing paired reviewer case')
        a=rr['one_word_top3'];b=rr['one_char_top3']; gold=a['original_gold_relation']
        wa=a['final_decision']==gold;wb=b['final_decision']==gold
        name=('both_correct' if wa and wb else 'both_wrong' if not wa and not wb else
              'word_correct_char_wrong' if wa else 'word_wrong_char_correct')
        joint[name]+=1
        detailed.append({'case_id':case[0],'doc_id':case[1],
                         'word_is_correct':wa,'char_is_correct':wb,
                         'word_proposal':a['final_decision'],'char_proposal':b['final_decision'],
                         'original_gold':gold,'reviewer_wrong_after_correct':wa and not wb,
                         'reviewer_right_after_wrong':not wa and wb,
                         'shared_wrong_error':not wa and not wb})
    assert len(detailed)==209
    return {**dict(joint),'total_matched_original_cases':209,
            'plain_char_override_recovers_word_errors':joint['word_wrong_char_correct'],
            'plain_char_override_breaks_correct_word_answers':joint['word_correct_char_wrong'],
            'at_least_one_model_wrong':joint['both_wrong']+joint['word_correct_char_wrong']+joint['word_wrong_char_correct'],
            'wrong_wrong_cases':joint['both_wrong']},detailed


def cluster_bootstrap(rows,first,second,samples=1000):
    aligned=defaultdict(dict)
    for row in rows:
        if row['policy'] in (first,second):aligned[(row['case_id'],row['doc_id'])][row['policy']]=row
    if len(aligned)!=209 or any(len(d)!=2 for d in aligned.values()):raise ValueError('unmatched original task')
    per_claim=defaultdict(list)
    for (cid,_),pair in aligned.items():per_claim[cid].append(pair)
    ids=sorted(per_claim)
    def score(group):
        vals=list(group)
        n=len(vals)
        metrics={}
        for measure,name in [(lambda r:int(r['incorrect_decisive']),'wrong_decisive'),
                             (lambda r:int(r['agrees_with_human_gold']),'correct_decisive'),
                             (lambda r:int(not r['is_decisive']),'abstain')]:
            metrics[name]=sum(measure(p[second])-measure(p[first]) for p in vals)/n
        return metrics
    point=score(aligned.values())
    rnd=random.Random(20261026)
    boots={name:[] for name in point}
    for _ in range(samples):
        draws=[p for cid in (rnd.choice(ids) for _ in ids) for p in per_claim[cid]]
        scores=score(draws)
        for name,x in scores.items():boots[name].append(x)
    return {'compare':second+' minus '+first,'distinct_claim_clusters':len(ids),
            'resamples':samples,'estimates':{name:{'difference':round(point[name],6),
              'descriptive_claim_cluster_interval_95pct':[round(sorted(v)[24],6),round(sorted(v)[975],6)]}
              for name,v in boots.items()}}


def source_authority_examples(mixed):
    return {'source_papers_with_both_expert_SUPPORT_and_CONTRADICT_relations_to_distinct_claims':len(mixed),
            'illustrative_distinct_claim_source_edges':[{'paper_id':key,'relationships':[
              {'source_split':s,'claim_id':cid,'expert_relation':gold}
              for s,cid,gold in rows[:12]]}
              for key,rows in sorted(mixed.items(),key=lambda it:(-len(it[1]),it[0]))[:8]],
            'authority_interpretation':'The source paper is not globally SUPPORT or CONTRADICT; expert polarity is scoped to the exact claim and its original rationale. Older user quotes are historical observations and cannot defeat a newer governing user correction.'}


def main():
    if sha(SOURCE)!=ORIGINAL_SHA256 or sha(ARCHIVED_DEV)!=ARCHIVED_SHA256:raise ValueError('upstream source hash changed')
    time_start=time.perf_counter()
    train,dev,scopes,mixed,source_wall=pairs_same_source()
    models={};train_cost={}
    for name in ('joint_pair_word','joint_pair_char'):
        models[name],train_cost[name]=classifier(name,train)
    with gzip.open(PRIOR_RESULTS,'rt',encoding='utf8') as f:prior=json.load(f)['cases']
    existing={(r['claim_id'],r['doc_id']):r for r in prior}
    assert len(existing)==len(dev)
    cases=[]
    # Fixed seed order prevents each arm always benefitting from first-run cache.
    for i,pair in enumerate(dev):
        order=list(POLICIES)
        random.Random('20261026'+str(pair['claim_id'])+'/'+pair['doc_id']).shuffle(order)
        for policy in order:
            row=one_case(pair,i,scopes,models,policy)
            cases.append(row)
        original=existing[(pair['claim_id'],pair['doc_id'])]
        if original['label']!=pair['label']:raise ValueError('prior gold changed')
    direct_map={}
    for row in cases:
        if row['policy'] in ('one_word_top3','one_char_top3','one_word_full'):
            direct_map[(row['case_id'],row['doc_id'],row['policy'])]=row
    for p in dev:
        previous=existing[(p['claim_id'],p['doc_id'])]['predictions']
        for policy,ctx,model in [('one_word_top3','bm25_top3','joint_pair_word'),
                                 ('one_char_top3','bm25_top3','joint_pair_char'),
                                 ('one_word_full','whole_abstract','joint_pair_word')]:
            current=direct_map[(p['claim_id'],p['doc_id'],policy)]
            if current['final_decision']!=previous[ctx][model]['label']:
                raise ValueError('rerun does not match original unchanged trained model decision')
            if abs(current['stages'][0]['raw_support_score_not_calibrated']-previous[ctx][model]['confidence_support_raw'])>0.00002:
                raise ValueError('source/model change alters probability')
    results={p:eval_stats([r for r in cases if r['policy']==p]) for p in POLICIES}
    reviewer,cases_reviewer=reviewer_joint_error(cases)
    bootstrap=cluster_bootstrap(cases,'one_word_top3','two_consensus_top3')
    all_cases=cases
    costs={'source_load_and_hash_seconds':round(source_wall,5),
           'classifier_training_wall_seconds':{key:cost['training_seconds'] for key,cost in train_cost.items()},
           'classification_total_wall_seconds_including_preprocessing':round(time.perf_counter()-time_start,5),
           'peak_resident_memory_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'actual_qwen_llm_calls':0,'actual_gpu_vram_attributable_to_calls':None,
           'all_cpu_classifier_calls':sum(x['model_calls'] for x in cases),
           'no_external_action_effects':True}
    summary={'version':VERSION,'source_reference':{
               'original_sci_fact_archive_sha256':ORIGINAL_SHA256,
               'original_BM25_evidence_prediction_sha256':ARCHIVED_SHA256,
               'original_classifier_code_sha256':sha(HERE/'evaluate_scifact_hard_polarity_20261026.py'),
               'protocol_sha256':sha(PROTOCOL),
               'previous_round_summary_sha256':sha(PRIOR_SUMMARY),
               'all_original_dev_pairs_reconciled_with_last_round':True,
               'training_independently_source_labeled_pairs':len(train),
               'original_eval_independently_labeled_pairs':len(dev),
               'distinct_eval_source_claims':len({x['claim_id'] for x in dev}),
           },
           'policy_definitions_fixed_before_running':'SOURCE_AUTHORITY_REVIEW_POLICY_PROTOCOL_2026-10-26.md',
           'independent_label_scope':'Expert support or contradiction for original SciFact claim/paper pair; not true-negative unrelated documents or SWAAG worker effects.',
           'results':results,'same_case_primary_reviewer_errors':reviewer,
           'uncertainty':bootstrap,'source_authority_relationships':source_authority_examples(mixed),
           'resource_costs':costs,
           'limitations':[
              'This is actual measured CPU lexical-model inference over exact source claim/paper cases, not Qwen model review or installed SWAAG production trajectories.',
              'The dev split has previous research usage and 124/209 paper IDs also exist in training; independent human gold is valid but not a novel untouched split.',
              'Original scientific labels describe source-specific entailment relations, not truth of the scientific claim globally.',
              'No clinical or scientific answer should be deployed from these low-performing exploratory baselines; an abstention is unresolved, not a correct answer.',
              'Model raw output 0.4..0.6 thresholds are heuristics, not calibrated uncertainty or a verified source-based grader.',
              'Pretrained weights are learned using original source expert rationale oracle; real retrieval and paper choice remain separate unvalidated costs.',
              'Source-whitespace words are not actual Qwen tokens and CPU work does not estimate GPU contention.',
           ]}
    dest=HERE/'evidence/source-authority-policy'
    dest.mkdir(exist_ok=True,parents=True)
    for path,records in [('policy_summary_v1.json',summary),('final_verdict_and_cost_ledger_v1.json',cases),
                         ('matched_reviewer_case_errors_v1.json',cases_reviewer)]:
        target=dest/path
        if target.exists():raise FileExistsError('refuse overwrite of earlier versioned experiment '+str(target))
        target.write_text(json.dumps(records,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'source_authority_mixed_papers':len(mixed),
        'costs':costs,'policies':{k:{x:v for x,v in val.items() if x in (
          'decisive_coverage','FP','FN','TP','TN','ABSTAIN_NEGATIVE','ABSTAIN_POSITIVE',
          'correct_decisive_total','model_inference_calls_total','measured_cpu_model_call_wall_milliseconds_total','source_whitespace_words_reprocessed_across_calls')}
        for k,val in results.items()},'paired_reviewer':reviewer,'bootstrap':bootstrap},indent=2))
    print('RESULTS',dest)

if __name__=='__main__':main()
