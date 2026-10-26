#!/usr/bin/env python3
"""CPU-only original-source claim polarity sanity check, not SWAAG Qwen review.

Method contract: HARD_SOURCE_POLARITY_PROTOCOL_2026-10-26.md.
No new Qwen calls, no production writes, no private source text in outputs.
"""
from __future__ import annotations
import collections
import gzip
import hashlib
import io
import json
import math
import pathlib
import random
import resource
import statistics
import tarfile
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE=pathlib.Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
SOURCE=pathlib.Path('/data/tmp/agent-model-independent-qrels-20261025/scifact_original.tar.gz')
ORIGINAL_SHA256='11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be'
ARCHIVED_DEV=HERE/'evidence/scifact-offline/source_evidence_case_predictions_v1.json.gz'
ARCHIVED_SHA256='0e0f797b9cafdd0abb7a15ff06b898b9ad8ef4d15bdf4f22ec609831872a076e'
VERSION='swaag-independent-scifact-polarity-hard-source-v1-20261026'
POSITIVE='SUPPORT';NEGATIVE='CONTRADICT';CONTEXTS=('oracle_min_rationale','bm25_top3','bm25_top5','whole_abstract')
MODELS=('support_prior','claim_only_word','evidence_only_word','joint_pair_word','joint_pair_char')


def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    if sha256(SOURCE)!=ORIGINAL_SHA256:raise ValueError('Original corpus changed; refuse newer source silently')
    if sha256(ARCHIVED_DEV)!=ARCHIVED_SHA256:raise ValueError('Archived BM25 predictions changed')
    with tarfile.open(SOURCE,'r:gz') as tar:
        def read(name):
            with io.TextIOWrapper(tar.extractfile('data/'+name),encoding='utf8') as f:
                return [json.loads(l) for l in f]
        corpus=read('corpus.jsonl'); train=read('claims_train.jsonl'); dev=read('claims_dev.jsonl')
    corpus.sort(key=lambda x:int(x['doc_id']))
    assert (len(corpus),len(train),len(dev))==(5183,809,300)
    claims_to_exclude={c['id'] for c in train if c['claim'] in {d['claim'] for d in dev}}
    if claims_to_exclude!={871,1291}:raise ValueError('Changed train/dev claim overlap')
    train=[c for c in train if c['id'] not in claims_to_exclude]
    with gzip.open(ARCHIVED_DEV,'rt',encoding='utf8') as f:prior=json.load(f)
    if prior['source_sha256']!=ORIGINAL_SHA256:raise ValueError('Source version mismatch in stored ranking')
    return corpus,train,dev,prior


def pairs(claims,corpus):
    docs={str(d['doc_id']):d for d in corpus};out=[]
    for c in claims:
        for docid,ref in (c.get('evidence') or {}).items():
            if not ref:continue
            d=docs.get(str(docid))
            if d is None:raise ValueError('Missing cited paper')
            labels={r['label'] for r in ref}
            if len(labels)!=1 or not labels<={POSITIVE,NEGATIVE}:
                raise ValueError('Inconsistent original polarity labels')
            gold=sorted({tuple(sorted({int(i) for i in r['sentences']})) for r in ref},key=lambda x:(len(x),x))
            if not gold or any(not g or min(g)<0 or max(g)>=len(d['abstract']) for g in gold):
                raise ValueError('Invalid original expert rationale')
            out.append({'claim_id':int(c['id']),'doc_id':str(docid),'claim':c['claim'],
                        'label':next(iter(labels)),'sentence_list':d['abstract'],'gold_sets':gold,
                        'multi_premise':all(len(s)>=2 for s in gold),'multiple_alternatives':len(gold)>1,
                        'oracle_indices':gold[0]})
    return sorted(out,key=lambda r:(r['claim_id'],int(r['doc_id'])))


def verify_dev_predictions(dev_pairs,prior):
    index={(r['claim_id'],str(r['doc_id'])):r for r in prior['cases']}
    if len(index)!=len(prior['cases']) or len(index)!=len(dev_pairs):
        raise ValueError('Current dev source pair set differs from original prediction archive')
    out={}
    for p in dev_pairs:
        k=(p['claim_id'],p['doc_id']);r=index.get(k)
        if r is None or r['label']!=p['label'] or {tuple(x) for x in r['gold_sets']}!=set(p['gold_sets']):
            raise ValueError('Original independent annotation drift '+str(k))
        ranking=r['methods']['bm25']['top5_sentence_ids']
        if (len(ranking)!=min(5,len(p['sentence_list'])) or len(set(ranking))!=len(ranking) or
                min(ranking)<0 or max(ranking)>=len(p['sentence_list'])):
            raise ValueError('Invalid recorded BM25 sentence indices')
        out[k]=ranking
    return out


def content_for(pair,variant,rankings):
    if variant=='oracle_min_rationale':ids=pair['oracle_indices']
    elif variant=='whole_abstract':ids=tuple(range(len(pair['sentence_list'])))
    elif variant=='bm25_top3':ids=rankings[(pair['claim_id'],pair['doc_id'])][:3]
    elif variant=='bm25_top5':ids=rankings[(pair['claim_id'],pair['doc_id'])][:5]
    else:raise ValueError('unknown evidence scope')
    # Source sequence order rather than inverse ranking order, to avoid
    # inventing a different relationship between adjacent original sentences.
    ids=tuple(sorted(set(ids)))
    text=' '.join(pair['sentence_list'][i] for i in ids)
    complete=any(set(g)<=set(ids) for g in pair['gold_sets'])
    return {'text':text,'ids':ids,'complete_rationale_retained':complete,
            'source_chars':sum(len(pair['sentence_list'][i]) for i in ids),
            'source_words_proxy':sum(len(pair['sentence_list'][i].split()) for i in ids)}


def input_text(pair,evidence,model):
    if model=='claim_only_word':return pair['claim']
    if model=='evidence_only_word':return evidence
    return 'CLAIM '+pair['claim']+' EVIDENCE '+evidence


def classifier(model,train_pairs):
    if model=='joint_pair_char':
        v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=30000,
                          lowercase=True,sublinear_tf=True,norm='l2')
    else:
        v=TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=2,max_features=50000,
                          lowercase=True,sublinear_tf=True,norm='l2')
    texts=[input_text(p,' '.join(p['sentence_list'][i] for i in p['oracle_indices']),model)
           for p in train_pairs]
    labels=[p['label'] for p in train_pairs]
    clf=LogisticRegression(C=1.0,class_weight='balanced',solver='liblinear',max_iter=1000,
                           random_state=20261026)
    start=time.perf_counter()
    matrix=v.fit_transform(texts)
    clf.fit(matrix,labels)
    sec=time.perf_counter()-start
    if list(clf.classes_)!=[NEGATIVE,POSITIVE]:raise ValueError('unexpected sklearn positive class order')
    return (v,clf),{'training_seconds':round(sec,5),'feature_count':len(v.vocabulary_),
                    'training_pairs':len(train_pairs),'train_nnz':matrix.nnz}


def predict(model,clf,pairs_,scope):
    start=time.perf_counter()
    if model=='support_prior':
        probabilities=[1.0]*len(pairs_)
    else:
        vec,lr=clf
        text=[input_text(p,sc['text'],model) for p,sc in zip(pairs_,scope)]
        probabilities=lr.predict_proba(vec.transform(text))[:,1].tolist()
    wall=time.perf_counter()-start
    labels=[POSITIVE if prob>=0.5 else NEGATIVE for prob in probabilities]
    return labels,probabilities,wall


def confusion(records,model,scope,mask=lambda r:True):
    valid=[r for r in records if mask(r)]
    c=collections.Counter()
    for p in valid:
        pred=p['predictions'][scope][model]['label']; gold=p['label']
        c['TP' if pred==POSITIVE and gold==POSITIVE else
          'TN' if pred==NEGATIVE and gold==NEGATIVE else
          'FP' if pred==POSITIVE else 'FN']+=1
    n=len(valid)
    return {'n':n,'TP':c['TP'],'TN':c['TN'],'FP':c['FP'],'FN':c['FN'],
            'accuracy':round((c['TP']+c['TN'])/n,6) if n else None,
            'support_recall':round(c['TP']/(c['TP']+c['FN']),6) if c['TP']+c['FN'] else None,
            'contradiction_recall':round(c['TN']/(c['TN']+c['FP']),6) if c['TN']+c['FP'] else None,
            'false_support_rate_among_contradictions':round(c['FP']/(c['TN']+c['FP']),6)
             if c['TN']+c['FP'] else None}


def bootstrap_effect(records,base,candidate,scope='oracle_min_rationale',samples=1000):
    groups=collections.defaultdict(list)
    for r in records:groups[r['claim_id']].append(r)
    ids=sorted(groups);rnd=random.Random(20261026)
    def correct(rr,model):return int(rr['predictions'][scope][model]['label']==rr['label'])
    est=(sum(correct(r,candidate)-correct(r,base) for r in records)/len(records))
    draws=[]
    for _ in range(samples):
        subset=[r for cid in (rnd.choice(ids) for _ in ids) for r in groups[cid]]
        draws.append(sum(correct(r,candidate)-correct(r,base) for r in subset)/len(subset))
    draws.sort()
    return {'candidate_minus_baseline_accuracy':round(est,6),
            'clustered_ci95':[round(draws[int(0.025*len(draws))],6),round(draws[int(0.975*len(draws))],6)],
            'cluster_count':len(ids),'bootstrap_resamples':samples}


def run():
    start=time.perf_counter()
    corpus,train,dev,previous=load()
    train_pairs=pairs(train,corpus)
    dev_pairs=pairs(dev,corpus)
    if len(train_pairs)!=564 or len(dev_pairs)!=209:raise ValueError('Unexpected source/evidence version')
    if len({p['claim_id'] for p in train_pairs})!=505 or len({p['claim_id'] for p in dev_pairs})!=188:
        raise ValueError('Unexpected independent claim counts')
    if {p['claim'] for p in train_pairs}&{p['claim'] for p in dev_pairs}:
        raise ValueError('Verbatim train/test claim leakage detected')
    lookup=verify_dev_predictions(dev_pairs,previous)
    sources={ctx:[content_for(p,ctx,lookup) for p in dev_pairs] for ctx in CONTEXTS}
    trained={};training={}
    for model in MODELS:
        if model=='support_prior':training[model]={'training_seconds':0.,'feature_count':0,'training_pairs':0};continue
        trained[model],training[model]=classifier(model,train_pairs)
    predictions=collections.defaultdict(dict)
    inference={}
    for ctx in CONTEXTS:
        for model in MODELS:
            labels,probabilities,sec=predict(model,trained.get(model),dev_pairs,sources[ctx])
            inference[f'{model}/{ctx}']={'total_wall_seconds':round(sec,5),
                'mean_wall_milliseconds_per_pair':round(1000*sec/len(dev_pairs),5),
                'non_llm_inference_calls':1 if model!='support_prior' else 0,
                'llm_calls':0}
            for i, (label,prob) in enumerate(zip(labels,probabilities)):
                predictions[i].setdefault(ctx,{})[model]={'label':label,'confidence_support_raw':round(float(prob),6)}
    multi_docs=collections.Counter(p['claim_id'] for p in dev_pairs)
    original_known_docs=set(p['doc_id'] for p in train_pairs)
    results=[]
    for i,p in enumerate(dev_pairs):
        results.append({'claim_id':p['claim_id'],'doc_id':p['doc_id'],'label':p['label'],
            'gold_complete_rationale_alternatives':[list(g) for g in p['gold_sets']],
            'min_gold_rationale_len':len(p['oracle_indices']),
            'multi_premise':p['multi_premise'],'multiple_alternatives':p['multiple_alternatives'],
            'same_claim_multiple_docs':multi_docs[p['claim_id']]>1,
            'seen_paper_id_in_training':p['doc_id'] in original_known_docs,
            'selected_source':{ctx:{k:v for k,v in sources[ctx][i].items() if k!='text'} for ctx in CONTEXTS},
            'predictions':predictions[i]})
    aggregate={}
    strata={
        'all':lambda r:True,
        'expert_SUPPORT':lambda r:r['label']==POSITIVE,
        'expert_CONTRADICT':lambda r:r['label']==NEGATIVE,
        'requires_multiple_gold_sentences':lambda r:r['multi_premise'],
        'has_single_sentence_alternative':lambda r:not r['multi_premise'],
        'has_multiple_expert_alternatives':lambda r:r['multiple_alternatives'],
        'has_one_expert_alternative':lambda r:not r['multiple_alternatives'],
        'same_claim_multiple_expert_papers':lambda r:r['same_claim_multiple_docs'],
        'paper_id_seen_in_train':lambda r:r['seen_paper_id_in_training'],
        'paper_id_unseen_in_train':lambda r:not r['seen_paper_id_in_training'],
    }
    for ctx in CONTEXTS:
        aggregate[ctx]={model:{key:confusion(results,model,ctx,check) for key,check in strata.items()} for model in MODELS}
    contexts={}
    for ctx in CONTEXTS:
        src=[x['selected_source'][ctx] for x in results]
        contexts[ctx]={'pairs':len(src),'complete_expert_rationale':sum(x['complete_rationale_retained'] for x in src),
            'complete_expert_rationale_support':sum(x['complete_rationale_retained'] for x,r in zip(src,results) if r['label']==POSITIVE),
            'complete_expert_rationale_contradict':sum(x['complete_rationale_retained'] for x,r in zip(src,results) if r['label']==NEGATIVE),
            'average_source_chars':round(statistics.mean(x['source_chars'] for x in src),2),
            'median_source_chars':round(statistics.median(x['source_chars'] for x in src),2),
            'average_whitespace_source_words_not_model_tokens':round(statistics.mean(x['source_words_proxy'] for x in src),2)}
    inconsistencies={}
    grouped=collections.defaultdict(list)
    for r in results:grouped[r['claim_id']].append(r)
    # This is an actual multiple-document consistency test: same original claim,
    # separately annotated source papers, same externally assigned polarity.
    for ctx in CONTEXTS:
        inconsistencies[ctx]={}
        for model in MODELS:
            mixed=[]
            for cid,rs in grouped.items():
                if len(rs)<2:continue
                if len({r['label'] for r in rs})!=1:raise ValueError('Different original labels for same claim: not scored as invariance')
                if len({r['predictions'][ctx][model]['label'] for r in rs})>1:mixed.append(cid)
            inconsistencies[ctx][model]={'claim_instances_with_multiple_papers':sum(len(rs)>1 for rs in grouped.values()),
                                        'inconsistent_claim_count':len(mixed),
                                        'inconsistent_claim_ids':mixed}
    overall={
        'original_archive_sha256':ORIGINAL_SHA256,
        'original_bm25_archive_sha256':ARCHIVED_SHA256,
        'train_claim_overlap_removed':[871,1291],
        'training_labeled_pairs':len(train_pairs),
        'training_unique_original_claims':len({p['claim_id'] for p in train_pairs}),
        'evaluation_labeled_pairs':len(dev_pairs),
        'evaluation_unique_original_claims':len(grouped),
        'heldout_expert_support_pairs':sum(r['label']==POSITIVE for r in results),
        'heldout_expert_contradict_pairs':sum(r['label']==NEGATIVE for r in results),
        'multi_premise_original_pairs':sum(r['multi_premise'] for r in results),
        'multiple_rationale_original_pairs':sum(r['multiple_alternatives'] for r in results),
        'same_claim_multiple_original_papers':sum(r['same_claim_multiple_docs'] for r in results),
        'doc_id_overlap_pairs':sum(r['seen_paper_id_in_training'] for r in results),
        'source_contexts':contexts,
        'classifier_training_cpu_wall':training,
        'classifier_inference_cpu_wall':inference,
        'total_experiment_wall_seconds':round(time.perf_counter()-start,5),
        'maxrss_linux_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'gpu_used_by_this_experiment':False,
        'llm_inference_calls':0,
        'source_text_exported_to_git':False,
        'source_type':'human_scientific_evidence_claim_pair_not_real_swaag_agent_task',
    }
    comparisons={
        'oracle_min_rationale':{
            'joint_word_minus_always_support':bootstrap_effect(results,'support_prior','joint_pair_word'),
            'joint_word_minus_evidence_only':bootstrap_effect(results,'evidence_only_word','joint_pair_word'),
            'joint_char_minus_always_support':bootstrap_effect(results,'support_prior','joint_pair_char'),
        }
    }
    report={'version':VERSION,'protocol':'HARD_SOURCE_POLARITY_PROTOCOL_2026-10-26.md',
        'source':overall,'matched_same_pairs_outcomes':aggregate,'clustered_paired_differences':comparisons,
        'same_claim_cross_paper_inconsistency':inconsistencies,
        'limitations':[
            'Original expert SUPPORT/CONTRADICT labels are real independent human-grounded polarity, but cannot establish true-negative irrelevant-doc classification.',
            'Evidence oracle and a known correct original paper are assumed, not discovered by an agent.',
            'This is a low-capacity, supervised CPU lexical model; no local Qwen or SWAAG worker/reviewer inference was performed.',
            'One dev set previously used for retrieval research and shared paper corpus may create selection and reuse bias; claim text itself is held out exactly.',
            'Classifier probabilities are not calibrated epistemic truth; report all hard negatives and incorrect outputs.',
            'Mean source whitespace words are not model tokens and CPU wall time under concurrent load is not GPU queue latency.',
        ]}
    return report,results


def main():
    report,records=run()
    out=HERE/'evidence/scifact-hard-polarity';out.mkdir(exist_ok=True,parents=True)
    (out/'summary_v1.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    with gzip.open(out/'case_errors_v1.json.gz','wt',encoding='utf8') as f:
        json.dump({'version':VERSION,'source_sha256':ORIGINAL_SHA256,'cases':records},f,sort_keys=True)
    print('HARD NEGATIVE HELDOUT SUMMARY')
    print(json.dumps({'source':{k:v for k,v in report['source'].items() if k not in ('classifier_inference_cpu_wall','classifier_training_cpu_wall','source_contexts')},
        'contexts':report['source']['source_contexts'],
        'oracle_confusions':{k:v['all'] for k,v in report['matched_same_pairs_outcomes']['oracle_min_rationale'].items()},
        'bm25_top3_confusions':{k:v['all'] for k,v in report['matched_same_pairs_outcomes']['bm25_top3'].items()},
        'clustering':report['clustered_paired_differences']},indent=2))
    print('REPORT',out/'summary_v1.json')
    print('CASE ERROR LEDGER',out/'case_errors_v1.json.gz')

if __name__=='__main__':main()
