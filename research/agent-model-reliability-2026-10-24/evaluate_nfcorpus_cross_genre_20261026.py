#!/usr/bin/env python3
"""Fully source-pinned NF Corpus cross-genre retrieval benchmark, CPU only.

Original NFCorpus test qrels are citation-derived graded relevance, not human
entailment gold. Unjudged is UNKNOWN, never a claimed negative semantic label.
All methods/acceptance preregistered in NFCORPUS_CROSS_GENRE_RELEVANCE_PROTOCOL_2026-10-26.md.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import resource
import statistics
import time
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERE=Path(__file__).resolve().parent
ARCHIVE=Path('/data/tmp/agent-model-independent-qrels-20261025/nfcorpus.zip')
VERSION='NFCorpus_cross_genre_original_grades_v1_20261026'
SHA_ARCHIVE='efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b'
MEMBER_HASHES={
 'nfcorpus/corpus.jsonl':'10cc83ef1826b1425e6a87090b5140b39b27755d5a27e48215a88611c899991f',
 'nfcorpus/queries.jsonl':'d024e6621b84925d485ae473d316a0c3af31c62c8068a59fb29d22f7613aef2a',
 'nfcorpus/qrels/train.tsv':'6336b80f9bffc4f063f3aa450047ad35c0b7c534efe4a6ba35e16dbace047f6a',
 'nfcorpus/qrels/dev.tsv':'b1d38b5e8f78c4a5820bce2b7ec2db54911d7690dc601e76811846b211180bd8',
 'nfcorpus/qrels/test.tsv':'f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a',
}
TOKEN=re.compile(r'(?u)\b\w+\b')
RANKERS=('hash_query_doc_control','train_popularity_query_blind_control','bm25_full_corpus','tfidf_full_corpus','rrf_bm25_tfidf')


def sha(content):return hashlib.sha256(content).hexdigest()


def tokens(text):return TOKEN.findall(text.lower())


def source():
    began=time.perf_counter()
    if sha(ARCHIVE.read_bytes())!=SHA_ARCHIVE:
        raise ValueError('NFCorpus original source archive checksum changed')
    with zipfile.ZipFile(ARCHIVE,'r') as zipfile_:
        if sorted(set(zipfile_.namelist())-{'nfcorpus/','nfcorpus/qrels/'})!=sorted(MEMBER_HASHES):
            raise ValueError('Unexpected original archive member set')
        contents={key:zipfile_.read(key) for key in MEMBER_HASHES}
    for key,data in contents.items():
        if sha(data)!=MEMBER_HASHES[key]:raise ValueError('NFCorpus source member changed: '+key)
    def read_jsonl(name):
        return [json.loads(line) for line in contents[name].decode('utf8').splitlines() if line.strip()]
    def read_tsv(name):
        lines=list(csv.DictReader(io.StringIO(contents[name].decode('utf8')),delimiter='\t'))
        if not lines or list(lines[0])!=['query-id','corpus-id','score']:
            raise ValueError('Original qrel source schema changed: '+name)
        return lines
    documents=read_jsonl('nfcorpus/corpus.jsonl')
    queries=read_jsonl('nfcorpus/queries.jsonl')
    train=read_tsv('nfcorpus/qrels/train.tsv')
    dev=read_tsv('nfcorpus/qrels/dev.tsv')
    test=read_tsv('nfcorpus/qrels/test.tsv')
    if tuple(map(len,(documents,queries,train,dev,test)))!=(3633,3237,110575,11385,12334):
        raise ValueError('NFCorpus source cardinality changed')
    if any(set(d)!=set(('_id','title','text','metadata')) for d in documents):
        raise ValueError('Unexpected corpus original fields')
    if any(set(q)!=set(('_id','text','metadata')) for q in queries):
        raise ValueError('Unexpected original query fields')
    docs={d['_id']:d for d in documents};qs={q['_id']:q for q in queries}
    if len(docs)!=len(documents) or len(qs)!=len(queries):raise ValueError('Nonunique source IDs')
    def by_query(rows,grades):
        output=defaultdict(dict)
        for row in rows:
            qid=row['query-id'];did=row['corpus-id'];grade=row['score']
            if qid not in qs or did not in docs or grade not in grades:
                raise ValueError('Unexpected original qrel or missing cited document/query')
            if did in output[qid]:raise ValueError('Duplicate graded query/document pair')
            output[qid][did]=int(grade)
        return dict(output)
    judged={'train':by_query(train,{'1'}),'dev':by_query(dev,{'1','2'}),'test':by_query(test,{'1','2'})}
    test_ids=set(judged['test'])
    if len(test_ids)!=323 or test_ids & set(judged['train']) or test_ids & set(judged['dev']):
        raise ValueError('Source train/dev/test query split leakage')
    if Counter(g for r in judged['test'].values() for g in r.values())!={1:11758,2:576}:
        raise ValueError('Original NF binary-ordinal test label distribution changed')
    high={q for q,labels in judged['test'].items() if 2 in labels.values()}
    both={q for q,labels in judged['test'].items() if set(labels.values())=={1,2}}
    if (len(high),len(both))!=(119,106):raise ValueError('No more original graded relevant groups')
    return docs,qs,judged,{'original_source_zip_sha256':SHA_ARCHIVE,'original_member_sha256':MEMBER_HASHES,
     'num_source_documents':len(docs),'num_original_queries':len(qs),
     'num_train_qrels':len(train),'num_dev_qrels':len(dev),'num_test_qrels':len(test),
     'test_queries':len(test_ids),'test_grade1_judgments':11758,'test_grade2_judgments':576,
     'test_queries_with_strong_grade2':len(high),'test_queries_with_both_graded_levels':len(both),
     'test_query_ids_overlap_train_or_dev':0,'original_label_kind':'citation_link_derived_relevance_grade_not_human_entailment',
     'source_read_sha_checks_seconds':round(time.perf_counter()-began,5)}


def lexical_index(doc_ids,doc_texts):
    tic=time.perf_counter()
    counterlist=[]; lengths=[]; index=defaultdict(list)
    for i,text in enumerate(doc_texts):
        count=Counter(tokens(text));counterlist.append(count)
        lengths.append(sum(count.values()))
        for token,tf in count.items():index[token].append((i,tf))
    N=len(doc_ids);avg=sum(lengths)/N
    if N!=3633:raise ValueError('wrong index cardinality')
    idf={term:math.log1p((N-len(postings)+0.5)/(len(postings)+0.5)) for term,postings in index.items()}
    k1=1.2;b=.75
    normalization=[k1*(1-b+b*doclen/avg) for doclen in lengths]
    def query(qtext):
        scores=np.zeros(N,dtype=np.float64)
        for term,query_tf in Counter(tokens(qtext)).items():
            if term not in index:continue
            value=idf[term]
            for i,tf in index[term]:
                scores[i]+=value*(tf*(k1+1))/(tf+normalization[i])
        return scores
    return query,{'index_build_seconds':round(time.perf_counter()-tic,5),
                  'indexed_document_token_total':sum(lengths),'unique_index_terms':len(index),
                  'mean_source_document_whitespace_words_proxy':round(statistics.mean(len(txt.split()) for txt in doc_texts),3),
                  'source_document_characters_indexed':sum(map(len,doc_texts))}


def tfidf_index(doc_texts):
    tic=time.perf_counter()
    vec=TfidfVectorizer(lowercase=True,token_pattern=r'(?u)\b\w+\b',
       ngram_range=(1,2),min_df=1,max_features=100000,sublinear_tf=True,norm='l2')
    mat=vec.fit_transform(doc_texts)
    return vec,mat,{'index_build_seconds':round(time.perf_counter()-tic,5),
        'feature_count':len(vec.vocabulary_),'nonzero_doc_features':int(mat.nnz)}


def rank(scores,ids):
    # Explicit lexicographic original document-ID tiebreak avoids accidental
    # corpus iteration and source-label order shortcuts.
    return sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]))


def hash_rank(query_id,doc_ids):
    return sorted(range(len(doc_ids)),key=lambda i:(
      hashlib.sha256(f'20261026|{query_id}|{doc_ids[i]}'.encode()).digest(),doc_ids[i]))


def source_popularity_ranking(doc_ids,train_qrels):
    cited=Counter(doc for rows in train_qrels.values() for doc in rows)
    return sorted(range(len(doc_ids)),key=lambda i:(-cited[doc_ids[i]],doc_ids[i])),{
      'training_labeled_query_ids':len(train_qrels),'unique_train_cited_docs':len(cited),
      'mean_train_citations_per_doc':round(statistics.mean(cited.values()),3)}


def ndcg_at(rank_ids,grades,k=10):
    from math import log2
    if not grades:return 0.0
    gained=[((2**int(grades.get(doc,0))-1)/log2(j+2)) for j,doc in enumerate(rank_ids[:k])]
    ideal=sorted((2**g-1 for g in grades.values()),reverse=True)[:k]
    dcg=sum(gained)
    idcg=sum(v/log2(j+2) for j,v in enumerate(ideal))
    return dcg/idcg if idcg else 0.0


def within_judged_auc(ranked,judged):
    high=[did for did,g in judged.items() if g==2]
    weak=[did for did,g in judged.items() if g==1]
    if not high or not weak:return None
    pos={doc:i for i,doc in enumerate(ranked)}
    strong_better=sum(pos[h]<pos[w] for h in high for w in weak)
    total=len(high)*len(weak)
    return {'auc':strong_better/total,'strong_better_pairs':strong_better,
      'strong_weaker_pairs':total,'inversion_pairs':total-strong_better}


def _ci(values,seed=20261026,n=1000):
    rnd=random.Random(seed)
    m=len(values)
    sample=[]
    for _ in range(n):
        ids=(rnd.randrange(m) for _ in range(m))
        sample.append(sum(values[i] for i in ids)/m)
    ordered=sorted(sample)
    return [round(ordered[24],6),round(ordered[975],6)]


def bootstrap(allrows):
    allids=sorted(allrows)
    strongids=[q for q in allids if allrows[q]['strong_grade2_count']>0]
    bothids=[q for q in allids if allrows[q]['both_strong_and_weak']]
    assert (len(allids),len(strongids),len(bothids))==(323,119,106)
    outputs={}
    metrics=[('graded_nDCG@10',allids,'nDCG_at_10'),
             ('strong_recall@10',strongids,'grade2_recall_at_10'),
             ('strong_recall@20',strongids,'grade2_recall_at_20'),
             ('judged_grade2_vs_grade1_AUC',bothids,'annotated_strong_weak_auc')]
    for label,ids,field in metrics:
        comparisons={}
        for method in RANKERS:
            sample=[allrows[q]['methods'][method][field] for q in ids]
            if any(v is None for v in sample):raise ValueError('missing labeled metric')
            comparisons[method]={'query_count':len(ids),'mean':round(statistics.mean(sample),6),
                'query_bootstrap_95pct':_ci(sample,seed=f'20261026|{label}|{method}')}
        pairs={}
        for method in RANKERS:
            for baseline in ('hash_query_doc_control','train_popularity_query_blind_control','bm25_full_corpus'):
                if method==baseline:continue
                delta=[allrows[q]['methods'][method][field]-allrows[q]['methods'][baseline][field] for q in ids]
                pairs[f'{method}_minus_{baseline}']={
                    'query_count':len(ids),'mean_difference':round(statistics.mean(delta),6),
                    'paired_query_bootstrap_95pct':_ci(delta,seed=f'20261026|{label}|{method}|{baseline}')}
        outputs[label]={'individual':comparisons,'paired_differences':pairs,
                        'bootstrap_query_replicates':1000,
                        'source_gold_caveat':'Original NFCorpus qrel grade 1/2 is citation-derived, not human entailment; unjudged unknown'}
    return outputs


def decision(metrics):
    m=metrics['graded_nDCG@10']['paired_differences']
    h=m['bm25_full_corpus_minus_hash_query_doc_control']
    p=m['bm25_full_corpus_minus_train_popularity_query_blind_control']
    bm_ok=(h['mean_difference']>=.03 and h['paired_query_bootstrap_95pct'][0]>0 and
           p['mean_difference']>=.03 and p['paired_query_bootstrap_95pct'][0]>0)
    candidates={}
    top_strong=metrics['strong_recall@10']['individual']['bm25_full_corpus']['mean']
    for name in ('tfidf_full_corpus','rrf_bm25_tfidf'):
        d=m[name+'_minus_bm25_full_corpus']
        recall=metrics['strong_recall@10']['individual'][name]['mean']
        candidates[name]={'mean_ndcg_gain_over_bm25':d['mean_difference'],
           'bootstrap_lower_bound_positive':d['paired_query_bootstrap_95pct'][0]>0,
           'strong_recall_point_delta_from_bm25':round(recall-top_strong,6),
           'eligible_to_replace_bm25':(d['mean_difference']>=.02 and
             d['paired_query_bootstrap_95pct'][0]>0 and recall-top_strong>=-.02)}
    selected=sorted((name for name,val in candidates.items() if val['eligible_to_replace_bm25']),
        key=lambda n:-candidates[n]['mean_ndcg_gain_over_bm25'])
    return {'bm25_vs_two_query_blind_negative_controls_passes_frozen_acceptance':bm_ok,
      'candidate_replacement_tests':candidates,
      'lexical_ranker_with_best_supported_cross_genre_evidence':selected[0] if selected else 'bm25_full_corpus',
      'qualification':'Ranking-only recommendation for original NFCorpus judged-document retrieval, not a semantic fact checker, Qwen, or SWAAG user/worker relevance winner.'}


def main():
    started=time.perf_counter()
    docs,queries,judged,source_manifest=source()
    ids=sorted(docs)
    text=[docs[i]['title']+' '+docs[i]['text'] for i in ids]
    bm,bb=lexical_index(ids,text)
    tf,docmat,tt=tfidf_index(text)
    popular,pop=source_popularity_ranking(ids,judged['train'])
    test_queries=judged['test']
    allrows={};costs={'bm25_index':bb,'tfidf_index':tt,'train_popularity':pop}
    time_scoring=Counter()
    for qid in sorted(test_queries):
        query=queries[qid]['text']
        d=test_queries[qid]
        t=time.perf_counter();scores_bm=bm(query);idx_bm=rank(scores_bm,ids)
        time_scoring['bm25_full_corpus']+=time.perf_counter()-t
        t=time.perf_counter();scores_tf=(tf.transform([query])@docmat.T).toarray()[0];idx_tf=rank(scores_tf,ids)
        time_scoring['tfidf_full_corpus']+=time.perf_counter()-t
        t=time.perf_counter();idx_hash=hash_rank(qid,ids)
        time_scoring['hash_query_doc_control']+=time.perf_counter()-t
        t=time.perf_counter();positions_bm=np.empty(len(ids),dtype=np.int32);positions_tf=np.empty(len(ids),dtype=np.int32)
        for j,i in enumerate(idx_bm):positions_bm[i]=j+1
        for j,i in enumerate(idx_tf):positions_tf[i]=j+1
        rrf=1/(60+positions_bm)+1/(60+positions_tf)
        idx_rrf=rank(rrf,ids)
        time_scoring['rrf_bm25_tfidf']+=time.perf_counter()-t
        t=time.perf_counter();idx_pop=popular
        time_scoring['train_popularity_query_blind_control']+=time.perf_counter()-t
        entries={'hash_query_doc_control':idx_hash,
                 'train_popularity_query_blind_control':idx_pop,
                 'bm25_full_corpus':idx_bm,'tfidf_full_corpus':idx_tf,'rrf_bm25_tfidf':idx_rrf}
        strong={doc for doc,grade in d.items() if grade==2}
        has_both={v for v in d.values()}=={1,2}
        metrics={}
        for method,ranking in entries.items():
            ranked_ids=[ids[i] for i in ranking]
            if len(set(ranked_ids))!=3633:raise ValueError('incomplete whole corpus method ranking')
            judgments=[{'source_doc_id':doc,'original_relevance_grade':d.get(doc,'unjudged')}
                       for doc in ranked_ids[:10]]
            pref=within_judged_auc(ranked_ids,d) if has_both else None
            metrics[method]={
               'nDCG_at_10':ndcg_at(ranked_ids,d,10),
               'grade2_recall_at_10':(sum(doc in strong for doc in ranked_ids[:10])/len(strong)) if strong else None,
               'grade2_recall_at_20':(sum(doc in strong for doc in ranked_ids[:20])/len(strong)) if strong else None,
               'annotated_strong_weak_auc':pref['auc'] if pref else None,
               'annotated_preference_correct_pairs':pref['strong_better_pairs'] if pref else None,
               'annotated_preference_total_pairs':pref['strong_weaker_pairs'] if pref else None,
               'unknown_unjudged_top10_source_documents':sum(doc not in d for doc in ranked_ids[:10]),
               'top_10_original_document_and_qrel_provenance':judgments,
            }
        allrows[qid]={'query_id':qid,'query_text_sha256':sha(query.encode()),
              'original_qrel_count':len(d),'weaker_grade1_count':sum(g==1 for g in d.values()),
              'strong_grade2_count':len(strong),'both_strong_and_weak':has_both,
              'methods':metrics}
    stats=bootstrap(allrows)
    status=decision(stats)
    result={'version':VERSION,'source_version':source_manifest,
        'prespecified_protocol_file':'NFCORPUS_CROSS_GENRE_RELEVANCE_PROTOCOL_2026-10-26.md',
        'prespecified_protocol_sha256':sha((HERE/'NFCORPUS_CROSS_GENRE_RELEVANCE_PROTOCOL_2026-10-26.md').read_bytes()),
        'case_count':len(allrows),'source_gold_kind':'citation_derived_grade1_and_grade2_not_independent_human_semantic_gold',
        'no_unjudged_source_is_labelled_semantically_irrelevant':True,
        'ranking_methods':list(RANKERS),
        'metrics_and_paired_bootstrap':stats,
        'frozen_acceptance_rule_outcome':status,
        'cost':{'method_index_build_and_source_data':costs,
           'per_query_ranking_wall_seconds_total':{k:round(v,5) for k,v in time_scoring.items()},
           'total_source_validation_indexing_scoring_bootstrap_seconds':round(time.perf_counter()-started,5),
           'peak_process_resident_memory_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'actual_qwen_model_calls':0,'actual_gpu_inference_calls':0,'real_SWAAG_end_to_end_actions':0,
           'token_estimator':'whole-source whitespace words only; no llama.cpp tokenizer'},
        'limitations':[
          'Citation-derived graded NFCorpus qrels, not independent human entailment or authentic user/worker SWAAG task effects.',
          'No test grade-zero judgments. A score1 remains source graded relevant; unjudged is UNKNOWN, only conventional benchmark nDCG uses zero gain.',
          '323 test queries are ID-disjoint from dev/train but document IDs and topics may repeat; bootstrap captures queries, not independent clinical facts.',
          'Only 119 queries contain any score2 positive; only 106 permit explicit grade2 versus grade1 ranking comparison.',
          'Whole-corpus retrieval is document ranking, not SciFact sentence-complete rationale or final factual correctness; never pool success percentages across tasks.',
          'CPU wall and lexical words are not actual Qwen GPU, batching, review latency, tokens or production actions.',
        ]}
    dest=HERE/'evidence/nfcorpus-cross-genre'
    dest.mkdir(parents=True,exist_ok=True)
    main=dest/'source_retrieval_metrics_v1.json'
    details=dest/'heldout_query_ranking_receipts_v1.json'
    for p in (main,details):
        if p.exists():raise FileExistsError('Frozen original NF source test already exists; cannot overwrite '+str(p))
    main.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    details.write_text(json.dumps([allrows[q] for q in sorted(allrows)],sort_keys=True,indent=2)+'\n')
    print('ORIGINAL CROSS-GENRE RESULT:')
    for metric in ('graded_nDCG@10','strong_recall@10','strong_recall@20','judged_grade2_vs_grade1_AUC'):
        print('===',metric)
        for method,v in stats[metric]['individual'].items():
            print(method,'mean',v['mean'],'CI',v['query_bootstrap_95pct'],'n',v['query_count'])
    print('=== acceptance',json.dumps(status,indent=2))
    print('=== CPU costs',json.dumps(result['cost'],indent=2))
    print('OUTPUT',dest)

if __name__=='__main__':main()
