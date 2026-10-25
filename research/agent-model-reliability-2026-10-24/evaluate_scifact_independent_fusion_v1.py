#!/usr/bin/env python3
"""Heldout SciFact train claim evidence ranking, no model calls or label training.

Development split already studied; excludes exact claim-text overlap. Preserves
entire expert rationale alternatives as conjunctions. Source descriptions and
method matrix frozen in SCIFACT_INDEPENDENT_FUSION_PROTOCOL_2026-10-25.md.
"""
from __future__ import annotations
import argparse, collections, gzip, hashlib, io, json, pathlib, random, statistics, tarfile
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluate_scifact_offline_v1 import BM25, ordered, rationale_complete, ARCHIVE_HASH

SEED = 20261025
RRF_K = 60
SENT_K = (1,2,3,4,5)
DOC_K = (1,5,10,20,50)
METHODS = ('tfidf','bm25','first','last','random','rrf_lexical','alternate_lexical',
           'rrf_tfidf_last','rrf_bm25_last','rrf_three')
DOCUMENT_METHODS = ('tfidf','bm25','rrf_lexical')


def source(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_HASH:
        raise ValueError('upstream original SciFact source hash differs')
    with tarfile.open(archive, 'r:gz') as tar:
        def rows(name):
            with io.TextIOWrapper(tar.extractfile('data/'+name),encoding='utf8') as f:
                return [json.loads(line) for line in f]
        corpus=rows('corpus.jsonl');train=rows('claims_train.jsonl');dev=rows('claims_dev.jsonl')
    corpus.sort(key=lambda d:int(d['doc_id']))
    assert len(corpus)==5183 and len(train)==809 and len(dev)==300
    dev_texts={c['claim'] for c in dev}
    excluded={c['id'] for c in train if c['claim'] in dev_texts}
    assert len(excluded)==2
    return corpus,[x for x in train if x['id'] not in excluded],excluded


def annotated(corpus,claims):
    lookup={str(doc['doc_id']):i for i,doc in enumerate(corpus)}
    gold_docs=collections.defaultdict(set)
    pairs=[]
    raw_count=0
    for c in claims:
        for did,rs in (c.get('evidence') or {}).items():
            if not rs:continue
            docid=str(did)
            if docid not in lookup:raise ValueError('unknown source document id')
            d=corpus[lookup[docid]]
            labels={r['label'] for r in rs}
            if len(labels)!=1 or labels-{'SUPPORT','CONTRADICT'}:raise ValueError('conflicting original evidence label')
            gold=[]
            for r in rs:
                sentence_ids=frozenset(int(i) for i in r['sentences'])
                if not sentence_ids or min(sentence_ids)<0 or max(sentence_ids)>=len(d['abstract']):raise ValueError('bad expert sentence indices')
                gold.append(sentence_ids)
                raw_count+=1
            unique=sorted(set(gold),key=lambda s:(len(s),tuple(sorted(s))))
            gold_docs[int(c['id'])].add(lookup[docid])
            pairs.append({'claim_id':int(c['id']),'doc_id':docid,'doc_index':lookup[docid],
                          'claim':c['claim'],'label':next(iter(labels)),'gold_sets':unique,
                          'sentence_count':len(d['abstract']),'sentence_chars':sum(len(s) for s in d['abstract'])})
    return pairs,gold_docs,raw_count


def reciprocal_rank_fusion(ranks,k=RRF_K):
    n=len(ranks[0]);out=np.zeros(n,dtype=np.float64)
    for order in ranks:
        if len(order)!=n:raise ValueError('rank lengths differ')
        if len(set(order))!=n:raise ValueError('rank has duplicates')
        inv=np.empty(n,dtype=np.int64)
        inv[np.asarray(order,dtype=int)]=np.arange(1,n+1)
        out+=1.0/(k+inv)
    return ordered(out)


def alternate(r1,r2):
    n=len(r1);out=[];seen=set();i1=i2=0
    while len(out)<n:
        for method in (1,2):
            if method==1:
                while i1<n and r1[i1] in seen:i1+=1
                if i1<n:v=r1[i1];i1+=1
                else:continue
            else:
                while i2<n and r2[i2] in seen:i2+=1
                if i2<n:v=r2[i2];i2+=1
                else:continue
            if v not in seen:
                out.append(v);seen.add(v)
    return out


def rank_documents(corpus, claims, gold_docs):
    texts=[d['title']+' '+' '.join(d['abstract']) for d in corpus]
    tf=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,norm='l2',dtype=np.float32)
    x=tf.fit_transform(texts)
    bm=BM25(texts)
    out=[]
    for c in claims:
        cid=int(c['id'])
        if cid not in gold_docs:continue
        q=c['claim']
        tfidf=ordered((x@tf.transform([q]).T).toarray().ravel())
        bm25=ordered(bm.score(q))
        ranks={'tfidf':tfidf,'bm25':bm25,'rrf_lexical':reciprocal_rank_fusion([tfidf,bm25])}
        gold=gold_docs[cid]
        out.append({'claim_id':cid,'known_gold_doc_indices':sorted(gold),'methods':{
             method:{'top50_doc_indices':rank[:50],
                     'any_known_gold_at_k':{str(k):bool(set(rank[:k])&gold) for k in DOC_K},
                     'known_gold_recall_at_k':{str(k):len(set(rank[:k])&gold)/len(gold) for k in DOC_K}}
             for method,rank in ranks.items()}})
    return out


def rank_sentences(corpus,pairs):
    all_sent=[];offset={}
    for idx,d in enumerate(corpus):
        start=len(all_sent);all_sent.extend(d['abstract']);offset[idx]=(start,len(all_sent))
    vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,norm='l2',dtype=np.float32)
    x=vec.fit_transform(all_sent)
    bm=BM25(all_sent)
    out=[];tf_cache={};bm_cache={}
    for p in pairs:
        qid=p['claim_id'];q=p['claim'];start,end=offset[p['doc_index']];n=end-start
        if qid not in tf_cache:tf_cache[qid]=vec.transform([q]);bm_cache[qid]=bm.score(q)
        rank_tf=ordered((x[start:end]@tf_cache[qid].T).toarray().ravel())
        rank_bm=ordered(bm_cache[qid][start:end])
        rank_first=list(range(n));rank_last=list(reversed(rank_first))
        ranks={'tfidf':rank_tf,'bm25':rank_bm,'first':rank_first,'last':rank_last,
               'rrf_lexical':reciprocal_rank_fusion([rank_tf,rank_bm]),
               'alternate_lexical':alternate(rank_bm,rank_tf),
               'rrf_tfidf_last':reciprocal_rank_fusion([rank_tf,rank_last]),
               'rrf_bm25_last':reciprocal_rank_fusion([rank_bm,rank_last]),
               'rrf_three':reciprocal_rank_fusion([rank_tf,rank_bm,rank_last])}
        shuffled=rank_first[:];random.Random(f'{SEED}/{p["claim_id"]}/{p["doc_id"]}').shuffle(shuffled)
        ranks['random']=shuffled
        gold=p['gold_sets'];d=corpus[p['doc_index']]
        result={'claim_id':qid,'doc_id':p['doc_id'],'doc_index':p['doc_index'],'label':p['label'],
                'gold_sets':[sorted(s) for s in gold],'min_complete_rationale_sentences':min(len(s) for s in gold),
                'sentence_count':n,'source_char_count':p['sentence_chars'],'methods':{}}
        for name,rank in ranks.items():
            if len(rank)!=n or len(set(rank))!=n:raise ValueError('rank invariant violated')
            result['methods'][name]={'top5_sentence_ids':rank[:5],
                'complete_any_gold_rationale_at_k':{str(k):rationale_complete(gold,rank[:k]) for k in SENT_K},
                'source_char_fraction_at_k':{str(k):sum(len(d['abstract'][i]) for i in rank[:k])/max(p['sentence_chars'],1) for k in SENT_K}}
        out.append(result)
    return out


def claim_cluster_bootstrap(pairs,m1,m2,rep=1000):
    groups=collections.defaultdict(list)
    for r in pairs:groups[r['claim_id']].append(r)
    ids=sorted(groups)
    rng=random.Random(SEED)
    def success(row,method):return int(row['methods'][method]['complete_any_gold_rationale_at_k']['3'])
    def difference(grouped):
        arr=[success(r,m1)-success(r,m2) for group in grouped for r in groups[group]]
        return sum(arr)/len(arr)
    direct=difference(ids)
    samples=sorted(difference([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(rep))
    return {'estimate':round(direct,5),'claim_cluster_bootstrap_interval_95pct':[round(samples[24],5),round(samples[975],5)],
            'clusters':len(ids),'resamples':rep,'interpretation':'paired train-split descriptive CI, not model semantic correctness'}


def summary(sent,docs,excluded,raw_count):
    total_claims=len(docs)
    groups=[]
    for label in ('ALL','SUPPORT','CONTRADICT'):
        cases=[r for r in sent if label=='ALL' or r['label']==label]
        multi=[r for r in cases if r['min_complete_rationale_sentences']>=2]
        scores=[]
        for method in METHODS:
            scores.append({'method':method,
               'complete_gold_rationale_at_k':{str(k):round(sum(r['methods'][method]['complete_any_gold_rationale_at_k'][str(k)] for r in cases)/len(cases),5) for k in SENT_K},
               'multisentence_complete_at3':round(sum(r['methods'][method]['complete_any_gold_rationale_at_k']['3'] for r in multi)/len(multi),5) if multi else None,
               'mean_source_char_fraction_at3':round(statistics.mean(r['methods'][method]['source_char_fraction_at_k']['3'] for r in cases),5)})
        groups.append({'label':label,'n_pairs':len(cases),'multi_sentence_rationale_n':len(multi),'methods':scores})
    doc_results=[{'method':method,'any_annotated_gold_hit_at_k':{
      str(k):round(sum(d['methods'][method]['any_known_gold_at_k'][str(k)] for d in docs)/len(docs),5) for k in DOC_K}}
      for method in DOCUMENT_METHODS]
    lookup={r['claim_id']:r for r in docs};pbyclaim=collections.defaultdict(list)
    for s in sent:pbyclaim[s['claim_id']].append(s)
    cascades=[]
    for dmethod in DOCUMENT_METHODS:
        for k_docs in (1,5,10):
            for smethod in METHODS:
                for k_sent in (1,3,5):
                    succ=0;doc_hit=0
                    for cid,pairs in pbyclaim.items():
                        top=set(lookup[cid]['methods'][dmethod]['top50_doc_indices'][:k_docs])
                        visible=[row for row in pairs if row['doc_index'] in top]
                        doc_hit+=bool(visible)
                        succ+=any(r['methods'][smethod]['complete_any_gold_rationale_at_k'][str(k_sent)] for r in visible)
                    cascades.append({'doc_method':dmethod,'doc_topk':k_docs,'sentence_method':smethod,'sentence_topk':k_sent,
                      'success_claims':succ,'n_claims':total_claims,'known_doc_hit_claims':doc_hit,'complete_rationale_pipeline_rate':round(succ/total_claims,5)})
    contingency=collections.Counter((r['methods']['tfidf']['complete_any_gold_rationale_at_k']['3'],
                                      r['methods']['bm25']['complete_any_gold_rationale_at_k']['3']) for r in sent)
    a=contingency[(True,True)];b=contingency[(True,False)];c=contingency[(False,True)];d=contingency[(False,False)]
    denom=((a+b)*(a+c)*(d+b)*(d+c))**.5
    return {'source_sha256':ARCHIVE_HASH,'split':'train','excluded_duplicate_dev_claim_ids':sorted(excluded),
       'n_annotated_claims':total_claims,'n_expert_annotated_claim_doc_pairs':len(sent),'original_rationale_alternatives':raw_count,
       'no_training_on_expert_annotations':True,'methods_frozen_before_run':list(METHODS),
       'groups':groups,'document_results':doc_results,'cascade':cascades,
       'lexical_selector_joint_errors_top3':{'both_hit':a,'tfidf_only':b,'bm25_only':c,'both_missed':d,
           'phi_correlation':round((a*d-b*c)/denom,5) if denom else None,
           'type':'lexical evidence retrieval, NOT model ensemble truth judgments'},
       'cluster_bootstrap_effects':{
          'rrf_lexical_minus_tfidf':claim_cluster_bootstrap(sent,'rrf_lexical','tfidf'),
          'rrf_lexical_minus_bm25':claim_cluster_bootstrap(sent,'rrf_lexical','bm25'),
          'rrf_tfidf_last_minus_tfidf':claim_cluster_bootstrap(sent,'rrf_tfidf_last','tfidf'),
          'rrf_three_minus_tfidf':claim_cluster_bootstrap(sent,'rrf_three','tfidf'),
          'tfidf_minus_first':claim_cluster_bootstrap(sent,'tfidf','first')},
       'critical_limitations':'Expert evidence source retrieval only; no final answer/worker effect accuracy. Dev-informed positional methods evaluated on train claims excluding identical dev texts; scientific papers may overlap across splits; no true-negative unannotated-paper ground truth.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=pathlib.Path,required=True)
    p.add_argument('--metrics',type=pathlib.Path,required=True);p.add_argument('--details',type=pathlib.Path,required=True)
    a=p.parse_args()
    if a.metrics.exists() or a.details.exists():raise SystemExit('refuse overwrite frozen evidence')
    corpus,train,excluded=source(a.archive)
    pairs,gold_docs,raw_count=annotated(corpus,train)
    docs=rank_documents(corpus,train,gold_docs)
    sentences=rank_sentences(corpus,pairs)
    metrics=summary(sentences,docs,excluded,raw_count)
    a.metrics.parent.mkdir(parents=True,exist_ok=True);a.details.parent.mkdir(parents=True,exist_ok=True)
    a.metrics.write_text(json.dumps(metrics,sort_keys=True,indent=2)+'\n')
    details={'source_sha256':ARCHIVE_HASH,'split':'train','excluded_duplicate_dev_claim_ids':sorted(excluded),
             'sentence_case_predictions':sentences,'document_query_predictions':docs}
    # All derived source IDs and gold rationale indices; no original source text.
    with gzip.GzipFile(filename=str(a.details),mode='wb',compresslevel=9,mtime=0) as f:
        f.write((json.dumps(details,sort_keys=True,indent=2)+'\n').encode('utf8'))
    print('COMPLETED train claims with source evidence',len(docs),'claim-paper pairs',len(sentences),
          'rationale alternatives',raw_count,'source text exclusions',len(excluded))
    all_group=metrics['groups'][0]
    print('THREE SENTENCE FULL RATIONALE',[(m['method'],m['complete_gold_rationale_at_k']['3']) for m in all_group['methods']])
    print('DOCUMENT RETRIEVAL TOP10',[(m['method'],m['any_annotated_gold_hit_at_k']['10']) for m in metrics['document_results']])
    print('SOURCE_LEXICAL ERROR JOINT',metrics['lexical_selector_joint_errors_top3'])
    print('BOOTSTRAP',metrics['cluster_bootstrap_effects'])

if __name__=='__main__':main()
