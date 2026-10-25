#!/usr/bin/env python3
"""Bounded source-exact SciFact retrieval with *complete-rationale* scoring.

CPU only. No LLM/model inference, no source text in public result artifacts.
Methods and evaluation outcomes are defined in frozen preregistration.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import statistics
import tarfile

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ARCHIVE_HASH = '11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be'
K_VALUES = (1, 2, 3, 4, 5)
DOC_K_VALUES = (1, 5, 10, 20, 50)
METHODS = ('first', 'last', 'random', 'overlap', 'tfidf', 'bm25')
SEED = 20261025
TOKEN_RE = re.compile(r'(?u)\b\w\w+\b')


def tokens(s):
    return TOKEN_RE.findall(s.lower())


def read_scifact(path: Path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_HASH:
        raise ValueError('different SciFact original source archive')
    with tarfile.open(path, 'r:gz') as archive:
        corpus = [json.loads(x) for x in io.TextIOWrapper(archive.extractfile('data/corpus.jsonl'), encoding='utf8')]
        claims = [json.loads(x) for x in io.TextIOWrapper(archive.extractfile('data/claims_dev.jsonl'), encoding='utf8')]
    corpus.sort(key=lambda x: int(x['doc_id']))
    assert len(corpus) == 5183 and len(claims) == 300
    doc_idx = {str(d['doc_id']):i for i,d in enumerate(corpus)}
    grouped = []
    gold_docs = defaultdict(set)
    source_label_counts = Counter()
    raw_rationale_count = 0
    for claim in claims:
        cid = int(claim['id'])
        for docid, evidences in (claim.get('evidence') or {}).items():
            docid = str(docid)
            if docid not in doc_idx: raise ValueError('unknown evidence document')
            document = corpus[doc_idx[docid]]
            rationale_sets = []
            labels = set()
            for ev in evidences:
                label = ev['label']
                labels.add(label)
                indices = frozenset(int(i) for i in ev['sentences'])
                if not indices or max(indices)>=len(document['abstract']) or min(indices)<0:
                    raise ValueError('bad rationale sentence index')
                rationale_sets.append(indices)
                raw_rationale_count+=1
            if len(labels) != 1 or not labels <= {'SUPPORT','CONTRADICT'}:raise ValueError('inconsistent labels')
            unique = sorted(set(rationale_sets), key=lambda s:(len(s),tuple(sorted(s))))
            if not unique:raise ValueError('no evidence')
            gold_docs[cid].add(doc_idx[docid])
            source_label_counts[next(iter(labels))] += 1
            grouped.append({'claim_id':cid,'doc_id':docid,'claim':claim['claim'],
                            'label':next(iter(labels)),'sentences':document['abstract'],
                            'gold_sets':unique,'doc_index':doc_idx[docid]})
    return corpus, claims, grouped, gold_docs,source_label_counts, raw_rationale_count


class BM25:
    """Exact frozen Robertson BM25 ranker, no learned query relevance."""
    def __init__(self, texts):
        self.cv = CountVectorizer(ngram_range=(1,1))
        x = self.cv.fit_transform(texts).astype(np.float64)
        self.x = csc_matrix(x)
        self.lengths = np.asarray(x.sum(axis=1)).ravel()
        self.n = x.shape[0]
        self.avglen = max(float(self.lengths.mean()),1e-12)
        self.idf = np.log1p((self.n - np.diff(self.x.indptr) + .5)/(np.diff(self.x.indptr) + .5))
        self.denominator_offset = 1.2*(.25 + .75*self.lengths/self.avglen)

    def score(self, query):
        row = self.cv.transform([query])
        score = np.zeros(self.n, dtype=np.float64)
        for term in row.indices:
            start,end=self.x.indptr[term],self.x.indptr[term+1]
            ids = self.x.indices[start:end]
            tf = self.x.data[start:end]
            score[ids] += self.idf[term]*(tf*2.2)/(tf+self.denominator_offset[ids])
        return score


def ordered(scores, tie_indices=None):
    # Stable ties use source index ascending: zero-score queries cannot hallucinate relevance.
    return np.argsort(-np.asarray(scores),kind='stable').tolist()


def rationale_complete(gold_sets, selected):
    have=set(selected)
    return any(variant.issubset(have) for variant in gold_sets)


def sentence_results(pairs):
    all_sentences=[]; offsets=[]
    for p in pairs:
        start=len(all_sentences)
        all_sentences+=p['sentences']
        offsets.append((start,len(all_sentences)))
    word_vect=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,norm='l2',dtype=np.float32)
    X=word_vect.fit_transform(all_sentences)
    bm=BM25(all_sentences)
    out=[]
    for pair,(start,end) in zip(pairs,offsets):
        n=end-start
        q=pair['claim']
        v=word_vect.transform([q]);sims=(X[start:end]@v.T).toarray().ravel()
        bms=bm.score(q)[start:end]
        qt=set(tokens(q))
        ovs=[len(qt&set(tokens(s))) for s in pair['sentences']]
        rng=random.Random(f'{SEED}/{pair["claim_id"]}/{pair["doc_id"]}')
        rnd=list(range(n));rng.shuffle(rnd)
        ranks={'first':list(range(n)), 'last':list(reversed(range(n))), 'random':rnd,
               'overlap':ordered(ovs),'tfidf':ordered(sims), 'bm25':ordered(bms)}
        total_chars=sum(len(s) for s in pair['sentences'])
        case={'claim_id':pair['claim_id'],'doc_id':pair['doc_id'],'label':pair['label'],
              'sentence_count':n,'min_rationale_length':min(len(g) for g in pair['gold_sets']),
              'num_gold_rationale_alternatives':len(pair['gold_sets']),
              'source_sentence_chars':total_chars,
              'gold_sets':[sorted(s) for s in pair['gold_sets']],
              'methods':{}}
        for method,rank in ranks.items():
            if len(rank)!=n or len(set(rank))!=n:raise ValueError('invalid ranker output')
            case['methods'][method]={'top5_sentence_ids':rank[:5],
              'any_complete_rationale':{str(k):rationale_complete(pair['gold_sets'],rank[:k]) for k in K_VALUES},
              'at_least_one_rationale_sentence':{str(k):bool(set(rank[:k]) & set().union(*pair['gold_sets'])) for k in K_VALUES},
              'retained_source_chars':{str(k):sum(len(pair['sentences'][i]) for i in rank[:k]) for k in K_VALUES}}
        out.append(case)
    return out


def document_results(corpus,claims,gold_docs):
    texts=[d['title']+' '+' '.join(d['abstract']) for d in corpus]
    vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,norm='l2',dtype=np.float32)
    corpus_matrix=vec.fit_transform(texts)
    doc_bm=BM25(texts)
    out=[]
    for claim in claims:
        cid=int(claim['id'])
        if cid not in gold_docs:continue
        q=claim['claim']
        tfs=(vec.transform([q])@corpus_matrix.T).toarray().ravel()
        bms=doc_bm.score(q)
        x={'claim_id':cid,'known_evidence_doc_indices':sorted(gold_docs[cid]),'methods':{}}
        for method,scores in [('tfidf',tfs),('bm25',bms)]:
            rank=ordered(scores)
            gold=gold_docs[cid]
            x['methods'][method]={'top50_doc_indices':rank[:50],
              'any_gold_hit':{str(k):bool(set(rank[:k])&gold) for k in DOC_K_VALUES},
              'known_gold_recall':{str(k):len(set(rank[:k])&gold)/len(gold) for k in DOC_K_VALUES}}
        out.append(x)
    return out


def bootstrap_claim_cluster(cases,method='tfidf',comparison='first',resamples=1000):
    grouped=defaultdict(list)
    for c in cases:grouped[c['claim_id']].append(c)
    keys=sorted(grouped)
    rng=random.Random(SEED)
    diffs=[]
    def success(c,m):return float(c['methods'][m]['any_complete_rationale']['3'])
    for j in range(resamples):
        sampled=[keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        selected=[c for key in sampled for c in grouped[key]]
        diffs.append(sum(success(c,method)-success(c,comparison) for c in selected)/len(selected))
    diffs.sort()
    actual=sum(success(c,method)-success(c,comparison) for c in cases)/len(cases)
    return {'effect_any_complete_rationale_at3':round(actual,5),
            'claim_cluster_bootstrap_95pct_interval':[round(diffs[24],5),round(diffs[975],5)],
            'claim_clusters':len(keys),'repetitions':resamples,
            'interpretation':'paired development-set descriptive interval, not a general LLM benefit or multiple-comparison-adjusted p-value'}


def summarise(sentences,docs,label_counts,raw_rationale_count):
    breakdown=[]
    for label in ('ALL','SUPPORT','CONTRADICT'):
        rows=sentences if label=='ALL' else [x for x in sentences if x['label']==label]
        nontriv=[x for x in rows if x['sentence_count']>3]
        multi=[x for x in rows if x['min_rationale_length']>1]
        vals=[]
        for method in METHODS:
            vals.append({'method':method,'complete_rationale_recall_at_k':{
                str(k):round(sum(bool(x['methods'][method]['any_complete_rationale'][str(k)]) for x in rows)/len(rows),5) for k in K_VALUES},
               'nontrivial_sentence_count_gt3_complete_at3':round(sum(x['methods'][method]['any_complete_rationale']['3'] for x in nontriv)/len(nontriv),5) if nontriv else None,
               'multi_sentence_only_complete_at3':round(sum(x['methods'][method]['any_complete_rationale']['3'] for x in multi)/len(multi),5) if multi else None,
               'average_chars_retained_at3':round(statistics.mean(x['methods'][method]['retained_source_chars']['3']/max(x['source_sentence_chars'],1) for x in rows),5)})
        breakdown.append({'label':label,'n_claim_abstract_pairs':len(rows),'nontrivial_abstracts_gt3_sentences':len(nontriv),
                          'multi_sentence_gold_set_min_gt1':len(multi),'methods':vals})
    doc_metrics=[]
    for method in ('tfidf','bm25'):
        doc_metrics.append({'method':method,'source_retrieval_at_k':{str(k):{
            'at_least_one_known_gold':round(sum(x['methods'][method]['any_gold_hit'][str(k)] for x in docs)/len(docs),5),
            'mean_known_gold_recall':round(statistics.mean(x['methods'][method]['known_gold_recall'][str(k)] for x in docs),5)} for k in DOC_K_VALUES}})
    return {'n_documented_source_claims':len(docs),'n_expert_claim_abstract_pairs':len(sentences),'n_gold_rationales_before_pair_grouping':raw_rationale_count,
       'source_expert_pairs_by_label':dict(label_counts),'document_retrieval':doc_metrics,'sentence_retrieval':breakdown,
       'clustered_pair_differences_at3':{'tfidf_minus_first':bootstrap_claim_cluster(sentences,'tfidf','first'),
                                        'bm25_minus_first':bootstrap_claim_cluster(sentences,'bm25','first'),
                                        'bm25_minus_tfidf':bootstrap_claim_cluster(sentences,'bm25','tfidf')},
       'important_caveat':'document recall refers to known evidence docs only, not exhaustive relevance; sentence retrieval uses annotated gold document oracle and does not determine factual SUPPORT/CONTRADICT label; multi-sentence rationale must be recovered in its entirety'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--details',type=Path,required=True);a=p.parse_args()
    if a.results.exists() or a.details.exists():raise SystemExit('existing evidence: refuse overwrite')
    corpus,claims,pairs,gold_docs,label_counts,raw_count=read_scifact(a.archive)
    sent=sentence_results(pairs)
    docs=document_results(corpus,claims,gold_docs)
    report=summarise(sent,docs,label_counts,raw_count)
    report.update({'dataset':'SciFact original release, development split', 'source_sha256':ARCHIVE_HASH,'seed':SEED,
                   'algorithms':'frozen first/last/random/simple word overlap, word-unigram BM25, unigram-bigram TFIDF; CPU only; expert labels used only for scoring',
                   'corpus_document_count':len(corpus),'dev_claims':len(claims)})
    a.results.parent.mkdir(parents=True,exist_ok=True)
    a.details.parent.mkdir(parents=True,exist_ok=True)
    a.results.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    a.details.write_text(json.dumps({'source_sha256':ARCHIVE_HASH,'cases':sent,'document_queries':docs},indent=2,sort_keys=True)+'\n')
    print('done pairs',len(sent),'document_queries',len(docs),'rationales',raw_count,'result_file_bytes',a.results.stat().st_size,'details_bytes',a.details.stat().st_size)
    for item in report['sentence_retrieval']:
        print('label',item['label'],'n',item['n_claim_abstract_pairs'],'nontrivial_n',item['nontrivial_abstracts_gt3_sentences'],'scores_at3',[(x['method'],x['complete_rationale_recall_at_k']['3']) for x in item['methods']])
    for item in report['document_retrieval']:
        print('doc',item['method'],'at10',item['source_retrieval_at_k']['10'])

if __name__=='__main__':main()
