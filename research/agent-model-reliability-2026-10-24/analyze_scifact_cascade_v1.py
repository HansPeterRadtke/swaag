#!/usr/bin/env python3
"""Exploratory post-hoc cascade analysis of the frozen SciFact rank outputs.

Not preregistered as a primary confirmatory metric: report as exploration.
Uses exact source IDs/rationale indices, never original paper text in artifacts.
"""
import argparse,collections,hashlib,io,json,math,pathlib,tarfile
from evaluate_scifact_offline_v1 import ARCHIVE_HASH


def evaluate(details,archive):
    if details['source_sha256']!=ARCHIVE_HASH:
        raise ValueError('source output version mismatch')
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_HASH:
        raise ValueError('original source mismatch')
    with tarfile.open(archive,'r:gz') as f:
        docs=[json.loads(x) for x in io.TextIOWrapper(f.extractfile('data/corpus.jsonl'),encoding='utf8')]
    docs.sort(key=lambda x:int(x['doc_id']))
    source_doc_idx={str(x['doc_id']):i for i,x in enumerate(docs)}
    document_queries={x['claim_id']:x for x in details['document_queries']}
    byclaim=collections.defaultdict(list)
    for pair in details['cases']:
        if pair['doc_id'] not in source_doc_idx:raise ValueError('unknown source doc')
        doc_idx=source_doc_idx[pair['doc_id']]
        assert doc_idx in document_queries[pair['claim_id']]['known_evidence_doc_indices']
        byclaim[pair['claim_id']].append((doc_idx,pair))
    assert len(byclaim)==len(document_queries)

    x=[]
    for doc_method in ('bm25','tfidf'):
        for doc_budget in (1,5,10,20):
            for sentence_method in ('bm25','tfidf','last','first'):
                for sent_budget in (1,2,3,4,5):
                    successes=[];stage1_successes=[]
                    for claim_id,pairs in byclaim.items():
                        top_docs=set(document_queries[claim_id]['methods'][doc_method]['top50_doc_indices'][:doc_budget])
                        visible=[pair for index,pair in pairs if index in top_docs]
                        stage1_successes.append(bool(visible))
                        successes.append(any(bool(p['methods'][sentence_method]['any_complete_rationale'][str(sent_budget)]) for p in visible))
                    x.append({'document_method':doc_method,'retrieved_document_budget':doc_budget,
                         'sentence_method':sentence_method,'selected_sentence_budget':sent_budget,
                         'total_claims_with_expert_evidence':len(byclaim),
                         'known_document_stage_hit_count':sum(stage1_successes),
                         'complete_expert_rationale_pipeline_success_count':sum(successes),
                         'complete_expert_rationale_pipeline_rate':round(sum(successes)/len(successes),5),
                         'complete_rationale_success_given_hit':round(sum(successes)/sum(stage1_successes),5) if sum(stage1_successes) else None})
    comparisons=collections.Counter()
    for p in details['cases']:
        a=bool(p['methods']['tfidf']['any_complete_rationale']['3'])
        b=bool(p['methods']['bm25']['any_complete_rationale']['3'])
        comparisons[('success' if a else 'failure','success' if b else 'failure')]+=1
    c11=comparisons[('success','success')];c10=comparisons[('success','failure')];c01=comparisons[('failure','success')];c00=comparisons[('failure','failure')]
    denom=math.sqrt((c11+c10)*(c11+c01)*(c00+c10)*(c00+c01))
    phi=(c11*c00-c10*c01)/denom if denom else None
    return {'source_sha256':ARCHIVE_HASH,'status':'EXPLORATORY POST-HOC analysis; not a preregistered hypothesis test',
        'source_document_retrieval':'Original full 5,183-paper SciFact corpus; claim-only query; known-evidence recall not exhaustive precision',
        'expert_rationale':'At least one FULL original rationale alternative must be retained in some retrieved expert-positive paper; no final SUPPORT/CONTRADICT entailment judgment',
        'claim_count':len(byclaim),'annotated_claim_document_pair_count':len(details['cases']),
        'cascade':x,
        'lexical_selector_joint_outcomes_at3':{'both_success':c11,'tfidf_success_bm25_failure':c10,
            'tfidf_failure_bm25_success':c01,'both_failure':c00,
            'paired_phi':round(phi,5) if phi is not None else None,
            'warning':'Lexical retrieval error dependence, not cross-LLM ensemble error correlation'},
        'limitations':'Known SciFact evidence annotations are incomplete negatives, study has oracle gold source only for sentence stage. Cascaded success is a necessary evidence-access condition but does not measure true final-answer correctness.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=pathlib.Path,required=True)
    p.add_argument('--details',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise SystemExit('no overwrite of old research output')
    details=json.loads(a.details.read_text());res=evaluate(details,a.archive)
    a.out.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
    print('source pairs',res['annotated_claim_document_pair_count'],'claims',res['claim_count'],'correlated_misses',res['lexical_selector_joint_outcomes_at3'])
    print('cascade example BM25 document top 10:')
    for row in res['cascade']:
        if row['document_method']=='bm25' and row['retrieved_document_budget']==10 and row['selected_sentence_budget']==3:
            print(row['sentence_method'],'success',row['complete_expert_rationale_pipeline_success_count'],'/',row['total_claims_with_expert_evidence'],'rate',row['complete_expert_rationale_pipeline_rate'])

if __name__=='__main__':main()
