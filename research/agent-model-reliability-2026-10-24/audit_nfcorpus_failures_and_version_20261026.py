#!/usr/bin/env python3
"""Source-exact post-result retrieval misses and BEIR-vs-native scope audit.

Never re-fits/reruns ranking algorithms; reads frozen source and stored top10
receipts. Native NFCorpus comparison is metadata-only and NOT rowwise verified.
"""
from __future__ import annotations
from collections import Counter
import hashlib,json
from pathlib import Path
from evaluate_nfcorpus_cross_genre_20261026 import HERE,source,RANKERS

ROOT=HERE/'evidence/nfcorpus-cross-genre'
OUTCOME_SHA='eb41d9687197277460393244314b5c6bfae61f539f0a34d8a132db2bdd51208b'
# The outcome source is validated by its own original archive/TSV hashes even
# if a later result-format hash changes; compare report before publishing.
VERSION='NFCorpus_BEIR_original_version_scope_and_false_miss_v1_20261026'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit():
    result=json.loads((ROOT/'source_retrieval_metrics_v1.json').read_text())
    rows=json.loads((ROOT/'heldout_query_ranking_receipts_v1.json').read_text())
    docs,queries,qrels,original=source()
    if len(rows)!=323 or result['source_version']['original_source_zip_sha256']!=original['original_source_zip_sha256']:
        raise ValueError('source version or frozen query count changed')
    keyed={x['query_id']:x for x in rows}
    if len(keyed)!=len(rows) or set(keyed)!=set(qrels['test']):raise ValueError('test identity changed')
    reports={};evidence={}
    for method in RANKERS:
        counts=Counter();pairs=Counter();bad=[]
        for qid in sorted(qrels['test']):
            gold=qrels['test'][qid];record=keyed[qid]['methods'][method]
            high={did for did,g in gold.items() if g==2}
            weak={did for did,g in gold.items() if g==1}
            first=[x['source_doc_id'] for x in record['top_10_original_document_and_qrel_provenance']]
            found=high.intersection(first);missing=high-set(first)
            counts['test_queries']+=1
            counts['top10_source_slots']+=len(first)
            counts['top10_unjudged_original_documents']+=sum(d not in gold for d in first)
            counts['top10_weaker_grade1_documents']+=sum(d in weak for d in first)
            counts['top10_strong_grade2_documents']+=len(found)
            counts['test_qrel_high_grade_records']+=len(high)
            if record['nDCG_at_10']==0:counts['test_queries_with_no_graded_top10_sources']+=1
            if high:
                counts['queries_with_any_high_grade_judgment']+=1
                counts['retrieved_high_grade_test_qrels_in_top10']+=len(found)
                counts['missed_high_grade_test_qrels_in_top10']+=len(missing)
                counts['queries_with_zero_high_grade_retrieved_top10']+=int(not found)
                if not found:bad.append({'query_id':qid,'n_original_high_grade_qrels':len(high),
                    'original_grade2_doc_ids':sorted(high),
                    'retrieved_top10_doc_ids':first,
                    'original_qrel_source':'BEIR NFCorpus test.tsv score2',
                    'semantic_false_positive_inference_disallowed':True})
            if record['annotated_preference_total_pairs'] is not None:
                counts['test_queries_with_both_grades']+=1
                pairs['pairs']+=record['annotated_preference_total_pairs']
                pairs['correct_graded_order_pairs']+=record['annotated_preference_correct_pairs']
                pairs['inverted_graded_order_pairs']+=record['annotated_preference_total_pairs']-record['annotated_preference_correct_pairs']
                counts['queries_with_stronger_lower_rank_than_weaker_at_least_half_pairs']+=int(record['annotated_strong_weak_auc']<=.5)
        assert counts['test_queries']==323 and counts['queries_with_any_high_grade_judgment']==119
        assert counts['test_queries_with_both_grades']==106 and counts['test_qrel_high_grade_records']==576
        assert counts['retrieved_high_grade_test_qrels_in_top10']+counts['missed_high_grade_test_qrels_in_top10']==576
        assert pairs['correct_graded_order_pairs']+pairs['inverted_graded_order_pairs']==pairs['pairs']
        reports[method]={**dict(counts),
          'annotated_lower_grade1_vs_higher_grade2_pairs':dict(pairs),
          'pair_weighted_strong_grade_preference_AUC':round(pairs['correct_graded_order_pairs']/pairs['pairs'],6),
          'mean_unknown_unjudged_top10_per_query_not_semantic_false_positives':round(counts['top10_unjudged_original_documents']/323,6),
          'mean_strong_grade2_qrel_recall_by_total_labels_micro':round(counts['retrieved_high_grade_test_qrels_in_top10']/576,6)}
        evidence[method]={'num_source_queries_with_no_grade2_in_top10':len(bad),
          'largest_original_grade2_miss_examples_ids_only':sorted(bad,key=lambda e:(-e['n_original_high_grade_qrels'],e['query_id']))[:8]}
    out={'version':VERSION,
       'source_retrieval_report_sha256':digest(ROOT/'source_retrieval_metrics_v1.json'),
       'source_top10_receipts_sha256':digest(ROOT/'heldout_query_ranking_receipts_v1.json'),
       'original_beir_archive_sha256':original['original_source_zip_sha256'],
       'methods':reports,'error_inventory':evidence,
       'version_reconciliation':{
         'tested_beir_processed_n_original_docs':3633,'tested_beir_processed_n_test_queries':323,
         'tested_beir_processed_test_qrel_count':12334,
         'tested_beir_grade1_count':11758,'tested_beir_grade2_count':576,
         'native_nfcorpus_reported_document_count_approx':5371,
         'native_nfcorpus_reported_test_query_count':325,
         'native_nfcorpus_reported_original_test_qrels':15820,
         'native_nfcorpus_reported_original_test_grade1_topic_links':3486,
         'native_nfcorpus_reported_original_test_grade2_indirect_links':11758,
         'native_nfcorpus_reported_original_test_grade3_direct_citations':576,
         'count_consistency_only':(12334==15820-3486 and 11758==11758 and 576==576),
         'NOT_verified_rowwise_original_to_BEIR_label_transform':True,
         'original_release_metadata_url':'https://semproc.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/',
         'original_ir_datasets_metadata_url':'https://ir-datasets.com/nfcorpus.html',
         'beir_repackaged_dataset_url':'https://huggingface.co/datasets/BeIR/nfcorpus',
       },
       'scope_limitations':[
          'The original NFCorpus and BEIR-preprocessed subsets are NOT identical source populations; do not transfer metrics to the full native release.',
          'Exact grade count coincidences are consistent with removal/remapping of the weakest original grade, but row-by-row equality is not proved.',
          'Unjudged top-10 results are UNKNOWN. Strong-grade omissions are FN only against known citation-derived stronger-grade judgments.',
          'Within-judged grades 1 and 2 reflect link strength not factual entailment or worker/user role priority.',
          'This is a separate post-result descriptive error inventory and does not modify the preregistered ranking/acceptance rule.',
       ]}
    return out


def main():
    out=audit()
    path=ROOT/'postresult_source_version_and_missed_strong_qrels_v1.json'
    if path.exists():raise FileExistsError('Never overwrite original postresult source-error manifest')
    path.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    for method in RANKERS:
        x=out['methods'][method]
        print(method,'strong_grade_2_retrieved_of_576',x['retrieved_high_grade_test_qrels_in_top10'],
          'queries_without_high_grade_hit',x['queries_with_zero_high_grade_retrieved_top10'],
          'unjudged_top10',x['top10_unjudged_original_documents'],
          'micro_graded_auc',x['pair_weighted_strong_grade_preference_AUC'])
    print('SAVED',path)

if __name__=='__main__':main()
