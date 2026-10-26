"""Original cross-source NFCorpus heldout graded judgments and retriever controls.

Read-only tests on immutable original source and previously generated receipts;
never treats missing grade as a verified irrelevant document, or calls Qwen.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import statistics
import unittest

from evaluate_nfcorpus_cross_genre_20261026 import (
 HERE,ARCHIVE,SHA_ARCHIVE,MEMBER_HASHES,RANKERS,
 source,bootstrap,decision,ndcg_at,hash_rank,source_popularity_ranking,
)

DEST=HERE/'evidence/nfcorpus-cross-genre'


class NFCorpusCrossGenreHeldoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((DEST/'source_retrieval_metrics_v1.json').read_text())
        cls.rows=json.loads((DEST/'heldout_query_ranking_receipts_v1.json').read_text())
        cls.byqid={r['query_id']:r for r in cls.rows}

    def test_original_source_hashes_independent_query_ids_and_no_unlabelled_negative_invention(self):
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),SHA_ARCHIVE)
        docs,qs,qrels,meta=source()
        self.assertEqual(meta['original_member_sha256'],MEMBER_HASHES)
        recorded=self.report['source_version']
        for field in meta:
            if field!='source_read_sha_checks_seconds':
                self.assertEqual(recorded[field],meta[field],field)
        self.assertGreater(recorded['source_read_sha_checks_seconds'],0)
        self.assertEqual(len(docs),3633)
        self.assertEqual(len(qs),3237)
        self.assertEqual(len(self.rows),323)
        self.assertEqual(len(self.byqid),323)
        self.assertEqual(set(self.byqid),set(qrels['test']))
        self.assertFalse(set(qrels['test']) & set(qrels['train']))
        self.assertFalse(set(qrels['test']) & set(qrels['dev']))
        self.assertEqual(set(g for scores in qrels['test'].values() for g in scores.values()),{1,2})
        self.assertTrue(self.report['no_unjudged_source_is_labelled_semantically_irrelevant'])
        self.assertEqual(self.report['case_count'],323)

    def test_metrics_use_actual_graded_original_labels_and_exact_top_ten_source_refs(self):
        docs,qs,qrels,_=source()
        for row in self.rows:
            query=row['query_id'];truth=qrels['test'][query]
            self.assertEqual(row['query_text_sha256'],hashlib.sha256(qs[query]['text'].encode()).hexdigest())
            self.assertEqual(row['original_qrel_count'],len(truth))
            self.assertEqual(row['weaker_grade1_count'],sum(g==1 for g in truth.values()))
            self.assertEqual(row['strong_grade2_count'],sum(g==2 for g in truth.values()))
            self.assertEqual(row['both_strong_and_weak'],set(truth.values())=={1,2})
            self.assertEqual(set(row['methods']),set(RANKERS))
            for method,d in row['methods'].items():
                results=d['top_10_original_document_and_qrel_provenance']
                self.assertEqual(len(results),10)
                ids=[r['source_doc_id'] for r in results]
                self.assertEqual(len(set(ids)),10)
                self.assertTrue(all(i in docs for i in ids))
                for it in results:
                    self.assertEqual(it['original_relevance_grade'],truth.get(it['source_doc_id'],'unjudged'))
                self.assertAlmostEqual(d['nDCG_at_10'],ndcg_at(ids,truth,10),places=10)
                self.assertEqual(d['unknown_unjudged_top10_source_documents'],sum(i not in truth for i in ids))
                if row['strong_grade2_count']:
                    expected=sum(truth.get(i)==2 for i in ids)/row['strong_grade2_count']
                    self.assertAlmostEqual(d['grade2_recall_at_10'],expected,places=10)
                    self.assertLessEqual(d['grade2_recall_at_10'],d['grade2_recall_at_20'])
                    self.assertLessEqual(d['grade2_recall_at_20'],1)
                else:self.assertIsNone(d['grade2_recall_at_10'])
                if row['both_strong_and_weak']:
                    self.assertGreater(d['annotated_preference_total_pairs'],0)
                    self.assertAlmostEqual(d['annotated_strong_weak_auc'],
                        d['annotated_preference_correct_pairs']/d['annotated_preference_total_pairs'],places=10)
                else:self.assertIsNone(d['annotated_strong_weak_auc'])

    def test_has_independent_negative_controls_and_no_spurious_human_entailment_claim(self):
        docs,qs,qrels,_=source()
        doc_ids=sorted(docs)
        query=sorted(qrels['test'])[0]
        hash_expected=[doc_ids[i] for i in hash_rank(query,doc_ids)[:10]]
        self.assertEqual(hash_expected,[x['source_doc_id'] for x in self.byqid[query]['methods']['hash_query_doc_control']['top_10_original_document_and_qrel_provenance']])
        pop,_=source_popularity_ranking(doc_ids,qrels['train'])
        ids=[doc_ids[i] for i in pop[:10]]
        for query in qrels['test']:
            self.assertEqual(ids,[x['source_doc_id'] for x in self.byqid[query]['methods']['train_popularity_query_blind_control']['top_10_original_document_and_qrel_provenance']])
        self.assertEqual(self.report['source_gold_kind'],'citation_derived_grade1_and_grade2_not_independent_human_semantic_gold')
        self.assertTrue(any('not independent human entailment' in x for x in self.report['limitations']))
        self.assertEqual(self.report['cost']['actual_qwen_model_calls'],0)
        self.assertEqual(self.report['cost']['actual_gpu_inference_calls'],0)
        self.assertEqual(self.report['cost']['real_SWAAG_end_to_end_actions'],0)

    def test_metric_macro_means_and_source_query_sizes_are_not_confused(self):
        metrics={'graded_nDCG@10':('nDCG_at_10',323),
                 'strong_recall@10':('grade2_recall_at_10',119),
                 'strong_recall@20':('grade2_recall_at_20',119),
                 'judged_grade2_vs_grade1_AUC':('annotated_strong_weak_auc',106)}
        for name,(key,size) in metrics.items():
            stats=self.report['metrics_and_paired_bootstrap'][name]
            for method in RANKERS:
                observed=[r['methods'][method][key] for r in self.rows if r['methods'][method][key] is not None]
                self.assertEqual(len(observed),size)
                self.assertAlmostEqual(stats['individual'][method]['mean'],statistics.mean(observed),delta=1e-6)
                self.assertEqual(stats['individual'][method]['query_count'],size)
                self.assertEqual(stats['bootstrap_query_replicates'],1000)
                self.assertLessEqual(*stats['individual'][method]['query_bootstrap_95pct'])
        self.assertEqual(self.report['source_version']['test_queries_with_strong_grade2'],119)
        self.assertEqual(self.report['source_version']['test_queries_with_both_graded_levels'],106)

    def test_frozen_acceptance_no_posthoc_tuning_and_target_outcome_not_input(self):
        self.assertEqual(self.report['prespecified_protocol_sha256'],hashlib.sha256((HERE/'NFCORPUS_CROSS_GENRE_RELEVANCE_PROTOCOL_2026-10-26.md').read_bytes()).hexdigest())
        expected=decision(self.report['metrics_and_paired_bootstrap'])
        self.assertEqual(self.report['frozen_acceptance_rule_outcome'],expected)
        self.assertIn(expected['lexical_ranker_with_best_supported_cross_genre_evidence'],
                      ('bm25_full_corpus','tfidf_full_corpus','rrf_bm25_tfidf'))
        self.assertIn('not a semantic fact checker',expected['qualification'])
        self.assertEqual(self.report['ranking_methods'],list(RANKERS))

    def test_paired_query_uncertainty_recomputed_on_source_case_receipts(self):
        metrics=self.report['metrics_and_paired_bootstrap']['graded_nDCG@10']['paired_differences']
        for method,base in [('bm25_full_corpus','hash_query_doc_control'),
                            ('bm25_full_corpus','train_popularity_query_blind_control'),
                            ('tfidf_full_corpus','bm25_full_corpus'),
                            ('rrf_bm25_tfidf','bm25_full_corpus')]:
            key=method+'_minus_'+base
            values=[r['methods'][method]['nDCG_at_10']-r['methods'][base]['nDCG_at_10'] for r in self.rows]
            self.assertAlmostEqual(metrics[key]['mean_difference'],statistics.mean(values),delta=1e-6)
            self.assertEqual(metrics[key]['query_count'],323)
            self.assertEqual(len(metrics[key]['paired_query_bootstrap_95pct']),2)

    def test_cross_genre_does_not_falsely_finish_original_qwen_comparison(self):
        limitations=' '.join(self.report['limitations'])
        self.assertIn('SWAAG',limitations)
        self.assertIn('Qwen',limitations)
        self.assertTrue(self.report['no_unjudged_source_is_labelled_semantically_irrelevant'])
        self.assertGreater(sum(r['unknown_unjudged_top10_source_documents'] for q in self.rows for r in q['methods'].values()),0)
        self.assertTrue(any('No test grade-zero judgments' in text for text in self.report['limitations']))
        self.assertTrue(self.report['cost']['peak_process_resident_memory_kib']>10000)

if __name__=='__main__':unittest.main()
