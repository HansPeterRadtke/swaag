"""Verify the separate BEIR source-subset and known-strong-grade-miss ledger.

No Qwen or retriever run and no original test dataset modified.
"""
from __future__ import annotations
from collections import Counter
import hashlib,json,unittest
from pathlib import Path
from evaluate_nfcorpus_cross_genre_20261026 import HERE,source,RANKERS
from audit_nfcorpus_failures_and_version_20261026 import audit

EVIDENCE=HERE/'evidence/nfcorpus-cross-genre'

class NFCorpusCitationDerivedFailureAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved=json.loads((EVIDENCE/'postresult_source_version_and_missed_strong_qrels_v1.json').read_text())
        cls.now=audit()

    def test_source_hash_and_post_hoc_scope_not_hidden(self):
        self.assertEqual(self.saved,self.now)
        self.assertEqual(self.saved['source_retrieval_report_sha256'],hashlib.sha256((EVIDENCE/'source_retrieval_metrics_v1.json').read_bytes()).hexdigest())
        self.assertEqual(self.saved['source_top10_receipts_sha256'],hashlib.sha256((EVIDENCE/'heldout_query_ranking_receipts_v1.json').read_bytes()).hexdigest())
        self.assertEqual(self.saved['original_beir_archive_sha256'],'efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b')
        self.assertEqual(set(self.saved['methods']),set(RANKERS))

    def test_genuine_strong_positive_omissions_and_unknown_not_false_negatives(self):
        for method,stat in self.saved['methods'].items():
            self.assertEqual(stat['test_queries'],323)
            self.assertEqual(stat['top10_source_slots'],3230)
            self.assertEqual(stat['queries_with_any_high_grade_judgment'],119)
            self.assertEqual(stat['test_queries_with_both_grades'],106)
            self.assertEqual(stat['test_qrel_high_grade_records'],576)
            self.assertEqual(stat['retrieved_high_grade_test_qrels_in_top10']+stat['missed_high_grade_test_qrels_in_top10'],576)
            self.assertEqual(stat['top10_unjudged_original_documents']+stat['top10_weaker_grade1_documents']+stat['top10_strong_grade2_documents'],3230)
            self.assertEqual(stat['top10_strong_grade2_documents'],stat['retrieved_high_grade_test_qrels_in_top10'])
            self.assertGreaterEqual(stat['top10_unjudged_original_documents'],0)
        bm=self.saved['methods']['bm25_full_corpus']
        self.assertEqual(bm['retrieved_high_grade_test_qrels_in_top10'],150)
        self.assertEqual(bm['missed_high_grade_test_qrels_in_top10'],426)
        self.assertEqual(bm['queries_with_zero_high_grade_retrieved_top10'],40)
        self.assertEqual(bm['top10_unjudged_original_documents'],2530)
        self.assertEqual(self.saved['methods']['rrf_bm25_tfidf']['retrieved_high_grade_test_qrels_in_top10'],147)

    def test_examples_are_actually_known_original_strong_grade_positive_not_hallucinated(self):
        _,_,gold,_=source()
        for method,examples in self.saved['error_inventory'].items():
            misses=examples['largest_original_grade2_miss_examples_ids_only']
            self.assertLessEqual(len(misses),8)
            self.assertEqual(examples['num_source_queries_with_no_grade2_in_top10'],
                             self.saved['methods'][method]['queries_with_zero_high_grade_retrieved_top10'])
            for miss in misses:
                expected={did for did,grade in gold['test'][miss['query_id']].items() if grade==2}
                self.assertEqual(set(miss['original_grade2_doc_ids']),expected)
                self.assertFalse(set(miss['retrieved_top10_doc_ids']) & expected)
                self.assertTrue(miss['semantic_false_positive_inference_disallowed'])

    def test_native_original_release_is_not_falsely_equated_with_beir_subset(self):
        meta=self.saved['version_reconciliation']
        self.assertEqual((meta['tested_beir_processed_n_original_docs'],meta['tested_beir_processed_n_test_queries'],meta['tested_beir_processed_test_qrel_count']),(3633,323,12334))
        self.assertEqual((meta['native_nfcorpus_reported_document_count_approx'],meta['native_nfcorpus_reported_test_query_count'],meta['native_nfcorpus_reported_original_test_qrels']),(5371,325,15820))
        self.assertEqual((meta['native_nfcorpus_reported_original_test_grade1_topic_links'],meta['native_nfcorpus_reported_original_test_grade2_indirect_links'],meta['native_nfcorpus_reported_original_test_grade3_direct_citations']),(3486,11758,576))
        self.assertTrue(meta['count_consistency_only'])
        self.assertTrue(meta['NOT_verified_rowwise_original_to_BEIR_label_transform'])
        self.assertTrue(any('not proved' in s for s in self.saved['scope_limitations']))

if __name__=='__main__':unittest.main()
