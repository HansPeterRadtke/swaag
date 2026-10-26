"""Independent original-source and archived-output checks for polarity study.

No training, inference, outbound request, model reload, or source mutation.
"""
import collections
import gzip
import hashlib
import json
import unittest
from pathlib import Path
from evaluate_scifact_hard_polarity_20261026 import (
    load,pairs,verify_dev_predictions,content_for,HERE,ORIGINAL_SHA256,
    ARCHIVED_SHA256,CONTEXTS,MODELS,POSITIVE,NEGATIVE
)

EVIDENCE=HERE/'evidence/scifact-hard-polarity'

class IndependentSourcePolarityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=json.loads((EVIDENCE/'summary_v1.json').read_text())
        with gzip.open(EVIDENCE/'case_errors_v1.json.gz','rt') as f:cls.cases=json.load(f)['cases']

    def test_original_source_and_previous_ranker_identical(self):
        obj=self.summary['source']
        self.assertEqual(obj['original_archive_sha256'],ORIGINAL_SHA256)
        self.assertEqual(obj['original_bm25_archive_sha256'],ARCHIVED_SHA256)
        corpus,train,dev,prior=load()
        self.assertEqual((len(corpus),len(train),len(dev)),(5183,807,300))
        self.assertEqual((len(pairs(train,corpus)),len(pairs(dev,corpus))),(564,209))
        # Actual original source gold alternatives must match recorded predictions.
        self.assertEqual(len(verify_dev_predictions(pairs(dev,corpus),prior)),209)

    def test_authoritative_positive_and_true_negative_counts(self):
        counts=collections.Counter(r['label'] for r in self.cases)
        self.assertEqual(counts,{POSITIVE:138,NEGATIVE:71})
        self.assertEqual(len({r['claim_id'] for r in self.cases}),188)
        self.assertEqual(len({(r['claim_id'],r['doc_id']) for r in self.cases}),209)
        self.assertTrue(all(r['gold_complete_rationale_alternatives'] for r in self.cases))
        # Refutation is real expert-labeled negative SUPPORT, not missing evidence.
        self.assertNotIn('NEI',counts)

    def test_all_methods_same_cases_and_auditable_confusions(self):
        for ctx in CONTEXTS:
            for model in MODELS:
                measured=self.summary['matched_same_pairs_outcomes'][ctx][model]['all']
                counts=collections.Counter()
                for r in self.cases:
                    x=r['predictions'][ctx][model]
                    self.assertGreaterEqual(x['confidence_support_raw'],0)
                    self.assertLessEqual(x['confidence_support_raw'],1)
                    observed=x['label']
                    self.assertIn(observed,(POSITIVE,NEGATIVE))
                    if observed==POSITIVE:counts['TP' if r['label']==POSITIVE else 'FP']+=1
                    else:counts['TN' if r['label']==NEGATIVE else 'FN']+=1
                self.assertEqual({k:measured[k] for k in ('TP','TN','FP','FN')},
                                 {k:counts[k] for k in ('TP','TN','FP','FN')})
                self.assertEqual(measured['n'],209)

    def test_retained_entire_gold_rationale_not_partial_fragments(self):
        for ctx in CONTEXTS:
            complete=0
            for r in self.cases:
                actual=r['selected_source'][ctx]
                original_sets=[set(x) for x in r['gold_complete_rationale_alternatives']]
                expected=any(ref.issubset(set(actual['ids'])) for ref in original_sets)
                self.assertEqual(actual['complete_rationale_retained'],expected)
                complete+=expected
            self.assertEqual(complete,self.summary['source']['source_contexts'][ctx]['complete_expert_rationale'])
        self.assertEqual([self.summary['source']['source_contexts'][key]['complete_expert_rationale']
                          for key in CONTEXTS],[209,168,185,209])

    def test_truly_short_abstract_keeps_all_sentences_without_fabrication(self):
        corpus,train,dev,prior=load()
        found=pairs(dev,corpus)
        rankings=verify_dev_predictions(found,prior)
        short=[p for p in found if len(p['sentence_list'])<5]
        self.assertEqual(len(short),8)
        for p in short:
            self.assertLessEqual(len(rankings[(p['claim_id'],p['doc_id'])]),len(p['sentence_list']))
            bm=content_for(p,'bm25_top5',rankings)
            self.assertEqual(len(bm['ids']),len(p['sentence_list']))
            self.assertTrue(bm['complete_rationale_retained'])

    def test_hard_source_group_and_original_instances_not_duplicate_pseudogold(self):
        self.assertEqual(sum(r['multi_premise'] for r in self.cases),14)
        self.assertEqual(sum(r['multiple_alternatives'] for r in self.cases),85)
        self.assertEqual(sum(r['same_claim_multiple_docs'] for r in self.cases),31)
        self.assertEqual(sum(r['seen_paper_id_in_training'] for r in self.cases),124)
        for ctx in CONTEXTS:
            for model in MODELS:
                x=self.summary['matched_same_pairs_outcomes'][ctx][model]
                self.assertEqual(x['requires_multiple_gold_sentences']['n'],14)
                self.assertEqual(x['has_multiple_expert_alternatives']['n'],85)
                self.assertEqual(x['same_claim_multiple_expert_papers']['n'],31)
                self.assertEqual(x['paper_id_seen_in_train']['n'],124)
                self.assertEqual(x['paper_id_unseen_in_train']['n'],85)

    def test_original_source_version_and_private_text_not_committed(self):
        keys={'source','version','limitations','clustered_paired_differences','matched_same_pairs_outcomes'}
        self.assertTrue(keys.issubset(self.summary))
        self.assertFalse(self.summary['source']['source_text_exported_to_git'])
        self.assertFalse(self.summary['source']['gpu_used_by_this_experiment'])
        self.assertEqual(self.summary['source']['llm_inference_calls'],0)
        for r in self.cases:
            self.assertNotIn('claim',r)
            self.assertNotIn('sentence_list',r)
            self.assertNotIn('source_text',r)
            self.assertNotIn('title',r)
            for ctx in CONTEXTS:
                self.assertNotIn('text',r['selected_source'][ctx])

    def test_false_support_and_false_rejection_source_examples_persist(self):
        byid={(r['claim_id'],r['doc_id']):r for r in self.cases}
        fp=byid[(1029,'13923140')]
        self.assertEqual(fp['label'],NEGATIVE)
        self.assertEqual(fp['predictions']['oracle_min_rationale']['joint_pair_word']['label'],POSITIVE)
        fn=byid[(1379,'16322674')]
        self.assertEqual(fn['label'],POSITIVE)
        self.assertEqual(fn['predictions']['oracle_min_rationale']['joint_pair_word']['label'],NEGATIVE)

    def test_matched_negative_review_net_cost_and_uncertainty(self):
        gold=self.summary['matched_same_pairs_outcomes']['oracle_min_rationale']
        b=gold['support_prior']['all']; w=gold['joint_pair_word']['all']; c=gold['joint_pair_char']['all']
        self.assertEqual((b['TP'],b['TN'],b['FP'],b['FN']),(138,0,71,0))
        self.assertEqual((w['TP'],w['TN'],w['FP'],w['FN']),(78,19,52,60))
        self.assertEqual((c['TP'],c['TN'],c['FP'],c['FN']),(80,23,48,58))
        # Compared with the always-support reference, 19 refutation rescues
        # but 60 originally correct supports broken by word classifier.
        self.assertEqual((w['TN'],w['FN']),(19,60))
        self.assertEqual(self.summary['clustered_paired_differences']['oracle_min_rationale']
                         ['joint_word_minus_always_support']['cluster_count'],188)

if __name__=='__main__':unittest.main()
