"""Pinned source-human-gold review-policy outcomes, never replaying model calls."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EVIDENCE=HERE/'evidence/source-authority-policy'
POLICIES=(
 'zero_call_support','zero_call_abstain','one_word_top3','one_word_full',
 'one_char_top3','two_consensus_top3','selective_review_uncertain',
 'selective_review_support','one_word_abstain_uncertain','two_consensus_full')


class OriginalExpertSourcePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=json.loads((EVIDENCE/'policy_summary_v1.json').read_text())
        cls.records=json.loads((EVIDENCE/'final_verdict_and_cost_ledger_v1.json').read_text())
        cls.joint=json.loads((EVIDENCE/'matched_reviewer_case_errors_v1.json').read_text())

    def test_archived_source_and_split_are_identical(self):
        a=self.summary['source_reference']
        orig=Path('/data/tmp/agent-model-independent-qrels-20261025/scifact_original.tar.gz')
        prev=HERE/'evidence/scifact-offline/source_evidence_case_predictions_v1.json.gz'
        self.assertEqual(hashlib.sha256(orig.read_bytes()).hexdigest(),a['original_sci_fact_archive_sha256'])
        self.assertEqual(hashlib.sha256(prev.read_bytes()).hexdigest(),a['original_BM25_evidence_prediction_sha256'])
        self.assertEqual(hashlib.sha256((HERE/'evaluate_scifact_hard_polarity_20261026.py').read_bytes()).hexdigest(),a['original_classifier_code_sha256'])
        self.assertEqual(hashlib.sha256((HERE/'SOURCE_AUTHORITY_REVIEW_POLICY_PROTOCOL_2026-10-26.md').read_bytes()).hexdigest(),a['protocol_sha256'])
        self.assertTrue(a['all_original_dev_pairs_reconciled_with_last_round'])
        self.assertEqual([a['training_independently_source_labeled_pairs'],a['original_eval_independently_labeled_pairs'],a['distinct_eval_source_claims']],[564,209,188])

    def test_per_case_expert_labels_and_paired_policy_completeness(self):
        self.assertEqual(len(self.records),209*len(POLICIES))
        by=defaultdict(set); gold={}
        for x in self.records:
            key=(x['case_id'],x['doc_id'])
            by[key].add(x['policy']);gold.setdefault(key,x['original_gold_relation'])
            self.assertEqual(gold[key],x['original_gold_relation'])
            self.assertIn(x['final_decision'],('SUPPORT','CONTRADICT','needs_independent_verification'))
            self.assertIn('cited paper',x['final_user_visible_reply'])
            self.assertEqual(x['model_calls'],len(x['stages']))
            self.assertEqual(x['model_calls'],sum(x['model_calls'] for y in [x] for _ in [0]))
            self.assertEqual(x['complete_source_rationale_was_available_to_any_called_model'],
                 any(s['full_rationale_was_available'] for s in x['stages']))
        self.assertEqual(len(by),209)
        self.assertTrue(all(s==set(POLICIES) for s in by.values()))
        self.assertEqual(Counter(gold.values()),{'SUPPORT':138,'CONTRADICT':71})

    def test_every_policy_scores_final_claims_not_abstentions_as_correct(self):
        for policy in POLICIES:
            batch=[x for x in self.records if x['policy']==policy]
            r=self.summary['results'][policy]
            self.assertEqual(len(batch),209)
            self.assertEqual(r['n_original_claim_paper_pairs'],209)
            self.assertEqual(r['TP'],sum(x['final_decision']=='SUPPORT' and x['original_gold_relation']=='SUPPORT' for x in batch))
            self.assertEqual(r['TN'],sum(x['final_decision']=='CONTRADICT' and x['original_gold_relation']=='CONTRADICT' for x in batch))
            self.assertEqual(r['FP'],sum(x['false_support'] for x in batch))
            self.assertEqual(r['FN'],sum(x['false_contradiction'] for x in batch))
            self.assertEqual(r['ABSTAIN_POSITIVE'],sum(x['final_decision']=='needs_independent_verification' and x['original_gold_relation']=='SUPPORT' for x in batch))
            self.assertEqual(r['ABSTAIN_NEGATIVE'],sum(x['final_decision']=='needs_independent_verification' and x['original_gold_relation']=='CONTRADICT' for x in batch))
            self.assertEqual(r['correct_decisive_total'],sum(x['agrees_with_human_gold'] for x in batch))
            self.assertEqual(r['incorrect_decisive_total'],sum(x['incorrect_decisive'] for x in batch))
            self.assertEqual(r['model_inference_calls_total'],sum(x['model_calls'] for x in batch))
            self.assertEqual(r['source_whitespace_words_reprocessed_across_calls'],sum(x['source_word_count_total'] for x in batch))
            self.assertEqual(r['source_characters_reprocessed_across_calls'],sum(x['source_character_count_total'] for x in batch))
            self.assertEqual(r['correct_decisive_without_complete_rationale_present'],sum(x['correct_final_without_complete_source_rationale'] for x in batch))
            self.assertEqual(r['TP']+r['TN']+r['FP']+r['FN']+r['ABSTAIN_POSITIVE']+r['ABSTAIN_NEGATIVE'],209)

    def test_correlated_reviewer_errors_and_an_observed_override_can_harm(self):
        r=self.summary['same_case_primary_reviewer_errors']
        self.assertEqual(r['total_matched_original_cases'],209)
        self.assertEqual([r['both_correct'],r['both_wrong'],r['word_correct_char_wrong'],r['word_wrong_char_correct']],[87,103,10,9])
        self.assertEqual(r['plain_char_override_recovers_word_errors'],9)
        self.assertEqual(r['plain_char_override_breaks_correct_word_answers'],10)
        self.assertEqual(len(self.joint),209)
        self.assertEqual(sum(x['shared_wrong_error'] for x in self.joint),103)
        self.assertEqual(sum(x['reviewer_wrong_after_correct'] for x in self.joint),10)
        self.assertEqual(sum(x['reviewer_right_after_wrong'] for x in self.joint),9)

    def test_reviewer_added_cost_and_error_tradeoff(self):
        a=self.summary['results'];primary=a['one_word_top3'];two=a['two_consensus_top3']
        self.assertEqual((primary['TP'],primary['TN'],primary['FP'],primary['FN']),(79,18,53,59))
        self.assertEqual((two['TP'],two['TN'],two['FP'],two['FN'],two['abstained_total']),(71,16,48,55,19))
        self.assertEqual(primary['model_inference_calls_total'],209)
        self.assertEqual(two['model_inference_calls_total'],418)
        self.assertEqual(two['correct_decisive_total'],primary['correct_decisive_total']-10)
        self.assertEqual(two['incorrect_decisive_total'],primary['incorrect_decisive_total']-9)
        self.assertEqual(a['selective_review_uncertain']['model_inference_calls_total'],324)
        self.assertEqual(a['selective_review_support']['model_inference_calls_total'],341)
        self.assertEqual(a['zero_call_abstain']['correct_decisive_total'],0)
        self.assertEqual(a['zero_call_abstain']['abstained_total'],209)
        self.assertEqual(a['zero_call_support']['FP'],71)

    def test_exact_not_optimistic_source_availability(self):
        a=self.summary['results']
        self.assertEqual(a['one_word_top3']['correct_decisive_without_complete_rationale_present'],20)
        self.assertEqual(a['one_word_full']['correct_decisive_without_complete_rationale_present'],0)
        self.assertEqual(a['two_consensus_top3']['correct_decisive_without_complete_rationale_present'],17)
        self.assertEqual(a['two_consensus_full']['correct_decisive_without_complete_rationale_present'],0)
        self.assertEqual(a['selective_review_uncertain']['correct_decisive_without_complete_rationale_present'],7)
        self.assertGreater(a['two_consensus_full']['source_whitespace_words_reprocessed_across_calls'],
                           a['one_word_full']['source_whitespace_words_reprocessed_across_calls'])

    def test_original_source_authority_scope_and_version(self):
        scope=self.summary['source_authority_relationships']
        self.assertEqual(scope['source_papers_with_both_expert_SUPPORT_and_CONTRADICT_relations_to_distinct_claims'],179)
        found=False
        for ex in scope['illustrative_distinct_claim_source_edges']:
            if ex['paper_id']=='16472469':
                self.assertEqual({v['expert_relation'] for v in ex['relationships']},{'SUPPORT','CONTRADICT'})
                found=True
        self.assertTrue(found)
        self.assertIn('not globally',scope['authority_interpretation'])
        self.assertEqual(self.summary['resource_costs']['actual_qwen_llm_calls'],0)
        self.assertTrue(self.summary['resource_costs']['no_external_action_effects'])
        self.assertEqual(self.summary['source_reference']['original_classifier_code_sha256'],
                         hashlib.sha256((HERE/'evaluate_scifact_hard_polarity_20261026.py').read_bytes()).hexdigest())

    def test_reviewers_only_abstain_on_disagreement_not_certain_proof(self):
        allrows={(r['case_id'],r['doc_id'],r['policy']):r for r in self.records}
        for e in self.joint:
            key=e['case_id'],e['doc_id']
            row=allrows[key+('two_consensus_top3',)]
            if e['word_proposal']==e['char_proposal']:
                self.assertEqual(row['final_decision'],e['word_proposal'])
            else:self.assertEqual(row['final_decision'],'needs_independent_verification')
            if e['shared_wrong_error']:
                self.assertFalse(row['agrees_with_human_gold'])
                self.assertTrue(row['incorrect_decisive'])

    def test_claim_cluster_uncertainty_is_descriptive_not_simulated_independence(self):
        x=self.summary['uncertainty']
        self.assertEqual(x['distinct_claim_clusters'],188)
        self.assertEqual(x['resamples'],1000)
        self.assertAlmostEqual(x['estimates']['wrong_decisive']['difference'],-9/209,places=6)
        self.assertAlmostEqual(x['estimates']['correct_decisive']['difference'],-10/209,places=6)
        self.assertAlmostEqual(x['estimates']['abstain']['difference'],19/209,places=6)

    def test_source_text_not_exported_and_cost_units_not_qwen(self):
        for x in self.records:
            self.assertNotIn('claim',x)
            self.assertNotIn('source_text',x)
            self.assertNotIn('sentence_list',x)
            self.assertNotIn('title',x)
            self.assertTrue(all('source_scope' in stage for stage in x['stages']))
        self.assertIsNone(self.summary['resource_costs']['actual_gpu_vram_attributable_to_calls'])
        self.assertEqual(self.summary['resource_costs']['all_cpu_classifier_calls'],2337)
        self.assertGreater(self.summary['resource_costs']['peak_resident_memory_kib'],10000)
        self.assertTrue(any('not Qwen' in x for x in self.summary['limitations']))

if __name__=='__main__':unittest.main()
