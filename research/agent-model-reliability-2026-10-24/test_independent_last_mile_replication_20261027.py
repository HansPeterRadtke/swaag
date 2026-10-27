"""Original Git-source regrade of persisted full SWAAG final-result study.

Pure offline verifier: never reruns a source tool, model inference or durable
control. The evaluator and the verifier have distinct implementations.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import unittest

from independent_regrade_source_receipts_20261027 import (
    HERE,EVIDENCE,SOURCE_RECORD,JOURNAL,ORIGINAL_ROUND9,
    SOURCE_SHA,JOURNAL_SHA,ROUND9_SHA,
    independent_gold_and_receipt_audit,
)

AUDIT=EVIDENCE/'independent_source_and_cost_regrade_v1.json'


class GitSourceEndToEndIndependentReplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=json.loads(AUDIT.read_text())
        cls.fresh=independent_gold_and_receipt_audit()
        cls.original=json.loads(SOURCE_RECORD.read_text())

    def test_independent_input_sha_pinned_sources_and_no_result_change(self):
        self.assertEqual(hashlib.sha256(SOURCE_RECORD.read_bytes()).hexdigest(),SOURCE_SHA)
        self.assertEqual(hashlib.sha256(JOURNAL.read_bytes()).hexdigest(),JOURNAL_SHA)
        self.assertEqual(hashlib.sha256(ORIGINAL_ROUND9.read_bytes()).hexdigest(),ROUND9_SHA)
        self.assertEqual(self.audit['round10_original_result_sha256'],SOURCE_SHA)
        self.assertEqual(self.audit['round10_immutable_ledger_sha256'],JOURNAL_SHA)
        self.assertEqual(self.audit['original_prior_round9_artifact_sha256'],ROUND9_SHA)
        self.assertTrue(self.audit['source_labels_independently_derived_from_original_git_bytes'])
        self.assertEqual(self.audit['total_frozen_original_git_source_cases'],6)

    def test_full_journal_cost_confusions_recomputed_not_stored_summary_claim(self):
        for k in ('original_case_arms','total_final_answer_policies_regraded',
                  'total_frozen_original_git_source_cases',
                  'posthoc_structural_verifier_diagnostic','independently_rechecked_cases'):
            self.assertEqual(self.audit[k],self.fresh[k])
        self.assertEqual(len(self.audit['independently_rechecked_cases']),24)
        a=self.audit['original_case_arms']
        self.assertEqual(set(a),{'read_accurate','read_inverted','direct_yes_no_tool','cautious_no_tool'})
        for arm,record in a.items():
            self.assertEqual(record['cases'],6)
            self.assertEqual(record['input_whitespace_words'],self.original['cost_and_final_outcomes_by_arm'][arm]['prompt_whitespace_word_proxy_total'])
            self.assertEqual(record['model_requests'],self.original['cost_and_final_outcomes_by_arm'][arm]['scripted_model_requests_total'])
            self.assertGreater(record['microseconds'],0)

    def test_wrong_final_after_actual_read_stays_wrong_even_with_source_receipt(self):
        rows=self.audit['independently_rechecked_cases']
        incorrect=[x for x in rows if x['policy']=='read_inverted']
        self.assertEqual(len(incorrect),6)
        self.assertTrue(all(x['real_read_file_and_receipt_observed'] for x in incorrect))
        self.assertTrue(all(x['final_grade_rederived'] in ('FP','FN') for x in incorrect))
        self.assertTrue(all(not x['source_and_final_correct_and_verified'] for x in incorrect))
        self.assertEqual(self.audit['original_case_arms']['read_inverted']['source_revealed_but_false'],6)
        self.assertEqual(self.audit['original_case_arms']['read_inverted']['model_requests'],18)
        self.assertEqual(self.audit['original_case_arms']['read_inverted']['input_whitespace_words'],44214)

    def test_source_grounded_correct_with_real_tool_receipts_and_same_cost_as_inverted(self):
        accurate=self.audit['original_case_arms']['read_accurate']
        inverse=self.audit['original_case_arms']['read_inverted']
        self.assertEqual(tuple(accurate.get(k,0) for k in ('TP','TN','FP','FN')),(3,3,0,0))
        self.assertEqual(tuple(inverse.get(k,0) for k in ('TP','TN','FP','FN')),(0,0,3,3))
        for k in ('model_requests','input_whitespace_words','real_source_reads'):
            self.assertEqual(accurate[k],inverse[k])
        self.assertEqual(accurate['source_verified_completed'],6)
        self.assertEqual(accurate['real_source_reads'],6)
        self.assertEqual(inverse['source_verified_completed'],0)
        self.assertEqual(inverse['real_source_reads'],6)

    def test_unverified_correct_by_coincidence_and_explicit_safe_abstention_not_success(self):
        direct=self.audit['original_case_arms']['direct_yes_no_tool']
        cautious=self.audit['original_case_arms']['cautious_no_tool']
        self.assertEqual(tuple(direct.get(k,0) for k in ('TP','TN','FP','FN')),(3,0,3,0))
        self.assertEqual(direct['source_verified_completed'],0)
        self.assertEqual(direct['coincidentally_correct_no_tool'],3)
        self.assertEqual(direct['real_source_reads'],0)
        self.assertEqual((cautious['ABSTAIN_POS'],cautious['ABSTAIN_NEG']),(3,3))
        self.assertEqual(cautious['safe_unresolved'],6)
        self.assertEqual(cautious['source_verified_completed'],0)
        self.assertEqual(cautious['model_requests'],12)

    def test_versioned_history_case_is_independently_regraded_without_effect_reexecution(self):
        self.assertEqual(self.audit['previous_round9_original_immutable_versioned_source_cases_regraded'],2)
        self.assertEqual(self.audit['previous_round9_negative_control_cases_checked'],4)
        original=json.loads(ORIGINAL_ROUND9.read_text())
        self.assertEqual([x['version'] for x in original['source_read_to_persisted_final_and_replay']],['old','new'])
        self.assertEqual(original['summary']['true_grounded_complete_tasks'],2)
        self.assertEqual(original['summary']['no_tool_direct_false_asserts'],1)
        self.assertTrue(original['saved_state_and_control_identity_checks']['old_control_historical_answer_replayed_after_file_changed'])
        self.assertEqual(original['saved_state_and_control_identity_checks']['new_tool_calls_during_both_restarts_replays'],0)

    def test_posthoc_exact_predicate_checker_has_bounded_source_scope(self):
        gate=self.audit['posthoc_structural_verifier_diagnostic']
        self.assertEqual(gate['accepted_correct_receipted_conclusions'],6)
        self.assertEqual(gate['withheld_wrong_or_unsupported_or_abstained'],18)
        self.assertEqual(gate['false_accepts_relative_to_exact_heading_truth'],0)
        self.assertEqual(gate['false_rejects_relative_to_exact_heading_truth'],0)
        self.assertTrue(gate['is_source_specific_and_was_developed_after_observing_results'])
        self.assertTrue(gate['not_independent_model_reviewer_and_not_deployable_as_general_semantic_truth_verifier'])
        self.assertTrue(gate['no_extra_llm_calls_and_cpu_only'])

    def test_no_fake_claim_of_independently_reconstituted_full_hash_chain(self):
        notes=' '.join(self.audit['notes'])
        self.assertIn('cryptographic event hash chain independently',notes)
        self.assertIn('full temporary SQLite source-event payloads',notes)
        self.assertIn('not local Qwen',notes)
        self.assertTrue(self.audit['labels_are_exact_heading_predicates_not_blinded_human_semantic_relevance'])
        self.assertTrue(self.original['models_scripted_not_qwen'])
        self.assertEqual(self.original['actual_qwen_llm_calls'],0)
        self.assertEqual(self.original['production_worker_side_effects'],0)

if __name__=='__main__':unittest.main()
