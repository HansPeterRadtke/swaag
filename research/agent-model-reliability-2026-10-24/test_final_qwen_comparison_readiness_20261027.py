"""Read-only original Qwen task-matrix and live-snapshot audit integrity.

Never asks a model anything or touches production SQLite in write mode.
No proposed SWAAG change or a reported missing task is scored as a wrong
model answer. All source records are hashed and original statuses retained.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import json
import unittest

from audit_qwen_source_leakage_20261026 import audit as audit_original_journals

from audit_final_qwen_comparison_readiness_20261027 import (
    ARCHIVE,CONTEXT,SOURCE_SHA,CONTEXT_SHA,
    read_source,cell_index,matched_costs,hash_file,HERE,
)

REPORT=HERE/'evidence/final-qwen-readiness/original_qwen_source_and_live_readonly_eligibility_v1.json'


class QwenEndToEndSourceEligibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive,cls.context=read_source()
        cls.cells=cell_index(cls.archive,cls.context)
        cls.costs=matched_costs(cls.archive,cls.cells)
        cls.report=json.loads(REPORT.read_text())

    def test_source_versions_and_leaked_relevance_label_ids_preserved(self):
        self.assertEqual(hash_file(ARCHIVE),SOURCE_SHA)
        self.assertEqual(hash_file(CONTEXT),CONTEXT_SHA)
        self.assertEqual(self.report['recorded_original_Qwen_evidence_sha256'],SOURCE_SHA)
        self.assertEqual(self.report['original_context_manifest_sha256'],CONTEXT_SHA)
        self.assertEqual(len(self.context['cases']),8)
        for case in self.context['cases']:
            self.assertEqual(case['support_ids'],['r1','r2'])
            self.assertEqual(len(case['passages']),5)
        self.assertTrue(self.report['source_role_encoded_expected_selection_all_cases'])
        self.assertFalse(self.report['original_gold_independent_for_end_to_end_task'])

    def test_every_original_planned_case_model_condition_kept_even_if_missing(self):
        self.assertEqual(len(self.cells),48)
        self.assertEqual(self.report['full_original_model_task_condition_cells'],48)
        expected={'completed_final_answer':13,'first_stage_selector_only_no_final_answer':8,
                  'admission_refused_before_inference':10,'no_recorded_attempt':17}
        self.assertEqual(Counter(r['status'] for r in self.cells.values()),expected)
        self.assertEqual(self.report['source_cell_status_counts'],expected)
        self.assertEqual(self.report['completed_full_direct_answers'],9)
        self.assertEqual(self.report['completed_privileged_researcher_gold_selected_answers'],4)
        self.assertEqual(self.report['completed_model_selected_stage_one_only'],8)
        self.assertEqual(self.report['completed_model_selected_end_to_end_chains'],0)
        self.assertEqual(len(self.report['source_cell_matrix']),48)
        self.assertEqual({(r['case_id'],r['model_family'],r['strategy']) for r in self.report['source_cell_matrix']},set(self.cells))

    def test_independent_original_event_journals_and_original_source_spans(self):
        # Go back to the pre-existing original SHA-pinned Qwen event files, not
        # just the derived current final source comparison summary.
        originals=audit_original_journals()
        source=originals['summary']
        self.assertEqual(source['original_event_records'],150)
        self.assertEqual(source['unique_events'],94)
        self.assertEqual(source['byte_identical_overlaps'],56)
        self.assertEqual(source['unchanged_source_passage_spans_verified'],40)
        self.assertEqual(source['full_planned_condition_cells'],48)
        self.assertEqual(source['completed_full_context_direct'],9)
        self.assertEqual(source['completed_researcher_oracle_gold_selection'],4)
        self.assertEqual(source['completed_model_source_selection_first_stage_only'],8)
        self.assertEqual(source['admission_refused_before_model_call'],10)
        self.assertEqual(source['not_attempted_any_recorded_call'],17)
        self.assertEqual(source['completed_real_two_call_selection_then_answer'],0)
        self.assertEqual(originals['costs']['paired_full_direct_evaluated_tokens'],5593)
        self.assertEqual(originals['costs']['paired_first_selector_evaluated_tokens'],5600)
        self.assertEqual(originals['costs']['same_case_model_paired_full_vs_selector_first_stage_count'],6)

    def test_stage_two_censorship_does_not_fake_model_incorrectness(self):
        for r in self.report['source_cell_matrix']:
            self.assertFalse(r['finished_actual_model_selected_two_stage_answer'])
            self.assertFalse(r['independent_final_task_outcome_available'])
            self.assertEqual(r['expected_answer_origin'],'researcher_authored_not_blinded_independent_gold')
            if r['status']=='first_stage_selector_only_no_final_answer':
                self.assertEqual(r['strategy'],'model_select')
                self.assertEqual(r['completed_model_calls'],1)
                self.assertEqual(r['stage2_final_correctness'],'unknown_stage2_not_executed')
            elif r['status']=='admission_refused_before_inference':
                self.assertEqual(r['completed_model_calls'],0)
                self.assertIsNone(r['observed_total_tokens'])
            elif r['status']=='no_recorded_attempt':
                self.assertIsNone(r['observed_total_tokens'])
                self.assertIsNone(r['observed_wall_seconds'])
            elif r['status']=='completed_final_answer':
                self.assertNotEqual(r['strategy'],'model_select')
                self.assertGreater(r['observed_total_tokens'],0)
                self.assertGreater(r['observed_wall_seconds'],0)

    def test_matched_actual_qwen_direct_vs_selector_lower_bound_not_complete_cost(self):
        stored=self.report['alternative_architecture_cost']
        self.assertEqual(stored,self.costs)
        self.assertEqual(stored['paired_original_model_task_units'],6)
        self.assertEqual((stored['full_direct_correct_against_researcher_key'],stored['full_direct_incorrect_against_researcher_key']),(5,1))
        self.assertEqual((stored['completed_direct_tokens'],stored['selector_stage1_alone_tokens']),(5593,5600))
        self.assertAlmostEqual(stored['completed_direct_wall_seconds'],20.874356,places=6)
        self.assertAlmostEqual(stored['selector_stage1_alone_wall_seconds'],21.826964,places=6)
        self.assertEqual(stored['selector_stage1_excess_tokens_before_any_answer'],7)
        self.assertAlmostEqual(stored['selector_stage1_excess_seconds_before_any_answer'],0.952608,places=6)
        self.assertEqual(stored['selector_stage1_already_slower_paired_tasks'],5)
        self.assertEqual(stored['max_possible_improvement_against_researcher_key_on_six_paired_direct_cases'],1)
        self.assertFalse(stored['staged_total_cost_available'])
        self.assertFalse(stored['staged_correctness_available'])

    def test_distinct_oracle_source_trials_not_misrepresented_as_completed_chain(self):
        s=self.report['alternative_architecture_cost']
        self.assertEqual(s['separately_completed_direct_plus_researcher_privileged_answer_pairs'],2)
        self.assertEqual(s['selector_stage1_plus_separately_completed_oracle_answer_pairs'],2)
        self.assertEqual(s['selector_plus_oracle_answer_pairs_with_exact_same_original_selected_source_IDS'],1)
        self.assertEqual(s['full_direct_plus_selector_stage1_plus_oracle_answer_triples'],0)
        self.assertEqual(s['genuinely_completed_model_selected_stage2_answers'],0)
        self.assertIn('different trial',s['critical_contraint'])

    def test_current_passive_nitro_state_not_falsely_generalized_to_thor(self):
        saved=self.report['runtime_passive_snapshot']
        self.assertEqual(saved['sqlite_open_mode'],'file:...mode=ro')
        self.assertIn('Nitro only',saved['source_host'])
        self.assertEqual(saved['current_nitro_original_SWAAG_SQLite_counts_by_table']['history.sqlite3']['counts']['events'],0)
        self.assertEqual(saved['current_nitro_original_SWAAG_SQLite_counts_by_table']['inference_requests.sqlite3']['counts']['inference_requests'],0)
        self.assertEqual(saved['current_nitro_original_SWAAG_SQLite_counts_by_table']['delegated_tools.sqlite3']['counts']['delegated_tool_calls'],0)
        self.assertEqual(saved['current_nitro_operations_log_event_kind_counts_only'],{'startup':15,'shutdown':13})
        self.assertFalse(saved['current_local_readonly_listening_port_snapshot']['14833'])
        self.assertFalse(saved['current_local_readonly_listening_port_snapshot']['14835'])
        self.assertTrue(saved['current_local_readonly_listening_port_snapshot']['19333'])
        self.assertTrue(any('SmolVLM2' in s for s in saved['confirmed_non_qwen_listener_port_19333_model_basenames']))
        self.assertGreater(saved['gpu_memory_used_MiB_passive_read'],0)
        self.assertGreater(saved['gpu_memory_total_MiB_passive_read'],saved['gpu_memory_used_MiB_passive_read'])
        self.assertTrue(saved['live_values_are_time_bounded_not_persistent_assertions'])
        datetime.fromisoformat(saved['sample_timestamp_UTC'])
        self.assertEqual(saved['GPU_queue_wait_for_archived_Qwen_calls'],'not recorded')

    def test_one_legacy_resolved_swe_task_not_a_matched_reviewer_study(self):
        x=self.report['runtime_passive_snapshot']['legacy_source_separate_single_swe_agent_outcome']
        self.assertTrue(x['available'])
        self.assertEqual(x['total_original_task_instances'],1)
        self.assertEqual(x['completed_instances'],1)
        self.assertEqual(x['resolved_instances'],1)
        self.assertFalse(x['has_matched_direct_and_reviewer_arms'])
        self.assertTrue(x['is_swe_agent_single_legacy_job_not_swaag_matched_qwen'])

    def test_every_original_question_acceptance_gate_stays_honest(self):
        answers=self.report['original_question_acceptance']
        self.assertTrue(answers['role_conditioned_relevance_direction_supported'])
        self.assertTrue(answers['one_user_initial_call_is_current_operational_default'])
        self.assertFalse(answers['automatic_correct_role_selector_verified'])
        self.assertFalse(answers['numeric_better_than_text_or_vice_versa_proven'])
        self.assertFalse(answers['decomposed_model_stage2_full_final_accuracy_proven'])
        self.assertFalse(answers['Qwen_peer_review_reliability_proven'])
        self.assertFalse(answers['Qwen_syntax_aux_training_gain_proven'])
        self.assertFalse(self.report['comparison_readiness_gate']['safe_to_claim_winning_live_Qwen_architecture'])
        self.assertFalse(self.report['Qwen_reviewer_correction_harm_confusion_available'])
        self.assertFalse(self.report['current_Qwen_training_syntax_aux_matched_FLOP_available'])
        self.assertFalse(self.report['numeric_vs_word_rating_independent_human_gold_available'])
        self.assertEqual(self.report['new_model_inference_requests'],0)
        self.assertFalse(self.report['previously_denied_operations_retried'])
        self.assertEqual(self.report['actual_production_SQLite_mutations'],0)
        self.assertTrue(self.report['no_model_service_unloaded_or_restarted'])

if __name__=='__main__':unittest.main()
