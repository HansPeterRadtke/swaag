"""Audit the actual FINAL SWAAG answer/turn/tool/control receipts, not JSON drafts.

Pure verification of already captured isolation-session outcomes. Never starts
production SWAAG, calls Qwen, or reruns uncertain tool effects.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EVIDENCE=HERE/'evidence/final-foreground-receipts'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalUserAnswerReceiptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=json.loads((EVIDENCE/'study_summary_v1.json').read_text())
        cls.rows=json.loads((EVIDENCE/'source_full_user_receipts_v1.json').read_text())
        cls.journal=[json.loads(x) for x in (EVIDENCE/'isolated_final_receipts_attempt02.jsonl').read_text().splitlines()]

    def test_source_version_and_provenance_are_not_conflated_with_real_qwen(self):
        summary=self.summary
        self.assertEqual(summary['original_guard_sha256'],digest(ROOT/'src/swaag/foreground_reply_evidence.py'))
        self.assertEqual(summary['frozen_cases_sha256'],
                         '18f11f8fa3fa38087a489c1caa9c7be19b6282ae34be9957990ca287a6e31ba8')
        for name,sha in summary['source_test_hashes'].items():self.assertEqual(sha,digest(ROOT/name))
        self.assertEqual(summary['llm_inference_completed'],0)
        self.assertFalse(summary['change_to_production_source'])
        self.assertTrue(summary['model_tasks_are_scripted'])
        self.assertEqual(len(summary['cases']),23)

    def test_append_only_all_case_attempts_and_no_double_completion(self):
        starts=[x for x in self.journal if x['event']=='case_begin']
        dones=[x for x in self.journal if x['event']=='case_completed']
        errors=[x for x in self.journal if x['event']=='case_failed']
        self.assertEqual(len(starts),46)
        self.assertEqual(len(dones)+len(errors),46)
        self.assertEqual(len({(x['id'],x['variant']) for x in starts}),46)
        self.assertEqual(len(self.rows),len(dones))
        self.assertEqual(len(errors),len(self.summary['runtime_failures_not_semantic_errors']))
        self.assertEqual(self.journal[-1]['event'],'run_complete')
        self.assertEqual(self.journal[-1]['completed_cells'],len(self.rows))
        for x in errors:self.assertTrue(x['failure']['never_count_as_semantic_incorrect'])

    def test_final_user_visible_answer_has_persisted_control_and_turn_receipts(self):
        for row in self.rows:
            self.assertTrue(row['final_user_answer'])
            self.assertTrue(row['durable_turn_finished_sequence']>row['durable_control_message_processed_sequence'])
            self.assertEqual(len(row['durable_turn_finished_hash']),64)
            self.assertEqual(len(row['durable_control_message_processed_hash']),64)
            self.assertEqual(row['completed_stream_event_kinds'][0],'accepted')
            self.assertEqual(row['completed_stream_event_kinds'][-1],'completed')
            self.assertEqual(row['model_contracts'][0],'fast_foreground_response')
            self.assertEqual(row['number_of_model_requests'],len(row['model_contracts']))
            self.assertTrue(row['replayed_no_duplicate_model_or_tool_request'])
            self.assertEqual(row['test_isolated_model_backend_url'],'http://127.0.0.1:9999')
            self.assertTrue(row['fake_model_only_no_actual_llm'])

    def test_concrete_calculator_effect_cannot_be_earned_by_fake_user_text(self):
        for row in self.rows:
            if row['real_calculator_tool_called'] or row['real_calculator_tool_result_count']:
                self.assertEqual((row['real_calculator_tool_called'],row['real_calculator_tool_result_count']),(1,1))
                self.assertTrue(row['verified_calculator_tool_receipt'])
                self.assertEqual(len(row['real_tool_receipt_sequences']),2)
                self.assertEqual(row['model_contracts'],['fast_foreground_response','agent_action','agent_action'])
                self.assertEqual(row['final_user_answer'],'The calculator returned 42.')
                self.assertTrue(row['final_task_verified_complete'])
            if row['task']=='calculator_required' and row['variant']=='baseline':
                # The baseline may naively publish text claiming success, but
                # zero actual tool receipts cannot count as task completion.
                self.assertFalse(row['verified_calculator_tool_receipt'])
                self.assertFalse(row['final_task_verified_complete'])

    def test_unsafe_and_false_reject_metrics_exact_from_completed_outputs(self):
        counts=self.summary['result']
        for variant in ('baseline','candidate'):
            group=[x for x in self.rows if x['variant']==variant]
            source=counts[variant]
            self.assertEqual(source['completed_cases'],len(group))
            self.assertEqual(source['unknown_due_to_runtime_or_streaming_error'],23-len(group))
            self.assertEqual(source['unsafe_final_false_accepts_in_observed'],
                             sum(x['unsafe_final_false_accept'] for x in group))
            self.assertEqual(source['safe_direct_false_rejects_in_observed'],
                             sum(x['safe_initial_reply_unnecessarily_escalated'] for x in group))
            self.assertEqual(source['safe_final_answers_in_observed'],
                             sum(x['safe_final_answer'] for x in group))
            self.assertEqual(source['verified_completed_tasks_in_observed'],
                             sum(x['final_task_verified_complete'] for x in group))
            self.assertEqual(source['model_request_total_observed'],
                             sum(x['number_of_model_requests'] for x in group))
        p=counts['paired_complete']
        self.assertEqual(p['count']+p['unknown_pair_count'],23)
        self.assertEqual(len(set(p['case_ids'])),p['count'])

    def test_scripted_safety_labels_separated_from_genuine_model_evaluation(self):
        self.assertIn('not independent blinded human gold',self.summary['adjudication'])
        self.assertTrue(any('natural Qwen responses' in item for item in self.summary['limitations']))
        for row in self.rows:
            if row['unsafe_final_false_accept']:
                self.assertFalse(row['gold_direct_allowed'])
                self.assertTrue(row['final_answer_matches_initial_proposal'])
                self.assertFalse(row['final_task_verified_complete'])
            if row['safe_initial_reply_unnecessarily_escalated']:
                self.assertTrue(row['gold_direct_allowed'])
                self.assertFalse(row['final_answer_matches_initial_proposal'])

if __name__=='__main__':unittest.main()
