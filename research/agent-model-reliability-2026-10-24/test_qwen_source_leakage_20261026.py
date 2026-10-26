"""Reproducible read-only controls for already executed, never replayed calls."""
import hashlib
import json
import unittest
from pathlib import Path
from audit_qwen_source_leakage_20261026 import audit, bm25_ids, dataset, HERE
from prepare_opaque_context_v2_20261026 import create

RESULT=HERE/'evidence/source-leakage/archived_qwen_source_and_cost_v1.json'
MODEL_INPUT=HERE/'evidence/source-leakage/source_opaque_model_input_v2.json'
RESEARCHER_LABELS=HERE/'evidence/source-leakage/source_opaque_researcher_labels_v2.json'

class QwenSourceLeakageAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=audit()
        cls.source=json.loads(RESULT.read_text())

    def test_archive_deduplicated_without_rerunning_or_relabeling(self):
        s=self.out['summary']
        self.assertEqual([s['planned'],s['observed_trial_ids'],s['original_event_records'],
                          s['unique_events'],s['byte_identical_overlaps']],[48,31,150,94,56])
        self.assertEqual(s['unchanged_source_passage_spans_verified'],40)
        self.assertEqual(s['not_attempted_any_recorded_call'],17)
        self.assertEqual(s['admission_refused_before_model_call'],10)
        self.assertEqual(s['completed_model_source_selection_first_stage_only'],8)
        self.assertEqual(s['completed_real_two_call_selection_then_answer'],0)

    def test_actual_qwen_selection_vs_label_leakage_negative_control(self):
        a=self.out['selection_confusions']
        self.assertEqual([a['8_existing_Qwen_first_stage_calls'][k] for k in ('TP','FP','FN','TN')],
                         [16,2,0,22])
        self.assertEqual([a['zero_inference_pid_constant_selector'][k] for k in ('TP','FP','FN','TN')],
                         [16,0,0,24])
        self.assertEqual([a['pid_blind_BM25_same_selected_passage_counts'][k] for k in ('TP','FP','FN','TN')],
                         [14,4,2,20])
        self.assertEqual(self.out['summary']['authored_support_label_identity'],'r1,r2 for every case')

    def test_full_call_vs_first_stage_matched_lower_bound(self):
        c=self.out['costs']
        self.assertEqual(c['first_selector_calls'],8)
        self.assertEqual(c['all_first_selector_evaluated_tokens'],7669)
        self.assertEqual(c['same_case_model_paired_full_vs_selector_first_stage_count'],6)
        self.assertEqual((c['paired_full_direct_evaluated_tokens'],c['paired_first_selector_evaluated_tokens']),(5593,5600))
        self.assertEqual(c['paired_selector_stage_already_slower_count'],5)
        self.assertFalse(c['reliable_total_model_select_final_cost_available'])
        self.assertEqual(c['llm_model_requests_in_this_audit'],0)

    def test_missing_cells_never_scored_as_failures(self):
        s=self.out['summary']
        self.assertEqual((s['completed_full_context_direct'],s['completed_researcher_oracle_gold_selection']),(9,4))
        self.assertEqual((s['direct_source_answer_author_anchor_correct'],s['direct_source_answer_author_anchor_incorrect']),(8,1))
        self.assertEqual(s['missing_full_direct_cells'],7)
        self.assertEqual((s['worst_case_correct_full_direct_of_16_if_all_missing_wrong'],
                          s['best_case_correct_full_direct_of_16_if_all_missing_correct']),(8,15))
        self.assertEqual(s['full_direct_by_model'],{
            'qwen3-coder':{'observed':5,'matched_researcher_answer':4},
            'qwen38':{'observed':4,'matched_researcher_answer':4},
        })

    def test_saved_result_matches_reanalysis_not_timing_guess(self):
        for k in ['version','source_checksums','original_project_source_checksums','summary',
                  'selection_confusions','costs','matched_direct_vs_first_selector',
                  'source_selector_records','direct_final_records','unfinished_selector_records']:
            self.assertEqual(self.source[k],self.out[k])

    def test_opaque_v2_preserves_original_source_but_no_labels_in_model_input(self):
        public,gold=create(); self.assertEqual(public,json.loads(MODEL_INPUT.read_text()))
        self.assertEqual(gold,json.loads(RESEARCHER_LABELS.read_text()))
        original,_=dataset()
        self.assertEqual(len(public['input_cases']),len(original))
        self.assertEqual(len(gold['labels']),len(original))
        self.assertNotIn('r1',MODEL_INPUT.read_text())
        self.assertNotIn('r2',MODEL_INPUT.read_text())
        for inp,ref,lab in zip(public['input_cases'],original,gold['labels']):
            self.assertEqual((inp['question'],inp['answer_options']),(ref['question'],ref['answer_options']))
            self.assertFalse(set(inp)&{'gold_answer','support_ids','source_basis'})
            self.assertEqual(len(inp['passages']),len(ref['passages']))
            self.assertTrue(all(k['source_id'].startswith('s_') for k in inp['passages']))
            self.assertTrue(all(not k['source_id'].startswith(('r','d')) for k in inp['passages']))
            self.assertEqual([(p['file'],p['start'],p['end'],p['text']) for p in inp['passages']],
                             [(p['file'],p['start'],p['end'],p['text']) for p in ref['passages']])
            v1_select=bm25_ids(ref['question'],ref['passages'],2)
            v2_select=bm25_ids(inp['question'],[
                {**p,'pid':p['source_id']} for p in inp['passages']],2)
            mapping=lab['original_source_label_mapping_for_audit_only']
            self.assertEqual([mapping[k] for k in v1_select],v2_select)
        self.assertEqual(hashlib.sha256(MODEL_INPUT.read_bytes()).hexdigest(),
                         'b8aaa4a567a14798338667173a3a0f8a319b9e8c63765f3db41624b9fa5363d8')
        self.assertEqual(hashlib.sha256(RESEARCHER_LABELS.read_bytes()).hexdigest(),
                         'a475a412615fffd85a2bd0dfab64d1d79de6643fc302b4640baf1ed219e0496a')

if __name__=='__main__':unittest.main()
