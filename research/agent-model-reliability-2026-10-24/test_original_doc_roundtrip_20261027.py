"""Read-only verification of genuine Git-source → tool → persisted answer receipts.

This does not repeat the already completed isolated tool calls, execute any
model or mutate the original owner worktree. Source truth is original Git bytes.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

# Pure artifact/source verification runs with the lightweight research Python,
# without importing SWAAG, loading a model, or requiring its production venv.
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNNER=HERE/'run_original_doc_roundtrip_20261027.py'
TREE=ast.parse(RUNNER.read_text())
def constant(name):
    matches=[node.value for node in TREE.body if isinstance(node,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)]
    if len(matches)!=1:raise AssertionError('source runner missing frozen constant '+name)
    return ast.literal_eval(matches[0])
SOURCE=constant('SOURCE')
GIT_VERSIONS=constant('GIT_VERSIONS')
VERSION=constant('VERSION')
SECTION=constant('SECTION')
OLD_ANSWER=constant('OLD_ANSWER')
NEW_ANSWER=constant('NEW_ANSWER')
ABSTAIN=constant('ABSTAIN')
def sha(data):return hashlib.sha256(data).hexdigest()
def originals():
    output={}
    for label,metadata in GIT_VERSIONS.items():
        raw=subprocess.run(['git','show',metadata['commit']+':'+SOURCE],cwd=ROOT,
                           check=True,capture_output=True).stdout
        assert sha(raw)==metadata['sha256']
        assert (SECTION in raw.decode('utf8'))==metadata['has_section']
        output[label]=raw
    return output

DATA=HERE/'evidence/original-doc-roundtrip/original_git_doc_source_to_final_state_v1.json'

class RealSourceStateRoundtripEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record=json.loads(DATA.read_text())
        cls.files=originals()

    def test_verified_original_source_versions_not_synthetic_gold(self):
        self.assertEqual(self.record['version'],VERSION)
        for role,rec in self.record['git_snapshot_source_sha256'].items():
            source=subprocess.run(['git','show',rec['commit']+':'+SOURCE],cwd=ROOT,
                check=True,capture_output=True).stdout
            self.assertEqual(source,self.files[role])
            self.assertEqual(hashlib.sha256(source).hexdigest(),rec['source_sha256'])
            self.assertEqual((SECTION in source.decode('utf8')),rec['header_presence_is_ground_truth'])
            self.assertEqual(rec['source_sha256'],GIT_VERSIONS[role]['sha256'])
        self.assertNotEqual(self.files['old'],self.files['new'])
        self.assertEqual(self.record['original_repository_source'],SOURCE)
        self.assertIn('exact immutable real Git source',self.record['authoritative_gold_type'])

    def test_real_source_tool_read_identity_and_final_answer_for_both_versions(self):
        cases=self.record['source_read_to_persisted_final_and_replay']
        self.assertEqual(len(cases),2)
        by={x['version']:x for x in cases}
        self.assertEqual(set(by),{'old','new'})
        for version,case in by.items():
            self.assertEqual(case['original_source_sha256'],sha(self.files[version]))
            self.assertEqual(case['source_section_present_exact_git_gold'],GIT_VERSIONS[version]['has_section'])
            self.assertTrue(case['decoy_exists_and_same_basename'])
            self.assertEqual(case['source_path_alias'],'active/history-index-cost-2026-10-16.md')
            self.assertEqual(case['decoy_path_alias'],'historical/history-index-cost-2026-10-16.md')
            self.assertEqual(case['final_answer'],NEW_ANSWER if version=='new' else OLD_ANSWER)
            self.assertTrue(case['correct_against_original_git_source'])
            self.assertTrue(case['verified_exact_byte_source_contents'])
            self.assertTrue(case['source_full_text_handed_to_model_after_tool'])
            self.assertEqual(case['verified_true_source_reads'],1)
            self.assertEqual(case['actual_read_file_tool_called'],1)
            self.assertEqual(case['actual_read_file_tool_result'],1)
            self.assertEqual(case['model_request_contracts'],['fast_foreground_response','agent_action','agent_action'])
            self.assertEqual(case['scripted_model_requests'],3)
            self.assertEqual(case['stream_event_names'],['accepted','assistant_progress','completed'])
            self.assertGreater(case['roundtrip_wall_seconds'],0)
            for field in ('control_receipt_hash','filesystem_read_receipt_hash','tool_called_receipt_hash',
                          'tool_result_receipt_hash','turn_finished_receipt_hash'):
                self.assertEqual(len(case[field]),64)
            self.assertLess(case['control_receipt_sequence'],case['tool_called_sequence'])
            self.assertLess(case['tool_called_sequence'],case['filesystem_read_sequence'])
            self.assertLess(case['filesystem_read_sequence'],case['tool_result_sequence'])
            self.assertLess(case['tool_result_sequence'],case['turn_finished_sequence'])
        self.assertEqual(by['old']['persisted_session_id'],by['new']['persisted_session_id'])
        self.assertNotEqual(by['old']['persisted_control_id'],by['new']['persisted_control_id'])
        self.assertEqual(by['old']['original_input_user_instruction_sha256'],by['new']['original_input_user_instruction_sha256'])
        self.assertLess(by['old']['turn_finished_sequence'],by['new']['control_receipt_sequence'])
        self.assertNotEqual(by['old']['tool_result_receipt_hash'],by['new']['tool_result_receipt_hash'])

    def test_restart_replay_same_control_is_historical_not_new_source_authority(self):
        stored=self.record['saved_state_and_control_identity_checks']
        expected={
            'two_original_distinct_source_versions':True,
            'reopened_original_persisted_session_twice':True,
            'old_control_historical_answer_replayed_after_file_changed':True,
            'both_old_and_new_controls_replayed_after_second_restart':True,
            'conflicting_user_instruction_same_control_id_rejected':True,
            'source_versioned_receipts_remain_distinct':True,
            'persisted_history_hash_chain_matches_terminal_sequence':True,
            'new_model_calls_during_both_restarts_replays':0,
            'new_tool_calls_during_both_restarts_replays':0,
            'persisted_final_tool_count':2,
            'persisted_final_turn_count':2,
            'persisted_final_filesystem_read_count':2,
        }
        for key,value in expected.items():self.assertEqual(stored[key],value,key)
        self.assertEqual(len(stored['last_event_hash']),64)
        self.assertGreater(stored['last_event_sequence'],0)
        self.assertEqual(stored['identical_original_user_message_sha256'],
            self.record['source_read_to_persisted_final_and_replay'][0]['original_input_user_instruction_sha256'])

    def test_negative_controls_count_false_assertion_and_unresolved_abstentions(self):
        observed=self.record['negative_controls_no_real_source_tool']
        self.assertEqual(len(observed),4)
        by={(x['version'],x['control']):x for x in observed}
        self.assertEqual(len(by),4)
        for ver in ('old','new'):
            wrong=by[ver,'unguarded_direct_yes']
            cautious=by[ver,'cautious_no_tool']
            self.assertEqual(wrong['final_user_answer'],NEW_ANSWER)
            self.assertTrue(wrong['asserted_unverified_fact'])
            self.assertEqual(wrong['factual_assertion_wrong_against_original_source'],ver=='old')
            self.assertFalse(wrong['source_verified_task_complete'])
            self.assertEqual(wrong['model_requests_to_scripted_client'],1)
            self.assertEqual(wrong['model_request_contracts'],['fast_foreground_response'])
            self.assertEqual(cautious['final_user_answer'],ABSTAIN)
            self.assertTrue(cautious['safe_unresolved_abstention'])
            self.assertFalse(cautious['source_verified_task_complete'])
            self.assertEqual(cautious['model_requests_to_scripted_client'],2)
            self.assertEqual(cautious['model_request_contracts'],['fast_foreground_response','agent_action'])
            for row in (wrong,cautious):
                self.assertTrue(row['no_real_file_tool_called'])
                self.assertEqual(row['original_source_sha256'],sha(self.files[ver]))
                self.assertEqual(len(row['control_receipt_hash']),64)
                self.assertEqual(len(row['turn_receipt_hash']),64)

    def test_complete_end_to_end_counts_and_true_model_resource_limits(self):
        summary=self.record['summary']
        self.assertEqual(summary['true_grounded_complete_tasks'],2)
        self.assertEqual(summary['original_source_tool_reads_executed'],2)
        self.assertEqual(summary['exact_original_source_byte_matches'],2)
        self.assertEqual(summary['original_source_claim_errors_in_grounded_arm'],0)
        self.assertEqual(summary['grounded_fake_model_requests'],6)
        self.assertEqual(summary['no_tool_direct_false_asserts'],1)
        self.assertEqual(summary['no_tool_direct_source_verified_answers'],0)
        self.assertEqual(summary['no_tool_cautious_safe_unresolved_answers'],2)
        self.assertEqual(summary['no_tool_control_fake_model_requests'],6)
        self.assertEqual(self.record['real_qwen_inference_calls'],0)
        self.assertEqual(self.record['real_external_services_modified'],0)
        self.assertTrue(self.record['model_outputs_are_scripted'])
        self.assertFalse(self.record['gpu_vram_or_true_llamacpp_tokens_measured'])
        self.assertGreater(summary['grounded_prompt_whitespace_words_proxy'],0)
        self.assertGreater(summary['grounded_2_turns_wall_seconds'],0)

    def test_evidence_does_not_embed_original_document_contents_or_secrets(self):
        saved=DATA.read_text()
        # Only source-version hashes and short final excerpts are archived; not
        # original scientific/engineering documents or complete prompt bodies.
        self.assertNotIn('# History search-index reconstruction cost',saved)
        self.assertNotIn('The installed long-history investigation separated model generation',saved)
        self.assertNotIn('BEGIN PRIVATE KEY',saved)
        self.assertNotIn('refresh_token',saved)
        self.assertNotIn('client_secret',saved)
        for case in self.record['source_read_to_persisted_final_and_replay']:
            self.assertNotIn('source_text',case)
            self.assertNotIn('model_prompt',case)
            self.assertNotIn('tool_result_full_text',case)
        self.assertTrue(any('scripted' in l for l in self.record['limitations']))

if __name__=='__main__':unittest.main()
