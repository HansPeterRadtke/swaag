"""Read-only source-expert final SWAAG user-visible answer receipts audit."""
import hashlib,json,unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DIR=HERE/'evidence/source-authority-policy'
POLICY_RESULT_SHA='9abb684833e3cd1ea76998c8ae049b7843c07c7025b3c2dd8bf383f9cde279a0'
SELECTED={(56,'4709641','SUPPORT'),(57,'4709641','CONTRADICT'),
          (1029,'13923140','CONTRADICT'),(1379,'16322674','SUPPORT'),
          (903,'10648422','CONTRADICT'),(179,'17450673','SUPPORT')}

class SourceAuthorityFinalUserBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=json.loads((DIR/'swaag_final_user_bridge_summary_v1.json').read_text())
        cls.journal=[json.loads(s) for s in (DIR/'swaag_final_user_bridge_receipts_v1.jsonl').read_text().splitlines()]
        cls.results=[r['result'] for r in cls.journal if r['event']=='isolated_case_completed']

    def test_source_and_gold_scopes_are_original_and_not_confused(self):
        self.assertEqual(hashlib.sha256((DIR/'final_verdict_and_cost_ledger_v1.json').read_bytes()).hexdigest(),POLICY_RESULT_SHA)
        self.assertEqual(self.summary['source_score_file_sha256'],POLICY_RESULT_SHA)
        self.assertEqual({(x['source_claim_id'],x['source_paper_id'],x['independent_expert_gold']) for x in self.results},SELECTED)
        self.assertEqual(self.summary['paired_source_cases'],6)
        self.assertEqual(self.summary['true_negative_CONTRADICT_source_cases'],3)
        self.assertEqual(self.summary['true_positive_SUPPORT_source_cases'],3)
        self.assertEqual({r['source_paper_id'] for r in self.results if r['source_claim_id'] in (56,57)},{'4709641'})
        self.assertEqual({r['independent_expert_gold'] for r in self.results if r['source_claim_id'] in (56,57)},{'SUPPORT','CONTRADICT'})

    def test_all_final_replies_and_receipts_complete_on_real_service(self):
        self.assertEqual(len(self.results),12)
        begins=[r for r in self.journal if r['event']=='isolated_case_begin']
        self.assertEqual(len(begins),12)
        self.assertEqual(len({(x['id'],x['paper'],x['policy']) for x in begins}),12)
        self.assertEqual(self.journal[-1]['event'],'all_complete')
        self.assertEqual(self.journal[-1]['final_trajectories'],12)
        for x in self.results:
            self.assertEqual(x['fake_model_contracts'],['fast_foreground_response'])
            self.assertEqual(x['fake_model_requests'],1)
            self.assertEqual(x['actual_model_qwen_inference_requests'],0)
            self.assertEqual(x['tool_effects_count'],0)
            self.assertTrue(x['turn_finished_receipt_seq']>x['control_processed_receipt_seq'])
            self.assertEqual(len(x['turn_finished_receipt_hash']),64)
            self.assertEqual(len(x['control_processed_receipt_hash']),64)
            self.assertTrue(x['reconnect_idempotent'])
            self.assertTrue(x['original_source_text_not_exported'])
            self.assertEqual(len(x['original_user_message_sha256']),64)
            self.assertNotIn('original_scientific_claim',x)
            self.assertIn('specific claim',x['final_user_visible_reply'])

    def test_review_agreement_cannot_override_expert_original_contradiction(self):
        by={(x['source_claim_id'],x['source_paper_id'],x['policy']):x for x in self.results}
        for cid,docid,gold in SELECTED:
            a=by[cid,docid,'one_word_top3'];b=by[cid,docid,'two_consensus_top3']
            self.assertEqual(a['final_source_scoped_verdict'],b['final_source_scoped_verdict'])
            self.assertEqual(a['correct_against_external_expert_source'],b['correct_against_external_expert_source'])
        for name in ('one_word_top3','two_consensus_top3'):
            r=self.summary['policy_results'][name]
            self.assertEqual((r['right_terminal_final_user_replies'],r['wrong_terminal_final_user_replies']),(1,5))
            self.assertEqual(r['n'],6)
            self.assertEqual(r['model_calls_to_mock'],6)
            self.assertEqual(r['actual_qwen_calls'],0)
            self.assertEqual(r['authentic_tool_receipts'],0)
            self.assertEqual(r['source_complete_context_cases'],5)
            self.assertGreater(r['cpu_swaag_harness_end_to_end_wall_milliseconds'],0)

    def test_source_text_and_real_remote_secrets_never_saved(self):
        for record in self.journal:
            self.assertNotIn('claim_text',record)
            self.assertNotIn('abstract',record)
            self.assertNotIn('source_text',record)
        self.assertTrue(self.summary['claim_paper_selected_after_model_outcomes'])
        self.assertIn('scripted FakeModelClient',self.summary['inference_backend'])
        self.assertTrue(any('not representative' in s for s in self.summary['limitations']))

if __name__=='__main__':unittest.main()
