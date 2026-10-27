"""Source-immutable read-only audit of historical isolated replay receipts."""
import json
import unittest
from pathlib import Path
from audit_final_duplicate_effect_receipts_20261027 import audit,HERE

FILE=HERE/'evidence/final-qwen-readiness/pre_0600_isolated_duplicate_effect_audit_v1.json'

class FinalDuplicateEffectsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved=json.loads(FILE.read_text())
        cls.fresh=audit()

    def test_independent_counts_from_immutable_original_journal_cases(self):
        self.assertEqual(self.saved,self.fresh)
        self.assertEqual(self.saved['sum_distinct_final_case_receipts'],82)
        self.assertEqual(self.saved['sum_verified_idempotent_replays'],82)
        self.assertEqual(self.saved['sum_sandbox_tool_results'],14)
        records=self.saved['inspected_studies']
        self.assertEqual({k:v['distinct_finished_final_case_receipts'] for k,v in records.items()},
                         {'foreground_guard':46,'source_policy':12,'independent_last_mile':24})
        self.assertEqual({k:v['sandbox_tool_result_receipts'] for k,v in records.items()},
                         {'foreground_guard':2,'source_policy':0,'independent_last_mile':12})
        self.assertTrue(all(v['technical_failed_case_receipts']==0 for v in records.values()))
        self.assertTrue(all(v['tracked_at_original_git_head'] for v in records.values()))

    def test_scope_limits_avoid_production_or_semantic_accuracy_claim(self):
        limitations=' '.join(self.saved['limits'])
        self.assertIn('not proof of absence of duplicate effects in live owner production services',limitations)
        self.assertIn('Original Qwen stage-2 final responses',limitations)
        self.assertIn('every original ephemeral SQLite payload',limitations)
        self.assertIn('did not read/write production SQLite',limitations)

if __name__=='__main__':unittest.main()
