"""Guard control regressions: exact source contracts and uncorrected negatives.

The six failures in the adversarial negatives are expected observations of
this frozen source version, not a claim that current production is repaired.
"""
import json
import unittest
from pathlib import Path
from verified_foreground_guard_v1 import evaluate, CASES, CASES_SHA256, SOURCE_SHA256

HERE=Path(__file__).resolve().parent
class FrozenGuardControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.report=evaluate()

    def test_source_and_case_frozen(self):
        self.assertEqual(self.report['source_sha256'], SOURCE_SHA256)
        self.assertEqual(self.report['frozen_cases_sha256'], CASES_SHA256)
        self.assertEqual(len(CASES),32)

    def test_anchored_contracts_and_adversarial_failures_distinct(self):
        self.assertEqual(self.report['summary']['anchored']['gate_correct'],24)
        self.assertEqual(self.report['summary']['anchored']['gate_false_accept'],0)
        self.assertEqual(self.report['summary']['heldout_negative']['gate_false_accept'],6)
        self.assertEqual(self.report['summary']['heldout_negative']['gate_false_reject'],0)
        self.assertEqual(self.report['summary']['heldout_negative']['gate_correct'],2)

    def test_zero_llm_budget_and_non_deceptive_provenance(self):
        self.assertEqual(self.report['full_model_calls'],0)
        self.assertIn('not model generations',self.report['model_output_provenance'])
        self.assertIn('NOT a completed LLM strategy',self.report['limitations'][0])

    def test_frozen_saved_observation_matches_case_verdicts(self):
        saved=json.loads((HERE/'evidence/foreground-verified-guard/frozen_same_budget_v1.json').read_text())
        self.assertEqual(saved['frozen_cases_sha256'],CASES_SHA256)
        old={x['id']:(x['gold_unchanged'],x['gate_unchanged']) for x in saved['cases']}
        new={x['id']:(x['gold_unchanged'],x['gate_unchanged']) for x in self.report['cases']}
        self.assertEqual(old,new)

if __name__=='__main__': unittest.main()
