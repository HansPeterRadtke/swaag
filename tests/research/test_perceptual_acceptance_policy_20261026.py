import unittest
import importlib.util
from pathlib import Path
import sys
_spec = importlib.util.spec_from_file_location("acceptance_policy_under_test", Path(__file__).resolve().parents[2] / "src/swaag/perception/acceptance_policy.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
Observation, Verdict, decide_property = _mod.Observation, _mod.Verdict, _mod.decide_property


class PerceptualAcceptancePolicyTests(unittest.TestCase):
    def item(self, outcome, provenance='verified_source', **kwargs):
        return Observation('rack_bar_contact', 'triangle_contact_v1', outcome, provenance, **kwargs)

    def test_source_contact_violation_is_auto_fail_only_with_complete_verification(self):
        verified = dict(reference_verified=True, actual_output_verified=True, measured_property=True)
        self.assertEqual(decide_property([self.item('fail', **verified)]), Verdict.FAIL)
        for omitted in verified:
            missing = dict(verified)
            missing[omitted] = False
            self.assertEqual(decide_property([self.item('fail', **missing)]), Verdict.REVIEW)

    def test_visual_model_predictions_cannot_auto_fail_or_auto_pass(self):
        all_yes = dict(reference_verified=True, actual_output_verified=True, measured_property=True)
        for outcome in ('pass', 'fail'):
            for provenance in ('visual_proposal', 'unverified_source'):
                self.assertEqual(decide_property([self.item(outcome, provenance, **all_yes)]), Verdict.REVIEW)

    def test_conflicting_source_and_verified_human_observation_abstains(self):
        verified = dict(reference_verified=True, actual_output_verified=True, measured_property=True)
        self.assertEqual(decide_property([self.item('fail', **verified), self.item('pass', 'human_reference', **verified)]), Verdict.REVIEW)

    def test_trusted_pass_does_not_override_unverified_fail(self):
        verified = dict(reference_verified=True, actual_output_verified=True, measured_property=True)
        self.assertEqual(decide_property([self.item('pass', **verified), self.item('fail', 'visual_proposal')]), Verdict.PASS)

    def test_missing_and_unknown_are_review_not_pass(self):
        self.assertEqual(decide_property([]), Verdict.REVIEW)
        self.assertEqual(decide_property([self.item('inconclusive')]), Verdict.REVIEW)
        with self.assertRaises(ValueError):
            decide_property([self.item('fail'), Observation('arrow_direction', 'ocr', 'pass', 'verified_source')])

if __name__ == '__main__':
    unittest.main()
