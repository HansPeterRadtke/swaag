"""Verify frozen toy evidence mechanically without retraining or importing torch."""
import json
import pathlib
import unittest

DIR=pathlib.Path(__file__).parent/'evidence'/'grammar-toy'

class GrammarEvidenceTests(unittest.TestCase):
    def test_all_arms_equal_optimizer_steps_and_parameter_count(self):
        a=json.loads((DIR/'grammar_aux_synthetic_results_20261024.json').read_text())
        r=a['results'];self.assertEqual(len(r),9)
        self.assertEqual({x['seed'] for x in r},{11,23,41})
        self.assertEqual({x['mode'] for x in r},{'syntax','plain','shuffled'})
        self.assertEqual({x['updates'] for x in r},{30})
        self.assertEqual({x['architecture_parameters'] for x in r},{6110})
        self.assertEqual({x['total_tokens_seen'] for x in r},{6720})
        self.assertEqual(a['splits'],{'train':192,'validation':48,'test':60})
    def test_no_synthetic_agreement_gain_was_observed(self):
        a=json.loads((DIR/'grammar_aux_synthetic_results_20261024.json').read_text())
        for m in ('plain','syntax','shuffled'):
            self.assertEqual(a['summary'][m]['verb_number_accuracy']['values'],[.5,.5,.5])
    def test_initial_error_remains_visible(self):
        msg=(DIR/'grammar_aux_run_failed_initial.txt').read_text()
        self.assertIn('224 versus 192',msg)
        self.assertIn('invalid setup',msg)
if __name__=='__main__': unittest.main()
