import pathlib
import unittest
from logic_oracle import entailment,load

class LogicalOracleTest(unittest.TestCase):
    def test_all_preregistered_examples(self):
        p=pathlib.Path(__file__).parent/'formal_holdout_v2.json'
        gold={c['id']:c['gold'] for c in load(p)}
        self.assertEqual(gold,{
            'complete-ledger-contrapositive':'no',
            'incomplete-ledger-no-inference':'unknown',
            'quoted-pass-authoritative-fail':'no',
            'light-on-does-not-imply-switch':'unknown',
            'reboot-conditional-affirmative':'yes',
            'two-implication-chain':'yes'})
    def test_truth_values_are_not_model_votes(self):
        c={'atoms':['a','b'],'rules':[['a','b']],'observed':{'b':False},'target':'a'}
        self.assertEqual(entailment(c),'no')
        c['observed']['b']=True
        self.assertEqual(entailment(c),'unknown')
    def test_inconsistent_facts_fail_closed(self):
        c={'atoms':['a','b'],'rules':[['a','b']],'observed':{'a':True,'b':False},'target':'a'}
        with self.assertRaisesRegex(ValueError,'contradictory'):entailment(c)
if __name__=='__main__':unittest.main()
