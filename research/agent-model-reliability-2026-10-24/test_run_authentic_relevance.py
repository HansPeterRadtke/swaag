import unittest
from pathlib import Path
import json
from run_authentic_relevance import plan,prompt,ROOT

class AuthenticTrialPlanTests(unittest.TestCase):
 def setUp(self):self.cases=json.loads((ROOT/'authentic_relevance_v1.json').read_text())['cases']
 def test_plan_is_balanced_and_source_messages_full(self):
  jobs=list(plan(self.cases))
  self.assertEqual(len(jobs),128)
  self.assertEqual(len({j['trial_id'] for j in jobs}),128)
  for role in ('spoken_status','fix_planning','coordination','identity_answer'):
   selected=[j for j in jobs if j['case']['goal_id']==role]
   self.assertEqual(len(selected),32)
   self.assertEqual(sum(j['case']['gold_score']==8 for j in selected),16)
  for j in jobs:
   messages=prompt(j['case'],j['encoding']);self.assertIn(j['case']['source_text'],messages[-1]['content'])
 def test_ordered_scale_has_same_information_both_encodings(self):
  j=list(plan(self.cases))[0]
  n=prompt(j['case'],'number')[-1]['content']
  s=prompt(j['case'],'category')[-1]['content']
  self.assertEqual(n.replace('Return the 0-to-8 integer as score.',''),s.replace('Return the exact named category as score.',''))
if __name__=='__main__':unittest.main()
