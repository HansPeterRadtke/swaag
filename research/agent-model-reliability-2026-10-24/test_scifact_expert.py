import unittest,json,pathlib
from run_scifact_expert_v1 import validate,plan,input_messages
from prepare_expert_scifact_v1 import build

class SciFactExpertSourceTests(unittest.TestCase):
 def setUp(self):
  self.parent=pathlib.Path(__file__).parent
  self.manifest=json.loads((self.parent/'scifact_expert_manifest_v1.json').read_text())
  self.private=pathlib.Path('/data/tmp/agent-model-independent-qrels-20261025/scifact_expert_private_cases_v1.json')
 def test_frozen_manifest_labels_and_rationale_indices(self):
  m=self.manifest
  self.assertEqual(len(m['cases']),48)
  self.assertEqual(sum(x['gold']=='SUPPORT' for x in m['cases']),24)
  self.assertEqual(sum(x['gold']=='CONTRADICT' for x in m['cases']),24)
  self.assertTrue(all(x['rationale_indices'] for x in m['cases']))
  self.assertEqual(len(set(x['case_id'] for x in m['cases'])),48)
 def test_manifest_does_not_contain_abstract_or_claim_content(self):
  self.assertFalse(any('claim' in row or 'abstract' in row for row in self.manifest['cases']))
 def test_plan_produces_unique_ordered_model_and_condition_trials(self):
  if not self.private.exists():self.skipTest('private source data not bundled with public repository')
  examples=validate(self.manifest,json.loads(self.private.read_text()))
  jobs=plan(examples)
  self.assertEqual(len(jobs),384)
  self.assertEqual(len({t['trial_id'] for t in jobs}),384)
  for j in jobs[:8]:
   a=input_messages(j['case'],'direct')
   self.assertIn(j['case']['claim'],a[1]['content'])
 def test_mixed_source_mismatch_fails(self):
  if not self.private.exists():self.skipTest('private source data not bundled')
  private=json.loads(self.private.read_text());private['cases'][0]['claim']='tampered claim'
  with self.assertRaisesRegex(ValueError,'claim_sha_mismatch'):validate(self.manifest,private)
if __name__=='__main__':unittest.main()
