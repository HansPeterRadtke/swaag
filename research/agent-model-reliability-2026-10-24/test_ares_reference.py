import json,hashlib,collections,unittest
from pathlib import Path
from run_ares_reference import plan

class AresReferenceTests(unittest.TestCase):
 def setUp(self):
  self.dataset=json.loads((Path(__file__).parent/'ares_nq_reference_v1.json').read_text());self.cases=self.dataset['items']
 def test_bounded_original_external_reference_data(self):
  self.assertEqual(len(self.cases),48)
  self.assertEqual(collections.Counter(x['label'] for x in self.cases),{0:24,1:24})
  self.assertEqual(len(set(x['query'] for x in self.cases)),24)
  self.assertEqual(len(set(x['item_id'] for x in self.cases)),48)
  self.assertEqual(len(plan(self.cases)),192)
  for x in self.cases:
   self.assertEqual(hashlib.sha256(x['document'].encode()).hexdigest(),x['passage_sha256'])
   self.assertGreaterEqual(len(x['document']),120)
   self.assertLessEqual(len(x['document']),1350)
 def test_no_human_claim(self):
  self.assertIn('do not claim',self.dataset['label_provenance_caution'])
if __name__=='__main__':unittest.main()
