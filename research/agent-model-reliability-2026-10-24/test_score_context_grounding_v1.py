import json,unittest
from pathlib import Path
from score_context_grounding_v1 import score

class ContextGroundingScorerTests(unittest.TestCase):
 def setUp(self):
  self.dataset=json.loads((Path(__file__).parent/'context_grounding_v1.json').read_text());self.case=self.dataset['cases'][0]
 def test_capacity_not_marked_incorrect(self):
  a=score(self.dataset,[{'trial_id':'r','event':'stopped','reason':'busy'}]);self.assertEqual(a['statuses'],{'admission_refused':1});self.assertEqual(a['valid_final_answers'],0)
 def test_source_tampering_detected(self):
  b={'trial_id':'r','event':'begin','variant':'full','model':'m','case':dict(self.case,gold_answer='invented'), 'ground_truth':'invented'}
  with self.assertRaisesRegex(ValueError,'source_or_expected_answer_changed'):score(self.dataset,[b])
if __name__=='__main__':unittest.main()
