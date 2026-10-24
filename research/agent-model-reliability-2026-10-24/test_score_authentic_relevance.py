import unittest
import json
from pathlib import Path
from score_authentic_relevance import summarize

class AuthenticScorerTests(unittest.TestCase):
 def setUp(self):
  f=Path(__file__).parent/'authentic_relevance_v1.json';self.fixture=json.loads(f.read_text());self.c=self.fixture['cases'][0]
 def request(self,encoding,model='coder'):
  c=self.c;return dict(trial_id='trial'+encoding+model,event='request_submitted',case_id=c['case_id'],
    model=model,encoding=encoding,gold=c['gold_score'],role=c['role'],goal_id=c['goal_id'],
    source_sha256=c['source_sha256'],source_char_start=c['source_char_start'],source_char_end=c['source_char_end'])
 def test_identical_case_has_two_distinct_grades(self):
  a=self.request('category');b=self.request('number')
  journal=[a,{'trial_id':a['trial_id'],'event':'response','score':8,'elapsed_seconds':.2},
           b,{'trial_id':b['trial_id'],'event':'response','score':3,'elapsed_seconds':.3}]
  s=summarize(self.fixture,journal)
  self.assertEqual(len(s['matched_encoding_pairs']),1)
  self.assertEqual((s['matched_encoding_pairs'][0]['numeric'],s['matched_encoding_pairs'][0]['category']),(3,8))
 def test_gold_change_refused(self):
  a=self.request('number');a['gold']=100
  with self.assertRaisesRegex(ValueError,'fixture_tampering'):summarize(self.fixture,[a])
if __name__=='__main__':unittest.main()
