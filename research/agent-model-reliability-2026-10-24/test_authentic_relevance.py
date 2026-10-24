import hashlib
import json
from pathlib import Path
import unittest
from authentic_relevance_v1 import build, ROOT

class SourceBoundRelevance(unittest.TestCase):
 def test_grounded_original_paragraphs_unchanged(self):
  cases=build()
  self.assertEqual(len(cases),64)
  self.assertEqual(len({c['case_id'] for c in cases}),64)
  for c in cases:
   p=ROOT/c['source'];text=p.read_text(encoding='utf8')
   self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),c['source_sha256'])
   self.assertEqual(text[c['source_char_start']:c['source_char_end']],c['source_text'])
 def test_scored_only_extremes_and_role_contrasts(self):
  cases=build()
  self.assertEqual(sum(c['gold_score']==8 for c in cases),32)
  self.assertEqual(sum(c['gold_score']==0 for c in cases),32)
  same={}
  for c in cases:
   same.setdefault((c['source'],c['source_char_start']),set()).add(c['gold_score'])
  self.assertTrue(any(x=={0,8} for x in same.values()))
 def test_frozen_snapshot_matches_generator(self):
  p=Path(__file__).parent/'authentic_relevance_v1.json'
  a=json.loads(p.read_text())
  self.assertEqual(a['cases'],build())
if __name__=='__main__':unittest.main()
