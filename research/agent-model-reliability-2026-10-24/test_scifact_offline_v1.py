"""Check source, exact evidence-set rules and ranking sanity, not model quality."""
import pathlib
import unittest
from evaluate_scifact_offline_v1 import BM25, rationale_complete, ordered, read_scifact


class SourceEvidenceRetrievalTests(unittest.TestCase):
 def test_multiple_alternative_gold_evidence_sets(self):
  gold=[frozenset({1,2}),frozenset({5})]
  self.assertFalse(rationale_complete(gold,[1,3]))
  self.assertTrue(rationale_complete(gold,[1,2,3]))
  self.assertTrue(rationale_complete(gold,[5]))
  self.assertFalse(rationale_complete(gold,[]))

 def test_bm25_term_hit_beats_unrelated_source(self):
  bm=BM25(['alpha beta gamma', 'zeta zeta rho', 'beta theta phi'])
  ranks=ordered(bm.score('alpha gamma'))
  self.assertEqual(ranks[0],0)

 def test_ties_are_source_position_stable(self):
  self.assertEqual(ordered([0.0,0.0,1.0,0.0]),[2,0,1,3])

 def test_real_source_has_distinct_evidence_groups(self):
  archive=pathlib.Path('/data/tmp/agent-model-independent-qrels-20261025/scifact_original.tar.gz')
  if not archive.exists():self.skipTest('exact private source archive unavailable')
  corpus,claims,pairs,gold_docs,labels,raw_rationales=read_scifact(archive)
  self.assertEqual((len(corpus),len(claims)),(5183,300))
  self.assertTrue(100<len(pairs)<=raw_rationales)
  self.assertTrue(all(p['gold_sets'] and p['claim_id'] in gold_docs for p in pairs))
  self.assertEqual(set(labels),{'SUPPORT','CONTRADICT'})

if __name__=='__main__':unittest.main()
