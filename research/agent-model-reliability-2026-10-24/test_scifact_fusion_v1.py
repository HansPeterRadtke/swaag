import pathlib
import unittest
from evaluate_scifact_independent_fusion_v1 import (
    alternate, reciprocal_rank_fusion, source, METHODS, DOCUMENT_METHODS
)

class FrozenFusionTests(unittest.TestCase):
    def test_rank_fusion_disagreement_not_counted_as_truth(self):
        r=reciprocal_rank_fusion([[0,1,2,3],[3,2,1,0]])
        self.assertEqual(sorted(r),[0,1,2,3])
        self.assertEqual(len(set(r)),4)
    def test_reversed_single_rank_is_unchanged(self):
        self.assertEqual(reciprocal_rank_fusion([[3,2,1,0]]),[3,2,1,0])
    def test_alternate_two_rankers(self):
        self.assertEqual(alternate([0,1,2,3],[2,3,0,1]),[0,2,1,3])
    def test_rank_fusion_invalid_duplicate_refused(self):
        with self.assertRaisesRegex(ValueError,'duplicates'):
            reciprocal_rank_fusion([[0,1,1]])
    def test_holdout_uses_source_train_not_dev(self):
        path=pathlib.Path('/data/tmp/agent-model-independent-qrels-20261025/scifact_original.tar.gz')
        if not path.exists():self.skipTest('private original source unavailable')
        c,claims,excl=source(path)
        self.assertEqual(len(c),5183)
        self.assertEqual(len(claims),807)
        self.assertEqual(len(excl),2)
        self.assertIn('rrf_lexical',METHODS)
        self.assertIn('rrf_lexical',DOCUMENT_METHODS)

if __name__=='__main__':unittest.main()
