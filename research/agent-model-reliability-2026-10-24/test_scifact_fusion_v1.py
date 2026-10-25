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

class DerivedHeldoutEvidenceIntegrityTests(unittest.TestCase):
    """Re-evaluate full original rationale sets from frozen ID-only predictions."""
    def test_every_precomputed_evidence_hit_matches_original_gold_set(self):
        import gzip,json
        path=pathlib.Path(__file__).parent/'evidence'/'scifact-heldout-fusion'
        file=path/'heldout_train_case_predictions_v1.json.gz'
        summary=path/'heldout_train_metrics_v1.json'
        if not file.exists() or not summary.exists():
            self.skipTest('derived study results not published in this checkout')
        with gzip.open(file,'rt') as f:detail=json.load(f)
        meta=json.loads(summary.read_text())
        cases=detail['sentence_case_predictions']
        self.assertEqual(len(cases),564)
        self.assertEqual(len(detail['document_query_predictions']),505)
        self.assertEqual(meta['n_expert_annotated_claim_doc_pairs'],564)
        self.assertEqual(meta['excluded_duplicate_dev_claim_ids'],[871,1291])
        for row in cases:
            gold=[set(g) for g in row['gold_sets']]
            self.assertTrue(gold)
            n=row['sentence_count']
            for method,record in row['methods'].items():
                rank=record['top5_sentence_ids']
                self.assertEqual(len(set(rank)),len(rank))
                self.assertTrue(all(0<=v<n for v in rank))
                for k in (1,2,3,4,5):
                    expected=any(g.issubset(set(rank[:k])) for g in gold)
                    self.assertEqual(expected,record['complete_any_gold_rationale_at_k'][str(k)])
        bymethod={x['method']:x for x in meta['groups'][0]['methods']}
        for m,x in bymethod.items():
            rate=sum(c['methods'][m]['complete_any_gold_rationale_at_k']['3'] for c in cases)/len(cases)
            self.assertAlmostEqual(rate,x['complete_gold_rationale_at_k']['3'],places=5)
