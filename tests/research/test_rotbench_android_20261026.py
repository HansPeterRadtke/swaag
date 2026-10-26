"""Offline integrity and falsification tests for actual external Android images."""
import json
from pathlib import Path
import unittest

DATA=Path(__file__).resolve().parents[2]/'docs/research/perceptual-rotbench-android-heldout-2026-10-26.json'


class TestRotBenchPublicEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=json.loads(DATA.read_text())
        cls.rows=cls.d['rows']

    def test_external_source_gold_and_independent_holdout_sample(self):
        d=self.d
        self.assertEqual(d['sample']['image_pairs'],100)
        self.assertEqual(d['sample']['unique_android_apps'],39)
        self.assertEqual(d['sample']['downloaded_distinct_original_PNGs'],262)
        self.assertEqual(len(set(x['id'] for x in self.rows)),100)
        self.assertEqual(sum(bool(x['source_author_bug']) for x in self.rows),50)
        self.assertEqual(sum(x['group']=='mutated-layout-clip' for x in self.rows),25)
        self.assertEqual(sum(x['group']=='mutated-layout-overlap' for x in self.rows),25)
        self.assertEqual(sum(x['group']=='original-right' for x in self.rows),50)
        self.assertTrue(d['source_and_predictions_SHA256']['predictions.json'])
        self.assertTrue(d['source_and_predictions_SHA256']['sham_reference_predictions.json'])

    def test_all_four_confusions_derive_from_retained_per_case_outputs(self):
        for field in ('aligned_localized','aligned_global','mismatched_localized','mismatched_global'):
            f=[x for x in self.rows if x[field] is not None]
            actual={'TP':sum(x['source_author_bug'] and x[field] for x in f),'FP':sum((not x['source_author_bug']) and x[field] for x in f),'FN':sum(x['source_author_bug'] and (not x[field]) for x in f),'TN':sum((not x['source_author_bug']) and (not x[field]) for x in f),'abstained':len(self.rows)-len(f)}
            self.assertEqual(self.d['confusion'][field],actual)
        self.assertEqual(self.d['confusion']['aligned_localized'],dict(TP=39,FP=0,FN=11,TN=50,abstained=0))
        self.assertEqual(self.d['confusion']['aligned_global'],dict(TP=3,FP=0,FN=47,TN=50,abstained=0))
        self.assertEqual(self.d['confusion']['mismatched_localized'],dict(TP=50,FP=50,FN=0,TN=0,abstained=0))

    def test_real_missed_clip_change_is_small_not_empty(self):
        misses=[x for x in self.rows if x['source_author_bug'] and not x['aligned_localized']]
        self.assertEqual(len(misses),11)
        self.assertTrue(all(x['group']=='mutated-layout-clip' for x in misses))
        self.assertTrue(all(0<x['changed_fraction_max']<=.0005 for x in misses))
        self.assertTrue(all(not all(x['reference_image_identical']) for x in misses))
        self.assertEqual(sum(x['aligned_localized'] for x in self.rows if x['group']=='mutated-layout-overlap'),25)
        self.assertEqual(sum(x['aligned_localized'] for x in self.rows if x['group']=='mutated-layout-clip'),14)

    def test_easy_controls_are_exact_same_bytes_not_dynamic_true_negatives(self):
        normals=[x for x in self.rows if x['group']=='original-right']
        self.assertEqual(len(normals),50)
        self.assertTrue(all(all(x['reference_image_identical']) for x in normals))
        self.assertTrue(all(x['changed_fraction_max']==0 for x in normals))
        self.assertTrue(all(not x['aligned_localized'] and x['mismatched_localized'] for x in normals))
        self.assertIn('not in git',self.d['copyright_and_license'].lower())
        self.assertIn('not new human-adjudicated',self.d['source_gold_scope'])
        self.assertTrue(any('legitimate changing' in t for t in self.d['scientific_limits']))

    def test_cost_captures_shared_both_detector_budget(self):
        cost=self.d['cost']
        self.assertFalse(cost['GPU_used'])
        self.assertGreater(cost['primary_and_global_shared_native_pixel_compute_seconds'],0)
        self.assertGreater(cost['sham_identical_operations_pixel_compute_seconds'],0)


if __name__=='__main__':unittest.main()
