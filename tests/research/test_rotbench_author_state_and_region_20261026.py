"""Independent offline tests of preserved author-right and bbox-source error evidence."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'docs/research/perceptual-android-author-state-and-region-audit-2026-10-26.json'
OLD=ROOT/'docs/research/perceptual-rotbench-android-heldout-2026-10-26.json'
NEW=ROOT/'docs/research/perceptual-rotbench-unused-heldout-replication-2026-10-26.json'


class TestRealSourceIdentityAndAnnotationAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=json.loads(R.read_text());cls.cases=cls.d['same_app_clean_cases']

    def test_full_frozen_cross_study_source_provenance(self):
        d=self.d
        o=json.loads(OLD.read_text());n=json.loads(NEW.read_text())
        self.assertEqual(d['source_version'],o['source_pinned_revision'])
        self.assertEqual(d['source_version'],n['source_revision'])
        self.assertEqual(d['source_test_json_sha256'],o['source_and_predictions_SHA256']['source_test.json'])
        self.assertEqual(d['source_test_json_sha256'],n['source_sha256'])
        self.assertEqual(len(d['evidence_SHA256']),9)
        self.assertTrue(all(len(x)==64 for x in d['evidence_SHA256'].values()))

    def test_all_clean_originals_remain_clean_despite_other_activity(self):
        c=self.cases;metrics=self.d['clean_alternative_state_negative']
        self.assertEqual(len(c),352)
        self.assertEqual(metrics['author_clean_distinct_activity_pairs'],352)
        self.assertEqual(metrics['paired_orientation_comparisons'],704)
        self.assertEqual(len({x['case_id'] for x in c}),137)
        self.assertEqual(len({x['alternate_id'] for x in c}),metrics['distinct_wrong_activity_sources'])
        self.assertEqual(sum(x['localized_flag'] for x in c),metrics['author_clean_bad_flags_localized_wrong_activity_ref'])
        self.assertEqual(sum(x['global_flag'] for x in c),metrics['author_clean_bad_flags_global_wrong_activity_ref'])
        self.assertEqual(sum(x['any_pixel_flag'] for x in c),metrics['author_clean_bad_flags_any_pixel_wrong_activity_ref'])
        self.assertEqual(sum(x['exact_original_flag'] for x in c),0)
        self.assertTrue(all(x['correct_source_binding']=='review' for x in c))
        self.assertEqual(metrics['correct_source_binding_review_for_wrong_activity'],len(c))

    def test_released_author_bboxes_are_not_silently_assumed_valid(self):
        gold=self.d['gold_region_cross_split']
        self.assertEqual(gold['exact_modified_file_matches'],72)
        self.assertEqual(gold['original_refs_available'],70)
        self.assertEqual(gold['abstained_missing_original_refs'],2)
        self.assertEqual(gold['author_bbox_within_original_png'],62)
        self.assertEqual(gold['author_bbox_out_of_original_png'],8)
        self.assertEqual(gold['bbox_valid_changed_pixels_denominator'],62)
        self.assertEqual(gold['bbox_valid_changed_pixels_inside_90pct_or_more'],51)
        self.assertEqual(gold['old_detector_misses_with_bbox']+gold['new_detector_misses_with_bbox'],7)
        self.assertEqual(len(gold['invalid_native_bbox_indices']),8)
        self.assertEqual(len(set(gold['invalid_native_bbox_indices'])),8)

    def test_all_missed_mutations_are_small_but_not_empty(self):
        rows=self.d['annotated_missed_cases'];self.assertEqual(len(rows),7)
        self.assertEqual(len({x['case_id'] for x in rows}),7)
        self.assertTrue(all(0<x['changed_fraction']<=.0005 for x in rows))
        self.assertTrue(all(x['bbox_valid'] is True for x in rows))

    def test_cost_and_generalization_limits_are_not_hidden(self):
        r=self.d
        self.assertFalse(r['cost']['GPU_calls'])
        self.assertGreater(r['cost']['clean_alternative_state_image_CPU_seconds'],0)
        self.assertGreater(r['cost']['region_author_bbox_native_CPU_seconds'],0)
        self.assertEqual(r['cost']['aborted_first_clean_state_CPU_wall_seconds'],512.764)
        self.assertIn('human',r['review_limitations'][0])
        self.assertIn('not 352 independent',r['review_limitations'][1])
        self.assertIn('not proven in a live tool call',r['review_limitations'][-1])

if __name__=='__main__':unittest.main()
