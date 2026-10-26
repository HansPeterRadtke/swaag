"""Offline integrity tests for the previously unused 240-case RotBench split."""
import json
from pathlib import Path
import unittest

DOC=Path(__file__).resolve().parents[2]/'docs/research/perceptual-rotbench-unused-heldout-replication-2026-10-26.json'
PREV=Path(__file__).resolve().parents[2]/'docs/research/perceptual-rotbench-android-heldout-2026-10-26.json'


class TestExternalAndroidUnusedHoldout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=json.loads(DOC.read_text());cls.a=cls.r['per_case']

    @staticmethod
    def count(rows,field):
        v=[(x['external_author_bug'],x['predictions']['aligned'][field]) for x in rows]
        return {'TP':sum(y and p for y,p in v if p is not None),'FP':sum((not y) and p for y,p in v if p is not None),'FN':sum(y and not p for y,p in v if p is not None),'TN':sum((not y) and not p for y,p in v if p is not None),'abstain':sum(p is None for _,p in v)}

    def test_staged_source_provenance_and_new_case_ids(self):
        before=json.loads(PREV.read_text())
        self.assertEqual(self.r['source_sha256'],before['source_and_predictions_SHA256']['source_test.json'])
        self.assertEqual(self.r['source_revision'],before['source_pinned_revision'])
        self.assertEqual(len(self.a),240)
        self.assertEqual(len({x['id'] for x in self.a}),240)
        self.assertFalse({x['id'] for x in self.a} & {x['id'] for x in before['rows']})
        for name in ('design.json','selection.json','predictions.json','evaluated.json'):
            self.assertEqual(len(self.r['frozen_receipts'][name]),64)

    def test_recompute_confusion_matrices_from_retained_rows(self):
        for method in ('localized','global','any_pixel'):
            self.assertEqual(self.r['counts'][method],self.count(self.a,method))
        self.assertEqual(sum(x['external_author_bug'] for x in self.a),120)
        self.assertEqual(sum(not x['external_author_bug'] for x in self.a),120)
        self.assertEqual(sum(x['type']=='right' for x in self.a),120)
        self.assertEqual(sum(x['type']=='layout-clip' for x in self.a),9)
        self.assertEqual(sum(x['type']=='layout-overlap' for x in self.a),9)
        self.assertEqual(sum(x['type']=='layout-miss' for x in self.a),34)
        self.assertEqual(sum(x['type']=='direction-mismatch' for x in self.a),34)
        self.assertEqual(sum(x['type']=='state-loseinput' for x in self.a),34)

    def test_missing_source_abstentions_cannot_be_counted_as_clean(self):
        missing=[x for x in self.a if x['predictions']['aligned']['localized'] is None]
        self.assertEqual(len(missing),self.r['errors']['cases_abstained'])
        self.assertTrue(all(x['predictions']['aligned']['global'] is None and x['predictions']['aligned']['any_pixel'] is None for x in missing))
        self.assertTrue(self.r['errors']['unavailable_pinned_source_files_count']>0)
        self.assertTrue(all(x.endswith('.png') for x in self.r['errors']['unavailable_pinned_source_paths']))
        self.assertTrue(self.r['errors']['no_missing_source_filled_with_guessed_media'])

    def test_bad_viewport_reference_is_not_pass_or_verified_fail(self):
        clean=[x for x in self.a if not x['external_author_bug']]
        self.assertEqual(sum(all(x['aligned_bytestream_identical']) for x in clean),self.r['negative_control']['exact_approved_reference_bytes'])
        self.assertEqual(sum(x['predictions']['wrong_viewport_forced']['localized'] is True for x in clean),self.r['negative_control']['wrong_viewport_forced_bad_flags_on_author_clean'])
        self.assertEqual(sum(x['predictions']['wrong_viewport_correct_gate']['abstain_due_to_dimensions'] for x in clean),self.r['negative_control']['wrong_viewport_source_binding_ABSTAINS_on_author_clean'])
        self.assertIn('wrong viewport',self.r['negative_control']['wrong_viewport_control_does_not_match_state'])

    def test_resource_and_selection_bias_recorded(self):
        self.assertEqual(self.r['resources']['gpu_inference'],False)
        self.assertGreater(self.r['resources']['total_native_RGB_CPU_plus_wrong_viewport_resizing_seconds'],0)
        self.assertEqual(self.r['selection']['prior_development_examples'],100)
        self.assertEqual(self.r['selection']['previously_unused_heldout'],240)
        self.assertGreater(self.r['selection']['shared_apps_across_rounds'],0)
        self.assertIn('not independent human',self.r['original_gold_status'].lower())
        self.assertIn('synthetic',self.r['limitations'][4].lower())

if __name__=='__main__':unittest.main()
