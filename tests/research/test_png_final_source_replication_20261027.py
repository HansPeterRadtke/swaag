"""Offline evidence/regression of truly decoded PNG source/capture integrity.

Uses compact source-report only; original publisher PNGs, PII and active
research history are not needed for CI. Never replays the stored 32 effects.
"""
from collections import Counter
import json
from pathlib import Path
import unittest

BASE=Path(__file__).resolve().parents[2]
PUBLIC=BASE/'docs/research/perceptual-real-png-pixel-and-capture-provenance-2026-10-27.json'

class NativePngFinalOutputEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=json.loads(PUBLIC.read_text())

    def test_publisher_release_and_independently_frozen_sources(self):
        d=self.d
        self.assertEqual(d['original_release_SHA256'],'0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7')
        for k in ('separate_frozen_study_SHA256','complete_actual_registered_runtime_result_SHA256',
                  'source_manifest_independent_SHA256','source_prior_full_image_predictions_SHA256',
                  'source_distinct_app_negative_manifest_SHA256','registered_tool_module_sha256',
                  'dependency_pyproject_sha256','dependency_lock_sha256'):
            self.assertEqual(len(d[k]),64)
        self.assertEqual(d['actual_attachment_tool_call_count'],32)
        self.assertEqual(len(d['outcomes_compact']),32)

    def test_each_class_confusion_rederived_from_original_case_receipts(self):
        d=self.d
        expected={'clean':'pass','changed':'fail','wrong_activity':'review','truncated_final_png':'review'}
        allids=set()
        for kind,verdict in expected.items():
            rows=[x for x in d['outcomes_compact'] if x['category']==kind]
            self.assertEqual(len(rows),8)
            self.assertTrue(all(x['expected']==verdict and x['verified_outcome']==verdict for x in rows))
            self.assertTrue(all(x['durable_event_hash'] and len(x['durable_event_hash'])==64 for x in rows))
            self.assertTrue(all(x['actual_SWAAG_tool_call_wall_s']>0 for x in rows))
            self.assertEqual(d['case_counts'][kind]['correct'],8)
            self.assertEqual(d['case_counts'][kind]['expected_verdict'],verdict)
            allids.update(x['durable_event_hash'] for x in rows)
        self.assertEqual(len(allids),32)
        cm=d['confusion_by_exact_defined_property']
        self.assertEqual(cm['source_correct_immutable_changed'],{'TP':8,'FN':0})
        self.assertEqual(cm['source_correct_exact_immutable_unchanged'],{'FP':0,'TN':8})
        self.assertEqual(cm['source_wrong_activity'],{'REVIEW':8,'auto_PASS':0,'auto_FAIL':0})
        self.assertEqual(cm['undecodable_actual_PNG'],{'REVIEW':8,'auto_PASS':0,'auto_FAIL':0})

    def test_renderability_negative_control_is_really_png_end_boundary(self):
        d=self.d
        self.assertIn('IEND',d['corrupted_negative'])
        self.assertIn('8/8',d['corrupted_negative'])
        self.assertIn('different-Activity',d['source_negative'])
        self.assertTrue(all(x['reason']=='png_final_pixels_or_end_boundary_not_verified'
                            for x in d['outcomes_compact'] if x['category']=='truncated_final_png'))
        self.assertTrue(all(x['reason']=='attested_source_state_or_contract_unavailable'
                            for x in d['outcomes_compact'] if x['category']=='wrong_activity'))
        self.assertIn('Pillow',d['PNG_pixel_decoder_boundary'])

    def test_no_false_universal_GUI_quality_claim(self):
        d=self.d
        self.assertFalse(d['cost']['GPU_used'])
        self.assertGreater(d['cost']['all_32_cases_actual_SWAAG_attachment_store_and_durable_history_wall_s'],0)
        self.assertGreater(d['cost']['max_call_wall_s'],d['cost']['min_call_wall_s'])
        self.assertTrue(d['cost']['old_3case_runtime_wall_s_not_matched_for_before_after'])
        self.assertIn('no new independent human',d['frozen_gold_authority'])
        self.assertTrue(any('not signed authority' in x.lower() for x in d['known_limitations']))
        self.assertIn('not an appraisal',d['PNG_pixel_decoder_boundary'])

    def test_descriptive_sample_CI_not_transfer_certificate(self):
        i=self.d['descriptive_uncertainty_not_general_population']
        self.assertAlmostEqual(i['n8_success_95pct_exact_binomial_lower'],.63058,places=4)
        self.assertAlmostEqual(i['n16_zero_unqualified_auto_decisions_95pct_exact_binomial_upper'],.20591,places=4)
        self.assertIn('NOT valid',i['warning'])

if __name__=='__main__':unittest.main()
