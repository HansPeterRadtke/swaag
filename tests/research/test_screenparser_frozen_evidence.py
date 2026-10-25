"""Verify source-linked ScreenParser LOCALIZATION evidence, not defect acceptance.

This is a fixed artifact-consistency regression, not proof of model quality.
No GPU, no model or network calls. Original source image bytes stay outside Git.
"""
from pathlib import Path
import json, unittest
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class RealWebsiteScreenParserEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads((ROOT/'perceptual-screenparser-verified-evidence-2026-10-25.json').read_text())
 def test_real_source_and_exact_bounded_target(self):
  a=self.d['case_scope']
  self.assertEqual(a['frozen_real_Firefox_screenshots'],32)
  self.assertEqual(a['known_browser_defective_targets'],16)
  self.assertEqual(a['paired_clean_controls'],16)
  self.assertEqual(a['actual_websites_or_site_families'],15)
  self.assertEqual(a['screenshot_dimensions_px'],[1920,994])
 def test_not_accidental_defect_accuracy_claim(self):
  self.assertIn('NOT a qualified',self.d['status'])
  self.assertIn('not exhaustive',self.d['case_scope']['note'].lower())
  self.assertIn('not evidence of valid standalone',self.d['conclusion'])
 def test_defective_versus_control_boxes(self):
  a=self.d['target_box_result']
  self.assertEqual(a['all']['iou_hits']['0.5'],11)
  self.assertEqual(a['defective']['iou_hits']['0.5'],3)
  self.assertEqual(a['clean_control']['iou_hits']['0.5'],8)
  self.assertEqual(self.d['paired_readonly_analysis']['clean_higher_IoU_than_bad_count'],15)
  self.assertEqual(self.d['paired_readonly_analysis']['same_pixel_pair_count'],1)
 def test_cpu_only_latency_and_source_provenance(self):
  a=self.d['model_inference']
  self.assertEqual(a['config']['device'],'cpu')
  self.assertGreater(a['total_cpu_seconds'],1800)
  self.assertGreater(a['actual_mean_seconds_per_image'],60)
  self.assertEqual(a['source_results_finished'],32)
  self.assertEqual(len(self.d['source_sha256']),5)
  self.assertFalse(self.d['licensed_source_media_in_git'])
 def test_runtime_not_unconditionally_commercially_licensed(self):
  self.assertIn('Ultralytics',self.d['model_weight_license'])
  self.assertIn('separate',self.d['model_weight_license'])
if __name__=='__main__':unittest.main()
