"""Check scientific source-output bookkeeping, NOT perceptual model quality.

No licensed images, model binaries, HTTP/GPU or third-party raw annotations
required. Explicitly preserves false alarms/misses and correct label scope.
"""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class SlideAuditHumanEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads((ROOT/'perceptual-slideaudit-human-visual-evidence-2026-10-25.json').read_text())
 def test_original_human_labels_are_balanced_matching_not_toy_metadata(self):
  d=self.d
  self.assertEqual(d['dataset']['all_labelled_slide_images'],2400)
  self.assertEqual(d['dataset']['source_original_unique_image_cases_tested'],343)
  self.assertEqual(d['dataset']['human_reviewers_per_image'],3)
  self.assertTrue(d['dataset']['human_has_strong_agreement_required'])
  self.assertTrue(d['dataset']['unverified_or_disagreed_annotations_excluded_per_category'])
  self.assertEqual(d['dataset']['origin_types'],['Gemini','gdcdataset','Google'])
  self.assertIn('NOT actual desktop',d['dataset']['original_GUI_not_proven'])
 def test_true_positive_false_positives_fn_reported_for_three_distinct_tasks(self):
  expected={'Content Overflow/Cut-off':82,'Occluded Content':105,'Insufficient Color Contrast for Readability':20}
  self.assertEqual(set(self.d['tasks']),set(expected))
  for name,n in expected.items():
   t=self.d['tasks'][name]
   self.assertEqual(t['human_strong_positive_count'],n)
   self.assertEqual(t['matched_strong_negative_count'],n)
   for kind in ['fixed_OCR_OpenCV','image_only_vision_CLIP_ViTB32','leave_one_slide_source_out_trained_logistic_pixel_features']:
    c=t[kind]
    self.assertEqual(c['TP']+c['FN'],n)
    self.assertEqual(c['FP']+c['TN'],n)
    self.assertEqual(c['n'],2*n)
    self.assertGreaterEqual(c['FP'],0);self.assertGreaterEqual(c['FN'],0)
   self.assertIn('matched_pairs_with_OCR_abstention',t['CV_ocr_operational_coverage'])
   self.assertEqual(len(t['source_disjoint_fold_details']),3)
   for m in ['image_only_continuous_indicator_AUROC','CLIP_vision_ranking_AUROC']:
    if t[m] is not None:self.assertLessEqual(t[m],1.0);self.assertGreaterEqual(t[m],0.)
 def test_source_model_latency_hardware_and_gold_limitations_preserved(self):
  d=self.d
  self.assertFalse(d['resource']['GPU_inference_used'])
  self.assertGreater(d['resource']['CLIP_local_CPU_wall_s'],0)
  self.assertGreater(d['resource']['OCR_OpenCV_pixel_only_wall_s'],0)
  self.assertTrue(d['source_data_all_outside_project_git'])
  self.assertTrue(d['overall_status'].startswith('PARTIAL'))
  self.assertIn('NOT proper foreground/background WCAG',str(d['limitations']))
 def test_provenance_six_frozen_source_sha_refs(self):
  a=self.d['source_sha256'];self.assertEqual(len(a),6)
  for name,value in a.items():
   self.assertEqual(len(value),64)
   self.assertTrue(all(c in '0123456789abcdef' for c in value))
 def test_no_sensitive_guipilot_source_paths_in_research(self):
  s=(ROOT/'perceptual-slideaudit-human-visual-evidence-2026-10-25.json').read_text()
  self.assertNotIn('image_bytes',s)
  self.assertNotIn('raw_ocr_text',s)
if __name__=='__main__':unittest.main()
