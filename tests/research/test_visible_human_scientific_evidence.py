"""Fixed independent-source acceptance/evidence regression; no model replays."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class HumanReviewedVisualEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads((ROOT/'perceptual-visible-human-reviewed-evidence-2026-10-24.json').read_text())
 def test_anchored_real_target_inputs_do_not_prove_visibility_of_flaw(self):
  c=self.d['real_site_target_visible_capture'];self.assertEqual(c['screenshots'],32)
  self.assertEqual(c['actual_external_layout_defects'],16)
  self.assertTrue(c['not_all_defects_necessarily_visually_decidable'])
 def test_blind_image_only_CLIP_failed_heldout(self):
  x=self.d['real_site_image_only_CLIP']['scores']
  self.assertEqual(x['test_redecheck:element-protrusion']['TP'],0)
  self.assertEqual(x['test_redecheck:viewport-protrusion']['TP'],0)
  self.assertEqual(x['dev_redecheck:element-protrusion']['FP'],2)
 def test_scry_no_tagged_controls_cannot_be_called_gold_clean(self):
  s=self.d['human_annotated_paired_UI_Scry'];self.assertIn('NOT',s['no_tagged_caveat'])
  self.assertEqual(s['calibration_and_heldout_grouped_by_app']['heldout_rows'],249)
  x=s['frozen_pixel_metrics']['ssim_difference']['heldout'];self.assertEqual((x['TP'],x['FP'],x['TN'],x['FN']),(157,41,22,29))
 def test_scry_grouped_split_and_localization_only(self):
  s=self.d['human_annotated_paired_UI_Scry']
  self.assertEqual(s['calibration_and_heldout_grouped_by_app']['heldout_app_groups'],38)
  l=s['heldout_localization_without_defect_tag']
  self.assertEqual((l['issues_with_box'],l['bbox_iou_recovered_issues_no_tag_check']),(458,253))
  self.assertGreater(l['no_tagged_control_any_box_flag_fraction'],.9)
 def test_official_w3c_contrast_has_abstentions_and_errors(self):
  w=self.d['W3C_ACT_color_contrast']
  self.assertEqual(w['testcase_count'],34)
  c=w['aggregate_counts'];self.assertEqual(c['TP'],5);self.assertEqual(c['FP'],1)
  self.assertEqual(c['ABSTAIN'],11);self.assertEqual(c['INAPPLICABLE_WRONG'],4)
 def test_third_party_image_bytes_not_embedded(self):
  data=(ROOT/'perceptual-visible-human-reviewed-evidence-2026-10-24.json').read_text()
  self.assertNotIn('image_a',data);self.assertNotIn('raw_output',data)
  self.assertTrue(self.d['raw_media_policy'].startswith('All Scry'))
if __name__=='__main__':unittest.main()
