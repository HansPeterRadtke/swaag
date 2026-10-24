"""Regression on exact source-linked research counts, not visual model acceptance."""
from pathlib import Path
import json,unittest
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class RealMediaResearchEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.data=json.loads((ROOT/'perceptual-visual-continuation-evidence-2026-10-24.json').read_text())
 def test_rapid_ocr_not_passed_as_unqualified(self):
  x=self.data['rapidocr_vs_tesseract']
  self.assertEqual(x['evaluated_screenshots'],70)
  self.assertEqual(x['scores']['test_contrast']['counts']['FP'],6)
  self.assertEqual(x['scores']['test_layout']['counts']['FN'],1)
 def test_browser_subtasks_diverge(self):
  x=self.data['firefox_dom_real_site_l1']['criterion_scores']
  self.assertEqual(x['dev_redecheck:element-protrusion']['TP'],7)
  self.assertEqual(x['test_redecheck:element-protrusion']['TP'],2)
  self.assertEqual(x['dev_redecheck:viewport-protrusion']['FN']+x['test_redecheck:viewport-protrusion']['FN'],5)
 def test_diagram_labels_recall_not_semantic(self):
  x=self.data['diagram_mmu_literal_labels'];self.assertEqual(x['gold_literal_node_labels'],134)
  self.assertEqual(x['psm6']['exact_labels_recovered'],42)
  self.assertEqual(x['psm11']['exact_labels_recovered'],30)
 def test_industrial_false_negatives(self):
  x=self.data['clip_local_patches_on_real_industrial_photos'];self.assertEqual(x['TP'],1);self.assertEqual(x['FN'],9)
 def test_video_identical_hold_and_freeze_counterexamples(self):
  x=self.data['real_render_video_controlled_frame_edits'];self.assertEqual(x['cases'],48)
  self.assertEqual(x['approved_hold_vs_freeze_identical_pixel_pairs'],8)
  self.assertEqual(x['scores']['has_diff_spike_six_mad']['all']['FP'],24)
 def test_unlicensed_originals_not_embedded(self):
  text=(ROOT/'perceptual-visual-continuation-evidence-2026-10-24.json').read_text()
  self.assertNotIn('image_bytes',text)
  self.assertNotIn('raw_output',text)

if __name__=='__main__':unittest.main()
