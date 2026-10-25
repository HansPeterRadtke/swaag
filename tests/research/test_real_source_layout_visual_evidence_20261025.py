"""Source/evidence consistency regression; NOT proof of visual model quality.

No original rendered media, user annotations, model weights, private accounts,
GPU/model calls or network needed. Check only published aggregate claims and
that source-layout roles are not mislabeled as final GUI acceptance.
"""
from pathlib import Path
import json,unittest
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class GenuineSourceVsRenderedHumanEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads((ROOT/'perceptual-real-source-layout-human-visual-2026-10-25.json').read_text())
  cls.registry=json.loads((ROOT/'perceptual-analyzer-registry-2026-10-24.json').read_text())
 def test_all_source_description_to_actual_human_reviewed_render_identity(self):
  d=self.d
  self.assertEqual(d['n_distinct_original_slide_rasters'],343)
  self.assertEqual(d['n_original_source_layout_descriptions'],343)
  self.assertEqual(d['n_valid_source_rectangles'],1684)
  self.assertTrue(d['overall_status'].startswith('PARTIAL'))
  self.assertFalse(d['original_licensed_or_private_media_in_git'])
 def test_source_geometry_weak_on_clipping_even_with_343_original_slides(self):
  t=self.d['target_human_category_results']['Content Overflow/Cut-off']
  a=t['source_rectangle_fixed_rule']
  self.assertEqual((a['TP'],a['FP'],a['FN'],a['TN']),(9,7,73,75))
  self.assertEqual((t['old_image_only_OCR_baseline']['TP'],t['old_image_only_OCR_baseline']['FP']),(58,26))
  self.assertAlmostEqual(t['source_primary_continuous_AUROC'],.5611986912552053)
  delta=t['source_image_primary_AUROC_delta_matched95']
  self.assertLess(delta[0],delta[1]);self.assertLess(delta[1],0)
 def test_occlusion_source_geometry_can_localize_but_false_alerts_remain(self):
  t=self.d['target_human_category_results']['Occluded Content']
  x=t['source_rectangle_fixed_rule']
  self.assertEqual((x['TP'],x['FP'],x['FN'],x['TN']),(93,33,12,72))
  self.assertAlmostEqual(t['source_primary_continuous_AUROC'],.7705215419501132)
  self.assertEqual((t['old_image_only_OCR_baseline']['TP'],t['old_image_only_OCR_baseline']['FP']),(24,12))
  self.assertEqual((t['old_real_zero_shot_CLIP']['TP'],t['old_real_zero_shot_CLIP']['FP']),(74,78))
  self.assertGreater(t['source_image_primary_AUROC_delta_matched95'][0],0)
  region=t['source_occlusion_localization_and_chance']
  self.assertEqual((region['n_occlusion_human_positive_slides_with_bbox'],region['source_localization_hits'],region['source_proposal_empty']),(104,85,12))
  self.assertGreater(region['expected_random_location_hits'],40)
  self.assertLess(region['expected_random_location_hits'],50)
  self.assertGreater(region['empirical_one_sided_probability_with_plus_one_correction'],0)
  self.assertLessEqual(region['empirical_one_sided_probability_with_plus_one_correction'],.0001)
 def test_or_and_show_false_positive_tradeoffs_not_universal_improvement(self):
  for task,expected in [('Content Overflow/Cut-off',(62,32,5,1)),('Occluded Content',(96,40,21,5))]:
   c=self.d['target_human_category_results'][task]
   self.assertEqual((c['geometry_OR_image_heuristic']['TP'],c['geometry_OR_image_heuristic']['FP'],c['geometry_AND_image_heuristic']['TP'],c['geometry_AND_image_heuristic']['FP']),expected)
 def test_video_source_is_candidate_not_installed_or_validated(self):
  x=self.d['separate_candidate_video_annotation_source']
  self.assertEqual(x['distinct_source_videoIDs_seen'],16351)
  self.assertGreater(x['annotated_events_per_frame_record'],400000)
  self.assertEqual(x['new_local_video_judge_accuracy'],'NOT TESTED')
  self.assertTrue(any('single' in v.lower() or 'one' in v.lower() for v in self.d['scientific_cautions']) or 'rating' in str(x).lower())
 def test_registry_explicitly_rejects_source_only_clipping_and_autonomous_occlusion(self):
  k='real_slide_source_geometry_rendered_visual_2026_10_25'
  self.assertIn(k,self.registry)
  self.assertTrue(self.registry[k]['state'].startswith('PARTIAL'))
  methods=[v for v in self.registry['analyzers'] if v.get('source_report')=='perceptual-source-layout-vs-actual-visual-2026-10-25.md']
  self.assertEqual(len(methods),2)
  self.assertTrue(all('NOT' in m['status'] or 'rejected' in m['status'] for m in methods))
  self.assertEqual(len(self.d['source_sha256']),9)
if __name__=='__main__':unittest.main()
