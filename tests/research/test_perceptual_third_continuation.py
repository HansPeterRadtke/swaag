"""Regression over independent source evidence, never an accuracy certificate."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class PerceptualThirdContinuationEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads((ROOT/'perceptual-third-continuation-evidence-2026-10-24.json').read_text())
  cls.registry=json.loads((ROOT/'perceptual-analyzer-registry-2026-10-24.json').read_text())
 def test_fps_confound_remains_explicit(self):
  a=self.d['real_physics_human_pair_confounded']
  self.assertEqual(a['total_video_pair_evaluations'],18)
  self.assertEqual(a['native_fps_alone_agreed_with_human_preferences'],18)
  self.assertIn('cannot imply physical reasoning',a['main_result'])
 def test_vlm_majority_does_not_pass_as_qualified(self):
  a=self.d['five_released_3d_human_VLM_votes']
  self.assertEqual(len(a['five_models']),5)
  m=a['expert_missing_parts_majority_3_of_5'];self.assertEqual((m['TP'],m['FP'],m['FN'],m['TN']),(8,8,23,60))
  any1=a['expert_missing_parts_any_1_of_5'];self.assertEqual((any1['TP'],any1['FP']),(20,29))
  x=a['expert_mechanical_missing_parts_majority_3_of_5'];self.assertEqual((x['TP'],x['FN']),(1,8))
 def test_GLBS_demonstrate_structural_label_mismatch(self):
  d=self.d['real_glb_deterministic_structure']
  self.assertEqual(d['checked_meshes'],10);self.assertEqual(d['submeshes_all_open'],10)
  openmesh=d['missingparts_as_open_submeshes'];self.assertEqual(openmesh['FP'],6)
  inter=d['missingparts_as_coarse_AABB_overlap'];self.assertEqual((inter['TP'],inter['FP'],inter['TN']),(4,3,3))
 def test_human_temporal_event_and_misleading_no_label_controls(self):
  d=self.d['spotlight_human_temporal_errors']
  self.assertEqual(d['selected_videos'],24);self.assertEqual(d['source_prompt_families'],8)
  self.assertEqual(d['videos_with_human_motion_label'],9)
  self.assertEqual(d['videos_without_motion_label_NOT_GOLD_CLEAN'],15)
  self.assertEqual(d['separately_human_localized_motion_intervals'],11)
  warp=d['low_level_analyzer_results']['warp_residual']
  self.assertEqual((warp['TP_on_labeled_motion'],warp['FN_on_labeled_motion'],warp['FP_proxy_on_no_motion_labels']),(5,4,0))
 def test_motion_localization_close_to_random_and_fullclip_labels(self):
  d=self.d['spotlight_human_temporal_errors']
  self.assertEqual(d['chance_audit']['random_95pct_interval_counts'],[4,9])
  self.assertEqual(d['chance_audit']['full_clip_motion_annotation_count'],1)
  self.assertAlmostEqual(d['chance_comparison']['warp_residual']['random_peak_ge_observed_pvalue'],.1028,places=4)
  self.assertGreater(d['chance_audit']['random_expected_events_out_of_11'],6.5)
 def test_no_general_gate_and_sources_not_embedded(self):
  self.assertTrue(self.d['overall_status'].startswith('PARTIAL'))
  self.assertIn('third_hourly_continuation_2026_10_24',self.registry)
  self.assertTrue(self.registry['third_hourly_continuation_2026_10_24']['decision'].startswith('PARTIAL'))
  original=(ROOT/'perceptual-third-continuation-evidence-2026-10-24.json').read_text()
  self.assertNotIn('model.safetensors',original)
  self.assertNotIn('glb/golden/',original)
  self.assertNotIn('source_url',original)
  self.assertNotIn('raw_video_frames',original)
if __name__=='__main__':unittest.main()
