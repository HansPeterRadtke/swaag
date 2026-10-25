"""Validate provenance and source-author class TP/FP/FN/TN of actual game videos.

A passed test means compact research reports agree with independently saved
source images/author label outcome, NOT video model performance or production
acceptance. No 716MB external source media, paid service, GPU or original labels
inside source Git. True-negative controls counted explicitly; uncertain aborted
original FFmpeg filter research attempt preserved outside Git and excluded.
"""
from pathlib import Path
import json,unittest
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class SourceFrozenCharacterVideoIntegrity(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.a=json.loads((ROOT/'perceptual-tempglitch-real-character-pose-video-control-2026-10-25.json').read_text())
  cls.registry=json.loads((ROOT/'perceptual-analyzer-registry-2026-10-24.json').read_text())
  cls.owner=json.loads((ROOT/'perceptual-owner-reconciled-decision-2026-10-25.json').read_text())
 def test_actual_original_engine_video_source_counts_and_rights(self):
  d=self.a
  self.assertEqual(d['original_independent_scene_pairs'],12)
  self.assertEqual(d['original_video_files_CPU_decoded'],24)
  self.assertTrue(all(v['TP']+v['FN']==12 and v['FP']+v['TN']==12 for v in d['observed_original_source_binary_and_rank_performance'].values()))
  self.assertGreater(d['source_original_total_media_megabytes'] if 'source_original_total_media_megabytes' in d else d['source_media_megabytes'],700)
  self.assertFalse(d['run_resource']['GPU_used'])
  self.assertFalse(d['run_resource']['original_source_files_in_project_git'])
  self.assertTrue(d['status'].startswith('PARTIAL'))
 def test_five_frozen_methods_do_not_hide_false_negatives_or_positives(self):
  d=self.a
  m=d['observed_original_source_binary_and_rank_performance']
  expected={'official_FFmpeg_full_frame_freezedetect','whole_frame_pixel_MAE_stall','center_raw_still_while_outer_motion','phase_aligned_center_still_while_outer_motion','Farneback_center_still_while_background_moves'}
  self.assertEqual(set(m),expected)
  for name,row in m.items():
   self.assertEqual(row['TP']+row['FN']+row['abstained_original_positive_n'],12)
   self.assertEqual(row['FP']+row['TN']+row['abstained_original_normal_control_n'],12)
   self.assertGreaterEqual(row['FN'],0)
   self.assertGreaterEqual(row['FP'],0)
   self.assertEqual(len(row['source_original_exact_recall95_CI']),2) if row['source_original_exact_recall95_CI'] is not None else None
   self.assertEqual(len(row['source_scene_matched_bootstrap_95pct_CI']['AUROC']),2) if row['source_scene_matched_bootstrap_95pct_CI']['AUROC'] else None
 def test_external_character_pose_not_confused_with_encoded_full_frame_freeze(self):
  d=self.a
  self.assertIn('character',d['independent_source_truth_scope'].lower())
  self.assertIn('no human blind',d['independent_source_truth_scope'].lower())
  self.assertIn('ffmpeg_full_frame_freezedetect',d['method_frozen_conditions'])
  self.assertIn('frame-only',str(d['scientific_limits']).lower())
 def test_privacy_source_sha_and_no_previous_denied_effects(self):
  d=self.a
  self.assertEqual(len(d['source_provenance_sha256']),4)
  self.assertTrue(all(len(s)==64 for s in d['source_provenance_sha256'].values()))
  self.assertIn('not reattempted',str(d['scientific_limits']).lower())
 def test_results_registered_as_partial_not_universal_visual_qa(self):
  key='external_real_Godot_Frozen_Character_Animation_vs_Normal_2026_10_25'
  self.assertIn(key,self.registry)
  self.assertIn('PARTIAL',self.registry[key]['status'])
  self.assertIn('Platformer',self.registry[key]['new_original_public_source'])
  self.assertIn('later_external_actual_gameplay_frozen_character_video',self.owner)
  self.assertTrue(self.owner['status'].startswith('PARTIAL'))
 def test_compact_evidence_does_not_redistribute_video_frames(self):
  s=(ROOT/'perceptual-tempglitch-real-character-pose-video-control-2026-10-25.json').read_text()
  self.assertLess(len(s),35000)
  for bad in ('Godot_Frozen_Animation_Platformer_Buggy_', 'data:video/mp4', 'sample_id_source_file_path', 'raw_original_video_frames','native_frame_jpeg'):
   self.assertNotIn(bad,s)
if __name__=='__main__':unittest.main()
