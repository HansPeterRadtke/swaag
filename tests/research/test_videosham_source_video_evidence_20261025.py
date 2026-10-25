"""Internal consistency/provenance checks for REAL external author-edited videos.

These do not replay videos, forge model quality, download source, access
restricted VideoSham original media or run on a GPU. They validate the saved
human-editor intent labels, source pair count, false-alarm and miss bookkeeping,
and crucially forbid interpreting author edits as natural judged playback faults.
"""
from pathlib import Path
import unittest,json
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class OriginalVideoShamSourceEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads((ROOT/'perceptual-videosham-real-temporal-editor-control-2026-10-25.json').read_text())
  cls.reg=json.loads((ROOT/'perceptual-analyzer-registry-2026-10-24.json').read_text())
 def test_fixed_human_authored_source_category_selection_and_pair_provenance(self):
  d=self.d
  self.assertEqual(d['selection']['independent_original_source_clip_families'],38)
  self.assertEqual(d['selection']['author_temporal_intent_positive_pairs'],19)
  self.assertEqual(d['selection']['author_other_edit_controls_pairs'],19)
  self.assertEqual(d['selection']['original_and_edited_video_files_total'],76)
  self.assertTrue(d['selection']['no_source_visual_features_used_for_label_selection'])
  self.assertIn('noncommercial',d['license'])
 def test_all_frozen_methods_count_false_alarms_misses_and_abstentions(self):
  for name,method in self.d['comparisons'].items():
   self.assertEqual(method['TP']+method['FN']+method['abstained_positive'],19)
   self.assertEqual(method['TN']+method['FP']+method['abstained_other_edit_controls'],19)
   self.assertGreaterEqual(method['FP'],0);self.assertGreaterEqual(method['FN'],0)
   for name2 in ['exact_binomial95_on_conditional_recall','exact_binomial95_on_conditional_false_alarm_rate']:
    if method[name2] is not None:
     self.assertEqual(len(method[name2]),2)
     self.assertLessEqual(method[name2][0],method[name2][1])
  self.assertIn('duration_only_CONFOUNDED_NOT_VISUAL',self.d['comparisons'])
  self.assertIn('reference_aware',self.d['comparisons'])
 def test_failure_and_abstention_preserved_no_fake_100pct_accuracy(self):
  self.assertEqual(self.d['resource']['GPU_used'],False)
  self.assertEqual(self.d['resource']['actual_video_model_inference_used'],False)
  self.assertEqual(self.d['resource']['raw_restricted_video_media_in_Git'],False)
  self.assertGreater(self.d['resource']['ffmpeg_original_and_edited_native_fps_gray_128x72_frames_fully_decoded'],100)
  self.assertEqual(len(self.d['source_hashes']),5)
  self.assertTrue(all(len(x)==64 for x in self.d['source_hashes'].values()))
 def test_other_processed_edits_are_controls_not_unedited_toy_negatives(self):
  txt=str(self.d['why_this_does_not_prove_objectionable_video_quality'])
  self.assertIn('other edits',txt.lower())
  self.assertIn('human',txt.lower())
  self.assertIn('no qualified',self.d['status'].lower())
 def test_registry_does_not_approve_auto_stutter_gate(self):
  self.assertIn('author_video_edit_temporal_external_controls_2026_10_25',self.reg)
  row=self.reg['author_video_edit_temporal_external_controls_2026_10_25']
  self.assertIn('PARTIAL',row['status'])
  self.assertIn('not',str(row).lower())
if __name__=='__main__':unittest.main()
