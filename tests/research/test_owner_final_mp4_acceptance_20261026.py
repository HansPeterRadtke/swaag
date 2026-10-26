import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'docs/research/perceptual-owner-final-mp4-playback-2026-10-26.json'

class FinalMovieReceipts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.d=json.loads(REPORT.read_text())
 def test_source_lineage(self):
  d=self.d
  self.assertEqual(d['source']['scene_sha256'],'6609f09ac0637dd232539f7d5de49ce3e59dc477a5bf25beb125ddcbf73849b6')
  self.assertEqual(d['source']['original_frame_count'],210)
  self.assertTrue(d['source']['all_source_frame_hashes_verified'])
 def test_encoded_final_source_pixel_fidelity(self):
  d=self.d['decoded'];w=d['warehouse'];dock=d['dock']
  self.assertEqual(d['source_matched_frame_sequence_property'],'PASS')
  self.assertEqual(w['original_frames_count'],210)
  self.assertGreaterEqual(w['source_alignment_PSN_dB_min'],25)
  self.assertGreaterEqual(w['source_alignment_PSN_dB_median'],30)
  self.assertGreaterEqual(w['SSIM_gray240_median'],.8)
  self.assertLess(dock['source_alignment_PSN_dB_median'],15)
  self.assertGreater(w['correct_minus_shifted_median_dB'],5)
 def test_exact_pts_and_hold(self):
  d=self.d
  for p in d['pts'].values():
   self.assertEqual(p['count'],210)
   self.assertEqual(p['nonmonotonic_count'],0)
   self.assertLess(p['frame_timebase_max_error_seconds'],1e-5)
  self.assertEqual(d['tail']['frames_after_source_camera_stop'],45)
  self.assertGreater(d['tail']['early_first_165_decoded_adjacent_frame_gray_motion_median'],d['tail']['tail_last_45_decoded_adjacent_frame_gray_motion_median'])
 def test_real_browser_playback_not_metadata_only(self):
  b=self.d['browser'];self.assertEqual(b['status'],'PASS')
  self.assertTrue(b['ended_successfully'])
  self.assertTrue(b['all_media_times_non_decreasing'])
  self.assertTrue(b['screenshots_saved'])
  self.assertEqual(b['last_view']['current_time'],14)
  self.assertIsNone(b['last_view']['media_error'])
 def test_limits_and_no_production_writes(self):
  d=self.d
  self.assertEqual(d['audio_synchronization'],'REVIEW')
  self.assertEqual(d['subjective_video_quality'],'REVIEW')
  self.assertEqual(d['physical_contact_from_video'],'REVIEW')
  self.assertFalse(d['gpu_used'])
  self.assertTrue(d['storyboard_transfer_denied'])
  self.assertTrue(d['original_mp4s_unmodified'])
if __name__=='__main__': unittest.main()
