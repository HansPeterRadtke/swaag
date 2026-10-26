"""Offline exact source/video empirical error/latency and source authority checks.

May run on other SWAAG checkout without owner proprietary 3D media installed;
verify all compact source receipts and matrices, do not replay encodes.
"""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
EVID=ROOT/'docs/research/perceptual-owner-mp4-verified-temporal-faults-2026-10-26.json'
POLICY=ROOT/'src/swaag/perception/acceptance_policy.py'

class RealApprovedOwnerMovieTemporalControl(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads(EVID.read_text())
  cls.v=cls.d['cases']

 def test_original_authorized_source_and_frozen_fault_gold(self):
  d=self.d
  self.assertEqual(d['source_video_original_SHA'],'e2ef866ec79318a6b3812865a51922141ad02be66b555900da311171f49b4adb')
  self.assertEqual(d['source_3D_blend_original_SHA'],'6609f09ac0637dd232539f7d5de49ce3e59dc477a5bf25beb125ddcbf73849b6')
  self.assertEqual(d['source_authorized_final_hold_frames_1based'],[166,210])
  self.assertTrue(d['prior_real_final_video_source_210_Sources_verified'])
  self.assertEqual(d['variant_count'],3)
  self.assertEqual(d['positive_scripted_frames'],9)
  self.assertEqual(d['source_checked_frame_population_total'],630)
  self.assertEqual(len(d['frozen_SHA256']),7)
  self.assertTrue(all(len(v)==64 for v in d['frozen_SHA256'].values()))
  self.assertEqual(self.v['clean_reencoded']['script_known_altered_frames_1based'],[])
  self.assertEqual(self.v['single_frame_holds']['script_known_altered_frames_1based'],[40,91,145])
  self.assertEqual(self.v['short_reversals']['script_known_altered_frames_1based'],[55,56,105,106,150,151])

 def test_all_matrices_derive_gold_and_emitted_frame_predictions(self):
  sums={method:Counter() for method in ('source_bound_confusion','no_reference_low_motion_confusion')}
  for film in self.v.values():
   gold=set(film['script_known_altered_frames_1based'])
   for kind,key in (('source_bound_confusion','source_bound_predicted_frame_indices'),('no_reference_low_motion_confusion','no_reference_predicted_frame_indices')):
    marked=set(film[key]);c=film[kind]
    self.assertEqual(len(marked),len(film[key]))
    self.assertTrue(all(1<=x<=210 for x in marked))
    expected={'TP':len(marked&gold),'FN':len(gold-marked),'FP':len(marked-gold),'TN':210-len(gold|marked)}
    self.assertEqual(c,expected)
    sums[kind].update(c)
  self.assertEqual(sums['source_bound_confusion'],Counter(TP=9,FN=0,FP=0,TN=621))
  self.assertEqual(sums['no_reference_low_motion_confusion'],Counter(TP=3,FN=6,FP=0,TN=621))
  self.assertEqual(self.d['methods']['reference_based_PSNR_28dB'],dict(sums['source_bound_confusion']))
  self.assertEqual(self.d['methods']['no_reference_low_motion_under1_bounded_early'],dict(sums['no_reference_low_motion_confusion']))

 def test_freeze_detection_false_alarm_on_intentional_hold(self):
  for film in self.v.values():
   self.assertEqual(film['FFmpeg_freezedetect_intended_last_hold_event'],1)
   self.assertIn(11.0,film['FFmpeg_freezedetect_events_start_s'])
  self.assertEqual(self.v['single_frame_holds']['FFmpeg_freezedetect_unintended_moving_region_events'],3)
  self.assertEqual(self.v['short_reversals']['FFmpeg_freezedetect_unintended_moving_region_events'],0)
  self.assertEqual(self.v['clean_reencoded']['FFmpeg_freezedetect_unintended_moving_region_events'],0)

 def test_no_pts_or_browser_failure_despite_faulted_visual_frames(self):
  for name,film in self.v.items():
   p=film['timebase']
   self.assertEqual(p['count'],210)
   self.assertEqual(p['nonmonotonic'],0)
   self.assertLess(p['max_target_pts_error_s'],1e-5)
  browser=self.d['browser_user_facing']
  self.assertEqual(set(browser),{'clean_reencoded','short_reversals'})
  self.assertTrue(all(x['browser_playback_success'] for x in browser.values()))
  self.assertTrue(all(x['browser_video_endpoint_seconds']==14.0 and x['browser_media_error'] is None for x in browser.values()))
  self.assertEqual(browser['short_reversals']['source_sequence_fidelity_fault_flags'],6)
  self.assertEqual(browser['clean_reencoded']['source_sequence_fidelity_fault_flags'],0)

 def test_actual_resource_cost_is_measured_not_invented(self):
  r=self.d['resources']
  self.assertFalse(r['GPU_used'])
  self.assertGreater(r['original_video_to_lossless_PNG_decode_wall_s'],0)
  self.assertAlmostEqual(r['three_new_video_H264_encode_wall_total_s'],sum(r['new_video_H264_encode_wall_s_each'].values()),places=3)
  self.assertAlmostEqual(r['measured_scoring_wall_sum_s'],r['source_original_PNG_verify_and_load_s']+r['three_video_native_frame_decode_and_same_budget_metrics_wall_s']+r['three_video_ffprobe_PTS_wall_s']+r['three_video_official_FFmpeg_freezedetect_wall_s'],places=3)
  self.assertTrue(r['source_bound_and_no_reference_methods_share_frame_decode'])
  self.assertTrue(r['full_total_includes_extra_process_overhead_unmeasured'])

 def test_actual_policy_results_not_unqualified_media_approval(self):
  policy=self.d['policy_offline']
  self.assertEqual({k:v['source_verified_verdict'] for k,v in policy.items()}, {'clean_reencoded':'pass','short_reversals':'fail','single_frame_holds':'fail'})
  self.assertTrue(all(x['unverified_motion_verdict']=='review' and x['ffmpeg_tail_verdict']=='review' for x in policy.values()))
  self.assertTrue(self.d['limitations'][0].startswith('Synthetic frame edits'))
  self.assertTrue(any('no live SWAAG' in x.lower() or 'No live SWAAG' in x for x in self.d['limitations']))
  spec=importlib.util.spec_from_file_location('owner_video_actual_policy_under_test',POLICY)
  mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
  o=mod.Observation('visual_freeze_intended','ffmpeg_proposal','fail','visual_proposal',reference_verified=True,actual_output_verified=True,measured_property=True)
  self.assertEqual(mod.decide_property([o]),mod.Verdict.REVIEW)
  o2=mod.Observation('verified_source_sequence','sha_validated_approved_mp4','fail','verified_source',reference_verified=True,actual_output_verified=True,measured_property=True)
  self.assertEqual(mod.decide_property([o2]),mod.Verdict.FAIL)

 def test_question_by_question_audit_explicitly_unproved(self):
  audit=self.d['original_goal_audit']
  self.assertEqual(len(audit),7)
  self.assertTrue(all('question' in a and 'measured' in a and 'missing' in a and 'status' in a for a in audit))
  self.assertTrue(any('audio' in a['question'].lower() for a in audit))
  self.assertTrue(any('SWAAG' in a['question'] for a in audit))
  self.assertTrue(any('PARTIAL' in a['status'] for a in audit))
  self.assertEqual(self.d['status'],'PARTIAL')

if __name__=='__main__':unittest.main()
