"""Offline evidence tests for preregistered unseen dock source temporal transfer."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

WORK=Path(__file__).resolve().parents[2]
REPORT=WORK/'docs/research/perceptual-dock-cross-source-video-transfer-2026-10-26.json'
POLICY=WORK/'src/swaag/perception/acceptance_policy.py'

class GenuineDifferentDockSourceTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.r=json.loads(REPORT.read_text());cls.c=cls.r['method_by_case']
 def test_new_scene_real_video_distinct_from_warehouse(self):
  x=self.r
  self.assertEqual(x['new_dock_source_final_mp4_SHA256'],'75ef01e7316d099759f4e1dccec304f4c3ea8a995949deebe7f6b41962c6ef93')
  self.assertEqual(x['new_dock_source_blend_SHA256'],'7c1bda0b4c78df455b609c1022b79e17c4e3bce48dbd6675c8d014425f97dbc4')
  self.assertEqual(x['preregistered_result'],'PASS')
  self.assertEqual(x['transferred_source_psnr_threshold_dB'],28)
  self.assertEqual(x['transferred_no_ref_gray_motion_threshold'],1)
  self.assertGreater(x['current_source_dock_original_against_Cycles_min_PSNR_dB'],28)
  self.assertEqual(len(x['source_data_receipts_SHA256']),7)
  self.assertTrue(all(len(z)==64 for z in x['source_data_receipts_SHA256'].values()))
  self.assertTrue(all(x['preregistered_checks'].values()))

 def test_unseen_pregold_fault_locations_and_real_clean_controls(self):
  c=self.c
  self.assertEqual(set(c),{'clean_crf17','clean_crf23','holds_crf17','reversals_crf17'})
  self.assertEqual(c['clean_crf17']['frozen_edit_positions'],[])
  self.assertEqual(c['clean_crf23']['frozen_edit_positions'],[])
  self.assertEqual(c['holds_crf17']['frozen_edit_positions'],[41,88,139])
  self.assertEqual(c['reversals_crf17']['frozen_edit_positions'],[63,64,121,122,161,162])
  self.assertGreater(c['clean_crf17']['PSNR_source_exact_frame_min_dB'],28)
  self.assertGreater(c['clean_crf23']['PSNR_source_exact_frame_min_dB'],28)
  self.assertEqual(c['clean_crf17']['encoder_CRf'],17)
  self.assertEqual(c['clean_crf23']['encoder_CRf'],23)

 def test_frame_level_confusion_recomputes_from_frozen_source_positions(self):
  total={key:dict(TP=0,FN=0,FP=0,TN=0) for key in ('source_method','no_reference_method')}
  for name,part in self.c.items():
   gold=set(part['frozen_edit_positions']);assert len(gold)<=6
   for key,poskey in (('source_method','source_bound_predicted_frame_indices'),('no_reference_method','no_reference_predicted_frame_indices')):
    predicted=set(part[poskey]);self.assertEqual(len(predicted),len(part[poskey]))
    conf={'TP':len(predicted&gold),'FN':len(gold-predicted),'FP':len(predicted-gold),'TN':210-len(gold|predicted)}
    self.assertEqual(conf,part[key],(name,key))
    for metric,value in conf.items():total[key][metric]+=value
  self.assertEqual(total['source_method'],{'TP':9,'FN':0,'FP':0,'TN':831})
  self.assertEqual(total['no_reference_method'],{'TP':3,'FN':6,'FP':0,'TN':831})
  self.assertEqual(total['source_method'],self.r['frame_confusion']['source_verified'])
  self.assertEqual(total['no_reference_method'],self.r['frame_confusion']['no_reference'])

 def test_wrong_product_and_no_reference_controls(self):
  self.assertLess(self.r['wrong_source_original_warehouse_reference_median_PSNR_dB'],20)
  self.assertTrue(self.r['wrong_source_ref_verdict'].startswith('REVIEW'))
  ff=self.r['FFmpeg_freezedetect_negative_control']
  self.assertEqual(ff['clean_dock_2_reencodes_events'],0)
  self.assertEqual(ff['faulted_dock_holds_events'],3)
  self.assertEqual(ff['faulted_dock_reversal_events'],0)
  for case in self.c.values():
   self.assertTrue(case['PTS_monotonic'])
   self.assertLess(case['PTS_max_error_s'],1e-5)
  self.assertGreater(self.c['reversals_crf17']['PSNR_source_exact_frame_median_dB'],28)

 def test_uncertainty_limits_do_not_claim_population_certification(self):
  x=self.r
  interval=x['conditional_descriptive_95pct_exact_binomial_intervals']
  self.assertAlmostEqual(interval['sensitivity_source'][0],.66373,places=5)
  self.assertEqual(interval['sensitivity_source'][1],1.)
  self.assertEqual(interval['false_positive_rate_source'][0],0.)
  self.assertAlmostEqual(interval['false_positive_rate_source'][1],.00443,places=5)
  self.assertTrue(any('correlated' in v.lower() or 'one previously unseen dock' in v.lower() for v in x['limits']))
  self.assertTrue(any('human' in v.lower() and 'natural' in v.lower() for v in x['limits']))
  self.assertFalse(x['GPU_used'])

 def test_html5_success_does_not_prove_frame_sequence(self):
  cases=self.r['browser_decoded_user_facing']
  self.assertEqual(set(cases),{'clean_crf23','reversals_crf17'})
  self.assertTrue(all(v['success'] and v['actual_end_playback_seconds']==14 for v in cases.values()))
  self.assertEqual(cases['clean_crf23']['source_frame_fault_candidates'],0)
  self.assertEqual(cases['reversals_crf17']['source_frame_fault_candidates'],6)

 def test_real_policy_replay_preserves_review_for_wrong_reference(self):
  d=self.r['source_attested_actual_SWAAG_policy_replay']
  self.assertEqual({k:v['verified_source_property'] for k,v in d.items()}, {'clean_crf17':'pass','clean_crf23':'pass','holds_crf17':'fail','reversals_crf17':'fail'})
  self.assertTrue(all(v['unverified_CV_proposal']=='review' and v['wrong_product_reference']=='review' for v in d.values()))
  spec=importlib.util.spec_from_file_location('dock_source_replay_policy',POLICY)
  mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
  missing=mod.Observation('dock_frame_sequence','other_product_reference','fail','verified_source',reference_verified=False,actual_output_verified=True,measured_property=True)
  self.assertEqual(mod.decide_property([missing]),mod.Verdict.REVIEW)
  verified=mod.Observation('dock_frame_sequence','attested_native_frames','fail','verified_source',reference_verified=True,actual_output_verified=True,measured_property=True)
  self.assertEqual(mod.decide_property([verified]),mod.Verdict.FAIL)

 def test_timing_is_actual_separate_components_not_claimed_live_latency(self):
  c=self.r['latency'];self.assertGreater(c['original_extract_s'],0)
  self.assertGreater(c['source_verify_and_decode_png_seconds'],0)
  self.assertGreater(sum(c['variants_encode_s'].values()),0)
  self.assertEqual(len(c['variant_CPU_decode_s']),4)
  self.assertTrue(all(v>0 for v in c['variant_CPU_decode_s'].values()))
  self.assertFalse(c['GPU_used'])
  self.assertEqual(self.r['overall'],'PARTIAL')
  self.assertTrue(any('Live SWAAG' in p or 'OFFLINE' in p for p in self.r['limits']))

if __name__=='__main__':unittest.main()
