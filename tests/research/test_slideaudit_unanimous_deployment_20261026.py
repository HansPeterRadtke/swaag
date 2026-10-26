"""Offline, zero-GPU falsification of SlideAudit original-human deployment receipts."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[2]
PATH=REPO/'docs/research/perceptual-human-unanimous-negative-deployment-2026-10-26.json'
POLICY=REPO/'src/swaag/perception/acceptance_policy.py'

class HumanOriginalSourceDeploymentTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads(PATH.read_text())

 def test_source_authority_and_unseen_negative_selection(self):
  d=self.d
  self.assertEqual(d['revision'],'642d490b7c1d2e78a50a631bfd359433397f3ecf')
  self.assertTrue(d['all_72_unanimously_negative'])
  self.assertTrue(d['unseen_72_source_stratified'])
  self.assertIn('unanimous 3/3',d['origin'])
  self.assertIn('Other defects',d['gold'])
  self.assertEqual(len(d['original_frozen_SHA256']),8)
  self.assertTrue(all(len(k)==64 and all(c in '0123456789abcdef' for c in k) for k in d['original_frozen_SHA256'].values()))
  self.assertEqual(len(d['case_hash_flags']),72)
  self.assertEqual(len({x['id_hash'] for x in d['case_hash_flags']}),72)
  self.assertEqual(sorted(Counter(x['source_stratum'] for x in d['case_hash_flags']).values()),[6]*12)

 def test_unanimous_human_negative_flag_counts_recomputed(self):
  d=self.d
  a=d['case_hash_flags'];counts=d['false_review_proposals_per_72_human_unanimous_negatives']
  self.assertEqual(sum(x['flag_OCR'] for x in a),counts['OCR_only'])
  self.assertEqual(sum(x['flag_CLIP'] for x in a),counts['CLIP_only'])
  self.assertEqual(sum(x['flag_source_geometry'] for x in a),counts['geometry_only'])
  self.assertEqual(sum(x['flag_union'] for x in a),counts['OCR_or_CLIP'])
  self.assertEqual(counts['OCR_only'],26)
  self.assertEqual(counts['CLIP_only'],66)
  self.assertEqual(counts['OCR_or_CLIP'],70)
  self.assertTrue(all(x['flag_union']==(x['flag_OCR'] or x['flag_CLIP']) for x in a))
  self.assertEqual(counts['null_no_proposal'],0)

 def test_no_source_geometry_claim_for_contrast(self):
  d=self.d
  by=d['false_alarms_on_unanimous_negatives_per_category']
  self.assertIsNone(by['Insufficient Color Contrast for Readability']['geometry_only'])
  self.assertEqual(by['Content Overflow/Cut-off']['CLIP_only'],49)
  self.assertEqual(by['Occluded Content']['OCR_only'],8)
  self.assertEqual(by['Insufficient Color Contrast for Readability']['OCR_only'],2)

 def test_previous_human_positive_versus_control_misses_and_false_alarms(self):
  d=self.d
  previous=d['earlier_343_source_category_strong_human_positive_and_matched_negative_confusions']
  for task,n in (('Content Overflow/Cut-off',82),('Occluded Content',105),('Insufficient Color Contrast for Readability',20)):
   for method,results in previous[task].items():
    if method=='geometry_only' and n==20:
     self.assertEqual(results['abstained'],40)
     continue
    self.assertEqual(results['TP']+results['FN'],n)
    self.assertEqual(results['FP']+results['TN'],n)
    self.assertEqual(results['abstained'],0)
    self.assertGreaterEqual(previous[task]['OCR_or_CLIP']['TP'],previous[task]['OCR_only']['TP'])
  self.assertEqual(previous['Content Overflow/Cut-off']['OCR_or_CLIP']['TP'],75)
  self.assertEqual(previous['Content Overflow/Cut-off']['OCR_or_CLIP']['FP'],68)
  self.assertEqual(previous['Occluded Content']['OCR_or_CLIP']['TP'],79)
  self.assertEqual(previous['Occluded Content']['OCR_or_CLIP']['FP'],81)
  self.assertEqual(previous['Insufficient Color Contrast for Readability']['OCR_or_CLIP']['TP'],15)

 def test_resource_accounting_and_unmeasured_end_to_end_cost(self):
  l=self.d['latency_and_resource']
  t=l['actually_measured_separate_method_72_wall_s']
  self.assertAlmostEqual(t['OCR_only'],170.992,places=2)
  self.assertAlmostEqual(t['CLIP_only'],61.341,places=2)
  self.assertAlmostEqual(t['geometry_only'],.05,places=2)
  self.assertAlmostEqual(l['combined_OCR_plus_CLIP_arithmetic_sequential_sum_72_s'],t['OCR_only']+t['CLIP_only'],places=3)
  self.assertTrue(l['combined_model_end_to_end_wall_time_unmeasured'])
  self.assertEqual(l['human_review_time_per_case'],'NOT MEASURED')
  self.assertFalse(l['GPU_used'])

 def test_source_conflicts_do_not_override_verified_property_policy(self):
  spec=importlib.util.spec_from_file_location('standalone_perceptual_acceptance_for_study',POLICY)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  obs=module.Observation('text_visible','OCR-proposal','fail','visual_proposal',reference_verified=True,actual_output_verified=True,measured_property=True)
  self.assertEqual(module.decide_property([obs]),module.Verdict.REVIEW)
  source=module.Observation('text_visible','author-source-coordinate-only','fail','verified_source',reference_verified=True,actual_output_verified=False,measured_property=True)
  self.assertEqual(module.decide_property([source]),module.Verdict.REVIEW)
  verified=module.Observation('actual_png_sha_matches_author_contract','independent_sha_violation','fail','verified_source',reference_verified=True,actual_output_verified=True,measured_property=True)
  self.assertEqual(module.decide_property([verified]),module.Verdict.FAIL)

if __name__=='__main__':unittest.main()
