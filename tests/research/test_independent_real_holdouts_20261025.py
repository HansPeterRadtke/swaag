"""Check preserved independent-source scientific claims, NOT visual acceptance.

This validates internal consistency of archived aggregate evidence without
rerunning CPU model/geometry operations or requiring licensed raw media in Git.
"""
from pathlib import Path
import json,unittest,math
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class HeldoutSourceEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.a=json.loads((ROOT/'perceptual-asset-and-text-holdout-uncertainty-2026-10-25.json').read_text())
  cls.registry=json.loads((ROOT/'perceptual-analyzer-registry-2026-10-24.json').read_text())
 def test_old_ten_glbs_excluded_and_real_human_missingparts_claim_is_negative(self):
  x=self.a['real_GLBS'];c=x['AABB_broadphase_missingparts_confusion']
  self.assertEqual(x['source_holdout_n'],119)
  self.assertEqual(x['label_missing_positives'],33)
  self.assertEqual([c[k] for k in ('TP','FP','FN','TN')],[9,20,24,66])
  self.assertAlmostEqual(c['recall'],9/33)
  self.assertAlmostEqual(c['specificity'],66/86)
  self.assertEqual(x['open_mesh_flag_n'],119)
  self.assertEqual(x['degenerate_face_flag_n'],0)
  self.assertEqual(x['bad_coords_flag_n'],0)
  ci=x['prompt_cluster_bootstrap']['intervals']['AABB_AUROC']
  self.assertLess(ci['low'],.5);self.assertGreater(ci['high'],.5)
 def test_expert_reference_crosswalk_does_not_overclaim_ambiguous_ids(self):
  x=self.a['real_GLBS']['expert_crosswalk']
  self.assertEqual(x['unambiguous_n'],97)
  self.assertEqual(x['ambiguous_prompt_generator_n'],2)
  self.assertEqual(x['not_found_n'],1)
  self.assertTrue(all(v['matched_unambiguously']==97 and v['agreement']==97 for v in x['full_agreement_on_unique_keys'].values()))
  self.assertIn('overwrote',x['note'])
 def test_paired_real_ocr_better_signal_but_far_too_many_control_flags(self):
  x=self.a['real_paired_mobile_screenshots']
  self.assertEqual([x['total_sample'],x['calibration_n'],x['app_disjoint_eval_n']],[132,35,97])
  self.assertEqual([x['calibration_app_n'],x['evaluation_app_n']],[14,38])
  c=x['heldout_ocr_missing_token_confusion']
  self.assertEqual([c[k] for k in ('TP','FP','FN','TN')],[21,34,7,35])
  self.assertAlmostEqual(c['precision'],21/55)
  self.assertAlmostEqual(c['recall'],21/28)
  self.assertLess(x['same_source_heldout_SSIM_auroc'],.5)
  self.assertGreater(x['heldout_ocr_auroc'],.6)
  delta=x['app_grouped_bootstrap']['intervals']['paired_AUROC_delta_OCR_minus_SSIM']
  self.assertGreater(delta['low'],0)
  self.assertLess(delta['high'],.4)
 def test_analyzer_registry_does_not_claim_reliable_visual_or_physics_gate(self):
  d=self.registry['independent_source_text_and_3d_holdouts_2026_10_25']
  self.assertTrue(d['status'].startswith('PARTIAL'))
  recent=[a for a in self.registry['analyzers'] if a.get('source_report')=='perceptual-analyzers-independent-heldouts-2026-10-25.md']
  self.assertEqual(len(recent),2)
  self.assertEqual({x['status'] for x in recent},{'rejected_as_semantic_missing_parts_judge','useful_change_evidence_not_acceptance_gate'})
  self.assertTrue(self.a['external_media_stays_local'])
if __name__=='__main__':unittest.main()
