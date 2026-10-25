"""Verify source provenance and exact heldout false-negative counts, NOT model QA.

No original source photos, copyrighted manufacturer media, GPU, image models or
human relabeling in this test. Confirms 24/20/90 split, frozen normal-only
calibration, source original label counts, exact TP/FP/FN/TN, PaDiM research
proxy vs global/HSV, paired POSTHOC uncertainty and the no-global-pass policy.
"""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class MVTecScrewBagOriginalSourceIntegrity(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.a=json.loads((ROOT/'perceptual-mvtec-screwbag-new-spatial-anomaly-evidence-2026-10-25.json').read_text())
  cls.post=json.loads((ROOT/'perceptual-mvtec-screwbag-posthoc-paired-uncertainty-2026-10-25.json').read_text())
  cls.reg=json.loads((ROOT/'perceptual-analyzer-registry-2026-10-24.json').read_text())
  cls.owner=json.loads((ROOT/'perceptual-owner-reconciled-decision-2026-10-25.json').read_text())
 def test_actual_new_industrial_source_split_and_rights(self):
  d=self.a
  self.assertEqual(d['original_photo_counts'],{'train_normal':24,'validation_normal':20,'heldout_good':30,'heldout_logical':30,'heldout_structural':30,'new_source_photos_total':134})
  self.assertIn('screw_bag',d['source_release']) if 'screw_bag' in d['source_release'] else self.assertIn('Screw Bag',d['source_publication'])
  self.assertIn('NonCommercial',d['license'])
  self.assertFalse(d['GPU_allocated'])
  self.assertIn('NOT original official',d['architecture'])
 def test_exact_binary_counts_and_auc_each_source_role(self):
  wanted={'ResNet18_global_cosine_nearest_normal':(12,3,48,27),'ResNet18_spatial_gaussian_mahalanobis_max':(16,10,44,20),'ResNet18_spatial_gaussian_mahalanobis_p95':(16,3,44,27),'HSV_spatial_color_nearest_normal':(11,2,49,28)}
  self.assertEqual(set(self.a['heldout_original_source_tp_fp_fn_tn_and_CI']),set(wanted))
  for method,expected in wanted.items():
   entry=self.a['heldout_original_source_tp_fp_fn_tn_and_CI'][method]
   pooled=entry['all_anomalies_vs_clean'];logical=entry['logical_vs_clean'];structural=entry['structural_vs_clean']
   self.assertEqual(tuple(pooled[k] for k in ('TP','FP','FN','TN')),expected)
   self.assertEqual(logical['TP']+logical['FN'],30)
   self.assertEqual(structural['TP']+structural['FN'],30)
   self.assertEqual(pooled['TP'],logical['TP']+structural['TP'])
   self.assertEqual((pooled['TP']+pooled['FN'],pooled['FP']+pooled['TN']),(60,30))
   self.assertTrue(0<=pooled['test_AUROC']<=1)
   self.assertTrue(len(pooled['original_source_image_stratified_bootstrap95']['AUROC'])==2)
   self.assertTrue(len(pooled['source_exact95_recall'])==2)
   self.assertTrue(len(pooled['source_exact95_specificity'])==2)
   self.assertIn('20 original source normal',self.a['normal_calibration'])
 def test_comparison_is_exploratory_and_does_not_prove_spatial_superiority(self):
  post=self.post
  self.assertIn('POSTHOC',post['scientific_status'])
  self.assertEqual(post['source_photo_n'],{'test_normal':30,'test_logical':30,'test_structural':30})
  delta=post['posthoc_paired_by_source_image_AUC_deltas']['ResNet18_global_cosine_nearest_normal']
  self.assertLess(delta['same_original_source_image_pair_bootstrap_95pct'][0],0)
  self.assertGreater(delta['same_original_source_image_pair_bootstrap_95pct'][1],0)
  self.assertEqual(len(post['source_anomaly_positive_and_good_control_detection_discordances']),3)
 def test_sha_sources_and_original_no_model_gating(self):
  d=self.a
  self.assertEqual(len(d['research_provenance_SHA256']),5)
  self.assertTrue(all(len(x)==64 for x in d['research_provenance_SHA256'].values()))
  self.assertIn('normal',d['normal_calibration'].lower())
  self.assertEqual(d['pretrained_model_checkpoint'].split()[-1], 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec')
  self.assertTrue(d['owned_original_data_isolation'])
 def test_owned_registry_direct_answer_and_no_auto_release(self):
  key='new_independent_MVTec_screwbag_PaDiM_spatial_2026_10_25'
  self.assertIn(key,self.reg)
  self.assertTrue(self.reg[key]['status'].startswith('PARTIAL'))
  self.assertIn('posthoc',str(self.reg[key]['after_the_fact_paired_method_comparison']).lower())
  recent=[x for x in self.reg['analyzers'] if x.get('source_report')=='perceptual-mvtec-screwbag-padim-spatial-original-photos-2026-10-25.md']
  self.assertEqual(len(recent),4)
  self.assertTrue(all('NOT' in x['status'] for x in recent))
  self.assertIn('posthoc_uncertainty',self.owner['later_MVTec_screwbag_spatial_source_results'])
 def test_project_report_no_source_photo_bytes_or_raw_models(self):
  raw=(ROOT/'perceptual-mvtec-screwbag-new-spatial-anomaly-evidence-2026-10-25.json').read_text()
  for marker in ('original_media/', 'frozen_actual_image_only_features/', 'raw_test_image_embedding_data', 'data:image/png', 'tensor_state_dict', 'model_weights_encoded'):
   self.assertNotIn(marker,raw)
  self.assertLess(len(raw),28000)
if __name__=='__main__':unittest.main()
