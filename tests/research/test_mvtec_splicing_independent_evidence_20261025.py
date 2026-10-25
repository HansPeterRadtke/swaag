"""Check NEW original 134-photo MVTec industrial negative control benchmark.

These assertions ensure original train/normal val/test data partition, source
external labels, 3 pixel-only predeclared methods, 20 normal source calibration,
heldout 30 clean+30 logical+30 structural, exact TP/FP/FN/TN denominators,
source-independent external license bounds and normal-only thresholds. They do
NOT prove model quality or extrapolation to 3D CAD/machinery reliability.
Source original licensed images and pretrained model weights stay out of Git.
"""
from pathlib import Path
import unittest,json
ROOT=Path(__file__).resolve().parents[2]/'docs/research'
MODES=['CLIP_global_nearest_normal','HSV_spatial_hist_nearest_normal','CLIP_p95_patch_memory']

class IndependentIndustrialOriginalImages(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads((ROOT/'perceptual-mvtec-splicing-new-industrial-anomaly-evidence-2026-10-25.json').read_text())
  cls.registry=json.loads((ROOT/'perceptual-analyzer-registry-2026-10-24.json').read_text())
 def test_new_original_product_source_and_disjoint_original_splits(self):
  d=self.d;sample=d['sample']
  self.assertEqual(sample['train_source_good_original_images'],24)
  self.assertEqual(sample['normal_source_validation_images_used_for_95pct_score_cutoff'],20)
  self.assertEqual(sample['source_original_test_good'],30)
  self.assertEqual(sample['source_original_test_logical_anomalies'],30)
  self.assertEqual(sample['source_original_test_structural_anomalies'],30)
  self.assertEqual(sample['total_raw_original_source_images'],134)
  self.assertTrue(sample['source_train_validation_test_disjoint'])
  self.assertTrue(sample['selection_hash_frozen_before_model_inference'])
  self.assertIn('splicing connectors',d['source'])
  self.assertIn('CC BY-NC-SA',d['license'])
 def test_all_methods_use_same_nonanomalous_calibration_and_failures(self):
  d=self.d
  self.assertEqual(len(d['original_heldout_score_and_errors']),3)
  for method in MODES:
   self.assertIn(method,d['original_heldout_score_and_errors'])
   self.assertIn(method,d['frozen_normal_only_thresholds'])
   group=d['original_heldout_score_and_errors'][method]
   for label,npositive in [('logical_vs_real_clean',30),('structural_vs_real_clean',30),('all_source_anomalies_vs_original_clean',60)]:
    m=group[label]
    self.assertEqual(m['TP']+m['FN'],npositive)
    self.assertEqual(m['FP']+m['TN'],30)
    self.assertEqual(m['n'],npositive+30)
    self.assertEqual(len(m['image_stratified_bootstrap_95pct_ROC_AUC']),2)
    self.assertEqual(len(m['image_stratified_bootstrap_95pct_balanced_accuracy']),2)
    self.assertAlmostEqual(m['frozen_normal_only_threshold'],d['frozen_normal_only_thresholds'][method])
    self.assertTrue(0<=m['source_original_test_ROC_AUC']<=1)
    self.assertEqual(len(m['source_positive_recall_exact_binomial95_CI']),2)
    self.assertEqual(len(m['source_control_specificity_exact_binomial95_CI']),2)
 def test_real_resource_use_and_no_unjustified_cad_or_owner_render_claims(self):
  d=self.d
  self.assertFalse(d['local_resource']['GPU_used'])
  self.assertTrue(d['local_resource']['source_bytes_and_images_outside_Git'])
  self.assertGreater(d['local_resource']['pixel_feature_CPU_wall_s'],0)
  self.assertEqual(len(d['local_resource']['frozen_transformers_checkpoint_sha256']),64)
  self.assertIn('not proof of CAD',str(d['owner_general_relevance']))
  self.assertTrue(d['status'].startswith('PARTIAL'))
 def test_source_hash_evidence_and_project_registry_scope(self):
  d=self.d
  self.assertEqual(len(d['source_only_private_SHA256']),5)
  self.assertTrue(all(len(s)==64 for s in d['source_only_private_SHA256'].values()))
  k='original_industrial_MVTec_LOCO_splicing_visual_2026_10_25'
  self.assertIn(k,self.registry)
  self.assertIn('PARTIAL',self.registry[k]['status'])
  recent=[x for x in self.registry['analyzers'] if x.get('source_report')=='perceptual-mvtec-splicing-industrial-visual-review-2026-10-25.md']
  self.assertEqual(len(recent),3)
 def test_no_raw_images_or_embeddings_in_compact_source_json(self):
  b=(ROOT/'perceptual-mvtec-splicing-new-industrial-anomaly-evidence-2026-10-25.json').read_text()
  for raw in ['patch_embeddings', 'original_media/', 'frozen_original_image_only_embeddings/', 'base64,']:
   self.assertNotIn(raw,b)
  self.assertNotIn('image_bytes',self.d)
  self.assertNotIn('raw_image_pixels',self.d)
  self.assertNotIn('per_image_embeddings',self.d)
  self.assertLess(len(b),35000)
if __name__=='__main__':unittest.main()
