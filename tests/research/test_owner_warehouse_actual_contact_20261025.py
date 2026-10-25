"""Verify real saved-scene mechanical proof provenance; not a QA pass by itself.

Consumes only the published source-linked aggregate from the actual owner
warehouse v30/v44/v47 Blender mesh ray experiments. No user videos, 400MB
Blender assets, GPU, live rendering or private runtime needed in CI.
"""
from pathlib import Path
import json,unittest
DOC=Path(__file__).resolve().parents[2]/'docs/research'

class OriginalWarehouseGeometryEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.a=json.loads((DOC/'perceptual-owner-warehouse-physical-contact-chain-2026-10-25.json').read_text())
  cls.registry=json.loads((DOC/'perceptual-analyzer-registry-2026-10-24.json').read_text())
 def test_24_actual_objects_and_stage_identity_consistency(self):
  x=self.a
  self.assertEqual(x['actual_source_scene_counts'],{'pallets':24,'steel_bearing_bars':72,'rack_bays':12})
  d=x['version_controlled_geometric_identity']
  self.assertEqual(d['pallet_world_triangle_meshes_identical_pre_v30_through_v47'],24)
  self.assertEqual(d['original_racks_same_v30_v44'],12)
  self.assertEqual(d['original_racks_CHANGED_world_geometry_v44_to_v47'],12)
  self.assertEqual(d['steel_bearing_bars_same_v44_v47_world_AABB'],72)
 def test_dangerous_partial_stage_vertical_only_false_acceptance(self):
  mid=self.a['real_intermediate_v44']
  self.assertEqual(mid['wood_to_steel_contact_bars_within_2mm'],72)
  self.assertEqual(mid['source_steel_bars_with_rack_contact_at_LEFT_end'],0)
  self.assertEqual(mid['source_steel_bars_with_rack_contact_at_RIGHT_end'],0)
  self.assertEqual(mid['source_steel_bars_with_full_wood_bearing_rack_contact_chain'],0)
 def test_corrected_full_source_triangle_geometric_contact_not_structural_strength(self):
  x=self.a['real_corrected_v47']
  self.assertEqual(x['wood_to_steel_contact_bars_within_2mm'],72)
  self.assertEqual(x['original_rack_triangle_geometric_hits_at_both_ends'],72)
  self.assertEqual(x['original_rack_left_end_hits'],72)
  self.assertEqual(x['original_rack_right_end_hits'],72)
  self.assertEqual(x['sampled_rack_endpoint_rays'],432)
  self.assertAlmostEqual(x['wood_to_steel_sampled_min_vertical_gap_m'],-.0003,places=6)
  self.assertEqual(x['dynamic_sampled_source_meshes_unchanged_per_frame'],108)
  self.assertEqual(x['sampled_scene_frame_count'],5)
  self.assertTrue(x['no_animation_data_on_these_108_mesh_objects'])
 def test_source_rig_contrast_not_fabricated_independent_samples(self):
  x=self.a
  self.assertTrue(x['source_provenance']['v47_matches_preexisting_independent_camera_source_receipt'])
  self.assertEqual(x['original_v30_pre_repair']['real_steel_bearing_bars'],0)
  self.assertEqual(x['original_v30_pre_repair']['predefined_endpoint_source_rack_rays'],144)
  self.assertEqual(x['original_v30_pre_repair']['rack_end_location_wood_to_rack_ray_hits'],0)
  self.assertIn('not a single-variable pair',x['original_v30_pre_repair']['limitation'])
  self.assertTrue(any('not' in y.lower() or 'no' in y.lower() for y in x['what_is_NOT_proved']))
  self.assertFalse(x['source_assets_STAY_OUTSIDE_GIT'] is False)
 def test_runtime_and_evidence_not_mistaken_for_general_vision_model(self):
  self.assertTrue(self.a['no_render_no_shared_GPU_no_model_called'])
  self.assertEqual(len(self.a['raw_unmodified_results_SHA256']),5)
  self.assertTrue(all(len(s)==64 for s in self.a['raw_unmodified_results_SHA256'].values()))
  k='actual_warehouse_3d_support_chain_2026_10_25'
  self.assertIn(k,self.registry)
  rec=[x for x in self.registry['analyzers'] if x.get('source_report')=='perceptual-direct-verdict-and-real-warehouse-contact-2026-10-25.md']
  self.assertEqual(len(rec),1)
  self.assertIn('one_original_real_asset_ONLY',rec[0]['status'])
  self.assertIn('PARTIAL',self.registry[k]['status'])
if __name__=='__main__':unittest.main()
