"""Integrity checks for new genuine source image/graph evaluation, NOT model QA.

This uses only compact project source reports; raw licensed original images,
OCR tokens and curated graph annotations remain outside Git. Tests verify exact
false positives/negatives, external dataset denominator, OCR-unknown coverage,
independent paired image CI, model-facing image-only feature extraction and
absent arrowhead confidence. Not a substitute for independent human quality.
"""
from pathlib import Path
import unittest,json
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class SourceBoundedRealDiagramTopologyEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads((ROOT/'perceptual-topobench-69-original-diagram-connector-evidence-2026-10-25.json').read_text())
  cls.registry=json.loads((ROOT/'perceptual-analyzer-registry-2026-10-24.json').read_text())
  cls.owner=json.loads((ROOT/'perceptual-owner-reconciled-decision-2026-10-25.json').read_text())
 def test_original_source_real_images_and_license_restrictions(self):
  d=self.d;s=d['source_annotation_counts'] if 'source_annotation_counts' in d else None
  self.assertIn('sixty-nine legally',d['source'])
  self.assertTrue(d['original_copyright_redistribution_risk_93_images_not_requested'])
  self.assertFalse(d['original_GPU_used'])
  self.assertTrue(d['source_graph_ground_truth_caveat'])
  self.assertGreater(d['original_pixel_ocr_elapsed_seconds'],100)
  self.assertEqual(len(d['source_SHA256']),3)
  self.assertTrue(all(len(h)==64 for h in d['source_SHA256'].values()))
 def test_hough_conditional_vs_source_nearest_neighbor_on_identical_pairs(self):
  a=self.d['PSM_both_predeclared_settings']['11']
  self.assertEqual(a['total_original_annotated_source_nodes'],740)
  self.assertEqual(a['unique_ocr_grounded_source_nodes'],229)
  self.assertEqual(a['source_annotated_undirected_connections_all_diagrams'],966)
  self.assertEqual(a['scorable_annotated_connections_both_node_labels_found'],198)
  self.assertEqual(a['unscorable_source_connections_due_ambiguous_or_missing_OCR_anchor'],768)
  self.assertEqual(a['actual_annotated_negative_nonconnection_controls_scored'],471)
  self.assertEqual(a['source_diagrams_with_at_least_one_grounded_pair'],38)
  x=a['source_conditional_hough_binary'];y=a['source_conditional_layout_nearest_neighbor_null']
  self.assertEqual([x[q] for q in ['TP','FP','FN','TN']],[151,158,47,313])
  self.assertEqual([y[q] for q in ['TP','FP','FN','TN']],[93,55,105,416])
  self.assertEqual(x['TP']+x['FN'],a['scorable_annotated_connections_both_node_labels_found'])
  self.assertAlmostEqual(x['precision'],151/309)
  self.assertGreater(a['source_line_score_ROC_AUC'],.75)
  ci=a['source_diagram_bootstrap95']['CI95']['Hough_AUROC_minus_layout_nearest_AUROC']
  self.assertGreater(ci[0],0)
 def test_independent_second_psm_is_not_overwritten_by_best_psm(self):
  self.assertEqual(set(self.d['PSM_both_predeclared_settings']),{'11','6'})
  b=self.d['PSM_both_predeclared_settings']['6']
  self.assertEqual((b['unique_ocr_grounded_source_nodes'],b['source_diagrams_with_at_least_one_grounded_pair']),(173,31))
  x=b['source_conditional_hough_binary']
  self.assertEqual([x[q] for q in ['TP','FP','FN','TN']],[102,124,38,210])
  self.assertEqual(b['scorable_annotated_connections_both_node_labels_found'],140)
  self.assertEqual(b['source_annotated_undirected_connections_all_diagrams'],966)
 def test_posthoc_continuous_distance_control_substantively_weakens_hough_claim(self):
  x=self.d['later_posthoc_continuous_distance_control']
  self.assertIn('POSTHOC',x['status'])
  self.assertEqual(len(x['source_saved_raw_sha256']),64)
  a=x['PSM11'];b=x['PSM6']
  self.assertEqual(a['original_node_pairs_both_anchors'],669)
  self.assertAlmostEqual(a['AUROC']['score_hough'],.7832732848656415)
  self.assertAlmostEqual(a['AUROC']['score_geometry_distance'],.7260074202749363)
  self.assertGreater(a['source_diagram_bootstrap_95pct_delta'][0],0)
  self.assertLess(a['Hough_AUROC_minus_CONTINUOUS_distance_AUROC'],.07)
  self.assertTrue(b['source_diagram_bootstrap_95pct_delta'][0]<0<b['source_diagram_bootstrap_95pct_delta'][1])
  self.assertTrue(x['PSM11']['no_model_training_or_threshold_selection'])
 def test_arrow_direction_not_inferrable_from_hough_segments(self):
  a=self.d['PSM_both_predeclared_settings']['11']['layout_reading_order_guessed_arrow_orientation_NOT_measured_vision']
  self.assertEqual((a['direction_right_under_layout_bias'],a['direction_wrong_under_layout_bias']),(80,67))
  self.assertEqual(a['hough_detected_unambiguous_directed_connections'],147)
  self.assertTrue(a['this_is_NOT_arrowhead_analysis'])
  self.assertGreater(a['two_sided_binomial_p_vs_half'],.05)
  self.assertTrue(a['exact_binomial_95pct_accuracy_CI'][0]<.5<a['exact_binomial_95pct_accuracy_CI'][1])
  b=self.d['PSM_both_predeclared_settings']['6']['layout_reading_order_guessed_arrow_orientation_NOT_measured_vision']
  self.assertEqual((b['direction_right_under_layout_bias'],b['hough_detected_unambiguous_directed_connections']),(44,100))
 def test_existing_registry_and_owner_conclusions_preserve_unknowns(self):
  k='TopoBench_69_original_diagram_undirected_connector_pixel_2026_10_25'
  self.assertIn(k,self.registry)
  self.assertIn('PARTIAL',self.registry[k]['state'])
  self.assertIn('NO_DIRECTED_ARROW',next(a['status'] for a in self.registry['analyzers'] if a.get('source_report')=='perceptual-topobench-real-directed-connector-research-2026-10-25.md'))
  x=self.owner['later_independent_real_diagram_connector_evidence']
  self.assertEqual([x['observed_undirected_connector_conditional'][s] for s in ['TP','FP','FN','TN']],[151,158,47,313])
  self.assertTrue(self.owner['status'].startswith('PARTIAL'))
  self.assertIn('UNCERTAIN',self.owner['calendar_delivery_status'])
 def test_only_compact_project_outputs_no_embedded_pixels(self):
  text=(ROOT/'perceptual-topobench-69-original-diagram-connector-evidence-2026-10-25.json').read_text()
  for forbidden in ('raw_tesseract_ocr','hough_line_segments','image_bytes','graph_annotation_raw','data:application/octet-stream'):
   self.assertNotIn(forbidden,text)
  self.assertLess(len(text),60000)
if __name__=='__main__':unittest.main()
