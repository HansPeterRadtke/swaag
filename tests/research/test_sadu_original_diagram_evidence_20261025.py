"""Integrity checks for independently scored real SADU diagrams, not QA passes."""
from pathlib import Path
import json,unittest
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class SADUSourceDiagramEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads((ROOT/'perceptual-sadu-154-real-diagrams-evidence-2026-10-25.json').read_text())
  cls.registry=json.loads((ROOT/'perceptual-analyzer-registry-2026-10-24.json').read_text())
 def test_independently_grounded_original_diagrams_and_qa_not_synthetic(self):
  s=self.d['source_original_structure']
  self.assertEqual((s['real_diagrams'],s['behavior'],s['structural'],s['ER']),(154,51,53,50))
  self.assertEqual(s['total_archived_QA_predictions_scored'],26741)
  self.assertEqual(s['independent_numeric_source_graph_assertions'],924)
  self.assertEqual(s['independent_numeric_assertion_mismatches'],0)
  self.assertEqual(len(self.d['primary_metrics']),11)
 def test_output_correctness_and_false_relations_are_both_preserved(self):
  m=self.d['primary_metrics'];self.assertEqual(m['gemini-3-flash-preview']['strict_correct_QA_n'],1743)
  self.assertEqual(m['gemini-2.5-flash']['strict_correct_QA_n'],1692)
  self.assertEqual(m['qwen_2.5-VL-32B']['strict_correct_QA_n'],1078)
  self.assertGreater(m['gemini-3-flash-preview']['retrieval_false_negative_source_entities'],1000)
  self.assertGreater(m['gemini-2.5-flash']['retrieval_false_positive_source_entities'],300)
  self.assertGreater(m['gemini-3-flash-preview']['format_or_answer_type_invalid'],200)
  self.assertLess(m['gemini-3-flash-preview']['strict_source_QA_accuracy'],.8)
 def test_qwen_indices_reordered_and_source_identity_matched(self):
  a=self.d['authored_index_alignment']
  for s in ['qwen_2.5-VL-32B','qwen_2.5-VL-3B','qwen_2.5-VL-7B']:
   self.assertEqual(a[s],2431)
  self.assertEqual(a['gemini-3-flash-preview'],0)
  self.assertIn('format',str(self.d['limitations']).lower())
 def test_model_selection_not_confused_with_question_only_free_prose(self):
  a=self.d['prompt_and_reasoning_ablation']
  self.assertTrue(a['strict_only_question_zero_is_prompt_contract_mismatch_NOT_model_comprehension'])
  self.assertEqual(len(a['source_question_only_not_strictly_scored']),4)
  r=a['valid_JSON_comparable']
  self.assertAlmostEqual(r['gemini3_full_low_thinking']['overall'],.71945701357)
  self.assertAlmostEqual(r['gemini3_full_default']['overall'],.71698889345)
  self.assertLess(r['gemini2p5_without_definition']['overall'],r['gemini2p5_full_default']['overall'])
 def test_ocr_recall_mixed_domains_not_automatic_diagram_qa(self):
  s=self.d['pixel_only_Tesseract_vs_entity_names']
  all=s['all'];self.assertEqual((all['total_diagrams'],all['scored_diagrams'],all['ocr_abstain_or_timeout']),(154,154,0))
  self.assertEqual((all['author_annotated_unique_entity_labels'],all['exact_labels_recognized']),(1733,820))
  self.assertGreater(s['per_domain']['ER']['exact_entity_name_recall'],.9)
  self.assertLess(s['per_domain']['structural']['exact_entity_name_recall'],.32)
  self.assertLess(s['per_domain']['behavior']['exact_entity_name_recall'],.41)
  self.assertIn('No false positive precision',s['note'])
 def test_registry_keeps_exact_scope_and_orig_media_not_committed(self):
  k='SADU_real_154_human_source_diagrams_2026_10_25'
  self.assertIn(k,self.registry)
  self.assertEqual(self.registry[k]['status'].split(',')[0],'PARTIAL')
  rec=[x for x in self.registry['analyzers'] if x.get('source_report')=='perceptual-real-diagram-benchmark-2026-10-25.md']
  self.assertEqual(len(rec),3)
  self.assertIn('not_rendered_visual',rec[2]['status'])
  self.assertEqual(self.d['overall_status'].split(' —')[0],'PARTIAL')
if __name__=='__main__':unittest.main()
