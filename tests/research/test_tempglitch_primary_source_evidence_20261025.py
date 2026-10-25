"""Validate original 1500-source and 12-VLM published evidence, not model QA.

No media, GPU, network or inference. The expected counts came from the original
TempGlitch arxiv HTML source Table3 and Appendix E EXACT author TP/TN/FP/FN,
independently recalculated by source snapshot script; metadata-only test used
complete HuggingFace original 1500 source video listing without decoding clips.
These checks prove artifact consistency only, not deployment reliability.
"""
from pathlib import Path
import json,unittest
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class TempGlitchEvidenceIntegrity(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.a=json.loads((ROOT/'perceptual-tempglitch-1500-source-evidence-2026-10-25.json').read_text())
  cls.registry=json.loads((ROOT/'perceptual-analyzer-registry-2026-10-24.json').read_text())
 def test_all_original_author_source_dataset_and_not_human_mos(self):
  a=self.a
  self.assertEqual(a['actual_author_evaluation']['dataset_video_n'],1500)
  self.assertEqual(a['actual_author_evaluation']['n_clean_original'],750)
  self.assertEqual(a['actual_author_evaluation']['n_glitchy_injected'],750)
  self.assertEqual(a['actual_author_evaluation']['n_vlms'],12)
  self.assertEqual(a['actual_author_evaluation']['n_model_input_settings_scored'],22)
  self.assertTrue(a['independent_video_download_or_local_VLM_inference'] is False)
  self.assertTrue(a['complete_real_source_records_not_committed_to_git'])
 def test_original_paper_winning_by_accuracy_still_misses_majority(self):
  x=self.a['actual_author_evaluation']['best_overall']
  self.assertEqual(x['model'],'GPT 5.4 Mini')
  self.assertEqual((x['true_positives'],x['false_positives'],x['false_negatives'],x['true_negatives']),(173,112,577,638))
  self.assertLess(x['accuracy_pct'],55)
 def test_qwen_and_gemini_are_opposite_failure_modes(self):
  x=self.a['actual_author_evaluation']['selected_model_exact_source_matrices']
  q=x['Qwen3VL 8B']['settings']['sampled_5fps']
  g=x['Gemini 3 Flash']['settings']['native_raw_video']
  self.assertEqual([q[k] for k in ['TP','FN','FP','TN']],[2,748,0,750])
  self.assertEqual([g[k] for k in ['TP','FN','FP','TN']],[654,96,650,100])
  self.assertGreater(g['false_positive_rate_pct'],85)
  self.assertGreater(q['false_negative_rate_pct'],99)
 def test_f1_all_positive_trivial_reference(self):
  t=self.a['actual_author_evaluation']['trivial_all_glitch_policy']
  self.assertEqual([t[k] for k in ['TP','FP','FN','TN']],[750,750,0,0])
  self.assertAlmostEqual(t['f1_pct'],200/3,places=3)
  self.assertAlmostEqual(t['accuracy_pct'],50.0)
 def test_complete_1500_metadata_pair_integrity_and_no_vision_inference(self):
  d=self.a['full_original_video_file_metadata_independent_audit']
  self.assertEqual(d['all_source_files'],1500)
  self.assertEqual(d['original_category_matched_pairs'],750)
  self.assertIn('75 Platformer',d['Stuck_in_Place_environment_reused_numeric_IDs'])
  held=d['metadata_size_only_source_pair_heldout_model']
  self.assertEqual((held['n_heldout_source_clips'],held['n_test_buggy'],held['n_test_normal']),(320,160,160))
  self.assertEqual([held[k] for k in ['TP','FP','FN','TN']],[87,66,73,94])
  self.assertAlmostEqual(held['test_roc_auc'],.59890625)
  low,hi=held['pair_group_bootstrap_2500_source95pct_AUROC']
  self.assertLess(.50,low)
  self.assertLess(low,hi)
  self.assertLess(hi,.70)
 def test_model_source_license_and_no_deployment_claim(self):
  a=self.a
  self.assertIn('MIT',a['source'])
  self.assertIn('not blind independent',str(a['limitations']).lower())
  key='TempGlitch_1500_original_full_source_model_accuracy_2026_10_25'
  self.assertIn(key,self.registry)
  self.assertTrue(self.registry[key]['status'].startswith('PARTIAL'))
  recent=[m for m in self.registry['analyzers'] if m.get('source_report')=='perceptual-tempglitch-primary-video-evidence-2026-10-25.md']
  self.assertEqual(len(recent),2)
  self.assertTrue(all('NOT_' in x['status'] or 'NOT' in x['status'] for x in recent))
  self.assertEqual(len(a['raw_research_snapshots_sha256']),2)
  self.assertEqual(len(a['primary_arxiv_html_source_sha256']),64)
if __name__=='__main__':unittest.main()
