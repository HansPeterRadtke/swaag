"""Source-pinned output checks; do not substitute software tests for model validation."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'docs/research'

class SourceCalibratedEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=json.loads((ROOT/'perceptual-source-calibrated-evidence-2026-10-24.json').read_text())
 def test_real_axe_core_includes_abstentions(self):
  a=self.d['W3C_ACT_axe_core']
  self.assertEqual(a['pin_version'],'axe-core 4.14.0')
  self.assertEqual(a['executed'],32);self.assertEqual(a['scripts_safely_excluded'],2)
  c=a['truth_to_reported'];self.assertEqual(c['failed->failed'],7)
  self.assertEqual(c['failed->cannot_tell'],4)
  self.assertEqual(c['passed->inapplicable'],1)
 def test_human_machinery_gold_and_5_matched_model_runs(self):
  s=self.d['3D_DefectBench_human']
  self.assertEqual(s['silver_n_assets'],998)
  self.assertEqual(s['expert_adjudicated_n_assets'],99)
  self.assertEqual(len(s['models']),5)
  for model,stats in s['models'].items():
   x=stats['silver_mechanical_missing'];g=stats['expert_all_asset_missing'];m=stats['expert_mechanical_missing']
   self.assertEqual(x['n'],368);self.assertEqual(x['TP']+x['FN'],47)
   self.assertEqual(g['n'],99);self.assertEqual(g['TP']+g['FN'],31)
   self.assertEqual(m['n'],30);self.assertEqual(m['TP']+m['FN'],9)
 def test_model_false_alarms_and_failures_remain(self):
  d=self.d['3D_DefectBench_human']['models']
  self.assertEqual(d['gemini-2.5-pro']['silver_mechanical_missing']['FP'],110)
  self.assertEqual(d['qwen3-vl-235b']['silver_mechanical_missing']['FN'],35)
  self.assertEqual(d['gpt-5']['expert_all_asset_missing']['FN'],25)
  self.assertEqual(d['qwen3-vl-235b']['expert_all_asset_missing']['TP'],4)
 def test_expert_uncertainty_is_grouped_not_certain(self):
  d=self.d['3D_DefectBench_human']['expert_missing_parts_prompt_group_bootstrap_intervals']
  self.assertEqual(d['gpt-5']['all_3d_expert']['prompt_groups'],50)
  self.assertEqual(d['gemini-2.5-pro']['mechanical_expert']['prompt_groups'],15)
  self.assertLess(d['qwen3-vl-235b']['all_3d_expert']['recall_95pct_group_bootstrap'][0],.1)
 def test_screenvlm_small_crop_model_run_complete_but_failed_text(self):
  s=self.d['ScreenVLM_real_input_task_limited']
  self.assertEqual(s['repository_revision'],'78df0cc8498bbea1a03eb2816056fc0a76d2f63d')
  self.assertTrue(s['CPU_only']);self.assertGreater(s['source_model_bytes'],700000000)
  self.assertEqual(len(s['cases']),4)
  self.assertTrue(all(c['status']=='completed' for c in s['cases']))
  self.assertEqual(s['expected_labels_recovered'],0)
  self.assertEqual(s['complete_closed_screen_tags'],2)
  self.assertAlmostEqual(s['sum_CPU_seconds'],1091.82,places=2)
  self.assertGreater(s['observed_peak_rss_kb'],2000000)
 def test_source_license_bytes_not_embedded(self):
  t=(ROOT/'perceptual-source-calibrated-evidence-2026-10-24.json').read_text()
  self.assertNotIn('model.safetensors',t)
  self.assertNotIn('glb/golden/',t)
  self.assertNotIn('tokenizer.json',t)
if __name__=='__main__':unittest.main()
