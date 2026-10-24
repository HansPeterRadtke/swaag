"""Small, bounded research-tool smoke tests; these do not certify model accuracy."""
import importlib.util
import json
import unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).parent
ROOT=HERE.parents[1]/'docs'/'research'

def load(script):
 spec=importlib.util.spec_from_file_location(script.stem,script)
 module=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module)
 return module

class PerceptualResearchContracts(unittest.TestCase):
 def test_prior_metadata_fixture_has_distinct_oracle(self):
  prior=load(HERE/'perceptual_cpu_fixtures.py')
  self.assertEqual(prior.EXPECTED['clipped'] if hasattr(prior,'EXPECTED') else prior.CASES['clipped'][0], 'clipped_text')
  self.assertEqual(prior.inspect([(1,1,401,30)],(0,0,0),(255,255,255)),'clipped_text')
  self.assertEqual(prior.inspect([(1,1,90,30),(85,22,140,53)],(0,0,0),(255,255,255)),'overlap')
 def test_new_pixels_and_gold_generated_separately(self):
  app=load(HERE/'perceptual_image_only_probe.py')
  self.assertEqual(app.gui_image('clean').size,(440,240))
  self.assertFalse(np.array_equal(np.asarray(app.gui_image('clipped_text')),np.asarray(app.gui_image('clean'))))
  self.assertEqual(len(app.render_video('camera_cut')[0]),16)
 def test_negative_cases_are_preserved_in_evidence(self):
  d=json.loads((ROOT/'perceptual-image-only-20261024/results.json').read_text())
  self.assertTrue(any(x['case']=='camera_cut' and x['gold']=='none' for x in d['video']['cases']))
  self.assertTrue(any(x['case']=='overprinted_connection' and x['gold']=='overprint' for x in d['diagram']['cases']))
  self.assertTrue(any(x['case']=='legitimate_recessed_track' and x['gold']=='none' for x in d['mechanical']['cases']))
  self.assertEqual(len(d['ocr_contract']['cases']),8)
 def test_real_asset_evidence_carries_unknown_ground_truth(self):
  d=json.loads((ROOT/'perceptual-real-assets-20261024/results.json').read_text())
  self.assertEqual(d['glb']['glb_version'],2)
  self.assertLess(d['video']['sampled_frames'],d['video']['frame_count'])
  self.assertIn('no timestamp ground truth',d['video']['limits'])
if __name__=='__main__':unittest.main()
