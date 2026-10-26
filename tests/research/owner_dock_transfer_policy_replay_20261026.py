#!/usr/bin/env python3
"""Actual SWAAG policy module replay on frozen different-source DOCK H264 cases.

Pure research offline, never invokes the production-facing analyzer service.
No CV/FFmpeg proposal can be promoted to verified failure automatically.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE=Path('/data/var/perceptual-independent-20261024/owner_dock_cross_source_transfer_20261026')
WORK=Path(__file__).resolve().parents[2]
POLICY=WORK/'src/swaag/perception/acceptance_policy.py'
SOURCE=Path('/data/var/3d-publication-20261024/approved_dock_cinematic_preview_v46.mp4')
SHA='75ef01e7316d099759f4e1dccec304f4c3ea8a995949deebe7f6b41962c6ef93'
DEST=BASE/'actual_policy_replay.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 if DEST.exists():raise RuntimeError('Policy replay receipt exists')
 spec=importlib.util.spec_from_file_location('transfer_dock_acceptance',POLICY)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 manifest=json.loads((BASE/'source_manifest.json').read_text());scored=json.loads((BASE/'transfer_scored.json').read_text());enc=json.loads((BASE/'encodes.json').read_text())
 assert scored['source_manifest_SHA']==sha(BASE/'source_manifest.json')
 assert manifest['dock_MP4_SHA']==sha(SOURCE)==SHA
 assert scored['encoded_variants_SHA']==sha(BASE/'encodes.json')
 results=[]
 for name,entry in sorted(enc['variants'].items()):
  assert sha(Path(entry['path']))==entry['sha256']
  c=scored['cases'][name];assert c['real_encoded_video_sha256']==entry['sha256']
  same_version=manifest['dock_scene_SHA']==scored['reference_dock_blend_SHA']
  final_provenance_valid=bool(same_version and c['PTS']['count']==210 and c['PTS']['nonmonotonic']==0)
  source_alerts=c['source_bound']['TP']+c['source_bound']['FP']
  source=mod.Observation('dock_approved_ordered_source_frames',f'exact_reference_dock_SHA_28dB_{name}','fail' if source_alerts else 'pass','verified_source',reference_verified=same_version,actual_output_verified=final_provenance_valid,measured_property=True)
  source_result=mod.decide_property([source])
  cv=mod.Observation('unwanted_video_motion','generic_low_motion_detector','fail' if c['no_reference']['TP']+c['no_reference']['FP'] else 'inconclusive','visual_proposal',reference_verified=True,actual_output_verified=True,measured_property=False)
  cv_result=mod.decide_property([cv])
  bad_original=mod.Observation('dock_approved_ordered_source_frames','warehouse_reference_wrong_product','fail','verified_source',reference_verified=False,actual_output_verified=True,measured_property=True)
  bad_result=mod.decide_property([bad_original])
  results.append({'case':name,'gold_frame_positions':entry['original_source_frame_override_positions'],'verified_frame_sequence_property':source_result.value,'generic_motion_proposal':cv_result.value,'invalid_warehouse_reference':bad_result.value,'source_valid':same_version,'final_output_provenance_valid':final_provenance_valid})
 assert {x['case']:x['verified_frame_sequence_property'] for x in results}=={'clean_crf17':'pass','clean_crf23':'pass','holds_crf17':'fail','reversals_crf17':'fail'}
 assert all(x['generic_motion_proposal']=='review' and x['invalid_warehouse_reference']=='review' for x in results)
 out={'status':'NARROW_OFFLINE_POLICY_REPLAY_PASS','source_dock_mp4_sha256':SHA,'policy_source_sha256':sha(POLICY),'verified_original_dock_source_manifest_sha256':sha(BASE/'source_manifest.json'),'frozen_transfer_scored_SHA256':sha(BASE/'transfer_scored.json'),'cases':results,'limits':'Off-line source-based verified property replay only. Live SWAAG tool and user interaction not exercised; unqualified visual cues REVIEW.'}
 tmp=DEST.with_suffix('.working');tmp.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');tmp.rename(DEST)
 print('DOCK_POLICY_REPLAY',[(z['case'],z['verified_frame_sequence_property'],z['generic_motion_proposal']) for z in results],'sha',sha(DEST))
if __name__=='__main__':main()
