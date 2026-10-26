#!/usr/bin/env python3
"""Offline SOURCE-ATTESTED policy replay using real owner encoded video results.

This is NOT a live SWAAG tool/service integration. It imports the actual
conservative acceptance policy and checks source/video hashes independently,
then proves no unqualified CV/FFmpeg proposal can become auto-fail.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE=Path('/data/var/perceptual-independent-20261024/owner_real_encoded_fault_injection_20261026')
WORK=Path(__file__).resolve().parents[2]
POLICY=WORK/'src/swaag/perception/acceptance_policy.py'
SOURCE=Path('/data/var/3d-publication-20261024/approved_warehouse_door_rear_corner_v47.mp4')
EXPECTED_SHA='e2ef866ec79318a6b3812865a51922141ad02be66b555900da311171f49b4adb'
OUTPUT=BASE/'source_attested_policy_replay_20261026.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 if OUTPUT.exists():raise RuntimeError('Replay already frozen; no re-run')
 spec=importlib.util.spec_from_file_location('owner_frame_policy_20261026',POLICY)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 assert sha(SOURCE)==EXPECTED_SHA
 manifests=BASE/'source_receipts.json';encodes=BASE/'encodes.json';scored=BASE/'final_video_fault_score.json';browse=BASE/'browser_faulted_video_and_clean_control_20261026.json'
 receipt=json.loads(manifests.read_text());encoded=json.loads(encodes.read_text());r=json.loads(scored.read_text());b=json.loads(browse.read_text())
 assert receipt['video_sha256']==EXPECTED_SHA
 assert r['frozen_manifest_sha256']==sha(manifests) and r['frozen_encoded_variant_sha256']==sha(encodes)
 assert b['status']=='PASS'
 outcomes=[]
 for name,media in sorted(encoded['variants'].items()):
  assert sha(Path(media['path']))==media['sha256']
  case=r['cases'][name]
  assert case['file_SHA256']==media['sha256']
  frame_count=case['source_bound_detection']['TP']+case['source_bound_detection']['FP']
  property_name='approved_v47_ordered_frame_pixels_match_expected_source_under_28dB_threshold'
  actual_verified=(case['PTS']['count']==210 and case['PTS']['nonmonotonic']==0)
  obs=mod.Observation(property_name, 'verified_source_frame_SHA+decode+ffprobe', 'fail' if frame_count else 'pass', 'verified_source',reference_verified=True,actual_output_verified=actual_verified,measured_property=True)
  source_verdict=mod.decide_property([obs])
  model_has_alert=case['no_reference_low_motion_detection']['TP']+case['no_reference_low_motion_detection']['FP']>0
  image_obs=mod.Observation('unintended_visible_freeze','motion_anomaly_suspect','fail' if model_has_alert else 'inconclusive','visual_proposal',reference_verified=True,actual_output_verified=True,measured_property=False)
  model_verdict=mod.decide_property([image_obs])
  tail=case['official_ffmpeg_freezedetect']['events_in_authorized_last_3sec']
  assert tail==1
  ffobs=mod.Observation('camera_should_always_move','FFmpeg-freeze-at-approved-camera-hold','fail','visual_proposal',reference_verified=True,actual_output_verified=True,measured_property=False)
  freeze_verdict=mod.decide_property([ffobs])
  outcomes.append({'name':name,'gold_frame_edits':media['edit_positions'],'reference_verified':True,'final_video_sha_verified':True,'PTS_verified':actual_verified,'source_verified_property':property_name,'source_verified_verdict':source_verdict.value,'no_reference_video_suspect':model_has_alert,'no_reference_verdict':model_verdict.value,'official_ffmpeg_tail_suspect':tail,'official_ffmpeg_tail_property_verdict':freeze_verdict.value,'source_detections':frame_count})
 assert [(x['name'],x['source_verified_verdict']) for x in outcomes]==[('clean_reencoded','pass'),('short_reversals','fail'),('single_frame_holds','fail')]
 assert all(x['no_reference_verdict']=='review' and x['official_ffmpeg_tail_property_verdict']=='review' for x in outcomes)
 report={'status':'PARTIAL','method':'Actual source-backed acceptance-policy replay, not live SWAAG tool registration or real user-visible assistant tool call','policy_path':str(POLICY.relative_to(WORK)),'policy_code_sha256':sha(POLICY),'source_video_SHA':EXPECTED_SHA,'original_frozen_receipts_SHA':sha(manifests),'encoded_variants_receipt_SHA':sha(encodes),'final_frame_scores_SHA':sha(scored),'browser_playback_receipt_SHA':sha(browse),'verified_source_policy_outcomes':outcomes,'observations':'2 source-defined fail vs1 source-defined pass; 3/3 FFmpeg hold alarms remain review; no-reference ambiguity remains review. Only matched identity and source contract permits property-specific fail.','no_CV_or_VLM_auto_fail':True,'runtime_service_integration_tested':False,'raw_media_outside_git':True}
 tmp=OUTPUT.with_suffix('.tmp');tmp.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');tmp.replace(OUTPUT)
 print('OWNER_SOURCE_ATTESTED_POLICY_REPLAY',[(x['name'],x['source_verified_verdict'],x['no_reference_verdict']) for x in outcomes],'sha',sha(OUTPUT))

if __name__=='__main__':main()
