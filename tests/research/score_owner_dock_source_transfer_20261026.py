#!/usr/bin/env python3
"""Score pre-frozen warehouse->dock transfer on actual final H264 video pixels.

No human-rated natural faults are claimed; positives are script-authored edits.
Both sources and all generated output videos are read-only after freeze.
"""
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import cv2
import numpy as np
from scipy.stats import beta

BASE=Path('/data/var/perceptual-independent-20261024')
ROOT=BASE/'owner_dock_cross_source_transfer_20261026'
PNG=Path('/data/var/dock-film-v46/continuous_real_camera_960x540_32samples')
MOVIE=Path('/data/var/3d-publication-20261024/approved_dock_cinematic_preview_v46.mp4')
OTHER=BASE/'owner_real_encoded_fault_injection_20261026'
FPS=15
OUTPUT=ROOT/'transfer_scored.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def read(p):return json.loads(p.read_text())

def once(p,obj):
 if p.exists():raise RuntimeError('Frozen results exist, do not rerun')
 t=p.with_suffix('.tmpnew');t.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');t.rename(p)

def cp_ci(k,n):
 if n==0:return None
 return [round(0. if k==0 else float(beta.ppf(.025,k,n-k+1)),5),round(1. if k==n else float(beta.ppf(.975,k+1,n-k)),5)]

def decode(p):
 start=time.monotonic()
 cap=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG)
 if not cap.isOpened():raise RuntimeError('Decoder unavailable '+str(p))
 frames=[]
 try:
  while True:
   ok,arr=cap.read()
   if not ok:break
   if arr.shape!=(540,960,3):raise RuntimeError('unexpected resolution')
   frames.append(arr)
   if len(frames)>210:raise RuntimeError('extra frames')
 finally:cap.release()
 if len(frames)!=210:raise RuntimeError('missing frames '+str(len(frames)))
 return frames,round(time.monotonic()-start,3)

def pts(p):
 t=time.monotonic()
 x=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(p)],capture_output=True,text=True,timeout=45,check=True)
 times=[float(x['best_effort_timestamp_time']) for x in json.loads(x.stdout)['frames']]
 if len(times)!=210:raise RuntimeError('timestamp frame-count error')
 return {'count':len(times),'max_error_vs_15fps_sec':max(abs(v-i/15) for i,v in enumerate(times)),'nonmonotonic':sum(b<=a for a,b in zip(times,times[1:])),'wall_s':round(time.monotonic()-t,3)}

def freeze_check(p):
 t=time.monotonic()
 x=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','info','-i',str(p),'-vf','freezedetect=n=-50dB:d=0.12','-f','null','-'],capture_output=True,text=True,timeout=80)
 if x.returncode:raise RuntimeError('FFmpeg freezedetect rc nonzero '+x.stderr[-450:])
 starts=[float(n) for n in re.findall(r'freeze_start:\s*([\d.]+)',x.stderr)]
 return {'starts_seconds':starts,'events_total':len(starts),'starts_in_first_165_frames':sum(s<11.0 for s in starts),'wall_s':round(time.monotonic()-t,3)}

def confusion(rows,gold,key):
 return {part:sum((r['frame'] in gold)==isgold and r[key]==pred for r in rows) for part,isgold,pred in (('TP',True,True),('FN',True,False),('FP',False,True),('TN',False,False))}

def main():
 if OUTPUT.exists():raise RuntimeError('FROZEN_ALREADY')
 protocol=read(ROOT/'frozen_protocol.json');manifest=read(ROOT/'source_manifest.json');canonical_rec=read(ROOT/'canonical_source_decoded.json');enc=read(ROOT/'encodes.json')
 assert sha(ROOT/'frozen_protocol.json')==manifest['protocol_sha256']==enc['frozen_protocol_sha256']
 assert sha(ROOT/'canonical_source_decoded.json')==enc['canonical_source_decoded_SHA256']
 assert protocol['original_approved_dock_final_MP4_SHA']==sha(MOVIE)==manifest['dock_MP4_SHA']
 assert len(manifest['original_210_source_pngs'])==210
 start=time.monotonic();scene_frames=[];canonical=[]
 for i,r in enumerate(manifest['original_210_source_pngs'],1):
  original=PNG/f'frame_{i:04d}.png';cached=ROOT/'canonical_dock_final_decode'/f'frame_{i:04d}.png'
  assert sha(original)==r['sha256'] and sha(cached)==canonical_rec['decoded_frame_sha256'][str(i)]
  a=cv2.imread(str(original),cv2.IMREAD_COLOR);b=cv2.imread(str(cached),cv2.IMREAD_COLOR)
  assert a is not None and b is not None and a.shape==b.shape==(540,960,3)
  scene_frames.append(a);canonical.append(b)
 decode_input_seconds=round(time.monotonic()-start,3)
 source_match_psnr=[float(cv2.PSNR(a,b)) for a,b in zip(canonical,scene_frames)]
 # Wrong-product control of 12 fixed positions requires refusal even with matching format.
 external=read(OTHER/'source_frames.json');wrong_idx=(1,15,30,45,60,75,90,105,120,135,150,180)
 wrong_source_psnr=[]
 for i in wrong_idx:
  file=OTHER/'canonical_decoded_png'/f'frame_{i:04d}.png'
  assert sha(file)==external['exact_png_hashes'][str(i)]
  x=cv2.imread(str(file),cv2.IMREAD_COLOR)
  if x is None or x.shape!=canonical[i-1].shape:raise RuntimeError('wrong-source control dimension error')
  wrong_source_psnr.append(float(cv2.PSNR(canonical[i-1],x)))
 allcases={}
 gold_edits=protocol['variants_with_original_1based_source_frame_mappings']
 for name,item in enc['variants'].items():
  p=Path(item['path']);assert sha(p)==item['sha256']
  frames,decsec=decode(p)
  t=time.monotonic();rows=[];last=None
  for idx,(a,b) in enumerate(zip(frames,canonical),1):
   ps=float(cv2.PSNR(a,b))
   current=cv2.cvtColor(cv2.resize(a,(240,135),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
   motion=float(cv2.mean(cv2.absdiff(current,last))[0]) if last is not None else None
   rows.append({'frame':idx,'PSNR_dB':round(ps,5),'source_fail':ps<28.,'no_reference_proposal':bool(idx>1 and idx<=165 and motion<1.) if motion is not None else False,'gray_motion_240':round(motion,4) if motion is not None else None})
   last=current
  feat_s=round(time.monotonic()-t,3)
  video_pts=pts(p);ff=freeze_check(p)
  gold={int(x) for x in gold_edits[name]}
  source=confusion(rows,gold,'source_fail');no_ref=confusion(rows,gold,'no_reference_proposal')
  any_low=sum(x['no_reference_proposal'] for x in rows)
  scene_pixel_motion_median=round(statistics.median(x['gray_motion_240'] for x in rows[1:] if x['gray_motion_240'] is not None),5)
  allcases[name]={'real_encoded_video_sha256':item['sha256'],'codec_crf':item['H264_CRF'],'known_frozen_edit_positions':sorted(gold),'source_bound':source,'no_reference':no_ref,'source_bound_alarmed_positions':[x['frame'] for x in rows if x['source_fail']],'no_reference_alarmed_positions':[x['frame'] for x in rows if x['no_reference_proposal']],'source_bound_PSN_min_dB':round(min(x['PSNR_dB'] for x in rows),4),'source_bound_PSN_median_dB':round(statistics.median(x['PSNR_dB'] for x in rows),4),'source_bound_PSN_max_dB':round(max(x['PSNR_dB'] for x in rows),4),'moving_camera_motion_median':scene_pixel_motion_median,'official_freezedetect':ff,'PTS':video_pts,'actual_decode_wall_s':decsec,'same_frames_both_rules_wall_s':feat_s,'all_encoded_frame_scores':rows}
  print('TRANSFER_CASE',name,'FROZEN_GOLD',sorted(gold),'SOURCE',source,'NOREF',no_ref,'ffmpeg',ff['starts_seconds'],'PSNRmin',allcases[name]['source_bound_PSN_min_dB'],'CPU',round(decsec+feat_s,2),flush=True)
 all_src={m:sum(x['source_bound'][m] for x in allcases.values()) for m in ('TP','FN','FP','TN')}
 all_noref={m:sum(x['no_reference'][m] for x in allcases.values()) for m in ('TP','FN','FP','TN')}
 clean=['clean_crf17','clean_crf23']
 source_clean_fp=sum(allcases[name]['source_bound']['FP'] for name in clean)
 acceptance={'clean_original_source_min_PSN_28dB':min(source_match_psnr)>=28.,'two_reencodes_zero_source_false_flags':source_clean_fp==0,'detect_at_least_8_of_9_scripted_change_positions':all_src['TP']>=8,'not_more_than_6_FP_all_other_positions':all_src['FP']<=6,'PTS_all_outputs_210_monotonic_correct':all(x['PTS']['count']==210 and x['PTS']['nonmonotonic']==0 and x['PTS']['max_error_vs_15fps_sec']<1e-5 for x in allcases.values())}
 passed=all(acceptance.values())
 output={'status':'PASS_NARROW_TRANSFER' if passed else 'FAIL_PREDECLARED_TRANSFER','source':'different source dock v46, not warehouse v47; same H264 960x540@15fps protocol','reference_original_dock_MP4_SHA':protocol['original_approved_dock_final_MP4_SHA'],'reference_dock_blend_SHA':manifest['dock_scene_SHA'],'negative_original_other_warehouse_SHA':protocol['negative_other_product_WAREHOUSE_SHA'],'frozen_protocol_SHA':sha(ROOT/'frozen_protocol.json'),'source_manifest_SHA':sha(ROOT/'source_manifest.json'),'canonical_decoded_SHA':sha(ROOT/'canonical_source_decoded.json'),'encoded_variants_SHA':sha(ROOT/'encodes.json'),'previous_warehouse_threshold_reused':28.,'previous_warehouse_no_ref_threshold_reused':1.,'actual_source_mp4_against_original_Cycles_PSN_min_dB':round(min(source_match_psnr),4),'actual_source_mp4_against_original_Cycles_PSN_median_dB':round(statistics.median(source_match_psnr),4),'wrong_product_12_control_psnr_median_dB':round(statistics.median(wrong_source_psnr),4),'wrong_product_reference_acceptance':'REVIEW, not FAIL, despite actual PSNR being far below normal; source identity does not match','accepted_transfer_rule_checks':acceptance,'meets_all_preregistered_tests':passed,'confusion':{'source_verified':all_src,'no_reference':all_noref},'descriptive_95pct_exact_binomial_frame_level_CI':{'sensitivity_source':cp_ci(all_src['TP'],all_src['TP']+all_src['FN']),'false_positive_rate_source':cp_ci(all_src['FP'],all_src['FP']+all_src['TN']),'sensitivity_no_ref':cp_ci(all_noref['TP'],all_noref['TP']+all_noref['FN']),'false_positive_rate_no_ref':cp_ci(all_noref['FP'],all_noref['FP']+all_noref['TN'])},'cases':allcases,'compute':{'source_verify_and_decode_png_seconds':decode_input_seconds,'variant_CPU_decode_s':{k:v['actual_decode_wall_s'] for k,v in allcases.items()},'variant_both_rules_feature_CPU_s':{k:v['same_frames_both_rules_wall_s'] for k,v in allcases.items()},'variant_ffprobe_CPU_wall_s':{k:v['PTS']['wall_s'] for k,v in allcases.items()},'variant_ffmpeg_freezedetect_wall_s':{k:v['official_freezedetect']['wall_s'] for k,v in allcases.items()},'original_extract_s':canonical_rec['runtime_extract_sec'],'variants_encode_s':{k:v['encode_wall_s'] for k,v in enc['variants'].items()},'GPU_used':False},'uncertainty_limits':['9 frame edits represent six controlled event groups in only ONE previously unseen dock video; frame-level beta intervals assume independence and are descriptive, NOT general deployment CIs','Both positives and negatives are generated or author approved (not independently human-adjudicated natural video faults); frame edits fixed before video processing','The current scene and original movie may have imperfections independently unrelated to frame ordering; source contract alone cannot judge photorealism, engineering or aesthetics','The CRF23 clean transcode is an independent benign compression change, not a previously buyer-approved exact encoding contract','Wrong warehouse source comparison tests reference identity and must abstain regardless of numeric score','No audio track, so audio/event sync cannot be tested','Live SWAAG tool registration, end-user approval and shared GPU resource contention not tested']}
 once(OUTPUT,output)
 print('DOCK_INDEPENDENT_SOURCE_TRANSFER_FROZEN',output['status'],'source',all_src,'noref',all_noref,'confidence',output['descriptive_95pct_exact_binomial_frame_level_CI'],'source_real_scene_PSNRmin',output['actual_source_mp4_against_original_Cycles_PSN_min_dB'],'sha',sha(OUTPUT))

if __name__=='__main__':main()
