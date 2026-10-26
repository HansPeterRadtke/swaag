#!/usr/bin/env python3
"""Independently score frozen time-coded controlled real-owner MP4 edit outputs.

Source-bound actual frame identity from previously SHA-verified canonical frame
sequence. PTS-only, no-reference adjacent motion, and FFmpeg freezedetect are
frozen comparison controls. No model inference or production writes.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import cv2
import numpy as np

BASE=Path('/data/var/perceptual-independent-20261024/owner_real_encoded_fault_injection_20261026')
ORIGINAL=Path('/data/var/3d-publication-20261024/approved_warehouse_door_rear_corner_v47.mp4')
ORIG_SHA='e2ef866ec79318a6b3812865a51922141ad02be66b555900da311171f49b4adb'
REPORT=BASE/'final_video_fault_score.json'
THRESH_PSNR_DB=28.
THRESH_NOREF_GRAYSCALE=1.
CAMERA_START_HOLD=166

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def save(path,obj):
 if path.exists():raise RuntimeError('Frozen prior scoring result exists')
 tmp=path.with_suffix('.newtmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def decode(path,canonical):
 cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG)
 if not cap.isOpened():raise RuntimeError('Could not read real encoded video '+str(path))
 t=time.monotonic();out=[];prev=None
 try:
  for idx in range(1,211):
   ok,frame=cap.read()
   if not ok:raise RuntimeError('Encoded MP4 frame missing '+str(idx))
   expected=canonical[idx-1]
   if expected.shape!=frame.shape:raise RuntimeError('Source shape differs from final output')
   psnr=float(cv2.PSNR(frame,expected))
   small=cv2.cvtColor(cv2.resize(frame,(240,135),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
   motion=None if prev is None else float(cv2.mean(cv2.absdiff(small,prev))[0])
   out.append({'frame':idx,'PTS_expected_seconds':round((idx-1)/15,6),'source_aligned_psnr_dB':round(psnr,4),'source_mismatch_candidate':bool(psnr<THRESH_PSNR_DB),'adjacent_gray_motion_240':round(motion,5) if motion is not None else None,'low_motion_candidate_pre_hold':bool(idx>1 and idx<CAMERA_START_HOLD and motion<THRESH_NOREF_GRAYSCALE) if motion is not None else False,'inside_authorized_camera_hold':idx>=CAMERA_START_HOLD})
   prev=small
  ok,_=cap.read()
  if ok:raise RuntimeError('Unexpected extra frame')
 finally:cap.release()
 return out,round(time.monotonic()-t,3)

def ffprobe_pts(path):
 t=time.monotonic(); p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)],capture_output=True,text=True,timeout=30,check=True)
 frame=json.loads(p.stdout)['frames'];arr=[float(x['best_effort_timestamp_time']) for x in frame]
 return {'count':len(arr),'nonmonotonic':sum(x<=y for x,y in zip(arr[1:],arr[:-1])),'max_target_pts_error_s':max(abs(x-i/15) for i,x in enumerate(arr)),'seconds':round(time.monotonic()-t,3)}

def freeze_detector(path):
 t=time.monotonic()
 result=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','info','-i',str(path),'-vf','freezedetect=n=-50dB:d=0.12','-f','null','-'],capture_output=True,text=True,timeout=80)
 if result.returncode!=0:raise RuntimeError('ffmpeg official freezedetect failed: '+result.stderr[-700:])
 starts=[float(x) for x in re.findall(r'freeze_start:\s*([0-9.]+)',result.stderr)]
 ends=[float(x) for x in re.findall(r'freeze_end:\s*([0-9.]+)',result.stderr)]
 return {'event_start_seconds':starts,'event_end_seconds':ends,'events_total':len(starts),'events_in_authorized_last_3sec':sum(x>=11.0 for x in starts),'events_in_moving_part':sum(x<11.0 for x in starts),'wall_s':round(time.monotonic()-t,3)}

def matrix(rows,gold,feature):
 return {'TP':sum(x['frame'] in gold and x[feature] for x in rows),'FN':sum(x['frame'] in gold and not x[feature] for x in rows),'FP':sum(x['frame'] not in gold and x[feature] for x in rows),'TN':sum(x['frame'] not in gold and not x[feature] for x in rows)}

def main():
 if REPORT.exists():raise RuntimeError('Previously scored; preserve receipts')
 manifest=BASE/'source_receipts.json';origframes=BASE/'source_frames.json';encodes=BASE/'encodes.json'
 m=json.loads(manifest.read_text());f=json.loads(origframes.read_text());e=json.loads(encodes.read_text())
 assert sha(ORIGINAL)==ORIG_SHA and m['video_sha256']==ORIG_SHA
 assert e['frozen_receipt_sha256']==sha(manifest) and e['source_frame_receipt_sha256']==sha(origframes)
 assert all(k in e['variants'] for k in ('clean_reencoded','single_frame_holds','short_reversals'))
 t=time.monotonic();canonical=[]
 for i in range(1,211):
  p=BASE/'canonical_decoded_png'/f'frame_{i:04d}.png'
  assert sha(p)==f['exact_png_hashes'][str(i)]
  im=cv2.imread(str(p),cv2.IMREAD_COLOR)
  if im is None or im.shape!=(540,960,3):raise RuntimeError('Corrupt source original '+str(i))
  canonical.append(im)
 original_decode_t=round(time.monotonic()-t,3)
 result={'status':'PARTIAL','source':'ACTUAL original approved owner MP4 reencoded with exact script-controlled mutations; independent script-time labels but not independently human-rated natural faults','source_SHA256':ORIG_SHA,'frozen_manifest_sha256':sha(manifest),'frozen_encoded_variant_sha256':sha(encodes),'source_decoded_original_PNG_SHA256':sha(origframes),'thresholds':{'source_psnr_min_dB':THRESH_PSNR_DB,'no_ref_gray_move_under':THRESH_NOREF_GRAYSCALE,'source_authorized_camera_hold_start_frame':CAMERA_START_HOLD},'no_real_human_gold':True,'resource':{'source_verify_and_load_wall_s':original_decode_t,'GPU_used':False},'cases':{}}
 for name,spec in e['variants'].items():
  path=Path(spec['path']);assert sha(path)==spec['sha256']
  rows,latency=decode(path,canonical)
  pts=ffprobe_pts(path);ff=freeze_detector(path)
  gold=set(spec['edit_positions'])
  source=matrix(rows,gold,'source_mismatch_candidate')
  noreference=matrix(rows,gold,'low_motion_candidate_pre_hold')
  psn=[x['source_aligned_psnr_dB'] for x in rows]
  motion=[x['adjacent_gray_motion_240'] for x in rows if x['adjacent_gray_motion_240'] is not None]
  holdmotion=[x['adjacent_gray_motion_240'] for x in rows if x['inside_authorized_camera_hold']]
  if len(rows)!=210 or pts['count']!=210 or pts['nonmonotonic']>0:raise RuntimeError('Original encoded delivery corrupted unexpectedly')
  result['cases'][name]={'file_SHA256':spec['sha256'],'full_decode_and_metric_wall_s':latency,'PTS':pts,'official_ffmpeg_freezedetect':ff,'frame_gold_synthetic_positions':sorted(gold),'source_bound_detection':source,'no_reference_low_motion_detection':noreference,'source_quality':{'psnr_min_dB':min(psn),'psnr_median_dB':round(statistics.median(psn),3),'below_28_frame_count':sum(p<28 for p in psn)},'real_authorized_hold':{'num_frames':len(holdmotion),'gray_motion_median':round(statistics.median(holdmotion),5),'low_motion_candidate_ungated_frames':sum((r['adjacent_gray_motion_240'] or 0)<1 for r in rows if r['inside_authorized_camera_hold'])},'decoded_frame_indices':[{'frame':x['frame'],'PSNR':x['source_aligned_psnr_dB'],'source_flag':x['source_mismatch_candidate'],'no_ref_flag':x['low_motion_candidate_pre_hold'],'motion_240':x['adjacent_gray_motion_240']} for x in rows],'exact_frozen_gold_not_used_by_detector':True}
  print('MEASURED_REAL_OWNER_FAULT_FILM',name,'SCRIPT_GOLD',sorted(gold),'SOURCE_FRAME_MATRIX',source,'NO_REF_MATRIX',noreference,'ffmpeg',ff,'psnrmin',min(psn),'cost_s',latency,flush=True)
 result['all_case_video_level']={'source_gate_positive_variants':sum(x['source_bound_detection']['TP']>0 for name,x in result['cases'].items() if name!='clean_reencoded'),'source_gate_clean_false_fail':bool(result['cases']['clean_reencoded']['source_bound_detection']['FP']),'no_reference_detect_positive_videos':sum(x['no_reference_low_motion_detection']['TP']>0 for name,x in result['cases'].items() if name!='clean_reencoded'),'no_reference_clean_false_alert':bool(result['cases']['clean_reencoded']['no_reference_low_motion_detection']['FP'])}
 result['limitations']=['Script-authored replay of a real owner VIDEO is not an independent human-adjudicated natural fault set','Individual frames and neighboring events from one scene are strongly correlated and do not justify generic FP/FN rate inference','A low source frame PSNR can prove an unexpected edited frame under a verified exact source/order contract only, not that a camera/static scene is visually ugly','Whole-frame FFmpeg freeze can flag the legitimate final three-second camera hold; no-ref heuristics need source time policy','All H264 files have same codec/fps, and all synthetically faulted films can remain HTML5 playable, so codec/probe success is not temporal semantic correctness','No audio stream so there is no audiovisual sync gold','No GPU inference or real SWAAG runtime tool calls']
 save(REPORT,result)
 print('SOURCE_BOUNDED_VIDEO_FRAME_FAULT_COMPARISON_FROZEN',sha(REPORT),'video summary',result['all_case_video_level'])

if __name__=='__main__':main()
