#!/usr/bin/env python3
"""Evaluate genuinely filmed vs generated physics video temporal visual signals.

24 independently sourced 5-second MP4s, 12 scenarios with blind human-vote
preferences. Compute RAW pixel/motion scores before reading vote directions.
Normalize input time to common 12 fps, aspect to 192x108 pixels. CPU only.
Metric direction fixed: lower residual/jerk = smoother, not necessarily more
PHYSICALLY realistic; a passing physics judge is NOT expected. Licenses apply.
"""
from __future__ import annotations
from pathlib import Path
import collections,hashlib,json,math,statistics,time,resource
import numpy as np
import cv2
ROOT=Path('/data/var/perceptual-independent-20261024/rapidata-physics-video')
SOURCE=ROOT/'selected_human_pair_clip_receipts.json'
HUMAN=ROOT/'frozen_human_pair_source_manifest.json'
RAW=ROOT/'24_video_temporal_raw_scores_20261024.json'
SCORED=ROOT/'12_human_rated_real_vs_ai_temporal_comparison_20261024.json'
CONFIG={'sampling_fps':12,'frame_count':60,'target_size':[192,108],'duration_seconds':5.0,'methods':['temporal_second_derivative_over_adjacent_change','flow_warp_residual_mean','interframe_exact_repeat_rate','flow_magnitude_acceleration'],'direction':'lower = less temporal change/flicker/jerk; NOT guaranteed more physically correct','farneback_parameters':{'pyr_scale':0.5,'levels':2,'winsize':13,'iterations':2,'poly_n':5,'poly_sigma':1.1,'flags':0},'pixel_interpolation':'INTER_AREA','comparison_rule':'pairwise lower raw score wins, near-exact ties -> abstain; no thresholds or source model labels for raw stage'}

def sample(path):
 c=cv2.VideoCapture(str(path));assert c.isOpened(),path
 fps=float(c.get(cv2.CAP_PROP_FPS));count=int(c.get(cv2.CAP_PROP_FRAME_COUNT));original=[int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))]
 assert fps>2 and count>=65
 target=np.rint(np.arange(CONFIG['frame_count'])*fps/12.0).astype(int)
 arr=[];i=0;t=0;inds=set(target.tolist())
 # Sequential decode, source media read-only.
 while len(arr)<len(target):
  ok,fr=c.read();assert ok and i<count,(path,i,count)
  if i in inds:
   x=cv2.cvtColor(cv2.resize(fr,(192,108),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
   arr.append(x)
  i+=1
  if i>max(target):break
 c.release()
 assert len(arr)==len(target),(len(arr),target)
 return np.asarray(arr,dtype=np.uint8),{'source_fps':fps,'source_frames':count,'source_size':original,'resampled_frames':len(arr)}

def compute(path):
 cv2.setNumThreads(1)
 start=time.perf_counter();frames,info=sample(path)
 values=frames.astype(np.float32)
 adjacent=np.abs(values[1:]-values[:-1]).mean(axis=(1,2))
 second=np.abs(values[1:-1]-(values[:-2]+values[2:])/2.).mean(axis=(1,2))
 second_normalized=float(second.sum()/(adjacent.sum()+1e-7))
 warp=[];median_flows=[]
 h,w=frames[0].shape;gridx,gridy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
 args=CONFIG['farneback_parameters']
 for a,b in zip(frames[:-1],frames[1:]):
  flow=cv2.calcOpticalFlowFarneback(a,b,None,args['pyr_scale'],args['levels'],args['winsize'],args['iterations'],args['poly_n'],args['poly_sigma'],args['flags'])
  # map f_{t+1} backward to f_t using the forward flow displacement
  warped=cv2.remap(b,gridx+flow[:,:,0],gridy+flow[:,:,1],interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
  warp.append(float(cv2.absdiff(a,warped).mean())/255.)
  median_flows.append(float(np.median(np.linalg.norm(flow,axis=-1))))
 flow_jerk=float(np.percentile(np.abs(np.diff(median_flows)),90)) if len(median_flows)>2 else None
 repeats=float(np.mean([np.array_equal(a,b) for a,b in zip(frames[:-1],frames[1:])]))
 info.update({'temporal_second_derivative_over_adjacent_change':round(second_normalized,7),'flow_warp_residual_mean':round(float(np.mean(warp)),7),'interframe_exact_repeat_rate':round(repeats,7),'flow_magnitude_acceleration':round(flow_jerk,7),'frame_mean_luma':round(float(values.mean()),5),'mean_interframe_MAE':round(float(adjacent.mean()/255),7),'wall_seconds':round(time.perf_counter()-start,2)})
 return info

def raw():
 if RAW.exists():print('RAW_ALREADY_FROZEN_NO_REPLAY');return
 receipts=json.loads(SOURCE.read_text());assert len(receipts['clips'])==24
 result={'method':CONFIG,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'clip_metrics':{},'notes':'Only pixels and source verification read here; no human labels or model names for metric choice.'}
 for idx,case in enumerate(sorted(receipts['clips'],key=lambda x:x['sha256'])):
  assert case['status'] in ('downloaded','existing_not_redownloaded')
  p=Path(case['file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==case['sha256']
  values=compute(p)
  result['clip_metrics'][case['sha256']]=values
  print('VIDEO',idx+1,'/',len(receipts['clips']),'RESAMPLED_FPS',CONFIG['sampling_fps'],'SCORES',{k:values[k] for k in CONFIG['methods']},'SECONDS',values['wall_seconds'],flush=True)
 assert len(result['clip_metrics'])==24
 RAW.write_text(json.dumps(result,indent=2)+'\n')
 print('SAVED_RAW_BEFORE_GOLD',RAW,flush=True)

def eval():
 if SCORED.exists():print('SCORED_COMPLETE_NOT_REPLAYED');return
 frozen=json.loads(RAW.read_text());assert frozen['method']==CONFIG and len(frozen['clip_metrics'])==24
 receipt=json.loads(SOURCE.read_text());human=json.loads(HUMAN.read_text());assert len(human['samples'])==12
 byurl={r['source_url']:r for r in receipt['clips']}
 results=[]
 for sample in human['samples']:
  real=frozen['clip_metrics'][byurl[sample['ground_truth_url']]['sha256']]
  synthetic=frozen['clip_metrics'][byurl[sample['generated_url']]['sha256']]
  decision={}
  for k in CONFIG['methods']:
   delta=real[k]-synthetic[k]
   # No adjustable near-tie tolerance, only exact floats
   decision[k]={'gt_score':real[k],'ai_score':synthetic[k],'gt_score_less_than_ai':delta<0,'equal':delta==0,'direction_fixed_lower_is_better':True}
  results.append({'prompt_sha256':sample['scenario_prompt_sha256'],'human_blind_prefers_real':sample['ground_truth_human_win_blind'],'human_with_reference_prefers_real':sample['ground_truth_human_win_with_reference'],'vote_real_blind':sample['blind_ground_truth_weighted_votes'],'vote_fake_blind':sample['blind_generated_weighted_votes'],'vote_real_reference':sample['with_reference_ground_truth_weighted_votes'],'vote_fake_reference':sample['with_reference_generated_weighted_votes'],'metrics':decision})
 comparisons={}
 for k in CONFIG['methods']:
  for mode in ['blind','with_reference']:
   gold=[r['human_'+mode+'_prefers_real'] for r in results]
   preds=[r['metrics'][k]['gt_score_less_than_ai'] for r in results]
   abstain=sum(r['metrics'][k]['equal'] for r in results)
   matches=sum(a==b for a,b,r in zip(preds,gold,results) if not r['metrics'][k]['equal'])
   comparisons[k+'_'+mode]={'matched_human_preference':matches,'cases_scored':len(results)-abstain,'ties':abstain,'failures':len(results)-abstain-matches,'source_real_preferred_by_humans':sum(gold),'not_independent_of_source_model_fps':True}
 scored={'source':'Rapidata Physics-IQ human preference, 12 independent real filmed vs Cosmos 2.5 5-second prompts','source_original_manifest_sha256':hashlib.sha256(HUMAN.read_bytes()).hexdigest(),'pixel_predictions_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'time_normalized_12fps':True,'comparator_results':comparisons,'cases':results,'limitations':'Human vote is comparative perceived realism, not a frame-exact stutter label. GT vs generated has participant confound: actual source 30fps, Cosmos output 16fps; resampling both at 12fps reduces but does not fully eliminate codec/resolution/model confounds. Only 12 scenarios and one generator; human preferences all happen to favor the real clip. Some real footage can contain movement jerk or compression. Scores cannot prove physics or detect audio/video synchronization. Never tune on these selected cases.'}
 SCORED.write_text(json.dumps(scored,indent=2)+'\n')
 print('HUMAN_COMPARISON',json.dumps(comparisons,indent=2))
if __name__=='__main__':raw();eval()
