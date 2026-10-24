#!/usr/bin/env python3
"""Temporal video signals vs independently human-annotated Spotlight motion faults.

Compute original-video pixel/optical-flow time series and freeze BEFORE any
annotation based scoring. 24 genuine human-reviewed T2V videos, 3 generators,
8 independent prompt families. 4 prompt families chosen >=2 motion labels;
4 have zero motion labels across all models (NOT independently clean).

This IS NOT judging physical plausibility, action correctness or generative
quality; comparison tests whether crude temporal peaks land near human motion
fault intervals. Score fixed thresholds/top-K before reading any labels.
CC BY-NC 4.0 media and raw results outside project Git. No model inference.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,subprocess,time,resource,math,collections,statistics
import cv2,numpy as np
ROOT=Path('/data/var/perceptual-independent-20261024/spotlight-videogen')
MANIFEST=ROOT/'motion_human_24_video_manifest.json'
RAW=ROOT/'spotlight_24_motion_pixel_only_predictions.json'
SCORES=ROOT/'spotlight_24_human_motion_eval.json'
FPS=8;FRAME_SIZE=(160,90)
METRICS=('flow_magnitude_jerk','warp_residual','frame_difference_mae')
CONFIG={'source':'ECCV 2026 Spotlight 24 original generated videos, human motion-error annotations','fps':FPS,'pixels':FRAME_SIZE,'metric_names':METRICS,'farneback':{'pyr_scale':.5,'levels':2,'winsize':13,'iterations':2,'poly_n':5,'poly_sigma':1.1,'flags':0},'top_peaks':3,'nonmaximum_suppression_seconds':.375,'anomaly_threshold_MAD_z':5.0,'robust_deviation_floor':{'flow_magnitude_jerk':0.025,'warp_residual':.002,'frame_difference_mae':.002},'expected_behavior':'Peak positions capture temporal errors; not a model of objective physics','no_threshold_tuning_or_calibration':True,'selection':'4 motion-group and 4 no-motion-group independent prompt IDs, each has all three model variants, constrained MP4 sizes <=12MB. Annotation presence enriches sample and is not population prevalence.'}

def decode(path):
 cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','1','-filter_threads','1','-i',str(path),'-an','-vf','fps=8,scale=160:90:flags=area','-pix_fmt','gray','-f','rawvideo','pipe:1']
 proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,check=False)
 if proc.returncode or len(proc.stdout)%(160*90) or len(proc.stdout)<15*(160*90) or len(proc.stdout)>180*(160*90):raise RuntimeError(f'FFmpeg invalid decode frames {len(proc.stdout)//(160*90)}, status {proc.returncode}, stderr {proc.stderr[:120]!r}')
 frames=np.frombuffer(proc.stdout,dtype=np.uint8).reshape(-1,90,160)
 return frames

def feature(frames):
 cv2.setNumThreads(1)
 a=CONFIG['farneback'];gridx,gridy=np.meshgrid(np.arange(160,dtype=np.float32),np.arange(90,dtype=np.float32))
 warp=[];magnitudes=[];diffs=[]
 for first,second in zip(frames[:-1],frames[1:]):
  flow=cv2.calcOpticalFlowFarneback(first,second,None,a['pyr_scale'],a['levels'],a['winsize'],a['iterations'],a['poly_n'],a['poly_sigma'],a['flags'])
  mag=np.linalg.norm(flow,axis=2)
  magnitudes.append(float(np.median(mag)))
  warped=cv2.remap(second,gridx+flow[:,:,0],gridy+flow[:,:,1],interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
  warp.append(float(cv2.absdiff(first,warped).mean())/255.0)
  diffs.append(float(cv2.absdiff(first,second).mean())/255.0)
 jerk=[0.]+[abs(u-v) for u,v in zip(magnitudes[:-1],magnitudes[1:])]
 return {'flow_magnitude_jerk':np.asarray(jerk,dtype=np.float64),'warp_residual':np.asarray(warp,dtype=np.float64),'frame_difference_mae':np.asarray(diffs,dtype=np.float64),'magnitudes':np.asarray(magnitudes,dtype=np.float64)}

def peak_features(vals,key):
 raw=vals.tolist();n=len(raw)
 median=float(np.median(vals));mad=float(np.median(np.abs(vals-median)))
 denom=max(1.4826*mad,CONFIG['robust_deviation_floor'][key])
 z=(vals-median)/denom
 ranked=np.argsort(-z,kind='stable')
 chosen=[]
 for idx in ranked:
  time_s=round((float(idx)+.5)/FPS,4)
  if all(abs(time_s-p['seconds'])>=CONFIG['nonmaximum_suppression_seconds'] for p in chosen):
   chosen.append({'seconds':time_s,'z':round(float(z[idx]),5),'value':round(float(vals[idx]),7),'i':int(idx)})
  if len(chosen)>=3:break
 return {'n_timepoints':n,'median':round(median,7),'mad':round(mad,7),'max_z':round(float(np.max(z)),5),'anomaly_flag':bool(float(np.max(z))>=CONFIG['anomaly_threshold_MAD_z']),'top3_local_peaks':chosen,'scores_time_series':[(round((float(i)+.5)/FPS,4),round(float(value),7)) for i,value in enumerate(raw)]}

def compute_raw():
 if RAW.exists():print('RAW_PREVIOUSLY_FROZEN_NO_REPLAY');return
 m=json.loads(MANIFEST.read_text());assert m['video_count']==24
 records={};start=time.monotonic()
 # The predictor accesses ONLY the independently downloaded media, its hash and id;
 # all human annotations are ignored until the raw input feature file is frozen.
 for i,c in enumerate(sorted(m['cases'],key=lambda a:a['id'])):
  path=Path(c['receipt']['path']);sha=c['receipt']['sha256'];t=time.monotonic()
  if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise RuntimeError('source hash mismatch '+c['id'])
  video=decode(path);f=feature(video);methods={key:peak_features(f[key],key) for key in METRICS}
  records[c['id']]={'source_video_sha256':sha,'frame_count':len(video),'duration_estimate_seconds':len(video)/FPS,'source_media_size_bytes':path.stat().st_size,'temporal_metrics':methods,'runtime_seconds':round(time.monotonic()-t,2)}
  print('VIDEO',i+1,'/',len(m['cases']),c['id'],'decoded',len(video),'seconds',records[c['id']]['runtime_seconds'],'flags',[methods[k]['anomaly_flag'] for k in METRICS],flush=True)
 data={'config':CONFIG,'source_receipt_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'predictions':records,'total_wall_seconds':round(time.monotonic()-start,2),'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'ground_truth_not_used_in_predictor':True}
 RAW.write_text(json.dumps(data,indent=2)+'\n')
 print('FROZEN_BEFORE_LABEL_SCORING',RAW,'VIDEOS',len(records),flush=True)

def score():
 if SCORES.exists():print('SCORES_EXIST_NOT_RECOMPUTED');return
 m=json.loads(MANIFEST.read_text());a=json.loads(RAW.read_text());assert json.dumps(a['config'],sort_keys=True)==json.dumps(CONFIG,sort_keys=True) and len(a['predictions'])==24
 samples=[]
 for case in m['cases']:
  pred=a['predictions'][case['id']];ann=case['motion_annotations']
  intervals=[]
  for item in ann:
   begin,end=item['segment'];intervals.append((max(0.,float(begin)),min(float(end),pred['duration_estimate_seconds'])))
  for begin,end in intervals:
   assert end>begin,(case['id'],begin,end)
  rec={'id':case['id'],'prompt_id':case['pid'],'model_generator':case['generator'],'human_motion_segments':[{'begin':b,'end':e} for b,e in intervals],'has_human_annotated_motion':bool(intervals),'other_annotated_error_types':sorted(set(x['type'] for x in case['other_annotations'])),'metrics':{}}
  for name in METRICS:
   f=pred['temporal_metrics'][name]
   top=[x['seconds'] for x in f['top3_local_peaks']]
   event_hits=[any(b<=s<=e for s in top) for b,e in intervals]
   rec['metrics'][name]={'anomaly_flag':f['anomaly_flag'],'events_found_by_top3':sum(event_hits),'human_motion_events':len(intervals),'top3_peaks_inside_any_labeled_motion_segment':sum(any(b<=s<=e for b,e in intervals) for s in top),'top3_peaks_outside_human_motion_segments':sum(not any(b<=s<=e for b,e in intervals) for s in top),'event_hits':event_hits,'top3_peak_seconds':top,'max_z':f['max_z']}
  samples.append(rec)
 summary={}
 for method in METRICS:
  tp=sum(x['has_human_annotated_motion'] and x['metrics'][method]['anomaly_flag'] for x in samples)
  fp=sum(not x['has_human_annotated_motion'] and x['metrics'][method]['anomaly_flag'] for x in samples)
  fn=sum(x['has_human_annotated_motion'] and not x['metrics'][method]['anomaly_flag'] for x in samples)
  tn=sum(not x['has_human_annotated_motion'] and not x['metrics'][method]['anomaly_flag'] for x in samples)
  events=sum(x['metrics'][method]['human_motion_events'] for x in samples)
  found=sum(x['metrics'][method]['events_found_by_top3'] for x in samples)
  summary[method]={'TP_on_labeled_motion':tp,'FP_proxy_on_no_motion_labels':fp,'FN_on_labeled_motion':fn,'TN_proxy_on_no_motion_labels':tn,'labeled_motion_videos':tp+fn,'not_motion_labeled_videos':fp+tn,'positive_recall':tp/(tp+fn) if tp+fn else None,'proxy_precision':tp/(tp+fp) if tp+fp else None,'events_localized_by_top3':found,'total_human_motion_events':events,'event_recall_top3':found/events if events else None}
 # Fixed random baseline for top3 temporal points per source, 1000 draws each
 rng=np.random.default_rng(20261024);random_hits=[]
 for x in samples:
  if not x['human_motion_segments']:continue
  dur=a['predictions'][x['id']]['duration_estimate_seconds']
  truth=[(b['begin'],b['end']) for b in x['human_motion_segments']]
  for _ in range(1000):
   times=rng.uniform(0,dur,size=3)
   random_hits.append(sum(any(b<=s<=e for s in times) for b,e in truth))
 baseline=float(np.mean(random_hits)) if random_hits else None
 result={'date':'2026-10-24','source':'Spotlight ECCV 2026 pinned CC BY-NC 4.0, 24 original T2V videos, nine with human motion errors, remainder no motion annotation but may have OTHER errors','human_annotations_are_NOT_certified_frame_artifact_or_clean_ground_truth':True,'raw_predictions_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'label_manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'seed_for_uniform_random_time_peaks':20261024,'classes_balanced_by_prompt_based_on_annotations':'four >=2/3 motion-positive prompt groups and four 0/3 motion groups across LTX-2, Seedance, Veo3','raw_metrics_frozen_before_gold':True,'no_mdl_or_threshold_tuning':True,'per_video':samples,'metric_summary':summary,'random_baseline_mean_event_hits_per_positive_video_per_3_uniform_points':baseline,'clear_scope':'Unlabeled motion intervals and clips not verified clean; do not interpret FP_proxy as real false alarm; model has no semantic understanding and cannot judge static physical realism, prompt adherence, object appearance, image-text discrepancies, or production video stutter. Human intervals may cover much of a clip. Multiple generator clips share same prompt.'}
 SCORES.write_text(json.dumps(result,indent=2)+'\n')
 print('HUMAN_EVENT_SUMMARY',json.dumps(summary,indent=2),'RANDOM_BASELINE_EVENT_HITS_MEAN',baseline)
if __name__=='__main__':compute_raw();score()
