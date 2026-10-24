#!/usr/bin/env python3
"""Audit Spotlight source-event coverage against random peak placement.

Only previously FROZEN raw temporal predictions and human annotation intervals.
No media decoder, no GPU, no model inference. Monte Carlo respects three peak
locations, grid of 8 FPS, >=0.375 second separation. No threshold tuning.
Human full-clip error windows inflate easy localization; count explicitly.
"""
from pathlib import Path
import json,hashlib,collections,statistics
import numpy as np
ROOT=Path('/data/var/perceptual-independent-20261024/spotlight-videogen')
RAW=ROOT/'spotlight_24_motion_pixel_only_predictions.json'
SCORE=ROOT/'spotlight_24_human_motion_eval.json'
OUT=ROOT/'spotlight_24_motion_random_peak_baseline.json'
N=10000
SEED=20261024
METRICS=('flow_magnitude_jerk','warp_residual','frame_difference_mae')

def random_indices(rng,n):
 # Draw shuffled grid positions sequentially without a second draw or rejection.
 # Selecting 3 from n>=40, no two closer than 3 frames.
 choices=[]
 for idx in rng.permutation(n):
  if all(abs(int(idx)-old)>=3 for old in choices):choices.append(int(idx))
  if len(choices)==3:break
 assert len(choices)==3
 return [(x+.5)/8. for x in choices]

def main():
 if OUT.is_file():print('EXISTING_EVIDENCE_NOT_REPLAYED');return
 raw=json.loads(RAW.read_text());d=json.loads(SCORE.read_text());assert len(raw['predictions'])==len(d['per_video'])==24
 positives=[x for x in d['per_video'] if x['has_human_annotated_motion']]
 assert len(positives)==9
 n_gold=sum(len(x['human_motion_segments']) for x in positives);assert n_gold==11
 # Each generator clip of a repeated prompt is correlated; this exploratory
 # random-time baseline is NOT an inferential test across 24 iid productions.
 random=np.random.default_rng(SEED)
 sim=np.zeros(N,dtype=np.int16)
 for x in positives:
  intervals=[(q['begin'],q['end']) for q in x['human_motion_segments']]
  frames=raw['predictions'][x['id']]['temporal_metrics'][METRICS[0]]['n_timepoints']
  for run in range(N):
   times=random_indices(random,frames)
   sim[run]+=sum(any(begin<=t<=end for t in times) for begin,end in intervals)
 stats={'random_expected_events_out_of_11':round(float(np.mean(sim)),4),'random_std':round(float(np.std(sim)),4),'random_95pct_interval_counts':[int(np.quantile(sim,x,method='nearest')) for x in (.025,.975)],'random_max':int(sim.max()),'seed':SEED,'draws':N,'topK':3,'separation_frames_min':3,'total_motion_events':n_gold,'full_clip_motion_annotation_count':sum((p['end']-p['begin'])>=raw['predictions'][x['id']]['duration_estimate_seconds']-.01 for x in positives for p in x['human_motion_segments'])}
 methods={}
 for name in METRICS:
  s=d['metric_summary'][name];hits=s['events_localized_by_top3'];stats_for=s.copy()
  stats_for.update(random_peak_ge_observed_pvalue=float(np.mean(sim>=hits)),excess_hits_over_random_mean=round(hits-float(np.mean(sim)),4))
  methods[name]=stats_for
 flag_groups={}
 for gen in ('ltx2','seedance','veo3'):
  g=[x for x in d['per_video'] if x['model_generator']==gen]
  flag_groups[gen]={name:{'human_motion_positives':sum(x['has_human_annotated_motion'] for x in g),'TP':sum(x['has_human_annotated_motion'] and x['metrics'][name]['anomaly_flag'] for x in g),'FP_proxy':sum((not x['has_human_annotated_motion']) and x['metrics'][name]['anomaly_flag'] for x in g),'events_found':sum(x['metrics'][name]['events_found_by_top3'] for x in g),'events_total':sum(x['metrics'][name]['human_motion_events'] for x in g)} for name in METRICS}
 report={'source':'Spotlight pinned 600 video full human event labels, fixed 24 video stratified pilot, 11 motion annotations in 9 videos','source_prediction_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'gold_eval_sha256':hashlib.sha256(SCORE.read_bytes()).hexdigest(),'sampling':'random peak positions on same 8 FPS frame grid with >=3 frame separation, fixed 10000 reps; no control over true physical plausibility and no unseen gold','chance':stats,'comparisons':methods,'generator_counts':flag_groups,'warning':'Model categories differ, videos share 8 source prompts, errors often broad (one entire clip); random peaks already frequently cover annotated segments. P-value here means fraction of uniformly random peak sets with >= observed segment hits, NOT probability motion detector is correct, and the method/data were previously seen. Do not treat no motion label as verified clean or event localization as semantic error identification.'}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('CHANCE_EXPECTED',stats['random_expected_events_out_of_11'],'OF 11','95%',stats['random_95pct_interval_counts'],'full_clip_annotations',stats['full_clip_motion_annotation_count'])
 for k,v in methods.items():print('METHOD',k,'HITS',v['events_localized_by_top3'],'p_random_ge',v['random_peak_ge_observed_pvalue'],'TP',v['TP_on_labeled_motion'],'FP_PROXY',v['FP_proxy_on_no_motion_labels'])
 print('GENERATORS',flag_groups)
if __name__=='__main__':main()
