#!/usr/bin/env python3
"""Fixed prior pixel-only temporal features on 72 prompt-DISJOINT Spotlight clips.

Reuses exact prior 24-video published script's 8fps, Farneback, robust z>=5,
3 peak and 0.375sec spacing. No retuning. Source 72 original human-reviewed
T2V clips from 24 untouched prompt families; motion annotations ONLY used
AFTER per-video image features and predictions are checkpointed.
External-source annotation absence is NOT a true clean label, and geometry /
semantics cannot be judged by optical-flow peaks.
No GPU, no new model load, no videos altered or redistributed.
"""
from __future__ import annotations
from pathlib import Path
import json,hashlib,time,resource,collections,datetime
import numpy as np
from spotlight_human_temporal_flow_eval import CONFIG, METRICS, decode, feature, peak_features
ROOT=Path('/data/var/perceptual-independent-20261024/spotlight-videogen')
SOURCE=ROOT/'holdout72_prompt_disjoint_source_20261025.json'
PRED=ROOT/'holdout72_temporal_frozen_pixel_features_20261025.json'
SCORES=ROOT/'holdout72_temporal_human_motion_scores_20261025.json'
N=10_000


def compute():
 source=SOURCE.read_bytes();m=json.loads(source)
 assert m['all_completed'] and m['n']==len(m['items'])==72
 raw=json.loads(PRED.read_text()) if PRED.exists() else {'config':CONFIG,'source_manifest_sha256':hashlib.sha256(source).hexdigest(),'status':'in_progress','source_human_annotations_NOT_read_by_feature_extractor':True,'cases':{},'created':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 assert json.dumps(raw['config'],sort_keys=True)==json.dumps(CONFIG,sort_keys=True)
 assert raw['source_manifest_sha256']==hashlib.sha256(source).hexdigest()
 start=time.monotonic()
 for row in sorted(m['items'],key=lambda x:x['id']):
  ident=row['id'];v=row['receipt'];path=Path(v['path'])
  if ident in raw['cases']:
   print('PRESERVED_EXISTING_RESULT',ident,flush=True);continue
  if hashlib.sha256(path.read_bytes()).hexdigest()!=v['sha256']:raise ValueError('Source video content mismatch: '+ident)
  t=time.monotonic();frames=decode(path);f=feature(frames)
  raw['cases'][ident]={'source_sha256':v['sha256'],'frame_count':len(frames),'duration_seconds':len(frames)/8,'methods':{k:peak_features(f[k],k) for k in METRICS},'seconds':round(time.monotonic()-t,3)}
  PRED.write_text(json.dumps(raw,indent=2)+'\n')
  print('PRED',len(raw['cases']),'/',len(m['items']),ident,'frames',len(frames),'seconds',raw['cases'][ident]['seconds'],'flags',{k:raw['cases'][ident]['methods'][k]['anomaly_flag'] for k in METRICS},flush=True)
 assert len(raw['cases'])==72
 raw['status']='complete_frozen_before_labels_scored'
 raw['runtime_current_iteration_seconds']=round(time.monotonic()-start,2)
 PRED.write_text(json.dumps(raw,indent=2)+'\n')
 print('FROZEN_ALL_72',PRED,'NEW_ITERATION_SECONDS',raw['runtime_current_iteration_seconds'])


def random_peaks(rng,n):
 # Same original pre-declared nonmaximum spacing >=3 x 8FPS frame steps.
 take=[]
 for idx in rng.permutation(n):
  if all(abs(int(idx)-j)>=3 for j in take):take.append(int(idx))
  if len(take)==3:break
 assert len(take)==3
 return [round((x+.5)/8,4) for x in take]


def indicators(rows,metric):
 a=[r for r in rows if r['has_human_motion']];b=[r for r in rows if not r['has_human_motion']]
 tp=sum(r['signals'][metric]['flag'] for r in a);fp=sum(r['signals'][metric]['flag'] for r in b)
 events=sum(len(r['motion_intervals']) for r in a);hit=sum(r['signals'][metric]['events_found'] for r in a)
 return {'human_motion_positive_videos':len(a),'no_motion_annotation_videos_NOT_CERTIFIED_CLEAN':len(b),'TP':tp,'FN':len(a)-tp,'FP_proxy':fp,'TN_proxy':len(b)-fp,'positive_recall':tp/len(a) if a else None,'motion_events':events,'top3_event_hits':hit,'event_hit_fraction':hit/events if events else None}


def score():
 if SCORES.is_file():print('PRIOR_COMPLETED_SCORES_NOT_REPLAYED');return
 manifest=json.loads(SOURCE.read_text());raw=json.loads(PRED.read_text());assert raw['status']=='complete_frozen_before_labels_scored'
 assert len(raw['cases'])==72 and len(manifest['items'])==72
 if len(set(x['prompt_id'] for x in manifest['items']))!=24:raise ValueError('unexpected prompt families')
 records=[]
 for source in manifest['items']:
  ident=source['id'];p=raw['cases'][ident]
  intervals=[]
  for ann in source['motion_annotations']:
   b,e=ann['segment'];b=max(0,float(b));e=min(float(e),float(p['duration_seconds']))
   if e<=b:raise ValueError('invalid human annotation times for '+ident)
   intervals.append([b,e])
  record={'id':ident,'prompt':source['prompt_id'],'generator':source['generator'],'positive_selected_prompt_group':source['selection_group']=='motion_label_enriched','has_human_motion':bool(intervals),'motion_intervals':intervals,'other_error_types':source['other_annotation_types'],'signals':{}}
  for key in METRICS:
   measured=p['methods'][key];peaks=[x['seconds'] for x in measured['top3_local_peaks']]
   hits=sum(any(b<=t<=e for t in peaks) for b,e in intervals)
   record['signals'][key]={'flag':measured['anomaly_flag'],'top3':peaks,'events_found':hits,'max_z':measured['max_z']}
  records.append(record)
 summary={key:indicators(records,key) for key in METRICS}
 rng=np.random.default_rng(20261025)
 simulated=np.zeros(N,dtype=np.int16)
 positive=[r for r in records if r['has_human_motion']]
 for r in positive:
  n=raw['cases'][r['id']]['methods'][METRICS[0]]['n_timepoints']
  for rep in range(N):
   tops=random_peaks(rng,n)
   simulated[rep]+=sum(any(b<=t<=e for t in tops) for b,e in r['motion_intervals'])
 reference={'random_peak_mean_event_hits':float(np.mean(simulated)),'random_peak_std_event_hits':float(np.std(simulated)),'random_peak_95pct_interval':[int(np.quantile(simulated,.025)),int(np.quantile(simulated,.975))],'draws':N,'rng_seed':20261025,'peak_limit':3,'minimum_distance_frames':3}
 for key,d in summary.items():
  d['random_top3_hit_mean']=reference['random_peak_mean_event_hits']
  d['random_peak_ge_observed_frequency']=float(np.mean(simulated>=d['top3_event_hits']))
 generators={g:{key:indicators([r for r in records if r['generator']==g],key) for key in METRICS} for g in ('ltx2','seedance','veo3')}
 # Each prompt shared by 3 models; bootstrap source prompt groups for imperfect
 # confidence on *binary* motion-positive recall only (external labels subset).
 clusters=collections.defaultdict(list)
 for r in records:clusters[r['prompt']].append(r)
 groups=list(clusters)
 boot={}
 for key in METRICS:
  rng=np.random.default_rng(20261025)
  recalls=[]
  for _ in range(2000):
   drawn=rng.choice(groups,size=len(groups),replace=True)
   items=[r for group in drawn for r in clusters[group] if r['has_human_motion']]
   if items:recalls.append(sum(r['signals'][key]['flag'] for r in items)/len(items))
  boot[key]={'recall_95pct_prompt_cluster_bootstrap':[float(np.quantile(recalls,x)) for x in (.025,.975)],'bootstrap_replicates':2000,'prompt_groups':len(groups)}
 out={'source':'Spotlight ECCV 2026 human motion annotations with 72 original T2V videos, 24 prompt families fully prompt-disjoint from prior 24','source_manifest_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'frozen_pixel_predictions_sha256':hashlib.sha256(PRED.read_bytes()).hexdigest(),'source_original_provenance':'UBC-ViL/Spotlight-VideoGen-Errors, CC BY-NC 4.0, pinned 4165ea17ece2fda7b6b0df52bcbf07348a31253e','same_predeclared_parameters_as_prior_pilot':True,'no_threshold_tuning_or_model_selection_on_this_data':True,'human_motion_positive_videos':len(positive),'human_motion_unlabeled_not_true_clean':len(records)-len(positive),'human_motion_intervals':sum(len(r['motion_intervals']) for r in positive),'prompt_groups':len(groups),'n_videos':len(records),'metrics':summary,'by_generator':generators,'random_three_peak_timepoint_control':reference,'prompt_cluster_bootstrap':boot,'cases':records,'limits':'24 prompt families chosen based on human motion annotations enriched 12 vs 12, not population random. Prior 24 had already revealed method weaknesses, so this is prompt-disjoint fixed-rule confirmatory diagnostic but not untouched high-stakes final holdout. Motion label absence is NOT verified clean and other errors are present. Human error segments often long, random-time control required. Fixed 8FPS frame/camera decoder may miss short motion flaws. Frame-signal detection is not semantic physics or video-language model accuracy.'}
 SCORES.write_text(json.dumps(out,indent=2)+'\n')
 print('CONFIRMATORY_SUMMARY',json.dumps(summary,indent=2),'RANDOM',reference)
 print('HELDOUT_GROUPS',generators)

if __name__=='__main__':compute();score()
