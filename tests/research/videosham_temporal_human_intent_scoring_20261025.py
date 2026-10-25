#!/usr/bin/env python3
"""Independent human-authored original-video EDIT LABELS vs frozen image signals.

Adobe VideoSham original Vimeo footage and published manually specified edits.
Thirty-eight distinct source video families, 19 specifically temporal edits
(duplicate/reverse/drop/speed), 19 other edited videos (adding/removing/text/
background/audio), all preselected by fixed original source category and SHA.

Ground labels are author EDIT TYPE, not blind human ratings of playback quality,
not precise frame-level defect truth, and are not perfectly clean no-temporal-
anomaly controls. Score source-timestamp/gray-frame/pHash models only AFTER all
image-only input predictions were saved, without retuning any thresholds.
The model's abstentions and failed decoding remain separate, never success.

AUC and fixed-threshold count on the same holdout from a source benchmark;
bootstrap by original video clip ID. Since all threshold rules predeclared,
these are exploratory zero-fit validation; subsequently comparing models on
this set consumes it for any selection. Source material licensed separately.
"""
from pathlib import Path
import json,hashlib,collections,statistics,math
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path('/data/var/perceptual-independent-20261024/videosham-20261025')
MAN=ROOT/'source_selection_original_38_video_pairs_20261025.json'
PIX=ROOT/'38_real_clip_source_temporal_pixels_frozen_20261025.json'
SCORES=ROOT/'38_real_manually_assigned_video_temporal_edit_scores_20261025.json'
CFG={'source':'Adobe VideoSham v1.0 authentic original+manually edited public Vimeo pairs','author_temporal_positives':19,'author_other_edit_controls':19,'no_retrain_or_threshold_change':True,'no_ref_flags':['no_ref_hold_flag','no_ref_jerk_flag','no_ref_combined_flag'],'reference_aware_flag':'ref_abnormal_flag only if source matching scorable, otherwise ABSTAIN','source_metadata_confounded_baseline':'abs log edit/original duration > log 1.3','bootstrap':3000,'seed':20261025,'source_truth_scope':'author-specified manipulation category, not viewer-objectionable visual stutter or physical plausibility','unqualified':'if reference frame content match<25pct or original static and no-ref does not flag, report unknown, not negative'}


def cm(rows,pred):
 evaluated=[r for r in rows if r[pred] is not None]
 tp=sum(r['positive'] and r[pred] for r in evaluated);fp=sum(not r['positive'] and r[pred] for r in evaluated)
 fn=sum(r['positive'] and not r[pred] for r in evaluated);tn=sum(not r['positive'] and not r[pred] for r in evaluated)
 return {'tested':len(evaluated),'total_available':len(rows),'abstained_positive':sum(r['positive'] and r[pred] is None for r in rows),'abstained_other_edit_controls':sum(not r['positive'] and r[pred] is None for r in rows),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'source_author_temporal_recall_when_scored':tp/(tp+fn) if tp+fn else None,'source_other_edit_specificity_when_scored':tn/(tn+fp) if tn+fp else None,'precision_on_balanced_author_edit_type':tp/(tp+fp) if tp+fp else None,'balanced_accuracy_on_scorable':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None}

def roc(rows,key):
 valid=[r for r in rows if r[key] is not None and math.isfinite(float(r[key]))]
 if len({r['positive'] for r in valid})<2:return None,len(valid)
 return float(roc_auc_score([r['positive'] for r in valid],[r[key] for r in valid])),len(valid)

def boot(rows,pred,score):
 rng=np.random.default_rng(CFG['seed']);result={'balanced_accuracy':[],'AUROC':[]}
 for _ in range(CFG['bootstrap']):
  draw=[rows[i] for i in rng.integers(0,len(rows),size=len(rows))]
  m=cm(draw,pred);a,n=roc(draw,score)
  if m['balanced_accuracy_on_scorable'] is not None:result['balanced_accuracy'].append(m['balanced_accuracy_on_scorable'])
  if a is not None:result['AUROC'].append(a)
 return {k:([round(float(np.quantile(values,.025)),4),round(float(np.quantile(values,.975)),4)] if len(values)>1000 else None) for k,values in result.items()}


def clip_type(text):
 x=text.lower()
 if any(k in x for k in ['repeat','duplicate']):return 'frame_repetition'
 if any(k in x for k in ['reverse','backward']):return 'frame_reversal'
 if 'drop frame' in x:return 'frame_dropping'
 if any(k in x for k in ['speed up','faster']):return 'speedup'
 if any(k in x for k in ['slow','speed','shorter','length']):return 'speed_or_timeline_change'
 return 'author_temporal_unspecified'


def score():
 if SCORES.exists():print('SOURCE_EDIT_LABEL_SCORES_ALREADY_SAVED_NOT_RECALCULATED');return
 origin=json.loads(MAN.read_text());pixels=json.loads(PIX.read_text())
 assert origin['source_pairs_expected']==38 and len(origin['items'])==38
 assert pixels['status']=='complete' and len(pixels['cases'])==38
 assert pixels['source_selection_SHA256']==hashlib.sha256(MAN.read_bytes()).hexdigest()
 records=[]
 for item in origin['items']:
  ident=str(item['source_video_id']);data=pixels['cases'][ident]
  truth=bool(item['gold_temporal_intent'])
  kind=clip_type(item['author_instruction']) if truth else item['author_manipulation_group'].lower()
  row={'source_video_id':ident,'positive':truth,'author_edit_class':kind,'author_manipulation_category':item['author_manipulation_group'],'source_decoding_status':data['status'],'no_ref_hold':None,'no_ref_jerk':None,'no_ref_combined':None,'ref_anomaly':None,'source_duration_confounded':None,'image_motion_jerk_score':None,'image_low_frame_change_rate':None,'image_ref_abnormal_score':None,'reference_content_match_coverage':None,'source_original_fps':None,'source_processed_fps':None,'source_original_duration':None,'source_processed_duration':None}
  if data['status']=='complete':
   sig=data['processed']['signals'];ref=data['reference']
   row.update(no_ref_hold=bool(sig['no_ref_hold_flag']),no_ref_jerk=bool(sig['no_ref_jerk_flag']),no_ref_combined=bool(data['no_ref_combined_flag']),source_duration_confounded=bool(data['duration_confounded_flag']),image_motion_jerk_score=sig['no_ref_jerk_fraction'],image_low_frame_change_rate=sig['source_same_grayscale_threshold_0p8_count']/max(1,sig['frame_count']-1),reference_content_match_coverage=ref['valid_matches_fraction'],source_original_fps=data['original']['video_info']['frame_rate'],source_processed_fps=data['processed']['video_info']['frame_rate'],source_original_duration=data['original']['video_info']['duration_seconds'],source_processed_duration=data['processed']['video_info']['duration_seconds'])
   if ref['status']=='scorable':
    row['ref_anomaly']=bool(ref['ref_abnormal_flag']);row['image_ref_abnormal_score']=ref['ref_abnormal_deviation_gt_4_frames_fraction']
  records.append(row)
 assert sum(r['positive'] for r in records)==19
 assert sum(not r['positive'] for r in records)==19
 methods={
  'no_ref_hold':('no_ref_hold','image_low_frame_change_rate'),
  'no_ref_jerk':('no_ref_jerk','image_motion_jerk_score'),
  'no_ref_combined':('no_ref_combined','image_motion_jerk_score'),
  'reference_aware':('ref_anomaly','image_ref_abnormal_score'),
  'duration_only_CONFOUNDED_NOT_VISUAL':('source_duration_confounded','duration_ratio_score')
 }
 for r in records:
  if r['source_original_duration'] and r['source_processed_duration']:
   r['duration_ratio_score']=abs(math.log(r['source_processed_duration']/r['source_original_duration']))
  else:r['duration_ratio_score']=None
 outcome={}
 for key,(pred,feature) in methods.items():
  x=cm(records,pred);area,num=roc(records,feature)
  x.update({'continuous_feature':feature,'continuous_roc_auc':area,'available_continuous_scores':num,'source_clip_bootstrap95':boot(records,pred,feature)})
  outcome[key]=x
  print('EXTERNAL_VIDEO_SHAM_HUMAN_EDITOR_TYPE',key,x,flush=True)
 types={}
 for cat in sorted({r['author_edit_class'] for r in records}):
  group=[r for r in records if r['author_edit_class']==cat]
  types[cat]={'n':len(group),'no_ref_detected':sum(r['no_ref_combined'] is True for r in group),'reference_scored':sum(r['ref_anomaly'] is not None for r in group),'reference_detected':sum(r['ref_anomaly'] is True for r in group),'median_edited_to_source_duration_ratio':statistics.median(r['source_processed_duration']/r['source_original_duration'] for r in group if r['source_original_duration'] and r['source_processed_duration']) if any(r['source_original_duration'] for r in group) else None}
 documented={'date':'2026-10-25','data_source':'Adobe VideoSham 2022 manually edited original Vimeo videos, release v1.0, Adobe Research License Version 1.1a noncommercial','source_clip_selection_sha256':hashlib.sha256(MAN.read_bytes()).hexdigest(),'frozen_real_video_pixel_feature_SHA256':hashlib.sha256(PIX.read_bytes()).hexdigest(),'label_scope':'Author selected 19 temporal manipulations (frame repeats, deletions, reversals, speed/slow effects) versus 19 non-temporal edited videos of matched original sources. No blind human-rater subjective score or frame/time spans; positives may be *intentionally artistically* distorted. Other edit controls are not perfectly clean video reference, but have no author temporal edit label.','config':CFG,'n_independent_source_vimeo_snippets':len(records),'n_temporal_author_labels':19,'n_other_edited_source_controls':19,'decoding_failures':sum(r['source_decoding_status']!='complete' for r in records),'methods':outcome,'by_authored_edit_class':types,'case_original_ids_and_truth':records,'limits':['Compared with exact corresponding original clips; clips within same author Vimeo sources might share artistic/static frames, source-ID clustering not independent filming families.','Matched original to edited reference comparisons are legitimate for provenance checking but cannot infer if a deliberate creative slow-motion or freeze is actually objectionable.','Non-temporal processed controls may have differences in reencoding, edited durations and backgrounds; detected differences can arise from codec rather than temporal manipulation.','No exact source frame-location human truth; video-level detection only, NOT temporal event F1, video-render alignment confidence or perceptual rating.','Reference-aware model abstentions due low pHash content matching are not treated as negatives; report coverage explicitly.','Simple fixed threshold methods tested as provided; any subsequent algorithm selection on these same clips consumes benchmark and needs external novel source families.','All source video media and human original instructions remain under /data/var outside Git; only aggregated evidence should be published.','No GPU, remote multimodal video model or new external paid API used.']}
 SCORES.write_text(json.dumps(documented,indent=2)+'\n')
 print('COMPLETE_MANUALLY_EDITED_REAL_VIDEO_BENCHMARK_SCORED',SCORES,'METHODS',len(outcome),'REAL_SOURCE_CLIPS',len(records))
if __name__=='__main__':score()
