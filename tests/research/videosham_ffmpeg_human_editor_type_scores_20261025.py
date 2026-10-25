#!/usr/bin/env python3
"""Score original author VIDEO EDIT TYPE using frozen official FFmpeg freezedetect.

No video images, editorial labels or source mutation used during actual FFmpeg
inference. Only after complete 38 original manipulated video clips with 19
intentional temporal edits and 19 intentionally edited but non-temporal source
controls, join fixed original public author manipulation categories. Compare
video-level freeze event flag, event count/longest duration, abstentions,
true-positive/false-positive/false-negative/true-negative. These labels are NOT
human-assessed objectionable playback quality or accurate event times.
"""
from pathlib import Path
import json,hashlib,collections,statistics,math
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path('/data/var/perceptual-independent-20261024/videosham-20261025')
SOURCE=ROOT/'source_selection_original_38_video_pairs_20261025.json'
RAW=ROOT/'videosham_38_external_real_ffmpeg_freezedetect_20261025.json'
OUT=ROOT/'videosham_official_ffmpeg_freezedetect_author_edit_scores_20261025.json'

def metric(rows):
 valid=[r for r in rows if r['predicted'] is not None]
 tp=sum(r['gold_temporal_editor'] and r['predicted'] for r in valid)
 fp=sum(not r['gold_temporal_editor'] and r['predicted'] for r in valid)
 fn=sum(r['gold_temporal_editor'] and not r['predicted'] for r in valid)
 tn=sum(not r['gold_temporal_editor'] and not r['predicted'] for r in valid)
 return {'scored_n':len(valid),'abstained_temporal_positive':sum(r['gold_temporal_editor'] and r['predicted'] is None for r in rows),'abstained_other_edited_control':sum(not r['gold_temporal_editor'] and r['predicted'] is None for r in rows),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall_on_editor_temporal_group':tp/(tp+fn) if tp+fn else None,'specificity_on_other_author_edited_control':tn/(tn+fp) if tn+fp else None,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None}

def auc(rows,feature):
 used=[r for r in rows if r[feature] is not None]
 return float(roc_auc_score([r['gold_temporal_editor'] for r in used],[r[feature] for r in used])) if len({r['gold_temporal_editor'] for r in used})==2 else None

def main():
 if OUT.exists():print('EXISTING_AUTHORED_FFMPEG_SCORE_PRESERVED');return
 s=json.loads(SOURCE.read_text());raw=json.loads(RAW.read_text())
 assert raw['status']=='complete' and len(raw['results'])==s['source_pairs_expected']==38
 assert raw['source_selection_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
 rows=[]
 for item in s['items']:
  id=str(item['source_video_id']);r=raw['results'][id]
  gold=bool(item['gold_temporal_intent'])
  rows.append({'source_id':id,'gold_temporal_editor':gold,'other_editor_category':item['author_manipulation_group'] if not gold else None,'status':r['status'],'predicted':bool(r['event_positive_flag']) if r['status']=='complete' else None,'event_count':r.get('event_count'),'max_recorded_duration_s':r.get('max_recorded_complete_freeze_seconds'),'total_completed_event_duration_s':r.get('known_freeze_duration_sum_seconds'),'unclosed_event':r.get('had_unclosed_freeze_event')})
 assert sum(r['gold_temporal_editor'] for r in rows)==19
 score=metric(rows)
 score['event_count_AUROC']=auc(rows,'event_count')
 score['longest_detected_freeze_AUROC']=auc(rows,'max_recorded_duration_s')
 rng=np.random.default_rng(20261025)
 boot={'balanced_accuracy':[],'AUROC':[]}
 for _ in range(2000):
  sampled=[rows[i] for i in rng.integers(0,len(rows),size=len(rows))]
  val=metric(sampled)['balanced_accuracy'];area=auc(sampled,'event_count')
  if val is not None:boot['balanced_accuracy'].append(val)
  if area is not None:boot['AUROC'].append(area)
 score['source_clip_bootstrap95']={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] if len(v)>800 else None for k,v in boot.items()}
 report={'date':'2026-10-25','scope':'Real Adobe VideoSham author-edited Vimeo footage, 19 source clips temporal edit versus 19 other editor-made modifications from other independent clips','source_selection_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'real_original_video_FFMPEG_frozen_output_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'filter':raw['config']['filter'],'not_trained_or_tuned_with_gold_labels':True,'result':score,'per_authored_edit_class':{cat:{'n':len(group),'freeze_event_flags':sum(r['predicted'] is True for r in group),'abstentions':sum(r['predicted'] is None for r in group)} for cat in {r['other_editor_category'] if r['other_editor_category'] else 'temporal_author_category' for r in rows} if (group:=[r for r in rows if (r['other_editor_category'] if r['other_editor_category'] else 'temporal_author_category')==cat])},'copyright_or_privacy':'Adobe Research License v1.1a noncommercial; no videos/raw source texts/model outputs published in project Git','limitations':['FFmpeg freezedetect flags pixel stasis for a certain duration; intentional freezes/slowed motion can be correctly flagged as phenomena but are not necessarily defects.','A video can have duplication, drop, reversal or speeding without a long pixel-identical hold; misses are expected for those temporal edit types.','Other edited source controls have changes unrelated to timing and may include static natural content or processing alterations; a flagged freeze is not necessarily false in pixel-difference terms.','This is video-level author-edit category truth, not independent human subjective acceptance labels or time-coded event truth.','Nineteen positives and controls give high uncertainty; no automatic production acceptance.']}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('INDEPENDENT_FFMPEG_FREEZE_ON_REAL_AUTHORED_EDITED_VIDEOS',score,'SAVED',OUT,flush=True)
if __name__=='__main__':main()
