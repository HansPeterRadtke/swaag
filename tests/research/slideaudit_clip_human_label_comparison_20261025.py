#!/usr/bin/env python3
"""Score frozen OpenAI CLIP ViT-B/32 on HUMAN SlideAudit visual flaws.

Task and prompts fixed before looking at labels. Source strong-agreement human
positive/negative pairs of the same source_type and alteration. Report fixed
P(defect)>0.5 confusion (FNs and false alarms), continuous ROC AUC, per-source
stratum and paired-stratified 3,000 bootstrap uncertainty. No post-hoc prompt
selection, fit or threshold tuning. Input is presentation slides (some altered),
NOT actual GUI application screenshots. Input labels not read by CLIP.
"""
from pathlib import Path
import json,hashlib,collections,statistics
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path('/data/var/perceptual-independent-20261024')
SOURCE=ROOT/'slideaudit_selection_20261025.json'
RAW=ROOT/'slideaudit_343_clip_image_only_20261025.json'
OUT=ROOT/'slideaudit_clip_343_human_flaw_comparison_20261025.json'
LABELS=['Content Overflow/Cut-off','Occluded Content','Insufficient Color Contrast for Readability']
B=3000;SEED=20261025

def cm(rows):
 tp=sum(x['truth'] and x['predicted'] for x in rows);fp=sum(not x['truth'] and x['predicted'] for x in rows)
 fn=sum(x['truth'] and not x['predicted'] for x in rows);tn=sum(not x['truth'] and not x['predicted'] for x in rows)
 return {'n':len(rows),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':tp/(tp+fn) if tp+fn else None,'specificity_on_strong_negatives':tn/(tn+fp) if tn+fp else None,'precision_on_balanced_source_sample':tp/(tp+fp) if tp+fp else None,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None}

def auc(rows):
 if len({r['truth'] for r in rows})<2:return None
 return float(roc_auc_score([r['truth'] for r in rows],[r['score'] for r in rows]))

def bootstrap(pairs):
 rng=np.random.default_rng(SEED);metrics={'recall':[],'specificity_on_strong_negatives':[],'roc_auc':[],'balanced_accuracy':[]}
 for _ in range(B):
  inds=rng.integers(0,len(pairs),size=len(pairs))
  data=[row for idx in inds for row in pairs[idx]]
  m=cm(data);m['roc_auc']=auc(data)
  for k,val in m.items():
   if k in metrics and val is not None:metrics[k].append(val)
 return {k:[round(float(np.quantile(v,.025)),4),round(float(np.quantile(v,.975)),4)] if len(v)>B/2 else None for k,v in metrics.items()}

def main():
 if OUT.exists():print('SOURCE_HUMAN_CLIP_SCORES_ALREADY_SAVED_NOT_RECOMPUTED');return
 src=json.loads(SOURCE.read_text());raw=json.loads(RAW.read_text())
 assert raw['status']=='complete' and len(raw['predictions'])==src['n_selected_unique']==343
 assert all(v['status']=='complete' for v in raw['predictions'].values())
 res={}
 for category in LABELS:
  selected=src['task_counts'][category];pos=selected['selected_positive_ids'];neg=selected['selected_negative_ids'];assert len(pos)==len(neg)
  pairs=[]
  for a,b in zip(pos,neg):
   pair=[]
   for sid,truth in [(a,True),(b,False)]:
    m=src['items'][str(sid)];prediction=raw['predictions'][f'slide_{sid:04d}'];p=prediction['positive_prompt_probability'][category]
    assert m['labels'][category]['strong'] and m['labels'][category]['positive']==truth
    pair.append({'slide_id_sha256':hashlib.sha256(str(sid).encode()).hexdigest(),'source_type':m['source_type'],'alteration':m['alteration'],'truth':truth,'score':p,'predicted':p>0.5})
   assert pair[0]['source_type']==pair[1]['source_type'] and pair[0]['alteration']==pair[1]['alteration']
   pairs.append(pair)
  rows=[r for pair in pairs for r in pair]
  results={'balanced_strong_positive_negative_pairs':len(pairs),'original_strong_positive_n':len(pos),'original_strong_negative_controls_n':len(neg),'binary_prompt_rule':cm(rows),'AUROC_positive_defect_prompt':auc(rows),'paired_bootstrap_95pct':bootstrap(pairs),'by_source':{s:{**cm([r for r in rows if r['source_type']==s]),'AUROC':auc([r for r in rows if r['source_type']==s])} for s in ('Gemini','gdcdataset','Google')},'source_original_weak_positive_excluded':selected['original_weak_positive_not_adjudicated'],'median_positive_probability':statistics.median(r['score'] for r in rows if r['truth']),'median_negative_probability':statistics.median(r['score'] for r in rows if not r['truth'])}
  res[category]=results
  print('CLIP_ZERO_SHOT_HUMAN_VERDICT',category,'N',len(rows),'AUC',results['AUROC_positive_defect_prompt'],'BIN',results['binary_prompt_rule'],'CI',results['paired_bootstrap_95pct'],flush=True)
 report={'date':'2026-10-25','source':'SlideAudit UIST25 CC BY4, 343 original actual and altered slide screenshots with strong human label agreement','model':'OpenAI CLIP ViT-B/32 local checkpoint, CPU-only, fixed six zero-shot phrases, no training on source labels','input_source_selection_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'model_frozen_predictions_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'model_weights_sha256':raw['model_weights_sha256'],'model_cpu_time_s':raw['total_cpu_wall_seconds'],'model_peak_rss_kb':raw['peak_rss_kb'],'tasks':res,'model_raw_config':raw['config'],'interpretation':'Zero-shot CLIP is a generic screenshot image encoder. Prompt probabilities are not calibrated defect confidence. Predicted similarities may correlate with source alterations, wording, text density and origin rather than human deficiency; no localization. 343 source slides are not application GUI. Strong negative means human annotated absence for ONE visual flaw only, not certified all flaw-free. A high score would require independent source-family and untouched project holdout before acceptance. This model is only a candidate benchmark, not live UI QA.'}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('CLIP_REAL_SLIDE_HUMAN_LABELS_EVALUATION_RECORDED',OUT)
if __name__=='__main__':main()
