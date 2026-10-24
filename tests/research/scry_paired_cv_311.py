#!/usr/bin/env python3
"""Blind paired-image CV baseline on 311 human-reviewed Scry mobile UI diffs.

Model stage reads ONLY image pair and app group/id, never human issue counts or
boxes. Calibration folds are selected by stable app_slug hash (not row split).
Calibrate score threshold on 14 app groups; evaluate on separate 38 groups.
No-tagged controls are NOT certified perfect screens: FPR is a diagnostic.
Spatial boxes are raw difference components, NOT semantic issue predictions.
All screenshots & original annotations stay outside Git; code/results aggregate only.
"""
from __future__ import annotations
import collections,hashlib,io,json,os,statistics,time,resource
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity
import pyarrow.parquet as pq

ROOT=Path('/data/var/perceptual-independent-20261024/scry311')
RAW=ROOT/'scry_cv_pixel_only_predictions.json'
SCORES=ROOT/'scry_cv_human_evaluation.json'
CONFIG={
 'input':'paired human-reviewed mobile reference A and generated implementation B images',
 'image_size_wh':[256,512],
 'transform':'resize both reference A and generated B to exactly 256x512; handles unequal screenshot sizes via geometric stretch; no registration, so variation in aspect/resolution causes confounds',
 'image_metrics':['normalized_RGB_MAE','1_minus_grayscale_SSIM'],
 'calibration_group':'int(sha256(app_slug)[:8],16) modulo 4 equals zero (14 app groups); rest 38 groups heldout',
 'calibration_grid':[round(x/100,2) for x in range(2,82,2)],
 'threshold_choice':'maximize calibration balanced accuracy across no-tagged-vs-tagged screen outcomes; tie choose highest threshold (fewer flags)',
 'which_candidate_selected':'best calibration balanced accuracy; tie favor SSIM',
 'pixel_change_threshold':64,
 'difference_gaussian_kernel':[5,5],
 'difference_morphology_close':[7,7],
 'component_min_area_px':80,
 'component_min_width_height_px':6,
 'max_boxes':25,
 'issue_box_overlap_iou':0.1,
 'no_tagged_caution':'77 no-tagged pairs are not audited as truly defect-free; compute control flag rate as proxy, not absolute specificity',
 'freeze':'parameters defined before computing predictions; no test labels used by predictor or threshold selection',
}

def metrics(data_a,data_b):
 with Image.open(io.BytesIO(data_a)).convert('RGB') as a:
  A=np.asarray(a.resize((256,512),Image.Resampling.BILINEAR),dtype=np.uint8)
 with Image.open(io.BytesIO(data_b)).convert('RGB') as b:
  B=np.asarray(b.resize((256,512),Image.Resampling.BILINEAR),dtype=np.uint8)
 mae=float(np.abs(A.astype(np.int16)-B.astype(np.int16)).mean()/255.)
 gray_a=cv2.cvtColor(A,cv2.COLOR_RGB2GRAY);gray_b=cv2.cvtColor(B,cv2.COLOR_RGB2GRAY)
 ssim=structural_similarity(gray_a,gray_b,data_range=255,gaussian_weights=False)
 anomaly=1-float(ssim)
 # Crude image-difference components, cannot emit semantic labels. Difference
 # evaluated in normalized pixel coordinates of B for comparison with box_b.
 raw_diff=np.max(cv2.absdiff(A,B),axis=2)
 smooth=cv2.GaussianBlur(raw_diff,(5,5),0)
 mask=(smooth>=64).astype(np.uint8)*255
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
 components=cv2.connectedComponentsWithStats(mask,connectivity=8)[2][1:]
 components=sorted(components,key=lambda x:int(x[4]),reverse=True)
 boxes=[]
 for x,y,w,h,area in components:
  if area>=80 and w>=6 and h>=6:
   boxes.append({'x':round(float(x)/256,6),'y':round(float(y)/512,6),'w':round(float(w)/256,6),'h':round(float(h)/512,6),'area_fraction':round(float(area)/(256*512),7)})
  if len(boxes)>=25:break
 return {'mae':round(mae,6),'ssim_difference':round(anomaly,6),'raw_diff_boxes':boxes,'diff_box_count':len(boxes)}

def compute():
 if RAW.exists():print('PREVIOUS_RAW_PREDICTIONS_PRESERVED_NOT_RECOMPUTED');return
 start=time.monotonic()
 results=[];shards=sorted(ROOT.glob('test-*-of-00016.parquet'))
 assert len(shards)==16
 cv2.setNumThreads(1)
 for shard in shards:
  table=pq.read_table(shard,columns=['id','app_slug','image_a','image_b'])
  for record in table.to_pylist():
   t=time.monotonic();a=record['image_a']['bytes'];b=record['image_b']['bytes']
   prediction=metrics(a,b)
   results.append({'id':record['id'],'app_slug':record['app_slug'],'image_a_sha256':hashlib.sha256(a).hexdigest(),'image_b_sha256':hashlib.sha256(b).hexdigest(),'seconds':round(time.monotonic()-t,4),**prediction})
  print('SHARD',shard.name,'ROWS',len(results),'SECONDS',round(time.monotonic()-start,2),flush=True)
 assert len(results)==311 and len({r['id'] for r in results})==311
 # Freeze raw predictions BEFORE reading or scoring any human annotations.
 RAW.write_text(json.dumps({'source':'Scrymore/scry-design-diff-eval observed 2026-10-24','config':CONFIG,'n':len(results),'predictions':results,'total_seconds':round(time.monotonic()-start,2),'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2)+'\n')
 print('FROZEN_RAW',RAW,'N',len(results),'seconds',round(time.monotonic()-start,2),flush=True)

def iou(a,b):
 x1=max(a['x'],b['x']);y1=max(a['y'],b['y']);x2=min(a['x']+a['w'],b['x']+b['w']);y2=min(a['y']+a['h'],b['y']+b['h'])
 inter=max(0,x2-x1)*max(0,y2-y1)
 area=a['w']*a['h']+b['w']*b['h']-inter
 return inter/area if area else 0

def confusion(items,name,threshold):
 tp=fp=tn=fn=0
 for case in items:
  pred=case[name]>threshold;gold=case['n_issues']>0
  if pred and gold:tp+=1
  elif pred and not gold:fp+=1
  elif not pred and gold:fn+=1
  else:tn+=1
 tpr=tp/(tp+fn) if tp+fn else None
 tnr=tn/(tn+fp) if tn+fp else None
 return {'TP':tp,'FP':fp,'TN':tn,'FN':fn,'sensitivity':tpr,'no_tagged_control_acceptance':tnr,'balanced_accuracy':(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
 'screen_precision_against_nonexhaustive_tags':tp/(tp+fp) if tp+fp else None}

def calibrate(items,name):
 grid=CONFIG['calibration_grid']
 scores=[(confusion(items,name,t)['balanced_accuracy'],t) for t in grid]
 scores=[(v,t) for v,t in scores if v is not None]
 return max(scores,key=lambda x:(x[0],x[1]))[1],max(scores,key=lambda x:(x[0],x[1]))[0]

def evaluate():
 if SCORES.exists():print('PREVIOUS_COMPLETE_SCORES_PRESERVED');return
 frozen=json.loads(RAW.read_text());assert frozen['n']==311 and frozen['config']==CONFIG
 rows={x['id']:x for x in frozen['predictions']}
 annotated={}
 for file in sorted(ROOT.glob('test-*-of-00016.parquet')):
  for record in pq.read_table(file,columns=['id','app_slug','n_issues','issue_labels','ground_truth_issues','split']).to_pylist():
   key=record['id'];assert key not in annotated
   annotated[key]=record
 assert len(annotated)==len(rows)==311
 paired=[]
 for key,pred in rows.items():
  label=annotated[key];assert pred['app_slug']==label['app_slug']
  fold='calibration' if int(hashlib.sha256(pred['app_slug'].encode()).hexdigest()[:8],16)%4==0 else 'heldout'
  paired.append({'id':key,'app_slug':pred['app_slug'],'fold':fold,'n_issues':label['n_issues'],'issues':json.loads(label['ground_truth_issues']),'issue_labels':label['issue_labels'],'mae':pred['mae'],'ssim_difference':pred['ssim_difference'],'boxes':pred['raw_diff_boxes']})
 cal=[r for r in paired if r['fold']=='calibration'];test=[r for r in paired if r['fold']=='heldout']
 assert len(cal)==62 and len(test)==249
 grid={}
 for model in ['mae','ssim_difference']:
  threshold,bacc=calibrate(cal,model)
  grid[model]={'threshold':threshold,'calibration_balanced_accuracy':bacc,'calibration':confusion(cal,model,threshold),'heldout':confusion(test,model,threshold)}
 selected=max(grid,key=lambda m:(grid[m]['calibration_balanced_accuracy'], int(m=='ssim_difference')))
 # Category/label localization diagnostic; NO predicted semantic label, so this
 # is box-IoU coverage ONLY and cannot be compared with published tag+box recall.
 boxes={}
 for fold,group in [('calibration',cal),('heldout',test)]:
  annotations=[];nmatched=0;models_boxes=0;controls=[];per_category=collections.defaultdict(lambda:[0,0])
  for row in group:
   proposed=row['boxes'];models_boxes+=len(proposed)
   if row['n_issues']==0:controls.append(len(proposed))
   for issue in row['issues']:
    truth=issue.get('box_b') or issue.get('box_a')
    if not truth:continue
    matched=any(iou(truth,p)>=CONFIG['issue_box_overlap_iou'] for p in proposed)
    nmatched+=matched
    annotations.append((row['id'],matched))
    for cat in issue.get('labels',[]):
     per_category[cat][0]+=int(matched);per_category[cat][1]+=1
  boxes[fold]={'issues_with_box':len(annotations),'bbox_iou_recovered_issues_no_tag_check':nmatched,'localization_only_recall':nmatched/len(annotations) if annotations else None,'predicted_boxes':models_boxes,'no_tagged_control_mean_candidate_boxes':statistics.mean(controls) if controls else None,'no_tagged_control_any_box_flag_fraction':sum(x>0 for x in controls)/len(controls) if controls else None,'per_category_loc_only':{k:{'matches':v[0],'total':v[1]} for k,v in per_category.items()}}
 # Confidence intervals cluster by app, controls are NOT truly certified clean.
 rng=np.random.default_rng(20261024)
 def cluster_interval(model):
  apps=sorted({r['app_slug'] for r in test});groups={a:[r for r in test if r['app_slug']==a] for a in apps}
  boot=[]
  for _ in range(500):
   selected_apps=rng.choice(apps,size=len(apps),replace=True)
   sample=[row for app in selected_apps for row in groups[app]]
   cm=confusion(sample,model,grid[model]['threshold'])
   if cm['balanced_accuracy'] is not None:boot.append(cm['balanced_accuracy'])
  return [float(np.quantile(boot,q)) for q in [.025,.975]] if len(boot)>30 else None
 for model in grid:grid[model]['heldout_group_bootstrap_balanced_accuracy_interval95']=cluster_interval(model)
 out={'date':'2026-10-24','dataset':'Scrymore/scry-design-diff-eval 311 human-reviewed paired screens','dataset_annotation_n':sum(r['n_issues'] for r in paired),'app_groups':len({r['app_slug'] for r in paired}),'folds':{'calibration_rows':len(cal),'calibration_app_groups':len({r['app_slug'] for r in cal}),'heldout_rows':len(test),'heldout_app_groups':len({r['app_slug'] for r in test})},'frozen_prediction_file_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'config':CONFIG,'threshold_methods':grid,'selected_calibration_method':selected,'localization_without_issue_semantics':boxes,'raw_compute_seconds':frozen['total_seconds'],'limitations':'Control images no-tagged but not exhaustively reviewed. Human issue boxes are positives, category labels not predicted by CV methods; localization-only recall cannot be compared with official strict tag-and-box known-issue recall. Different source A and implementation B may contain purposeful aesthetic changes. Aspect-resizing introduces image registration errors; app groups correlated. No claims of reliable fully automated UI QA.'}
 SCORES.write_text(json.dumps(out,indent=2)+'\n')
 print('SELECTED',selected,'FOLDS',out['folds']);print('RESULTS',grid);print('SPATIAL',boxes,flush=True)

def main():
 compute()
 if RAW.exists():evaluate()
if __name__=='__main__':main()
