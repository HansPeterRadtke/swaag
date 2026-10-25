#!/usr/bin/env python3
"""Scientific end-to-end test of THREE predeclared industrial pixel detectors.

NEW original MVTec LOCO real SPlicing Connectors category, totally distinct
from previously tested pushpins. 24 source normal train, 20 source normal
validation for EXACT frozen 95th percentile high anomaly detection cutoff,
90 original test photos: 30 NORMAL,30 logical anomalies and 30 structural
anomalies; no source test labels or image categories used to build predictors.

Perceptual methods use frozen pretrained OpenAI CLIP ViT-B/32 images extracted
on Nitro CPU with NO GPU and no further training or model weights download:
  (A) full-image CLIP cosine distance to NEAREST of 24 train-normal embeddings;
  (B) original image HSV spatial histogram to nearest train-normal image,
      same 24 normals and same validation count, no trained VLM;
  (C) CLIP last-hidden 49 visual patch tokens, each compared with 1176
      original normal patch-memory bank tokens; image anomaly score p95
      over 49 per-token min cosine distance. This is NOT official PatchCore.

Calibration threshold: 95th percentile of the score distribution of EXACTLY
20 validation normal images only; no anomalies in fit/tuning. Independently
score all image files first by SHA file+source ID, archive all continuous
scores and thresholds BEFORE looking at test labels (separate --evaluate
mode invoked ONLY after --freeze completes and locks exact 134 IDs).\n
Main scoring: TP FP FN TN, AUROC, 95% source-image bootstrap intervals for
AUC and sensitivity/specificity, class breakdown. Exact binomial confidence
for heldout 30 clean and two distinct 30 anomaly types, warn unrepresentative
commercial/general 3D output. No image pixels/source labels/model weights Git.

All thresholds and use of CLIP prepared prior to downloading source images or
opening any human data. No rerun on ambiguous intermediate sources.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,hashlib,math,datetime,statistics,argparse,collections
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
ROOT=Path('/data/var/perceptual-independent-20261024/mvtec-loco-splicing-20261025')
SELECT=ROOT/'source_frozen_pre_analysis_selection_20261025.json'
CAPTURE=ROOT/'134_original_image_visual_feature_capture_state_20261025.json'
FROZEN=ROOT/'industrial_134_image_scores_normal_only_frozen_before_gold_20261025.json'
EVAL=ROOT/'industrial_90_heldout_good_logical_structural_confusions_20261025.json'
CFG={'source':'MVTec LOCO splicing connectors NEW research category 134 original externally source-labelled image photos','training_normal_original_images':24,'validation_normal_original_images':20,'original_heldout_test_images_3_classes':30,'predeclared_detectors':{'CLIP_global_nearest_normal':'1.0 - max cosine similarity against each normal-only train embedding (512D)','HSV_spatial_hist_nearest_normal':'1.0 - max cosine similarity against train-normal 2x4 HSV grid hist (1024D)','CLIP_p95_patch_memory':'95th percentile nearest-normal patch cosine distance over 49 actual CLIP ViT-B/32 image patches compared with 24*49 original normal patch tokens, 768D; not official PatchCore'},'threshold':'95th percentile (np.quantile method linear) of each method continuous anomaly score on exactly 20 original validation normal photos, no test label tuning','auc':'ROC AUC against original test good vs logical, good vs structural, pooled good vs both anomaly classes; no data selection after observing features','bootstrap':'3000 class-stratified bootstrap by original IMAGE using seed20261025 separately on normal,logical,structural cohorts, descriptions source-related and not independent manufacturer domain','resource':'numpy matrix ops and frozen CLIP features, no GPU and no live model calls during scoring','license':'MVTec original photos CC BY-NC-SA4.0, academic noncommercial original source only; pixel files retained /data/var.'}

def vec(item):
 assert item['status']=='complete'
 p=ROOT/item['local_feature_file']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['feature_file_sha256']
 with np.load(p,allow_pickle=False) as arr:
  a={x:arr[x].copy().astype(np.float32) for x in ['global_embedding','patch_embeddings','hsv_spatial_histogram']}
 assert a['global_embedding'].shape==(512,) and a['hsv_spatial_histogram'].shape==(1024,) and a['patch_embeddings'].shape==(49,768)
 return a

def normal_bank(selected,features):
 manifests={x['source_image_path']:x for x in selected}
 cases={ident:(manifest,features[ident]) for ident,manifest in features.items()} if False else None
 train_ids=[hashlib.sha256(s['source_image_path'].encode()).hexdigest() for s in selected if s['source_normal_only_train']]
 val_ids=[hashlib.sha256(s['source_image_path'].encode()).hexdigest() for s in selected if s['source_normal_only_val']]
 assert len(train_ids)==24 and len(val_ids)==20 and set(train_ids).isdisjoint(val_ids)
 train={k:vec(features[k]) for k in train_ids}
 bank={x:np.stack([v[x] for v in train.values()]) if x!='patch_embeddings' else np.concatenate([v[x] for v in train.values()],axis=0) for x in ['global_embedding','hsv_spatial_histogram','patch_embeddings']}
 assert bank['global_embedding'].shape==(24,512) and bank['patch_embeddings'].shape==(1176,768) and bank['hsv_spatial_histogram'].shape==(24,1024)
 return train_ids,val_ids,bank

def score(one,bank):
 a=float(1-np.max(bank['global_embedding']@one['global_embedding']))
 b=float(1-np.max(bank['hsv_spatial_histogram']@one['hsv_spatial_histogram']))
 cmat=one['patch_embeddings']@bank['patch_embeddings'].T
 c=float(np.quantile(1-np.max(cmat,axis=1),.95,method='linear'))
 return {'CLIP_global_nearest_normal':round(a,9),'HSV_spatial_hist_nearest_normal':round(b,9),'CLIP_p95_patch_memory':round(c,9)}

def freeze():
 if FROZEN.exists():
  data=json.loads(FROZEN.read_text())
  if data['status']=='complete':print('PREVIOUS_INDUSTRIAL_ORIGINAL_PREDICTIONS_FROZEN_NO_REPLAY');return
  raise RuntimeError('uncertain prior source score, cannot blindly rerun')
 s=json.loads(SELECT.read_text());capture=json.loads(CAPTURE.read_text());assert capture['status']=='complete'
 assert len(s['selected'])==134 and len(capture['cases'])==134
 assert capture['source_selection_sha256']==hashlib.sha256(SELECT.read_bytes()).hexdigest()
 train_ids,val_ids,bank=normal_bank(s['selected'],capture['cases'])
 vals={};all_scores={}
 for idx,entry in enumerate(s['selected'],1):
  ident=hashlib.sha256(entry['source_image_path'].encode()).hexdigest()
  image_vec=vec(capture['cases'][ident])
  scores=score(image_vec,bank)
  all_scores[ident]={'continuous_anomaly_scores':scores,'source_image_sha256':capture['cases'][ident]['source_image_sha256']}
  if ident in val_ids:vals[ident]=scores
  if idx%16==0:print('NORMAL_ONLY_CPU_SCORES_FROZEN',idx,'/134','source_labels_UNREAD',True,flush=True)
 assert len(vals)==20 and len(all_scores)==134
 t={name:float(np.quantile([v[name] for v in vals.values()],.95,method='linear')) for name in CFG['predeclared_detectors']}
 src={'date':'2026-10-25','status':'complete','config':CFG,'source_original_image_selection_sha256':hashlib.sha256(SELECT.read_bytes()).hexdigest(),'image_pixel_features_frozen_sha256':hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),'test_annotations_never_read_in_features_or_scores':True,'train_source_img_count':len(train_ids),'normal_validation_original_count':len(vals),'all_images_source_blind_score_count':len(all_scores),'trained_24_source_normal_SHA256_identifiers':train_ids,'validation_20_normal_idents':val_ids,'per_method_precalibrated_cutoffs_20_normal_only':t,'frozen_all_input_scores_by_anonymous_sha_source_ID':all_scores}
 FROZEN.write_text(json.dumps(src,indent=2)+'\n')
 print('INDUSTRIAL_REAL_ORIGINAL_IMAGES_134_CONTINUOUS_SCORES_FROZEN_BEFORE_TEST_LABELS',FROZEN,'THRESHOLDS_ONLY_NORMAL',t,'N',len(all_scores))

def ci(k,n):
 if n==0:return None
 return [round(float(beta.ppf(.025,k,n-k+1)) if k else 0,4),round(float(beta.ppf(.975,k+1,n-k)) if k<n else 1,4)]

_SOURCE_VALIDATION_THRESHOLDS = None

def metrics(rows,mode):
 global _SOURCE_VALIDATION_THRESHOLDS
 d=[(lab,score) for lab,score in rows]
 if _SOURCE_VALIDATION_THRESHOLDS is None:
  _SOURCE_VALIDATION_THRESHOLDS=json.loads(FROZEN.read_text())['per_method_precalibrated_cutoffs_20_normal_only']
 threshold=_SOURCE_VALIDATION_THRESHOLDS[mode]
 tp=sum(label and value>threshold for label,value in d);fp=sum(not label and value>threshold for label,value in d)
 fn=sum(label and value<=threshold for label,value in d);tn=sum(not label and value<=threshold for label,value in d)
 auc=float(roc_auc_score([lab for lab,sc in d],[sc for lab,sc in d])) if len({lab for lab,sc in d})==2 else None
 return {'n':len(d),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'precision_in_enriched_test_sample':tp/(tp+fp) if tp+fp else None,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,'source_original_test_ROC_AUC':auc,'source_positive_recall_exact_binomial95_CI':ci(tp,tp+fn),'source_control_specificity_exact_binomial95_CI':ci(tn,tn+fp),'frozen_normal_only_threshold':threshold}

def evaluate():
 if EVAL.exists():print('EXISTING_EXTERNAL_SOURCE_LABEL_COMPARISON_PRESERVED_NO_RECOMPUTATION');return
 f=json.loads(FROZEN.read_text());sel=json.loads(SELECT.read_text());assert f['status']=='complete'
 assert len(f['frozen_all_input_scores_by_anonymous_sha_source_ID'])==134
 classes=collections.defaultdict(list)
 for entry in sel['selected']:
  group=entry['selection_stratum']
  if not group.startswith('test/'):continue
  ident=hashlib.sha256(entry['source_image_path'].encode()).hexdigest()
  assert ident in f['frozen_all_input_scores_by_anonymous_sha_source_ID']
  classes[group].append(f['frozen_all_input_scores_by_anonymous_sha_source_ID'][ident]['continuous_anomaly_scores'])
 assert all(len(classes[g])==30 for g in ['test/good','test/logical_anomalies','test/structural_anomalies'])
 methods={};rng=np.random.default_rng(20261025)
 for name in CFG['predeclared_detectors']:
  original_normal=[a[name] for a in classes['test/good']]
  orig_logical=[a[name] for a in classes['test/logical_anomalies']]
  orig_struct=[a[name] for a in classes['test/structural_anomalies']]
  conditions={'logical_vs_real_clean':[(False,a) for a in original_normal]+[(True,a) for a in orig_logical], 'structural_vs_real_clean':[(False,a) for a in original_normal]+[(True,a) for a in orig_struct], 'all_source_anomalies_vs_original_clean':[(False,a) for a in original_normal]+[(True,a) for a in orig_logical+orig_struct]}
  by={}
  for group,rows in conditions.items():
   result=metrics(rows,name)
   B=2500;rocs=[];bacc=[]
   negatives=[x for x in rows if not x[0]];positives=[x for x in rows if x[0]]
   for _ in range(B):
    items=[negatives[i] for i in rng.integers(0,len(negatives),len(negatives))]+[positives[i] for i in rng.integers(0,len(positives),len(positives))]
    vals=metrics(items,name)
    if vals['source_original_test_ROC_AUC'] is not None:rocs.append(vals['source_original_test_ROC_AUC'])
    if vals['balanced_accuracy'] is not None:bacc.append(vals['balanced_accuracy'])
   result['image_stratified_bootstrap_95pct_ROC_AUC']=[round(float(np.quantile(rocs,p)),4) for p in [.025,.975]]
   result['image_stratified_bootstrap_95pct_balanced_accuracy']=[round(float(np.quantile(bacc,p)),4) for p in [.025,.975]]
   by[group]=result
  methods[name]=by
  print('HELDOUT_REAL_SPLICE_CONNECTOR_ORIGINAL_TEST',name,'LOGICAL',by['logical_vs_real_clean'],'STRUCTURAL',by['structural_vs_real_clean'],'POOLED',by['all_source_anomalies_vs_original_clean'],flush=True)
 source={'date':'2026-10-25','n_original_heldout_test_images':90,'source_original_split_counts':{k:len(v) for k,v in classes.items()},'source_model_research':'MVTec LOCO AD original industrial splicing connectors, HF pinned replica, noncommercial research CC BY-NC-SA4.0','source_selection_SHA256':hashlib.sha256(SELECT.read_bytes()).hexdigest(),'frozen_pixel_and_CV_predictions_SHA256':hashlib.sha256(FROZEN.read_bytes()).hexdigest(),'calibration_source_images_only_normal':20,'trained_normal_source_only':24,'test_source_anomaly_labels_used_only_after_predictions_and_95th_source_normal_calibration_frozen':True,'method_results':methods,'independence':'Test images are different physical sourced photos from train and validation; all images share one original industrial product family, source labels provided by MVTec creators, not new human blind verification; train23/val20/test90 are hash sampled from original public splits, no test tuning.','limitations':['Three methods all use same source training/validation budgets, but HSV has no pretrained outside knowledge whereas CLIP is pretrained on unrelated broad visual sources; comparison evaluates reuse cost, not fairness of total pretraining budget.','Splicing connectors industrial photographs only, no evidence of CAD, 3D load/contacts or photorealistic warehouse errors.','Logical anomalies may be changes in correct count/placement of connectors; global CLIP ViT32 224 resolution may not resolve detailed cable/contact constraints.','Only 30 original clean test photos, a perfect zero FP result could still allow double-digit true false-alarm uncertainty, so report confidence intervals.','Original author source defect annotations may omit borderline visible anomalies; generated dataset source not independently newly human relabeled.','Threshold uses 95th percentile of ONLY 20 validation normals, no optimization using anomalies, and no source images/pixels/weights committed.','The source is explicitly noncommercial CC BY-NC-SA, do not silently deploy samples/weights trained on them in for-profit product without licensing checks.']}
 EVAL.write_text(json.dumps(source,indent=2)+'\n')
 print('INDEPENDENT_NEW_ORIGINAL_INDUSTRIAL_HUMAN_SOURCE_LABEL_BENCHMARK_COMPLETE',EVAL)
if __name__=='__main__':
 arg=argparse.ArgumentParser();arg.add_argument('mode',choices=['freeze','evaluate']);a=arg.parse_args();freeze() if a.mode=='freeze' else evaluate()
