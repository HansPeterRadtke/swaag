#!/usr/bin/env python3
"""POSTHOC, not untouched: same-image paired AUC uncertainty of four fixed modes.

Four models/methods and original heldout source photo selection/thresholds
were frozen BEFORE MVTec anomaly test labels. We have now seen their results,
so this *additional comparative inference* is exploratory/development, not
an untouched super-test. Independently source-selected screw_bag original
camera images, 30 clean/30 logical/30 structural test (MVTec source author
labels), 24 normal train/20 normal calibration. Source images and scores under
/data/var only; only signed aggregate in Git. Pair bootstrap resamples exact
image IDs jointly across all method scores within source category, 4000 draws.
No inference, threshold changes, source file edits, model loads or GPU usage.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,numpy as np
ROOT=Path('/data/var/perceptual-independent-20261024/mvtec-loco-screwbag-20261025')
MAN=ROOT/'source_frozen_pre_analysis_selection_20261025.json'
SCORE=ROOT/'134_screwbag_normalonly_resnet_global_padim_HSV_scores_frozen_20261025.json'
OUT=ROOT/'posthoc_PaDiM_screwbag_exact_paired_method_comparison_20261025.json'
MODES=['ResNet18_global_cosine_nearest_normal','ResNet18_spatial_gaussian_mahalanobis_max','ResNet18_spatial_gaussian_mahalanobis_p95','HSV_spatial_color_nearest_normal']

def auc(p,n):
 return float(np.mean(p[:,None]>n[None])+0.5*np.mean(p[:,None]==n[None]))

def main():
 if OUT.exists():print('POSTHOC_CURRENT_ORIGINAL_SOURCE_EVIDENCE_PRESERVED');return
 src=json.loads(MAN.read_text());frozen=json.loads(SCORE.read_text());assert frozen['status']=='complete'
 if frozen['source_media_selection_sha256']!=hashlib.sha256(MAN.read_bytes()).hexdigest():raise RuntimeError('source mismatch')
 groups={'test/good':[],'test/logical_anomalies':[],'test/structural_anomalies':[]}
 for row in src['selected']:
  if row['selection_stratum'] not in groups:continue
  key=hashlib.sha256(row['source_image_path'].encode()).hexdigest()
  x=frozen['per_source_anonymous_image_score'][key]['continuous_model_scores']
  groups[row['selection_stratum']].append([x[m] for m in MODES])
 assert all(len(v)==30 for v in groups.values())
 normal=np.asarray(groups['test/good']);logical=np.asarray(groups['test/logical_anomalies']);structural=np.asarray(groups['test/structural_anomalies']);positives=np.concatenate([logical,structural],axis=0)
 observed={m:auc(positives[:,i],normal[:,i]) for i,m in enumerate(MODES)}
 rng=np.random.default_rng(20261025);delta={m:[] for m in MODES if m!='ResNet18_spatial_gaussian_mahalanobis_p95'}
 ncase=4000
 for _ in range(ncase):
  norm=normal[rng.integers(0,30,30)]
  pos=np.concatenate([logical[rng.integers(0,30,30)],structural[rng.integers(0,30,30)]],axis=0)
  au=[auc(pos[:,i],norm[:,i]) for i in range(len(MODES))]
  for i,m in enumerate(MODES):
   if m in delta:delta[m].append(au[2]-au[i])
 ci={m:{'ROC_AUC_spatial_p95_minus_other':observed[MODES[2]]-observed[m],'same_original_source_image_pair_bootstrap_95pct':[float(np.quantile(v,.025)),float(np.quantile(v,.975))]} for m,v in delta.items()}
 flags={m:((positives[:,i]>frozen['normal_only_predeclared_thresholds'][m]),(normal[:,i]>frozen['normal_only_predeclared_thresholds'][m])) for i,m in enumerate(MODES)}
 comparisons={}
 for m in MODES:
  if m==MODES[2]:continue
  a,b=flags[MODES[2]][0],flags[m][0];c,d=flags[MODES[2]][1],flags[m][1]
  comparisons[m]={'positive_images_flagged_only_by_spatial_p95':int(np.sum(a&~b)),'positive_images_flagged_only_by_comparator':int(np.sum(~a&b)),'positive_images_flagged_by_both':int(np.sum(a&b)),'good_images_false_flagged_only_by_spatial_p95':int(np.sum(c&~d)),'good_images_false_flagged_only_by_comparator':int(np.sum(~c&d)),'good_images_false_flagged_by_both':int(np.sum(c&d))}
 output={'date':'2026-10-25','explicit_posthoc_not_independent_supertest':True,'data_selection':'Original external MVTec screw bag independent original images 30 source good/30 logical/30 structural, source label joined only after four methods frozen image scores, one source product family','original_sha256':{'immutable_source_selection':hashlib.sha256(MAN.read_bytes()).hexdigest(),'all_original_image_scores_before_test_gold':hashlib.sha256(SCORE.read_bytes()).hexdigest()},'auc_over_same_90_original_images':observed,'paired_spatial_p95_vs_global_max_HSV':ci,'paired_event_discordances':comparisons,'bootstrap_4000_source_image_stratified_same_indices_all_methods':True,'limits':'Model comparison post hoc on consumed original test set, not preregistered relative superiority claim and not unchanged heldout independent training domain. Correlated original photos, not independent manufacturing plant. Conditional sensitivity on one source may misrepresent production false negatives. Does not certify physical assembly correctness, photorealism or human intent.'}
 OUT.write_text(json.dumps(output,indent=2)+'\n')
 print('NEW_REAL_INDUSTRIAL_PAIRED_SOURCE_SPATIAL_AUC_UNCERTAINTY',observed,'DELTA_CI',ci,'DISCORDANCES',comparisons,'SAVED',OUT)
if __name__=='__main__':main()
