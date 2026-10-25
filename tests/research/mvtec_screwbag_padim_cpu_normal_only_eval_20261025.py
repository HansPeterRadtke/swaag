#!/usr/bin/env python3
"""Head-to-head independent industrial visual inference on NEW Screw Bag family.

MVTec LOCO original screw_bag categories NEW versus previous SWAAG pushpins
and splicing connectors, 134 selected from original release before images
or model scores: 24 train/good,20 val/good normal-only, test30 good,30 logical,
30 structural. No source test anomaly labels in score training/freezing code.

All image-only features from pretrained official ResNet18 ImageNet1K frozen
are saved FIRST, normalized source hashes pinned and no GPU used. Four
FIXED methods with same source 24+20 training/validation splits:
 - ResNet18 global feature nearest normal cosine distance;
 - ResNet18 PaDiM-inspired 28x28 spatial per-location 32dim Gaussian with
   pretrained selected layers1/2/3, covariance shrinkage 0.01I using only 24
   normal images, image maximum mahalanobis distance (sensitive to tiny flaws);
 - Same spatial Mahalanobis image p95 percentile, lower outlier sensitivity;
 - Original HSV spatial tile histogram to nearest source normal image.
This is not the official released PaDiM model, and the source labels are not
human-blind quality ratings.

Thresholds frozen as original 20 val/good anomaly score empirical q95 using
np.quantile(method=linear), before viewing ANY source TEST labels. All 134
continuous image scores and thresholds saved under /data/var BEFORE label
scoring. Heldout 90 original test photographs then scored by external source
MVTec good/logical/structural metadata, TP/FP/FN/TN and class-wise AUROC and
bootstrap by source image, true negatives, sensitivity/specificity, cost.

No original photo, mask, model weights, embeddings or per-case results enter
project Git. Restricted CC BY-NC-SA4.0; no right to deploy commercially.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,hashlib,collections,statistics,time,resource,argparse
from pathlib import Path
import numpy as np
from scipy.stats import beta
ROOT=Path('/data/var/perceptual-independent-20261024/mvtec-loco-screwbag-20261025')
MAN=ROOT/'source_frozen_pre_analysis_selection_20261025.json'
FEATURES=ROOT/'134_source_pixelonly_resnet18_imagenet_hsv_features_20261025.json'
PRED=ROOT/'134_screwbag_normalonly_resnet_global_padim_HSV_scores_frozen_20261025.json'
SCORE=ROOT/'90_screwbag_original_heldout_good_logical_structural_confusions_20261025.json'
MODES=['ResNet18_global_cosine_nearest_normal','ResNet18_spatial_gaussian_mahalanobis_max','ResNet18_spatial_gaussian_mahalanobis_p95','HSV_spatial_color_nearest_normal']
CFG={'source':'MVTec LOCO AD externally MVTec-labeled original screw_bag real photos, public HF Rafa612/mvtec_loco revision 839fee8178c10d2e50afddc671238ba85d9975d9, noncommercial CC BY-NC-SA4.0','normal_only_reference_n':24,'normal_only_validation_n':20,'test_normal_n':30,'test_logical_n':30,'test_structural_n':30,'embedding':'torchvision official ResNet18 Imagenet1K V1 frozen global512 and spatial selected32 x28x28','statistic':'per 28x28 pixel location 32D empirical mean covariance from 24 training normal spatial features only, covariance shrinkage +0.01 identity, invert 32x32, mahalanobis sqrt distance','spatial_scores':['maximum all784 spatial mahalanobis scores','empirical quantile 0.95 over 784 spatial mahalanobis scores'],'global_score':'nearest of24 original ResNet18 global image embeddings 1-max dot cosine','HSV_baseline':'nearest of24 original source normal 1024D histogram image embeddings 1-max dot cosine','all_thresholds':'empirical q0.95 on exactly 20 normal validation scores independently for each of four modes, no test selection','inference':'CPU one BLAS/PyTorch thread, no GPU, no anomaly trained labels, actual image pixel only','uncertainty':'2500 bootstrap samples stratified by original test source file label, seed20261025, source category one product family not manufacturer independent','statistics':'exact Clopper Pearson binomial95 sensitivity and specificity, ROC AUC on heldout source anomaly vs normal, no posthoc threshold search'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def vec(record):
 assert record['status']=='complete'
 p=ROOT/record['feature_relpath'];assert sha(p)==record['feature_file_sha256']
 with np.load(p,allow_pickle=False) as d:
  v={key:d[key].copy().astype(np.float32) for key in ['global_resnet_embedding','padim_spatial_selected_32ch','hsv_tile_histogram']}
 assert v['global_resnet_embedding'].shape==(512,) and v['padim_spatial_selected_32ch'].shape==(784,32) and v['hsv_tile_histogram'].shape==(1024,)
 return v

def get_keys(source):
 groups=collections.defaultdict(list)
 for entry in source['selected']:
  groups[entry['selection_stratum']].append(hashlib.sha256(entry['source_image_path'].encode()).hexdigest())
 assert {k:len(v) for k,v in groups.items()}=={'train/good':24,'validation/good':20,'test/good':30,'test/logical_anomalies':30,'test/structural_anomalies':30}
 assert len(set(sum(groups.values(),[])))==134
 return groups

def normal_distribution(train):
 normals=np.stack([v['padim_spatial_selected_32ch'] for v in train.values()]).astype(np.float32)
 assert normals.shape==(24,784,32)
 mean=normals.mean(axis=0)
 centered=normals-mean
 cov=np.einsum('npi,npj->pij',centered,centered,optimize=True)/(len(normals)-1)
 cov+=np.eye(32,dtype=np.float32)[None]*.01
 prec=np.linalg.inv(cov).astype(np.float32)
 assert prec.shape==(784,32,32)
 # Parameters deterministic function of train-normal source only, never
 # fitted with test labels or any validation anomaly scores.
 return mean,prec

def scores(v,glob,hsv,mean,prec):
 a=float(1-np.max(glob@v['global_resnet_embedding']))
 b=float(1-np.max(hsv@v['hsv_tile_histogram']))
 delta=v['padim_spatial_selected_32ch']-mean
 sq=np.einsum('pi,pij,pj->p',delta,prec,delta,optimize=True)
 maha=np.sqrt(np.maximum(sq,0))
 c=float(np.max(maha));d=float(np.quantile(maha,.95,method='linear'))
 return dict(zip(MODES,[round(a,9),round(c,9),round(d,9),round(b,9)]))

def freeze():
 if PRED.exists():
  old=json.loads(PRED.read_text())
  if old['status']=='complete':print('ALREADY_FROZEN_ALL_SCREWBAG_SCORES_NOT_REPLAYED');return
  raise RuntimeError('uncertain prior anomaly score output, no rerun')
 original=json.loads(MAN.read_text());base=json.loads(FEATURES.read_text())
 assert base['status']=='complete' and len(base['results'])==134
 assert base['source_selection_sha256']==sha(MAN)
 grouped=get_keys(original);t=time.monotonic()
 train={i:vec(base['results'][i]) for i in grouped['train/good']}
 global_bank=np.stack([v['global_resnet_embedding'] for v in train.values()])
 hsv_bank=np.stack([v['hsv_tile_histogram'] for v in train.values()])
 mean,prec=normal_distribution(train)
 private_model=ROOT/'frozen_spatial_normal_only_784x32_covariance_20261025.npz'
 assert not private_model.exists()
 with private_model.open('xb') as f:np.savez_compressed(f,mean=mean,precision=prec,global_bank=global_bank,hsv_bank=hsv_bank)
 source_models_sha=sha(private_model)
 results={}
 for idx,entry in enumerate(original['selected'],1):
  ident=hashlib.sha256(entry['source_image_path'].encode()).hexdigest()
  inp=vec(base['results'][ident]);x=scores(inp,global_bank,hsv_bank,mean,prec)
  results[ident]={'continuous_model_scores':x,'pixel_source_image_sha256':base['results'][ident]['original_pixel_sha256']}
  if idx%25==0:print('SPATIAL_ORIGINAL_SOURCE_SCORE_FROZEN',idx,'/134','TEST_TRUTH_UNSEEN',True,flush=True)
 thresholds={method:float(np.quantile([results[x]['continuous_model_scores'][method] for x in grouped['validation/good']],.95,method='linear')) for method in MODES}
 assert len(results)==134 and len(thresholds)==len(MODES)
 source={'date':'2026-10-25','status':'complete','config':CFG,'source_media_selection_sha256':sha(MAN),'frozen_pixel_model_features_SHA256':sha(FEATURES),'frozen_normal_train_mean_cov_banks_SHA256':source_models_sha,'training_normal_only_source_count':24,'validation_normal_only_source_count':20,'all_image_scores_saved_before_gold_label_use':True,'normal_only_predeclared_thresholds':thresholds,'per_source_anonymous_image_score':results,'elapsed_seconds':round(time.monotonic()-t,2),'max_RSS_KB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 PRED.write_text(json.dumps(source,indent=2)+'\n')
 print('NEW_INDUSTRIAL_SCREWBAG_NORMAL_ONLY_ALL_134_PREDICTIONS_FROZEN',PRED,'THRESHOLDS',thresholds,'TIME_S',source['elapsed_seconds'],flush=True)

def ci(k,n):
 if not n:return None
 return [round(float(beta.ppf(.025,k,n-k+1)) if k else 0,4),round(float(beta.ppf(.975,k+1,n-k)) if k<n else 1,4)]

def AUC(positives,negatives):
 a=np.asarray(positives,dtype=float);b=np.asarray(negatives,dtype=float)
 return float(np.mean(a[:,None]>b[None])+0.5*np.mean(a[:,None]==b[None]))

def metric(positives,normal,threshold):
 tp=int(sum(v>threshold for v in positives));fn=int(len(positives)-tp)
 fp=int(sum(v>threshold for v in normal));tn=int(len(normal)-fp)
 return {'n':int(len(normal)+len(positives)),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2),'test_AUROC':AUC(positives,normal),'normal_only_calibration_threshold':threshold,'source_exact95_recall':ci(tp,tp+fn),'source_exact95_specificity':ci(tn,tn+fp)}

def evaluate():
 if SCORE.exists():print('PRIOR_ORIGINAL_SCREWBAG_LABEL_SCORES_PRESERVED');return
 fixed=json.loads(PRED.read_text());assert fixed['status']=='complete' and fixed['all_image_scores_saved_before_gold_label_use']
 source=json.loads(MAN.read_text());grouped=get_keys(source)
 assert fixed['source_media_selection_sha256']==sha(MAN)
 cohort={k:{name:[fixed['per_source_anonymous_image_score'][key]['continuous_model_scores'][name] for key in ids] for name in MODES} for k,ids in grouped.items() if k.startswith('test/')}
 outcomes={}
 rng=np.random.default_rng(20261025)
 for name in MODES:
  controls=np.asarray(cohort['test/good'][name])
  logical=np.asarray(cohort['test/logical_anomalies'][name])
  structural=np.asarray(cohort['test/structural_anomalies'][name])
  threshold=fixed['normal_only_predeclared_thresholds'][name]
  results={}
  for label,ps in [('logical_vs_clean',logical),('structural_vs_clean',structural),('all_anomalies_vs_clean',np.concatenate([logical,structural]))]:
   m=metric(ps,controls,threshold)
   intervals={'AUROC':[],'balanced_accuracy':[]}
   for _ in range(2500):
    p=ps[rng.integers(0,len(ps),size=len(ps))]
    n=controls[rng.integers(0,len(controls),size=len(controls))]
    vals=metric(p,n,threshold)
    intervals['AUROC'].append(vals['test_AUROC']);intervals['balanced_accuracy'].append(vals['balanced_accuracy'])
   m['original_source_image_stratified_bootstrap95']={key:[round(float(np.quantile(vals,.025)),4),round(float(np.quantile(vals,.975)),4)] for key,vals in intervals.items()}
   results[label]=m
  outcomes[name]=results
  print('NEW_SCREWBAG_HELDOUT_PRETRAINED_NORMAL_ONLY',name,'LOGICAL',results['logical_vs_clean'],'STRUCTURAL',results['structural_vs_clean'],'POOLED',results['all_anomalies_vs_clean'],flush=True)
 out={'date':'2026-10-25','source':'MVTec LOCO AD original screw_bag industrial photographs 24 train good / 20 val good / 30 test good / 30 test logical / 30 test structural','original_source_license':'CC BY-NC-SA 4.0, original copyright MVTec; no paid product reuse implied','source_SHA256':{'official_source_selection':sha(MAN),'actual_original_image_only_ResNet18_HSV_feature_receipts':sha(FEATURES),'normal_only_frozen_134_predictions_and_thresholds_before_gold':sha(PRED)},'source_original_n':{'train_normal':24,'validation_normal':20,'heldout_good':30,'heldout_logical':30,'heldout_structural':30,'new_source_photos_total':134},'source_labels_source':'Original MVTec authored normal/logical/structural test directories, NOT independent blind human rereview of images; test labels joined only after frozen model and normal-only cutoff persisted','source_original_group_n_of_independent_image_families':'Within same screw-bag category and fixed camera setup; no unique manufacturer holdout, no cross-domain generalization. Different dataset class from previously analyzed splicing connectors and pushpins.','config':CFG,'full_results':outcomes,'importance':'PaDiM-style Gaussian relative patch spatial features can be compared on real source logical and structural defects against same-budget untrained HSV and pretrained global ResNet; not official Padim result or proof of arbitrary photorealistic 3D correctness.','runtime':'Source feature and normal model CPU-only. No production GPU, 3D render, or other agent interrupted.','limitations':['Normal-only 24 train and 20 calibration points do not make stable commercial factory threshold; exact binomial false alarms on just 30 true normals are highly uncertain.','Logical/structural categories are author-labeled, not blind independent human acceptance ratings.','ResNet18 weights official ImageNet1K pretrained; training cost not equal to untrained HSV and possible MVTec source inclusion in pretrained corpus unknown.','PaDiM-style spatial empirical mean/cov with 32 selected channels and 0.01 regularization is a research proxy, not official PaDiM implementation/benchmark or license to claim official SOTA.','No comparison with deployed owner machine components, CAD identities, physically certified load, collision or final rendered photorealism.','Even a good conditional detection AUC is not equivalent to reliable accept/reject under actual low defect prevalence and unseen manufacturers.','Choosing best method after viewing this dataset consumes test source; require wholly untouched third-party mechanical product and independent exact engineering gold.']}
 SCORE.write_text(json.dumps(out,indent=2)+'\n')
 print('NEW_INDEPENDENT_MANUFACTURER_SOURCE_ARTIFACT_COMPLETED',SCORE,'METHODS',len(outcomes))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','evaluate']);a=ap.parse_args()
 if a.mode=='freeze':freeze()
 else:evaluate()
