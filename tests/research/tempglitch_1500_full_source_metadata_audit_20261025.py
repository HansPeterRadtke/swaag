#!/usr/bin/env python3
"""Source-wide TempGlitch metadata-only shortcut audit, NOT vision evaluation.

Original MIT HuggingFace asgaardlab/TempGlitch source (2026 arxiv 2605.21443)
contains five balanced in-engine video fault types, 750 original buggy and
750 category-matched normal Godot clips. The Stuck in Place type has two distinct
environments (TPS and Platformer) each with 75 IDs that overlap numerically.
Never pair by numeric suffix without environment! Preserve all 1500 source
files' size and path/source commit revision, no 60GB videos downloaded, no GPU.

Hypothesis: can a classifier using ONLY source encoded file size plus category
(and two-engine environment) predict original author injected-bug label? If yes,
file metadata is a potential construction confound; not accessible to VLM prompt
as such and does NOT imply VLM exploited it. Predeclare grouping, 80/20 split
by SHA256(category,environment,numeric_id) so category-matched source pairs are
in same fold; fit one L2 logistic with category-specific log-size slopes, test
untouched source pair IDs. No hyperparameter/threshold tuning or metadata
feature selection post-hoc; stats about published source, not model quality.

Upstream source originals and every complete 150-file folder list preserved
under /data/var/perceptual-independent-20261024/tempglitch-control-20261025.
No source media, thumbnails or model weights in Git. No new Calendar effects.
"""
from pathlib import Path
import urllib.request,urllib.parse,json,hashlib,collections,datetime,math,statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score
from scipy.stats import binomtest
ROOT=Path('/data/var/perceptual-independent-20261024/tempglitch-control-20261025')
OUT=ROOT/'full_1500_original_metadata_and_pair_integrity_20261025.json'
CATS=['Blinking','Frozen Animation','Shooting Error','Stuck in Place','Velocity Bug']
REPO='asgaardlab/TempGlitch'
CFG={'source':'asgaardlab/TempGlitch official HG release all 1500 original video paths + compressed file sizes, MIT dataset tag, NO video pixels','split':'SHA256 category|engine|numeric index modulo 5 ==0 reserved for evaluation; matched clean/bug same fold; no tuning or other dataset','model':'sklearn LogisticRegression(C=1,solver=lbfgs,max_iter=600,random_state=20261025,class_weight=balanced) on fixed category+environment interaction with natural-log encoded MP4 file bytes. Only first training portion fits coefficients.','positive_label':'author-engine-injected temporal glitch, NOT human rated unwanted natural playback freeze','original_pair_integrity':'Category-environment-number identity; two environments Stuck in Place both contain indexes 1–75 and are NOT cross-paired','random_seed':'sha256-category-environment-index-static-20261025','no_visual_or_model_inference':True,'no_media_download':True}

def api(path):
 u='https://huggingface.co/api/datasets/'+REPO+'/tree/main/'+urllib.parse.quote(path,safe='/')+'?limit=1000'
 with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'SWAAG-visual-research/1.0'}),timeout=21) as f:items=json.load(f)
 if len(items)!=150:raise RuntimeError('expected 150 original cases '+path)
 out={}
 for x in items:
  assert x['type']=='file' and x['path'].endswith('.mp4')
  name=x['path'].split('/')[-1]
  import re
  match=re.search(r'_(\d+)\.mp4$',name)
  if not match:raise RuntimeError('unrecognized original video filename')
  idx=int(match.group(1))
  if path.startswith('Stuck in Place/'):
   env='Platformer' if 'Platformer' in name else 'TPS' if '_TPS_' in name else None
   assert env
  else:env='Default'
  key=(env,idx)
  assert key not in out,('duplicate pair identity',path,key)
  out[key]={'file_path':x['path'],'bytes':int(x['size']),'original_git_blob_sha':x.get('oid'),'hf_original_lfs_sha256':(x.get('lfs') or {}).get('oid')}
 assert len(out)==150
 return out

def evaluate(rows):
 # One independent SOURCE scene index group in both classes; do not treat two
 # variants of same scene as independent source crossvalidation groups.
 labels=[];groups=[];features=[];names=sorted({(r['category'],r['environment']) for r in rows})
 for r in rows:
  cat=(r['category'],r['environment']);log=float(math.log(r['bytes']))
  vec=[]
  for name in names:
   indicator=float(cat==name)
   vec.extend((indicator, indicator*(log-16.0)))
  features.append(vec);labels.append(int(r['label_buggy']))
  digest=hashlib.sha256((r['category']+'|'+r['environment']+'|'+str(r['index'])).encode()).digest()
  groups.append(int.from_bytes(digest[:8],'big')%5)
 Y=np.asarray(labels,dtype=int);X=np.asarray(features,dtype=float);G=np.asarray(groups,dtype=int)
 train=G!=0;test=G==0
 assert len({int(y) for y in Y[train]})==2 and len({int(y) for y in Y[test]})==2
 assert sum(Y[test])==sum(1-Y[test])
 # Source encoded size only (plus category, environment one-hot) is available
 # from archive listing, not provided to VLM frame prompt. No CV/video model.
 classifier=LogisticRegression(C=1.0,solver='lbfgs',max_iter=600,random_state=20261025,class_weight='balanced')
 classifier.fit(X[train],Y[train]);pred=classifier.predict_proba(X[test])[:,1]
 hard=pred>=.5
 tn,fp,fn,tp=confusion_matrix(Y[test],hard).ravel()
 full={'n_total_video_files':len(rows),'n_total_source_pairs':len(rows)//2,'n_fit_source_clips':int(train.sum()),'n_heldout_source_clips':int(test.sum()),'n_fit_buggy':int(Y[train].sum()),'n_test_buggy':int(Y[test].sum()),'n_test_normal':int((1-Y[test]).sum()),'test_roc_auc':float(roc_auc_score(Y[test],pred)),'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn),'test_balanced_accuracy':float(balanced_accuracy_score(Y[test],hard)),'test_feature_only_model_logistic_fit_completed':True}
 # Source-pair clustered bootstrap on completely withheld source indexes,
 # including category clustering by retaining both class videos per group.
 bkeys=[(r['category'],r['environment'],r['index']) for i,r in enumerate(rows) if test[i]]
 unique=sorted(set(bkeys))
 ids=collections.defaultdict(list)
 for j,k in enumerate(bkeys):ids[k].append(j)
 yt=Y[test]
 rng=np.random.default_rng(20261025);auc=[];bacc=[]
 for _ in range(2500):
  picked=rng.integers(0,len(unique),len(unique))
  inds=[j for i in picked for j in ids[unique[i]]]
  a=yt[inds];p=pred[inds]
  if len(set(a))<2:continue
  auc.append(float(roc_auc_score(a,p)))
  bacc.append(float(balanced_accuracy_score(a,p>=.5)))
 full['pair_group_bootstrap_2500_source95pct_AUROC']=[round(float(np.quantile(auc,q)),4) for q in (.025,.975)]
 full['pair_group_bootstrap_2500_source95pct_balanced_accuracy']=[round(float(np.quantile(bacc,q)),4) for q in (.025,.975)]
 xcat={}
 for cat in CATS:
  index=np.array([rows[i]['category']==cat for i in range(len(rows))]) & test
  assert index.sum()>10
  y=Y[index];p=classifier.predict_proba(X[index])[:,1]
  b=(p>=.5);tn,fp,fn,tp=confusion_matrix(y,b).ravel()
  xcat[cat]={'test_n':len(y),'test_buggy':int(y.sum()),'ROC_AUC':float(roc_auc_score(y,p)),'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn),'balanced_accuracy':float(balanced_accuracy_score(y,b))}
 return full,xcat

def main():
 if OUT.exists():print('PREVIOUS_FULL_SOURCE_METADATA_AUDIT_PRESERVED_NOT_REPEATED');return
 ROOT.mkdir(mode=0o700,parents=True,exist_ok=True)
 with urllib.request.urlopen(urllib.request.Request('https://huggingface.co/api/datasets/'+REPO,headers={'User-Agent':'SWAAG-research/1.0'}),timeout=15) as f:meta=json.load(f)
 revision=meta['sha'];rows=[];summary={}
 for cat in CATS:
  bug=api(cat+'/Buggy'+(' ' if cat=='Stuck in Place' else ''))
  norm=api(cat+'/Normal')
  assert set(bug)==set(norm)
  assert len(bug)==150
  diffs=[];pairs=[]
  for env,id in sorted(bug):
   x=bug[(env,id)];y=norm[(env,id)]
   rows.append({'category':cat,'environment':env,'index':id,'label_buggy':True,'bytes':x['bytes'],'file_path':x['file_path'],'source_lfs_sha256':x['hf_original_lfs_sha256']})
   rows.append({'category':cat,'environment':env,'index':id,'label_buggy':False,'bytes':y['bytes'],'file_path':y['file_path'],'source_lfs_sha256':y['hf_original_lfs_sha256']})
   diffs.append(math.log(x['bytes']/y['bytes']));pairs.append((x['bytes'],y['bytes']))
  s=summary[cat]={'pairs':len(pairs),'buggy_size_median_MiB':round(statistics.median(x for x,y in pairs)/1048576,2),'normal_size_median_MiB':round(statistics.median(y for x,y in pairs)/1048576,2),'fraction_pairs_buggy_file_larger_than_clean':sum(x>y for x,y in pairs)/len(pairs),'median_log_buggy_clean_size_ratio':statistics.median(diffs),'fraction_pairs_abs_log_ratio_over_30pct':sum(abs(x)>.2624 for x in diffs)/len(diffs),'paired_sign_test_p_two_sided':float(binomtest(sum(x>y for x,y in pairs),n=len(pairs),p=.5).pvalue),'original_pair_env_counts':dict(collections.Counter(env for env,id in bug))}
  print('FULL_TEMPGITCH_AUTHOR_SOURCE_GROUP',cat,'N',len(pairs),'BUGGY_MEDIAN_MIB',s['buggy_size_median_MiB'],'CLEAN_MEDIAN_MIB',s['normal_size_median_MiB'],'BUG_LARGER_RATE',round(s['fraction_pairs_buggy_file_larger_than_clean'],3),'P',s['paired_sign_test_p_two_sided'],flush=True)
 assert len(rows)==1500
 held,byclass=evaluate(rows)
 report={'date':'2026-10-25','source':'TempGlitch original HuggingFace asgaardlab/TempGlitch, all actual 1500 Godot original videos source file entries, MIT release card','revision':revision,'original_paper':'https://arxiv.org/html/2605.21443','original_source_release_size_GB':60.3,'no_media_download_or_CPU_heavy_decoding':True,'config':CFG,'original_source_file_listing':rows,'category_source_pair_size_distributions':summary,'paired_source_crossvalidated_metadata_only_binary_result':held,'per_category_metadata_only_heldout':byclass,'limits':['File size is not part of tested VLM frame input; high metadata classification is evidence of benchmark shortcut potential, NOT evidence that VLMs used the shortcut.','Original video sizes can encode duration, bitrate, resolution, code/theme/visual texture and are NOT direct proof of perceptual temporal glitch.','Same Godot assets and two game environments, source group split within categories not generalizable outside engine or production.','Author-assigned in-engine glitch labels without external third-party perceptual acceptance raters per video. Clean and glitchy videos category-matched with numeric IDs, not necessarily identical camera sequences.','No trained video classifier or physical simulation was run on actual image pixels in this metadata-only source audit.','The source model paper evaluated the 12 VLMs on all 1500 original clips; source-reported accuracy numbers remain published, not our locally reproduced model results.']}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('FULL_1500_TEMPGITCH_PUBLIC_SOURCE_METADATA_EVIDENCE',OUT,'BYTES',OUT.stat().st_size,'HELDOU_FINAL',held,'BY_CATEGORY',byclass,flush=True)
if __name__=='__main__':main()
