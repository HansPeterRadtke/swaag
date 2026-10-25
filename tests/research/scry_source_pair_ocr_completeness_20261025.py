#!/usr/bin/env python3
"""Test text completeness from reference/implementation real mobile screenshots.

Human-reviewed Scry Design Diff Eval, 311 source pairs spanning 52 apps.
Select at most THREE distinct pairs per APP by SHA256(pair_id) (independent of
human issue tags). Freeze source OCR and simple visible-token loss/addition
signals BEFORE opening issue annotations. Tesseract 5.3.4 PSM11 CPU only.

Compare an independently checked content-based OCR hypothesis with the prior
paired-pixel SSIM/MAE baseline ON THE SAME selection, using app-separated
calibration and evaluation groups; prior model-selection history means these
are exploratory, NOT untouched super-test evidence. No-tagged is not truly
certified clean. OCR cannot detect pure font-size, style or partial clipping
when letters are hallucinated or still readable. No image content in Git.
"""
from __future__ import annotations
from pathlib import Path
import collections,hashlib,io,json,time,datetime,re,statistics,subprocess,tempfile,os
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.metrics import roc_auc_score

BASE=Path('/data/var/perceptual-independent-20261024/scry311')
PRED=BASE/'paired_ocr_psm11_20261025_frozen_predictions.json'
SCORES=BASE/'paired_ocr_psm11_20261025_human_issue_scores.json'
CFG={'name':'Tesseract PSM 11 source-visible token completeness from real paired mobile screens','tool_version':'5.3.4','psm':11,'max_image_width_px':640,'max_image_height_px':1152,'min_ocr_confidence_percent':40.0,'min_alpha_numeric_chars':2,'keep_repeated_tokens':False,'eligible_dataset':'Scrymore/scry-design-diff-eval 311 reference A vs implementation B pair screens','screen_sampling':'up to 3 smallest SHA256(id) among each app_slug group; independent of human issue tags','calibration_group':'SHA256(app_slug) numeric prefix modulo 4 == 0 (used in prior SSIM study); other apps are app-disjoint exploratory evaluation','positive_label':'human issue label includes Typography OR Missing Content, regardless of any other labels; not necessarily missing readable text','negative_label':'none of those two human issue tags; may contain other visual bugs, not certified clean','confidence_grid':[round(x/100,2) for x in range(0,91,5)],'training_selection':'Select among ref_token_missing_fraction, 1_minus_bag_Jaccard, 1_minus_text_difflib_ratio by calibration balanced accuracy; tie more conservative threshold','max_ocr_seconds_per_image':10,'cpu_threads':1,'no_training_on_eval_app_groups':True}


def nwords(record):
 # No labels are read by predictor; OCR gets ONLY A/B screenshot bytes.
 words=[]
 for line in record.splitlines()[1:]:
  parts=line.split('\t',11)
  if len(parts)!=12:continue
  try:confidence=float(parts[10])
  except Exception:continue
  if confidence<CFG['min_ocr_confidence_percent']:continue
  word=''.join(ch for ch in parts[11].casefold() if ch.isalnum())
  if len(word)>=CFG['min_alpha_numeric_chars']:words.append(word)
 return words


def tokenize(buf):
 start=time.perf_counter()
 with Image.open(io.BytesIO(buf)) as original:
  im=original.convert('RGB');size=list(im.size)
  scale=min(1.0,CFG['max_image_width_px']/size[0],CFG['max_image_height_px']/size[1])
  if scale<1:im=im.resize((max(1,int(size[0]*scale)),max(1,int(size[1]*scale))),Image.Resampling.LANCZOS)
  with tempfile.TemporaryDirectory(prefix='ocr-real-scry-',dir='/data/var/tmp') as root:
   tmp=Path(root)/'screenshot.png';im.save(tmp)
   env={**os.environ,'OMP_THREAD_LIMIT':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
   try:
    child=subprocess.run(['tesseract',str(tmp),'stdout','--psm',str(CFG['psm']),'tsv'],capture_output=True,text=True,timeout=CFG['max_ocr_seconds_per_image'],check=False,env=env)
    if child.returncode:raise ValueError(f'tesseract status {child.returncode} stderr {child.stderr[:120]}')
    words=nwords(child.stdout)
    return {'status':'completed','words':sorted(set(words)), 'recognized_token_count':len(words),'unique_token_count':len(set(words)),'raw_image_size':size,'ocr_image_size':list(im.size),'elapsed_seconds':round(time.perf_counter()-start,3)}
   except subprocess.TimeoutExpired:return {'status':'timed_out','elapsed_seconds':round(time.perf_counter()-start,3),'raw_image_size':size,'ocr_image_size':list(im.size)}


def text_scores(a,b):
 from difflib import SequenceMatcher
 A=set(a);B=set(b)
 missing=1-len(A&B)/max(1,len(A))
 jaccard=1-len(A&B)/max(1,len(A|B))
 seq=1-SequenceMatcher(None,sorted(A),sorted(B),autojunk=False).ratio()
 return {'ref_token_missing_fraction':round(missing,7),'1_minus_bag_Jaccard':round(jaccard,7),'1_minus_text_difflib_ratio':round(seq,7)}


def selected_items():
 files=sorted(BASE.glob('test-*-of-00016.parquet'))
 assert len(files)==16
 index=[]
 for f in files:
  for item in pq.read_table(f,columns=['id','app_slug']).to_pylist():index.append({'id':item['id'],'app_slug':item['app_slug'],'file':f.name})
 assert len(index)==311 and len({r['id'] for r in index})==311
 grouped=collections.defaultdict(list)
 for row in index:grouped[row['app_slug']].append(row)
 selected=[]
 for app in sorted(grouped):
  chosen=sorted(grouped[app],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:3]
  selected.extend(chosen)
 assert len(grouped)==52 and len(selected)>100
 return index,selected


def freeze():
 if PRED.exists():
  a=json.loads(PRED.read_text())
  if a['status']=='complete':print('RAW_OCR_ALREADY_FROZEN_NOT_REPEATED');return
 allrows,chosen=selected_items();selected={r['id'] for r in chosen}
 state=json.loads(PRED.read_text()) if PRED.is_file() else {'source':'Scrymore/scry-design-diff-eval', 'config':CFG,'selected_ids_sha256':hashlib.sha256('\n'.join(sorted(selected)).encode()).hexdigest(),'source_dataset_n':len(allrows),'selected_n':len(chosen),'app_groups':len({r['app_slug'] for r in chosen}),'status':'in_progress','frozen_pair_predictions':{},'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 assert state['config']==CFG and state['selected_ids_sha256']==hashlib.sha256('\n'.join(sorted(selected)).encode()).hexdigest()
 start=time.monotonic()
 for f in sorted(BASE.glob('test-*-of-00016.parquet')):
  arr=pq.read_table(f,columns=['id','app_slug','image_a','image_b'])
  for r in arr.to_pylist():
   if r['id'] not in selected or r['id'] in state['frozen_pair_predictions']:continue
   key=r['id'];a=r['image_a']['bytes'];b=r['image_b']['bytes']
   # Persist attempt BEFORE OCR; a timeout or ambiguous interruption is not retried.
   item={'id':key,'app_slug':r['app_slug'],'sha256_ref':hashlib.sha256(a).hexdigest(),'sha256_impl':hashlib.sha256(b).hexdigest(),'status':'pending_or_uncertain'}
   state['frozen_pair_predictions'][key]=item
   PRED.write_text(json.dumps(state,indent=2)+'\n')
   first=tokenize(a);second=tokenize(b)
   if first['status']=='completed' and second['status']=='completed':
    item.update(status='completed',source_a=first,implementation_b=second,features=text_scores(first['words'],second['words']))
   else:item.update(status='ocr_incomplete_or_timeout',source_a=first,implementation_b=second)
   PRED.write_text(json.dumps(state,indent=2)+'\n')
   print('OCR_PAIR',len(state['frozen_pair_predictions']),'/',len(selected),key,'app',r['app_slug'],'A',first['status'],first['unique_token_count'] if first['status']=='completed' else '-', 'B',second['status'],'TIME',round(first['elapsed_seconds']+second['elapsed_seconds'],2),flush=True)
  del arr
 assert len(state['frozen_pair_predictions'])==len(selected)
 state['status']='complete';state['runtime_iteration_seconds']=round(time.monotonic()-start,2)
 PRED.write_text(json.dumps(state,indent=2)+'\n')
 print('RAW_FREEZE_COMPLETE',len(state['frozen_pair_predictions']),'SEC',state['runtime_iteration_seconds'],flush=True)


def tally(rows,feature,threshold):
 tp=fp=tn=fn=0
 for r in rows:
  pred=r['features'][feature]>threshold
  truth=r['text_issue']
  if pred and truth:tp+=1
  elif pred and not truth:fp+=1
  elif not pred and truth:fn+=1
  else:tn+=1
 rec=tp/(tp+fn) if tp+fn else None
 spec=tn/(tn+fp) if tn+fp else None
 return {'n':len(rows),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':rec,'specificity_against_no_text_issues':spec,'precision':tp/(tp+fp) if tp+fp else None,'balanced_accuracy':(rec+spec)/2 if rec is not None and spec is not None else None}


def score():
 if SCORES.exists():print('SCORES_EXIST_NOT_RECOMPUTED');return
 prediction=json.loads(PRED.read_text());assert prediction['status']=='complete'
 assert prediction['config']==CFG
 labs={}
 for f in sorted(BASE.glob('test-*-of-00016.parquet')):
  for r in pq.read_table(f,columns=['id','n_issues','issue_labels','app_slug']).to_pylist():labs[r['id']]=r
 assert len(labs)==311
 valid=[]
 for k,r in prediction['frozen_pair_predictions'].items():
  if r['status']!='completed':continue
  lab=labs[k];assert r['app_slug']==lab['app_slug']
  text_issue=any(t in ('Typography','Missing Content') for t in lab['issue_labels'])
  fold='calibration' if int(hashlib.sha256(r['app_slug'].encode()).hexdigest()[:8],16)%4==0 else 'heldout_other_apps'
  valid.append({'id_sha256':hashlib.sha256(k.encode()).hexdigest(),'app_slug':r['app_slug'],'fold':fold,'text_issue':text_issue,'n_issues':lab['n_issues'],'issue_types':lab['issue_labels'],'features':r['features'],'ref_text_token_count':r['source_a']['unique_token_count'],'implementation_text_token_count':r['implementation_b']['unique_token_count']})
 cal=[r for r in valid if r['fold']=='calibration'];test=[r for r in valid if r['fold']=='heldout_other_apps']
 if len(cal)<15 or len(test)<50:raise ValueError('insufficient complete OCR evaluation')
 grid=[]
 for metric in ['ref_token_missing_fraction','1_minus_bag_Jaccard','1_minus_text_difflib_ratio']:
  for threshold in CFG['confidence_grid']:
   cm=tally(cal,metric,threshold)
   if cm['balanced_accuracy'] is not None:grid.append((cm['balanced_accuracy'],threshold,metric))
 if not grid:raise ValueError('could not fit calibration policy')
 best=max(grid,key=lambda a:(a[0],a[1],a[2]=='ref_token_missing_fraction'))
 scores={}
 for metric in ['ref_token_missing_fraction','1_minus_bag_Jaccard','1_minus_text_difflib_ratio']:
  threshold=max((p for p in grid if p[2]==metric),key=lambda z:(z[0],z[1]))[1]
  group={'calibration':tally(cal,metric,threshold),'heldout':tally(test,metric,threshold),'calibration_threshold':threshold}
  for field,rows in [('calibration',cal),('heldout',test)]:
   truth=np.array([r['text_issue'] for r in rows],dtype=int);vals=[r['features'][metric] for r in rows]
   group[field]['AUROC']=float(roc_auc_score(truth,vals)) if len(set(truth))>1 else None
  scores[metric]=group
 # Existing SSIM pixel-only evaluated on exactly the same selected pairs.
 ss=json.loads((BASE/'scry_cv_pixel_only_predictions.json').read_text())
 byid={r['id']:r for r in ss['predictions']};assert len(byid)==311
 comparisons={}
 for dataset,rows in [('calibration',cal),('heldout',test)]:
  lbl=np.array([r['text_issue'] for r in rows],dtype=int)
  seq=[ss['predictions'][0]['ssim_difference']] if not rows else [byid[labsid]['ssim_difference'] for labsid in [next(lid for lid in prediction['frozen_pair_predictions'] if hashlib.sha256(lid.encode()).hexdigest()==r['id_sha256']) for r in rows]]
  comparisons[dataset]={'Scry_SSIM_vs_human_text_tag_AUROC':float(roc_auc_score(lbl,seq)) if len(set(lbl))>1 else None,'n':len(rows),'label_text_n':int(sum(lbl)),'label_no_text_n':int(len(rows)-sum(lbl))}
 result={'source':'Scry 311 human-reviewed reference/implementation UI image pairs, preselected at most 3 per 52 apps independent from labels, OCR PSM11 frozen before labels','source_prediction_sha256':hashlib.sha256(PRED.read_bytes()).hexdigest(),'label_text_definition':CFG['positive_label'],'no_text_label_not_verified_good':True,'selected_source_pairs':prediction['selected_n'],'complete_ocr_pairs':len(valid),'skipped_incomplete_ocr_pairs':prediction['selected_n']-len(valid),'app_group_split':{'calibration_n':len(cal),'evaluation_n':len(test),'calibration_apps':len({r['app_slug'] for r in cal}),'evaluation_apps':len({r['app_slug'] for r in test})},'fixed_ocr_config':CFG,'calibration_selected':{'metric':best[2],'threshold':best[1],'calibration_bacc':best[0]},'per_metric_results':scores,'prior_frozen_SSIM_AUROC_on_same_screen_subset':comparisons,'limits':'Human missing-content is often a non-text icon/object and typography changes may preserve text. OCR word drops also reflect font, color, OCR failures, compression and content updates that were not separately adjudicated. Calibration app split was used in prior Scry analysis, so it is NOT a new untouched super-test; no-tagged status is not absence of all defects. One fixed OCR PSM, no retune using evaluation apps.'}
 SCORES.write_text(json.dumps(result,indent=2)+'\n')
 print('OCR_SOURCE_IMAGE_TEXT_EVALUATION',json.dumps({'samples':len(valid),'selected':result['calibration_selected'],'same_subset_SSIM':comparisons,'features':scores},indent=2))

if __name__=='__main__':freeze();score()
