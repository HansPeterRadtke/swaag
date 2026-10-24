#!/usr/bin/env python3
"""Screenshot-only detectors evaluated on independently labeled real-site UIJudge pages.

Predeclared, non-trained baselines: Tesseract confidence and local text-pixel
contrast. Inputs are PNG bytes ONLY. Ground truth never enters detectors.
Predictions persist before scoring. Snapshot v0.4.0, no original files changed.
"""
from __future__ import annotations
from pathlib import Path
import json, time, datetime, hashlib, resource, subprocess, os, math
import numpy as np
from PIL import Image
from collections import defaultdict
ROOT=Path('/data/var/perceptual-independent-20261024/uijudge-v0.4.0')
MANIFEST=ROOT/'real_ui_frozen_manifest.json';PRED=ROOT/'tesseract_immutable_predictions.json'
CONFIG={'tesseract_version':'5.3.4','tesseract_psm':11,'min_word_characters':2,'min_word_confidence':0,'flag_median_confidence_below':65,'flag_any_contrast_below_wcag_ratio':4.5,'minimum_words_for_confidence':2,'budget_per_page_seconds':9,'cpu_threads':1,'image_transform':'original pixels RGBA->RGB only','criterion_routing':'layout: median OCR confidence below 65 OR at least one low contrast OCR region; contrast: at least one low-contrast OCR region','parameter_tuning':'none; thresholds fixed before reading predictions'}

def contrast_ratio(fore,back):
 def linear(rgb):
  arr=np.asarray(rgb,dtype=float)/255
  return np.where(arr<=.04045,arr/12.92,((arr+.055)/1.055)**2.4)
 def l(rgb):return float(np.dot(linear(rgb),[.2126,.7152,.0722]))
 a,b=sorted((l(fore),l(back)))
 return (b+.05)/(a+.05)
def img_run(page,rec):
 start=time.monotonic();path=Path(rec['path'])
 result={'page_id':page,'sha256':rec['sha256'],'status':'pending'}
 try:
  with Image.open(path) as source:
   image=source.convert('RGB');im=np.asarray(image);h,w=im.shape[:2];result['size']=[w,h]
  # Tesseract TSV captures coordinate/text confidence; no gold or DOM read by predictor.
  child=subprocess.run(['tesseract',str(path),'stdout','--psm','11','tsv'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=CONFIG['budget_per_page_seconds'],check=False,env={**os.environ,'OMP_THREAD_LIMIT':'1'})
  result['returncode']=child.returncode
  if child.returncode!=0:raise RuntimeError('tesseract failed '+child.stderr[:130])
  words=[];low_contrast_ratios=[]
  lines=child.stdout.splitlines();columns=lines[0].split('\t')
  for line in lines[1:]:
   parts=line.split('\t',len(columns)-1)
   if len(parts)!=len(columns):continue
   d=dict(zip(columns,parts));text=d.get('text','').strip()
   if len(text)<CONFIG['min_word_characters']:continue
   try:conf=float(d['conf']);x=int(d['left']);y=int(d['top']);bw=int(d['width']);bh=int(d['height'])
   except Exception:continue
   if conf<CONFIG['min_word_confidence'] or bw<2 or bh<2:continue
   words.append((text,conf))
   # Image-only local foreground/background grayscale percentiles are NOT
   # guaranteed to isolate text, especially on images and tiny font sizes.
   x0=max(0,x);y0=max(0,y);x1=min(w,x+bw);y1=min(h,y+bh)
   if x1-x0>4 and y1-y0>3:
    crop=im[y0:y1,x0:x1,:].reshape((-1,3))
    # Select representative bright/dark pixels, do not use CSS colors or the label.
    lum=crop.astype(float)@np.array([.2126,.7152,.0722]);ix=np.argsort(lum)
    n=max(1,len(ix)//10)
    fore=np.median(crop[ix[:n]],axis=0);back=np.median(crop[ix[-n:]],axis=0)
    ratio=contrast_ratio(fore,back)
    low_contrast_ratios.append(ratio)
  med=float(np.median([w[1] for w in words])) if words else None
  minratio=float(min(low_contrast_ratios)) if low_contrast_ratios else None
  result.update(status='ok',n_words=len(words),median_word_confidence=round(med,2) if med is not None else None,
                min_word_region_contrast=round(minratio,3) if minratio is not None else None,
                flag_low_ocr_conf=bool(len(words)>=2 and med<CONFIG['flag_median_confidence_below']) if med is not None else None,
                flag_any_low_contrast=bool(minratio<CONFIG['flag_any_contrast_below_wcag_ratio']) if minratio is not None else None)
 except subprocess.TimeoutExpired:result.update(status='timed_out',error='9-second per-page OCR budget exceeded; never interpreted as clean')
 except Exception as e:result.update(status='error',error=type(e).__name__+': '+str(e)[:220])
 result['seconds']=round(time.monotonic()-start,3)
 return result

def score(pages,cases):
 # Aggregate by independent page. Do not imply 89 L1 items are 89 independent screens.
 out={}
 for split in ('dev','test'):
  for track in ('layout','contrast'):
   group=[r for r in cases if r['split']==split and (r['track']=='layout' if track=='layout' else r['criterion_code']=='wcag:1.4.3')]
   cm={'TP':0,'FP':0,'TN':0,'FN':0,'ABSTAIN':0};scored=[]
   for r in group:
    pred=pages.get(r['page_id'],{});bad=r['ground_truth']=='no'
    if pred.get('status')!='ok':cm['ABSTAIN']+=1;continue
    f=pred.get('flag_any_low_contrast')
    c=pred.get('flag_low_ocr_conf')
    if track=='contrast':flag=f
    else:flag=(f or c) if f is not None and c is not None else None
    if flag is None:cm['ABSTAIN']+=1;continue
    cm['TP' if flag and bad else 'FP' if flag and not bad else 'FN' if bad else 'TN']+=1
    scored.append({'id':r['item_id'],'page_id':r['page_id'],'criterion':r['criterion_code'],'true_violation':bad,'predicted_violation':bool(flag)})
   p=cm['TP']/(cm['TP']+cm['FP']) if cm['TP']+cm['FP'] else None
   rec=cm['TP']/(cm['TP']+cm['FN']) if cm['TP']+cm['FN'] else None
   out[split+'_'+track]={'counts':cm,'precision':p,'recall':rec,'n_items':len(group),'n_distinct_pages':len({r['page_id'] for r in group}),'scored':scored}
 return out

def main():
 m=json.loads(MANIFEST.read_text());cases=m['items'];source=m['pages']
 state=json.loads(PRED.read_text()) if PRED.exists() else {'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config':CONFIG,'images':{},'scores':{},'notes':'All predictions from pixels only. Posthoc labels added for scoring, not predictor.'}
 for i,page in enumerate(sorted(source)):
  if page in state['images']:continue
  # Must finish all pages before looking at aggregate scores. Report progress only.
  info=img_run(page,source[page]);state['images'][page]=info
  PRED.write_text(json.dumps(state,indent=2)+'\n')
  print('PAGE',i+1,'/',len(source),info['status'],info['seconds'],flush=True)
 state['scores']=score(state['images'],cases)
 PRED.write_text(json.dumps(state,indent=2)+'\n')
 for key,v in state['scores'].items():print('SCORE',key,v['n_items'],v['counts'],'precision',v['precision'],'recall',v['recall'])
 print('TOTAL',len(source),'OK',sum(x['status']=='ok' for x in state['images'].values()),'CPU_SECONDS',round(sum(x.get('seconds',0) for x in state['images'].values()),2))
if __name__=='__main__':main()
