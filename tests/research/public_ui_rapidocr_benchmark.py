#!/usr/bin/env python3
"""Frozen second OCR engine check: genuine UIJudge screenshots, model sees pixels only.

Run using the isolated numpy1 RapidOCR virtual environment. Keeps checkpointed
per-page predictions in var, not Git. Uses precisely the previously fixed
65-percent OCR / 4.5:1 region contrast thresholds, never optimized on test.
"""
from pathlib import Path
import os,json,time,datetime,sys
import numpy as np
from PIL import Image
ROOT=Path('/data/var/perceptual-independent-20261024/uijudge-v0.4.0')
SCRIPT=Path(__file__).resolve().parent/'public_ui_ocr_benchmark.py'
import importlib.util
spec=importlib.util.spec_from_file_location('independent_ui_metrics',SCRIPT)
utils=importlib.util.module_from_spec(spec);spec.loader.exec_module(utils)
from rapidocr_onnxruntime import RapidOCR
OUT=ROOT/'rapidocr_immutable_predictions.json'
CONFIG={'model':'RapidOCR ONNXRuntime 1.2.3','numpy':np.__version__,'onnxruntime_version':__import__('onnxruntime').__version__,'engine':'RapidOCR default text detection recognition models','min_text_chars':2,'flag_median_confidence_below':65,'flag_any_region_contrast_below_ratio':4.5,'minimum_words_for_confidence':2,'tuning':'none: previous Tesseract comparator exact thresholds, locked before predictions','inference':'CPU single-thread native ONNX'}

def classify(image_name,src,ocr):
 t=time.monotonic();p=Path(src['path'])
 outcome={'page_id':image_name,'sha256':src['sha256'],'status':'pending'}
 try:
  # Only image RGB pixels, no ground truth, filename, CSS, DOM or receipt.
  with Image.open(p) as image: pixels=np.asarray(image.convert('RGB'))
  rows,elapsed=ocr(pixels)
  confs=[];contrasts=[]
  for row in rows or []:
   quad,txt,score=row
   if not isinstance(txt,str) or len(txt.strip())<2:continue
   confs.append(float(score)*100)
   coords=np.asarray(quad,dtype=int);minx=max(0,int(coords[:,0].min()));maxx=min(pixels.shape[1],int(coords[:,0].max()));miny=max(0,int(coords[:,1].min()));maxy=min(pixels.shape[0],int(coords[:,1].max()))
   if maxx-minx>4 and maxy-miny>3:
    local=pixels[miny:maxy,minx:maxx].reshape((-1,3))
    intensity=local.astype(float)@np.array([.2126,.7152,.0722]);ix=np.argsort(intensity);n=max(1,len(ix)//10)
    contrasts.append(utils.contrast_ratio(np.median(local[ix[:n]],axis=0),np.median(local[ix[-n:]],axis=0)))
  median=float(np.median(confs)) if confs else None
  minimum=float(min(contrasts)) if contrasts else None
  outcome.update(status='ok',seconds=round(time.monotonic()-t,3),n_words=len(confs),median_word_confidence=round(median,3) if median is not None else None,min_word_region_contrast=round(minimum,3) if minimum is not None else None,flag_low_ocr_conf=bool(len(confs)>=2 and median<65) if median is not None else None,flag_any_low_contrast=bool(minimum<4.5) if minimum is not None else None)
 except Exception as e:outcome.update(status='error',seconds=round(time.monotonic()-t,3),error=type(e).__name__+': '+str(e)[:180])
 return outcome

def main():
 manifest=json.loads((ROOT/'real_ui_frozen_manifest.json').read_text());state=json.loads(OUT.read_text()) if OUT.exists() else {'config':CONFIG,'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'images':{},'scores':{}}
 ocr=RapidOCR()
 for i,k in enumerate(sorted(manifest['pages'])):
  if k in state['images']:continue
  data=classify(k,manifest['pages'][k],ocr)
  state['images'][k]=data
  OUT.write_text(json.dumps(state,indent=2)+'\n')
  print('PAGE',i+1,'/',len(manifest['pages']),data['status'],data['seconds'],flush=True)
 state['scores']=utils.score(state['images'],manifest['items']);OUT.write_text(json.dumps(state,indent=2)+'\n')
 for k,v in state['scores'].items():print('SCORE',k,'n',v['n_items'],v['counts'],'precision',v['precision'],'recall',v['recall'])
 print('TOTAL',len(state['images']),'SECONDS',round(sum(x.get('seconds',0) for x in state['images'].values()),3),flush=True)
if __name__=='__main__':main()
