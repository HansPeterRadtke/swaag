#!/usr/bin/env python3
"""Evaluate REAL human-judged visual design errors using PIXELS ONLY.

SlideAudit UIST'25 2400 slide images with 3 independent human raters; selected
343 original/synthesized actual slide raster images by independent human label
positive/control matching fixed BEFORE classifier predictions. Only source
JPEG/PNG pixel and file identity reach detector; original human categories,
positive/negative and bounding boxes never passed until raw feature checkpoint
has completed. The source dataset is not application GUIs and is partially
synthetically altered. Do NOT claim success on GUI product rendering from it.

Frozen cheap candidate checks:
 (1) Text color contrast inferred from inside Tesseract word glyph boxes,
     not true WCAG text CSS! Positive if at least 35% word boxes inferred
     contrast ratio < 4.5. Fonts/background images confound.
 (2) Clipping: any recognized word near 2% outer canvas border. Border pixels
     can be an intentional design choice; clipped content may be internal.
 (3) Occluded/overlap text: any two recognized word boxes overlap IoU >=0.15.
     Overlap of shapes not necessarily visible as overlapping OCR words.
Image complexity, text confidences and edge gradients also frozen for AUROC
comparison; no score threshold tuning on source labels in this experiment.

One CPU thread, Tesseract 5.3.4 PSM11 with 10s per slide upper limit. Store
raw OCR text and pixel diagnostics exclusively in /data/var. No GPUs, no user
product renders, original slide images are not edited or added to project Git.
"""
from pathlib import Path
import hashlib,json,datetime,time,os,subprocess,collections,statistics,io,csv,math,resource
from PIL import Image
import numpy as np
import cv2
ROOT=Path('/data/var/perceptual-independent-20261024/slideaudit-public/data/images')
SRC=Path('/data/var/perceptual-independent-20261024/slideaudit_selection_20261025.json')
VAR=Path('/data/var/perceptual-independent-20261024')
RAW=VAR/'slideaudit_343_image_only_ocr_cv_frozen_20261025.json'
CFG={'date':'2026-10-25','source':'SlideAudit UIST25 original 343 human strongly annotated actual/synthetically altered slide images','source_repository_commit':'642d490b7c1d2e78a50a631bfd359433397f3ecf','num_images_expected':343,'tesseract':'Tesseract 5.3.4 PSM 11 TSV','max_image_long_edge_px':1600,'ocr_timeout_seconds':12,'tesseract_cpu_threads':1,'word_min_confidence_for_visual_bbox':25.0,'near_canvas_border_fraction_of_min_dimension':0.02,'low_word_contrast_wcag_like_ratio_threshold':4.5,'slide_low_contrast_fraction_trigger':0.35,'word_overlap_min_iou':0.15,'proxy_contrast':'quantile grayscale relative-luminance contrast between 10th and 90th percentiles INSIDE OCR word rectangle; not glyph/background segmentation or WCAG conformance','Canny':'cv2 Canny thresholds 60,140, image grayscaled, border 2% min dimension','scope':'image-only binary baseline for human-reported design issue categories; not model trained for semantic correctness; no labels in predictor'}

def iou(a,b):
 x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3]);area=max(0,x1-x0)*max(0,y1-y0)
 denom=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
 return area/denom if denom>0 else 0.0

def contrast(sub):
 if not sub.size:return None
 # sRGB grayscale linear luminance approximation. Contrast cannot be judged
 # without actual foreground/background separation, this is an image proxy.
 a=np.percentile(sub,[10,90])/255.
 def lin(v):return v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4
 x,y=lin(float(a[0])),lin(float(a[1]))
 return float((max(x,y)+.05)/(min(x,y)+.05))

def estimate(path):
 t=time.monotonic()
 with Image.open(path) as im:
  orig=list(im.size);w,h=im.size;scale=min(1.0,CFG['max_image_long_edge_px']/max(w,h))
  if scale<1:im=im.resize((max(1,int(w*scale)),max(1,int(h*scale))),Image.Resampling.LANCZOS)
  arr=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
 H,W=arr.shape[:2]
 gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
 edge=cv2.Canny(gray,60,140)
 band=max(2,round(min(W,H)*CFG['near_canvas_border_fraction_of_min_dimension']))
 border=np.zeros((H,W),dtype=bool);border[:band,:]=True;border[-band:,:]=True;border[:,:band]=True;border[:,-band:]=True
 edges_border=float(np.mean(edge[border]>0));edges_middle=float(np.mean(edge[~border]>0))
 # Generic image-only mask for content reaching absolute physical slide border.
 screen_means=[float(gray[band:-band,band:-band].mean()),float(gray[border].mean())]
 env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMP_THREAD_LIMIT':'1'}
 try:
  ocr=subprocess.run(['tesseract',str(path),'stdout','--psm','11','tsv'],capture_output=True,text=True,timeout=CFG['ocr_timeout_seconds'],env=env)
  if ocr.returncode:raise RuntimeError('OCR return nonzero '+str(ocr.returncode)+': '+ocr.stderr[-100:])
  parsed=csv.DictReader(io.StringIO(ocr.stdout),delimiter='\t');words=[]
  for field in parsed:
   if field.get('level')!='5':continue
   try:conf=float(field.get('conf') or -1);x=float(field.get('left'));y=float(field.get('top'));ww=float(field.get('width'));hh=float(field.get('height'))
   except (ValueError,TypeError):continue
   if not field.get('text','').strip() or ww<=0 or hh<=0:continue
   # Tesseract works at ORIGINAL pixel size; map bboxes to resized image.
   x0=max(0,int(x*W/orig[0]));y0=max(0,int(y*H/orig[1]));x1=min(W,int((x+ww)*W/orig[0])+1);y1=min(H,int((y+hh)*H/orig[1])+1)
   if x1<=x0 or y1<=y0:continue
   words.append({'confidence':round(conf,3),'box':[x0,y0,x1,y1],'text':field['text'][:100]})
  status='complete'
 except subprocess.TimeoutExpired:words=[];status='ocr_timeout_no_retry'
 except Exception as e:words=[];status='ocr_error_no_retry:'+type(e).__name__
 eligible=[w for w in words if w['confidence']>=CFG['word_min_confidence_for_visual_bbox']]
 contrast_words=[]
 for w in eligible:
  x0,y0,x1,y1=w['box'];c=contrast(gray[y0:y1,x0:x1])
  if c is not None:contrast_words.append(c)
 near=[w for w in eligible if min(w['box'][0],w['box'][1],W-w['box'][2],H-w['box'][3])<=band]
 overlap_pairs=[]
 for i,a in enumerate(eligible):
  for j,b in enumerate(eligible[i+1:],i+1):
   v=iou(a['box'],b['box'])
   if v>=CFG['word_overlap_min_iou']:overlap_pairs.append((v,i,j))
 max_overlap=max((r[0] for r in overlap_pairs),default=0)
 results={'status':status,'original_dimensions_wh':orig,'evaluated_dimensions_wh':[W,H],'source_image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ocr_raw_word_count':len(words),'ocr_word_count_conf25':len(eligible),'mean_ocr_word_confidence':statistics.mean(w['confidence'] for w in eligible) if eligible else None,'ocr_low_conf_fraction':sum(w['confidence']<60 for w in words)/len(words) if words else None,'ocr_conf25_near_border_word_count':len(near),'ocr_conf25_near_border_fraction':len(near)/len(eligible) if eligible else 0,'ocr_conf25_overlapping_word_pairs':len(overlap_pairs),'ocr_conf25_max_word_iou':round(max_overlap,6),'ocr_conf25_median_word_height':statistics.median(w['box'][3]-w['box'][1] for w in eligible) if eligible else None,'word_contrast_values_sorted':[round(x,4) for x in sorted(contrast_words)],'minimum_estimated_word_contrast':min(contrast_words) if contrast_words else None,'p25_estimated_word_contrast':float(np.percentile(contrast_words,25)) if contrast_words else None,'estimated_fraction_words_under_4p5':sum(c<4.5 for c in contrast_words)/len(contrast_words) if contrast_words else None,'estimated_fraction_words_under_3':sum(c<3 for c in contrast_words)/len(contrast_words) if contrast_words else None,'edge_density_on_canvas_border':round(edges_border,7),'edge_density_inside':round(edges_middle,7),'edge_density_border_minus_interior':round(edges_border-edges_middle,7),'canvas_luma_interior':screen_means[0],'canvas_luma_border':screen_means[1],'predicted_low_contrast':(sum(c<4.5 for c in contrast_words)/len(contrast_words)>=.35) if contrast_words else False,'predicted_clipped_text_near_canvas_boundary':len(near)>=1,'predicted_ocr_box_overlap':len(overlap_pairs)>=1,'predicted_candidate_lowcontrast_boxes':[w['box'] for w in sorted(eligible,key=lambda w:contrast(gray[w['box'][1]:w['box'][3],w['box'][0]:w['box'][2]]) or 99)[:5]],'predicted_near_boundary_boxes':[w['box'] for w in near[:10]],'predicted_overlapping_word_boxes':[[eligible[i]['box'],eligible[j]['box']] for v,i,j in sorted(overlap_pairs,reverse=True)[:10]],'elapsed_s':round(time.monotonic()-t,3)}
 return results


def main():
 selection=json.loads(SRC.read_text())
 expected=set(selection['image_source_paths_expected'])
 available={str(Path('data/images')/p.name) for p in ROOT.glob('slide_*.png')}
 assert len(expected)==len(available)==CFG['num_images_expected'] and expected==available
 if RAW.exists():
  existing=json.loads(RAW.read_text())
  if existing['status']=='complete':print('COMPLETED_PREDICTIONS_PRESERVED_NOT_REPLAYED');return
 else:existing={'config':CFG,'source_selection_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'source_selection_count':len(expected),'source_git_commit':selection['official_source_commit'],'status':'in_progress','input_predictions':{},'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 assert existing['config']==CFG and existing['source_selection_sha256']==hashlib.sha256(SRC.read_bytes()).hexdigest()
 t0=time.monotonic();cv2.setNumThreads(1)
 for i,path in enumerate(sorted(ROOT.glob('slide_*.png')),1):
  key=path.stem
  if key in existing['input_predictions']:continue
  existing['input_predictions'][key]={'status':'pending_or_uncertain'}
  RAW.write_text(json.dumps(existing,indent=2)+'\n')
  try:existing['input_predictions'][key]=estimate(path)
  except Exception as e:existing['input_predictions'][key]={'status':'unhandled_error_no_retry','error':type(e).__name__+': '+str(e)[:200]}
  RAW.write_text(json.dumps(existing,indent=2)+'\n')
  r=existing['input_predictions'][key]
  if i%7==0 or r['status']!='complete':print('SLIDE_IMG',i,'/',len(expected),key,r['status'],'OCR_WORDS',r.get('ocr_word_count_conf25'),'SECONDS',r.get('elapsed_s'),flush=True)
 assert len(existing['input_predictions'])==CFG['num_images_expected']
 existing['status']='complete';existing['current_iteration_wall_s']=round(time.monotonic()-t0,2)
 existing['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 RAW.write_text(json.dumps(existing,indent=2)+'\n')
 print('SLIDEAUDIT_343_FROZEN_IMAGE_ONLY',len(existing['input_predictions']),'S',existing['current_iteration_wall_s'],'STATUSES',dict(collections.Counter(r['status'] for r in existing['input_predictions'].values())))
if __name__=='__main__':main()
