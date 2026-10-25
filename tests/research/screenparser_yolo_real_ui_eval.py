#!/usr/bin/env python3
"""Independent YOLO11-L ScreenParser bounding-box feasibility on real web pages.

32 Firefox full target-centered screenshots from archived UIJudgeBench real sites,
independently verified target locator rect/gold. YOLO sees ONLY screenshot PNG.
Predict all 55 classes, NO source DOM/selector/content/defect label in inference.
Raw model outputs frozen BEFORE target-box comparison; checkpoints one file per
screen to avoid repeating work. CPU ONLY, imgsz=1280/conf=.10/iou=.10 as author
card. This is screen element localization NOT reliable clipping classifier.
"""
from __future__ import annotations
import json,hashlib,time,statistics,resource,datetime,collections
from pathlib import Path
import numpy as np
ROOT=Path('/data/var/perceptual-independent-20261024')
SOURCE=ROOT/'screenparser-fullpage-real/fullpage_manifest.json'
MODEL=ROOT/'models/ScreenParser-YOLO/model_manifest.json'
PRED=ROOT/'screenparser-fullpage-real/screenparser_yolo_32_frozen_predictions.json'
EVAL=ROOT/'screenparser-fullpage-real/screenparser_yolo_32_target_localization_eval.json'
CONFIG={'checkpoint':'docling-project/ScreenParser best.pt pinned f029e565f1206577402e43206454522075be3f72','runtime':'Ultralytics 8.3.221 YOLO11-L, PyTorch 2.5.1 CPU FP32','imgsz':1280,'confidence_min':0.10,'iou_nonmax_suppression':0.10,'max_det':1000,'device':'cpu','torch_threads':1,'evaluation_iou_thresholds':[0.10,0.30,0.50],'exclusion_generic_classes':['Screen','Window','Column/Browser','Navigation Bar'],'scope':'class-agnostic target IoU and center coverage; not classification of visual defect'}

def calculate_iou(box,target):
 a,b,c,d=box; x,y,z,w=target
 area1=max(0,c-a)*max(0,d-b);area2=max(0,z-x)*max(0,w-y)
 i=max(0,min(c,z)-max(a,x))*max(0,min(d,w)-max(b,y))
 return i/(area1+area2-i) if area1+area2-i>0 else 0.

def predict():
 if PRED.is_file():
  previous=json.loads(PRED.read_text())
  if previous['status']=='complete':print('MODEL_RAW_PREVIOUSLY_FROZEN_NOT_REPEATED');return
 src=json.loads(SOURCE.read_text());m=json.loads(MODEL.read_text())
 assert len(src['cases'])==32 and all(v['status']=='ok' for v in src['cases'].values())
 assert m['revision']=='f029e565f1206577402e43206454522075be3f72'
 raw=json.loads(PRED.read_text()) if PRED.exists() else {'config':CONFIG,'checkpoint_sha256':m['sha256'],'source_manifest_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'predictions':{},'status':'in_progress','created':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 assert raw['config']==CONFIG and raw['source_manifest_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
 import torch
 from ultralytics import YOLO
 torch.set_num_threads(1)
 start=time.monotonic()
 checkpoint=ROOT/'models/ScreenParser-YOLO/best.pt';assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==m['sha256']
 model=YOLO(str(checkpoint))
 print('UI_DETECTOR_LOADED',m['revision'],'N_CLASSES',len(model.names),flush=True)
 for i,(key,source) in enumerate(sorted(src['cases'].items())):
  if key in raw['predictions']:
   print('PRESERVE_ALREADY_EVALUATED',i+1,flush=True);continue
  import cv2
  path=Path(source['image_path']);inp=cv2.imread(str(path),cv2.IMREAD_COLOR)
  assert inp is not None and hashlib.sha256(path.read_bytes()).hexdigest()==source['image_sha256']
  t=time.monotonic()
  results=model.predict(source=inp,device='cpu',imgsz=1280,conf=.10,iou=.10,verbose=False,max_det=1000)
  assert len(results)==1
  result=results[0]
  boxes=result.boxes.xyxy.detach().cpu().numpy().tolist()
  confidences=result.boxes.conf.detach().cpu().numpy().tolist()
  classids=result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
  predictions=[{'label':str(model.names[cid]),'confidence':round(float(conf),6),'xyxy':[round(float(v),3) for v in bbox]} for bbox,conf,cid in zip(boxes,confidences,classids)]
  raw['predictions'][key]={'input_sha256':source['image_sha256'],'width':int(inp.shape[1]),'height':int(inp.shape[0]),'predictions':predictions,'count':len(predictions),'seconds':round(time.monotonic()-t,3)}
  PRED.write_text(json.dumps(raw,indent=2)+'\n')
  print('SCREEN',i+1,'/',len(src['cases']),'BOXES',len(predictions),'RUNTIME',raw['predictions'][key]['seconds'],'LABELS_TOP',list(collections.Counter(x['label'] for x in predictions).most_common(4)),flush=True)
 raw['status']='complete';raw['cpu_elapsed_seconds']=round(time.monotonic()-start,2);raw['rss_peak_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 PRED.write_text(json.dumps(raw,indent=2)+'\n')
 print('SOURCE_ONLY_PREDICTIONS_FROZEN',PRED,'TOTAL_S',raw['cpu_elapsed_seconds'],'PEAK_RSS_KB',raw['rss_peak_kb'])

def evaluate():
 if EVAL.exists():print('PREVIOUS_SCORES_PRESERVED_NO_RECOMPUTE');return
 source=json.loads(SOURCE.read_text())['cases'];raw=json.loads(PRED.read_text())
 assert raw['status']=='complete' and len(raw['predictions'])==len(source)==32
 rows=[]
 for key,truth in sorted(source.items()):
  box=truth['target_visible_rect'];p=raw['predictions'][key]
  preds=[x for x in p['predictions'] if x['label'] not in CONFIG['exclusion_generic_classes']]
  ious=[calculate_iou(x['xyxy'],box) for x in preds]
  if ious:
   chosen=preds[int(np.argmax(ious))];maxiou=max(ious)
  else:chosen=None;maxiou=0
  center=((box[0]+box[2])/2,(box[1]+box[3])/2)
  contained=[x for x in preds if x['xyxy'][0]<=center[0]<=x['xyxy'][2] and x['xyxy'][1]<=center[1]<=x['xyxy'][3]]
  rows.append({'source_item_SHA256':hashlib.sha256(key.encode()).hexdigest(),'source_img_sha256':p['input_sha256'],'independent_defect_gold':truth['independent_gold_violation'],'criterion':truth['criterion'],'split':truth['split'],'target_dom_tag':truth['target_dom_tag'],'target_visual_rect':box,'best_class_agnostic_iou':round(maxiou,5),'best_label':chosen['label'] if chosen else None,'best_confidence':chosen['confidence'] if chosen else None,'target_center_covered_by_nongeneric_box':bool(contained),'n_total_boxes':p['count'],'n_nongeneric_boxes':len(preds),'latency_s':p['seconds']})
 summary={}
 for title,part in [('all',rows),('clipped', [x for x in rows if x['criterion']=='redecheck:element-protrusion']),('viewport_protrusion',[x for x in rows if x['criterion']=='redecheck:viewport-protrusion']),('defective',[x for x in rows if x['independent_defect_gold']]),('control',[x for x in rows if not x['independent_defect_gold']])]:
  summary[title]={'n':len(part),'iou_hits':{str(t):sum(x['best_class_agnostic_iou']>=t for x in part) for t in CONFIG['evaluation_iou_thresholds']},'center_covered':sum(x['target_center_covered_by_nongeneric_box'] for x in part),'mean_best_iou':statistics.mean(x['best_class_agnostic_iou'] for x in part) if part else None,'median_boxes':statistics.median(x['n_total_boxes'] for x in part) if part else None}
 result={'dataset':'UIJudgeBench v0.4.0 real webpages rendered full-page view after centering the target; independent DOM-target geometry, 32 cases with 16 verified DOM defective and 16 paired controls','model':'docling-project/ScreenParser YOLO11-L 55 classes pinned v2, CPU','source_capture_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'raw_detector_output_sha256':hashlib.sha256(PRED.read_bytes()).hexdigest(),'config':CONFIG,'summary':summary,'cases':rows,'interpretation':'Only source-identified target element localization. This YOLO model emits box/class, NO text and NO defect quality verdict. Element boxes may be too broad or nested (a source target can include child labels), especially heavily clipped/out-of-window source elements. Without exhaustive source-labeled UI boxes there is NO parser precision or overall PageIoU. Stronger localization would improve support for OCR but cannot itself prove clipping/overlap/contrast. Reference bbox/selector NEVER used for prediction.','latency_mean_s':statistics.mean(x['latency_s'] for x in rows),'cpu_total_s':raw['cpu_elapsed_seconds'],'peak_rss_kb':raw['rss_peak_kb']}
 EVAL.write_text(json.dumps(result,indent=2)+'\n')
 print('SCREENPARSER_LOCALIZATION',json.dumps(summary,indent=2),'MEAN_LATENCY',result['latency_mean_s'],'TOTAL_S',result['cpu_total_s'])

if __name__=='__main__':predict();evaluate()
