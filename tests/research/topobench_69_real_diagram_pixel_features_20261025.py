#!/usr/bin/env python3
"""Source-BLIND OCR and line geometry extraction from 69 real TopoBench diagrams.

External original: WayneGuo0011/TopoBench-180 pinned Hugging Face revision
'df366de6a2c8aaf83c0a95ba8bfb1a28e231cb14'. Complete 162 source graphs,
only 69 with legally redistributed original images (54 AI2D CC BY-SA, 15
explicitly licensed network diagrams). The source embeds directed graph gold,
but this module reads ONLY original image bytes/sample ID: zero graph nodes,
annotations, question answers or target edge labels used for predictions.

All 69 natural-source diagrams, not previous SADU or synthetic fixtures. Freeze
actual image-derived OCR word boxes plus Canny+Hough line segments BEFORE source
labels can be opened in scoring. Two fixed independent OCR PSM settings (6,11)
are compared but never dynamically selected based on human/source annotations.
One CPU thread at nice19; no local VLM or GPU, no user production renders.
Source licensed pixels and word predictions retained outside project Git.

This is exploratory on an externally curated original test-only dataset. Human
original source graph annotation is not necessarily exhaustive visual ground
truth, and numerical PDF or code verification is not substitute for pixels.
"""
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
import numpy as np,cv2
import subprocess,tempfile,json,hashlib,io,os,time,datetime,collections,resource,math
VAR=Path('/data/var/perceptual-independent-20261024/topobench-independent-20261025')
SOURCE=VAR/'official_topobench_162_test_original.parquet'
OUT=VAR/'original_69_raster_OCR_line_geometry_frozen_20261025.json'
CFG={'source':'WayneGuo0011/TopoBench-180 actual 69 redistributed test images in 162 test-only source rows; 54 original AI2D, 15 source-licensed public network diagrams','dataset_revision':'df366de6a2c8aaf83c0a95ba8bfb1a28e231cb14','image_only_features_before_gold':True,'source_image_count':69,'preprocessed_max_longside_px':1400,'ocr':'Tesseract 5.3.4 native default English, TSV, two separately frozen sparse PSM11 and single-block PSM6, conf>=30 for later mapping; no exact source node labels given to OCR','psm_modes':[11,6],'tesseract_per_image_timeout_seconds':13,'opencv':'Canny(gray,80,180); HoughLinesP rho1 theta1deg threshold=max(12,0.015*diagonal), minLineLength=max(12,.024diag), maxLineGap=max(4,.013diag)','max_hough_segments_to_store':2000,'source_image_license':'AI2D CC BY-SA 4.0, 15 network images have individual evidenced redistribution license; original media not published to project Git','CPU_threads':1,'anticipated_timeout_handling':'pending_or_uncertain never retried; fail/timeout not considered clean','no_model_inference_or_gold_labels':True}

def ocr(p,psm):
 t=time.monotonic()
 env={**os.environ,'OMP_THREAD_LIMIT':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
 cmd=['tesseract',str(p),'stdout','--psm',str(psm),'tsv']
 try:
  ret=subprocess.run(cmd,capture_output=True,text=True,timeout=CFG['tesseract_per_image_timeout_seconds'],env=env)
  if ret.returncode:raise RuntimeError('Tesseract exit '+str(ret.returncode))
  import csv
  parsed=csv.DictReader(io.StringIO(ret.stdout),delimiter='\t')
  rows=[]
  for r in parsed:
   if r.get('level')!='5' or not r.get('text','').strip():continue
   try:
    conf=float(r['conf']);x=int(r['left']);y=int(r['top']);w=int(r['width']);h=int(r['height'])
   except (ValueError,TypeError):continue
   if w<=0 or h<=0:continue
   rows.append({'text':r['text'][:110],'x':x,'y':y,'w':w,'h':h,'conf':conf,'page':r['page_num'],'block':r['block_num'],'paragraph':r['par_num'],'line':r['line_num']})
  return {'status':'complete','words':rows,'elapsed_seconds':round(time.monotonic()-t,2)}
 except subprocess.TimeoutExpired:
  return {'status':'timeout_unknown_no_retry','words':[],'elapsed_seconds':round(time.monotonic()-t,2)}
 except Exception as e:
  return {'status':'error_unknown_no_retry','words':[],'error_type':type(e).__name__,'elapsed_seconds':round(time.monotonic()-t,2)}

def features(blob):
 start=time.monotonic()
 with Image.open(io.BytesIO(blob)) as src:
  img=src.convert('RGB');original_size=list(img.size)
  scale=min(1.0,CFG['preprocessed_max_longside_px']/max(img.size))
  if scale<1:img=img.resize((max(1,round(img.width*scale)),max(1,round(img.height*scale))),Image.Resampling.LANCZOS)
  rgb=np.asarray(img);height,width=rgb.shape[:2];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  cv2.setNumThreads(1)
  edge=cv2.Canny(gray,80,180);diag=math.hypot(width,height)
  threshold=max(12,round(diag*.015));minlength=max(12,round(diag*.024));maxgap=max(4,round(diag*.013))
  lines=cv2.HoughLinesP(edge,1,np.pi/180,threshold=threshold,minLineLength=minlength,maxLineGap=maxgap)
  if lines is None:linelist=[]
  else:
   linelist=[[int(v) for v in x] for x in lines[:,0].tolist()]
   linelist.sort(key=lambda x:-math.hypot(x[2]-x[0],x[3]-x[1]))
  total_lines=len(linelist)
  linelist=linelist[:CFG['max_hough_segments_to_store']]
  with tempfile.TemporaryDirectory(prefix='topobench-visual-',dir='/data/var/tmp') as td:
   tmp=Path(td)/'source.png';img.save(tmp)
   words={str(psm):ocr(tmp,psm) for psm in CFG['psm_modes']}
  return {'status':'complete','source_image_sha256':hashlib.sha256(blob).hexdigest(),'original_size_wh':original_size,'working_size_wh':[width,height],'working_scale':scale,'edge_pixels_fraction':round(float(np.mean(edge>0)),6),'hough_detected_count_before_cap':total_lines,'hough_cap_reached':total_lines>CFG['max_hough_segments_to_store'],'hough_line_segments':linelist,'raw_tesseract_ocr':words,'pixel_feature_elapsed_seconds':round(time.monotonic()-start,2)}

def run():
 if OUT.exists():
  old=json.loads(OUT.read_text())
  if old['status']=='complete':print('TOPOBENCH_ALL_69_ALREADY_FROZEN_NOT_REPLAYED');return
 else:old={'created':'2026-10-25','config':CFG,'source_parquet_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'status':'in_progress','source_original_69_results':{}}
 assert old['config']==CFG and old['source_parquet_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
 tbl=pq.read_table(SOURCE,columns=['sample_id','image','has_image'])
 rows=tbl.to_pylist();assert len(rows)==162 and sum(x['has_image'] for x in rows)==69
 t0=time.monotonic();k=0
 for row in rows:
  if not row['has_image']:continue
  k+=1;ident=row['sample_id']
  if ident in old['source_original_69_results']:continue
  old['source_original_69_results'][ident]={'status':'pending_or_uncertain'}
  OUT.write_text(json.dumps(old,indent=2)+'\n')
  try:res=features(row['image']['bytes'])
  except Exception as e:res={'status':'error_no_retry','error_type':type(e).__name__,'exception_preview':str(e)[:150]}
  old['source_original_69_results'][ident]=res
  OUT.write_text(json.dumps(old,indent=2)+'\n')
  print('REAL_TOPOBENCH_RASTER_ONLY',k,'/69',ident,'STATUS',res['status'],'OCR11',len(res.get('raw_tesseract_ocr',{}).get('11',{}).get('words',[])),'HOUGH',res.get('hough_detected_count_before_cap'),'SEC',res.get('pixel_feature_elapsed_seconds'),flush=True)
 assert len(old['source_original_69_results'])==69
 old['status']='complete';old['this_iteration_seconds']=round(time.monotonic()-t0,2);old['peak_RSS_KB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 OUT.write_text(json.dumps(old,indent=2)+'\n')
 print('ALL_69_REAL_SOURCE_DIAGRAM_PIXEL_FEATURES_FROZEN',OUT,'SECONDS',old['this_iteration_seconds'],'OCR_STATUSES',dict(collections.Counter(v['status'] for v in old['source_original_69_results'].values())),flush=True)
if __name__=='__main__':run()
