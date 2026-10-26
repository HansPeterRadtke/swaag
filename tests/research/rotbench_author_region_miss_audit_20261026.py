#!/usr/bin/env python3
"""Post-hoc audit of earlier frozen UI pixel classifier using separate author bboxes.

Author's classification/localization test file provides matched modified files
and exact source-provided rectangles. It is NOT independent human gold.
Raw source/PNG/ROI detail stays outside Git. No threshold retuning.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import hashlib
import json
import re
import time

import numpy as np
from PIL import Image
import requests

DATA=Path('/data/var/perceptual-independent-20261024')
OLD=DATA/'rotbench-android-heldout-20261026'
NEW=DATA/'rotbench-unused-source-split-20261026'
ROOT=DATA/'rotbench-author-region-frozen-audit-20261026'
REV='25b6e8214c5be600f54dc0834eaf4597fa9192e6'
SRC='https://huggingface.co/datasets/ImDim/RotBench/resolve/'+REV+'/bug_classification_and_localization/test.json'
BASE_SHA='0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7'

def hash_bytes(b):return hashlib.sha256(b).hexdigest()
def hash_file(p):return hash_bytes(p.read_bytes())
def write_once(path, obj):
 if path.exists():raise RuntimeError('Frozen file already exists: '+str(path))
 temp=path.with_suffix('.writing')
 temp.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
 temp.replace(path)

def local_image(path):
 for root in (OLD/'media',NEW/'media'):
  x=root/path
  if x.is_file():return x
 raise FileNotFoundError(path)

def inventory():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'inventory.json').exists() or (ROOT/'classification_source.json').exists(): raise RuntimeError('No repeated source acquisition')
 assert hash_file(OLD/'source_test.json')==BASE_SHA
 r=requests.get(SRC,timeout=30);r.raise_for_status();blob=r.content
 data=json.loads(blob)
 assert len(data)==340
 classification={path:(j,row) for j,row in enumerate(data) for path in row['images'] if re.search(r'_(?:layout-clip|layout-overlap|layout-miss|direction-mismatch|state-loseinput)_[0-9]+\.png$',path)}
 det=json.loads((OLD/'source_test.json').read_text());assert len(det)==340
 selected=[]
 for i,row in enumerate(det):
  pairs=[(p,*classification[p]) for p in row['images'] if p in classification]
  if len(pairs)!=1:continue
  p,j,match=pairs[0]
  selected.append({'detect_id':i,'classify_id':j,'modified_png':p,'classify_images':match['images'],'detector_images':row['images']})
 if len(selected)<40:raise RuntimeError('Insufficient independently released exact image matches')
 (ROOT/'classification_source.json').write_bytes(blob)
 write_once(ROOT/'inventory.json',{'release_sha':REV,'detection_source_sha':BASE_SHA,'separate_classification_file_sha':hash_bytes(blob),'num_classification_test_rows':len(data),'exact_image_matches':selected,'all_earlier_predictions_frozen':True,'old_predict_sha':hash_file(OLD/'predictions.json'),'new_predict_sha':hash_file(NEW/'predictions.json')})
 print('AUTHOR_SOURCE_REGION_FROZEN',len(selected),'classify_sha',hash_bytes(blob),'inventory_sha',hash_file(ROOT/'inventory.json'))

def evaluate():
 if (ROOT/'evaluated.json').exists():raise RuntimeError('No relabel or rerun')
 inventory=json.loads((ROOT/'inventory.json').read_text());assert hash_file(ROOT/'classification_source.json')==inventory['separate_classification_file_sha']
 raw=json.loads((ROOT/'classification_source.json').read_text())
 old=json.loads((OLD/'scored.json').read_text());new=json.loads((NEW/'evaluated.json').read_text())
 oldpred={x['record_index']:x for x in old['selected_cases']};newpred={x['id']:x for x in new['table']}
 t=time.perf_counter();rows=[]
 for x in inventory['exact_image_matches']:
  idx=x['detect_id'];loc=x['modified_png'];case=raw[x['classify_id']]
  answer=case['conversations'][1]['value']
  m=re.search(r'\{\s*"type"\s*:\s*"([^"]+)"\s*,\s*"image"\s*:\s*"([^"]+)"\s*,\s*"bbox_2d"\s*:\s*(\[[^\]]+\])\s*\}',answer)
  if not m: raise ValueError('Gold bbox parser fail id '+str(idx))
  defect_type,orient=m.group(1),m.group(2);bb=json.loads(m.group(3))
  assert defect_type in loc and orient in loc and len(bb)==4
  if idx in oldpred:
   outcome=oldpred[idx]['predictions']['changed_area_0p05pct'];previous_truth=oldpred[idx]['gold_bug'];group=oldpred[idx]['group']
  else:
   outcome=newpred[idx]['predictions']['aligned']['localized'];previous_truth=newpred[idx]['external_author_bug'];group=newpred[idx]['type']
  assert previous_truth is True
  if outcome is None:
   rows.append({'detect_id':idx,'gold_author_type':defect_type,'original_result':'ABSTAIN','bbox_status':'source_reference_missing'});continue
  image=np.asarray(Image.open(local_image(loc)).convert('RGB'))
  from tests_source_reference_path import reference_path
  original=from_ref=reference_path(loc)
  ref=np.asarray(Image.open(local_image(original)).convert('RGB'))
  if image.shape!=ref.shape: raise RuntimeError('Source/version size conflict '+loc)
  mask=(np.max(np.abs(image.astype(np.int16)-ref.astype(np.int16)),axis=2)>=24)
  h,w=mask.shape
  left,top,right,bottom=[int(z) for z in bb]
  bbox_valid=0<=left<right<=w and 0<=top<bottom<=h
  inside=int(mask[top:bottom,left:right].sum()) if bbox_valid else None
  count=int(mask.sum())
  area=(right-left)*(bottom-top) if bbox_valid else None
  rows.append({'detect_id':idx,'gold_author_type':defect_type,'gold_orientation':orient,'gold_bbox_xyxy':bb,'bbox_valid_at_native_resolution':bbox_valid,'modified_image_sha256':hash_file(local_image(loc)),'reference_image_sha256':hash_file(local_image(original)), 'native_hw':[h,w],'detector_flag':outcome,'pixels24_changed_total':count,'pixels24_in_author_bbox':inside,'author_bbox_area_px':area,'fraction_of_detected_changes_inside_bbox':(inside/count if bbox_valid and count else None),'changed_fraction_screen':count/(w*h), 'gold_author_bug':True,'prediction_source':'previous_100' if idx in oldpred else 'unused_240','description':'author synthetically changed same actual screenshot, separate released author bbox; not human quality certification'})
 summary={'source':'RotBench pinned classification/localization/test.json and separately pinned detection/test.json','dataset_revision':REV,'classification_source_sha':inventory['separate_classification_file_sha'],'original_previous_predictions_shas':[inventory['old_predict_sha'],inventory['new_predict_sha']],'matching_modified_png_paths':len(rows),'valid_original_pixels':sum('detector_flag' in z for z in rows),'abstain_missing_png_reference':sum(z.get('original_result')=='ABSTAIN' for z in rows),'old_previous_misses_with_author_bbox':sum(z.get('detector_flag') is False and z['prediction_source']=='previous_100' for z in rows if 'prediction_source' in z),'new_previous_misses_with_author_bbox':sum(z.get('detector_flag') is False and z['prediction_source']=='unused_240' for z in rows if 'prediction_source' in z),'bbox_invalid':sum(z.get('bbox_valid_at_native_resolution') is False for z in rows),'by_type':{},'cost':{'CPU_seconds_to_analyze_native_delta_inside_original_annotations':None,'GPU_used':False},'caveat':'bbox is an author-generated synthetic mutation target region, not independent human visible severity or actual runtime clipping truth', 'rows':rows}
 by=defaultdict(list)
 for z in rows:by[z['gold_author_type']].append(z)
 for k,v in by.items():summary['by_type'][k]={'n':len(v),'misses':sum(z.get('detector_flag') is False for z in v),'detected':sum(z.get('detector_flag') is True for z in v),'abstain':sum(z.get('detector_flag') is None for z in v)}
 summary['cost']['CPU_seconds_to_analyze_native_delta_inside_original_annotations']=round(time.perf_counter()-t,3)
 write_once(ROOT/'evaluated.json',summary)
 print('GOLD_REGION_SCORED',json.dumps({k:v for k,v in summary.items() if k not in ('rows','source')},sort_keys=True)[:2000], 'sha',hash_file(ROOT/'evaluated.json'))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('inventory','evaluate'));globals()[ap.parse_args().stage]()
