#!/usr/bin/env python3
"""Audit real, author-approved Android GUI states versus wrong SAME-APP references.

Stage freeze picks identity-valid original screens and invalid alternative clean
states without reading pixel differences. Stage pixels records image-only rules
with identical budgets. Stage evaluate joins source-author clean labels and
requires source-version/activity binding to prevent a false auto-fail.

Third-party media and all raw receipts remain isolated under /data/var.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import numpy as np
import cv2
from PIL import Image

BASE=Path('/data/var/perceptual-independent-20261024')
OLD=BASE/'rotbench-android-heldout-20261026'
NEW=BASE/'rotbench-unused-source-split-20261026'
OUT=BASE/'rotbench-same-app-clean-state-negative-20261026'
SOURCE_SHA='0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7'
REV='25b6e8214c5be600f54dc0834eaf4597fa9192e6'
DESIGN={
 'date':'2026-10-26',
 'frozen_source':'ImDim/RotBench bug_detection/test.json, pinned revision '+REV,
 'source_sha256':SOURCE_SHA,
 'source_authored_truth':'original RIGHT screen pairs only (author manually selected clean before mutation); not independent human validation of WCAG or all defects',
 'population':'all 170 original source RIGHT cases, two orientations each; eligible if same APP has another distinct original RIGHT activity with same native dimensions and orientation',
 'selection':'for each eligible original right case, select up to 3 distinct other activities by deterministic SHA256 ordering. Both original orientation screenshots compared to the corresponding other activity; no pixels or anomaly scores used to choose alternates',
 'source_version':'all images must exist in previously frozen original benchmark image receipts under same pinned source SHA, all local sha256 must match; no new fetch and no substitution of missing sources',
 'interpretation_1_valid_GUIs':'both images are author-approved original correct GUI states, though different activity contents are expected; an alarm about GUI defect due solely to pixel change is an invalid inference',
 'interpretation_2_immutable_screenshot':'only the EXACT same app+activity+viewport+source SHA reference qualifies for difference verdict; mismatched reference means REVIEW, not FAIL or PASS',
 'candidate':'maxRGB >=24 changed-pixel area >0.0005 (same threshold as earlier rounds)',
 'same_pixels_baseline':'global RGB mean absolute difference >1.5 (same threshold as earlier rounds)',
 'extra_upper_bound':'any changed pixel (not a semantic quality judge)',
 'null':'always no defect (not capable of defect recall)',
 'negative_caveat':'These are legitimate source GUI states but not naturally varying same-activity updates, cannot prove dynamic-state tolerance or arbitrary clean GUI suitability',
 'exact_control':'each clean captured PNG compared to itself; zero false alarms expected mathematically, never counted as proof of robust specificity',
 'frozen_gold':'original author RIGHT filenames used solely as source class eligibility; no novel independent gold and no trained thresholds',
 'no_GPU':True,
}

def sha(b):return hashlib.sha256(b).hexdigest()
def sha_path(p):return sha(p.read_bytes())
def put(path,obj):
 if path.exists():raise RuntimeError('do not overwrite frozen output: '+str(path))
 data=(json.dumps(obj,sort_keys=True,indent=2)+'\n').encode()
 temp=path.with_suffix('.staging')
 temp.write_bytes(data);os.replace(temp,path)

def original_media(path):
 for root in (OLD/'media',NEW/'media'):
  p=root/path
  if p.is_file():return p
 raise FileNotFoundError(path)

def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 if any((OUT/f).exists() for f in ('design.json','frozen_pairs.json','image_only_results.json','scored.json')):raise RuntimeError('frozen outputs already exist')
 assert sha_path(OLD/'source_test.json')==SOURCE_SHA
 records=json.loads((OLD/'source_test.json').read_text())
 assert len(records)==340
 receipts={}
 for data in ((OLD/'predictions.json'),(NEW/'predictions.json')):
  js=json.loads(data.read_text());fs=js.get('each_file_original_sha256') or js.get('files')
  for path,v in fs.items():
   s=v.get('sha256')
   if not s:continue
   if path in receipts and receipts[path]!=s:raise RuntimeError('source artifact conflict: '+path)
   receipts[path]=s
 right=[(i,row['images']) for i,row in enumerate(records) if len(row['images'])==2 and all(x.startswith('image/right/') for x in row['images'])]
 assert len(right)==170
 clean={}
 for ident,paths in right:
  assert all(p in receipts for p in paths)
  assert all(sha_path(original_media(p))==receipts[p] for p in paths)
  clean[ident]={'id':ident,'app':paths[0].split('/')[2], 'activity':paths[0].split('/')[-1].rsplit('_',1)[0], 'screenshots':paths, 'native_sizes':[list(Image.open(original_media(p)).size) for p in paths]}
 byapp=defaultdict(list)
 for item in clean.values():byapp[item['app']].append(item)
 manifest=[]
 for case in sorted(clean.values(),key=lambda x:x['id']):
  eligible=[v for v in byapp[case['app']] if v['id']!=case['id'] and v['activity']!=case['activity'] and v['native_sizes']==case['native_sizes']]
  alts=sorted(eligible,key=lambda x:sha((case['screenshots'][0]+'\n'+x['screenshots'][0]).encode()))[:3]
  for alt in alts:
   manifest.append({'case_id':case['id'],'other_case_id':alt['id'],'app':case['app'], 'activity':case['activity'],'other_activity':alt['activity'],'native_sizes':case['native_sizes'],'screenshots':case['screenshots'],'other_clean_screenshots':alt['screenshots'], 'actual_sha':[receipts[p] for p in case['screenshots']],'other_ref_sha':[receipts[p] for p in alt['screenshots']], 'same_source_revision':True,'same_app':True,'same_activity':False, 'same_orientation':True,'same_native_size':True})
 assert manifest and all(x['app'] and x['activity']!=x['other_activity'] and x['same_native_size'] for x in manifest)
 put(OUT/'design.json',DESIGN)
 put(OUT/'frozen_pairs.json',{'design_sha':sha_path(OUT/'design.json'),'pinned_source_sha':SOURCE_SHA,'prior_source_file_receipts_SHA':[sha_path(OLD/'predictions.json'),sha_path(NEW/'predictions.json')],'original_correct_cases':len(right),'eligible_source_cases':len(set(x['case_id'] for x in manifest)), 'comparisons':manifest})
 print('FROZEN',len(manifest),'matched pairs each 2 orientations','distinct_cases',len(set(x['case_id'] for x in manifest)),'apps',len(set(x['app'] for x in manifest)),'sha',sha_path(OUT/'frozen_pairs.json'))

@lru_cache(maxsize=48)
def read_rgb(s):return np.array(Image.open(original_media(s)).convert('RGB'))

def features(a,b):
 if a.shape!=b.shape:return None
 delta=cv2.absdiff(a,b); maxi=np.max(delta,axis=2)
 return {'frac_24':float(np.mean(maxi>=24)),'mae':float(np.mean(delta)),'any_changed':bool(np.any(maxi)), 'pixel_count':int(maxi.size)}

def pixels():
 if (OUT/'image_only_results.json').exists():raise RuntimeError('frozen image pixels already exist')
 plan=json.loads((OUT/'frozen_pairs.json').read_text());assert plan['pinned_source_sha']==SOURCE_SHA
 start=time.perf_counter();rows=[]
 unique=sorted({p for pair in plan['comparisons'] for p in pair['screenshots']+pair['other_clean_screenshots']})
 verified={p:sha_path(original_media(p)) for p in unique}
 for pair in plan['comparisons']:
  assert all(verified[p]==s for p,s in zip(pair['screenshots'],pair['actual_sha']))
  assert all(verified[p]==s for p,s in zip(pair['other_clean_screenshots'],pair['other_ref_sha']))
 for i,pair in enumerate(plan['comparisons']):
  a=[read_rgb(p) for p in pair['screenshots']];b=[read_rgb(p) for p in pair['other_clean_screenshots']]
  # Exact SAME budget & inputs for localized and global baseline.
  v=[features(x,y) for x,y in zip(a,b)]
  exact=[features(x,x) for x in a]
  if any(x is None for x in v+exact):raise AssertionError('native sizes differ; frozen selection invalid')
  rows.append({'case_id':pair['case_id'],'other_case_id':pair['other_case_id'],'actual_sha':pair['actual_sha'],'other_ref_sha':pair['other_ref_sha'],'opposite_clean_source_features':v,'exact_reference_control':exact,
   'candidate_localized_wrong_ref':any(x['frac_24']>.0005 for x in v),
   'same_budget_global_wrong_ref':any(x['mae']>1.5 for x in v),
   'any_pixel_wrong_ref':any(x['any_changed'] for x in v),
   'exact_ref_candidate':any(x['frac_24']>.0005 for x in exact),
   'source_activity_mismatch':pair['activity']!=pair['other_activity']})
  if (i+1)%50==0:print('PIXEL_PROGRESS',i+1,'/',len(plan['comparisons']),'elapsed',round(time.perf_counter()-start,2),flush=True)
 duration=time.perf_counter()-start
 put(OUT/'image_only_results.json',{'frozen_selection_sha':sha_path(OUT/'frozen_pairs.json'),'pinned_source_sha':SOURCE_SHA,'CPU_wall_seconds':round(duration,3),'GPU_calls':0,'shared_decode_three_pixel_features':True,'source_author_labels_not_joined':True,'cases':rows})
 print('PIXELS_FROZEN',len(rows),'s',round(duration,3),'sha',sha_path(OUT/'image_only_results.json'))

def score():
 if (OUT/'scored.json').exists():raise RuntimeError('do not rewrite scored experiment')
 plan=json.loads((OUT/'frozen_pairs.json').read_text());raw=json.loads((OUT/'image_only_results.json').read_text())
 assert raw['frozen_selection_sha']==sha_path(OUT/'frozen_pairs.json')
 cases=raw['cases']; lookup={(x['case_id'],x['other_case_id']):x for x in plan['comparisons']}
 assert len(cases)==len(lookup)
 perapp=defaultdict(list)
 for row in cases:
  p=lookup[(row['case_id'],row['other_case_id'])]
  assert row['source_activity_mismatch'] and p['same_app'] and p['same_orientation'] and p['same_native_size']
  assert all(x['any_changed']==False for x in row['exact_reference_control'])
  perapp[p['app']].append(row)
 n=len(cases)
 metrics={
  'author_clean_distinct_activity_pairs':n,
  'paired_orientation_comparisons':2*n,
  'author_clean_bad_flags_localized_wrong_activity_ref':sum(x['candidate_localized_wrong_ref'] for x in cases),
  'author_clean_bad_flags_global_wrong_activity_ref':sum(x['same_budget_global_wrong_ref'] for x in cases),
  'author_clean_bad_flags_any_pixel_wrong_activity_ref':sum(x['any_pixel_wrong_ref'] for x in cases),
  'exact_self_ref_bad_flags_localized':sum(x['exact_ref_candidate'] for x in cases),
  'correct_source_binding_review_for_wrong_activity':sum(x['source_activity_mismatch'] for x in cases),
  'distinct_author_clean_cases':len(set(x['case_id'] for x in cases)),
  'distinct_apps':len(perapp),
  'distinct_wrong_activity_sources':len(set(x['other_case_id'] for x in cases)),
  'max_reuses_of_one_source_case':max(Counter(x['case_id'] for x in cases).values()),
  'per_app':{app:{'comparisons':len(rows),'localized_bad_flags':sum(x['candidate_localized_wrong_ref'] for x in rows),'global_bad_flags':sum(x['same_budget_global_wrong_ref'] for x in rows)} for app,rows in perapp.items()}
 }
 put(OUT/'scored.json',{'source':'ImDim RotBench published right screenshot states, pinned SHA; all test originals author-right but not human certified', 'prior_source':REV,'frozen_selection_sha':sha_path(OUT/'frozen_pairs.json'),'frozen_predictions_sha':sha_path(OUT/'image_only_results.json'),'source_version_verified_against_previous_SHA_receipts':True,'metrics':metrics,'source_property':'same-app and viewport yet DIFFERENT ACTIVITY, not same source state; unbound reference must REVIEW, no legitimate semantic defect verdict','interpretations':{'pixel_difference':True,'actual_GUI_defect':'unproved; original screenshot is right by source author','verified_immutable_screenshot_reference':'no; alternate activity mismatch, hence REVIEW'},'cost':{'CPU_wall_seconds':raw['CPU_wall_seconds'],'GPU_calls':0}, 'cases':[{'id':x['case_id'],'alt_id':x['other_case_id'],'localized_bad_flag':x['candidate_localized_wrong_ref'],'global_bad_flag':x['same_budget_global_wrong_ref'],'any_pixel_bad_flag':x['any_pixel_wrong_ref'],'exact_control_bad_flag':x['exact_ref_candidate'],'wrong_version_correct_gate':'review'} for x in cases], 'limitations':['Not an independently certified human true-negative visual quality set; screenshots manually selected right by source authors.','Alternates are genuine clean screens but DIFFERENT ACTIVITY; source-validated pixel diff should NOT compare them.','Duplicate source activities and applications are correlated; per-comparison counts do not imply  independent samples.','Same activity naturally changing screens, real heldout developer-confirmed bug screenshots and owner final outputs remain untested.']})
 print('SCORED',json.dumps({k:v for k,v in metrics.items() if k!='per_app'},sort_keys=True),'sha',sha_path(OUT/'scored.json'))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','pixels','score']);globals()[ap.parse_args().stage]()
