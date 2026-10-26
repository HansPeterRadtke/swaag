#!/usr/bin/env python3
"""Locked follow-up: previously UNUSED RotBench test examples, source pixels first.

Three stages `freeze`, `predict`, `evaluate`. Existing earlier 100 cases and
previous thresholds fixed; never replay or rewrite those original receipts.
Native screenshots and new outputs stay under /data/var, not in project Git.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
from PIL import Image
import requests

ROOT = Path('/data/var/perceptual-independent-20261024/rotbench-unused-source-split-20261026')
OLD = Path('/data/var/perceptual-independent-20261024/rotbench-android-heldout-20261026')
REV = '25b6e8214c5be600f54dc0834eaf4597fa9192e6'
REMOTE = f'https://huggingface.co/datasets/ImDim/RotBench/resolve/{REV}/'
TARGET_OLD_SHA = '2dee311bfaa8a4161178f37eb93ad5af8bcd55695ec9ab0fab5401caf7f93a86'
TARGET_SRC_SHA = '0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7'
DESIGN = {
  'source': 'ImDim/RotBench official bug_detection/test.json pinned revision '+REV,
  'source_scope': '240 previously unused test cases, disjoint record indices from original first 100; same parent dataset and possibly related apps.',
  'prediction_blinding': 'Source author bug labels/conversations never opened until after frozen predictions SHA. Image filenames and path classes used to find source reference, not in prediction functions.',
  'thresholds_fixed_prior_round': {'localized': 'frac(max_RGB_abs_delta>=24)>0.0005 across EITHER native orientation', 'global': 'mean_abs_RGB_delta>1.5 across EITHER orientation', 'any_change': 'fraction(max_RGB_abs_delta>=1)>0', 'null': 'always no defect'},
  'negatives': 'matched right source exact captures; ordinary zero-delta controls are easy. Additional different viewport orientation is genuinely original source-author clean rendering of same activity, not the same requested state; correct behavior is abstain, counterfactual forced resized pixel comparison exposes false alarms.',
  'source_binding': 'exact package, activity, orientation and original source version required for verified SOURCE property. Opposite orientation deliberately fails binding; never auto-fail GUI on a bad reference.',
  'gold_limit': 'Author intentionally mutated Android screenshot pairs, not new human-rated visual severity; normal portrait and landscape are author-approved only within original release.',
  'selection_bias': 'This is a follow-on heldout set within the same source benchmark; thresholds picked in earlier 100-case developer sample. App/source grouping may overlap. Right cases are byte-identical to their own approved reference.',
  'resources': 'isolated original PNGs in /data/var, CPU only, no GPU, no production UI/test app restart',
}

def sha(b):return hashlib.sha256(b).hexdigest()
def save(path, obj):
  if path.exists():raise RuntimeError('no overwrite allowed '+str(path))
  data=(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode()
  tmp=path.with_suffix('.newtmp');tmp.write_bytes(data);os.replace(tmp,path)

def reference(x):
  p=x.split('/')
  if len(p)<4 or p[0]!='image':raise ValueError(x)
  file=p[-1]
  match=re.match(r'^(.+)_(portrait|landscape)(?:_[A-Za-z0-9-]+_[0-9]+)?\.png$',file)
  if not match:raise ValueError(x)
  return '/'.join(['image','right',p[-2],match.group(1)+'_'+match.group(2)+'.png'])

def freeze():
  ROOT.mkdir(parents=True,exist_ok=True)
  if any((ROOT/x).exists() for x in ('design.json','selection.json','predictions.json','evaluated.json')):raise RuntimeError('frozen paths already exist')
  assert sha((OLD/'selection.json').read_bytes())==TARGET_OLD_SHA
  assert sha((OLD/'source_test.json').read_bytes())==TARGET_SRC_SHA
  prev=json.loads((OLD/'selection.json').read_text())
  skipped=set(x['record_index'] for x in prev['selection'])
  data=json.loads((OLD/'source_test.json').read_bytes()); assert len(data)==340 and len(skipped)==100
  # No author conversation text/labels accessed.
  cases=[]
  for i,row in enumerate(data):
    if i in skipped:continue
    paths=row['images'];assert len(paths)==2
    assert all(s.endswith('.png') for s in paths)
    refs=[reference(x) for x in paths]
    assert len(set(refs))==2
    cases.append({'id':i,'native_screenshots':paths,'same_orientation_source_refs':refs})
  assert len(cases)==240
  # Preserve original split source hash, do not copy labels into prediction space.
  save(ROOT/'design.json',DESIGN)
  save(ROOT/'selection.json',{'design_sha256':sha((ROOT/'design.json').read_bytes()),'prior_manifest_SHA':TARGET_OLD_SHA,'external_split_SHA':TARGET_SRC_SHA,'parent_test_cases':340,'already_evaluated':len(skipped),'unused_samples':cases})
  print('FROZEN_SELECTION',len(cases),'sha',sha((ROOT/'selection.json').read_bytes()),'source',TARGET_SRC_SHA)

# Two exact original clean reference paths independently returned 404 at the
# pinned revision (one GET, both HEAD); do not repeat these failed URLs.
KNOWN_MISSING_404 = frozenset({
  'image/right/com.fsck.k9/app.k9mail.feature.widget.unread.UnreadWidgetChooseAccountActivity_landscape.png',
  'image/right/com.fsck.k9/app.k9mail.feature.widget.unread.UnreadWidgetChooseAccountActivity_portrait.png',
})

def fetch(path):
  if path in KNOWN_MISSING_404:
    return path,None,0,'verified_source_404_no_repeat'
  local=ROOT/'media'/path
  if local.is_file():return path,sha(local.read_bytes()),local.stat().st_size,'already'
  existing=OLD/'media'/path
  if existing.is_file():
    data=existing.read_bytes();method='copy_local'
  else:
    r=requests.get(REMOTE+path,timeout=50)
    if r.status_code==404:return path,None,0,'original_release_404'
    r.raise_for_status();data=r.content;method='original_download'
  if not data.startswith(b'\x89PNG\r\n\x1a\n'):raise RuntimeError('bad PNG '+path)
  local.parent.mkdir(parents=True,exist_ok=True)
  # Create new bytes, no editing original or old research media.
  with local.open('xb') as f:f.write(data)
  return path,sha(data),len(data),method

def read_rgb(path):return np.asarray(Image.open(ROOT/'media'/path).convert('RGB'))

def measures(img,ref):
  if img.shape!=ref.shape:return None
  d=np.abs(img.astype(np.int16)-ref.astype(np.int16))
  perpx=np.max(d,axis=2)
  return {'changed_fraction_24':round(float(np.count_nonzero(perpx>=24))/perpx.size,10),
          'changed_fraction_any':round(float(np.count_nonzero(perpx>=1))/perpx.size,10),
          'global_MAE':round(float(np.mean(d)),7), 'shape_hw':[int(img.shape[0]),int(img.shape[1])]}

def predict():
  if (ROOT/'predictions.json').exists():raise RuntimeError('frozen prediction exists; do not repeat')
  sel=ROOT/'selection.json';plan=json.loads(sel.read_text())
  files=sorted(set(z for c in plan['unused_samples'] for z in c['native_screenshots']+c['same_orientation_source_refs']))
  t=time.perf_counter(); blobs={}; download_source={}
  with ThreadPoolExecutor(max_workers=8) as ex:
    for f in as_completed([ex.submit(fetch,x) for x in files]):
      p,dig,size,origin=f.result();blobs[p]={'sha256':dig,'bytes':size};download_source[origin]=download_source.get(origin,0)+1
  download_sec=time.perf_counter()-t
  t=time.perf_counter()
  rows=[]
  for c in plan['unused_samples']:
    missing=[f for f in c['native_screenshots']+c['same_orientation_source_refs'] if blobs[f]['sha256'] is None]
    if missing:
      pred={'aligned':{x:None for x in ('localized','global','any_pixel')},
            'wrong_viewport_forced':{x:None for x in ('localized','global','any_pixel')},
            'wrong_viewport_correct_gate':{'abstain_due_to_dimensions':True},
            'always_clean':{'null':False}}
      rows.append({'id':c['id'],'imgs_hash':[blobs[f]['sha256'] for f in c['native_screenshots']],
                   'refs_hash':[blobs[f]['sha256'] for f in c['same_orientation_source_refs']],
                   'aligned_bytestream_identical':[False,False], 'aligned_measures':[None,None],
                   'wrong_viewport_measures':[None,None], 'predictions':pred,
                   'abstained_missing_pinned_source':missing})
      continue
    imgs=[read_rgb(f) for f in c['native_screenshots']]
    refs=[read_rgb(f) for f in c['same_orientation_source_refs']]
    aligned=[measures(a,b) for a,b in zip(imgs,refs)]
    # Real AUTHOR RIGHT portrait and landscape are legitimate but DIFFERENT
    # actual device states. A simplistic wrong-viewport selector is invalid.
    opposite=[measures(a,b) for a,b in zip(imgs,refs[::-1])]
    # Diagnostic ONLY: force-resize opposite viewport and compute exact same
    # features; that shortcut is an invalid semantic reference, not a label.
    bad=[]
    for a,r in zip(imgs,refs[::-1]):
      re=np.asarray(Image.fromarray(r).resize((a.shape[1],a.shape[0]),Image.Resampling.BILINEAR))
      bad.append(measures(a,re))
    pred={}
    for label,x in [('aligned',aligned),('wrong_viewport_forced',bad)]:
      pred[label]={
        'localized':None if any(v is None for v in x) else any(v['changed_fraction_24']>.0005 for v in x),
        'global':None if any(v is None for v in x) else any(v['global_MAE']>1.5 for v in x),
        'any_pixel':None if any(v is None for v in x) else any(v['changed_fraction_any']>0 for v in x),
      }
    pred['wrong_viewport_correct_gate']={'abstain_due_to_dimensions':any(v is None for v in opposite)}
    pred['always_clean']={'null':False}
    rows.append({'id':c['id'],'imgs_hash':[blobs[f]['sha256'] for f in c['native_screenshots']], 'refs_hash':[blobs[f]['sha256'] for f in c['same_orientation_source_refs']],'aligned_bytestream_identical':[blobs[x]['sha256']==blobs[y]['sha256'] for x,y in zip(c['native_screenshots'],c['same_orientation_source_refs'])], 'aligned_measures':aligned,'wrong_viewport_measures':bad,'predictions':pred})
  feature_s=time.perf_counter()-t
  save(ROOT/'predictions.json',{'selection_sha256':sha(sel.read_bytes()),'external_split_sha256':TARGET_SRC_SHA,'author_labels_unread':True,'num_original_PNGs':len(files),'files':blobs,'download_origin':download_source,'missing_original_release_files':[x for x in files if blobs[x]['sha256'] is None],'download_seconds':round(download_sec,3),'pixel_feature_seconds':round(feature_s,3),'cases':rows})
  print('FROZEN_PREDICTIONS',len(rows),'files',len(files),'download',round(download_sec,1),'pixelCPU',round(feature_s,1),'SHA',sha((ROOT/'predictions.json').read_bytes()),'abstained',sum(x['predictions']['aligned']['localized'] is None for x in rows))

def confusion(ys,ps):
  return {'TP':sum(y and p for y,p in zip(ys,ps) if p is not None),'FP':sum(not y and p for y,p in zip(ys,ps) if p is not None),'FN':sum(y and not p for y,p in zip(ys,ps) if p is not None),'TN':sum(not y and not p for y,p in zip(ys,ps) if p is not None),'abstain':sum(p is None for p in ps)}

def evaluate():
  if (ROOT/'evaluated.json').exists():raise RuntimeError('frozen evaluation exists, do not repeat')
  plan=json.loads((ROOT/'selection.json').read_text());frozen=(ROOT/'predictions.json').read_bytes();pred=json.loads(frozen)
  assert pred['selection_sha256']==sha((ROOT/'selection.json').read_bytes())
  assert pred['external_split_sha256']==TARGET_SRC_SHA
  gold=json.loads((OLD/'source_test.json').read_bytes());assert len(gold)==340
  table=[]
  groups=defaultdict(list)
  for case in pred['cases']:
    ref=next(x for x in plan['unused_samples'] if x['id']==case['id'])
    truth=gold[case['id']]
    assert ref['native_screenshots']==truth['images']
    answer=truth['conversations'][1]['value'].lower()
    if '"bug": true' in answer:y=True
    elif '"bug": false' in answer:y=False
    else:raise ValueError('missing external gold')
    typ='right' if all(s.startswith('image/right') for s in ref['native_screenshots']) else next(s.split('/')[2] for s in ref['native_screenshots'] if s.startswith('image/defective/'))
    app=ref['same_orientation_source_refs'][0].split('/')[2]
    record={'id':case['id'],'external_author_bug':y,'type':typ,'app':app,'reference_name':ref['same_orientation_source_refs'][0].split('/')[-1],'predictions':case['predictions'], 'changed_fraction24_max':max((s['changed_fraction_24'] for s in case['aligned_measures'] if s is not None),default=None), 'changed_fraction_any_max':max((s['changed_fraction_any'] for s in case['aligned_measures'] if s is not None),default=None),'aligned_bytestream_identical':case['aligned_bytestream_identical']}
    table.append(record);groups[typ].append(record)
  assert len(table)==240 and len(set(x['id'] for x in table))==240
  def metrics(cases):
    y=[x['external_author_bug'] for x in cases];d={}
    for method in ('localized','global','any_pixel'):
      d[method]=confusion(y,[x['predictions']['aligned'][method] for x in cases])
      d['wrong_viewport_'+method]=confusion(y,[x['predictions']['wrong_viewport_forced'][method] for x in cases])
    d['always_clean']=confusion(y,[False]*len(cases))
    d['wrong_viewport_correct_gate_abstain']=sum(x['predictions']['wrong_viewport_correct_gate']['abstain_due_to_dimensions'] for x in cases)
    return d
  out={'evaluation_sha256_of_frozen_predictions':sha(frozen),'selection_sha256':sha((ROOT/'selection.json').read_bytes()),'external_source_sha':TARGET_SRC_SHA,'source_gold':'author generated display mutations + right original screenshots, no independently human verified visual defect severity', 'original_source_media_in_repo':False, 'all':metrics(table),'by_type':{k:metrics(v) for k,v in groups.items()},'table':table,'known_bias':'source labels from synthetic mutations; clean controls exactly equal approved per-viewport reference; opposed real portrait/landscape clean but wrong state and only a deliberately invalid reference diagnostic; app family correlated', 'cost':{'native_pixel_source_pair_seconds':pred['pixel_feature_seconds'],'download_seconds':pred['download_seconds'],'original_distinct_pngs':pred['num_original_PNGs'],'gpu_calls':0}}
  save(ROOT/'evaluated.json',out)
  print('HELDOUT_SCORE',json.dumps(out['all'],sort_keys=True),'TYPE_COUNTS',{k:len(v) for k,v in groups.items()},'sha',sha((ROOT/'evaluated.json').read_bytes()))

if __name__=='__main__':
  ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','predict','evaluate']);stage=ap.parse_args().stage;globals()[stage]()
