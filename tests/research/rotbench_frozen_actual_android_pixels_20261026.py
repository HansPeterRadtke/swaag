#!/usr/bin/env python3
"""Source-blind pixel-first RotBench Android heldout evaluation.

Stages MUST execute freeze -> extract -> score. Test labels are loaded only by
score stage. External original screenshots remain under /data/var, not in Git.
Author labels are synthetic mutations, not blinded human semantic ratings.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
import requests

REVISION = '25b6e8214c5be600f54dc0834eaf4597fa9192e6'
ROOT = Path('/data/var/perceptual-independent-20261024/rotbench-android-heldout-20261026')
REMOTE = f'https://huggingface.co/datasets/ImDim/RotBench/resolve/{REVISION}/'
# Fixed *before* any heldout pixel feature extraction or author-label read.
DESIGN = {
    'source': 'ImDim/RotBench bug_detection/test.json, external author-generated rotation-bug screenshot dataset',
    'revision': REVISION,
    'test_split': 'author heldout test, distinct from other local UI studies',
    'selection': 'hash rank screenshot-path pair within author path classes: 25 layout-clip, 25 layout-overlap, 50 right. The path categories are used for stratification ONLY, never in predictions.',
    'gold_blinding': 'Author labels not parsed in freeze/extract; score joins author labels only after prediction SHA frozen.',
    'source_reference': 'original author captured unmodified right screenshot of same Android activity and orientation',
    'primary': 'per orientation, max RGB channel absolute change >=24; flag case if changed area fraction > 0.0005 (0.05%) in either orientation; no labels or filename categories read into prediction function',
    'matched_baseline': 'same decoded pixels, reference and compute budget: flag case if mean RGB absolute change >1.5 (0-255) in either orientation',
    'negative_control': 'same native screenshot/reference pixel compare with identical threshold but opposite/randomly paired clean reference drawn from another case, same orientation and resize. Deliberately mismatched reference tests source-identity necessity; control must not be mistaken for valid production detection.',
    'null_control': 'always say no defect; count FNs and TNs',
    'unverifiable': 'real benign state transitions, human visible severity of synthetic edits, live native app GUI, production fonts/animations, owner warehouse/video, third-party source reference license',
    'pixel_processing': 'RGB PIL decode at native resolution; if actual and matched right screenshot dimensions disagree, abstain (no relabel to pass). For null wrong right, resize preserving 1-to-1 array shape.',
    'confounds': 'right examples compare byte-identical images against themselves; filenames leak dataset labels, so all claim is conditional on exact aligned reference. No model trained/tuned here.'
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_atomic(path: Path, obj) -> None:
    b=(json.dumps(obj,indent=2,ensure_ascii=True,sort_keys=True)+'\n').encode()
    tmp=path.with_suffix('.tmp')
    tmp.write_bytes(b)
    os.replace(tmp,path)


def download(rel: str, root=ROOT):
    p=root/'media'/rel
    if p.is_file():
        return rel,sha(p.read_bytes()),p.stat().st_size,'existing'
    r=requests.get(REMOTE+rel,timeout=25)
    r.raise_for_status()
    data=r.content
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise RuntimeError('Non-PNG '+rel)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(data)
    return rel,sha(data),len(data),'new'


def reference_path(img):
    # Never inspect 'defective' or its bug class to make a prediction.
    parts=img.split('/')
    if len(parts)<4 or parts[0]!='image':
        raise ValueError('bad image path '+img)
    # right/<package>/<activity>_orientation.png OR defective/<type>/<package>/<activity>...
    app,stem=parts[-2:]
    m=re.match(r'^(.+)_(portrait|landscape)(?:_[A-Za-z0-9-]+_[0-9]+)?\.png$',stem)
    if not m:raise ValueError('unknown orientation in '+img)
    return 'image/right/'+app+'/'+m.group(1)+'_'+m.group(2)+'.png'


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    for p in ('design.json','selection.json','source_test.json'):
        if (ROOT/p).exists():raise RuntimeError('FROZEN_EXISTS: refuse to replace '+p)
    r=requests.get(REMOTE+'bug_detection/test.json',timeout=25);r.raise_for_status()
    data=r.content
    records=json.loads(data)
    assert len(records)==340
    groups={'right':[],'layout-clip':[],'layout-overlap':[]}
    for i,row in enumerate(records):
        paths=row['images']
        if len(paths)!=2:continue
        if all(x.startswith('image/right/') for x in paths):cls='right'
        elif all(x.startswith('image/defective/layout-clip/') for x in paths):cls='layout-clip'
        elif all(x.startswith('image/defective/layout-overlap/') for x in paths):cls='layout-overlap'
        else:continue
        groups[cls].append((sha(('\n'.join(paths)).encode()),i,paths))
    assert len(groups['right'])>=50 and len(groups['layout-clip'])>=25 and len(groups['layout-overlap'])>=25
    choice=[]
    for k,n in (('layout-clip',25),('layout-overlap',25),('right',50)):
        for _,i,paths in sorted(groups[k])[:n]:
            choice.append({'record_index':i,'images':paths,'references':[reference_path(x) for x in paths]})
    assert len(choice)==100
    plan={'design':DESIGN,'source_sha256':sha(data),'source_rows':len(records),'heldout_count':len(choice),'selection':choice}
    dump_atomic(ROOT/'design.json',DESIGN)
    dump_atomic(ROOT/'selection.json',plan)
    (ROOT/'source_test.json').write_bytes(data)
    print('FROZEN',ROOT/'selection.json','sha256',sha((ROOT/'selection.json').read_bytes()),'source_sha',sha(data),'classes selected by file naming only',[(k,len(v)) for k,v in groups.items()])


def color_stats(img_path, right_path):
    im=np.asarray(Image.open(img_path).convert('RGB'))
    ref=np.asarray(Image.open(right_path).convert('RGB'))
    if im.shape!=ref.shape:return None
    diff=np.abs(im.astype(np.int16)-ref.astype(np.int16))
    return {'changed_frac24':float(np.mean(np.max(diff,axis=2)>=24)), 'mean_abs_rgb':float(np.mean(diff)), 'size_hw':[int(im.shape[0]),int(im.shape[1])]}


def extract():
    if (ROOT/'predictions.json').exists():raise RuntimeError('FROZEN_PREDICTIONS_EXISTS refuse to rerun')
    select=ROOT/'selection.json'
    plan=json.loads(select.read_text());pairs=plan['selection']
    files=sorted(set(x for row in pairs for x in (row['images']+row['references'])))
    start=time.perf_counter()
    blobs={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        todo=[pool.submit(download,p) for p in files]
        for job in as_completed(todo):
            path,hashval,size,state=job.result()
            blobs[path]={'sha256':hashval,'bytes':size}
    download_s=time.perf_counter()-start
    score_start=time.perf_counter()
    out=[]
    for row in pairs:
        obs=[color_stats(ROOT/'media'/x,ROOT/'media'/y) for x,y in zip(row['images'],row['references'])]
        # Primary, baseline both use EXACTLY the same arrays and image inputs.
        pred={'changed_area_0p05pct':None if None in obs else any(q['changed_frac24']>0.0005 for q in obs),
              'global_meanabs_gt1p5':None if None in obs else any(q['mean_abs_rgb']>1.5 for q in obs),
              'always_no_defect':False}
        out.append({'record_index':row['record_index'],'images_sha256':[blobs[x]['sha256'] for x in row['images']],'reference_sha256':[blobs[x]['sha256'] for x in row['references']],'images_reference_equal_sha256':[blobs[x]['sha256']==blobs[y]['sha256'] for x,y in zip(row['images'],row['references'])], 'orientation_pixel_stats':obs,'predictions':pred})
    score_s=time.perf_counter()-score_start
    report={'source_sha256':plan['source_sha256'],'selection_sha256':sha(select.read_bytes()),'design_sha256':sha((ROOT/'design.json').read_bytes()),'num_case_pairs':len(out),'num_distinct_pngs':len(files),'download_seconds':round(download_s,3),'source_check_pixel_feature_seconds':round(score_s,3),'download_bytes_total':sum(v['bytes'] for v in blobs.values()),'pixel_features_frozen_before_author_gold':True,'each_file_original_sha256':blobs,'cases':out}
    dump_atomic(ROOT/'predictions.json',report)
    print('PREDICTIONS_FROZEN',len(out),'images',len(files),'sec',round(score_s,2),'sha256',sha((ROOT/'predictions.json').read_bytes()),'unknown',sum(x['predictions']['changed_area_0p05pct'] is None for x in out))


def confusion(y,p):
    valid=[(a,b) for a,b in zip(y,p) if b is not None]
    return {'TP':sum(a and b for a,b in valid),'FP':sum((not a) and b for a,b in valid),'FN':sum(a and (not b) for a,b in valid),'TN':sum((not a) and (not b) for a,b in valid),'abstained':len(y)-len(valid)}


def score():
    if (ROOT/'scored.json').exists():raise RuntimeError('SCORED_ALREADY')
    frozen=(ROOT/'predictions.json').read_bytes();data=json.loads(frozen);plan=json.loads((ROOT/'selection.json').read_text())
    gold=json.loads((ROOT/'source_test.json').read_text());assert len(gold)==340
    results=defaultdict(lambda:defaultdict(list))
    cases=[]
    for row in data['cases']:
        record=gold[row['record_index']]
        assert record['images']==next(item['images'] for item in plan['selection'] if item['record_index']==row['record_index'])
        answer=record['conversations'][1]['value'].lower()
        if '"bug": true' in answer:gold_label=True
        elif '"bug": false' in answer:gold_label=False
        else:raise ValueError('unknown gold format '+repr(answer))
        group='original-right' if all(s.startswith('image/right/') for s in record['images']) else 'mutated-'+record['images'][0].split('/')[2]
        cases.append({'record_index':row['record_index'],'gold_bug':gold_label,'group':group,'predictions':row['predictions'],'stats':row['orientation_pixel_stats']})
        for method,pred in row['predictions'].items():
            results['all'][method].append((gold_label,pred))
            results[group][method].append((gold_label,pred))
    metrics={group:{name:confusion([i[0] for i in arr],[i[1] for i in arr]) for name,arr in bymethod.items()} for group,bymethod in results.items()}
    assert sum(c['gold_bug'] for c in cases)==50 and sum(not c['gold_bug'] for c in cases)==50
    report={'source_sha256':sha((ROOT/'source_test.json').read_bytes()),'frozen_prediction_sha256':sha(frozen),'source_license_status':'Dataset card contains no explicit image license; originals isolated, not committed or redistributed', 'source_author_gold':'Synthetic author-injected display/layout defects on actual captured Android screenshot pairs; author labels not independent human severity certification', 'confusion_by_group':metrics,'selected_cases':cases,'interpretation':'Only verifies exact source-reference pixel difference on this benchmark; does not prove semantic clipping/overlap or benign variation handling.'}
    dump_atomic(ROOT/'scored.json',report)
    print('GOLD_JOINED',json.dumps(metrics['all'],sort_keys=True),'sha',sha((ROOT/'scored.json').read_bytes()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('freeze','extract','score'));args=ap.parse_args();globals()[args.stage]()
