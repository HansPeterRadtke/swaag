#!/usr/bin/env python3
"""Frozen-design sham-reference negative control for 2026-10-26 RotBench.

Uses identical image decoding, thresholds, and native resolution as aligned
source test. Replaces the approved reference with a different activity's
*same-resolution* clean reference. No gold labels are read here.
"""
from collections import defaultdict
from pathlib import Path
import hashlib, json, time
from PIL import Image
from rotbench_frozen_actual_android_pixels_20261026 import ROOT, color_stats, dump_atomic, sha


def main():
    out=ROOT/'sham_reference_predictions.json'
    if out.exists():raise RuntimeError('SHAM_PREDICTIONS_ALREADY_FROZEN: no rerun')
    plan=json.loads((ROOT/'selection.json').read_text())
    pool=sorted({ref for case in plan['selection'] for ref in case['references']})
    dims={ref:Image.open(ROOT/'media'/ref).size for ref in pool}
    byshape=defaultdict(list)
    for ref in pool:byshape[dims[ref]].append(ref)
    start=time.perf_counter()
    cases=[]
    for row in plan['selection']:
        choice=[]
        stats=[]
        for source,ref in zip(row['images'],row['references']):
            # Match native image shape and orientation, but not source activity.
            pool2=[x for x in byshape[dims[ref]] if x.split('/')[2] != ref.split('/')[2]]
            pool2.sort(key=lambda x:hashlib.sha256((source+'\n'+x).encode()).hexdigest())
            other=pool2[0] if pool2 else None
            choice.append(other)
            stats.append(color_stats(ROOT/'media'/source,ROOT/'media'/other) if other else None)
        good=all(v is not None for v in stats)
        cases.append({'record_index':row['record_index'],'sham_ref_sha256':[sha((ROOT/'media'/x).read_bytes()) if x else None for x in choice],'stats':stats,'predictions':{'changed_area_0p05pct_sham':any(s['changed_frac24']>0.0005 for s in stats) if good else None,'global_meanabs_gt1p5_sham':any(s['mean_abs_rgb']>1.5 for s in stats) if good else None}})
    dump_atomic(out,{'selection_sha256':sha((ROOT/'selection.json').read_bytes()),'no_author_gold_read':True,'native_shape_only':True,'different_source_activity':True,'same_methods_and_thresholds_as_source_aligned':True,'cases':cases,'pixel_feature_seconds':round(time.perf_counter()-start,3)})
    print('SHAM_FROZEN',len(cases),'fully_matched',sum(all(x is not None for x in c['stats']) for c in cases),'sha256',sha(out.read_bytes()))

if __name__=='__main__':main()
