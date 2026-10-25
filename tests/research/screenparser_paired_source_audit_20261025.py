#!/usr/bin/env python3
"""Independent paired error audit on completed 32 real website ScreenParser predictions.

Only existing screenshot-independent DOM target boxes and already frozen YOLO
outputs. NO new inference or image download. Compares paired clean/defect UI
source families with the same target selector/criterion; location-to-target
IoU, not defect classification, and no text prediction. Pre-existing IoU cutoffs
0.10/0.30/0.50, no threshold search. Cluster bootstrap by source website.
"""
from pathlib import Path
import json,hashlib,statistics,collections
import numpy as np
from scipy.stats import binomtest
ROOT=Path('/data/var/perceptual-independent-20261024/screenparser-fullpage-real')
EVAL=ROOT/'screenparser_yolo_32_target_localization_eval.json'
SOURCE=ROOT/'fullpage_manifest.json'
OUT=ROOT/'screenparser_paired_source_audit_20261025.json'
THRESHOLDS=(.1,.3,.5)


def family(name):
 return name.replace('--clip-overflow','').replace('--protrude-viewport','')

def audit():
 if OUT.exists():print('EXISTING_AUDIT_PRESERVED_NO_RERUN');return
 raw=EVAL.read_bytes();source=SOURCE.read_bytes()
 model=json.loads(raw);cases=json.loads(source)['cases']
 assert len(model['cases'])==32 and len(cases)==32
 lookup={row['source_item_SHA256']:row for row in model['cases']}
 joined=[]
 for key,x in sorted(cases.items()):
  sha=hashlib.sha256(key.encode()).hexdigest();r=lookup[sha]
  assert r['source_img_sha256']==x['image_sha256']
  assert r['independent_defect_gold']==x['independent_gold_violation']
  joined.append({'pair_key':(family(x['page_id']),x['criterion']),'family':family(x['page_id']),'source_id_sha256':sha,'gold_bad':x['independent_gold_violation'],'criterion':x['criterion'],'split':x['split'],'IoU':r['best_class_agnostic_iou'],'best_label':r['best_label'],'best_confidence':r['best_confidence'],'center_covered':r['target_center_covered_by_nongeneric_box'],'total_model_detections':r['n_total_boxes'],'time_seconds':r['latency_s'],'target_rect':r['target_visual_rect']})
 grouped=collections.defaultdict(list)
 for x in joined:grouped[x['pair_key']].append(x)
 assert len(grouped)==16
 pairs=[]
 for (name,criterion),group in sorted(grouped.items()):
  assert len(group)==2 and {x['gold_bad'] for x in group}=={True,False}
  clean=next(x for x in group if not x['gold_bad']);defect=next(x for x in group if x['gold_bad'])
  assert clean['split']==defect['split']
  pairs.append({'family':name,'criterion':criterion,'split':clean['split'],'clean_IoU':clean['IoU'],'defect_IoU':defect['IoU'],'delta_clean_minus_defect':round(clean['IoU']-defect['IoU'],6),'clean_center_detected':clean['center_covered'],'defect_center_detected':defect['center_covered'],'clean_best_label':clean['best_label'],'defect_best_label':defect['best_label'],'clean_full_bbox':clean['target_rect'],'defect_visible_bbox':defect['target_rect'],'different_image_sha256':cases[next(k for k,x in cases.items() if hashlib.sha256(k.encode()).hexdigest()==clean['source_id_sha256'])]['image_sha256']!=cases[next(k for k,x in cases.items() if hashlib.sha256(k.encode()).hexdigest()==defect['source_id_sha256'])]['image_sha256']})
 # Similar source images are not necessarily identical CLEAN/DEFECT pairs.
 rng=np.random.default_rng(20261025)
 websites=sorted({p['family'] for p in pairs});cluster={site:[p for p in pairs if p['family']==site] for site in websites}
 samples=[]
 for rep in range(2000):
  selected=rng.choice(websites,size=len(websites),replace=True)
  observed=[p for site in selected for p in cluster[site]]
  samples.append(statistics.mean(x['delta_clean_minus_defect'] for x in observed))
 diffs=[p['delta_clean_minus_defect'] for p in pairs]
 descriptive={
 'n_pairs':len(pairs),'independent_website_families':len(websites),'same_pixel_pair_count':sum(not p['different_image_sha256'] for p in pairs),
 'mean_clean_IoU':statistics.mean(p['clean_IoU'] for p in pairs),'mean_defect_IoU':statistics.mean(p['defect_IoU'] for p in pairs),
 'mean_paired_IoU_delta':statistics.mean(diffs),'median_paired_IoU_delta':statistics.median(diffs),
 'paired_source_cluster_bootstrap_95pct_mean_delta':[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))],
 'clean_higher_IoU_than_bad_count':sum(d>0 for d in diffs),'bad_higher_IoU_than_clean_count':sum(d<0 for d in diffs),'tie_count':sum(d==0 for d in diffs),
 'sign_test_two_sided_p':float(binomtest(sum(d>0 for d in diffs),n=sum(d!=0 for d in diffs),p=.5).pvalue),
 'paired_threshold_comparison':{str(t):{'clean_visible_element_located':sum(p['clean_IoU']>=t for p in pairs),'defect_visible_element_located':sum(p['defect_IoU']>=t for p in pairs)} for t in THRESHOLDS},
 'by_defect_type':{typ:{'n_pairs':len(a),'mean_iou_delta':statistics.mean(x['delta_clean_minus_defect'] for x in a),'clean_iou_at_least_point5':sum(x['clean_IoU']>=.5 for x in a),'defect_iou_at_least_point5':sum(x['defect_IoU']>=.5 for x in a)} for typ in sorted({p['criterion'] for p in pairs}) if (a:=[p for p in pairs if p['criterion']==typ])},
 }
 times=[x['time_seconds'] for x in joined]
 temporal={'total_cpu_seconds':sum(times),'mean_seconds_per_screenshot':statistics.mean(times),'median_seconds_per_screenshot':statistics.median(times),'p10_seconds':float(np.quantile(times,.1)),'p90_seconds':float(np.quantile(times,.9)),'min_seconds':min(times),'max_seconds':max(times)}
 report={'date':'2026-10-25','source':'UIJudgeBench v0.4.0 real archived website defects paired with control screenshot, 32 full viewport captured real Firefox renderings','frozen_predictions_SHA256':hashlib.sha256(raw).hexdigest(),'capture_manifest_SHA256':hashlib.sha256(source).hexdigest(),'model':'docling-project/ScreenParser v2 YOLO11-L pinned f029e565f1206577402e43206454522075be3f72, Ultralytics 8.3.221 on loaded Nitro CPU','no_new_model_calls':True,'target_metric':'class agnostic box overlap with only ONE source identified DOM element per page, NOT all element parser recall/precision and NOT defect detection accuracy','paired_cases':pairs,'paired_stats':descriptive,'resource':temporal,'limitations':'Severe ground truth modality mismatch: source element CSS bounding box can include long clipped or overflowing region while YOLO training labels target only visible distinct UI child pixels. Defective targets systematically harder to match with IoU, but this does not necessarily mean generic parser fails at other element labels. Two different mutated source pages share an identical screenshot hash, but they are NOT necessarily matched clean/defect pairs. Repeated websites and only one target per page make inference provisional. No threshold fit; no real global parser box gold or independent subjective visible defect rating.'}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('PAIRED_REAL_SITES',descriptive,'CPU',temporal)
if __name__=='__main__':audit()
