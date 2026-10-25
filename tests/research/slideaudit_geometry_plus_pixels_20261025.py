#!/usr/bin/env python3
"""Two complementary detectors on SAME externally reviewed slide visual defects.

Using previously FROZEN independent SlideAudit actual 343-source rectangle scores,
actual pixel/OCR scores, and strong-human label matched slide controls.
No re-inference, no source media edits, no retune, no source labels in detector.
Fixed alternate policies for occlusion/cut-off: source OR image (sensitive) and
source AND image (selective). Distinguishes false alarms/misses for both,
matched pair-bootstrap confidence and paired image/sourced overlap AUROC.

Secondary OCClUSION localization: intersections of source layout rectangles
are independently computed from original authored descriptions *before* source
human bbox labels join; select top 3 intersections by existing overlap fraction
>=0.10 of smaller object. Compare top3 proposal centers with strong-positive
human annotated region; chance probability from region area approximated by
20x20 fixed grid. Large gold boxes may make random points look successful.
This is exploratory on used benchmark, not untouched final test selection.
"""
from pathlib import Path
import json,hashlib,math,collections,statistics
import numpy as np
from sklearn.metrics import roc_auc_score
VAR=Path('/data/var/perceptual-independent-20261024')
SOURCE=VAR/'slideaudit_selection_20261025.json'
GEOM=VAR/'slideaudit_343_source_geometry_frozen_20261025.json'
PIX=VAR/'slideaudit_343_image_only_ocr_cv_frozen_20261025.json'
PROPOSALS=VAR/'slideaudit_343_layout_intersection_proposals_frozen_20261025.json'
OUT=VAR/'slideaudit_343_source_pixel_complementarity_20261025.json'
ROOT=VAR/'slideaudit-public/data/descriptions'
TASKS={'Content Overflow/Cut-off':('source_predicted_overflow','predicted_clipped_text_near_canvas_boundary','max_source_text_offscreen_area_fraction','ocr_conf25_near_border_fraction'),'Occluded Content':('source_predicted_any_element_overlap','predicted_ocr_box_overlap','max_source_any_pair_intersection_of_smaller','ocr_conf25_max_word_iou')}


def bbox(desc):
 raw=(desc.get('positionAndSize') or {}).get('normalized') or {}
 if any(k not in raw for k in ('x','y','width','height')):return None
 x,y,w,h=[raw[k] for k in ('x','y','width','height')]
 if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in [x,y,w,h]) or w<=0 or h<=0:return None
 return [float(x),float(y),float(x+w),float(y+h)]

def intersects(a,b):
 return [max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]

def area(a):return max(0,a[2]-a[0])*max(0,a[3]-a[1])


def freeze_proposals():
 if PROPOSALS.exists():
  a=json.loads(PROPOSALS.read_text())
  if a.get('status')=='complete':print('SOURCE_PROPOSALS_ALREADY_FROZEN_NOT_REPEATED');return
  raise RuntimeError('incomplete prior proposal computation, inspect first')
 source=json.loads(SOURCE.read_text());ids=sorted(map(int,source['items']));assert len(ids)==343
 proposals={}
 for i in ids:
  p=ROOT/(f'slide_{i:04d}.json');raw=p.read_bytes();doc=json.loads(raw)
  rects=[(bbox(x),str(x.get('type','unknown'))) for x in doc['elements']]
  rects=[x for x in rects if x[0] is not None]
  r=[]
  for obj_idx,a in enumerate(rects):
   for b in rects[obj_idx+1:]:
    overlap=intersects(a[0],b[0]);inter=area(overlap)
    if inter<=0:continue
    smaller=min(area(a[0]),area(b[0]));fraction=inter/smaller
    if fraction>=.10:
     r.append({'center_normalized':[(overlap[0]+overlap[2])/2,(overlap[1]+overlap[3])/2],'intersection_normalized':overlap,'overlap_fraction_of_smaller':round(fraction,6),'rect_types':[a[1],b[1]]})
  r.sort(key=lambda d:(-d['overlap_fraction_of_smaller'],-area(d['intersection_normalized'])))
  proposals[f'slide_{i:04d}']={'input_source_sha256':hashlib.sha256(raw).hexdigest(),'source_candidates_total':len(r),'top3':r[:3]}
 assert len(proposals)==343 and set(proposals)=={f'slide_{i:04d}' for i in ids}
 PROPOSALS.write_text(json.dumps({'status':'complete','config':{'source_layout_proposals_only':True,'at_least_10pct_smaller_object_intersection':True,'max_three_ranked_proposals':3,'source_human_labels_not_exposed_to_proposal':True},'slides':proposals},indent=2)+'\n')
 print('ORIGINAL_LAYOUT_343_SOURCE_INTERSECTION_PROPOSALS_FROZEN',len(proposals),'WITH_NONEMPTY',sum(bool(x['top3']) for x in proposals.values()))


def cm(rows,key):
 tp=sum(r['gold'] and r[key] for r in rows);fp=sum(not r['gold'] and r[key] for r in rows)
 fn=sum(r['gold'] and not r[key] for r in rows);tn=sum(not r['gold'] and not r[key] for r in rows)
 return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':tp/(tp+fn),'specificity':tn/(tn+fp),'precision_balanced_sample':tp/(tp+fp) if tp+fp else None,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2}


def interval(pairs,metric,n=3000):
 rng=np.random.default_rng(20261025);values=[]
 for _ in range(n):
  x=[t for i in rng.integers(0,len(pairs),size=len(pairs)) for t in pairs[i]]
  try:q=metric(x)
  except ValueError:continue
  if q is not None and math.isfinite(q):values.append(q)
 return [round(float(np.quantile(values,t)),4) for t in (.025,.975)] if len(values)>n//2 else None


def evaluate():
 if OUT.exists():print('PRIOR_COMBINED_GEOMETRY_PIXEL_RESULTS_PRESERVED');return
 selected=json.loads(SOURCE.read_text());geometry=json.loads(GEOM.read_text());pixels=json.loads(PIX.read_text());proposals=json.loads(PROPOSALS.read_text())
 assert geometry['status']==pixels['status']==proposals['status']=='complete'
 assert set(geometry['features_only_NO_source_elements_or_text'])==set(pixels['input_predictions'])==set(proposals['slides'])
 summary={}
 for defect,(gf,pf,gscore,pscore) in TASKS.items():
  cfg=selected['task_counts'][defect];pairs=[]
  for yes,no in zip(cfg['selected_positive_ids'],cfg['selected_negative_ids']):
   pair=[]
   for id,gold in [(yes,True),(no,False)]:
    item=selected['items'][str(id)];key=f'slide_{id:04d}';g=geometry['features_only_NO_source_elements_or_text'][key]['source_element_geometry'];p=pixels['input_predictions'][key]
    assert item['labels'][defect]['strong'] and item['labels'][defect]['positive']==gold
    pair.append({'id':id,'origin':item['source_type'],'gold':gold,'geom':bool(g[gf]),'pixel':bool(p[pf]),'OR':bool(g[gf] or p[pf]),'AND':bool(g[gf] and p[pf]),'gscore':g[gscore],'pscore':p[pscore]})
   assert pair[0]['origin']==pair[1]['origin']
   pairs.append(pair)
  rows=[r for pair in pairs for r in pair]
  outcomes={k:cm(rows,k) for k in ['geom','pixel','OR','AND']}
  feature_roc={'source_original_geometry':roc_auc_score([r['gold'] for r in rows],[r['gscore'] for r in rows]),'actual_pixels_OCR':roc_auc_score([r['gold'] for r in rows],[r['pscore'] for r in rows])}
  auroc_delta=interval(pairs,lambda x:float(roc_auc_score([r['gold'] for r in x],[r['gscore'] for r in x])-roc_auc_score([r['gold'] for r in x],[r['pscore'] for r in x])))
  stats={'strong_pairs':len(pairs),'human_strong_positives':len(pairs),'human_strong_category_negative_controls':len(pairs),'binary_policy':outcomes,'predeclared_continuous_roc_auc':feature_roc,'paired_bootstrap_95pct_auc_difference_geom_minus_pixel':auroc_delta,'paired_bootstrap_95pct_bacc_OR':interval(pairs,lambda x:cm(x,'OR')['balanced_accuracy']),'paired_bootstrap_95pct_bacc_AND':interval(pairs,lambda x:cm(x,'AND')['balanced_accuracy'])}
  if defect=='Occluded Content':
   positive=[x['id'] for x in rows if x['gold']]
   localization=[]
   for id in positive:
    item=selected['items'][str(id)];box=item['labels'][defect]['boxes']
    if not box:continue
    W,H=item['width'],item['height'];regions=[[b['x']/W,b['y']/H,(b['x']+b['width'])/W,(b['y']+b['height'])/H] for b in box]
    def inside(x,y):return any(a<=x<=c and b<=y<=d for a,b,c,d in regions)
    candidates=proposals['slides'][f'slide_{id:04d}']['top3'];hit=any(inside(*c['center_normalized']) for c in candidates)
    grid=[inside((i+.5)/25,(j+.5)/25) for i in range(25) for j in range(25)]
    frac=sum(grid)/len(grid)
    localization.append({'n_proposals':len(candidates),'human_bbox_n':len(box),'hit_any':hit,'random_any_center_hit_prob_area_proxy':1-(1-frac)**len(candidates) if candidates else 0,'gold_region_area_fraction_on_grid':frac})
   stats['source_intersection_localization']={'annotated_occluded_positive_slides':len(localization),'source_proposals_at_least_one':sum(x['n_proposals']>0 for x in localization),'source_top3_hits_human_bbox':sum(x['hit_any'] for x in localization),'mean_random_matched_number_points_hit_probability_proxy':statistics.mean(x['random_any_center_hit_prob_area_proxy'] for x in localization),'gold_bbox_mean_canvas_fraction':statistics.mean(x['gold_region_area_fraction_on_grid'] for x in localization),'limits':'Region boxes sometimes cover large areas, source boxes and human altered raster may have different layout frames. Hit is center in any source human annotated area, not full precision/recall; random points have high baseline.'}
  summary[defect]=stats
  print('BINARY_COMPLEMENTARY',defect,outcomes,'AUROC',feature_roc,'AUC_DIFF_CI',auroc_delta,flush=True)
  if 'source_intersection_localization' in stats:print('SOURCE_RECT_CANDIDATE_SPATIAL_VALIDATION',stats['source_intersection_localization'],flush=True)
 out={'date':'2026-10-25','property':'Real human cut-off and occlusion from actual slide raster, with independent source-layout metadata and earlier frozen screenshot-only OCR signal, fixed OR/AND combinations','source_selection_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'source_geometry_frozen_sha256':hashlib.sha256(GEOM.read_bytes()).hexdigest(),'source_pixel_detection_sha256':hashlib.sha256(PIX.read_bytes()).hexdigest(),'source_intersection_proposals_sha256':hashlib.sha256(PROPOSALS.read_bytes()).hexdigest(),'fixed_algorithm_and_no_retraining':True,'same_original_343_rasters_and_matched_category_controls':True,'human_label_scoring_is_exploratory_data_already_seen_in_prior_SLIDEAUDIT_pass':True,'tasks':summary,'scientific_limitations':['Slide source geometry is not current browser DOM or physically verified render geometry; author layout descriptions may precede alterations','Balanced strong-agreement categories and shared source families can confound; positive/negative controls not certified totally error-free','OR and AND were not trained on labels, but evaluated after prior separate baselines consumed source; exploratory diagnostic not unbiased final test','Source overlap includes intentional stacking and background shapes; does not by itself prove hidden content','Human annotated bounding boxes can cover broad slide regions; spatial candidate hit chance often nontrivial','Container internal clip is not captured by slide exterior coordinates; label validity depends on source/raster correspondence']}
 OUT.write_text(json.dumps(out,indent=2)+'\n')
 print('FROZEN_TWO_MODALITIES_FULL_343_EVALUATION_SAVED',OUT)
if __name__=='__main__':freeze_proposals();evaluate()
