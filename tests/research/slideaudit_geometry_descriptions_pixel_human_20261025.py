#!/usr/bin/env python3
"""Independently evaluate actual SOURCE rectangle geometry vs REAL slide defects.

2400 original SlideAudit authored+modified presentations; all 343 previously
source-human-selected positive/control slides, 2400 original description and
metadata files pinned at 642d490b7c1d2e78a50a631bfd359433397f3ecf.

Step 1 freeze source geometry of all 343 slide object descriptions (TEXT_BOX,
IMAGE, TABLE, LINE), WITHOUT reading any human annotations, output image or
other detectors' outcome. Predeclared rules: >2% source-object off physical
slide, >10% intersection of smaller axis-aligned rectangles among two text
objects or any two objects. These are SOURCE metadata checks, not perception;
wrongly or intentionally overlapping text boxes are possible and do not prove
actual rendered occlusion, nor a human-visible defect.

Step 2 use pre-existing human task-specific positive/negative paired controls
(SlideAudit has_strong_agreement true) to independently score true/false.
Controls matched on source_type and alteration, selection independent of
source features. Compare to earlier frozen screenshot-only OCR and CLIP on the
EXACT same instances, no tuning. Source JSON text is NOT copied into Git.
This can prove only correlation of source geometry with human visible issues;
source description may precede later raster alterations; no output claims
about source clipping without rendered-image proof.
"""
from pathlib import Path
import hashlib,json,collections,statistics,math
import numpy as np
from sklearn.metrics import roc_auc_score
VAR=Path('/data/var/perceptual-independent-20261024')
ROOT=VAR/'slideaudit-public/data'
RAW=VAR/'slideaudit_343_source_geometry_frozen_20261025.json'
SCORE=VAR/'slideaudit_343_source_geometry_human_defects_20261025.json'
LABELS=('Content Overflow/Cut-off','Occluded Content','Insufficient Color Contrast for Readability')
CONFIG={
 'source':'SlideAudit original authored slide TEXT_BOX / IMAGE / TABLE / LINE object positions from pinned source descriptions, 343 actual human-reviewed slide images used solely for ID selection',
 'source_type':'source_metadata_not_image_pixels',
 'source_revision':'642d490b7c1d2e78a50a631bfd359433397f3ecf',
 'predefined_out_of_canvas_fraction_of_object_area':.02,
 'predefined_pair_overlap_fraction_of_smaller_object_area':.10,
 'invalid_source_geometry_is_unknown_not_automatic_defect':True,
 'binary_overflow':'at least one positive-area TEXT_BOX extends >2% of own area outside slide canvas',
 'binary_occlusion':'at least one pair of positive-area authored objects intersects >=10% of the smaller source rectangle, unrestricted type',
 'binary_text_only_occlusion':'two TEXT_BOX rectangles intersect >=10% of smaller source rectangle',
 'limited_transfer_to_pixels':'Source rectangles may be original pre-alteration or design-intent boxes, and cannot detect text exceeding its bounding container when container remains within canvas. Layering, opacity and occlusion semantics not present.',
 'control_labels':'binary strong-agreement human positive and matched category-negative source slide samples from previous independent selection; labels never accessible during freeze',
 'auc_features':['max_source_text_offscreen_area_fraction','max_source_any_offscreen_area_fraction','max_source_any_pair_intersection_of_smaller','max_source_text_pair_intersection_of_smaller','max_source_text_image_intersection_of_smaller','sum_source_pair_overlap_area_in_slide_units'],
}


def box(el):
 pos=(el.get('positionAndSize') or {}).get('normalized') or {}
 if any(k not in pos for k in ('x','y','width','height')):return None
 x,y,w,h=[pos[k] for k in ('x','y','width','height')]
 if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (x,y,w,h)):return None
 if w<=0 or h<=0:return None
 return {'bounds':(float(x),float(y),float(x+w),float(y+h)),'area':float(w*h),'type':str(el.get('type','unknown'))}


def intersect(a,b):
 a0,a1,a2,a3=a; b0,b1,b2,b3=b
 return max(0,min(a2,b2)-max(a0,b0))*max(0,min(a3,b3)-max(a1,b1))


def analyze(objects):
 parsed=[box(el) for el in objects]
 valid=[b for b in parsed if b is not None]
 text=[b for b in valid if b['type']=='TEXT_BOX']
 invalid=len(parsed)-len(valid)
 visible=(0.0,0.0,1.0,1.0)
 for b in valid:
  b['out_frac']=max(0,1-intersect(visible,b['bounds'])/b['area'])
 pairs=[]
 for idx,a in enumerate(valid):
  for b in valid[idx+1:]:
   amount=intersect(a['bounds'],b['bounds'])
   if amount<=0:continue
   f=amount/min(a['area'],b['area'])
   types={a['type'],b['type']}
   pairs.append({'fraction':f,'slide_area':amount,'two_text':a['type']==b['type']=='TEXT_BOX','text_image':types=={'TEXT_BOX','IMAGE'}})
 feat={
  'source_element_count':len(objects),'source_valid_geometry_count':len(valid),'source_invalid_geometry_count':invalid,
  'source_textbox_count':len(text),'source_image_count':sum(b['type']=='IMAGE' for b in valid),
  'max_source_text_offscreen_area_fraction':max((b['out_frac'] for b in text),default=0.),
  'max_source_any_offscreen_area_fraction':max((b['out_frac'] for b in valid),default=0.),
  'source_text_offscreen_count_over_2pct':sum(b['out_frac']>.02 for b in text),
  'max_source_any_pair_intersection_of_smaller':max((p['fraction'] for p in pairs),default=0.),
  'max_source_text_pair_intersection_of_smaller':max((p['fraction'] for p in pairs if p['two_text']),default=0.),
  'max_source_text_image_intersection_of_smaller':max((p['fraction'] for p in pairs if p['text_image']),default=0.),
  'sum_source_pair_overlap_area_in_slide_units':sum(p['slide_area'] for p in pairs),
  'source_pair_intersection_count':len(pairs),
  'source_intentional_overlap_uncertain':True,
  'source_predicted_overflow':any(b['out_frac']>.02 for b in text),
  'source_predicted_any_element_overlap':any(p['fraction']>=.1 for p in pairs),
  'source_predicted_text_text_overlap':any(p['fraction']>=.1 and p['two_text'] for p in pairs),
 }
 return feat


def freeze():
 if RAW.exists():
  old=json.loads(RAW.read_text());
  if old['status']=='complete':print('PREVIOUS_343_SOURCE_LAYOUT_FROZEN_NOT_REPLAYED');return
  raise RuntimeError('prior uncertain geometry extraction; inspect first')
 paths=sorted((ROOT/'images').glob('slide_*.png'))
 assert len(paths)==343
 rows={}
 for p in paths:
  dpath=ROOT/'descriptions'/(p.stem+'.json')
  data=dpath.read_bytes();d=json.loads(data)
  assert str(d['slide_id'])==p.stem
  rows[p.stem]={'original_source_description_sha256':hashlib.sha256(data).hexdigest(),'source_element_geometry':analyze(d['elements'])}
 result={'date':'2026-10-25','config':CONFIG,'status':'complete','descriptions_n':len(rows),'image_input_ids_only':343,'features_only_NO_source_elements_or_text':rows}
 RAW.write_text(json.dumps(result,indent=2)+'\n')
 print('FROZEN_SOURCE_343_GEOMETRY_NO_HUMAN_LABELS_YET',RAW,'COUNT',len(rows),'OFFTEXT',sum(x['source_element_geometry']['source_predicted_overflow'] for x in rows.values()),'TEXT_OVERLAP',sum(x['source_element_geometry']['source_predicted_text_text_overlap'] for x in rows.values()),'ANY_OVERLAP',sum(x['source_element_geometry']['source_predicted_any_element_overlap'] for x in rows.values()),'VALID_ELEMENTS',sum(x['source_element_geometry']['source_valid_geometry_count'] for x in rows.values()))


def cm(rows,key='pred'):
 tp=sum(x['gold'] and x[key] for x in rows);fp=sum(not x['gold'] and x[key] for x in rows)
 fn=sum(x['gold'] and not x[key] for x in rows);tn=sum(not x['gold'] and not x[key] for x in rows)
 return {'n':len(rows),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'precision_on_balanced_sample':tp/(tp+fp) if tp+fp else None,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None}


def groupboot(pairs,metric,n=3000):
 rng=np.random.default_rng(20261025);samples=[]
 for i in range(n):
  ii=rng.integers(0,len(pairs),len(pairs));allrows=[x for j in ii for x in pairs[j]]
  try:val=metric(allrows)
  except ValueError:continue
  if val is not None and math.isfinite(val):samples.append(val)
 return [round(float(np.quantile(samples,q)),4) for q in (.025,.975)] if len(samples)>=n/2 else None


def evaluate():
 if SCORE.exists():print('SOURCE_LAYOUT_HUMAN_SCORES_ALREADY_SAVED');return
 frozen=json.loads(RAW.read_text());assert frozen['status']=='complete' and len(frozen['features_only_NO_source_elements_or_text'])==343
 selection=json.loads((VAR/'slideaudit_selection_20261025.json').read_text());assert selection['n_selected_unique']==343
 prev=json.loads((VAR/'slideaudit_human_visual_label_performance_20261025.json').read_text())
 clip=json.loads((VAR/'slideaudit_clip_343_human_flaw_comparison_20261025.json').read_text())
 kinds={'Content Overflow/Cut-off':'source_predicted_overflow','Occluded Content':'source_predicted_any_element_overlap','Insufficient Color Contrast for Readability':'source_predicted_text_text_overlap'}
 results={}
 for label,flag in kinds.items():
  task=selection['task_counts'][label];p=task['selected_positive_ids'];q=task['selected_negative_ids'];assert len(p)==len(q)
  pairs=[]
  for pos,neg in zip(p,q):
   pair=[]
   for ident,gold in [(pos,True),(neg,False)]:
    m=selection['items'][str(ident)];desc=frozen['features_only_NO_source_elements_or_text'][f'slide_{ident:04d}']['source_element_geometry']
    assert m['labels'][label]['strong'] and m['labels'][label]['positive']==gold
    pair.append({'id_sha256':hashlib.sha256(str(ident).encode()).hexdigest(),'origin':m['source_type'],'alteration':m['alteration'],'gold':gold,'pred':desc[flag],'numeric_features':{feature:desc[feature] for feature in CONFIG['auc_features']},'has_source_text':desc['source_textbox_count']>0,'invalid_source_rect_count':desc['source_invalid_geometry_count']})
   assert pair[0]['origin']==pair[1]['origin'] and pair[0]['alteration']==pair[1]['alteration']
   pairs.append(pair)
  rows=[x for pair in pairs for x in pair]
  scores={}
  for feat in CONFIG['auc_features']:
   yy=[x['gold'] for x in rows];vv=[x['numeric_features'][feat] for x in rows]
   scores[feat]={'auroc':float(roc_auc_score(yy,vv)),'paired_bootstrap95':groupboot(pairs,lambda r:float(roc_auc_score([x['gold'] for x in r],[x['numeric_features'][feat] for x in r])))}
  tp=cm(rows);boot=groupboot(pairs,lambda r:cm(r)['balanced_accuracy'])
  results[label]={'source_positive_n':len(p),'source_negative_n':len(q),'frozen_geometry_rule':flag,'confusion':tp,'paired_bootstrap95_balanced_accuracy':boot,'all_six_predeclared_continuous_sources':scores,'matched_pair_n':len(pairs),'comparators_same_raw_slides':{'previous_OCR_OpenCV_confusion':prev['tasks'][label]['confusion_on_balanced_human_source'],'previous_CLIP_confusion':clip['tasks'][label]['binary_prompt_rule']},'by_slide_origin':{origin:cm([r for r in rows if r['origin']==origin]) for origin in ('Gemini','gdcdataset','Google')},'source_geometry_contains_valid_textboxes':sum(x['has_source_text'] for x in rows),'invalid_source_geometry_rows':sum(x['invalid_source_rect_count']>0 for x in rows)}
  print('HUMAN_VISIBLE_DEFECT_VS_SOURCE_GEOMETRY',label,'N',len(rows),'FLAGS',tp,'AUC',[(k,round(v['auroc'],4)) for k,v in scores.items()],'CI',boot,flush=True)
 report={'date':'2026-10-25','study':'source slide object rectangles versus 343 REAL rendered human-reviewed defects, matched positives and category-negative slide controls, source-only geometry frozen before reference labels','source_geometry_prediction_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'source_human_selection_sha256':hashlib.sha256((VAR/'slideaudit_selection_20261025.json').read_bytes()).hexdigest(),'config':CONFIG,'outcomes':results,'model_training_or_threshold_tuning':False,'limits':['Original slide source element descriptions may predate visual augmentation and need not be pixels of final rendering','Out-of-slide object bounds not sufficient for cropped TEXT INSIDE a container; object overlap may be intentional. No layering/order/opacity/style in geometry','Category-negative means strongly absent one human annotated visual issue, not guaranteed no defects; balanced from three categories, some same image repeated across tasks','Image-only OCR and CLIP source-independent comparison already frozen on same 343 source inputs','AABB source geometry is NOT a vision model and no general web DOM accuracy is established on presentation slides','Source image and description content stored outside repository.']}
 SCORE.write_text(json.dumps(report,indent=2)+'\n')
 print('SOURCE_LAYOUT_343_HUMAN_RESULTS_SAVED',SCORE)
if __name__=='__main__':freeze();evaluate()
