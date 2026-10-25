#!/usr/bin/env python3
"""Source-reviewed slide visual errors versus FIXED image-only checks.

Scores original 343 annotated screenshots (SlideAudit UIST25, 3 raters) using
ONLY source-blind frozen visual feature extraction. Positive/negative human
strong agreement labels come from a separate selection file read AFTER the
343 screenshot-only predictions are finished and immutable; control pairs
matched by source_type and alteration before model metrics were run.

Exact property heads are predeclared: (A) edge text for cut-off/overflow,
(B) OCR word overlap for occlusion, (C) estimated patch contrast for low text
contrast. No training or threshold tuning. Report FP and FN, AUROC for fixed
continuous indicators and parent-slide-pair bootstrap uncertainty. Many
source slides are altered/synthetic, NOT app GUI screenshots; a slide annotation
for a category is not proof all other visible defects are absent. Bbox label
may describe whole region and not precise text glyphs. No live GPU inference.
"""
from pathlib import Path
import json,hashlib,collections,statistics,math,datetime
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path('/data/var/perceptual-independent-20261024')
SRC=ROOT/'slideaudit_selection_20261025.json'
RAW=ROOT/'slideaudit_343_image_only_ocr_cv_frozen_20261025.json'
OUT=ROOT/'slideaudit_human_visual_label_performance_20261025.json'
LABELS={'Content Overflow/Cut-off':{'flag':'predicted_clipped_text_near_canvas_boundary','scores':{'ocr_word_border_frac':('ocr_conf25_near_border_fraction',1),'canvas_border_edge_density':('edge_density_on_canvas_border',1),'canvas_border_vs_center_edge_density':('edge_density_border_minus_interior',1),'negative_mean_ocr_confidence':('mean_ocr_word_confidence',-1)}},'Occluded Content':{'flag':'predicted_ocr_box_overlap','scores':{'ocr_max_word_iou':('ocr_conf25_max_word_iou',1),'ocr_low_conf_frac':('ocr_low_conf_fraction',1),'image_inner_edge_density':('edge_density_inside',1),'negative_mean_ocr_confidence':('mean_ocr_word_confidence',-1)}},'Insufficient Color Contrast for Readability':{'flag':'predicted_low_contrast','scores':{'estimated_low_contrast_word_fraction':('estimated_fraction_words_under_4p5',1),'negative_p25_estimated_contrast':('p25_estimated_word_contrast',-1),'estimated_very_low_contrast_word_fraction':('estimated_fraction_words_under_3',1),'negative_mean_ocr_confidence':('mean_ocr_word_confidence',-1),'ocr_low_conf_frac':('ocr_low_conf_fraction',1)}}}
CFG={'dataset':'SlideAudit UIST 2025 2400 all annotations, original CC BY4, subset 343 selected no use of image feature','matched_stratum':['source_type','alteration'],'pos_and_controls':'all strong-agreement positive slides, equally sized strong negative controls per category matched on origin+alteration','fixed_binary_flags':{k:v['flag'] for k,v in LABELS.items()},'continuous_score_directions':{k:v['scores'] for k,v in LABELS.items()},'no_fitting_or_threshold_selection_on_human_labels':True,'BOOTSTRAP_REPLICATES':3000,'BOOTSTRAP_SEED':20261025,'risk':'Estimating category false-alarm proxy against category-specific strong-negative human annotation, not absence of every GUI/slide defect; OCR only sees text, occlusion often shapes.'}

def summarize(data):
 tp=sum(x['gold_positive'] and x['pred'] for x in data);fp=sum(not x['gold_positive'] and x['pred'] for x in data)
 fn=sum(x['gold_positive'] and not x['pred'] for x in data);tn=sum(not x['gold_positive'] and not x['pred'] for x in data)
 return {'n':len(data),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'sensitivity':tp/(tp+fn) if tp+fn else None,'specificity_vs_human_strong_negative':tn/(tn+fp) if tn+fp else None,'precision_in_enriched_balanced_source':tp/(tp+fp) if tp+fp else None,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,'positive_count':tp+fn,'control_count':fp+tn}

def auroc(pairs,name):
 data=[(bool(x['gold_positive']),x['features'].get(name)) for x in pairs]
 valid=[(g,float(score)) for g,score in data if score is not None and math.isfinite(float(score))]
 if len({g for g,y in valid})<2:return None,len(valid)
 return float(roc_auc_score([g for g,y in valid],[y for g,y in valid])),len(valid)

def bootstrap(pairs,score_name=None):
 rng=np.random.default_rng(CFG['BOOTSTRAP_SEED']);B=CFG['BOOTSTRAP_REPLICATES']
 res={'sensitivity':[],'specificity_vs_human_strong_negative':[],'balanced_accuracy':[]}
 if score_name:res['auroc']=[]
 for _ in range(B):
  index=rng.integers(0,len(pairs),size=len(pairs))
  sample=[x for i in index for x in pairs[i]]
  stats=summarize(sample)
  for k in ['sensitivity','specificity_vs_human_strong_negative','balanced_accuracy']:
   if stats[k] is not None:res[k].append(stats[k])
  if score_name:
   auc,_=auroc(sample,score_name)
   if auc is not None:res['auroc'].append(auc)
 return {k:([round(float(np.quantile(v,.025)),4),round(float(np.quantile(v,.975)),4)] if len(v)>=B//2 else None) for k,v in res.items()}

def localize_positive(source,image,flaw):
 # Compare top predictions against source human drawn boxes; only for positive
 # examples. Candidate prediction list determined from pixels ONLY. Ground boxes
 # used AFTER all proposals frozen, not fed to predictor. Center inclusion is a
 # permissive localization criterion; bbox may cover most of the canvas.
 annotations=source['labels'][flaw].get('boxes',[])
 preds={'Insufficient Color Contrast for Readability':image.get('predicted_candidate_lowcontrast_boxes',[]),'Content Overflow/Cut-off':image.get('predicted_near_boundary_boxes',[]),'Occluded Content':[x for pair in image.get('predicted_overlapping_word_boxes',[]) for x in pair]}.get(flaw,[])
 if not annotations:return {'with_human_box':False,'candidate_count':len(preds)}
 sw,sh=image['original_dimensions_wh'];w,h=image['evaluated_dimensions_wh']
 bboxes=[]
 for box in annotations:
  bboxes.append((box['x']*w/sw,box['y']*h/sh,(box['x']+box['width'])*w/sw,(box['y']+box['height'])*h/sh))
 def is_inside(x,y):return any(a<=x<=c and b<=y<=d for a,b,c,d in bboxes)
 hit=sum(is_inside((x0+x1)/2,(y0+y1)/2) for x0,y0,x1,y1 in preds)
 # Empirical random-position candidate baseline for SOURCE annotation area only;
 # deterministic 20x20 evenly spaced evaluation grid (not a model benchmark).
 grid=[is_inside(w*(x+.5)/20,h*(y+.5)/20) for x in range(20) for y in range(20)]
 return {'with_human_box':True,'human_bounding_box_count':len(bboxes),'predicted_candidate_count':len(preds),'at_least_one_predicted_center_inside_human_box':bool(hit),'center_hit_count':hit,'human_box_canvas_area_proxy':sum(grid)/len(grid),'random_candidate_location_ge_one_hit_proxy':1-(1-sum(grid)/len(grid))**min(5,len(preds)) if preds else 0}


def main():
 if OUT.exists():print('PRIOR_SCORED_RESULTS_PRESERVED_NO_REPLAY');return
 frozen=json.loads(RAW.read_text());assert frozen['status']=='complete' and len(frozen['input_predictions'])==343
 selection=json.loads(SRC.read_text());assert len(selection['items'])==343
 assert frozen['source_selection_sha256']==hashlib.sha256(SRC.read_bytes()).hexdigest()
 cases=frozen['input_predictions']
 assert sorted(int(k[6:]) for k in cases)==sorted(int(k) for k in selection['items'])
 summaries={};extra={};allcases=[]
 for flaw,definition in LABELS.items():
  chosen=selection['task_counts'][flaw]
  positive=chosen['selected_positive_ids'];negative=chosen['selected_negative_ids']
  assert len(positive)==len(negative) and len(positive)>0
  pairrows=[];singles=[]
  for pos,neg in zip(positive,negative):
   pair=[]
   for i,is_positive in [(pos,True),(neg,False)]:
    src=selection['items'][str(i)];image=cases[f'slide_{i:04d}'];flag=bool(image[definition['flag']])
    assert src['labels'][flaw]['strong'] and src['labels'][flaw]['positive']==is_positive
    row={'source_sha256':image['source_image_sha256'],'slide_id':i,'source_type':src['source_type'],'alteration':src['alteration'],'gold_positive':is_positive,'pred':flag,'status':image['status'],'features':{},'human_box_localization':localize_positive(src,image,flaw) if is_positive else None}
    for alias,(name,direction) in definition['scores'].items():
     v=image.get(name)
     row['features'][alias]=float(v)*direction if v is not None and math.isfinite(float(v)) else None
    pair.append(row);singles.append(row)
   assert pair[0]['source_type']==pair[1]['source_type'] and pair[0]['alteration']==pair[1]['alteration']
   pairrows.append(pair)
  assert len(pairrows)==len(positive)
  pred=summarize(singles)
  complete_pairs=[pair for pair in pairrows if all(row['status']=='complete' for row in pair)]
  complete_only=summarize([row for pair in complete_pairs for row in pair]) if complete_pairs else None
  score_operational_coverage={'matched_pairs_total':len(pairrows),'matched_pairs_with_both_OCR_complete':len(complete_pairs),'matched_pairs_with_OCR_abstention':len(pairrows)-len(complete_pairs),'individual_positive_OCR_abstentions':sum(row['status']!='complete' for row in singles if row['gold_positive']),'individual_control_OCR_abstentions':sum(row['status']!='complete' for row in singles if not row['gold_positive'])}
  ranks={}
  for feature in definition['scores']:
   auc,available=auroc(singles,feature)
   ranks[feature]={'AUROC':round(auc,5) if auc is not None else None,'source_cases_with_score':available,'source_cases_abstained':len(singles)-available}
  best_predeclared='ocr_word_border_frac' if flaw=='Content Overflow/Cut-off' else 'ocr_max_word_iou' if flaw=='Occluded Content' else 'estimated_low_contrast_word_fraction'
  boot=bootstrap(pairrows,best_predeclared)
  poscases=[r for r in singles if r['gold_positive']]
  present=[r['human_box_localization'] for r in poscases if r['human_box_localization']['with_human_box']]
  localization={'source_positive_cases_with_bbox':len(present),'source_positive_cases_no_bbox':len(poscases)-len(present),'positive_bbox_proposals_with_any_hit':sum(x['at_least_one_predicted_center_inside_human_box'] for x in present),'source_positive_cases_with_no_proposals':sum(x['predicted_candidate_count']==0 for x in present),'mean_random_center_hit_fraction':statistics.mean(x['human_box_canvas_area_proxy'] for x in present) if present else None,'mean_random_any_candidate_hit_prob_proxy':statistics.mean(x['random_candidate_location_ge_one_hit_proxy'] for x in present) if present else None}
  source_types={t:summarize([x for x in singles if x['source_type']==t]) for t in ('Gemini','gdcdataset','Google')}
  alterations={t:summarize([x for x in singles if x['alteration']==t]) for t in ('nojitter','texture','layout','alignment')}
  failures=[r for r in singles if r['gold_positive']!=r['pred']]
  summaries[flaw]={'paired_cases':len(pairrows),'strong_gold_positive':len(poscases),'strong_negative_control':len(pairrows),'fixed_binary_flags':definition['flag'],'confusion_on_balanced_human_source':pred,'confusion_when_both_OCR_completed':complete_only,'operational_OCR_coverage':score_operational_coverage,'continuous_scores_no_threshold_fitting':ranks,'selected_primary_predeclared_continuous_feature':best_predeclared,'matched_pair_bootstrap_95pct':boot,'positive_human_box_coverage':localization,'source_type_subgroups':source_types,'alteration_subgroups':alterations,'failures_by_source':dict(collections.Counter(x['source_type'] for x in failures)),'original_unagreed_positive_excluded':chosen['original_weak_positive_not_adjudicated']}
  print('SLIDEAUDIT_GENUINE_HUMAN_LABEL',flaw,'STRONG_POS',len(poscases),'FLAGS',pred,'METRIC_AUROC',ranks,'BOOTSTRAP',boot,'LOCALIZATION',localization,'BY_SOURCE',source_types,flush=True)
 src_errors=collections.Counter(row['status'] for row in cases.values())
 report={'date':'2026-10-25','source':'SlideAudit UIST 2025 2400 human labelled slides, 3 crowd annotators, CC BY4 public pinned 642d490b7c1d2e78a50a631bfd359433397f3ecf','source_selection_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'frozen_prediction_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'predictions_completion_counts':dict(src_errors),'evaluated_image_count':len(cases),'source_strata':CFG,'tasks':summaries,'scope_limits':['Slide images, including intentional layout/texture modifications: source domain is presentation slides, NOT Android/mobile native GUIs or 3D industrial machinery','Source positive/control pairs balanced by source_type and alteration, so precision on this artificially enriched sample is NOT deployment PPV','Negative human label is strong disagreement with the category, not an independently proven fully defect-free screen; other flaws may co-occur','Fixed binary rules return a negative default when OCR times out; these must be regarded as abstentions, not guaranteed true negatives, and paired complete-case performance/coverage is separately reported','The model has no visual intentionality: OCR may ignore clipped words or overlapped shapes, and word-patch contrast estimated from grayscale percentiles is NOT computed foreground/background CSS WCAG contrast','Tesseract human bounding boxes are not fed to predictor; spatial human boxes may cover broad regions, so center-hit chance must be reported','Only three fixed heuristics and descriptive AUROC comparisons, not an automatically trained visual judge; no threshold optimization or model selection using source labels','Human annotation strong_agreement not necessarily unanimous; absent category weak/disagreement items excluded, low-contrast positives only 20','Original image/annotations and recovered OCR text remain outside Git, only pseudonymous source aggregate evidence in project docs']}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('SLIDEAUDIT_343_SCORED_ALL_TASKS',OUT,'TOTAL_IMAGES',len(cases),'ORIGINAL_HUMAN_POS',sum(v['strong_gold_positive'] for v in summaries.values()),'ORIGINAL_SOURCE_RETENTION',dict(src_errors))
if __name__=='__main__':main()
