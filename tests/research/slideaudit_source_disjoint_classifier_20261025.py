#!/usr/bin/env python3
"""Transfer check: do screenshot-only generic features predict real human defects?

SlideAudit pinned original 343 balanced matched real/synthetic slides with three
trained human annotators each. Only frozen image pixels + OCR/CV features are
fed into a fixed, lightweight LogisticRegression. Train on TWO distinct slide
source families and evaluate entirely on the THIRD (Gemini, gdcdataset,
Google), rotating leave-one-source-family-out. This prevents within-source
slide styling/augmentation leakage. No tuning of feature list, coefficients,
source grouping, class weights or probability threshold on heldout labels.

This is an exploratory trained pixel-only heuristic, not a pretrained visual
model. Small positive counts for contrast make that fold unstable. The data
are slides, not GUI apps; source negatives are only strongly no-TARGET-label.
No licensed input bytes or private data in project Git. No GPU use.
"""
from pathlib import Path
import json,hashlib,collections,statistics,datetime
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
VAR=Path('/data/var/perceptual-independent-20261024')
SRC=VAR/'slideaudit_selection_20261025.json'
RAW=VAR/'slideaudit_343_image_only_ocr_cv_frozen_20261025.json'
OUT=VAR/'slideaudit_343_human_label_source_transfer_20261025.json'
LABELS={
 'Content Overflow/Cut-off':'predicted_clipped_text_near_canvas_boundary',
 'Occluded Content':'predicted_ocr_box_overlap',
 'Insufficient Color Contrast for Readability':'predicted_low_contrast'
}
FEATURES=[
 'edge_density_on_canvas_border','edge_density_inside','edge_density_border_minus_interior',
 'canvas_luma_interior','canvas_luma_border','ocr_raw_word_count',
 'ocr_word_count_conf25','mean_ocr_word_confidence','ocr_low_conf_fraction',
 'ocr_conf25_near_border_fraction','ocr_conf25_max_word_iou',
 'ocr_conf25_median_word_height','minimum_estimated_word_contrast',
 'p25_estimated_word_contrast','estimated_fraction_words_under_4p5',
 'estimated_fraction_words_under_3'
]
SOURCES=['Gemini','gdcdataset','Google']
CFG={'source':'SlideAudit UIST25 343 strongly human-labelled slide visual outputs','input':'fixed 16 source screenshot-derived pixel/OCR features, NO category/text labels, no source_type/alteration features','features':FEATURES,'outer_split':'leave one slide source_type completely OUT: Gemini, gdcdataset, Google','pipeline':'sklearn SimpleImputer(strategy=median,keep_empty_features=True) + StandardScaler + LogisticRegression(C=1.0,class_weight=balanced,solver=lbfgs,max_iter=400,random_state=20261025)','fixed_decision_probability_threshold':0.5,'positive_definitions':LABELS,'no_parameter_tuning_or_feature_selection':True,'coverage':'return explicit noOCR abstentions; OCR missing features imputed by train-only median; per-source performance and aggregate source confusion, not population prevalence.'}

def metrics(y,scores,threshold=.5):
 pred=np.array(scores)>=threshold;gold=np.array(y,dtype=bool)
 tp=int(np.sum(pred&gold));fp=int(np.sum(pred&~gold));fn=int(np.sum(~pred&gold));tn=int(np.sum(~pred&~gold))
 return {'n':len(y),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'precision':tp/(tp+fp) if tp+fp else None,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if (tp+fn and tn+fp) else None,'roc_auc':float(roc_auc_score(gold,scores)) if len(set(gold))==2 else None}

def main():
 if OUT.exists():print('SOURCE_TRANSFER_PREVIOUS_RESULTS_NOT_RECALCULATED');return
 source=json.loads(SRC.read_text());raw=json.loads(RAW.read_text())
 assert raw['status']=='complete' and len(raw['input_predictions'])==source['n_selected_unique']==343
 assert raw['source_selection_sha256']==hashlib.sha256(SRC.read_bytes()).hexdigest()
 all_results={}
 for task,baseline_flag in LABELS.items():
  selected=source['task_counts'][task]
  positives=set(selected['selected_positive_ids']);controls=set(selected['selected_negative_ids'])
  assert len(positives)==len(controls)
  ids=sorted(positives|controls)
  inputs=[]
  for slide_id in ids:
   meta=source['items'][str(slide_id)];img=raw['input_predictions'][f'slide_{slide_id:04d}']
   assert meta['labels'][task]['strong']
   assert bool(meta['labels'][task]['positive'])==(slide_id in positives)
   inputs.append({'id':slide_id,'source_type':meta['source_type'],'gold':slide_id in positives,'baseline':bool(img[baseline_flag]),'ocr_status':img['status'],'features':[float(img[k]) if img.get(k) is not None else float('nan') for k in FEATURES]})
  folds={};heldall=[]
  for heldout in SOURCES:
   train=[r for r in inputs if r['source_type']!=heldout]
   held=[r for r in inputs if r['source_type']==heldout]
   assert train and held and len({r['gold'] for r in train})==2
   assert len({r['gold'] for r in held})==2,(task,heldout,'not enough positive/negative')
   X=np.asarray([r['features'] for r in train],dtype=float)
   Y=np.asarray([r['gold'] for r in train],dtype=int)
   Xtest=np.asarray([r['features'] for r in held],dtype=float)
   pipeline=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('linear',LogisticRegression(C=1.0,class_weight='balanced',solver='lbfgs',max_iter=400,random_state=20261025))])
   pipeline.fit(X,Y)
   p=pipeline.predict_proba(Xtest)[:,1].tolist()
   labels=[r['gold'] for r in held]
   bas=[int(r['baseline']) for r in held]
   evals=metrics(labels,p)
   prior=metrics(labels,bas)
   folds[heldout]={'train_n':len(train),'heldout_n':len(held),'positives':sum(labels),'negatives':len(held)-sum(labels),'model_withheld_source_confusion':evals,'prior_frozen_boolean_rule':prior,'inference_missing_OCR_status':collections.Counter(r['ocr_status'] for r in held),'model_calibration_apparent_train_score':metrics(Y,pipeline.predict_proba(X)[:,1]).get('balanced_accuracy'),'source_group':heldout}
   heldall.extend([{'id':r['id'],'source_type':heldout,'gold':r['gold'],'model_score':p[i],'baseline_score':bas[i],'ocr_status':r['ocr_status']} for i,r in enumerate(held)])
  overall=metrics([r['gold'] for r in heldall],[r['model_score'] for r in heldall]);b=metrics([r['gold'] for r in heldall],[r['baseline_score'] for r in heldall])
  # This is leave-one-out by only 3 broad source origins, not an unseen
  # deployment super-holdout; retain in outputs for future independent use.
  all_results[task]={'source_folds':folds,'out_of_source_combined_model':overall,'same_examples_untrained_prior_threshold':b,'n_source_family_folds':len(folds),'source_type_class_counts':{g:{'yes':sum(r['gold'] for r in inputs if r['source_type']==g),'no':sum(not r['gold'] for r in inputs if r['source_type']==g)} for g in SOURCES}}
  print('LEAVE_ONE_SOURCE_OUT',task,'MODEL',overall,'FIXED_RULE',b,'FOLDS', {name:(v['model_withheld_source_confusion']['roc_auc'],v['model_withheld_source_confusion']['recall'],v['model_withheld_source_confusion']['specificity']) for name,v in folds.items()},flush=True)
 result={'date':'2026-10-25','config':CFG,'source_selection_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'raw_343_pixels_model_predictions_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'trained_on_human_categories_must_be_treated_as_training':True,'original_source_type_domain_groups':SOURCES,'task_results':all_results,'limitations':'Only three source origins, nested alterations within source may leak visual templates, class prevalence artificially balanced, contrast positives only 20, source human labels strong agreement not guaranteed perfect. This simple logistic classifier is not a purpose-built available vision model; source overfit cannot be excluded. The output here is exploratory signal from generic pixel+OCR features, not certified UI clipping/overlap or WCAG compliance. Any later selection from these evaluations consumes the dataset and requires fresh external holdout.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('SLIDEAUDIT_SOURCE_DISJOINT_PIXEL_CLASSIFIER_SAVED',OUT)
if __name__=='__main__':main()
