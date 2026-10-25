#!/usr/bin/env python3
"""Source-grouped uncertainty and independent-label crosswalk for two REAL tests.

Read-only analysis of already frozen human-labelled 119 generated GLBs and
132 paired screenshot OCR predictions. No OCR/mesh re-execution, no model/GPU,
no threshold changes. Group bootstrap by ORIGINAL prompt or mobile app;
script reports descriptive uncertainty and never claims fresh super-test.
Annotator/source links by unique (prompt,generator), never guess duplicate keys.
All original media, per-screenshot words and public benchmark rows remain in
/data/var, only reproducible aggregate metrics published to owning project.
"""
from __future__ import annotations
from pathlib import Path
import collections,csv,hashlib,json,statistics
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
VAR=Path('/data/var/perceptual-independent-20261024')
ROOT=Path(__file__).resolve().parents[2]/'docs/research'
OUT=ROOT/'perceptual-asset-and-text-holdout-uncertainty-2026-10-25.json'
SEED=20261025
ITER=2000


def read(path):
 p=VAR/path;data=p.read_bytes()
 return json.loads(data),hashlib.sha256(data).hexdigest()


def cm(rows,pred,truth):
 tp=sum(bool(x[pred]) and bool(x[truth]) for x in rows)
 fp=sum(bool(x[pred]) and not bool(x[truth]) for x in rows)
 fn=sum(not bool(x[pred]) and bool(x[truth]) for x in rows)
 tn=sum(not bool(x[pred]) and not bool(x[truth]) for x in rows)
 return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'precision':tp/(tp+fp) if tp+fp else None,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None}


def clustered(rows,group,metrics,iterations=ITER,seed=SEED):
 groups=collections.defaultdict(list)
 for r in rows:groups[r[group]].append(r)
 sites=sorted(groups)
 if len(sites)<5:raise ValueError('too few source groups to bootstrap')
 rng=np.random.default_rng(seed);sampled={key:[] for key in metrics}
 for _ in range(iterations):
  draws=rng.choice(sites,size=len(sites),replace=True)
  data=[x for s in draws for x in groups[s]]
  for key,f in metrics.items():
   try:val=f(data)
   except ValueError:continue
   if val is not None and np.isfinite(val):sampled[key].append(float(val))
 return {'clusters':len(sites),'iterations':iterations,'seed':seed,'intervals':{k:{'low':round(float(np.quantile(v,.025)),4),'high':round(float(np.quantile(v,.975)),4),'successful_replicates':len(v)} if len(v)>=1000 else None for k,v in sampled.items()}}


def industrial():
 raw,hraw=read('3d-defectbench/holdout119_goldenglb_geometry_frozen_20261025.json')
 scored,hsc=read('3d-defectbench/holdout119_glb_geometry_vs_human_20261025.json')
 src,hm=read('3d-defectbench/all129_glb_source_manifest_20261025.json')
 assert raw['status']=='complete' and len(raw['cases'])==119 and scored['successfully_inspected_n']==119
 source_rows=list(csv.DictReader((VAR/'3d-defectbench/golden_labels.csv').open()))
 lookup={r['object_id']:r for r in source_rows}
 records=[]
 for ident,x in raw['cases'].items():
  assert x['status']=='completed' and not src['labels'][ident]['previously_selected_before_this_holdout']
  m=x['features'];flags=m['detector_flags'];r=src['labels'][ident]
  records.append({'id':ident,'prompt':lookup[ident]['prompt'],'generator':lookup[ident]['model_version'],'missing':r['gold_missing_parts_majority']==1,'agreement':r['gold_missing_parts_agreement'],'overlap':bool(flags['coarse_aabb_overlap']),'open':bool(flags['any_open_or_nonmanifold_submesh']),'degenerate':bool(flags['degenerate_triangle_faces']),'badcoords':bool(flags['nonfinite_vertex_coordinates']),'AABB_frac':m['max_aabb_overlap_fraction_of_smaller_part']})
 confusion=cm(records,'overlap','missing')
 assert confusion=={'TP':9,'FP':20,'FN':24,'TN':66,'recall':9/33,'specificity':66/86,'precision':9/29,'balanced_accuracy':(9/33+66/86)/2}
 auc=float(roc_auc_score([r['missing'] for r in records],[r['AABB_frac'] for r in records]))
 def metric_conf(key):return lambda r:cm(r,'overlap','missing')[key]
 unc=clustered(records,'prompt',{'recall':metric_conf('recall'),'specificity':metric_conf('specificity'),'precision':metric_conf('precision'),'AABB_AUROC':lambda r:float(roc_auc_score([x['missing'] for x in r],[x['AABB_frac'] for x in r]))})
 # Source expert crosswalk: only UNIQUE prompt+model aliases. Do not map dragon
 # or guitar duplicate names to arbitrary individual GLBs, even if labels match.
 import pandas as pd
 expert_path=VAR/'3d-defectbench-2026-paper/golden_adjudicated_truth.csv'
 experts=pd.read_csv(expert_path).to_dict('records')
 groups=collections.defaultdict(list)
 for r in source_rows:groups[(r['prompt'],'Gen-'+r['model_version'][-1])].append(r)
 paired=[];ambiguous=[];unmatched=[]
 for ex in experts:
  key=(ex['prompt'],ex['model']);candidates=groups[key]
  if len(candidates)==1:paired.append((candidates[0],ex))
  elif len(candidates)>1:ambiguous.append({'source_prompt':ex['prompt'],'generator':ex['model'],'possible_source_ids':[c['object_id'] for c in candidates]})
  else:unmatched.append({'prompt_sha256':hashlib.sha256(ex['prompt'].encode()).hexdigest(),'generator':ex['model']})
 assert len(paired)==97 and len(ambiguous)==2 and len(unmatched)==1
 agreement={}
 for golden,expert in [('q_missing_parts_majority_vote','Incorrect Prompt Quantity/Missing Parts'),('q_fused_incomplete_majority_vote','Fused or Incomplete Parts'),('q_form_surface_majority_vote','Form and Surface Quality Issues')]:
  agreement[expert]={'matched_unambiguously':len(paired),'agreement':sum(int(x[golden])==int(y[expert]) for x,y in paired)}
  assert agreement[expert]['agreement']==97
 return {'evidence_SHA256':{'full129_manifest':hm,'frozen_119_GLBS':hraw,'heldout_119_scoring':hsc,'expert_adjudicated_reference_SHA256':hashlib.sha256(expert_path.read_bytes()).hexdigest()},'source_claims':'129 golden labeled original generated GLBs, 10 used in prior geometry pilot; 119 heldout original geometry run, 33 human missing-parts majority, 86 not missing; same preset mesh numerical flags','source_holdout_n':len(records),'label_missing_positives':sum(x['missing'] for x in records),'bad_coords_flag_n':sum(x['badcoords'] for x in records),'open_mesh_flag_n':sum(x['open'] for x in records),'degenerate_face_flag_n':sum(x['degenerate'] for x in records),'AABB_broadphase_missingparts_confusion':confusion,'AABB_overlap_fraction_AUROC':auc,'prompt_cluster_bootstrap':unc,'rater_agreement_subsets':{str(agreement):{'n':len(sub),'missing_positives':sum(x['missing'] for x in sub),'AABB_counts':cm(sub,'overlap','missing')} for agreement in sorted({x['agreement'] for x in records}) if (sub:=[x for x in records if x['agreement']==agreement])},'expert_crosswalk':{'unambiguous_n':len(paired),'ambiguous_prompt_generator_n':len(ambiguous),'not_found_n':len(unmatched),'full_agreement_on_unique_keys':agreement,'ambiguous':ambiguous,'not_found':unmatched,'note':'Earlier preliminary 98/99 claim used a dict that overwrote two duplicate prompt/generator keys, therefore was overly broad. Only 97 unambiguous matches support a source-to-expert agreement statement. Do NOT merge duplicated objects by guessed identity.'},'limitations':'Released generated GLBs rather than industrial CAD metrology, benchmark human majority from two annotators with conflicts, generated object prompts repeat, classes include nonmechanical props. Open mesh/AABB flags are not intended to detect semantic missing parts, and aabb overlaps are not narrowphase actual collisions.'}


def uiocr():
 raw,hraw=read('scry311/paired_ocr_psm11_20261025_frozen_predictions.json')
 src,hsc=read('scry311/paired_ocr_psm11_20261025_human_issue_scores.json')
 old,hss=read('scry311/scry_cv_pixel_only_predictions.json')
 assert raw['status']=='complete' and len(raw['frozen_pair_predictions'])==src['selected_source_pairs']==132
 assert src['complete_ocr_pairs']==132
 assert src['calibration_selected']['threshold']==.3 and src['calibration_selected']['metric']=='ref_token_missing_fraction'
 from pathlib import Path
 labels={}
 for p in sorted((VAR/'scry311').glob('test-*-of-00016.parquet')):
  for item in pq.read_table(p,columns=['id','app_slug','issue_labels']).to_pylist():labels[item['id']]=item
 assert len(labels)==311
 byid={r['id']:r for r in old['predictions']};assert len(byid)==311
 records=[]
 for id,x in raw['frozen_pair_predictions'].items():
  assert x['status']=='completed'
  label=labels[id];assert label['app_slug']==x['app_slug']
  fold='calibration' if int(hashlib.sha256(x['app_slug'].encode()).hexdigest()[:8],16)%4==0 else 'evaluation'
  features=x['features'];assert features['ref_token_missing_fraction']>=0
  records.append({'id':hashlib.sha256(id.encode()).hexdigest(),'app':x['app_slug'],'fold':fold,'human_typography':('Typography' in label['issue_labels']),'human_missing_content':('Missing Content' in label['issue_labels']),'text_issue':any(y in ('Typography','Missing Content') for y in label['issue_labels']),'pred_text_issue':features['ref_token_missing_fraction']>.3,'text_loss':features['ref_token_missing_fraction'],'jaccard':features['1_minus_bag_Jaccard'],'ssim_delta':byid[id]['ssim_difference']})
 cal=[r for r in records if r['fold']=='calibration'];test=[r for r in records if r['fold']=='evaluation']
 assert len(cal)==35 and len(test)==97
 c=cm(test,'pred_text_issue','text_issue')
 assert (c['TP'],c['FP'],c['FN'],c['TN'])==(21,34,7,35)
 auc_text=float(roc_auc_score([r['text_issue'] for r in test],[r['text_loss'] for r in test]))
 auc_ssim=float(roc_auc_score([r['text_issue'] for r in test],[r['ssim_delta'] for r in test]))
 unc=clustered(test,'app',{'recall':lambda r:cm(r,'pred_text_issue','text_issue')['recall'],'specificity_against_unlabeled_text':lambda r:cm(r,'pred_text_issue','text_issue')['specificity'],'precision_vs_annotations':lambda r:cm(r,'pred_text_issue','text_issue')['precision'],'OCR_AUROC':lambda r:float(roc_auc_score([x['text_issue'] for x in r],[x['text_loss'] for x in r])),'SSIM_AUROC':lambda r:float(roc_auc_score([x['text_issue'] for x in r],[x['ssim_delta'] for x in r])),'paired_AUROC_delta_OCR_minus_SSIM':lambda r:float(roc_auc_score([x['text_issue'] for x in r],[x['text_loss'] for x in r])-roc_auc_score([x['text_issue'] for x in r],[x['ssim_delta'] for x in r]))},seed=SEED+1)
 return {'source_SHA256':{'frozen_Tesseract_pair_scores':hraw,'human_issue_scoring':hsc,'frozen_old_SSIM_for_same_cases':hss},'total_sample':132,'calibration_n':35,'app_disjoint_eval_n':97,'calibration_app_n':len({r['app'] for r in cal}),'evaluation_app_n':len({r['app'] for r in test}),'human_typography_n_in_eval':sum(r['human_typography'] for r in test),'human_missing_content_n_in_eval':sum(r['human_missing_content'] for r in test),'human_either_text_tag_n_in_eval':sum(r['text_issue'] for r in test),'calibration_selected_OCR_token_loss_threshold':.3,'heldout_ocr_missing_token_confusion':c,'heldout_ocr_auroc':auc_text,'same_source_heldout_SSIM_auroc':auc_ssim,'app_grouped_bootstrap':unc,'limitations':'These are actual paired human-reviewed app screenshots but positive labels for typography/missing content are neither exhaustive OCR-loss labels nor necessarily text-specific: some missing content is icons. Text changes due intentional design and OCR errors may be false alarms against available tags. Split app group was reused from previous Scry benchmark, so not untouched final evaluation. Calibration threshold .3 chosen on 35 cases and tested on other app groups only.'}


def main():
 if OUT.exists():print('EXISTING_AGGREGATE_NOT_REWRITTEN');return
 a=industrial();b=uiocr()
 result={'date':'2026-10-25','topic':'independent external human labels versus frozen numeric 3D source checks and paired image OCR, with source grouped uncertainty','real_GLBS':a,'real_paired_mobile_screenshots':b,'external_media_stays_local':True,'no_new_inference_or_untouched_holdout_claim':True,'interpretation':'AABB overlap and generic watertightness cannot detect missing semantic 3D parts (119 original human labeled holdout). Paired OCR word loss has modest text-tag ranking beyond SSIM (97 other-app screens), but screen rejection false alarm proxy extremely high. No general visual acceptance judge qualified.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('SOURCE_GROUNDED_UNCERTAINTY_SAVED',OUT,'BYTES',OUT.stat().st_size)
 print('REAL_GLB',a['AABB_broadphase_missingparts_confusion'],'AUC',a['AABB_overlap_fraction_AUROC'],'CIs',a['prompt_cluster_bootstrap']['intervals'])
 print('OCR',b['heldout_ocr_missing_token_confusion'],'AUROC',b['heldout_ocr_auroc'],'vs_SSIM',b['same_source_heldout_SSIM_auroc'],'CIs',b['app_grouped_bootstrap']['intervals'])
 print('LABEL_CROSSWALK',a['expert_crosswalk']['unambiguous_n'],a['expert_crosswalk']['full_agreement_on_unique_keys'])
if __name__=='__main__':main()
