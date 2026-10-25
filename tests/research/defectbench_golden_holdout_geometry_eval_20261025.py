#!/usr/bin/env python3
"""Verify exact frozen old topology rules on 119 unseen human-rated GLB assets.

Human 3D-DefectBench golden set 129 meshes, 10 explicitly handselected earlier
for pilot and excluded here. The 119 source-asset remainder is a holdout for the
same fixed geometry flags (nonfinite vertices, open meshes, degenerate faces,
AABB overlap fraction >.02), no tuning or human-label-based model changes.
Raw GLB topology predictions checkpoint BEFORE opening human labels for score.
This tests raw topology versus human semantic missing parts, NOT perception,
physical collision, manufacturing correctness or photorealism. CPU only.
"""
from pathlib import Path
import hashlib,json,time,datetime,resource,collections,statistics
import numpy as np
from sklearn.metrics import roc_auc_score
from real_mechanical_mesh_invariants_20261024 import inspect, FLAGS, TOL
ROOT=Path('/data/var/perceptual-independent-20261024/3d-defectbench')
MANIFEST=ROOT/'all129_glb_source_manifest_20261025.json'
RAW=ROOT/'holdout119_goldenglb_geometry_frozen_20261025.json'
SCORES=ROOT/'holdout119_glb_geometry_vs_human_20261025.json'
CONFIG={'source':'zzhao0500/3D-DefectBench golden human-majority generated 3D glTF assets, original CC BY-NC4, frozen 129 authors labels','selected': 'ALL 129 source golden objects minus exactly ten previously inspected pilot asset IDs (untouched 119)', 'prior_fixed_rules':{'TOL_relative_face_area':TOL,'aabb_overlap_fraction':.02},'flags':FLAGS,'classifier':'trimesh source-space topology invariants, geometry metadata only, no rendered visual analysis, no labels given to inspect()', 'cpu_threads':1,'evaluation':'against independent human majority label missing/fused/surface, a semantically different task, so disagreement does not imply numeric topology rule is wrong'}


def cm(rows,label,feature):
 pos=[r for r in rows if r[label]]; neg=[r for r in rows if not r[label]]
 tp=sum(r['features']['detector_flags'][feature] for r in pos)
 fp=sum(r['features']['detector_flags'][feature] for r in neg)
 fn=len(pos)-tp;tn=len(neg)-fp
 return {'n':len(rows),'gold_pos':len(pos),'gold_neg':len(neg),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':tp/len(pos) if pos else None,'specificity':tn/len(neg) if neg else None,'precision':tp/(tp+fp) if tp+fp else None}


def freeze():
 if RAW.exists():
  state=json.loads(RAW.read_text())
  if state.get('status')=='complete':print('RAW_HOLDOUT_ALREADY_COMPLETE_PRESERVED');return
 src=json.loads(MANIFEST.read_text());assert src['all_source_content_verified'] and src['total_cases']==129
 keep={i:r for i,r in src['labels'].items() if not r['previously_selected_before_this_holdout']}
 assert len(keep)==119
 state=json.loads(RAW.read_text()) if RAW.is_file() else {'date':'2026-10-25','source_manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'config':CONFIG,'status':'in_progress','cases':{},'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 assert state['config']==CONFIG and state['source_manifest_sha256']==hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
 t0=time.monotonic()
 for ident,r in sorted(keep.items(),key=lambda t:int(t[0])):
  if ident in state['cases']:continue
  p=Path(r['source']['path']);checksha=r['source']['sha256'];assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==checksha
  row={'id':ident,'input_sha256':checksha,'status':'pending_or_uncertain'}
  state['cases'][ident]=row;RAW.write_text(json.dumps(state,indent=2)+'\n')
  try:
   data=inspect(p)
   assert data['source_file_sha256']==checksha
   row.update(status='completed',features=data)
  except Exception as e:row.update(status='error_or_ambiguous',error=type(e).__name__+': '+str(e)[:190])
  RAW.write_text(json.dumps(state,indent=2)+'\n')
  if len(state['cases'])%10==0 or row['status']!='completed':
   print('REAL_GENERATED_GLB',len(state['cases']),'/',len(keep),'STATUS',row['status'],'TRIS',row.get('features',{}).get('triangle_faces'),'SOURCE_ID',ident,'SECONDS',row.get('features',{}).get('seconds'),flush=True)
 assert len(state['cases'])==119
 state['status']='complete';state['wall_seconds_this_iteration']=round(time.monotonic()-t0,2)
 state['max_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 RAW.write_text(json.dumps(state,indent=2)+'\n')
 print('FREEZE_COMPLETE',RAW,'N',len(state['cases']),'OK',sum(r['status']=='completed' for r in state['cases'].values()),'WALL_S',state['wall_seconds_this_iteration'])


def evaluate():
 if SCORES.exists():print('SCORES_ALREADY_COMPLETE_PRESERVED');return
 raw=json.loads(RAW.read_text());assert raw['status']=='complete' and json.dumps(raw['config'],sort_keys=True)==json.dumps(CONFIG,sort_keys=True)
 src=json.loads(MANIFEST.read_text())
 success=[];incomplete=[]
 for ident,r in raw['cases'].items():
  if r['status']!='completed':incomplete.append({'id':ident,'status':r['status']});continue
  gold=src['labels'][ident]
  success.append({'id':ident,'source_sha256':r['input_sha256'],'model_version':None,'rating_agreement':gold['gold_missing_parts_agreement'],'gold_missing_parts':bool(gold['gold_missing_parts_majority']),'gold_fused_incomplete':bool(gold['gold_fused_incomplete']),'gold_surface_form':bool(gold['gold_surface_form']),'features':r['features']})
 results={task:{flag:cm(success,task,flag) for flag in FLAGS} for task in ['gold_missing_parts','gold_fused_incomplete','gold_surface_form']}
 auc={}
 for label in results:
  y=np.asarray([x[label] for x in success],dtype=int)
  measure=np.asarray([x['features']['max_aabb_overlap_fraction_of_smaller_part'] for x in success],dtype=float)
  auc[label]=float(roc_auc_score(y,measure)) if len(set(y))>1 else None
 ratings={}
 for rate in sorted({x['rating_agreement'] for x in success}):
  candidates=[r for r in success if r['rating_agreement']==rate]
  ratings[str(rate)]={'n':len(candidates),'missing_gold_positive':sum(x['gold_missing_parts'] for x in candidates),'fixed_AABB_rule':cm(candidates,'gold_missing_parts','coarse_aabb_overlap')}
 t=sum(x['features']['seconds'] for x in success)
 result={'date':'2026-10-25','design':'119 previously unseen (vs 10 pilot) generated 3D-DefectBench human-majority golden GLBs, frozen topology flags, no threshold refitting','source_manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'raw_prediction_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'gold_source_SHA256':src['gold_labels_sha256'],'total_heldout_source':119,'successfully_inspected_n':len(success),'abstained_or_error':incomplete,'CPU_inspection_seconds_sum':round(t,2),'runtime_max_rss_kb':raw['max_rss_kb'],'confusion':results,'overlap_max_score_auc':auc,'human_label_agreement_subgroups':ratings,'no_calibration_or_retraining':True,'scope':'Trimesh source numeric topology, NOT photo/render vision model. Quality labels measure intended missing/fused parts, not mesh watertightness; no engineering physics oracle. AABB overlap is a very crude broadphase, not penetration test; open/nonmanifold meshes often normal for game assets.','limitations':['Non-exhaustive expert label majority (two reviewers on gold source; 0.5 agreement for conflicts)','129 prompts include animals and props, not industrial machines exclusively; holdout sampled by excluding previous pilot IDs, source model families correlated','AABB and watertightness cannot know which semantic component is absent, or whether a contact is intended','All threshold settings from prior 10 pilot rule fixed; this 119 is a within-public-repository confirmatory evaluation only, cannot prove generalization to unseen generator families','Noncommercial source files remain outside Git'],'case_metadata':[{'id_sha256':hashlib.sha256(x['id'].encode()).hexdigest(),'gold_missing_parts':x['gold_missing_parts'],'flags':x['features']['detector_flags'],'n_meshes':x['features']['mesh_nodes'],'AABB_fraction':x['features']['max_aabb_overlap_fraction_of_smaller_part']} for x in success]}
 SCORES.write_text(json.dumps(result,indent=2)+'\n')
 print('HELDOUT119_LABELLED_GLB_RESULTS','N',len(success),'MISSING',results['gold_missing_parts'],'AUC',auc,'LABEL_AGREEMENT',ratings,'TOTAL_INSPECT_S',round(t,2))
if __name__=='__main__':freeze();evaluate()
