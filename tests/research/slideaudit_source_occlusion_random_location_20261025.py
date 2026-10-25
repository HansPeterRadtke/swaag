#!/usr/bin/env python3
"""Validate source-geometry localized occlusion proposals vs human region chance.

104 strongly annotated occlusion-positive original SlideAudit images provide
human bounding boxes, all unavailable to source-layout proposal generator.
A separate frozen 343-slide source-layout output proposes up to 3 rectangle
intersection centers, ranked by overlap relative to smaller source box.

Compare actual candidate-in-human-box hits to 10,000 independently drawn sets
of the SAME candidate count uniformly placed across original slide area.
Source human broad bbox area can itself yield huge chance hit probability:
chance baseline is essential. Site-family pooling is not independent; use
per-slide bootstrap for descriptive range, no claim deployment prediction.
No model/image relaunch, no threshold tuning, only archived source numbers.
"""
from pathlib import Path
import json,hashlib,collections,math
import numpy as np
ROOT=Path('/data/var/perceptual-independent-20261024')
SEL=ROOT/'slideaudit_selection_20261025.json'
PRO=ROOT/'slideaudit_343_layout_intersection_proposals_frozen_20261025.json'
OUT=ROOT/'slideaudit_occlusion_source_localization_random_baseline_20261025.json'
N=10000;SEED=20261025


def main():
 if OUT.exists():print('PREVIOUS_SOURCE_OCCLUSION_CHANCE_EVAL_PRESERVED');return
 s=json.loads(SEL.read_text());a=json.loads(PRO.read_text())
 assert a['status']=='complete' and len(a['slides'])==343
 ids=s['task_counts']['Occluded Content']['selected_positive_ids'];assert len(ids)==105
 rng=np.random.default_rng(SEED)
 hist=np.zeros(N,dtype=np.uint16);rows=[]
 for i in ids:
  doc=s['items'][str(i)];annotations=doc['labels']['Occluded Content']['boxes']
  if not annotations:continue
  W,H=doc['width'],doc['height'];regions=[(max(0,b['x']/W),max(0,b['y']/H),min(1,(b['x']+b['width'])/W),min(1,(b['y']+b['height'])/H)) for b in annotations]
  points=[r['center_normalized'] for r in a['slides'][f'slide_{i:04d}']['top3']]
  def hits(x,y):
   h=np.zeros(np.broadcast_shapes(np.shape(x),np.shape(y)),dtype=bool)
   for lo,top,hi,bottom in regions:
    h|=(x>=lo)&(x<=hi)&(y>=top)&(y<=bottom)
   return h
  true_hit=any(bool(hits(x,y)) for x,y in points)
  if points:
   xy=rng.uniform(0,1,size=(N,len(points),2))
   random_hit=np.any(hits(xy[:,:,0],xy[:,:,1]),axis=1)
  else:random_hit=np.zeros(N,dtype=bool)
  hist+=random_hit.astype(np.uint16)
  rows.append({'id_sha256':hashlib.sha256(str(i).encode()).hexdigest(),'generator_source':doc['source_type'],'alteration':doc['alteration'],'source_candidates':len(points),'original_human_regions':len(regions),'actual_source_candidate_hits_any':true_hit,'random_uniform_center_hit_rate_for_same_candidate_count':float(np.mean(random_hit))})
 assert len(rows)==104
 observed=sum(r['actual_source_candidate_hits_any'] for r in rows)
 boot_rng=np.random.default_rng(SEED+1)
 observed_matrix=np.array([int(r['actual_source_candidate_hits_any']) for r in rows])
 null_expect=np.array([r['random_uniform_center_hit_rate_for_same_candidate_count'] for r in rows])
 sample=boot_rng.integers(0,len(rows),size=(5000,len(rows)),dtype=np.int16)
 difference=(observed_matrix[sample]-null_expect[sample]).mean(axis=1)
 cats={k:{'n':sum(r['generator_source']==k for r in rows),'hits':sum(r['actual_source_candidate_hits_any'] for r in rows if r['generator_source']==k),'random_expected_hits':sum(r['random_uniform_center_hit_rate_for_same_candidate_count'] for r in rows if r['generator_source']==k)} for k in ['Gemini','gdcdataset','Google']}
 stats={'n_occlusion_human_positive_slides_with_bbox':len(rows),'source_localization_hits':observed,'source_proposal_empty':sum(not r['source_candidates'] for r in rows),'hit_fraction':observed/len(rows),'expected_random_location_hits':float(np.mean(hist)),'random_simulation_95pct_hit_count_interval':[int(np.quantile(hist,.025)),int(np.quantile(hist,.975))],'num_random_trials':N,'seed':SEED,'number_random_trials_ge_observed':int((hist>=observed).sum()),'empirical_one_sided_probability_with_plus_one_correction':float((1+np.sum(hist>=observed))/(N+1)),'observed_minus_random_expected_hit_fraction':float((observed-np.mean(hist))/len(rows)),'per_slide_bootstrap_95pct_delta_hit_fraction':[float(np.quantile(difference,q)) for q in (.025,.975)],'human_categories_by_original_slide_source':cats}
 result={'source':'SlideAudit 2025 CC-BY4 original real/altered slide human bounding-box occlusion annotations','source_human_selection_sha256':hashlib.sha256(SEL.read_bytes()).hexdigest(),'source_generated_intersection_proposals_sha256':hashlib.sha256(PRO.read_bytes()).hexdigest(),'fixed_generation':'Original normalized authored object rectangle intersections >=10% of smaller box; up to 3 centers independently predicted without labels, 104 annotated positive source slides','random_baseline':'10k sets of uniformly random 2D points in the slide at the same per-slide proposal count, testing whether ANY point is inside ANY human bbox, inclusive boundaries','results':stats,'source_case_references':rows,'caveats':'Human positive occlusion bboxes may cover broad slide area; source layout positions may reflect later visual alteration. Correct candidate-center inside human-region shows localization association but not that detected source overlap is actually visible, obstructive, unintended, or complete. No false-positive localization on human negative objects, only earlier source classification FP separately scored. Per-slide bootstrap treats different slide IDs as independent despite shared source/alteration; descriptive only.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('GENUINE_OCCLUSION_SOURCE_SPATIAL_RANDOM_COMPARISON',stats,'LOCAL_RESULTS',OUT)
if __name__=='__main__':main()
