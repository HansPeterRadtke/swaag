#!/usr/bin/env python3
"""Compare actual source labels and frozen model outputs under deployment choices.

Human positives/category-matched controls are FROM PREVIOUSLY COMPLETED
source-fixed 343 cases. New triple-category unanimous-negative cases are
72 disjoint original PNGs. NO rerunning models or selecting thresholds.
Human review time unmeasured, and no source geometry gets automatic authority
for appearance properties unless verified against final state + contract.
"""
from pathlib import Path
import collections
import hashlib
import json
import random
import statistics

RAW=Path('/data/var/perceptual-independent-20261024')
NEW=RAW/'slideaudit_unseen_triple_neg_20261026'
OUTPUT=NEW/'deployment_policy_tradeoffs.json'
T=("Content Overflow/Cut-off", "Occluded Content", "Insufficient Color Contrast for Readability")
RULES={
 'OCR_only':('OCR',),
 'CLIP_only':('CLIP',),
 'geometry_only':('GEO',),
 'OCR_or_CLIP':('OCR','CLIP'),
 'OCR_or_geometry':('OCR','GEO'),
 'CLIP_or_geometry':('CLIP','GEO'),
 'all_three_union':('OCR','CLIP','GEO'),
 'null_no_proposal':(),
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get(path):return json.loads(path.read_text())
def frozen_old():
 sel=get(RAW/'slideaudit_selection_20261025.json')
 cv=get(RAW/'slideaudit_343_image_only_ocr_cv_frozen_20261025.json')['input_predictions']
 clip=get(RAW/'slideaudit_343_clip_image_only_20261025.json')['predictions']
 geo=get(RAW/'slideaudit_343_source_geometry_frozen_20261025.json')['features_only_NO_source_elements_or_text']
 assert len(sel['items'])==len(cv)==len(clip)==len(geo)==343
 features={}
 for sid in sel['items']:
  ident=f'slide_{int(sid):04d}';x=cv[ident];cl=clip[ident];g=geo[ident]['source_element_geometry']
  assert x['source_image_sha256']==cl['source_sha256']
  features[int(sid)]={
   'OCR':{T[0]:x['predicted_clipped_text_near_canvas_boundary'],T[1]:x['predicted_ocr_box_overlap'],T[2]:x['predicted_low_contrast']},
   'CLIP':{k:cl['positive_prompt_probability'][k]>.5 for k in T},
   'GEO':{T[0]:g['source_predicted_overflow'],T[1]:g['source_predicted_any_element_overlap'],T[2]:None},
   'status':x['status'],
   'source_stratum':sel['items'][sid]['source_type']+'|'+sel['items'][sid]['alteration'],
  }
 return sel,features

def flags(row,rule,category):
 methods=RULES[rule]
 if not methods:return False
 candidates=[row[m][category] for m in methods]
 if all(v is None for v in candidates):return None
 return any(v for v in candidates if v is not None)

def cmp(gold,vals):
 yes=[p for y,p in zip(gold,vals) if y and p is not None]
 no=[p for y,p in zip(gold,vals) if not y and p is not None]
 return {'TP':sum(yes),'FN':len(yes)-sum(yes),'FP':sum(no),'TN':len(no)-sum(no),'abstained':sum(p is None for p in vals)}

def boot_rate(rows,rule,B=4000):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['stratum']].append(r)
 strata=sorted(groups)
 assert len(strata)==12 and all(len(groups[k])==6 for k in strata)
 rng=random.Random(20261026)
 stats=[]
 for i in range(B):
  chosen=[rng.choice(groups[k]) for k in strata for _ in range(len(groups[k]))]
  stats.append(sum(r['proposals'][rule]['any'] for r in chosen)/len(chosen))
 stats.sort()
 return [round(stats[int(.025*B)],4),round(stats[int(.975*B)],4)]

def main():
 if OUTPUT.exists():raise RuntimeError('Do not rerun already scored deployment comparison')
 sel,old=frozen_old()
 selection=get(NEW/'selection.json');newscore=get(NEW/'scored.json')
 assert newscore['predeclared_selection_SHA256']==sha(NEW/'selection.json')
 cases=[]
 for item in selection['selected']:
  sid=item['id'];p=next(x for x in newscore['cases'] if x['slide_id']==sid)
  assert p['verified_unanimous_negative'] and p['source_SHA']
  row={'OCR':p['cv'],'CLIP':p['CLIP'],'GEO':{**p['source_geometry'],T[2]:None}}
  out={rule:{'properties':{k:flags(row,rule,k) for k in T},'any':any(flags(row,rule,k) is True for k in T)} for rule in RULES}
  cases.append({'id':sid,'stratum':item['stratum'],'proposals':out})
 assert len(cases)==72
 screened={rule:sum(x['proposals'][rule]['any'] for x in cases) for rule in RULES}
 by_label={k:{rule:(None if all(x['proposals'][rule]['properties'][k] is None for x in cases) else sum(x['proposals'][rule]['properties'][k] is True for x in cases)) for rule in RULES} for k in T}
 prior={}
 for category in T:
  task=sel['task_counts'][category]
  ids=task['selected_positive_ids']+task['selected_negative_ids']
  gold=[True]*len(task['selected_positive_ids'])+[False]*len(task['selected_negative_ids'])
  assert len(ids)==len(gold)
  row={}
  for method in RULES:
   vals=[flags(old[sid],method,category) for sid in ids]
   row[method]=cmp(gold,vals)
  prior[category]=row
 assert prior[T[0]]['OCR_only']=={'TP':58,'FN':24,'FP':26,'TN':56,'abstained':0}
 assert prior[T[1]]['OCR_only']=={'TP':24,'FN':81,'FP':12,'TN':93,'abstained':0}
 assert prior[T[2]]['OCR_only']=={'TP':8,'FN':12,'FP':0,'TN':20,'abstained':0}
 assert prior[T[0]]['CLIP_only']['TP']==52 and prior[T[0]]['CLIP_only']['FP']==62
 assert prior[T[1]]['CLIP_only']['TP']==74 and prior[T[1]]['CLIP_only']['FP']==78
 assert prior[T[2]]['CLIP_only']['TP']==12 and prior[T[2]]['CLIP_only']['FP']==16
 cost=newscore['cost']
 sample_count=72
 timings={
 'OCR_only':cost['pixel_OCR_CV_total_wall_s'],
 'CLIP_only':cost['CLIP_total_wall_s'],
 'geometry_only':cost['source_geometry_total_wall_s'],
 'OCR_or_CLIP':cost['pixel_OCR_CV_total_wall_s']+cost['CLIP_total_wall_s'],
 'OCR_or_geometry':cost['pixel_OCR_CV_total_wall_s']+cost['source_geometry_total_wall_s'],
 'CLIP_or_geometry':cost['CLIP_total_wall_s']+cost['source_geometry_total_wall_s'],
 'all_three_union':cost['pixel_OCR_CV_total_wall_s']+cost['CLIP_total_wall_s']+cost['source_geometry_total_wall_s'],
 'null_no_proposal':0.0,
 }
 for k,v in timings.items():assert v>=0
 result={
 'date':'2026-10-26',
 'purpose':'deployment triage tradeoffs, NOT a trained classifier',
 'source_human_gold':'UIST 2025 SlideAudit, unanimous 3/3 for negative on all three target categories; previous all strong positives and category-specific strong negative controls, 600 original/1800 altered slides',
 'source_authority':'Strong agreement means all 3 annotators agree; source object rectangles and model probabilities are not identical to human visual experience. Source metadata may predate visual alterations.',
 'source_revision':'642d490b7c1d2e78a50a631bfd359433397f3ecf',
 'frozen_source_SHA':{k:sha(p) for k,p in {'new_gold':NEW/'selection.json','new_cv':NEW/'pixel_features.json','new_clip':NEW/'clip_features.json','new_source_geometry':NEW/'source_features.json','old_gold':RAW/'slideaudit_selection_20261025.json','old_cv':RAW/'slideaudit_343_image_only_ocr_cv_frozen_20261025.json','old_clip':RAW/'slideaudit_343_clip_image_only_20261025.json','old_geometry':RAW/'slideaudit_343_source_geometry_frozen_20261025.json'}.items()},
 'heldout_72_unanimous_negative_per_category':by_label,
 'heldout_72_unanimous_negative_screen_any_alerts':screened,
 'heldout_72_strata_bootstrap_95pct_false_proposal_rate':{rule:boot_rate(cases,rule) for rule in RULES},
 'old_343_human_positive_and_matched_negative_by_task':prior,
 'computation':{'measured_batch_wall_72_by_rule_s':{k:round(v,3) for k,v in timings.items()},'measured_average_cpu_wall_per_image_by_rule_s':{k:round(v/sample_count,4) for k,v in timings.items()},'parallel_or_shared_compute_not_measured':True,'download_time_excluded':cost['download_wall_s'],'human_review_time_unmeasured':True,'CLIP_cold_load_wall_s':cost['CLIP_load_wall_s'],'GPU_used':False},
 'qualified_auto_fail':'No source metadata, OCR pixels or generic VLM positive becomes a verified human-visible defect automatically. Only exact property with source/final-state/contract verification may fail.',
 'policy_comparison':{
  'automated_detector_flags_are_final_failures':'DISALLOWED by independently rated false positive and negative counts; would wrongly reject at least new unseen 72 flagged screens as if known defects',
  'proposal_only_human_review':'Use one or more detectors only to prioritize examples for human review; previous positive cases unflagged would be missed if non-proposals are skipped',
  'human_review_all_ambiguous':'72/72 heldout human-negative would be reviewed, plus source positives in other holdout; quality depends on qualified reviewer not measured and no automatic time estimate possible',
  'source_geometry_verified_property':'Allow auto-fail only when actual final sourced geometry, intent, coordinate scale, correct source version and property are all verified; SlideAudit source boxes do not certify visible occlusion.'
 },
 'cases':cases,
 'warnings':['New 72 unanimous triple-negative image set is not an independent institution or a native mobile app, but previously UNUSED instances of same published slide corpus', 'Old positives/category-negatives were balanced and all positive categories selected, so cannot infer actual deployment prevalence or cross-domain precision', 'Combined OR rules are POSTHOC cost/error diagnostics on same frozen predictions, not individually preregistered new winners', 'Zero flags in one category means finite uncertainty; domain/source correlation invalidates naive iid confidence guarantees', 'Source geometry is not image perception; source rectangles may not reflect final altered pixels', 'Human review labor time and real owner audio/3D/perceptual quality have not been measured.']
 }
 OUTPUT.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 print('DEPLOYMENT_COMPARISON_SAVED',OUTPUT)
 for task,val in prior.items():print('HUMAN_POS+CONTROL',task,[(k,v) for k,v in val.items() if k in ('OCR_only','CLIP_only','geometry_only','OCR_or_CLIP','OCR_or_geometry','all_three_union')])
 print('NEW_UNANIMOUS_NEGATIVE',screened)
 print('COST_PER_SCREEN',result['computation']['measured_average_cpu_wall_per_image_by_rule_s'])

if __name__=='__main__':main()
