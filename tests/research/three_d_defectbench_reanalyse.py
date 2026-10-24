#!/usr/bin/env python3
"""Independent confusion-table replay of published VLM predictions on human labels.

The VLM generations are released by 3D-DefectBench authors, NOT executed here.
Select five identical baseline protocols: 998 silver assets x 9 human-labelled
defects per run. Don't compare author 2026 Gemini 3 pro aggregate run that
mixes multiple experimental configurations and duplicate asset/defect rows.
All raw licensed data remain outside Git; aggregate counts and source hashes only.
"""
from pathlib import Path
import hashlib,json,collections,math,time
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from sklearn.metrics import matthews_corrcoef,balanced_accuracy_score,precision_recall_fscore_support

ROOT=Path('/data/var/perceptual-independent-20261024/3d-defectbench-2026-paper')
OUTPUT=ROOT/'independent_vlm_baseline_replay.json'
CONFIG={
 'claude-sonnet-4':'silver_cs4_binary',
 'gemini-2.5-pro':'silver_g25p_binary',
 'gpt-4.1':'silver_gpt41_binary',
 'gpt-5':'silver_gpt5_binary',
 'qwen3-vl-235b':'silver_qwen3_binary',
}
GEOMETRY=('Incorrect Prompt Quantity/Missing Parts','Incorrect Prompt Pose/Placement','Fused or Incomplete Parts','Unrecognizable Extra Geometry','Form and Surface Quality Issues')
TEXTURE=('Noisy/Blurry/Graniness Texture','Misplaced/Overlapping texture','Baked in Lighting/Shadow','Incorrect Prompt Visual/Textual Attributes')
MECHANICAL=('Land Vehicles','Air Vehicles','Robots','Water Vehicles','Tools','Technology & Electronics')

def cm(rows):
 truth=rows.silver_majority.to_numpy(dtype=np.int8);prediction=rows.vlm_pred.to_numpy(dtype=np.int8)
 tp=int(((truth==1)&(prediction==1)).sum());fp=int(((truth==0)&(prediction==1)).sum())
 tn=int(((truth==0)&(prediction==0)).sum());fn=int(((truth==1)&(prediction==0)).sum())
 return {'n':len(rows),'TP':tp,'FP':fp,'TN':tn,'FN':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'mcc':float(matthews_corrcoef(truth,prediction)) if len(set(truth))>1 and len(set(prediction))>1 else None}

def score_group(m):
 sub={}
 for aspect,cats in [('geometry',GEOMETRY),('texture',TEXTURE)]:
  a=m[m.defect.isin(cats)]
  per={name:cm(a[a.defect==name]) for name in cats}
  vals=[v['mcc'] for v in per.values() if v['mcc'] is not None]
  sub[aspect]={'aggregate_confusion':cm(a),'macro_mcc_mean_nonconstant':sum(vals)/len(vals) if vals else None,'per_defect':per}
 sub['physical_mechanical_categories']={'all_mechanical_cells':cm(m[m.prompt_category.isin(MECHANICAL)]),'mechanical_missing_parts':cm(m[(m.prompt_category.isin(MECHANICAL))&(m.defect==GEOMETRY[0])]),'mechanical_fused_parts':cm(m[(m.prompt_category.isin(MECHANICAL))&(m.defect==GEOMETRY[2])])}
 return sub

def main():
 if OUTPUT.exists():print('EXISTING_COMPLETE_RESULT_NOT_RERUN');return
 p=ROOT/'vlm_cell_predictions.parquet';source_sha=hashlib.sha256(p.read_bytes()).hexdigest()
 cols=['run_folder','model_family','strategy','reasoning_mode','seed','view_count','view_layout','texture_present','prompt_packing','output_format','url','defect','silver_majority','vlm_pred','prompt_category','agreement_rate']
 df=pq.read_table(p,columns=cols).to_pandas()
 result={'source':'aieval2026/3d-defectbench pinned model family predictions, released 2026','source_revision':json.loads((ROOT/'provenance.json').read_text())['revision'],'source_prediction_parquet_sha256':source_sha,'reference':'3D-DefectBench authors own 3 human label silver-majority assets, not locally adjudicated; published VLM predictions, not new local inference','selection_predeclared':'one specific 8-view RGB/geometry baseline binary per-aspect, single seed1 run per model; Gemini3Pro baseline excluded because asset/defect duplicated under a single run_folder; use only five clean comparable protocol groups','models':{},'protocol':{'number_assets':998,'each_defects':9,'view_count':8,'view_layout':'grid','seed':1,'reasoning_mode':'off','strategy':'binary','prompt_packing':'per_aspect','output_format':'free_text','geometry_input':'geo_only','texture_input':'with_texture'},'limitations':'Multiple defect cells per asset correlated; public benchmark human silver majority is not unanimous expert ground truth. Post-hoc mechanical-category subgroup exploratory. Model labels not stable current served versions. Real 3D human-labelled generated meshes but pipeline rendering was performed by dataset authors, not this local run.'}
 first=None
 for model,folder in CONFIG.items():
  a=df[(df.model_family==model)&(df.run_folder==folder)]
  assert len(a)==8982,(model,len(a))
  assert a.url.nunique()==998 and len(a.defect.unique())==9
  assert not a.duplicated(['url','defect']).any()
  for key,value in [('seed',1),('view_count',8),('view_layout','grid'),('strategy','binary'),('reasoning_mode','off'),('output_format','free_text'),('prompt_packing','per_aspect')]:
   assert a[key].eq(value).all(),(model,key,a[key].unique())
  assert a[a.defect.isin(GEOMETRY)].texture_present.eq('geo_only').all()
  assert a[a.defect.isin(TEXTURE)].texture_present.eq('with_texture').all()
  assert set(a.vlm_pred.unique()).issubset({0,1}) and set(a.silver_majority.unique()).issubset({0,1})
  # Ensure all five VLMs evaluated same canonical asset/defect/labels.
  signature=hashlib.sha256('\n'.join(f'{u}|{d}|{y}' for u,d,y in a[['url','defect','silver_majority']].sort_values(['url','defect']).itertuples(index=False,name=None)).encode()).hexdigest()
  if first is None:first=signature
  assert signature==first,'model rows or labels diverge'
  result['models'][model]={'run_folder':folder,'same_gold_signature':signature,'scored_cells':len(a),'human_label_agreement_stats':{'mean':float(a.agreement_rate.mean()),'median':float(a.agreement_rate.median())},'metrics':score_group(a)}
  print('MODEL',model,'GEOMETRY_MCC',result['models'][model]['metrics']['geometry']['macro_mcc_mean_nonconstant'],'TEXTURE_MCC',result['models'][model]['metrics']['texture']['macro_mcc_mean_nonconstant'],'MISSING_MECH',result['models'][model]['metrics']['physical_mechanical_categories']['mechanical_missing_parts'],flush=True)
 OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
 print('COMPLETED',OUTPUT,'MODELS',len(result['models']),'source SHA',source_sha,flush=True)
if __name__=='__main__':main()
