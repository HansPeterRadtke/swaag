#!/usr/bin/env python3
"""Replay five SAME-protocol VLM predictions vs expert-adjudicated 3D gold.

Separate from silver majority results, avoids treating noisy crowd labels as
unanimous truth. Published VLM outputs (NOT newly inferred locally) joined
by immutable source asset URL and exact defect category. Independent heldout
human annotations: 99 of 100 expert assets have this common-protocol result.
"""
from pathlib import Path
import json,hashlib,pandas as pd,numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import matthews_corrcoef
ROOT=Path('/data/var/perceptual-independent-20261024/3d-defectbench-2026-paper')
OUT=ROOT/'five_vlms_expert_adjudicated_replay.json'
RUNS={'claude-sonnet-4':'silver_cs4_binary','gemini-2.5-pro':'silver_g25p_binary','gpt-4.1':'silver_gpt41_binary','gpt-5':'silver_gpt5_binary','qwen3-vl-235b':'silver_qwen3_binary'}
GEOM=['Incorrect Prompt Quantity/Missing Parts','Incorrect Prompt Pose/Placement','Fused or Incomplete Parts','Unrecognizable Extra Geometry','Form and Surface Quality Issues']
TEXT=['Noisy/Blurry/Graniness Texture','Misplaced/Overlapping texture','Baked in Lighting/Shadow','Incorrect Prompt Visual/Textual Attributes']

def summarize(a):
 g=a.expert_adjudicated.to_numpy(dtype=np.int8);p=a.vlm_pred.to_numpy(dtype=np.int8)
 tp=int(((g==1)&(p==1)).sum());fp=int(((g==0)&(p==1)).sum());fn=int(((g==1)&(p==0)).sum());tn=int(((g==0)&(p==0)).sum())
 return {'n':len(a),'positive_gold':tp+fn,'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'specificity':tn/(fp+tn) if fp+tn else None,'mcc':float(matthews_corrcoef(g,p)) if len(set(g))>1 and len(set(p))>1 else None}

def main():
 if OUT.exists():print('PREVIOUS_SCORES_NOT_REPLAYED');return
 gold=pd.read_csv(ROOT/'golden_adjudicated_truth.csv',dtype={'url':str})
 manifest=pd.read_csv(ROOT/'asset_manifest.csv')
 assert gold.url.nunique()==100 and all(k in gold for k in GEOM+TEXT)
 long=gold.melt(id_vars=['url','prompt','model'],value_vars=GEOM+TEXT,var_name='defect',value_name='expert_adjudicated')
 assert len(long)==900 and not long.duplicated(['url','defect']).any()
 cols=['run_folder','model_family','url','defect','silver_majority','vlm_pred','view_count','seed','strategy','reasoning_mode','texture_present','view_layout']
 allpred=pq.read_table(ROOT/'vlm_cell_predictions.parquet',columns=cols).to_pandas()
 results={'source':'3D-DefectBench pinned latest revision, 100 expert-adjudicated golden assets vs author-recorded 8-view VLM predictions','model_predictions_source_sha256':hashlib.sha256((ROOT/'vlm_cell_predictions.parquet').read_bytes()).hexdigest(),'golden_source_sha256':hashlib.sha256((ROOT/'golden_adjudicated_truth.csv').read_bytes()).hexdigest(),'gold_assets_total':100,'models':{},'critical_note':'Predictions were downloaded, NOT run on Nitro or Thor. Model/checkpoint versions and viewing protocols are historic author experiments; not automatically transferable to current systems or owner machines.'}
 ref_ids=None
 for model,run in RUNS.items():
  raw=allpred[(allpred.model_family==model)&(allpred.run_folder==run)]
  assert len(raw)==8982 and not raw.duplicated(['url','defect']).any()
  df=raw.merge(long[['url','defect','expert_adjudicated']],on=['url','defect'],how='inner',validate='many_to_one')
  assets=sorted(df.url.unique())
  assert len(assets)>=98,(model,len(assets))
  if ref_ids is None:ref_ids=assets
  assert ref_ids==assets,'different reference assets per model'
  per={k:summarize(df[df.defect==k]) for k in GEOM+TEXT}
  geom_mcc=[per[k]['mcc'] for k in GEOM if per[k]['mcc'] is not None]
  tex_mcc=[per[k]['mcc'] for k in TEXT if per[k]['mcc'] is not None]
  # This fixed 'missing parts' diagnosis is directly relevant to machines.
  mechanical_urls=manifest.loc[manifest.category.isin(['Land Vehicles','Air Vehicles','Robots','Water Vehicles','Tools','Technology & Electronics']),'url']
  missing=df[(df.defect==GEOM[0])&(df.url.isin(mechanical_urls))]
  results['models'][model]={'run_folder':run,'expert_assets':len(assets),'human_labeled_cells':len(df),'geometry':{'macro_mcc':sum(geom_mcc)/len(geom_mcc) if geom_mcc else None,'per_defect':{k:per[k] for k in GEOM}},'texture':{'macro_mcc':sum(tex_mcc)/len(tex_mcc) if tex_mcc else None,'per_defect':{k:per[k] for k in TEXT}},'mechanical_missing_parts':summarize(missing)}
  print('MODEL',model,'EXPERT_ASSETS',len(assets),'MISSING_ALL',per[GEOM[0]],'MECH_MISSING',results['models'][model]['mechanical_missing_parts'],'GEOM_MCC',results['models'][model]['geometry']['macro_mcc'],flush=True)
 results['expert_assets_common']=len(ref_ids)
 results['limitations']='100-asset expert set only, 99 common predictions, some defect categories rare (pose, extra geometry), macro MCC comparisons fragile. Expert agreement adjudicated by authors, not local re-review. Mechanical subset may have very few true positives. Public release, potential VLM training exposure.'
 OUT.write_text(json.dumps(results,indent=2)+'\n')
 print('SAVED',OUT,'BYTES',OUT.stat().st_size)
if __name__=='__main__':main()
