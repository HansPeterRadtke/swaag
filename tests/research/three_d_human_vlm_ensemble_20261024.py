#!/usr/bin/env python3
"""Predeclared fixed five-VLM disagreement policies on expert-rated real 3D meshes.

Only author-published model image/view predictions. Never submit a new VLM call.
Labels are human expert-adjudicated missing-parts and other geometry faults.
Thresholds {1,3,4,5}/5 all defined a priori, NOT chosen by scoring data.
Declares engineering failure if neither recall nor specificity adequately high.
Raw copyrighted source material stays outside Git; aggregate confusion only.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path('/data/var/perceptual-independent-20261024/3d-defectbench-2026-paper')
OUT=ROOT/'expert_5vlm_fixed_vote_ensemble_20261024.json'
RUNS={'claude-sonnet-4':'silver_cs4_binary','gemini-2.5-pro':'silver_g25p_binary','gpt-4.1':'silver_gpt41_binary','gpt-5':'silver_gpt5_binary','qwen3-vl-235b':'silver_qwen3_binary'}
GEOM=['Incorrect Prompt Quantity/Missing Parts','Incorrect Prompt Pose/Placement','Fused or Incomplete Parts','Unrecognizable Extra Geometry','Form and Surface Quality Issues']
POLICIES={'any_1_of_5':1,'majority_3_of_5':3,'supermajority_4_of_5':4,'unanimous_5_of_5':5}
MECHANICAL={'Land Vehicles','Air Vehicles','Robots','Water Vehicles','Tools','Technology & Electronics'}

def count(rows,col):
 truth=rows.expert_truth.to_numpy(dtype=np.int8);pred=rows[col].to_numpy(dtype=np.int8)
 tp=int(((truth==1)&(pred==1)).sum());fp=int(((truth==0)&(pred==1)).sum());fn=int(((truth==1)&(pred==0)).sum());tn=int(((truth==0)&(pred==0)).sum())
 return {'n':len(rows),'positive':tp+fn,'negative':tn+fp,'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'fpr':fp/(tn+fp) if tn+fp else None}

def result_frame(df):
 res={}
 for name,threshold in POLICIES.items():
  t=df.copy();t[name]=(t.vote_count>=threshold).astype(np.int8)
  res[name]=count(t,name)
 for model in RUNS:res['single_'+model]=count(df,model)
 return res

def main():
 if OUT.is_file():print('EXISTING_IMMUTABLE_RESULTS_NOT_REPLAYED');return
 labels=pd.read_csv(ROOT/'golden_adjudicated_truth.csv')
 selected=labels.melt(id_vars=['url','prompt','model'],value_vars=GEOM,var_name='defect',value_name='expert_truth')
 assert not selected.duplicated(['url','defect']).any() and len(selected)==500
 manifest=pd.read_csv(ROOT/'asset_manifest.csv')
 data=pq.read_table(ROOT/'vlm_cell_predictions.parquet',columns=['url','model_family','run_folder','defect','vlm_pred'])
 all_data=data.to_pandas()
 joint=selected[['url','prompt','defect','expert_truth']]
 for model,run in RUNS.items():
  sub=all_data[(all_data.model_family==model)&(all_data.run_folder==run)&(all_data.defect.isin(GEOM))]
  assert not sub.duplicated(['url','defect']).any()
  sub=sub[['url','defect','vlm_pred']].rename(columns={'vlm_pred':model})
  joint=joint.merge(sub,on=['url','defect'],how='inner',validate='one_to_one')
 # 99 of 100 expert assets common to five models; nine non-geometry texture labels omitted.
 assert len(joint)==99*5 and joint.url.nunique()==99
 joint['vote_count']=joint[list(RUNS)].sum(axis=1).astype(int)
 assert joint.vote_count.between(0,5).all()
 mech=set(manifest.loc[manifest.category.isin(MECHANICAL),'url'])
 slices={'all_expert_geometry_cells':joint,'expert_missing_parts_all':joint[joint.defect==GEOM[0]],'expert_mechanical_missing_parts':joint[(joint.defect==GEOM[0]) & (joint.url.isin(mech))],'expert_fused_incomplete_all':joint[joint.defect==GEOM[2]]}
 observations={k:result_frame(frame) for k,frame in slices.items()}
 disagreements=[]
 for key,frame in slices.items():
  if key not in ['expert_missing_parts_all','expert_mechanical_missing_parts']:continue
  wrong=np.sum(((frame.vote_count>=3)&(frame.expert_truth==0))|((frame.vote_count<3)&(frame.expert_truth==1)))
  disagreement=(frame.vote_count.isin([1,2,3,4])).mean()
  disagreements.append({'slice':key,'n':len(frame),'majority_errors':int(wrong),'fraction_nonunanimous':round(float(disagreement),5)})
 report={'source':'3D-DefectBench independently published expert labels and five published model-image predictions, pinned same run config as previous study','expert_sha256':hashlib.sha256((ROOT/'golden_adjudicated_truth.csv').read_bytes()).hexdigest(),'published_model_predictions_sha256':hashlib.sha256((ROOT/'vlm_cell_predictions.parquet').read_bytes()).hexdigest(),'prompt_source_categories':'fused/extra/missing/shape/pose; no temporal physics','models':RUNS,'fixed_voting_thresholds':POLICIES,'note':'No thresholds fit to gold; all four methods reported. Same 99 expert assets, 5 categories; generated 3D assets with published human defect ratings, not physically metrologized machines. Some source generators/videos might contaminate model training; not new model inference.','results':observations,'disagreement':disagreements}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 for key,data in observations.items():
  print('SLICE',key,'n',data['majority_3_of_5']['n'],'gold_positives',data['majority_3_of_5']['positive'])
  for m in POLICIES:
   x=data[m];print('  POLICY',m,'TP',x['TP'],'FP',x['FP'],'FN',x['FN'],'TN',x['TN'],'RECALL',x['recall'],'SPEC',x['specificity'])
 print('RAW_SAVED',OUT,'BYTES',OUT.stat().st_size)
if __name__=='__main__':main()
