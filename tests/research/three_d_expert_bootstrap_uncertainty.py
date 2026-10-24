#!/usr/bin/env python3
"""Grouped-by-prompt uncertainty for expert-adjudicated 3D-DefectBench.

Standalone reenactment using released VLM predictions, NOT local VLM inference.
No model selection or threshold tuning; stratified calibration not inferred.
Bootstrap resamples PROMPT identities so two generator variants stay together.
"""
import json,csv,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
ROOT=Path('/data/var/perceptual-independent-20261024/3d-defectbench-2026-paper')
OUT=ROOT/'expert_label_prompt_cluster_intervals.json'
R={'claude-sonnet-4':'silver_cs4_binary','gemini-2.5-pro':'silver_g25p_binary','gpt-4.1':'silver_gpt41_binary','gpt-5':'silver_gpt5_binary','qwen3-vl-235b':'silver_qwen3_binary'}
DEFECT='Incorrect Prompt Quantity/Missing Parts'
MECH=['Land Vehicles','Air Vehicles','Robots','Water Vehicles','Tools','Technology & Electronics']

def bootstrap(groups,seed):
 # Input per-prompt independent-ish confusion counts; multiple generator
 # variants share a prompt and should not be bootstrapped independently.
 rng=np.random.default_rng(seed)
 arr=np.asarray(groups,dtype=np.int64)
 n=len(arr);draw=rng.integers(0,n,size=(1000,n),dtype=np.int32)
 totals=arr[draw].sum(axis=1)
 tp,fp,fn,tn=totals.T
 def interval(numer,denom):
  vals=np.divide(numer,denom,out=np.full_like(numer,np.nan,dtype=float),where=denom>0)
  vals=vals[np.isfinite(vals)]
  return [round(float(np.quantile(vals,x)),4) for x in (.025,.975)] if len(vals)>=250 else None
 return {'recall_95pct_group_bootstrap':interval(tp,tp+fn),'precision_95pct_group_bootstrap':interval(tp,tp+fp),'false_positive_rate_95pct_group_bootstrap':interval(fp,fp+tn),'prompt_groups':n,'bootstrap_replicates':1000,'seed':seed}

def main():
 if OUT.exists():print('EXISTING_NOT_RERUN');return
 cols=['run_folder','model_family','url','defect','vlm_pred','prompt']
 p=pq.read_table(ROOT/'vlm_cell_predictions.parquet',columns=cols).to_pandas()
 gold=pd.read_csv(ROOT/'golden_adjudicated_truth.csv')
 manifest=pd.read_csv(ROOT/'asset_manifest.csv')
 values={}
 for idx,(model,folder) in enumerate(R.items()):
  a=p[(p.run_folder==folder)&(p.model_family==model)&(p.defect==DEFECT)]
  a=a.merge(gold[['url',DEFECT]],on='url',validate='one_to_one').rename(columns={DEFECT:'gold'})
  assert a.url.nunique()==99
  out={}
  for category,df in [('all_3d_expert',a),('mechanical_expert',a[a.url.isin(manifest.loc[manifest.category.isin(MECH),'url'])])]:
   scores=[]
   for prompt,g in df.groupby('prompt'):
    tp=int(((g.gold==1)&(g.vlm_pred==1)).sum());fp=int(((g.gold==0)&(g.vlm_pred==1)).sum());fn=int(((g.gold==1)&(g.vlm_pred==0)).sum());tn=int(((g.gold==0)&(g.vlm_pred==0)).sum())
    scores.append((tp,fp,fn,tn))
   out[category]=bootstrap(scores,20261024+idx)
  values[model]=out
 report={'dataset':'aieval2026/3d-defectbench 100 human expert adjudicated, 99 matching multi-view VLM predictions','gold_sha256':hashlib.sha256((ROOT/'golden_adjudicated_truth.csv').read_bytes()).hexdigest(),'VLM_source_sha256':hashlib.sha256((ROOT/'vlm_cell_predictions.parquet').read_bytes()).hexdigest(),'sampling_method':'1000 prompt-cluster bootstrap iterations; preserve both generated assets with same prompt within one group; each model bootstrapped separately, confidence intervals marginal not paired model difference','model_intervals':values,'limitations':'Small expert sample and class imbalance. Human expert labels are source-released rather than local video/CAD physical ground truth; grouped prompt bootstrap not external validation. Avoid ranking models by overlapping marginal 95% intervals.'}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 for model,d in values.items():print('UNCERTAINTY',model,'all',d['all_3d_expert'],'mech',d['mechanical_expert'],flush=True)
if __name__=='__main__':main()
