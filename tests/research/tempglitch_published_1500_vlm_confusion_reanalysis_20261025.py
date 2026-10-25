#!/usr/bin/env python3
"""Independently recompute 12 published VLM confusion tables on TempGlitch.

Read exact arxiv HTML paper v1 2605.21443 May 2026, snapshot SHA-256 and
source truth original Table3 summary, Table4 model/checkpoint version, Table5
exact TP,TN,FP,FN. Derive precision/recall/F1/accuracy ourselves and compare
with authors' rounded table. Authors ran VLM inference on 1500 original Godot
videos (750 injected temporal bugs + 750 category-matched normals). We DO NOT
rerun any VLM or issue a production GPU workload, and model accuracies are
published, not locally measured on Nitro. Dataset is 60.3 GB originally,
source metadata for ALL 1500 clips independently audited in separate script.

Important: correctness of published author's in-engine source labels differs
from independent human subjective playback/render quality. One dataset of two
Godot environments cannot be assumed to generalize. Earlier local Qwen3VL
clipped-text failure is separate from these author's Qwen3-VL-8B metrics.
"""
from pathlib import Path
from urllib.request import urlopen,Request
from bs4 import BeautifulSoup
import hashlib,datetime,json,math,statistics
import numpy as np
ROOT=Path('/data/var/perceptual-independent-20261024/tempglitch-control-20261025')
HTML=ROOT/'source_paper_arxiv_2605_21443v1.html'
OUT=ROOT/'all_12_source_vlm_exact_error_matrices_20261025.json'
URL='https://arxiv.org/html/2605.21443'
EPS=.11

def load_html():
 if HTML.exists():return HTML.read_bytes()
 req=Request(URL,headers={'User-Agent':'Mozilla/5.0 (research source audit)'})
 with urlopen(req,timeout=20) as f:data=f.read(800000)
 if not(100_000<len(data)<800000 and b'TempGlitch' in data and b'Qwen3VL' in data and b'GPT 5.4 Mini' in data):raise ValueError('unexpected primary arxiv source')
 ROOT.mkdir(parents=True,exist_ok=True);HTML.write_bytes(data)
 return data

def rows(table):
 return [[c.get_text(' ',strip=True).replace('\u00a0',' ').strip() for c in tr.find_all(['th','td'])] for tr in table.find_all('tr')]

def calc(TP,TN,FP,FN):
 total=TP+TN+FP+FN
 return {'TP':TP,'TN':TN,'FP':FP,'FN':FN,'glitch_truth_n':TP+FN,'clean_truth_n':TN+FP,'n':total,'accuracy_pct':100*(TP+TN)/total,'precision_pct':100*TP/(TP+FP) if TP+FP else 0.,'recall_pct':100*TP/(TP+FN) if TP+FN else 0.,'f1_pct':100*(2*TP)/(2*TP+FP+FN) if 2*TP+FP+FN else 0.,'false_positive_rate_pct':100*FP/(TN+FP) if TN+FP else None,'false_negative_rate_pct':100*FN/(TP+FN) if TP+FN else None,'balanced_accuracy_pct':50*(TP/(TP+FN)+TN/(TN+FP))}

def main():
 if OUT.exists():print('PAPER_CONFUSION_TABLE_PREVIOUSLY_INDEPENDENTLY_AUDITED');return
 raw=load_html();soup=BeautifulSoup(raw,'html.parser')
 table=[rows(t) for t in soup.find_all('table')]
 assert len(table)==5
 metrics,versions,conf=table[2],table[3],table[4]
 assert len(metrics)==len(conf)==14 and len(versions)==13
 assert all(len(v)==13 for v in metrics[1:] if v)
 version_by_model={x[0]:{'provider':x[1],'original_exact_checkpoint':x[2]} for x in versions[1:]}
 assert len(version_by_model)==12
 # Source primary paper names lack version suffix for Claude Opus/Haiku. Crosswalk
 # directly from paper Appendix A, NEVER guess absent checkpoint versions.
 revisions={'Claude Opus':'Claude Opus 4.7','Claude Haiku':'Claude Haiku 4.5','GPT 5.5':'GPT 5.5','GPT 5.4 Mini':'GPT-5.4 Mini','Gemini 3.1 Pro':'Gemini 3.1 Pro','Gemini 3 Flash':'Gemini 3 Flash','Qwen3.6 27B':'Qwen3.6 27B','Qwen3VL 8B':'Qwen3VL 8B','Gemma4 31B':'Gemma4 31B','Gemma4 4B':'Gemma4 E4B','Ministral3 14B':'Ministral3 14B','Ministral3 3B':'Ministral3 3B'}
 checked={};all_metrics=[]
 for high,low in zip(metrics[2:],conf[2:]):
  assert len(high)==len(low)==13 and high[0]==low[0]
  name=high[0];version=version_by_model[revisions[name]]
  checked[name]={'provider':version['provider'],'exact_published_model_version':version['original_exact_checkpoint'],'source_evaluations':{}}
  for setting,start in [('native_raw_video',1),('sampled_1fps',5),('sampled_5fps',9)]:
   observed=high[start:start+4];counts=low[start:start+4]
   if all(x in ('–','—','-') for x in observed+counts):continue
   assert all(x not in ('–','—','-') for x in observed+counts)
   nums=list(map(int,counts))
   result=calc(*nums)
   assert (result['glitch_truth_n'],result['clean_truth_n'],result['n'])==(750,750,1500)
   reported=[float(x) for x in observed]
   for label,y in zip(['accuracy_pct','precision_pct','recall_pct','f1_pct'],reported):
    if abs(result[label]-y)>EPS:raise ValueError('source primary reported metric does not match actual original confusion counts '+name+' '+setting+' '+label+' '+str(result[label])+' vs '+str(y))
   result['source_authors_original_reported_accuracy_precision_recall_f1_pct']=reported
   checked[name]['source_evaluations'][setting]=result
   all_metrics.append({'model':name,'setting':setting,**result})
 assert len(checked)==12
 assert all(sum(1 for key in d['source_evaluations'])>=1 for d in checked.values())
 best=max(all_metrics,key=lambda row:row['accuracy_pct'])
 assert best['model']=='GPT 5.4 Mini' and best['setting']=='sampled_5fps' and (best['TP'],best['TN'],best['FP'],best['FN'])==(173,638,112,577)
 assert checked['Qwen3VL 8B']['source_evaluations']['sampled_5fps']['TP']==2
 assert checked['Qwen3VL 8B']['source_evaluations']['sampled_5fps']['FP']==0
 assert (checked['Gemini 3 Flash']['source_evaluations']['native_raw_video']['TP'],checked['Gemini 3 Flash']['source_evaluations']['native_raw_video']['FP'])==(654,650)
 # balanced dataset generic always-positive predicts EVERY source as glitch,
 # gets F1 exactly 66.67 despite 750 false alarms. Always-clean 50% accuracy.
 naive=calc(750,0,750,0)
 normal=calc(0,750,0,750)
 report={'date':'2026-10-25','external_original_paper':'TempGlitch arXiv:2605.21443v1 (20 May 2026) source author paper, Section 5 Table 3 and Appendix E exact TP/TN/FP/FN, Appendix A checkpoint IDs','source_url':URL,'original_arxiv_html_sha256':hashlib.sha256(raw).hexdigest(),'authors_dataset':'asgaardlab/TempGlitch Godot in-engine injected temporal gameplay glitches, five types x 150 source faulty and x150 source normal 750/750; full 1500 source videos','published_model_results_locally_recomputed_not_inference':True,'author_model_count':12,'author_archived_runs':len(all_metrics),'models':checked,'best_overall_author_accuracy':{'model':best['model'],'setting':best['setting'],'accuracy_pct':best['accuracy_pct'],'true_positives':best['TP'],'false_positives':best['FP'],'false_negatives':best['FN'],'true_negatives':best['TN']},'simple_untrained_input_ignoring_policies':{'always_predict_glitch':naive,'always_predict_clean':normal},'validation':'for every published model+input setting, our recomputed classification accuracy, precision, recall and F1 from authors exact Table5 confusion matrix agrees with authors displayed Table3 numbers within 0.11 percentage points; all 750 actual clean + 750 authored bug source items accounted for','scope_limits':['These are author-published 12 VLM results on 1500 source clips; no 12 local VLM calls on Nitro/Thor, all model inference by authors with four L40S for their open-source subset.','Only two Godot environments. Controlled author injected glitches with self-assigned positive/normal labels; not independent blind human subjective review, and not naturally occurring 3D production rendering stalls.','Authors retry malformed JSON responses until valid; model output-format failures, inference-time overhead and API costs therefore are not represented by strict end-to-end accuracy.','Direct native video was evaluated only for Gemini; other models received up to 50 sampled images (1 or 5 FPS). An input format comparison is not randomized model-matched except within listed families.','Qwen3-VL-8B authors quantization is Unsloth bnb 4bit and not necessarily same weights, context or runtime as owner Thor Qwen3-VL or other workers. Qwen3.6 27B is not owner Qwen3.8 27B service.','Models have correlated errors on same original clips, paper tables do not provide raw predictions or per-case timing/latency here; simple binomial CI across repeated clips does not establish independent gameplay engine generalization.','High F1 such as 66% can be achieved trivially by predicting ALL videos glitchy on a 50:50 dataset; report precision, specificity, false alerts and misses, not just F1.','File-size-only metadata audit of original 1500 release is a separate local source experiment; paper models did not receive original MP4 bytes as an explicitly labeled metadata feature.']}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('SOURCE_PAPER_ALL_12_VLM_COUNTS_INDEPENDENTLY_RECOMPUTED','MODELS',len(checked),'MODEL_INPUT_SETTING_RESULTS',len(all_metrics),'BEST_ACC',round(best['accuracy_pct'],2),'SOURCE_SHA256',report['original_arxiv_html_sha256'],'ALWAYS_GLITCH_F1',round(naive['f1_pct'],3),'SAVED',OUT,flush=True)
 for name in ['GPT 5.4 Mini','Qwen3VL 8B','Gemini 3 Flash','Qwen3.6 27B']:
  for mode,vals in checked[name]['source_evaluations'].items():
   print('PUBLISHED_EXACT_REAL_DATA_CONFUSION',name,mode,'TP',vals['TP'],'FN',vals['FN'],'FP',vals['FP'],'TN',vals['TN'],'ACC',round(vals['accuracy_pct'],2),'FPR',round(vals['false_positive_rate_pct'],2),flush=True)
if __name__=='__main__':main()
