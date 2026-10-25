#!/usr/bin/env python3
"""Re-score archived high/low-thinking and prompt-ablation VLM predictions.

Real SADU 154 real source manually labeled architecture diagrams; 2,431 QA.
Authors already ran the model calls and published full original response logs
in Zenodo 19339991. No new model GPU/inference or remote prompts. Compare
same model + same original diagram QA under different externally documented
'full', 'only_question', 'without_definition', 'without_rules' settings and
thinking 'low' vs 'high'. Primary property strict source-answer correctness.

This is exploratory on ALREADY PUBLIC full SADU QA, not independent untouched
model selection. Output-token budget fixed at 512 in author experiment; output
truncation can masquerade as perceptual failure. Usage tokens are only cost
proxies, not measured wall-clock latency or current currency cost.
"""
import hashlib,json,zipfile,collections,statistics,sys
from pathlib import Path
import numpy as np
from sadu_11_vlm_real_diagrams_reanalysis import score,MODELS,RELATION_SUBTYPES,COUNT_SUBTYPES
ZIP=Path('/data/var/perceptual-independent-20261024/sadu-diagrams/SADU_ASE_submission.zip')
RAW=ZIP.parent/'sadu_prompt_thinking_model_output_comparisons_20261025.json'
SRC='SADU-main/experiment_result/response_result/'
CONFIG={
 'gemini3_full_default': 'default/gemini-3-flash-preview_response_0223_t_0.log',
 'gemini3_full_low_thinking':'thinking_level/low_gemini-3-flash-preview_response_0223_t_0.log',
 'gemini3_full_high_thinking':'thinking_level/high_gemini-3-flash-preview_response_0223_t_0.log',
 'gemini3_question_only_default':'thinking_level/only_question_gemini-3-flash-preview_response_0223_t_0.log',
 'gemini3_question_only_low_thinking':'thinking_level/low_only_question_gemini-3-flash-preview_response_0223_t_0.log',
 'gemini3_question_only_high_thinking':'thinking_level/high_only_question_gemini-3-flash-preview_response_0223_t_0.log',
 'gemini2p5_full_default':'default/gemini-2.5-flash_response_0223_t_0.log',
 'gemini2p5_question_only':'different_prompt/only_question_gemini-2.5-flash_response_0306_t_0.log',
 'gemini2p5_without_definition':'different_prompt/without_definition_gemini-2.5-flash_response_0306_t_0.log',
 'gemini2p5_without_rules':'different_prompt/without_rules_gemini-2.5-flash_response_0306_t_0.log',
}
B=3000
SEED=20261025

def canonical_key(item):
 return (item['diagram_type'],item['file'],item['question'],json.dumps(item['answer'],sort_keys=True,ensure_ascii=False),item['metadata']['sub_type'])

def metric(cases):
 n=len(cases);valid=[r for r in cases if r['parse_status']=='ok']
 nums=[r for r in cases if r['gold_type']=='numeric']
 rel=[r for r in cases if r['task_subtype'] in RELATION_SUBTYPES]
 return {'n':n,'exact_correct':sum(r['is_correct'] for r in cases),'exact_accuracy':sum(r['is_correct'] for r in cases)/n,'format_or_type_invalid':n-len(valid),'valid_answer_fraction':len(valid)/n,'parsed_only_accuracy':sum(r['is_correct'] for r in valid)/len(valid) if valid else None,'numeric_count_accuracy':sum(r['is_correct'] for r in nums)/len(nums) if nums else None,'relations_accuracy':sum(r['is_correct'] for r in rel)/len(rel) if rel else None}

def main():
 if RAW.exists():print('PROMPT_VARIANT_RESULTS_ALREADY_PRESERVED');return
 with zipfile.ZipFile(ZIP) as z:
  src={k:z.read(SRC+path) for k,path in CONFIG.items()}
  logs={name:[json.loads(row) for row in contents.decode().splitlines() if row] for name,contents in src.items()}
  for name,rows in logs.items():assert len(rows)==2431 and len({canonical_key(x) for x in rows})==2431,(name,len(rows))
  canon={canonical_key(x):i for i,x in enumerate(logs['gemini3_full_default'])}
  assert len(canon)==2431
  by_variant={};groups={};usage={}
  for name,rows in logs.items():
   assert set(map(canonical_key,rows))==set(canon),(name,'different source questions')
   scores={}
   tks=[]
   for row in rows:
    idx=canon[canonical_key(row)];r=score(row.get('response'),row['answer'])
    scores[idx]={**r,'source_group':row['diagram_type']+'|'+row['file'],'task_subtype':row['metadata']['sub_type']}
    tk=row.get('token_usage') or {}
    if tk.get('total_tokens') is not None:tks.append(tk['total_tokens'])
   assert len(scores)==2431
   by_variant[name]=[scores[i] for i in sorted(scores)]
   usage[name]={'n_reported_token_use':len(tks),'source_log_mean_total_tokens':statistics.mean(tks) if tks else None,'source_log_median_total_tokens':statistics.median(tks) if tks else None,'all_original_questions_aligned':2431,'archived_response_log_sha256':hashlib.sha256(src[name]).hexdigest()}
   print('VARIANT',name,metric(by_variant[name]),'MEAN_TOKENS',usage[name]['source_log_mean_total_tokens'],flush=True)
  # Preserve direct-pair difference and grouped uncertainty (same per-QA graph).
  qnames=sorted({x['source_group'] for x in by_variant['gemini3_full_default']})
  assert len(qnames)==154
  groups_per={name:collections.defaultdict(list) for name in CONFIG}
  for name,rows in by_variant.items():
   for x in rows:groups_per[name][x['source_group']].append(x)
  rng=np.random.default_rng(SEED)
  source_indices=rng.integers(0,len(qnames),size=(B,len(qnames)),dtype=np.int16)
  paired={}
  for name,rows in by_variant.items():
   base='gemini3_full_default' if name.startswith('gemini3') else 'gemini2p5_full_default'
   if name==base:continue
   orig=by_variant[base]
   wins=sum(a['is_correct'] and not b['is_correct'] for a,b in zip(rows,orig));losses=sum(b['is_correct'] and not a['is_correct'] for a,b in zip(rows,orig))
   nby=np.array([len(groups_per[name][g]) for g in qnames]);corr=np.array([sum(x['is_correct'] for x in groups_per[name][g]) for g in qnames]);baseline=np.array([sum(x['is_correct'] for x in groups_per[base][g]) for g in qnames])
   assert np.array_equal(nby,np.array([len(groups_per[base][g]) for g in qnames]))
   sums=(corr[source_indices]-baseline[source_indices]).sum(axis=1)
   totals=nby[source_indices].sum(axis=1)
   differences=sums/totals
   paired[name]={'baseline':base,'delta_accuracy':(wins-losses)/len(rows),'improved_QAs':wins,'regressed_QAs':losses,'unchanged_QAs':len(rows)-wins-losses,'source_diagram_clustered_95pct_delta':[float(np.quantile(differences,.025)),float(np.quantile(differences,.975))],'cannot_establish_current_model_causal_effects':True}
   print('PAIRED_ABLATION',name,'BASE',base,'DELTA',paired[name]['delta_accuracy'],'CI',paired[name]['source_diagram_clustered_95pct_delta'],'WINS_LOSSES',wins,losses,flush=True)
  report={'date':'2026-10-25','source':'SADU real 154 original software architecture diagrams, 2431 manual-source QA each, archived historical Gemini Flash output with full/only question and high/low thinking or prompt omissions','archive_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),'model_output_snapshots_NOT_new_inference':True,'comparison_source_model_settings':CONFIG,'all_QA_multisets_and_source_labels_match':True,'model_results':{name:metric(rows) for name,rows in by_variant.items()},'source_log_usage':usage,'paired_vs_full_default':paired,'BOOTSTRAP_SAMPLES':B,'SEED':SEED,'limits':'Authors used max output 512 tokens, many text answers truncated especially gemini3 high/low/onlyquestion. There is no independent heldout prompt data here, this is reused same public benchmark and model snapshot; token usage is author reported not latency/cost. This tests Q&A format, recognition+relation reasoning as a complete task, not pure image defect classification. Multiple hypotheses tested; trust per-setting outcomes not posthoc best choice.'}
  RAW.write_text(json.dumps(report,indent=2)+'\n')
  print('SADU_ABLATION_SOURCE_REANALYSIS_SAVED',RAW,'CONFIGS',len(CONFIG))
if __name__=='__main__':main()
