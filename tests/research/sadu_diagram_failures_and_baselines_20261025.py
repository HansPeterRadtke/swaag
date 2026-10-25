#!/usr/bin/env python3
"""Harder cross-model task/error analysis of archived SADU real architecture diagrams.

All 11 source VLM outputs independently scored already, and source labels are
manual QA for 154 real architectural drawings. This script analyzes only the
source-question-grounded result, not raw image/model inference.

Predeclared baselines: always answer integer zero on *numeric counting* tasks,
then source-diagram grouped paired bootstrap of Gemini 3 Flash vs Gemini 2.5
Flash; 24 task subtype macro-averages, count under/over asymmetry, structured
format output validity, retrieval FP/FN, and overlap/disagreement with source
LLM-as-a-judge. No threshold tuning, no uncertainty claims beyond SADU source.
"""
from pathlib import Path
import json,hashlib,collections,statistics
import numpy as np
ROOT=Path('/data/var/perceptual-independent-20261024/sadu-diagrams')
RAW=ROOT/'sadu_11_models_independent_qa_decisions_20261025.json'
SUMMARY=ROOT/'sadu_11_models_aggregate_20261025.json'
OUT=ROOT/'sadu_real_11_models_task_balance_and_judge_audit_20261025.json'
MODELS=('gemini-3-flash-preview','gemini-2.5-flash','claude-sonnet-4-5','qwen_2.5-VL-32B','qwen_2.5-VL-7B','gpt-5-nano','gpt-4o-mini')
B=4000;SEED=20261025

def main():
 if OUT.exists():print('PRIOR_SADU_DIAGRAM_TASK_AUDIT_PRESERVED');return
 r=json.loads(RAW.read_text());s=json.loads(SUMMARY.read_text());assert len(r['cases_by_model'])==len(s['models'])==11
 base=r['cases_by_model']['gemini-3-flash-preview']
 ids=sorted(x['index'] for x in base)
 assert ids==list(range(2431))
 aligned={name:{x['index']:x for x in rows} for name,rows in r['cases_by_model'].items()}
 assert all(set(d)==set(ids) for d in aligned.values())
 numeric=[x for x in base if x['gold_type']=='numeric'];assert len(numeric)==1372
 zero=sum(x['numeric_true']==0 for x in numeric);by_sub=collections.defaultdict(list)
 for x in numeric:by_sub[x['task_subtype']].append(x)
 zero_task=[{'task':key,'questions':len(a),'author_true_zero_fraction':sum(x['numeric_true']==0 for x in a)/len(a),'mean_gold_answer':statistics.mean(x['numeric_true'] for x in a)} for key,a in sorted(by_sub.items())]
 # Strict zero count baseline is an input-independent constant, not learned.
 zero_baseline={'always_zero_correct':zero,'total_numeric_qa':len(numeric),'exact_accuracy':zero/len(numeric),'by_subtype':zero_task}
 overall={}
 for name,rows in r['cases_by_model'].items():
  valid=[x for x in rows if x['parse_status']=='ok']
  sub=collections.defaultdict(list)
  for row in rows:sub[row['task_subtype']].append(row)
  allsub=[sum(r['is_correct'] for r in items)/len(items) for items in sub.values()]
  numerical=[row for row in rows if row['gold_type']=='numeric'];readable=[x for x in rows if x['gold_type'] in ('text','oriented_pairs')]
  valid_num=[x for x in numerical if x['numeric_pred'] is not None]
  nonzero=[x for x in valid_num if x['numeric_error']!=0]
  missed=sum(x['set_fn'] for x in readable);false=sum(x['set_fp'] for x in readable)
  judge_numeric=[x for x in numerical if x['author_judge_positive'] is not None]
  judge_false_yes=sum(x['author_judge_positive'] and not x['is_correct'] for x in judge_numeric)
  judge_false_no=sum(not x['author_judge_positive'] and x['is_correct'] for x in judge_numeric)
  valid_rate=len(valid)/len(rows)
  overall[name]={'n':len(rows),'valid_json_answer_count':len(valid),'valid_json_answer_fraction':valid_rate,'valid_response_accuracy':sum(x['is_correct'] for x in valid)/len(valid),'strict_unconditional_accuracy':sum(x['is_correct'] for x in rows)/len(rows),'task_macro_averaged_exact_accuracy':statistics.mean(allsub),'subtypes_n':len(sub),'num_exact_correct':sum(x['is_correct'] for x in numerical),'num_gold_cases':len(numerical),'number_zero_baseline_accuracy':zero/len(numerical),'num_under_count_cases':sum(x['numeric_error']<0 for x in nonzero),'num_over_count_cases':sum(x['numeric_error']>0 for x in nonzero),'num_nonzero_mistakes':len(nonzero),'relation_retrieval_spurious_false_elements':false,'relation_retrieval_missing_expected_elements':missed,'LLM_judge_numeric_false_yes_vs_exact':judge_false_yes,'LLM_judge_numeric_false_no_vs_exact':judge_false_no,'LLM_judge_numeric_disagreement_rate':(judge_false_no+judge_false_yes)/len(judge_numeric),'source_index_order_mismatch_n':r['index_integrity'][name]['source_question_ordinals_differ_from_canonical']}
 # Paired matrix across original 154 diagrams, not questions: identical source
 # evaluation set; per-image subquestions correlated, so resample whole images.
 a=aligned['gemini-3-flash-preview'];b=aligned['gemini-2.5-flash']
 diagrams=sorted({x['source_group_sha256'] for x in a.values()});assert len(diagrams)==154
 per_diagram=[]
 for graph in diagrams:
  qa=[i for i in ids if a[i]['source_group_sha256']==graph]
  per_diagram.append({'n':len(qa),'g3_correct':sum(a[i]['is_correct'] for i in qa),'g25_correct':sum(b[i]['is_correct'] for i in qa),'num_count':sum(a[i]['gold_type']=='numeric' for i in qa),'g3_numeric_correct':sum(a[i]['is_correct'] for i in qa if a[i]['gold_type']=='numeric'),'g25_numeric_correct':sum(b[i]['is_correct'] for i in qa if b[i]['gold_type']=='numeric')})
 array=np.asarray([[v['n'],v['g3_correct'],v['g25_correct'],v['num_count'],v['g3_numeric_correct'],v['g25_numeric_correct']] for v in per_diagram],dtype=np.int32)
 rng=np.random.default_rng(SEED)
 count_deltas=[];num_deltas=[]
 for _ in range(B):
  sample=array[rng.integers(0,len(array),len(array))].sum(axis=0)
  count_deltas.append((sample[1]-sample[2])/sample[0]);num_deltas.append((sample[4]-sample[5])/sample[3])
 g3_n=sum(a[i]['is_correct'] for i in ids);g25_n=sum(b[i]['is_correct'] for i in ids)
 paired={'gemini3_vs_gemini2p5_flash_exact_accuracy_difference':(g3_n-g25_n)/len(ids),'numeric_accuracy_difference':(sum(a[i]['is_correct'] for i in ids if a[i]['gold_type']=='numeric')-sum(b[i]['is_correct'] for i in ids if a[i]['gold_type']=='numeric'))/1372,'diagram_cluster_bootstrap_95pct_diff':[float(np.quantile(count_deltas,.025)),float(np.quantile(count_deltas,.975))],'diagram_cluster_bootstrap_95pct_numeric_diff':[float(np.quantile(num_deltas,.025)),float(np.quantile(num_deltas,.975))],'paired_question_outcomes':{'both_correct':sum(a[i]['is_correct'] and b[i]['is_correct'] for i in ids),'gemini3_only':sum(a[i]['is_correct'] and not b[i]['is_correct'] for i in ids),'gemini2p5_only':sum(b[i]['is_correct'] and not a[i]['is_correct'] for i in ids),'both_wrong':sum(not a[i]['is_correct'] and not b[i]['is_correct'] for i in ids)},'resample_unit':'source diagram, 154 independent-ish diagrams, 4000 repeated draws','seed':SEED}
 result={'source':'154 real manually structured SADU architecture diagrams, 2431 original QA each, 11 archived VLM outputs from authors','source_results_SHA256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'ground_qa_verified':r['dataset_verified'],'date':'2026-10-25','numeric_zero_baseline':zero_baseline,'per_model':overall,'paired_gemini_source_grouped':paired,'limitations':'Source counts include many zeros and common categories. Strict output format and prompt/truncation errors count against product-level usability, not solely vision understanding; valid-only denominator is selected and optimistic. Task macro average gives rare relation subtypes equal influence and is NOT the original benchmark metric. Original VLM-as-judge may allow semantic synonyms, but its numeric true/false disagreement against exact number is independently measurable. External author predictions, no new model inference. Current/latest model versions may differ.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('ZERO_BASELINE',zero,'/',len(numeric),'=',round(zero/len(numeric),4),'PAIRED_GEMINI',paired)
 for name,x in sorted(overall.items(),key=lambda kv:-kv[1]['strict_unconditional_accuracy']):
  print('MODEL',name,'STRICT',round(x['strict_unconditional_accuracy'],4),'VALID_ONLY',round(x['valid_response_accuracy'],4),'MACRO_TASK',round(x['task_macro_averaged_exact_accuracy'],4),'ZERO_BASE',round(x['number_zero_baseline_accuracy'],4),'COUNT_OVER_UNDER',(x['num_over_count_cases'],x['num_under_count_cases']),'JUDGE_NUMERIC_DISAGREE',round(x['LLM_judge_numeric_disagreement_rate'],4),'INDEX_REMAP',x['source_index_order_mismatch_n'])
if __name__=='__main__':main()
