#!/usr/bin/env python3
"""Publish compact, SOURCE-GROUNDED visual-diagram reliability results only.

Origin SADU 154 real architecture diagram PNGs + human graph source JSON +
2431 QA per 11 already archived model runs. No VLM inference, no reading
private token folders, no licensed images/raw responses committed. All source
per-case output retained separately under /data/var. This aggregator requires
all inputs to be complete, including real pixel-only Tesseract coverage.
"""
from pathlib import Path
import hashlib,json,collections,datetime
VAR=Path('/data/var/perceptual-independent-20261024/sadu-diagrams')
DOCS=Path(__file__).resolve().parents[2]/'docs/research'
OUT=DOCS/'perceptual-sadu-154-real-diagrams-evidence-2026-10-25.json'


def source(name):
 p=VAR/name;b=p.read_bytes()
 return json.loads(b),hashlib.sha256(b).hexdigest()


def main():
 if OUT.exists():print('COMPACT_ARCHIVED_REPORT_PRESERVED');return
 raw,hraw=source('sadu_11_models_independent_qa_decisions_20261025.json')
 metrics,hagg=source('sadu_11_models_aggregate_20261025.json')
 difficult,hdiff=source('sadu_real_11_models_task_balance_and_judge_audit_20261025.json')
 graph,hgraph=source('sadu_154_independent_source_graph_QA_audit_20261025.json')
 ocr,hocr=source('sadu_154_image_only_ocr_vs_graph_labels_20261025.json')
 ocrraw,hocrraw=source('sadu_154_original_png_frozen_ocr_psm11_20261025.json')
 ablate,hablate=source('sadu_prompt_thinking_model_output_comparisons_20261025.json')
 assert raw['status']=='complete' and len(raw['cases_by_model'])==11
 assert raw['dataset_verified']['human_diagram_groups']==154
 assert raw['dataset_verified']['QAs_matched_to_original_source_json']==2431
 assert len(metrics['models'])==11
 assert all(v['total']['n']==2431 for v in metrics['models'].values())
 assert len(ocr['cases'])==154 and len(ocrraw['predictions'])==154
 assert graph['numeric_QA_invariants_tested']==924 and not graph['individual_mismatches']
 assert ocr['n_diagrams']==154
 assert raw['index_integrity']['qwen_2.5-VL-3B']['source_question_ordinals_differ_from_canonical']==2431
 assert ablate['model_results']['gemini3_question_only_default']['valid_answer_fraction']==0.0
 assert metrics['models']['gemini-3-flash-preview']['total']['exact_correct']==1743
 assert metrics['models']['gemini-2.5-flash']['total']['exact_correct']==1692
 assert metrics['models']['qwen_2.5-VL-32B']['total']['exact_correct']==1078
 assert difficult['numeric_zero_baseline']['always_zero_correct']==345
 result={
  'date':'2026-10-25',
  'goal':'Reliably judge real GUI/scientific/architecture diagram correctness from available deterministic checks and vision models',
  'overall_status':'PARTIAL — no single vision-language or OCR diagram judgment model qualifies for automatic structural/visual defect acceptance',
  'primary_source':'SADU, Shuyin Ouyang et al. arXiv:2604.04009, CC BY4 original Zenodo DOI 10.5281/zenodo.19339991',
  'source_zip_sha256':raw['source_original_ZIP_SHA256'],
  'source_original_structure':{'real_diagrams':154,'behavior':51,'structural':53,'ER':50,'author_manual_graph_and_QA':True,'model_response_logs':11,'model_QAs_each':2431,'total_archived_QA_predictions_scored':11*2431,'independent_numeric_source_graph_assertions':924,'independent_numeric_assertion_mismatches':0,'author_original_graph_entities':1842,'author_original_graph_relations':1887},
  'raw_source_reproducibility_sha256':{'original_source_QA_model_decisions':hraw,'independent_model_aggregate':hagg,'quality_count_baselines_and_discrepancy_audit':hdiff,'original_graph_oracle_consistency':hgraph,'image_only_OCR_labels':hocr,'pixel_only_OCR_raw':hocrraw,'author_prompt_thinking_output_ablation':hablate},
  'scoring_protocol':metrics['evaluation_version'],
  'model_archived_output_not_live_inference':True,
  'primary_metrics':{},
  'zero_count_baseline':difficult['numeric_zero_baseline'],
  'model_comparison':difficult['paired_gemini_source_grouped'],
  'authored_index_alignment':{name:r['source_question_ordinals_differ_from_canonical'] for name,r in raw['index_integrity'].items()},
  'source_graph_QA_internal_consistency':graph['results'],
  'pixel_only_Tesseract_vs_entity_names':{'ocr_config':ocr['ocr_config'],'whole_original_real_diagrams':ocr['n_diagrams'],'all':ocr['overall'],'per_domain':ocr['by_diagram_type'],'note':'Expected entity-name recall only. No false positive precision because unannotated legitimate relation names and legends appear in rendered diagram. Pixel-only OCR not semantic graph/edge understanding.'},
  'prompt_and_reasoning_ablation':{
   'source_metadata':'Same source 154 diagrams, 2431 QA per variant; authors already ran all inference, max_completion_tokens 512',
   'valid_JSON_comparable':{k:{'overall':v['exact_accuracy'],'numeric':v['numeric_count_accuracy'],'relation':v['relations_accuracy'],'format_invalid':v['format_or_type_invalid'],'source_mean_tokens':ablate['source_log_usage'][k]['source_log_mean_total_tokens']} for k,v in ablate['model_results'].items() if 'question_only' not in k},
   'source_question_only_not_strictly_scored':['gemini3_question_only_default','gemini3_question_only_low_thinking','gemini3_question_only_high_thinking','gemini2p5_question_only'],
   'strict_only_question_zero_is_prompt_contract_mismatch_NOT_model_comprehension':True,
   'compatible_paired_QA_differences':{k:v for k,v in ablate['paired_vs_full_default'].items() if 'question_only' not in k},
  },
  'limitations':[
   'This source original archive includes a personal_token directory never opened and never committed. Original benchmark model outputs and source images kept outside Git.',
   'QA model predictions are historical author inference, not live 2026-current model checks on owners own diagrams.',
   'Original QA derived from human annotated source graph; 924/924 graph numeric consistency does not certify each PNG depicts the source graph perfectly.',
   'Strict deterministic normalized set-match rejects some semantically equivalent text and counts source prompt formatting truncations as end-to-end failure.',
   'Generic VLM task QA does not directly measure false alarms on generated defective vs true clean diagram variants, because SADU original rendered diagrams are reference inputs.',
   'Tesseract entity name recall is not edge direction or graph semantic accuracy; OCR extra tokens cannot be called false-positive entities without comprehensive text annotation.',
   'Question-only model outputs are natural-language responses rather than required JSON; cannot be ranked as 0% diagram understanding under the JSON exact format.',
   'Bootstrap confidence intervals resample original diagrams, not independent downstream user design projects; public source possible benchmark exposure.'
  ],
  'recommended_use':'Treat source-coded diagram graph topology/direction/count/labels as the contract; compare actual image label/connector visibility with deterministic OCR/geometry and independently reviewed discrepancies. Use VLMs only as evidence/proposal with verified false-alarm/negative rates, not unconditional acceptance.',
 }
 for name,metric in metrics['models'].items():
  v=difficult['per_model'][name]
  result['primary_metrics'][name]={'strict_source_QA_accuracy':metric['total']['exact_accuracy'],'strict_correct_QA_n':metric['total']['exact_correct'],'numeric_QA_accuracy':metric['numeric_count_tasks']['exact_accuracy'],'relation_QA_accuracy':metric['relation_retrieval']['exact_accuracy'],'relation_text_retrieval_precision':metric['total']['retrieval_micro_precision'],'relation_text_retrieval_recall':metric['total']['retrieval_micro_recall'],'retrieval_false_positive_source_entities':metric['total']['retrieved_spurious_elements_FP'],'retrieval_false_negative_source_entities':metric['total']['retrieved_missing_elements_FN'],'format_or_answer_type_invalid':metric['total']['malformed_or_wrong_answer_type'],'source_model_judge_disagreement':metric['total']['author_vs_exact_mismatch'],'cluster_95pct_strict_accuracy':metric['source_cluster_bootstrap_CI95']['overall'],'task_macro_average_accuracy':v['task_macro_averaged_exact_accuracy'],'conditional_accuracy_on_valid_JSON':v['valid_response_accuracy']}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('SADU_REAL_DIAGRAM_SOURCE_EVIDENCE_SAVED',OUT,'BYTES',OUT.stat().st_size,'OCR',result['pixel_only_Tesseract_vs_entity_names']['all'],'ARCHIVED_MODELS',len(result['primary_metrics']))
if __name__=='__main__':main()
