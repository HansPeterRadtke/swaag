#!/usr/bin/env python3
"""Verify current independent W3C and human-annotated 3D evidence without replay.

Aggregate-only report under SWAAG docs/research. Never commit sourced images,
GLBs, video, model weights, source per-cell predictions or original dataset CSV.
Running ScreenVLM cases remain explicitly pending, not fabricated scores.
"""
from __future__ import annotations
import hashlib,json,datetime,collections,re,statistics
from pathlib import Path
VAR=Path('/data/var/perceptual-independent-20261024')
OWN=Path(__file__).resolve().parents[2]
OUT=OWN/'docs/research/perceptual-source-calibrated-evidence-2026-10-24.json'

def source(path):
 data=path.read_bytes()
 return json.loads(data),hashlib.sha256(data).hexdigest()

def main():
 axe,h_axe=source(VAR/'w3c_act_contrast/axe_core_4_14_0_evaluation.json')
 model,h_model=source(VAR/'models/ScreenVLM/research_manifest.json')
 silver,h_silver=source(VAR/'3d-defectbench-2026-paper/independent_vlm_baseline_replay.json')
 expert,h_expert=source(VAR/'3d-defectbench-2026-paper/five_vlms_expert_adjudicated_replay.json')
 old,h_old=source(VAR/'w3c_act_contrast/independent_min_contrast_results.json')
 uncertainty,h_uncertainty=source(VAR/'3d-defectbench-2026-paper/expert_label_prompt_cluster_intervals.json')
 ui_source=VAR/'visible-ui-real/screenvlm_v2_first_run.json'
 parser,h_parser=source(ui_source) if ui_source.is_file() else ({'cases':[]},None)
 assert axe['summary']['cases_recorded']==34
 assert axe['summary']['coverage']==32
 assert axe['summary']['excluded_scripts']==2
 assert axe['summary']['cross_tab_expected_to_observed']=={'passed->passed':7,'failed->failed':7,'inapplicable->inapplicable':9,'passed->inapplicable':1,'failed->cannot_tell':4,'passed->cannot_tell':4,'inapplicable->cannot_tell':2}
 assert model['model']=='docling-project/ScreenVLM'
 assert model['revision']=='78df0cc8498bbea1a03eb2816056fc0a76d2f63d'
 assert model['license']=='apache-2.0'
 assert len(silver['models'])==5 and len(expert['models'])==5
 assert expert['expert_assets_common']==99
 assert len(uncertainty['model_intervals'])==5
 assert {v['scored_cells'] for v in silver['models'].values()}=={8982}
 assert {v['human_labeled_cells'] for v in expert['models'].values()}=={891}
 # Context/reference visible and no gold labels were passed in ScreenVLM prompt.
 assert all(x['prompt']==parser.get('instruction') for x in parser.get('cases',[]))
 assert len(parser['cases'])==4 and all(x['status']=='completed' for x in parser['cases'])
 parse_cases=[]
 for row in parser.get('cases',[]):
  raw=row.get('raw_screen_tag','')
  if row['status']=='completed':
   assert raw.startswith('<screentag>')
  stripped_text=re.sub(r'<[^>]*>','',raw.replace('<|end_of_text|>','')).strip()
  original_text=row.get('original_source_text','').strip()
  text_recovered=bool(original_text) and original_text.casefold() in stripped_text.casefold()
  parse_cases.append({'source_item_sha256':row['source_item_sha256'],'status':row['status'],'seconds':row.get('seconds'),'generated_plain_text':stripped_text[:120],'recovered_full_expected_label':text_recovered,'well_formed_complete_screentag':('</screentag>' in raw),'predicted_location_tokens':row.get('location_tokens'),'output_chars':row.get('output_chars'),'external_gold_violation':row.get('external_gold_violation')})
 w3c={
  'pin_version':axe['tool'],'source_bundle_sha256':axe['pinned_js_sha256'],
  'official_cases':34,'executed':32,'scripts_safely_excluded':2,'truth_to_reported':axe['summary']['cross_tab_expected_to_observed'],
  'correct_decided_cases':23,'pass_misclassified_inapplicable':1,'cannot_tell_including_excluded':10,
  'note':'Color-contrast is a real standard browser analyzer. Eight executed cases inconclusive, plus two script-bearing source cases intentionally excluded. No false violation in determinate reported cases; not a full WCAG certification.',
 }
 mechanical={}
 for key in sorted(silver['models']):
  s=silver['models'][key]['metrics']['physical_mechanical_categories']['mechanical_missing_parts']
  g=expert['models'][key]['geometry']['per_defect']['Incorrect Prompt Quantity/Missing Parts']
  mg=expert['models'][key]['mechanical_missing_parts']
  assert s['n']==368 and s['TP']+s['FN']==47
  assert g['n']==99 and g['TP']+g['FN']==31
  assert mg['n']==30 and mg['TP']+mg['FN']==9
  mechanical[key]={
   'protocol':'8-view binary per-aspect released author predictions (historic provider versions)','silver_mechanical_missing':s,
   'expert_all_asset_missing':g,'expert_mechanical_missing':mg,
   'silver_geometry_macro_MCC':silver['models'][key]['metrics']['geometry']['macro_mcc_mean_nonconstant'],
   'expert_geometry_macro_MCC':expert['models'][key]['geometry']['macro_mcc']
  }
 output={
  'date':'2026-10-24','state':'PARTIAL; no model qualifies for automatic GUI/video/industrial 3D approval',
  'kind':'independent recomputation of released original VLM predictions versus external human defect reference, plus local version-pinned real axe-core and local ScreenVLM source parsing cases',
  'source_sha256':{'axe_run':h_axe,'earlier_custom_checker':h_old,'ScreenVLM_model_file_manifest':h_model,'ScreenVLM_input_run':h_parser,'3D_silver_comparable_runs':h_silver,'3D_expert_adjudicated':h_expert,'3D_expert_prompt_cluster_bootstrap':h_uncertainty},
  'W3C_ACT_axe_core':w3c,
  'W3C_ACT_custom_previous':{'counts':old['counts'],'testcases':len(old['cases']),'purpose':'compare coverage against off-the-shelf ACT-aware axe engine, not certify custom checker'},
  'ScreenVLM_real_input_task_limited':{'model':'docling-project/ScreenVLM v2 Apache2','repository_revision':model['revision'],'source_model_bytes':model['total_bytes'],'CPU_only':True,'instruction':parser.get('instruction'),'cases':parse_cases,'test_cases':4,'gold_clipped':2,'gold_clean':2,'expected_labels_recovered':sum(int(x['recovered_full_expected_label']) for x in parse_cases),'complete_closed_screen_tags':sum(int(x['well_formed_complete_screentag']) for x in parse_cases),'sum_CPU_seconds':round(sum(x['seconds'] or 0 for x in parse_cases),2),'mean_CPU_seconds':round(statistics.mean(x['seconds'] for x in parse_cases),2),'observed_peak_rss_kb':max(x['peak_rss_kb'] for x in parser['cases']),'correctness':'Parse outputs are screen elements, not defect classifications; real small target crops outside full-web distribution. Only completed cases analyzed. CPU latency important.'},
  '3D_DefectBench_human':{'source':'3D-DefectBench aieval2026 pinned commit','revision':silver['source_revision'],'silver_n_assets':998,'expert_adjudicated_n_assets':99,'categories_mechanical': ['Land Vehicles','Air Vehicles','Robots','Water Vehicles','Tools','Technology & Electronics'],'published_model_predictions_recomputed_locally_NOT_new_inference':True,'comparability':'identical raw eight-view binary per-aspect seed-one prompt/evidence protocol for five models; Gemini3Pro excluded because duplicated configs cannot be fairly isolated','models':mechanical,'expert_missing_parts_prompt_group_bootstrap_intervals':uncertainty['model_intervals']},
  'original_external_data_remains_in_var':True,
  'caveats':['The released 3D prediction data are generated by authors, not inference performed on Nitro; historical model versions and public training contamination unknown.','Mechanical category selection exploratory and may combine vehicles, robotics, tools and electronics; geometry defects are human labels, not numerical engineering physics.','Only 99 golden assets common to all five baseline models, human labels are adjudicated but nonzero human disagreement and rare categories.','W3C ACT contains author-scripted and image-background cases; scripts excluded and proxy blocks network to protect system.','Small cropped target screenshots may remove important context for ScreenVLM; no extrapolation to normal full-page parsing.','NVIDIA physical plausibility video head reports approximately 80.52% bad recall and 31.33% good rejection on its heldout same-generator set, H100, not validated locally on machinery videos.'],
  'next_required':['Independent clean and physically impossible REAL mechanical CAD and render examples beyond generative text-to-3D source labels, with source geometry and multiview captures','Video model comparison on human temporal-discontinuity/real stutter annotations and approved timelines','ScreenVLM full-page and screenshot-only defect detector comparison with same developer-separate real-site holdout; parser output alone not acceptance','Official PatchCore/EfficientAD domain-trained industrial anomaly comparison on independent actual generated 3D textures','Cross-browser axe rule coverage, known intention for ellipsis and valid overlap exceptions']
 }
 OUT.write_text(json.dumps(output,indent=2)+'\n')
 print('SOURCE_EVIDENCE_SAVED',OUT,'BYTES',OUT.stat().st_size,'MODEL_CASES',len(parse_cases),'EXPERT_MODELS',len(mechanical))
if __name__=='__main__':main()
