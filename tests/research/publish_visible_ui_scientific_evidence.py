#!/usr/bin/env python3
"""Verify new external-source tests; publish small aggregate only, never media."""
from __future__ import annotations
import hashlib,json,collections,datetime
from pathlib import Path
VAR=Path('/data/var/perceptual-independent-20261024')
OUT=Path(__file__).resolve().parents[2]/'docs/research/perceptual-visible-human-reviewed-evidence-2026-10-24.json'

def load(name):
 p=VAR/name;raw=p.read_bytes();return json.loads(raw),hashlib.sha256(raw).hexdigest()

def main():
 screen,screen_hash=load('visible-ui-real/anchored_manifest.json')
 clip,clip_hash=load('visible-ui-real/clip_zeroshot_result.json')
 scry_src,scry_src_hash=load('scry311/shard_manifest.json')
 scry_raw,scry_raw_hash=load('scry311/scry_cv_pixel_only_predictions.json')
 scry_eval,scry_eval_hash=load('scry311/scry_cv_human_evaluation.json')
 w3c_src,w3c_src_hash=load('w3c_act_contrast/manifest.json')
 w3c_eval,w3c_eval_hash=load('w3c_act_contrast/independent_min_contrast_results.json')
 assert len(screen['cases'])==32 and all(x['status']=='ok' and x['anchor_intersects_image'] for x in screen['cases'].values())
 assert len(clip['cases'])==32 and clip['scores']['test_redecheck:element-protrusion']['TP']==0
 assert scry_src['num_shards']==16 and len(scry_src['files'])==16 and all(x['status'] in ('new_download_ok','existing_verified_size') for x in scry_src['files'])
 assert scry_src['expected_total_bytes']==198677141
 assert scry_raw['n']==311 and scry_eval['dataset_annotation_n']==557 and scry_eval['app_groups']==52
 assert scry_eval['folds']=={'calibration_rows':62,'calibration_app_groups':14,'heldout_rows':249,'heldout_app_groups':38}
 held=scry_eval['threshold_methods']['ssim_difference']['heldout'];cal=scry_eval['threshold_methods']['ssim_difference']['calibration']
 assert (held['TP'],held['FP'],held['TN'],held['FN'])==(157,41,22,29)
 assert (cal['TP'],cal['FP'],cal['TN'],cal['FN'])==(42,4,10,6)
 localization=scry_eval['localization_without_issue_semantics']['heldout']
 assert (localization['issues_with_box'],localization['bbox_iou_recovered_issues_no_tag_check'])==(458,253)
 assert len(w3c_src['cases'])==34 and len(w3c_eval['cases'])==34
 assert w3c_eval['counts']=={'TP':5,'FP':1,'FN':0,'TN':6,'ABSTAIN':11,'INAPPLICABLE_CORRECT':7,'INAPPLICABLE_WRONG':4}
 c=collections.Counter(x['gold_violation'] for x in screen['cases'].values());assert c[True]==16 and c[False]==16
 summary={
  'date':'2026-10-24',
  'status':'PARTIAL: no tested image-only or vision-language judge qualified for reliable autonomous GUI/3D/video defect acceptance',
  'new_source_receipt_hashes_sha256':{'UIJudge_reanchored_real_site_manifest':screen_hash,'CLIP_blind_pixels':clip_hash,'Scry_shard_manifest':scry_src_hash,'Scry_frozen_pixel_predictions_before_gold':scry_raw_hash,'Scry_externally_human_annotated_scores':scry_eval_hash,'W3C_ACT_min_contrast_official_manifest':w3c_src_hash,'W3C_independent_flat_color_checker':w3c_eval_hash},
  'real_site_target_visible_capture':{'screenshots':len(screen['cases']),'actual_external_layout_defects':c[True],'clean_controls':c[False],'all_element_rectangles_intersect_capture':True,'not_all_defects_necessarily_visually_decidable':True,'crop_raw_pngs_external':True},
  'real_site_image_only_CLIP':{'model':clip['model'],'fixed_prompts':clip['prompts_predefined'],'run_seconds':clip['run_seconds'],'scores':clip['scores'],'status':'REJECTED on this task: heldout recall zero of three'},
  'human_annotated_paired_UI_Scry':{'dataset':'Scry Design Diff Eval 311 pairs, 557 human-tagged individual issues, 77 no-tagged control pairs, 52 app groups','download_bytes':scry_src['expected_total_bytes'],'calibration_and_heldout_grouped_by_app':scry_eval['folds'],'frozen_pixel_metrics':scry_eval['threshold_methods'],'chosen_using_calibration_only':scry_eval['selected_calibration_method'],'heldout_localization_without_defect_tag':localization,'total_prediction_cpu_seconds':scry_eval['raw_compute_seconds'],'no_tagged_caveat':'No-tagged controls are NOT independently certified defect-free; no-tagged flag rate is diagnostic rather than proven false-positive rate','localization_caveat':'Spatial IoU >= 0.10 against human boxes without predicting issue class, so cannot be compared with the official category-and-location recall reported for contemporary VLMs'},
  'W3C_ACT_color_contrast':{'rule':'afw4f7 WCAG 1.4.3 Minimum, independent W3C ACT testcases','testcase_count':34,'expected_counts':{'passed':12,'failed':11,'inapplicable':11},'checker':'Firefox 146 computed CSS colors, opacity/background approximation; no axe-core; frozen rule before expected outcomes scored','three_way_confusion':w3c_eval['confusion'],'aggregate_counts':w3c_eval['counts'],'decision':'CANNOT serve as complete contrast checker: abstentions and false applicability decisions, despite correct decisions on some flat-color cases','limitations':w3c_eval['limitations']},
  'research_necessary_next':['Human-adjudicated overlap and low-contrast screenshots across natural apps where defect is definitely observable','Head-to-head ScreenVLM (element parsing) and specialized UI-UX (reasoning) using frozen source + variant images with separate independent labels','CAD mechanical correctness from authoritative real 3D simulation geometry plus externally verified good/bad physically plausible actual renders','Naturally occurring defective and correct final 3D animation/video clips with independently marked frame-time defects, not only controlled injections','Source-specific model latency, GPU/VRAM/license limits under demonstrably free production capacity'],
  'raw_media_policy':'All Scry third-party app screenshot pairs, public W3C HTML, and reanchored real-site PNGs remain under /data/var, NOT tracked in Git. Reproduction files and source hashes are in the project research folder.'
 }
 OUT.write_text(json.dumps(summary,indent=2)+'\n')
 print('VERIFIED',OUT,'BYTES',OUT.stat().st_size,'Scry best',held,'W3C',w3c_eval['counts'])
if __name__=='__main__':main()
