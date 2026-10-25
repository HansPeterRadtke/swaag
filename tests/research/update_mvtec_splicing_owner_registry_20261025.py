#!/usr/bin/env python3
"""Attach independently verified NEW industrial source test to owner registry.

Only after original MVTec LOCO 134 source images were SHA-verified, 134 CLIP/HSV
features frozen, normal-only thresholds saved, 90 source original test classes
scored, and compact companion report written. Extend existing analyzer registry
and owner direct-answer checkpoint without touching older model evidence or
concurrent project files. Source photos/weights stay /data/var, not Git.
"""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2]/'docs/research'
DATA=ROOT/'perceptual-mvtec-splicing-new-industrial-anomaly-evidence-2026-10-25.json'
REPORT='perceptual-mvtec-splicing-industrial-visual-review-2026-10-25.md'
REG=ROOT/'perceptual-analyzer-registry-2026-10-24.json'
OWNER=ROOT/'perceptual-owner-reconciled-decision-2026-10-25.json'
KEY='original_industrial_MVTec_LOCO_splicing_visual_2026_10_25'

def main():
 evidence=json.loads(DATA.read_text());r=json.loads(REG.read_text());owner=json.loads(OWNER.read_text())
 if KEY in r:
  print('PRIOR_ORIGINAL_INDUSTRIAL_RESULT_REGISTERED_NO_DUPLICATE');return
 assert evidence['sample']['total_raw_original_source_images']==134
 assert evidence['sample']['source_original_test_good']==30
 assert len(evidence['original_heldout_score_and_errors'])==3
 if 'later_original_industrial_splicing_camera_visual_evidence' in owner:raise RuntimeError('Owner result already exists: never duplicate')
 summaries={}
 for method,group in evidence['original_heldout_score_and_errors'].items():
  pooled=group['all_source_anomalies_vs_original_clean']
  logical=group['logical_vs_real_clean'];structural=group['structural_vs_real_clean']
  assert (pooled['TP']+pooled['FN'],pooled['FP']+pooled['TN'])==(60,30)
  assert (logical['TP']+logical['FN'],structural['TP']+structural['FN'])==(30,30)
  r['analyzers'].append({'task':'Industrial camera visual screening of source label normal versus logical and structural connector assembly anomalies','method':method+' source normal 24 images + validation-normal-only calibrated 20 images, then heldout 30 correct,30 logical defects,30 structural defects; source images externally curated','model_version':'Original MVTec LOCO AD splicing connectors, pinned HF replica 839fee8178c10d2e50afddc671238ba85d9975d9; frozen OpenAI CLIP ViT-B32 CPU only or no pretrained HSV baseline','status':'RESEARCH_ONLY_UNQUALIFIED_TO_AUTOMATICALLY_CERTIFY_3D_MACHINERY_PHOTOREALISM_OR_INDUSTRIAL_ASSEMBLY','scope':'Real MVTec LOCO original industrial photographs distinct from earlier 18 pushpin sample. CC BY-NC-SA noncommercial. 90 heldout original test with source normal versus logical/structural author labels, no test parameter tuning.','result':f"Pooled industrial defects TP={pooled['TP']} FP={pooled['FP']} FN={pooled['FN']} TN={pooled['TN']} ROC_AUC={pooled['source_original_test_ROC_AUC']:.3f}; logical TP={logical['TP']}/30, structural TP={structural['TP']}/30, false good flags={pooled['FP']}/30; threshold 95th pct 20 heldout source-validation good photos; AUC95CI={pooled['image_stratified_bootstrap_95pct_ROC_AUC']}",'source_report':REPORT,'limitations':'Pretrained full-image CLIP or CLIP patch proxy is not a certified industrial anomaly detector (not PatchCore/EfficientAD); in-domain 24 source normal references and 20 normal validation samples, just 30 test true good controls. Original category assigned by MVTec authors, no blind owner assessor. Source licensing restricts commercial use. Industrial photos not correct physical CAD support or warehouse photorealistic render.'})
  summaries[method]={'TP':pooled['TP'],'FP':pooled['FP'],'FN':pooled['FN'],'TN':pooled['TN'],'AUC':pooled['source_original_test_ROC_AUC'],'logical_correct':logical['TP'],'structural_correct':structural['TP'],'source_anomaly_threshold_fixed_before_test_labels':True}
 r[KEY]={'status':'PARTIAL: 134 actual MVTec industrial source photographs, 3 predeclared CPU visual detectors compared against original external clean/logical/structural labels; no full physical/photorealism acceptance model qualified','report':REPORT,'evidence':'perceptual-mvtec-splicing-new-industrial-anomaly-evidence-2026-10-25.json','original_pinned_source':'MVTec LOCO AD splicing connectors public HF 839fee8178c10d2e50afddc671238ba85d9975d9, CC BY-NC-SA research only, new independent source product vs older pushpins','source_counts':'24 train good, 20 validation good, 30 test true good, 30 test logical anomalies, 30 test structural anomalies','method_concise_original_TP_FP_FN_TN':summaries,'cannot_prove':'Visual camera defects in one assembly photo family DO NOT establish CAD load/geometry, correct 3D scene materials/shaders, physics, general human-perceived photorealism, or general low false alarms on unseen manufacturers. No commercial MVTec photo usage implied.','no_effects':'No GPU, model services, 3D renders, uncertain Calendar submissions, denied video retrieval or third-party media in Git.'}
 owner['later_original_industrial_splicing_camera_visual_evidence']={'source_report':REPORT,'source_compact_report_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'source_class_counts':'24 normal train;20 normal validation;test90 with normal30 logical30 structural30','screening_model_results':summaries,'owner_decision':'Use independently verified CAD/physical contact for physical soundness. CLIP or HSV camera-image anomaly scores can only nominate review where false negatives/alarms permit; cannot approve machine CAD or final photorealism automatically.','scope':'External original real industrial source photos under restrictive noncommercial license, not owner CGTrader renders.'}
 REG.write_text(json.dumps(r,indent=2)+'\n')
 OWNER.write_text(json.dumps(owner,indent=2,ensure_ascii=False)+'\n')
 print('EXISTING_ANALYZER_REGISTRY_AND_OWNER_DECISION_UPDATED_NEW_REAL_INDUSTRIAL_SOURCE',len(summaries),'METHODS',KEY)
if __name__=='__main__':main()
