#!/usr/bin/env python3
"""Insert only the verified NEW Screw Bag spatial model research into SWAAG.

Original external source MVTec LOCO screw_bag 134 photos, 90 heldout source
labels, normal-only calibration score and exact counts were independently
recorded before these project-level decisions. No source media, model weights,
pixel patches or anomaly source per-case info enters Git, no credential or
Calendar action. Preserve previous other-product MVTec splicing/pushpins runs.
"""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2]/'docs/research'
COMPACT=ROOT/'perceptual-mvtec-screwbag-new-spatial-anomaly-evidence-2026-10-25.json'
REG=ROOT/'perceptual-analyzer-registry-2026-10-24.json'
OWNER=ROOT/'perceptual-owner-reconciled-decision-2026-10-25.json'
OWNER_MD=ROOT/'perceptual-owner-reconciled-decision-2026-10-25.md'
K='new_independent_MVTec_screwbag_PaDiM_spatial_2026_10_25'
MODES=['ResNet18_global_cosine_nearest_normal','ResNet18_spatial_gaussian_mahalanobis_max','ResNet18_spatial_gaussian_mahalanobis_p95','HSV_spatial_color_nearest_normal']


def run():
 src=json.loads(COMPACT.read_text());reg=json.loads(REG.read_text());owner=json.loads(OWNER.read_text())
 if K in reg:print('PRIOR_SOURCE_LINKED_NEW_SCREWBAG_REGISTRY_RESULT_PRESERVED');return
 assert set(src['heldout_original_source_tp_fp_fn_tn_and_CI'])==set(MODES)
 assert src['original_photo_counts']['new_source_photos_total']==134
 assert src['original_photo_counts']['heldout_good']==30
 assert src['original_photo_counts']['heldout_logical']==src['original_photo_counts']['heldout_structural']==30
 assert src['GPU_allocated'] is False
 items={}
 for mode in MODES:
  d=src['heldout_original_source_tp_fp_fn_tn_and_CI'][mode]
  allout=d['all_anomalies_vs_clean'];lo=d['logical_vs_clean'];st=d['structural_vs_clean']
  assert (allout['TP']+allout['FN'],allout['FP']+allout['TN'])==(60,30)
  assert (lo['TP']+lo['FN'],st['TP']+st['FN'])==(30,30)
  items[mode]={'source_original_true_logical':lo['TP'],'source_original_true_structural':st['TP'],'FP_on_30_good':allout['FP'],'TP_of_60_positives':allout['TP'],'FN_of_60_positives':allout['FN'],'TN_of_30_controls':allout['TN'],'AUROC':allout['test_AUROC'],'AUROC95_bootstrap':allout['original_source_image_stratified_bootstrap95']['AUROC'],'normal_only_95percent_threshold':allout['normal_only_calibration_threshold']}
  reg['analyzers'].append({'task':'Original real industrial screw-bag logical/component-completeness and structural abnormality screening','method':mode,'model_version':'Official pretrained Torchvision ResNet18 ImageNet1K V1 verified f37072fd SHA or no-pretraining HSV, 32 fixed per-pixel channels and normal-only Gaussian shrinkage for spatial probes; NOT official PaDiM/PatchCore','status':'TRIAGE_ONLY_NOT_ACCURATE_ENOUGH_FOR_AUTOMATIC_INDUSTRIAL_OR_3D_CAD_ACCEPTANCE','scope':'New MVTec LOCO AD screw_bag source category, externally labelled original real photos 24 normal train20 normal validation30 clean test30 logical30 structural; CC BY-NC-SA noncommercial and original source outside Git','result':f"Original heldout images TP {allout['TP']}, FP {allout['FP']}, FN {allout['FN']}, TN {allout['TN']}, ROC AUC {allout['test_AUROC']:.3f}, 95 CI {items[mode]['AUROC95_bootstrap']}, logical TP {lo['TP']}/30 and structural TP {st['TP']}/30 at a 20-normal-only validation 95p fixed threshold",'source_report':'perceptual-mvtec-screwbag-padim-spatial-original-photos-2026-10-25.md','limitations':'One different original object/source product family, author external labels not new blind human reannotation; no proof of mechanical CAD contacts/photorealistic source render; far too many missed logical parts, and 30 correct test samples too few to bound production false alarm. Research PaDiM-inspired proxy not official PaDiM.'})
 reg[K]={'status':'PARTIAL: genuine NEW industrial product category, official ImageNet pretrained ResNet18 spatial Gaussian and same normal-only budget global ResNet/HSV alternatives tested; all miss many actual source logical and structural faults and not approved automatic inspection','dated_report':'perceptual-mvtec-screwbag-padim-spatial-original-photos-2026-10-25.md','new_source_pinned_evidence':'perceptual-mvtec-screwbag-new-spatial-anomaly-evidence-2026-10-25.json','original_source_sha256':hashlib.sha256(COMPACT.read_bytes()).hexdigest(),'physical_source_rights':'Original MVTec 134 photographs CC BY-NC-SA and all ResNet ImageNet weights/patch embeddings retained /data/var; not permitted commercial product training from those pixels.','independent_sample':'New screw_bag 24 normal-train+20 normal-calibration+30 normal controls+30 logical+30 structural, no source test labels used in feature scoring/thresholding. Distinct from earlier splicing connectors and pushpins.','all_predeclared_method_outcomes':items,'operational_result':'PaDiM-inspired spatial p95 detects a few more structural errors than global pretrained embedding on this new product class, but recall low, source test insufficient for acceptance of whole industrial assemblies/3D graphics. Need source CAD/physical rules and separate visual review.','no_production_effects':'No GPU or owner render load, no Calendar or earlier uncertain delivery replay, licensed source images and 134 embeddings outside Git.'}
 regout=json.dumps(reg,indent=2)+'\n'
 assert 'later_MVTec_screwbag_spatial_source_results' not in owner
 owner['later_MVTec_screwbag_spatial_source_results']={'source_new_original_industrial_product':'MVTec LOCO screw_bag external author labels, 134 real original photos including 90 clean/defective test, distinct from earlier splicing/pushpins','source_report':'perceptual-mvtec-screwbag-padim-spatial-original-photos-2026-10-25.md','source_compact_report_SHA256':hashlib.sha256(COMPACT.read_bytes()).hexdigest(),'methods_with_independent_clean_negative_counts':items,'owner_recommendation':'The spatial ResNet anomaly detector was a modest ranker improvement but at frozen source-normal threshold misses many logical and structural defects. Use only for visual triage; physical load/contact assembly integrity remains source-CAD invariant plus qualified inspection. No universal quality gate.'}
 md=OWNER_MD.read_text();tag='## New genuine third MVTec industrial source class, spatial anomaly proxy versus controls'
 if tag not in md:
  md+='''\n\n## New genuine third MVTec industrial source class, spatial anomaly proxy versus controls\n\nThe additional research report [`perceptual-mvtec-screwbag-padim-spatial-original-photos-2026-10-25.md`](perceptual-mvtec-screwbag-padim-spatial-original-photos-2026-10-25.md) extends the industrial positive/negative evidence beyond pushpins and splicing connectors to original **MVTec LOCO Screw Bag** photographs. A pinned separate external MVTec author-labeled source supplied **24 new normal-train,20 normal-calibration and90 heldout original test photos** containing **30 actual normal,30 logical and30 structural defects**, with no overlap across splits or earlier SWAAG MVTec categories. Frozen CPU ResNet18 ImageNet pretrained full-image embeddings, a **PaDiM-inspired 28×28 per-location multivariate Gaussian source-normal anomaly detector** with fixed32 features and0.01 covariance regularization, and the same-budget no-pretraining HSV spatial baseline were evaluated. All 134 image-only predictions and source-normal-only thresholds were frozen before author test labels were compared. Real source photo and model license bytes remain outside Git, original photos noncommercial CC BY-NC-SA.\n\n**None is a reliable acceptance judge for industrial component completeness or owner 3D photorealism.** The spatial p95 Gaussian showed better ranking than global ResNet18 and HSV on this one source class, but was calibrated to accept more than seventy percent of actual structural or logical anomalies. It cannot certify missing screws/required parts, and this original external industrial camera benchmark is not a physical CAD or rendered 3D assembly benchmark. Full numeric heldout true/false positives and negatives and source SHA receipts are in the linked report. The original first-tier automatic fail for authoritative CAD relationship and support-path violations remains justified; unverified learned image anomalousness triggers review only.\n'''
 REG.write_text(regout)
 OWNER.write_text(json.dumps(owner,indent=2,ensure_ascii=False)+'\n')
 OWNER_MD.write_text(md)
 print('NEW_THIRD_INDUSTRIAL_CATEGORY_ADDED_TO_OWNER_SOURCE_REGISTRY',len(items),'METHODS_AND_NUMERIC_SOURCE_ERRORS_PRESERVED')
if __name__=='__main__':run()
