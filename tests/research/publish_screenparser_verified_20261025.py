#!/usr/bin/env python3
"""Publish independently verified whole-site ScreenParser inference evidence only.

No model invocation, no source-download, no GPU. Original screenshot files,
third-party source HTML, raw 32-case predictions kept outside Git.
Check source/manifests/metrics and basic error-pair results; this is only
UI-element localization, NOT visual defect classification/acceptance.
"""
from pathlib import Path
import hashlib,json,collections
DATA=Path('/data/var/perceptual-independent-20261024')
PROJECT=Path(__file__).resolve().parents[2]
OUT=PROJECT/'docs/research/perceptual-screenparser-verified-evidence-2026-10-25.json'


def load(rel):
 p=DATA/rel;raw=p.read_bytes()
 return json.loads(raw),hashlib.sha256(raw).hexdigest()


def main():
 if OUT.exists():
  print('VERIFIED_AGGREGATE_ALREADY_EXISTS_DO_NOT_OVERWRITE',OUT)
  return
 model,mhash=load('models/ScreenParser-YOLO/model_manifest.json')
 capture,chash=load('screenparser-fullpage-real/fullpage_manifest.json')
 raw,rhash=load('screenparser-fullpage-real/screenparser_yolo_32_frozen_predictions.json')
 evaldata,ehash=load('screenparser-fullpage-real/screenparser_yolo_32_target_localization_eval.json')
 paired,phash=load('screenparser-fullpage-real/screenparser_paired_source_audit_20261025.json')
 assert model['revision']=='f029e565f1206577402e43206454522075be3f72'
 assert model['weight_bytes']==153259543
 assert model['sha256']=='dbcb4f583ccfdb8100a68e606525c247890a2de4c1a54b14741e0ee29ce0ab88'
 assert len(capture['cases'])==len(raw['predictions'])==len(evaldata['cases'])==32
 assert raw['status']=='complete' and len(paired['paired_cases'])==16
 assert sum(r['independent_defect_gold'] for r in evaldata['cases'])==16
 assert sum(1 for x in capture['cases'].values() if x['status']!='ok')==0
 assert sum(1 for x in evaldata['cases'] if x['best_class_agnostic_iou']>=.5)==11
 assert (evaldata['summary']['defective']['iou_hits']['0.5'],evaldata['summary']['control']['iou_hits']['0.5'])==(3,8)
 assert paired['paired_stats']['clean_higher_IoU_than_bad_count']==15
 assert paired['paired_stats']['same_pixel_pair_count']==1
 assert len({x['family'] for x in paired['paired_cases']})==15
 assert raw['checkpoint_sha256']==model['sha256']
 assert raw['source_manifest_sha256']==chash
 assert evaldata['raw_detector_output_sha256']==rhash
 assert paired['frozen_predictions_SHA256']==ehash
 assert paired['capture_manifest_SHA256']==chash
 assert raw['config']['device']=='cpu'
 labels=collections.Counter(p['label'] for c in raw['predictions'].values() for p in c['predictions'])
 report={
  'date':'2026-10-25',
  'status':'COMPLETE narrow test: real site UI ELEMENT PARSING only; NOT a qualified clipping/overlap/contrast DEFECT classifier',
  'model':'docling-project/ScreenParser YOLO11-L, ScreenParse-v2 55 classes, source pinned f029e565f1206577402e43206454522075be3f72',
  'model_weight_license':'Apache-2.0 source card, 153.26MB; Ultralytics 8.3.221 runtime AGPL-3.0/enterprise is a separate commercial-license issue',
  'runtime':'Nitro RTX 3060 not used; loaded production-host CPU one torch thread, 1280-pixel input, raw source native 1920x994 Firefox raster, device=cpu',
  'source_sha256':{'model_manifest':mhash,'captured_Firefox_32':chash,'frozen_YOLO_predictions':rhash,'direct_target_localization_scores':ehash,'source_paired_analysis':phash},
  'case_scope':{'actual_websites_or_site_families':15,'paired_clean_defective_comparisons':16,'frozen_real_Firefox_screenshots':32,'known_browser_defective_targets':16,'paired_clean_controls':16,'criterion_types':['redecheck:element-protrusion','redecheck:viewport-protrusion'],'screenshot_dimensions_px':[1920,994],'note':'Not exhaustively annotated all UI boxes. Source target DOM bounding box may include clipped invisible CSS extent, so IoU is only a narrow target localization metric, not parser average precision.'},
  'model_inference':{'config':raw['config'],'source_results_finished':32,'actual_mean_seconds_per_image':evaldata['latency_mean_s'],'total_cpu_seconds':raw['cpu_elapsed_seconds'],'peak_RSS_KB':raw['rss_peak_kb'],'runtime_distribution':paired['resource'],'observed_nonexhaustive_element_class_counts':dict(labels)},
  'target_box_result':{'all':evaldata['summary']['all'],'clipped':evaldata['summary']['clipped'],'viewport_protrusion':evaldata['summary']['viewport_protrusion'],'defective':evaldata['summary']['defective'],'clean_control':evaldata['summary']['control']},
  'paired_readonly_analysis':paired['paired_stats'],
  'conclusion':'The checkpoint demonstrably outputs many plausible UI-element boxes from real Firefox whole-viewport screens. Yet independent source target IoU >=0.5 in only 11/32 cases (3/16 defect target vs 8/16 controls); paired IoU lower for 15/16 defective source boxes. This is not evidence of valid standalone defect inspection, and visual clipping changes source CSS target vs visible parsed UI geometry. CPU latency is too high for live quality gating on busy Nitro. Precise DOM client-scroll bounds + standard accessibility checks remain preferable for narrow measurable properties; YOLO could help find element candidates in screenshot-only cases only after separate full element precision/recall and text validation.',
  'remaining_qualification':['Exhaustive human-labelled UI box/text class ground truth, not one source selected target','True visible defects and non-defects matched for scale and rendering/intent, not only source CSS-bbox mutations','ScreenParser alternate runtime/license, speed and model size on genuinely idle hardware','Complete production GUI clip/overlap/contrast decision accuracy, false alarms, abstain policy','Physical 3D and temporal video quality are separate tasks not addressed by this model'],
  'licensed_source_media_in_git':False,
 }
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('SCREENPARSER_32_FINISHED_VERIFIED',OUT,'BYTES',OUT.stat().st_size,'MEAN_SECONDS',report['model_inference']['actual_mean_seconds_per_image'],'ALL_IOU_HALF',report['target_box_result']['all']['iou_hits']['0.5'],'PAIRED_SOURCE_DIFF_POSITIVE',report['paired_readonly_analysis']['clean_higher_IoU_than_bad_count'])
if __name__=='__main__':main()
