#!/usr/bin/env python3
"""Source-linked aggregate for final hourly visual-analysis research continuation.

Read-only verification of independently preserved raw bench results from real
3D meshes, released human VLM annotations, published real/generated videos,
Spotlight motion-error intervals, and fixed random event-time baselines.
NO raw licensed media, personal data, or model binaries in project Git.
Does NOT invoke or repeat original external model requests or download steps.
"""
from pathlib import Path
import hashlib,json,collections,datetime
VAR=Path('/data/var/perceptual-independent-20261024')
OUT=Path(__file__).resolve().parents[2]/'docs/research/perceptual-third-continuation-evidence-2026-10-24.json'

def get(relative):
 p=VAR/relative
 data=p.read_bytes()
 return json.loads(data),hashlib.sha256(data).hexdigest()

def main():
 manifest,hm=get('rapidata-physics-video/frozen_human_pair_source_manifest.json')
 fps,hf=get('rapidata-physics-video/fps_only_preference_confound_20261024.json')
 generated,hg=get('rapidata-physics-video/six_generated_pair_transfer_20261024.json')
 raw_vid,hrv=get('rapidata-physics-video/24_video_temporal_raw_scores_20261024.json')
 human_vid,hhv=get('rapidata-physics-video/12_human_rated_real_vs_ai_temporal_comparison_20261024.json')
 votes,hv=get('3d-defectbench-2026-paper/expert_5vlm_fixed_vote_ensemble_20261024.json')
 mesh,hmesh=get('3d-defectbench/mesh_structure_vs_human_missing_parts_20261024.json')
 spotlight_src,hsm=get('spotlight-videogen/motion_human_24_video_manifest.json')
 spotlight_pred,hsp=get('spotlight-videogen/spotlight_24_motion_pixel_only_predictions.json')
 spotlight_score,hss=get('spotlight-videogen/spotlight_24_human_motion_eval.json')
 spotlight_chance,hsc=get('spotlight-videogen/spotlight_24_motion_random_peak_baseline.json')
 previous, hp=get_repo_prior()
 # Fixed exact original counts, not output-only self-referential "passed tests".
 assert len(manifest['samples'])==12 and len(human_vid['cases'])==12
 assert fps['real_vs_Cosmos']['n']==12 and fps['real_vs_Cosmos']['metadata_fps_matches']==12
 assert fps['Cosmos_vs_MiniMax']['n']==6 and fps['Cosmos_vs_MiniMax']['metadata_fps_matches']==6
 assert len(generated['pairs'])==6 and len(raw_vid['clip_metrics'])==24
 assert votes['results']['expert_missing_parts_all']['majority_3_of_5']['n']==99
 assert (votes['results']['expert_missing_parts_all']['majority_3_of_5']['TP'],votes['results']['expert_missing_parts_all']['majority_3_of_5']['FP'],votes['results']['expert_missing_parts_all']['majority_3_of_5']['FN'],votes['results']['expert_missing_parts_all']['majority_3_of_5']['TN'])==(8,8,23,60)
 assert votes['results']['expert_mechanical_missing_parts']['majority_3_of_5']['n']==30
 assert votes['results']['expert_mechanical_missing_parts']['majority_3_of_5']['TP']==1
 assert len(mesh['cases'])==10 and sum(x['gold_missing_parts'] for x in mesh['cases'])==4
 assert sum(x['metrics']['detector_flags']['any_open_or_nonmanifold_submesh'] for x in mesh['cases'])==10
 assert (mesh['comparison_to_human_labels']['gold_missing_parts']['coarse_aabb_overlap']['TP'],mesh['comparison_to_human_labels']['gold_missing_parts']['coarse_aabb_overlap']['FP'])==(4,3)
 assert spotlight_src['video_count']==len(spotlight_pred['predictions'])==len(spotlight_score['per_video'])==24
 assert len({x['pid'] for x in spotlight_src['cases']})==8
 assert sum(bool(c['motion_annotations']) for c in spotlight_src['cases'])==9
 assert sum(len(c['motion_annotations']) for c in spotlight_src['cases'])==11
 assert abs(spotlight_chance['chance']['random_expected_events_out_of_11']-6.6951)<.00001
 assert abs(spotlight_chance['comparisons']['warp_residual']['random_peak_ge_observed_pvalue']-.1028)<.00001
 assert spotlight_score['metric_summary']['warp_residual']['TP_on_labeled_motion']==5
 assert spotlight_score['metric_summary']['warp_residual']['FP_proxy_on_no_motion_labels']==0
 assert spotlight_score['metric_summary']['warp_residual']['events_localized_by_top3']==9
 assert len(previous['3D_DefectBench_human']['models'])==5
 assert previous['ScreenVLM_real_input_task_limited']['expected_labels_recovered']==0
 assert previous['ScreenVLM_real_input_task_limited']['test_cases']==4
 expected_sources={
  'rapidata_human_choices':hm,'rapidata_fps_confound':hf,'rapidata_generated_pair':hg,'rapidata_temporal_pixel_scores':hrv,'rapidata_human_vote_scoring':hhv,
  'expert_vlm_votes':hv,'human_glb_mesh_structure':hmesh,'spotlight_original_manifest':hsm,'spotlight_frozen_pixel_only_predictions':hsp,'spotlight_human_motion_labels_scored':hss,'spotlight_random_peak_chance':hsc,'prior_3d_gui_report':hp}
 excerpt={
  'date':'2026-10-24','research_ownership':'SWAAG visual perception, original owner testing/GUI/media-editing recordings','overall_status':'PARTIAL — no tested general vision/video judge qualified for autonomous acceptance','all_raw_source_and_prediction_SHA256':expected_sources,
  'real_physics_human_pair_confounded':{'pair_count_real_vs_Cosmos':12,'pair_count_Cosmos_vs_MiniMax':6,'native_fps_alone_agreed_with_human_preferences':18,'total_video_pair_evaluations':18,'test_method':'human aggregate source vote after frozen 12fps optical flow; FPS-only metadata diagnostic post-hoc','main_result':'Apparent 10/12 and 6/6 optical-flow/human preference agreement cannot imply physical reasoning; metadata native FPS alone gives 12/12 and 6/6, separate model pipelines and encodings confounded','independent_gold':'Rapidata physics human PREFERENCE, not true physical law or exact stutter time labels','relevance':'no-model physics judgement qualification'},
  'five_released_3d_human_VLM_votes':{'source':'3D-DefectBench expert-adjudicated 99 GLB/generative mesh assets, historical author VLM inference (not new local model calls)','five_models':list(votes['models']),'expert_missing_parts_majority_3_of_5':votes['results']['expert_missing_parts_all']['majority_3_of_5'],'expert_missing_parts_any_1_of_5':votes['results']['expert_missing_parts_all']['any_1_of_5'],'expert_mechanical_missing_parts_majority_3_of_5':votes['results']['expert_mechanical_missing_parts']['majority_3_of_5'],'expert_mechanical_missing_parts_any_1_of_5':votes['results']['expert_mechanical_missing_parts']['any_1_of_5'],'policy':'unweighted fixed votes any/3of5/4of5/5of5, all four scored; no threshold tuning or heldout statistical fit','finding':'ensemble consensus may reduce false alarms by missing most defects; not trustworthy physical-engineering QA'},
  'real_glb_deterministic_structure':{'source':'10 human-labelled actual 3D-DefectBench golden GLBs, four human majority missing-parts, six not','checked_meshes':10,'submeshes_all_open':10,'missingparts_as_nonfinite_coordinates':mesh['comparison_to_human_labels']['gold_missing_parts']['nonfinite_vertex_coordinates'],'missingparts_as_open_submeshes':mesh['comparison_to_human_labels']['gold_missing_parts']['any_open_or_nonmanifold_submesh'],'missingparts_as_coarse_AABB_overlap':mesh['comparison_to_human_labels']['gold_missing_parts']['coarse_aabb_overlap'],'physical_meaning':'GLB watertight is not necessary for every render asset, AABB overlap not narrowphase collision or penetration, mesh validity cannot detect absent semantic part'},
  'spotlight_human_temporal_errors':{'original_source':'Spotlight ECCV 2026 external human annotations, 600 original videos, 1604 localizable errors across motion, physics, semantics and anatomy; v1 revision pinned','selected_videos':24,'source_prompt_families':8,'source_generators':['ltx2','seedance','veo3'],'videos_with_human_motion_label':9,'videos_without_motion_label_NOT_GOLD_CLEAN':15,'separately_human_localized_motion_intervals':11,'low_level_optical_flow_raw_cpu_seconds':spotlight_pred['total_wall_seconds'],'low_level_analyzer_results':spotlight_score['metric_summary'],'chance_audit':spotlight_chance['chance'],'chance_comparison':{k:{'events_localized_by_top3':v['events_localized_by_top3'],'random_peak_ge_observed_pvalue':v['random_peak_ge_observed_pvalue'],'TP_labeled':v['TP_on_labeled_motion'],'FN_labeled':v['FN_on_labeled_motion'],'FP_proxy_unlabeled':v['FP_proxy_on_no_motion_labels']} for k,v in spotlight_chance['comparisons'].items()},'caveat':'Human motion annotations may be broad, one covers full clip. Out-of-labeled moments and videos not necessarily clean. Random-time baseline for same 3 peak slots detects average 6.70/11 and >=9/11 in about 10.3% cases, so 9/11 localizations not convincing. No video language model was locally tested on this sample; paper author results separate.'},
  'previous_local_specialized_model':{'ScreenVLM':'316M model, 4 real target crops, 0 full expected labels, ~273 CPU seconds per image, not a clipping classifier; full pages unmeasured','axe_core':'real axe-core 4.14.0 32 of 34 official W3C ACT text-contrast source cases executed, 7 confirmed failed, 7 passed, 9 N/A correctly, 8 inconclusive and 1 applicability mistake; 2 scripted cases not executed; component only','Thor_Qwen3VL':'original 3/4, clipped text wrongly accepted, acceptance gate FAILED'},
  'recommended_narrow_tools':{'instrumented_DOM_web':'tested axe-core WCAG contrast rules and DOM scroll/client/ancestor bounds with actual screenshot; must abstain unsupported cases','screenshot_UI':'expected-text OCR + independently rated visual evidence, screenshot-only CLIP/ScreenVLM not qualified; SSIM difference only change triage','diagrams':'authoritative graph topology/chart data/label manifest plus OCR and geometry; 42/134 source labels recovered by Tesseract PSM6 on source-backed diagrams, not semantic reasoning accuracy','industrial_3D':'actual source CAD world transforms, contacts/constraints/normals and multiview targeted verification, with human/independent good/bad mechanical reference; generic CLIP and released VLMs unqualified','video':'frame PTS/hash/source timelines plus semantic intentional cut/hold contract, optional flow visualization; no freeze detector or learned physics judge yet validated on user product'},
  'next_open':['externally grounded true clean and defective industrial 3D renders (correct linkage/contact, authentic source dimensions) rather than only generated geometry defects','human-verified naturally occurring video frame-drop/judder/artifact dataset with truly clean same codec/fps controls and frame timestamps; test modern video VLM and deterministic candidates','full-page specialized ScreenVLM/OmniParser & UI-UX on diverse real GUI clipping/overlap/contrast independently seen by humans','independently calibrated error rates and selection thresholds outside publicly consumed development datasets, broad host/resource/latency tests'],
  'evaluation_integrity':['Source videos and mechanical source assets are noncommercial licensed and remain exclusively in /data/var, no model weights or original private recordings in Git','All algorithms fixed before scoring but same public sample used for diagnostic discussion; not an unseen deployment holdout','No actual live trading/3D render/model GPU servers interrupted, no new Calendar events created; no denied tool action retried']
 }
 rendered=json.dumps(excerpt,indent=2)+'\n'
 if OUT.exists() and OUT.read_text()!=rendered:raise RuntimeError('existing aggregate differs, preserve old evidence and investigate')
 if not OUT.exists():OUT.write_text(rendered)
 print('VALIDATED_EVIDENCE_SAVED',OUT,'BYTES',OUT.stat().st_size,'STATUS',excerpt['overall_status'])

def get_repo_prior():
 p=Path(__file__).resolve().parents[2]/'docs/research/perceptual-source-calibrated-evidence-2026-10-24.json';data=p.read_bytes();return json.loads(data),hashlib.sha256(data).hexdigest()
if __name__=='__main__':main()
