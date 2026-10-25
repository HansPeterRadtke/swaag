#!/usr/bin/env python3
"""Append genuine new 12 original source video pair result to owner registry.

Only after 24 original source videos verified and pixel-based CV/FFmpeg signals
frozen BEFORE creator bug/normal labels, and compact scientific report saved.
Older VideoSham/TempGlitch author-model reports untouched; never present this
as human temporal fault onset/offset gold or qualified video quality. No
production effects, Calendar sends, GPU models, media or source paths in Git.
"""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2]/'docs/research'
DATA=ROOT/'perceptual-tempglitch-real-character-pose-video-control-2026-10-25.json'
REPORT='perceptual-tempglitch-frozen-pose-source-video-2026-10-25.md'
REG=ROOT/'perceptual-analyzer-registry-2026-10-24.json'
OWNER=ROOT/'perceptual-owner-reconciled-decision-2026-10-25.json'
OWNER_MD=ROOT/'perceptual-owner-reconciled-decision-2026-10-25.md'
KEY='external_real_Godot_Frozen_Character_Animation_vs_Normal_2026_10_25'

def run():
 a=json.loads(DATA.read_text());reg=json.loads(REG.read_text());owner=json.loads(OWNER.read_text())
 if KEY in reg:print('PRIOR_NEW_FROZEN_ANIMATION_RESEARCH_REGISTERED_NO_DUPLICATE');return
 assert a['original_independent_scene_pairs']==12 and a['original_video_files_CPU_decoded']==24
 assert len(a['observed_original_source_binary_and_rank_performance'])==5
 assert not a['run_resource']['GPU_used']
 if 'later_external_actual_gameplay_frozen_character_video' in owner:raise RuntimeError('original owner registered source, no duplicate')
 aggregate={}
 for method,x in a['observed_original_source_binary_and_rank_performance'].items():
  assert x['TP']+x['FN']+x['abstained_original_positive_n']==12
  assert x['FP']+x['TN']+x['abstained_original_normal_control_n']==12
  aggregate[method]={'TP':x['TP'],'FP':x['FP'],'FN':x['FN'],'TN':x['TN'],'abstained_author_bug':x['abstained_original_positive_n'],'abstained_author_normal':x['abstained_original_normal_control_n'],'AUROC_author_label':x['continuous_run_score_AUROC'],'source_pair_auc95_CI':x['source_scene_matched_bootstrap_95pct_CI']['AUROC']}
  reg['analyzers'].append({'task':'Video character animation pose frozen vs full-screen video freeze, real original Godot source footage','method':method,'model_version':'FFmpeg native 6.1.1 frozen video filter or original OpenCV grayscale region/phase/Farneback analysis at 5 FPS on Nitro CPU nice19, no trained VLM/character pose model','status':'SOURCE_ENGINE_CREATOR_LABELS_TESTED_ONLY_NOT_APPROVED_FOR_GENERAL_VIDEO_SEMANTIC_TEMPORAL_RELEASE_GATE','scope':'24 actual original Platformer source MP4s,12 creator-injected character frozen-animation fault clips +12 same scene/index author-normal clips, exact original source LFS SHA/byte verified. Creator injected labels not independent human subjective faults or time-coded event truth.','result':f"Original source author type TP{x['TP']}, FP{x['FP']}, FN{x['FN']}, TN{x['TN']}, abstain {x['abstained_original_positive_n']} bug/{x['abstained_original_normal_control_n']} normal, AUROC {x['continuous_run_score_AUROC']}, source-scene bootstrap95 {x['source_scene_matched_bootstrap_95pct_CI']['AUROC']}",'source_report':REPORT,'limitations':'Character animation freezing is not full framebuffer stall: world camera may move while actor pose frozen. No actor pixel bbox or 3D rig/pose ground truth, only guessed ROI; one small size-biased original game family. No human timestamped fault labels. Do not infer physical animation correctness or audiovisual sync. Source video outside Git.'})
 reg[KEY]={'status':'PARTIAL: genuinely new 24 ORIGINAL third-party gameplay video CPU visual fault measurement, but no qualified character pose, motion physics or natural audiovisual perceptual release gate','report':REPORT,'compact_source_hash_evidence':'perceptual-tempglitch-real-character-pose-video-control-2026-10-25.json','new_original_public_source':'TempGlitch original 2026 author engineered frozen CHARACTER animation vs normal matched Platformer scene, source selection hash before pixels, video license MIT dataset with third-party game imagery rights checked separately','source_original_n':'12 matched scene pairs, 24 distinct original videos, about717MB exact original SHA verified, no source result from owner video retrieved','method_error_counts_and_false_alarms':aggregate,'full_frame_vs_subject_differentiation':'Full-frame FFmpeg or frame-MAE can never prove absence of a character pose frozen while the rest of scene changes. Fixed ROIs/flow are imperfect, not semantic rig-motion analysis.','automated_video_default':'Verify approved frame/PTS/audio timeline and in-engine body pose/velocity constraints first when source available, and direct original video human/qualified ROI inspection for semantics. Never greenlight based on global freeze detector or VLM confidence.','invalid_preceding_read_only_test':'Initial ffmpeg filter string contained explanatory prose; six partial attempts stopped and invalid state preserved outside Git, no score. Correct new output path and filter verified; no production or other agent service affected.','no_effects':'No GPU, no source owner render, no Calendar replay/earlier denied 72-video or source-resolution retrieval, no license-restricted video in Git.'}
 REG.write_text(json.dumps(reg,indent=2)+'\n')
 owner['later_external_actual_gameplay_frozen_character_video']={'new_source_report':REPORT,'source_compact_SHA256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'original_clean_and_buggy_scene_pairs':12,'source_author_truth_not_human_MOS':'author-coded character frozen vs normal, no event-time masks, not genuine unplanned playback defect','source_measured_method_TP_FP_FN_TN':aggregate,'deployment_decision':'A frozen character pose can leave each video frame moving. FFmpeg whole frame is a narrow frame-stall check, not character animation quality. Need engine animation state plus actual seen motion and independent human event labels for release.'}
 OWNER.write_text(json.dumps(owner,indent=2,ensure_ascii=False)+'\n')
 md=OWNER_MD.read_text()
 tag='## Original-source video character freeze, not framebuffer freeze — later Round 4 source test'
 if tag not in md:
  md+='''\n\n## Original-source video character freeze, not framebuffer freeze — later Round 4 source test\n\nA new, distinct actual video test in [`perceptual-tempglitch-frozen-pose-source-video-2026-10-25.md`](perceptual-tempglitch-frozen-pose-source-video-2026-10-25.md) downloaded exactly **24 original Godot rendered videos** with officially verified SHA-256, twelve creator-injected *character animation frozen* faults and twelve source-scene-matched author normal clips. This is not a synthetic frame-drop test, the preceding VideoSham edit class study, or a new author-submitted Calendar message. Predictions from official FFmpeg native `freezedetect` and fixed frame, central ROI, camera-phase alignment, and optical-flow signals were frozen on the actual source pixels **before** joining author bug labels, and false alarm and missed positives on 12 real normal controls are in the compact source report. The source includes only one Platformer game engine and selection constrained to <=70MB per matched source pair. Author ground truth is an injected **pose animation fault**, not independent subjective human frame-onset ground truth or an unintended encoded video freeze. Even complete absence of whole-frame freeze does not prove that the character is animated correctly while backgrounds and camera continue changing. The physical engine pose/joint update and event timing is the authoritative source property; image motion or semantic VLM descriptions remain candidate evidence until externally validated. No production rendering/GPU, original video media in Git, uncertain Calendar send or earlier denied result was affected.\n'''
 OWNER_MD.write_text(md)
 print('NEW_INDEPENDENT_REAL_GAMEPLAY_VISUAL_MOTION_RESEARCH_REGISTERED',len(aggregate),'ALTERNATIVE_METHODS',KEY)
if __name__=='__main__':run()
