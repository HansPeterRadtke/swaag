#!/usr/bin/env python3
"""Compact source-frozen different owner product/scene transfer research report."""
from pathlib import Path
import hashlib
import json

BASE=Path('/data/var/perceptual-independent-20261024/owner_dock_cross_source_transfer_20261026')
WORK=Path(__file__).resolve().parents[2]
PUBLIC=WORK/'docs/research/perceptual-dock-cross-source-video-transfer-2026-10-26.json'
DOC=PUBLIC.with_suffix('.md')
SOURCE_MAP={name:BASE/name for name in ('frozen_protocol.json','source_manifest.json','canonical_source_decoded.json','encodes.json','transfer_scored.json','firefox_transferred_output.json','actual_policy_replay.json')}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 if PUBLIC.exists() or DOC.exists():raise RuntimeError('Published report exists; no rewrite')
 p=json.loads(SOURCE_MAP['frozen_protocol.json'].read_text());m=json.loads(SOURCE_MAP['source_manifest.json'].read_text());e=json.loads(SOURCE_MAP['encodes.json'].read_text());r=json.loads(SOURCE_MAP['transfer_scored.json'].read_text());b=json.loads(SOURCE_MAP['firefox_transferred_output.json'].read_text());policy=json.loads(SOURCE_MAP['actual_policy_replay.json'].read_text())
 assert r['frozen_protocol_SHA']==sha(SOURCE_MAP['frozen_protocol.json'])
 assert r['source_manifest_SHA']==sha(SOURCE_MAP['source_manifest.json'])
 assert r['canonical_decoded_SHA']==sha(SOURCE_MAP['canonical_source_decoded.json'])
 assert r['encoded_variants_SHA']==sha(SOURCE_MAP['encodes.json'])
 assert b['status']=='PASS' and policy['status']=='NARROW_OFFLINE_POLICY_REPLAY_PASS'
 assert r['reference_dock_blend_SHA']==p['source_blend_SHA']==m['dock_scene_SHA']
 assert len(m['original_210_source_pngs'])==210
 assert r['meets_all_preregistered_tests'] and all(r['accepted_transfer_rule_checks'].values())
 assert r['confusion']['source_verified']=={'TP':9,'FN':0,'FP':0,'TN':831}
 assert r['confusion']['no_reference']=={'TP':3,'FN':6,'FP':0,'TN':831}
 cases={}
 for name,row in r['cases'].items():
  enc=e['variants'][name]
  assert enc['sha256']==row['real_encoded_video_sha256']
  g=sorted(int(x) for x in p['variants_with_original_1based_source_frame_mappings'][name])
  assert g==row['source_bound_alarmed_positions'] if name in ('holds_crf17','reversals_crf17') else (not row['source_bound_alarmed_positions'])
  cases[name]={'encoded_MP4_SHA256':row['real_encoded_video_sha256'],'frozen_edit_positions':row['known_frozen_edit_positions'],'PSNR_source_exact_frame_min_dB':row['source_bound_PSN_min_dB'],'PSNR_source_exact_frame_median_dB':row['source_bound_PSN_median_dB'],'source_method':row['source_bound'],'no_reference_method':row['no_reference'],'source_bound_predicted_frame_indices':row['source_bound_alarmed_positions'],'no_reference_predicted_frame_indices':row['no_reference_alarmed_positions'],'FFmpeg_freezedetect_start_seconds':row['official_freezedetect']['starts_seconds'],'FFmpeg_freezedetect_events_total':row['official_freezedetect']['events_total'],'PTS_monotonic':row['PTS']['nonmonotonic']==0,'PTS_max_error_s':row['PTS']['max_error_vs_15fps_sec'],'encoder_CRf':row['codec_crf'],'CPU_decode_wall_s':row['actual_decode_wall_s'],'CPU_both_feature_rules_wall_s':row['same_frames_both_rules_wall_s'],'ffprobe_wall_s':row['PTS']['wall_s'],'FFmpeg_freezedetect_wall_s':row['official_freezedetect']['wall_s']}
 prior=WORK/'docs/research/perceptual-owner-mp4-verified-temporal-faults-2026-10-26.json'
 earlier=json.loads(prior.read_text())
 assert earlier['methods']['reference_based_PSNR_28dB']=={'TP':9,'FN':0,'FP':0,'TN':621}
 report={'date':'2026-10-26','overall':'PARTIAL','test':'Warehouse v47 to independent dock v46 scene and finished MP4, frozen threshold/source-protocol cross-product transfer','source_prior_warehouse_report_sha256':sha(prior),'new_dock_source_final_mp4_SHA256':p['original_approved_dock_final_MP4_SHA'],'new_dock_source_blend_SHA256':p['source_blend_SHA'],'frozen_reference_warehouse_transfer_code_SHA256':p['original_rack_warehouse_threshold_source_file_SHA'],'source_version_and_original_png_sha_verification':'210 original dock Cycles PNGs independently match their original SHA manifest; dock final H264 source-to-Cycles min PSNR>35dB; no source missing','source_data_receipts_SHA256':{name:sha(path) for name,path in SOURCE_MAP.items()},'frozen_source_protocol_rule':p['primary_acceptance'],'preregistered_checks':r['accepted_transfer_rule_checks'],'preregistered_result':'PASS' if r['meets_all_preregistered_tests'] else 'FAIL','transferred_source_psnr_threshold_dB':r['previous_warehouse_threshold_reused'],'transferred_no_ref_gray_motion_threshold':r['previous_warehouse_no_ref_threshold_reused'],'current_source_dock_original_against_Cycles_min_PSNR_dB':r['actual_source_mp4_against_original_Cycles_PSN_min_dB'],'wrong_source_original_warehouse_reference_median_PSNR_dB':r['wrong_product_12_control_psnr_median_dB'],'wrong_source_ref_verdict':'REVIEW for all; unrelated product source hash disqualifies it regardless of image difference','frame_confusion':r['confusion'],'conditional_descriptive_95pct_exact_binomial_intervals':r['descriptive_95pct_exact_binomial_frame_level_CI'],'method_by_case':cases,'FFmpeg_freezedetect_negative_control':{'clean_dock_2_reencodes_events':sum(len(cases[k]['FFmpeg_freezedetect_start_seconds']) for k in ('clean_crf17','clean_crf23')),'faulted_dock_holds_events':len(cases['holds_crf17']['FFmpeg_freezedetect_start_seconds']),'faulted_dock_reversal_events':len(cases['reversals_crf17']['FFmpeg_freezedetect_start_seconds']),'different_from_warehouse_camera_hold':'New dock source uses continuous camera and no authorized final three-second static hold'},'browser_decoded_user_facing':{name:{'video_SHA256':x['SHA'],'actual_end_playback_seconds':x['playback']['last']['currentTime'],'success':x['playback']['success'],'wall_s':x['playback']['wall_s'],'source_frame_fault_candidates':x['source_frame_faults']} for name,x in b['cases'].items()},'source_attested_actual_SWAAG_policy_replay':{x['case']:{'verified_source_property':x['verified_frame_sequence_property'],'unverified_CV_proposal':x['generic_motion_proposal'],'wrong_product_reference':x['invalid_warehouse_reference']} for x in policy['cases']},'latency':r['compute'],'limits':r['uncertainty_limits']+['A different owner Blender scene and a different product is genuinely different source; the 9 faults were scripted on authentic output and NOT independently adjudicated natural failures','This validates detection under a predeclared ordered-frame integrity contract, NOT mechanical physics, photorealism, visual intention, game animation or AV sync','CLI policy replay is OFFLINE, SWAAG registered/user-facing live tool and GPU contention unproved','Source-content comparison can verify only an approved frame contract; clips legitimately edited without reference updates must REVIEW rather than auto-fail'],'video_binary_bytes_embedded_in_git':False,'GPU_used':False}
 PUBLIC.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
 x=cases
 c=r['confusion']
 cm=lambda q:f"{q['TP']} TP / {q['FN']} FN / {q['FP']} FP / {q['TN']} TN"
 t=r['compute'];encode_total=sum(t['variants_encode_s'].values());score_sum=t['source_verify_and_decode_png_seconds']+sum(t['variant_CPU_decode_s'].values())+sum(t['variant_both_rules_feature_CPU_s'].values())+sum(t['variant_ffprobe_CPU_wall_s'].values())+sum(t['variant_ffmpeg_freezedetect_wall_s'].values())
 DOC.write_text(f'''# Dock v46 heldout transfer of warehouse source-frame integrity checker — 26 October 2026

**Outcome: PREDECLARED NARROW TRANSFER PASS; entire GUI/3D/video perception objective remains PARTIAL.** The new source is the owner's completely different dock/loading-handling-kit Blender scene `{p['source_blend_SHA']}`, finished video `{p['original_approved_dock_final_MP4_SHA']}`. Dock shows a continuous real camera movement and outward-opening cabinet; the previous warehouse v47's three-second authorized camera hold does not exist here. This is a true different product/video/source, **not a new natural-fault/human gold benchmark**. Source frame identities were verified against all 210 pinned original Cycles PNG SHA values, plus scene and finished video SHA before any edits. The dock source MP4 versus its Cycles source frames has a **{r['actual_source_mp4_against_original_Cycles_PSN_min_dB']:.2f} dB** minimum PSNR and **{r['actual_source_mp4_against_original_Cycles_PSN_median_dB']:.2f} dB** median.

## Preregistered rule and exact controls

The frozen selection (SHA `{sha(SOURCE_MAP['frozen_protocol.json'])}`) was created **before decoding the original dock video or viewing any pixel differences**. It transferred the prior warehouse thresholds UNCHANGED: source PSNR less than **28 dB** is a *candidate verified frame sequence violation* only with exact bound reference; generic no-reference grayscale 240×135 adjacent delta less than **1** is merely a motion proposal; official FFmpeg `freezedetect` negative 50 dB, minimum 0.12 seconds. The frozen rule required (a) no source violations in both clean CRF 17 and CRF 23 native videos, (b) at least **8 of 9** deliberately changed source frames detected, (c) no more than **6** false flags on all unchanged frame positions, (d) original dock MP4 matched original Cycles images with minimum PSNR at least 28dB, and (e) exact monotonic 210 PTS at 15fps. **All five checks passed**, with no threshold retuning.

Dock faults (fixed before media processing): one-frame holds at indexes **{x['holds_crf17']['frozen_edit_positions']}**; three local two-frame reversals at indexes **{x['reversals_crf17']['frozen_edit_positions']}**. The three positive local events per class and the two clean negatives used real original dock MP4 frames. Four final H264 outputs were encoded, with two controls including a benign CRF 23 change, and all output SHA and source PNG links preserved outside Git.

| Actual *decoded* dock output, 210 frames each | Scripted altered frame positions | Verified source method TP/FN/FP/TN | No-reference method TP/FN/FP/TN | FFmpeg freeze starts |
|---|---:|---|---|---|
| Clean CRF 17 | 0 | {cm(x['clean_crf17']['source_method'])} | {cm(x['clean_crf17']['no_reference_method'])} | {x['clean_crf17']['FFmpeg_freezedetect_start_seconds']} |
| Clean CRF 23 | 0 | {cm(x['clean_crf23']['source_method'])} | {cm(x['clean_crf23']['no_reference_method'])} | {x['clean_crf23']['FFmpeg_freezedetect_start_seconds']} |
| Three real dock one-frame holds | 3 | {cm(x['holds_crf17']['source_method'])} | {cm(x['holds_crf17']['no_reference_method'])} | {x['holds_crf17']['FFmpeg_freezedetect_start_seconds']} |
| Three real dock pair reversals | 6 | {cm(x['reversals_crf17']['source_method'])} | {cm(x['reversals_crf17']['no_reference_method'])} | {x['reversals_crf17']['FFmpeg_freezedetect_start_seconds']} |
| **All 840 frame positions** | **9** | **{cm(c['source_verified'])}** | **{cm(c['no_reference'])}** | False camera-hold complaints on clean dock: **0** |

The transferred source rule detected **9/9 deliberately changed positions, missed zero and accused 0/831 unaltered positions**, even including the separately reencoded and more compressed clean control. Its descriptive exact-binomial 95% sensitivity limits are **{r['descriptive_95pct_exact_binomial_frame_level_CI']['sensitivity_source']}**, and the frame-level false-flag limits **{r['descriptive_95pct_exact_binomial_frame_level_CI']['false_positive_rate_source']}**. This does **not** estimate a population reliability: frames share one source video and only **six intervention episodes**; proper independent clip/film confidence limits are unavailable. The no-reference detector found 3/9 and missed all six reverse-swapped frames; FFmpeg found the three holds but no reversals. These negatives agree with the original warehouse results, now transferred to a genuinely different product.

## Version validity and final user-facing playback

An unrelated warehouse source video (same duration/fps/dimensions) compared to 12 clean dock frames has median PSNR **{r['wrong_product_12_control_psnr_median_dB']:.2f} dB**, much lower than a correct match. This is a deliberately **invalid reference**, never proof of an undesirable dock movie. The policy must return **REVIEW** when source product or version is wrong. Actual offline SWAAG acceptance-policy module replay returned PASS for both clean approved-sequence source properties and FAIL for the two mutated sequence properties; its no-ref CV proposals and invalid warehouse reference returned REVIEW. This is **not a verified live SWAAG registered tool invocation**.

An isolated Firefox native HTML5 viewer successfully played the more compressed clean CRF 23 and the reversal-corrupted CRF 17 files to their **14-second `ended` state**, with no media errors. Successful playback once again does not certify actual frame ordering or semantic animation. The clips are silent; AV synchronization remains unknown.

## Actual cost, uncertainty and remaining work

Raw approved dock video decoder extraction **{t['original_extract_s']:.3f} s**; original source image SHA verification/load **{t['source_verify_and_decode_png_seconds']:.3f} s**; four actual H264 encodes together **{encode_total:.3f} s**; four video CPU decodes, both rules on the same frames, PTS checks and independent FFmpeg processing totaled **{score_sum:.3f} s** of measured components inclusive of source PNG reload. All original files, finished media and source Blender version preserved unchanged; **no GPU model or production render was invoked**. The reporter does not equate batch sums to live one-call latency or infer human review cost. Timing is dependent on busy Nitro CPU and tested H264 settings.

**What is proved:** a source-attested deterministic ordered-frame integrity contract transfers unchanged from warehouse to a second original owner CG film across real encoding outputs. **What remains unproved:** naturally occurring human-annotated temporal defects, source-independent video semantics, native GUI human-rated clipping/contrast, exact visible diagram arrowheads, machine part and photorealism correctness, synchronized audio+video events, and the complete user-facing registered SWAAG workflow. Overall **PARTIAL**.

Reproducible code: `tests/research/owner_dock_cross_source_transfer_20261026.py` (`freeze`, `extract`, `encode`), `tests/research/score_owner_dock_source_transfer_20261026.py`, `tests/research/owner_dock_transfer_firefox_20261026.py`, `tests/research/owner_dock_transfer_policy_replay_20261026.py`, `tests/research/publish_owner_dock_transfer_20261026.py`. Each stage rejects overwriting frozen outputs; original PNG hashes, final MP4s, frame-level scores, and all prior effects are isolated under `/data/var/perceptual-independent-20261024/owner_dock_cross_source_transfer_20261026/`. No file is uploaded to a gallery, no old denied retrieval retried, and no calendar/ChatGPT message scheduled.
''')
 print('PUBLISHED_DIFFERENT_SOURCE_DOCK_TRANSFER',PUBLIC,sha(PUBLIC),'status',report['preregistered_result'])

if __name__=='__main__':main()
