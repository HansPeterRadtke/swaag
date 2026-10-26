#!/usr/bin/env python3
"""Compact shareable source-audited report for real owner-film temporal faults."""
import hashlib
import json
from pathlib import Path

BASE=Path('/data/var/perceptual-independent-20261024/owner_real_encoded_fault_injection_20261026')
REPO=Path(__file__).resolve().parents[2]
DOC=REPO/'docs/research/perceptual-owner-mp4-verified-temporal-faults-2026-10-26.md'
JSON=DOC.with_suffix('.json')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 if DOC.exists() or JSON.exists():raise RuntimeError('Preserve previously published receipt, no overwrites')
 paths={k:BASE/k for k in ('frozen_design.json','source_receipts.json','source_frames.json','encodes.json','final_video_fault_score.json','browser_faulted_video_and_clean_control_20261026.json','source_attested_policy_replay_20261026.json')}
 source=json.loads(paths['source_receipts.json'].read_text())
 frame=json.loads(paths['source_frames.json'].read_text())
 enc=json.loads(paths['encodes.json'].read_text())
 data=json.loads(paths['final_video_fault_score.json'].read_text())
 browser=json.loads(paths['browser_faulted_video_and_clean_control_20261026.json'].read_text())
 policy=json.loads(paths['source_attested_policy_replay_20261026.json'].read_text())
 assert data['frozen_manifest_sha256']==sha(paths['source_receipts.json'])
 assert data['frozen_encoded_variant_sha256']==sha(paths['encodes.json'])
 assert data['source_decoded_original_PNG_SHA256']==sha(paths['source_frames.json'])
 assert policy['final_frame_scores_SHA']==sha(paths['final_video_fault_score.json'])
 assert browser['status']=='PASS' and all(m['playback']['success'] for m in browser['measurements'].values())
 assert source['video_sha256']==frame['source_video_sha256']==data['source_SHA256']
 assert frame['all_210_frames_complete'] and len(frame['exact_png_hashes'])==210
 assert policy['runtime_service_integration_tested'] is False
 breakdown={}
 for name,v in data['cases'].items():
  m=v['source_bound_detection'];n=v['no_reference_low_motion_detection'];known=set(v['frame_gold_synthetic_positions'])
  gold_edit_count=len(known);assert m['TP']+m['FN']==gold_edit_count
  assert m['FP']+m['TN']==210-gold_edit_count
  assert n['TP']+n['FN']==gold_edit_count
  assert n['FP']+n['TN']==210-gold_edit_count
  flagged_source=[x['frame'] for x in v['decoded_frame_indices'] if x['source_flag']]
  flagged_noref=[x['frame'] for x in v['decoded_frame_indices'] if x['no_ref_flag']]
  assert len(flagged_source)==m['TP']+m['FP']
  assert len(flagged_noref)==n['TP']+n['FP']
  breakdown[name]={'video_SHA256':v['file_SHA256'],'script_known_altered_frames_1based':v['frame_gold_synthetic_positions'],'source_bound_confusion':m,'no_reference_low_motion_confusion':n,'source_bound_predicted_frame_indices':flagged_source,'no_reference_predicted_frame_indices':flagged_noref,'FFmpeg_freezedetect_events_start_s':v['official_ffmpeg_freezedetect']['event_start_seconds'],'FFmpeg_freezedetect_intended_last_hold_event':v['official_ffmpeg_freezedetect']['events_in_authorized_last_3sec'],'FFmpeg_freezedetect_unintended_moving_region_events':v['official_ffmpeg_freezedetect']['events_in_moving_part'],'min_native_source_comparison_PSNR_dB':v['source_quality']['psnr_min_dB'],'median_native_source_comparison_PSNR_dB':v['source_quality']['psnr_median_dB'],'last_45_frame_camera_hold_gray_motion_median':v['real_authorized_hold']['gray_motion_median'],'latency':{'native_frame_decode_plus_both_candidate_rules_wall_s':v['full_decode_and_metric_wall_s'],'ffprobe_pts_wall_s':v['PTS']['seconds'],'FFmpeg_freezedetect_wall_s':v['official_ffmpeg_freezedetect']['wall_s']},'timebase':v['PTS'],'Gold_from_script_not_human':True}
 M={k:sum(breakdown[n][k]['TP'] for n in breakdown) for k in ('source_bound_confusion','no_reference_low_motion_confusion')}
 N={k:sum(breakdown[n][k]['FN'] for n in breakdown) for k in ('source_bound_confusion','no_reference_low_motion_confusion')}
 FP={k:sum(breakdown[n][k]['FP'] for n in breakdown) for k in ('source_bound_confusion','no_reference_low_motion_confusion')}
 TN={k:sum(breakdown[n][k]['TN'] for n in breakdown) for k in ('source_bound_confusion','no_reference_low_motion_confusion')}
 assert (M['source_bound_confusion'],N['source_bound_confusion'],FP['source_bound_confusion'],TN['source_bound_confusion'])==(9,0,0,621)
 assert (M['no_reference_low_motion_confusion'],N['no_reference_low_motion_confusion'],FP['no_reference_low_motion_confusion'],TN['no_reference_low_motion_confusion'])==(3,6,0,621)
 assert breakdown['clean_reencoded']['FFmpeg_freezedetect_events_start_s']==[11.0]
 assert breakdown['short_reversals']['FFmpeg_freezedetect_unintended_moving_region_events']==0
 assert breakdown['single_frame_holds']['FFmpeg_freezedetect_unintended_moving_region_events']==3
 assert all(v['FFmpeg_freezedetect_intended_last_hold_event']==1 for v in breakdown.values())
 wall_decode=frame['decode_wall_s'];wall_encode=sum(x['encode_wall_s'] for x in enc['variants'].values())
 wall_scoring_load=data['resource']['source_verify_and_load_wall_s']
 wall_scoring_computation=sum(x['latency']['native_frame_decode_plus_both_candidate_rules_wall_s'] for x in breakdown.values())
 wall_pts=sum(x['latency']['ffprobe_pts_wall_s'] for x in breakdown.values())
 wall_ff=sum(x['latency']['FFmpeg_freezedetect_wall_s'] for x in breakdown.values())
 audit=[
 {'question':'Exact GUI clipping, overlap, text contrast, OCR','measured':'UI-like human-reviewed SlideAudit OCR clipping 58/82 TP and 26/82 FP; new triple-unanimous-negative original slides OCR flags 26/72, generic CLIP flags 66/72','status':'PARTIAL','missing':'independently human-rated real native owner applications with true dynamic controls and source/output identity'},
 {'question':'Exact diagrams: source graph/labels/visible directional arrows','measured':'TopoBench uniquely OCR-grounded 198 of 966 source edges; 151 detected, 158 false links; directions 80/147 correctly guessed without arrowheads','status':'PARTIAL','missing':'independently verified actually drawn arrowheads, source graph-to-pixel correspondence and heldout direction scores'},
 {'question':'Real 3D mesh instances, industrial part completeness, support/contact','measured':'owner v44 0/72 steel-to-rack two-end contacts vs corrected v47 72/72, one correlated engineered source; 119 generated GLBs AABB 9/33 missing part TP,20/86 FP','status':'PARTIAL','missing':'engineering-certified unrelated assemblies; independent finished-owner-render part/photorealism ratings'},
 {'question':'Final encoded user-facing output and authorized camera hold','measured':'original approved silent warehouse 210/210 decoded frames, 15fps PTS, source PSNR min33.28dB and median35.70dB, native Firefox reached ended 14s; intended 3s camera hold','status':'NARROW_PASS','missing':'human quality and physics judgments cannot be inferred from source-frame similarity'},
 {'question':'Temporal video faults, false alarms, source-vs-no-source','measured':'this study 9 scripted frame faults in actual warehouse MP4: source detector 9/9 TP, 0/621 FP; no-ref low motion 3/9 TP; FFmpeg flags true approved tail in all 3 videos','status':'NARROW_PASS_ON_SCRIPTED_FAULTS','missing':'time-coded naturally occurring independent-human/expert video fault labels, different products/codecs and latency on production hardware'},
 {'question':'Audio events and A/V synchronization in complete captured gameplay/movie','measured':'approved source movie has no audio stream; source archive and browser playback cannot establish actual audio event alignment','status':'UNVERIFIED','missing':'recorded real audiovisual event-timed independent audio/video ground truth'},
 {'question':'SWAAG live agent toolkit and source-backed release gate','measured':'existing conservative property policy applied offline to exact encoded video receipts: clean PASS and 2 altered FAIL; 3/3 visual or FFmpeg proposals remain REVIEW','status':'OFFLINE_VERIFIED_ONLY','missing':'true live SWAAG registered tool call, permission/version binding and user-facing error/review behavior'},
 ]
 report={'date':'2026-10-26','status':'PARTIAL','source':'Actual original final owner-approved 210-frame v47 silent warehouse MP4, not fabricated synthetic image or metadata-only test','source_video_original_SHA':source['video_sha256'],'source_3D_blend_original_SHA':source['source_blend_sha256'],'source_authorized_final_hold_frames_1based':source['author_intent_hold_indices_1based'],'frozen_SHA256':{k:sha(v) for k,v in paths.items()},'prior_real_final_video_source_210_Sources_verified':True,'variant_count':3,'positive_scripted_frames':9,'source_checked_frame_population_total':630,'methods':{'reference_based_PSNR_28dB':{'TP':9,'FN':0,'FP':0,'TN':621},'no_reference_low_motion_under1_bounded_early':{'TP':3,'FN':6,'FP':0,'TN':621},'FFmpeg_freezedetect_minus50dB_0p12s':{'flagged_approved_camera_hold_per_video':1,'temporal_edited_stall_events_detected':3,'reversal_events_detected':0}},'cases':breakdown,'browser_user_facing':{k:{'source_video_SHA':v['sha256'],'browser_playback_success':v['playback']['success'],'browser_video_endpoint_seconds':v['playback']['last']['currentTime'],'browser_media_error':v['playback']['last']['error'],'browser_wall_time_s':v['playback']['wall_s'],'source_sequence_fidelity_fault_flags':v['source_FIDELITY_flags']} for k,v in browser['measurements'].items()},'policy_offline':{x['name']:{'source_verified_verdict':x['source_verified_verdict'],'unverified_motion_verdict':x['no_reference_verdict'],'ffmpeg_tail_verdict':x['official_ffmpeg_tail_property_verdict']} for x in policy['verified_source_policy_outcomes']},'resources':{'original_video_to_lossless_PNG_decode_wall_s':wall_decode,'new_video_H264_encode_wall_s_each':{k:v['encode_wall_s'] for k,v in enc['variants'].items()},'three_new_video_H264_encode_wall_total_s':round(wall_encode,3),'source_original_PNG_verify_and_load_s':wall_scoring_load,'three_video_native_frame_decode_and_same_budget_metrics_wall_s':round(wall_scoring_computation,3),'three_video_ffprobe_PTS_wall_s':round(wall_pts,3),'three_video_official_FFmpeg_freezedetect_wall_s':round(wall_ff,3),'measured_scoring_wall_sum_s':round(wall_scoring_load+wall_scoring_computation+wall_pts+wall_ff,3),'source_bound_and_no_reference_methods_share_frame_decode':True,'GPU_used':False,'time_to_conduct_browser_viewer_2_videos_s':round(sum(v['playback']['wall_s'] for v in browser['measurements'].values()),3),'full_total_includes_extra_process_overhead_unmeasured':True},'original_goal_audit':audit,'limitations':['Synthetic frame edits created by fixed independent script on REAL approved owner video; NOT independent external human-labeled natural defects','Frames from same video are highly correlated; 9 positives and 621 negatives cannot be interpreted as 630 independent production videos or real-world error-rate confidence intervals','Source-bound PSNR checks a predefined ordered frame sequence tolerance, not subjective motion correctness or video quality across unknown edits','No-reference early low-motion rule intentionally exempted author-approved camera hold by source policy. Without that policy the final hold creates false alerts','Even deliberately altered film reaches HTML5 ended; playback success and normal PTS cannot certify user-perceived correctness','Original film has no audio track and cannot validate AV synchronization','No live SWAAG registered tool invocation. Policy evaluation uses actual source-attested evidence offline.','No production media, 3D render, gallery, model service or previously denied retrieval modified or retried.']}
 JSON.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 DOC.write_text(f'''# Owner GUI / diagram / 3D / video perception research — source-frozen final MP4 temporal acceptance (26 October 2026)

**Overall PARTIAL.** Original owner September 23 testing recording read again in full, plus current `/data/infra/AGENTS.md`, UI test/acceptance guidelines and all recent relevant SWAAG source reports. Research branch was clean at `a6f0f42`. We audited all original questions one by one (machine-readable audit in accompanying compact JSON). The most valuable feasible unresolved new empirical test was detecting actual final owner encoded movie **time-localized** frame sequence faults while preserving the legitimate source-approved final hold, not repeating prior slides or benign MP4 timestamp checks.

## A real finished product, frozen faults, and same-input controls

Original source is the owner's final 14-second 960×540, 15fps, 210-frame silent warehouse H264 movie at immutable SHA-256 `{source['video_sha256']}`, scene SHA `{source['source_blend_sha256']}`. The preceding independent production movie study verified all 210 original Cycles-rendered source PNG hashes and Firefox HTML5 playback. In this round an entirely isolated **additional** decoding of the actual final MP4 yielded 210 lossless PNG frames and source SHA ledger `{sha(paths['source_frames.json'])}`. Before decoding, the script froze a manifest of explicit **synthetic** edits: replace individual film frames **40,91,145** by their immediate predecessor; reverse each local pair **55–56,105–106,150–151**. Original footage, camera hold frames 166–210, resolution, output count, H264 preset (`fast`, CRF17) and FPS were held unchanged. This yields one clean reencoded control and two edited 14s clips, all real decoded/output H264 images. **Gold labels are independently fixed script mutations, not independently human-rated natural errors.**

| Full decoded video result, same 210 frames and same H264 settings | Scripted changed frames | Source-bound per-frame PSNR <28 dB, TP/FN/FP/TN | No-ref pixel motion under 1.0, TP/FN/FP/TN | Official FFmpeg `freezedetect` |
|---|---:|---|---|---|
| Clean re-encoded real owner movie | 0 | 0 / 0 / 0 / 210 | 0 / 0 / 0 / 210 | **1 false freeze candidate** at allowed hold start 11s |
| Three single-frame holds in motion | 3 | **3 / 0 / 0 / 207** | **3 / 0 / 0 / 207** | 3 edit freezes **plus** 1 authorized tail hold candidate |
| Three short two-frame local reversals | 6 | **6 / 0 / 0 / 204** | **0 / 6 / 0 / 204** | No reversal detected, but 1 authorized tail hold candidate |
| **Frame-level totals** | **9** | **9 / 0 / 0 / 621** | **3 / 6 / 0 / 621** | Author-approved still camera yields one apparent freeze in every output |

All three output files have 210 monotonic presentation timestamps at 15fps and 14 seconds, despite introduced frame-order faults. The clean re-encode has source-PSNR minimum **{breakdown['clean_reencoded']['min_native_source_comparison_PSNR_dB']:.2f} dB** (above the **predeclared** 28dB threshold); the hold-edited and reversed copies reach **{breakdown['single_frame_holds']['min_native_source_comparison_PSNR_dB']:.2f} dB** and **{breakdown['short_reversals']['min_native_source_comparison_PSNR_dB']:.2f} dB** at offending frames. The original 3-second end-camera pause is a **true source-authorized negative** and should not auto-fail video quality. FFmpeg flags its near-identical images in all three movies. Without author intent, a whole-frame freeze detector cannot distinguish an appropriate cinematic hold from an undesired freeze.

A *real Firefox native HTML5 media viewer* played both clean and reversal-corrupted movies to **14.0 seconds, ended=true, error=null**, verified against exact output video SHA. Successful browser playback/PTS therefore cannot certify the frame sequence; the source-matched check still correctly identifies six reversed frames. These are actual encoded final deliverables under isolated controlled edits, **not merely intermediate JSON/model opinions**.

## Verifiable resource/cost and acceptance-policy limits

- Original approved full-MP4 PNG extraction: **{wall_decode:.3f}s** CPU; identical codec settings for all new H264 copies, total encoding **{wall_encode:.3f}s** (clean {enc['variants']['clean_reencoded']['encode_wall_s']:.3f}s, holds {enc['variants']['single_frame_holds']['encode_wall_s']:.3f}s, reversals {enc['variants']['short_reversals']['encode_wall_s']:.3f}s).
- Source-file SHA verification/load **{wall_scoring_load:.3f}s**; 3 complete RGB decodes computing *both* reference and no-reference proposals on same frames **{wall_scoring_computation:.3f}s**; 3 ffprobe PTS scans **{wall_pts:.3f}s** and 3 independent FFmpeg freeze checks **{wall_ff:.3f}s**. This gives about **{wall_scoring_load+wall_scoring_computation+wall_pts+wall_ff:.3f}s** summed measured scoring components, not a separate live deployed tool latency. All decoding, model-free feature checks and playback done without production GPU interference.
- The real `src/swaag/perception/acceptance_policy.py` was exercised **offline** against independently SHA-checked frozen source+encoded outputs: **PASS** clean ordered-frame integrity, **FAIL** the two mutated ordered-frame properties, **REVIEW** for low-level CV or FFmpeg proposals. This proves the *module's narrow decision behavior*, **not registration or live use** as a SWAAG user-facing tool.

## Original owner question-by-question disposition

1. **GUI clipping/overlap/contrast:** Earlier human-slide evaluation had OCR 58/82 cutoffs found and 26/82 false alarms on negatives; generic CLIP flagged 66/72 previously unseen triple-human-negative slides. Source DOM/axe-core can prove specific computed properties, but standalone image defects still need native GUI/human gold. **PARTIAL.**
2. **OCR, chart/diagram arrow and label:** TopoBench grounded only 198/966 authored connections via OCR; 151 correct link proposals with 158 false links, and 80/147 right/left reading-order direction guesses. Source topology is not actual arrowhead pixels. **PARTIAL.**
3. **Real 3D mesh, industrial parts and physics:** Corrected owner's warehouse support contacts passed 72/72 from earlier 0/72 incomplete support chain, but engineering-certified part completeness and genuinely photorealistic renders remain independently unjudged. **PARTIAL.**
4. **Encoded final movie and visual frame sequence:** Clean actual source movie and Firefox full playback verified; this new actual *script-corrupted encoded owner video* matches expected source-frame faults in 9/9 altered positions with 0/621 other frame flags. **NARROW SOURCE-CONTRACT PASS ON CONTROLLED EDITS ONLY.**
5. **Natural video behavior, acoustic events/audio-visual synchronization:** FFmpeg and simple motion detect freeze but not reversed frames reliably; no independent human onset/offset of unplanned faults, and the tested film has no audio. **PARTIAL / AV UNVERIFIED.**
6. **Reliable reusable SWAAG acceptance stack:** Verified-source per-property PASS/FAIL/REVIEW module responds appropriately offline; `InspectImageTool` remains disabled by default and live end-to-end workflow approval has **not** been demonstrated. **PARTIAL.**

**Hard remaining evidence gap:** independently certified real dynamic production GUI positives/negatives, specifically correct diagram arrowheads and machine identity/components, source-matched but naturally occurring *timed* moving-video and audiovisual faults, genuine photorealism judgments and full user-visible SWAAG tool integration. No human gold or real fault-rate confidence interval was fabricated. Original production media, render services, uploads, prior v52 explicitly denied extraction and old uncertain ChatGPT effects were not touched. Status **PARTIAL**.

## Reproduction and receipts

Original source, input frame copies, 3 newly encoded clips, predeclared fault manifest, all source-frame/encoded SHA receipts, decoded per-frame metrics and browser controls are isolated under `/data/var/perceptual-independent-20261024/owner_real_encoded_fault_injection_20261026/`; no third-party bytes or large binary videos are committed. Code: `tests/research/owner_real_encoded_video_faults_20261026.py` (`freeze`, `extract`, `encode`), `tests/research/owner_real_encoded_video_fault_scoring_20261026.py`, `tests/research/owner_faulted_mp4_browser_playback_20261026.py`, `tests/research/owner_video_source_attested_policy_replay_20261026.py`, and `tests/research/publish_owner_encoded_fault_comparison_20261026.py`. Every stage refuses to overwrite its previous verified outputs; **do not replay effects in the recorded run**. Only exact machine-readable compact scores and source hashes are included in Git.
''')
 print('OWNER_VIDEO_FAULT_COMPACT_REPORT',JSON,sha(JSON))

if __name__=='__main__':main()
