#!/usr/bin/env python3
"""Publish cross-source PNG integrity and narrow final screenshot verdicts."""
import hashlib
import json
from pathlib import Path
import statistics

RAW=Path('/data/var/perceptual-independent-20261024/snapshot_media_source_replication-20261027')
WORK=Path(__file__).resolve().parents[2]
PUBLIC=WORK/'docs/research/perceptual-real-png-pixel-and-capture-provenance-2026-10-27.json'
DOC=PUBLIC.with_suffix('.md')
PRIOR=WORK/'docs/research/perceptual-real-swaag-attachment-snapshot-roundtrip-2026-10-27.json'
SOURCE=WORK/'src/swaag/tools/verified_snapshots.py'
REPO=WORK/'pyproject.toml'
LOCK=WORK/'uv.lock'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for frag in iter(lambda:f.read(1048576),b''):h.update(frag)
 return h.hexdigest()


def exact_cp_95_success_8():
 # Exact two-sided Clopper-Pearson lower bound for 8 of 8 successes.
 return round(.025**(1/8),5)


def exact_cp_95_fp_0of16():
 # Exact two-sided Clopper-Pearson upper bound for zero of 16 false flags.
 return round(1-.025**(1/16),5)


def main():
 if PUBLIC.exists() or DOC.exists():raise RuntimeError('Preserve published audit; do not overwrite')
 frozen=RAW/'frozen.json'; scored=RAW/'runtime_results.json'
 plan=json.loads(frozen.read_text());obs=json.loads(scored.read_text())
 assert obs['source_manifest_sha']==sha(frozen)
 assert plan['frozen_before_evaluation']
 assert len(plan['cases'])==len(obs['results'])==32
 assert plan['n_by_class']=={'clean':8,'changed':8,'wrong_activity':8,'truncated_final_png':8}
 counts=obs['totals'];assert set(counts)==set(plan['n_by_class'])
 for x in plan['n_by_class']:
  assert counts[x]=={'n':8,'correct':8}
 for i,row in enumerate(obs['results']):
  assert row['kind']==plan['cases'][i]['kind']
  assert row['expected']==plan['cases'][i]['expected']
  assert row['tool_verdict']==row['expected']
  assert row['source_sha_match'] and row['capture_sha_match']
  assert row['event_hash'] and len(row['event_hash'])==64
 prior=json.loads(PRIOR.read_text())
 assert prior['new_registered_tool']=='verify_immutable_snapshot'
 assert prior['real_source_to_target_end_state_results']['wrong_activity']['first_real_runtime_tool_verdict']=='review'
 rows_by={kind:[x for x in obs['results'] if x['kind']==kind] for kind in plan['n_by_class']}
 reasons={kind:sorted({x['tool_reason'] for x in rows}) for kind,rows in rows_by.items()}
 assert reasons=={'clean':['verified_immutable_png_bytes_equal'],
                  'changed':['verified_immutable_png_bytes_changed'],
                  'wrong_activity':['attested_source_state_or_contract_unavailable'],
                  'truncated_final_png':['png_final_pixels_or_end_boundary_not_verified']}
 stats={}
 for kind,items in rows_by.items():
  latency=[x['tool_wall_s'] for x in items]
  stats[kind]={'n':len(items),'expected_verdict':items[0]['expected'],
    'correct':sum(x['tool_verdict']==x['expected'] for x in items),
    'median_seconds':round(statistics.median(latency),4),
    'min_seconds':round(min(latency),4),'max_seconds':round(max(latency),4)}
 synopsis={
   'date':'2026-10-27','overall':'PARTIAL','source':'Real original RotBench Android PNG screenshots from an author-pinned 340-case release. The new evaluated source cases were taken from the previously unused 240-case portion plus independent same-app different-activity source controls. Invalid PNGs are SCRIPT-TRUNCATED duplicates of these real original PNGs.',
   'central_finding':'The old registered SWAAG tool required only 24 PNG header bytes and attachment self-checksums for exact-byte PASS/FAIL. Strict current tool requires host-attested independent SOURCE and TARGET capture SHA digests, original content SHA receipts, full pixel decode and verified terminal IEND boundary. Unknown/corrupt media -> REVIEW.',
   'original_release_SHA256':plan['source_release_sha'],
   'separate_frozen_study_SHA256':sha(frozen),
   'complete_actual_registered_runtime_result_SHA256':sha(scored),
   'prior_round9_real_runtime_report_SHA256':sha(PRIOR),
   'source_manifest_independent_SHA256':plan['source_selection_sha'],
   'source_prior_full_image_predictions_SHA256':plan['prior_prediction_sha'],
   'source_distinct_app_negative_manifest_SHA256':plan['wrong_activity_source_sha'],
   'registered_tool_module_sha256':sha(SOURCE),
   'dependency_pyproject_sha256':sha(REPO),
   'dependency_lock_sha256':sha(LOCK),
   'actual_attachment_tool_call_count':len(obs['results']),
   'confusion_by_exact_defined_property':{
       'source_correct_immutable_changed':{'TP':8,'FN':0},
       'source_correct_exact_immutable_unchanged':{'FP':0,'TN':8},
       'source_wrong_activity':{'REVIEW':8,'auto_PASS':0,'auto_FAIL':0},
       'undecodable_actual_PNG':{'REVIEW':8,'auto_PASS':0,'auto_FAIL':0},
   },
   'case_counts':stats,
   'source_capture_integrity_check':'8/8 changed source snapshots have author-modified decoded pixels, 8/8 clean snapshots match approved decoded pixels; each attachment SHA matches an independently frozen release or script-controlled capture SHA; each final PNG is actually decoded, not just its filename or dimensions.',
   'corrupted_negative':'8/8 of the original unmodified publisher PNGs had ONLY the 12-byte IEND end marker removed in a NEW derived capture, preserving the authentic source IHDR and most compressed pixel bytes. They were intentionally script-controlled invalid final media and returned REVIEW, not an incorrect definitive snapshot verdict.',
   'source_negative':'8/8 preserved, legitimate different-Activity screens from 8 distinct apps were REVIEW, not an automatic visual FAIL.',
   'host_attestation_boundary':'Source SHA expected is independently host-pinned publisher manifest data, capture SHA expected is the trusted screenshot capture receipt, both bound to actual content-addressed attachment bytes. A malicious/incorrectly implemented host could still fabricate both and the manifest identity; this is NOT a cryptographically authenticated live Android capture service.',
   'PNG_pixel_decoder_boundary':'Opt-in Pillow>=11,<13; any missing Pillow or invalid pixel data, malformed CRC or absent final PNG IEND returns REVIEW; no blanket image/video quality approval. Pixel decoding/CRC validation is not an appraisal of readability, lighting or semantics.',
   'frozen_gold_authority':'Author-produced synthetic defects and author-approved Android original images; no new independent human visual annotations. The eight end-boundary negatives are script-authored PNG truncations, not natural error prevalence.',
   'cost':{'all_32_cases_actual_SWAAG_attachment_store_and_durable_history_wall_s':obs['wall_s_including_attachment_and_history'],
           'median_call_wall_s_by_class':{k:v['median_seconds'] for k,v in stats.items()},
           'min_call_wall_s':min(x['tool_wall_s'] for x in obs['results']),
           'max_call_wall_s':max(x['tool_wall_s'] for x in obs['results']),
           'GPU_used':False,
           'old_3case_runtime_wall_s_not_matched_for_before_after':True,
           'no_claim_compressed_decoder_pure_CPU_time_from_runtime_wall':True},
   'descriptive_uncertainty_not_general_population':{
       'n8_success_95pct_exact_binomial_lower':exact_cp_95_success_8(),
       'n16_zero_unqualified_auto_decisions_95pct_exact_binomial_upper':exact_cp_95_fp_0of16(),
       'warning':'These intervals are descriptive conditional on an assumed iid model that is NOT valid for correlated modified screenshots; cannot be extrapolated to native GUI deployment.'},
   'outcomes_compact':[
       {'hash_id':hashlib.sha256(str(row['source_case_id']).encode()).hexdigest(),
        'category':row['kind'],'expected':row['expected'],'verified_outcome':row['tool_verdict'],
        'reason':row['tool_reason'],'actual_SWAAG_tool_call_wall_s':row['tool_wall_s'],
        'durable_event_hash':row['event_hash']}
       for row in obs['results']],
   'known_limitations':[
       'The extra host SHA claims must be verified by a trusted live host from a source manifest and actual output capture; the tool only checks consistency between claims and attachments, not signed authority.',
       'This exact-byte contract applies only where snapshot equality is genuinely intended; normal dynamic GUIs, alternate rendering/encoding and valid edits must REVIEW.',
       'Actual screenshot human clipping, occlusion, contrast, diagram arrow direction, 3D industrial semantic parts and video audiovisual synchronization are NOT classified by these byte tests.',
       'Current tests are opt-in isolated SWAAG research runtime; no production workers, source services or GPU models were restarted, and original asset bytes are unchanged.',
       'The previously rejected storyboard transfer and absent RotBench source URLs were not retried.',
       'Original 32 controls do not provide a valid population error rate; separate independent human source labels for real native GUI dynamic changes remain missing.',
   ]
 }
 PUBLIC.write_text(json.dumps(synopsis,sort_keys=True,indent=2)+'\n')
 md=f'''# Independently replicated PNG source/pixel/end-boundary acceptance — October 27, 2026

**Status PARTIAL:** the exact immutable snapshot contract now validates *both independently host-attested artifact hashes and the actual final PNG pixels and end boundary*. Before this patch, the registered tool only inspected the first 24 bytes (PNG signature and IHDR width/height); it could treat a non-renderable/truncated final image as a valid snapshot if an attachment hash matched itself. The host attestor also provided no independent expected approved-image SHA or captured-target SHA, so a mismatched publisher source could be mistaken for authoritative input if its attachment content was otherwise internally consistent.

## Source-frozen real-image replication

The independent protocol SHA is `{sha(frozen)}`; publisher test release SHA `{plan['source_release_sha']}`. It selects 8 different author-approved right screens, 8 author-injected original pixel changes (no later selection by detector result), 8 real same-app/different-Activity clean screens across 8 Android applications, plus 8 **script-truncated** final PNG copies of the correct real source (only the terminal 12-byte `IEND` removed, IHDR retained). Source/capture artifact hashes were frozen **before** their SWAAG tool calls. No other media or original study receipts were changed.

| Actual SWAAG tool category | Source/script expected verdict | Result | Incorrect automatic decisions |
|---|---|---:|---:|
| 8 original clean PNG reference->unchanged output | PASS exact immutable bytes | **8/8 PASS** | 0 |
| 8 real author-injected different output screenshots | FAIL exact immutable bytes | **8/8 FAIL** | 0 |
| 8 correct alternate Activity screens, wrong source identity | REVIEW | **8/8 REVIEW** | 0 |
| 8 original final PNGs truncated at 12-byte end marker | REVIEW, not a decodable screenshot | **8/8 REVIEW** | 0 |

This is **32 correct outcomes in 32 instances under these defined source-property/script labels**, not 32 human-rated semantic GUI judgments. The source-only positive/no-change outcomes remain separate from the error/abstention controls. Eight source negatives and eight truncated negatives are correlated with the existing RotBench source and cannot justify independent population error rates. For an artificial iid assumption, 8/8 has a 95% exact two-sided lower bound of **{exact_cp_95_success_8():.1%}** and 0/16 unjustified decisions an upper bound of **{exact_cp_95_fp_0of16():.1%}**; the iid assumption is not defensible for real-world reliability here.

## Verified software repair and actual cost

The registered pure SWAAG `verify_immutable_snapshot` tool now requires `HostAttestedSnapshot.approved_source_sha256` and `.actual_capture_sha256`, which are checked against both persisted SHA-addressed attachment bytes. The trusted host must independently pin these values to the authoritative source manifest and captured final output, not simply echo the model's input. The tool also requires an optional **Pillow local image decoder** (`swaag[perceptual-snapshots]`), checks the final PNG `IEND` boundary, verifies PNG chunk integrity, and fully loads both images' pixels. If dependencies or data are unavailable/corrupt, it returns **REVIEW**, not PASS/FAIL. Source identity, app, orientation and viewport remain mandatory. Existing visual-model proposals never gain authority from this change.

The **complete 32-case run** through SWAAG's attachment store, registered tool, and durable history took **{obs['wall_s_including_attachment_and_history']:.3f}s** on busy Nitro CPU, with no model/GPU call. Median full dispatch time: clean {stats['clean']['median_seconds']:.3f}s, changed {stats['changed']['median_seconds']:.3f}s, wrong Activity {stats['wrong_activity']['median_seconds']:.3f}s, invalid final PNG {stats['truncated_final_png']['median_seconds']:.3f}s. The earlier header-only 3-case study took about 0.15-0.19s per tool call, but differed in data, source and system load, so this does **not** establish an experimentally matched per-image decoding overhead. The optional library is specified in `pyproject.toml`/`uv.lock` and installed only in the isolated research venv; production defaults remain off.

## Original owner questions — defensible vs unproved

- **GUI clipping, overlap, contrast / OCR:** independent human SlideAudit test: OCR found 58/82 clipped controls with 26/82 false alarms; generic CLIP proposed problems on 66/72 unanimously negative slides. Source-verified exact image, app and pixel integrity is narrow PASS, not automatic semantic GUI correctness. **Original question PARTIAL.**
- **Diagrams, labels, arrows:** prior TopoBench evaluated 198 scorable links, found 151 but 158 false links and correct direction only 80/147. No independent final arrowhead-to-pixel validated detector. **PARTIAL.**
- **3D parts, mesh, support and real physical output:** corrected owner warehouse validated 72/72 intended rack contacts; the final owner movie's 210 original Cycles frames matched the encoded final video. These are narrow contracts, not component-level independent human/CAD ground truth or structural engineering. **PARTIAL.**
- **Video timing and intentional pauses:** owner warehouse and distinct dock source videos passed frozen 210-frame source integrity; controlled script faults were detected 9/9 per film, but no-reference algorithms missed 6 reverse swaps. Real unforeseen human-labeled defects and true time-coded errors are unproved. **PARTIAL.**
- **Audio synchronization:** owner's verified silent warehouse/dock films have no audio tracks; there is no real event-timestamp A/V gold. **UNVERIFIED.**
- **Actual SWAAG integration:** opt-in tool now works through the real temporary SWAAG dispatcher/attachment/event history and persisted session recovery, including verified PNG bytes. This is not a production-trusted live Android state capture attestor, not active user workflows and not a certified GUI quality gate. **NARROW INTERNAL PASS; complete objective PARTIAL.**

## What remains and provenance

The highest-value unresolved evidence is authenticated capture of **actual** dynamic GUI states (not retrospective filename metadata) and independent human-rated native screen outcomes; actual visible diagram arrows, finished industrial part/engineering truth, natural video faults and recorded audio/video event sync. No old explicitly denied media transfers were repeated. No original source, video, Blender asset, private runtime or active GPU process was altered. The source releases and all derived 32-case executions remain in isolated `/data/var/perceptual-independent-20261024/snapshot_media_source_replication-20261027/`, and only source code plus compact hashed receipts enter Git. No previous frozen research session/effect was repeated.

Reproduction: `tests/research/swaag_snapshot_media_boundary_transfer_20261027.py freeze|run` in a **new** isolated root; `tests/research/publish_png_final_source_boundary_replication_20261027.py` to derive the report. Already completed stages refuse overwrite. The test environment uses the previously verified isolated SQLite 3.54.1 and optional Pillow; never install it globally to claim release readiness. Overall **PARTIAL**.
'''
 DOC.write_text(md)
 print('PUBLISHED_NEW_FINAL_SCREENSHOT_MEDIA_PROVENANCE',PUBLIC,'sha',sha(PUBLIC), 'derived',stats)

if __name__=='__main__':main()
