#!/usr/bin/env python3
"""Publish bounded source-identity failure repair; no redistribution of PNGs."""
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
BASE=Path('/data/var/perceptual-independent-20261024/rotbench_source_binding_replay_20261026')
PUBLIC=REPO/'docs/research/perceptual-source-identity-fail-closed-repair-2026-10-26.json'
DOC=PUBLIC.with_suffix('.md')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def main():
 if PUBLIC.exists() or DOC.exists():raise RuntimeError('Prior publication exists; cannot overwrite')
 spec=BASE/'frozen_protocol.json';raw=BASE/'source_binding_replay.json';data=json.loads(raw.read_text());p=json.loads(spec.read_text())
 assert sha(spec)==data['frozen_protocol_SHA256']
 assert data['policy_code_SHA256']==sha(REPO/'src/swaag/perception/acceptance_policy.py')
 assert data['real_author_clean_wrong_activity_pairs']==352
 assert data['real_author_clean_wrong_activity_orientation_trials']==704
 assert data['actual_correct_source_binding_decisions_on_mismatched_orientation_trials']=={'review':704}
 assert data['same_right_original_exact_byte_snapshots']=={'pass':704}
 assert data['source_file_hashes_all_match_original_receipts']
 assert data['source_generated_paired_real_screen_shots_100']['correct_source_exact_immutable_snapshot_fail_on_author_edits']==50
 assert data['source_generated_paired_real_screen_shots_100']['correct_source_exact_immutable_snapshot_pass_on_original_right']==50
 assert len(data['previously_unavailable_original_right_refs_without_retry'])==2
 negative=data['bad_metadata_and_missing_file_fail_closed']
 assert negative['true_self_snapshot']=='pass'
 assert all(value=='review' for key,value in negative.items() if key!='true_self_snapshot')
 wrong=data['comparison_per_case'];synth=data['source_mutation_per_case']
 assert len(wrong)==704 and len(synth)==100
 assert all(x['wrong_identity_outcome']=='review' for x in wrong)
 assert all(x['same_author_source_against_itself']=='pass' for x in wrong)
 assert len({x['case_id'] for x in wrong})==137
 assert len({x['source_test_id'] for x in synth})==100
 exact_source_errors=[x for x in synth if x['source_author_injected_change'] and x['byte_snapshot_group_outcome']=='fail']
 assert len(exact_source_errors)==50
 old_pixel_misses=[x for x in synth if x['source_author_injected_change'] and not x['earlier_visual_detector_flag']]
 assert len(old_pixel_misses)==11
 source_sha=data['source_pinned_release_SHA256']
 report={
 'date':'2026-10-26','status':'PARTIAL',
 'source':'Existing pinned RotBench original author Android screenshots and independently frozen negative/positive controls; all original image bytes remain isolated; no original file changed',
 'source_release_sha256':source_sha,
 'source_version_origin':'previously obtained public author release, originally frozen before this study; screenshots are not newly human-adjudicated',
 'source_manifest_replay_SHA256':sha(spec),
 'raw_source_attested_replay_SHA256':sha(raw),
 'source_policy_module_SHA256':data['policy_code_SHA256'],
 'previous_sources_SHA256':p['inputs_from_preregistered_prior_work'],
 'cause':'The low-level source-verified Observation interface has only caller-asserted booleans. An actual app can be compared to the wrong activity at the same app, native viewport and source revision; prior image-only regression flags all 352 otherwise-correct alternate-activity screen pairs.',
 'repair':'Added strict exact-immutable-snapshot source-bound verdict: require trusted external manifest and actual render-state attestation, matching source revision/subject/activity/orientation/dimensions, and recompute source+actual SHA-256 from existing files. Missing/stale/wrong bindings => REVIEW. Source-equal exact-byte contract => PASS; source-different exact-byte contract => property-only FAIL.',
 'evidence':{
  'previous_wrong_activity_pairs_flagged_naively':352,
  'wrong_reference_unique_author_clean_original_cases':137,
  'wrong_reference_actual_author_apps':28,
  'wrong_reference_orientation_comparisons':704,
  'wrong_reference_flag_after_binding_gate_as_verified_FAIL':0,
  'wrong_reference_REVIEW_after_binding_gate':704,
  'correct_self_reference_source_byte_contract_PASS':704,
  'previously_frozen_100_Android_cases':100,
  'pinned_author_injected_mutation_snapshot_FAIL':50,
  'pinned_author_right_unmodified_snapshot_PASS':50,
  'previous_localized_visual_detector_author_changes_detected':39,
  'previous_localized_visual_detector_author_changes_missed':11,
  'previous_unavailable_source_reference_cases_REVIEW':2,
  'all_408_distinct_source_artifact_SHA_verified':True,
  'distinct_source_artifacts_reverified':data['source_file_hashes_recomputed'],
 },
 'negative_controls':negative,
 'not_claimed':{
  '50_exact_byte_snapshot_change_detections_are_50_human_GUI_clipping_detections':False,
  'source_manifest_client_flags_prove_authenticated_host_state':False,
  'GUI_dynamic_appearance_change_is_automatically_failure':False,
  'real_SWAAG_user_facing_workflow_enabled_or_integration_verified':False,
  'film_order_source_psnr_acceptance_or_3D_engineering_covered_by_byte_snapshot_gate':False,
 },
 'per_case_hashed_samples':[
  {'case_digest':hashlib.sha256(str(x['case_id']).encode()).hexdigest(),
   'orientation':x['orientation'],
   'original_actual_sha256':x['actual_sha'],
   'baseline_byte_different':x['naive_byte_mismatch'],
   'strict_decision':x['wrong_identity_outcome'],
   'same_snapshot_control':x['same_author_source_against_itself']}
  for x in wrong],
 'source_modified_100_case_outcomes':[
  {'case_sha256':hashlib.sha256(str(x['source_test_id']).encode()).hexdigest(),
   'source_author_mutant':x['source_author_injected_change'],
   'prior_pixel_rule_proposed':x['earlier_visual_detector_flag'],
   'exact_byte_contract_verdict':x['byte_snapshot_group_outcome']}
  for x in synth],
 'performance':{'CPU_only_wall_seconds':data['measured_CPU_wall_s'],
  'peak_process_RSS_KB':data['peak_RSS_KB'],
  'GPU_used':data['GPU_used'],
  'Nitro_live_resource_snapshot':{'load_1min':34.62,'GPU_total_MiB':6144,'GPU_used_MiB':4735,'GPU_free_MiB':1070,'snapshot_time':'2026-10-26T22:23:24+01:00','other_work_not_stopped':True},
  'note':'Resource metrics are a one-time snapshot plus replay wall time; contention affects latency, but this does not prove how much time it added.'},
 'prior_denials_and_incidents':[
  'Older explicitly denied high-resolution video tail source retrieval was not retried.',
  'Attempted transfer of local storyboard to the chat was denied; no reroute or retry.',
  'Two author clean references previously returned HTTP404 at pinned release; no new HTTP access or substitute image performed.',
  'Eight previously invalid native-coordinate RotBench author bounding rectangles remain excluded pending documented transform; no guessed remap.',
  'Current replay harness initially mistook 352 pairs for distinct source cases (137 unique), and new-holdout `id` key for `record_index`. These local code errors were corrected before the final frozen output; no original input was changed.'
 ],
 'remaining':[
  'Trusted source identity and activity-state attestation in a live rendering/tool workflow is not implemented by this offline helper; caller flags alone are not evidence.',
  'Exact-byte comparator is not a calibrated semantic visual defect model; changing GUI states must return REVIEW rather than FAIL unless an immutable source-byte contract is explicitly justified.',
  'Independent human native GUI and diagram-arrowhead gold, industrial rendered/physics certification, naturally occurring video/audio temporal gold and real SWAAG end-to-end release acceptance remain incomplete.'
 ],
 }
 PUBLIC.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
 ev=report['evidence'];perf=report['performance']
 DOC.write_text(f'''# Exact source binding repair on real Android captured screens — October 26, 2026

**PARTIAL; concrete source-identity failure corrected for an opt-in exact immutable snapshot contract.** This round reread the complete original owner testing recording and Nitro Infra/testing/GUI guidelines, checked Git state clean at `a47f174`, inspected the original 100+240 RotBench results, 352 different-Activity clean-control receipts, the earlier 72 human-slide test and previous denied operations. No production GPU model/renderer/service was changed. Original data and uncertainty/denial receipts remain untouched.

## The measured real cause

A source-frame lookup can be **wrong even if the app/package, viewport size, orientation and source revision all match**. The already frozen external RotBench author-clean negative set contains 352 correct screenshots compared to a clean reference from a **different activity in the same app**; the image-only detector would accuse **352/352 pairs** of a defect. They represent only **137 distinct right/source cases across 28 Android apps**, not 352 independent applications. The original lower-level `decide_property` consumes caller-asserted verification flags but does not verify source identifiers, metadata binding or bytes itself. This is an integration vulnerability, not proof that the low-level function was actually called wrong in production.

Added `RenderSourceBinding` and `decide_bound_immutable_snapshot(...)` to `src/swaag/perception/acceptance_policy.py`. It requires exact source kind/revision/subject/activity or render-state/orientation/dimensions, independently attested source manifest and rendered state, prior SHA-256 receipts recomputed from **both actual original files**, and an explicit *immutable byte-identical snapshot* contract. Unknown binding, other activity, stale output/reference SHA, absent actual file, different source version, unknown source authority or permitted changes => **REVIEW**, never PASS/FAIL. When all preconditions are verified, it returns PASS for identical bytes and FAIL for changed bytes **only for that exact contract**. Attestation booleans must be established by a trusted runtime; this helper does not cryptographically prove a caller's metadata.

## Real source controls replayed, not a synthetic model test

| Original author data case | Previous unqualified interpretation | Strict new exact-source-bound outcome |
|---|---|---|
| 352 correct screens against **wrong activity** source, two orientations each | Previous pixel model flags 352/352 pairs | **0 auto-fails; {ev['wrong_reference_REVIEW_after_binding_gate']}/704 REVIEW** |
| Same 352 right screens compared to their own correct exact bytes, both orientations | Not a visual acceptance question | **{ev['correct_self_reference_source_byte_contract_PASS']}/704 PASS** for byte-equality ONLY |
| 50 author-injected source screenshot modifications | Previous localized pixel screen anomaly method 39/50, missed 11 | **{ev['pinned_author_injected_mutation_snapshot_FAIL']}/50** changed-byte snapshot contract FAIL; NOT fifty human clipping recognitions |
| 50 unmodified source-right samples | Previous detector no discrepancy | **{ev['pinned_author_right_unmodified_snapshot_PASS']}/50** unchanged-byte PASS |
| Two source tests with missing reference PNGs at pinned original release | Previous correctly abstained | **2/2 REVIEW** with no HTTP request or invented bytes |

Original source SHA-256 `{source_sha}` and the separately frozen replay protocol SHA-256 `{sha(spec)}` bind the tested data; {ev['distinct_source_artifacts_reverified']} distinct actual PNGs rehashed and matched their historical receipts. Stale metadata version, wrong reference or observed byte checksum, missing source proof, false activity attestation, non-immutable contract and unavailable reference each returned **REVIEW**; a correct self-reference alone returned PASS. No original file was modified and the old wrongly bound classifier outputs were preserved for audit.

**Costs:** {perf['CPU_only_wall_seconds']:.3f} seconds low-priority CPU, peak resident memory {perf['peak_process_RSS_KB']:,} KiB, no GPU. Concurrent live Nitro sample load average {perf['Nitro_live_resource_snapshot']['load_1min']}, GPU {perf['Nitro_live_resource_snapshot']['GPU_used_MiB']} MiB/{perf['Nitro_live_resource_snapshot']['GPU_total_MiB']} MiB occupied; trading and Chromium/3D processes remained untouched. This is not a clean single-request latency guarantee.

## Limits and next operational path

The 50 injected changes were authored on real Android screenshots; an immutable-byte check catches edits because they change bytes. It cannot tell whether text is clipped, contrast is inadequate or a redraw is legitimate. Use this gate only after a trusted source and **actual app/UI state** have been attested and the exact-byte property is warranted. General source/provenance checkers on previously approved *lossy* H264 movies require their own PSNR/timebase contract, not this byte-comparison function. `decide_property` is retained as the low-level assertion-only primitive for backward compatibility; live SWAAG `inspect_image` remains an evidence-only, default-disabled proposal tool. No live registered decision tool has been qualified by this offline replay.

**Previously observed errors and denials:** old storyboard transfer and source-resolution v52 retrieval denied, not retried; two original absent author references were not fetched again; eight source annotation rectangles remain outside native bounds and no coordinate remapping was invented. Two local *new* replay-harness schema mistakes (352 pairs versus 137 underlying source cases, and `id` versus `record_index`) were repaired before the frozen final result, without touching originals. Prior human-rated SlideAudit OCR/CLIP false alerts, TopoBench arrow-direction failures and industrial/temporal model misses remain unchanged.

**Remaining priority:** independent truly clean dynamic GUI and calibrated semantic defects; visible diagram arrowheads, engineering-rated machinery appearance, naturally occurring time-coded video/audiovisual errors, and authenticated source identity captured within a live SWAAG tool. Status **PARTIAL**.

Reproduction scripts: `tests/research/rotbench_source_binding_repair_replay_20261026.py freeze|evaluate`; results/referenced media under `/data/var/perceptual-independent-20261024/rotbench_source_binding_replay_20261026/` and earlier original pinned studies. Stages refuse to overwrite receipts. Compact report contains derived SHA and case-level hashed IDs; no third-party images, private assets or secrets were committed. Do not reissue already attempted denied tool operations.
''')
 print('PUBLISHED_SOURCE_BINDING_REPAIR',PUBLIC,'SHA',sha(PUBLIC))

if __name__=='__main__':main()
