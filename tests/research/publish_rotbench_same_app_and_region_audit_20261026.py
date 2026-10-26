#!/usr/bin/env python3
"""Publish bounded reproducible summaries of two completed independent source audits.

All original screenshots, raw bbox annotations and full case receipts stay
outside Git. Public report keeps compact audit numbers and source SHA only.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

WORK=Path(__file__).resolve().parents[2]
DOC=WORK/'docs/research/perceptual-android-author-state-and-region-audit-2026-10-26.md'
PUBLIC=WORK/'docs/research/perceptual-android-author-state-and-region-audit-2026-10-26.json'
BASE=Path('/data/var/perceptual-independent-20261024')
CLEAN=BASE/'rotbench-same-app-clean-state-negative-20261026'
REG=BASE/'rotbench-author-region-frozen-audit-20261026'
OLD=BASE/'rotbench-android-heldout-20261026'
UNUSED=BASE/'rotbench-unused-source-split-20261026'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 if DOC.exists() or PUBLIC.exists():raise RuntimeError('Do not overwrite published research')
 clean=json.loads((CLEAN/'scored.json').read_text())
 gold=json.loads((REG/'evaluated.json').read_text())
 gold_rows=gold['rows'];match=clean['metrics'];cases=clean['cases']
 assert len(cases)==match['author_clean_distinct_activity_pairs']==352
 assert match['paired_orientation_comparisons']==2*len(cases)
 assert all(x['wrong_version_correct_gate']=='review' for x in cases)
 assert len(gold_rows)==gold['matching_modified_png_paths']==72
 region_valid=[r for r in gold_rows if r.get('bbox_valid_at_native_resolution')]
 region_in=[r for r in region_valid if r.get('fraction_of_detected_changes_inside_bbox') is not None]
 assert len(region_valid)==62 and len(region_in)==62
 assert gold['bbox_invalid']==8 and gold['valid_original_pixels']==70 and gold['abstain_missing_png_reference']==2
 assert gold['old_previous_misses_with_author_bbox']+gold['new_previous_misses_with_author_bbox']==7
 inside90=sum(r['fraction_of_detected_changes_inside_bbox']>=.9 for r in region_in)
 misses=[r for r in gold_rows if r.get('detector_flag') is False]
 assert len(misses)==7 and all(0<r['changed_fraction_screen']<=.0005 for r in misses)
 prior=[OLD/'predictions.json', UNUSED/'predictions.json']
 assert [sha(p) for p in prior]==gold['original_previous_predictions_shas']
 report={
 'date':'2026-10-26','overall':'PARTIAL','source':'https://huggingface.co/datasets/ImDim/RotBench','source_version':'25b6e8214c5be600f54dc0834eaf4597fa9192e6','source_test_json_sha256':'0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7',
 'evidence_SHA256':{str(p.relative_to(BASE)):sha(p) for p in (CLEAN/'design.json',CLEAN/'frozen_pairs.json',CLEAN/'image_only_results.json',CLEAN/'scored.json',REG/'inventory.json',REG/'classification_source.json',REG/'evaluated.json',*prior)},
 'clean_alternative_state_negative':{k:v for k,v in match.items() if k!='per_app'},
 'per_app':match['per_app'],
 'correct_source_binding_verdict':'REVIEW for every alternate same-app different-activity reference; PASS only for verified immutable exact self reference in this sample, not a blanket GUI acceptance',
 'gold_region_cross_split':{'exact_modified_file_matches':len(gold_rows),'original_refs_available':gold['valid_original_pixels'],'abstained_missing_original_refs':gold['abstain_missing_png_reference'],'author_bbox_within_original_png':len(region_valid),'author_bbox_out_of_original_png':gold['bbox_invalid'],'old_detector_misses_with_bbox':gold['old_previous_misses_with_author_bbox'],'new_detector_misses_with_bbox':gold['new_previous_misses_with_author_bbox'],'bbox_valid_changed_pixels_inside_90pct_or_more':inside90,'bbox_valid_changed_pixels_denominator':len(region_in),'all_seven_missed_synthetic_changes_smaller_than_screen_fraction':.0005,'by_type':gold['by_type'],'invalid_native_bbox_indices':[r['detect_id'] for r in gold_rows if r.get('bbox_valid_at_native_resolution') is False]},
 'cost':{'clean_alternative_state_image_CPU_seconds':clean['cost']['CPU_wall_seconds'],'region_author_bbox_native_CPU_seconds':gold['cost']['CPU_seconds_to_analyze_native_delta_inside_original_annotations'],'aborted_first_clean_state_CPU_wall_seconds':512.764,'aborted_first_attempt_no_frozen_predictions':True,'GPU_calls':0},
 'review_limitations':['The original RIGHT GUI screens are author-filtered negatives, not independently human adjudicated; a clean screen from different Activity must not be used as a valid immutable source match.','352 comparisons are correlated repetitions of 137 original clean screen pairs from 28 apps, with at most 3 different other Activities per original case. These are not 352 independent application samples.','Author bbox coordinates in 8 available annotated cases extend beyond actual rendered native PNG, so those bboxes cannot be used as exact pixel localization truth without an author-documented map.','The published RotBench natural 100 developer-confirmed bug set was not available as raw matched labelled media in this project; author-reported RotVL model scores are not local inference or our measured performance.','No independent human judgments on native Android interaction, no genuinely changing same Activity captures, no owner warehouse final photorealism and no time-coded natural video fault/clean gold.','SWAAG InspectImageTool default disabled and existing conservative policy not proven in a live tool call; this study measures off-line CPU pixels only.'],
 'same_app_clean_cases':[{'case_id':r['id'],'alternate_id':r['alt_id'],'localized_flag':r['localized_bad_flag'],'global_flag':r['global_bad_flag'],'any_pixel_flag':r['any_pixel_bad_flag'],'exact_original_flag':r['exact_control_bad_flag'],'correct_source_binding':r['wrong_version_correct_gate']} for r in cases],
 'annotated_missed_cases':[{'case_id':r['detect_id'],'defect_type':r['gold_author_type'],'changed_fraction':r['changed_fraction_screen'],'bbox_valid':r['bbox_valid_at_native_resolution'],'changed_pixel_fraction_inside_gold_bbox':r['fraction_of_detected_changes_inside_bbox']} for r in misses],
 }
 PUBLIC.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
 flag=match['author_clean_bad_flags_localized_wrong_activity_ref'];glob=match['author_clean_bad_flags_global_wrong_activity_ref'];anyc=match['author_clean_bad_flags_any_pixel_wrong_activity_ref'];n=len(cases)
 DOC.write_text(f'''# Real valid-screen instances and original defect-region audit — October 26, 2026

**Result: PARTIAL; a stricter reference-identity acceptance rule is necessary, and independent human visual gold remains absent.** This work extends, but does not replay, the previous 100 + 240 author-labelled real Android screenshot runs. We first read Infra `AGENTS.md`, the entire September 23 original testing recording, GUI guidelines and previous SWAAG research reports. The SWAAG worktree was clean at commit `bcdb4bf` before this new experiment. Other agents' Infra edits were not touched.

## Hard genuine author-clean controls, SAME app and dimensions but different activity

The pinned source [RotBench author-released original images](https://huggingface.co/datasets/ImDim/RotBench) contain 170 author-approved *right* portrait/landscape screen pairs. Every one of their 340 PNGs was located locally and **reverified by SHA-256** against the preceding two frozen original-source extraction manifests. No discrepancy was found and no new image was downloaded. Of the 170 right cases, **{match['distinct_author_clean_cases']}** had a different, author-approved activity screen from the *same package, with the same native dimensions and orientation*. The protocol froze up to three deterministically hash-ranked alternatives, producing **{n}** clean-vs-clean image-pair comparisons across **{match['distinct_apps']} Android apps** and {match['paired_orientation_comparisons']} orientation-level comparisons. The frozen manifest SHA-256 is `{sha(CLEAN/'frozen_pairs.json')}`. These are correlated and not 352 independent clean applications.

| Pre-existing frozen pixel rule, same actual native images and comparison budget | Author-clean screenshot pairs falsely labelled changed/defective if wrong activity is treated as approved reference | Correct interpretation |
|---|---:|---|
| Local changed-pixel area >0.05% at max channel delta >=24 | **{flag}/{n}** | Unverified semantic GUI defect, REVIEW because activity ID mismatched |
| Global pixel mean absolute change >1.5 | **{glob}/{n}** | Likewise REVIEW; same exact native pixels and reference |
| Any pixel changed | **{anyc}/{n}** | Detects different screens, not a quality failure |
| Exact original screenshot as its *own* reference | **{match['exact_self_ref_bad_flags_localized']}/{n}** | PASS for that single immutable screenshot-equality property only |
| Version/activity-aware source binding | **{match['correct_source_binding_review_for_wrong_activity']}/{n} REVIEW** | No auto-FAIL and no universal GUI PASS |

The actual original screenshots in every comparison were author-accepted correct GUI states. The alternate activity may have different content while remaining completely valid; a pixel difference is not a GUI clipping, contrast or overlap defect. Same-app and same-orientation do not establish equivalence to the same activity/state. **This is deliberately an invalid-reference stress test**, not a fair production false-positive estimate for a correctly bound immutable regression screenshot. Unlike the prior other-app and wrong-orientation shams, it isolates missing *activity/state identity* while holding app, viewport and version constant. A separate null always-accept classifier would have zero alerts here but no ability to catch an actual defect.

## Independent released source annotation cross-check of previous false negatives

A separately published `bug_classification_and_localization/test.json` file at the same exact RotBench revision has SHA-256 `{gold['classification_source_sha']}`. Matching the exact *mutated PNG path* (not numeric case IDs) found **{len(gold_rows)}** original prior detection mutations with a source-author region annotation; **{gold['valid_original_pixels']}** had their original right reference PNG and **{gold['abstain_missing_png_reference']}** could not be scored because original clean source images were absent. The previous prediction SHA receipts were validated before this posthoc comparison; no prior threshold was changed or prior denied operation retried.

Of the {gold['valid_original_pixels']} evaluable mutations, **63 were detected and 7 missed** with the frozen changed-area rule. All seven true changed-pixel masks were below the 0.05%-of-screen threshold, despite a concrete author-injected mutation. In **{gold['bbox_invalid']} of the {gold['valid_original_pixels']}** available cases the author-provided rectangle lies outside the source PNG's actual native dimensions (including right/bottom coordinate 728 for 720-pixel image sides). Do **not** use those eight rectangles as exact pixel-level source gold without a separately justified coordinate transform or correction. Of the other **{len(region_in)}** geometrically valid author rectangles, **{inside90}** contain at least 90% of pixels changed by 24 or more in their corresponding source-mutated image. The seven misses include types: {', '.join(f'{k} {v}' for k,v in sorted(Counter(r['gold_author_type'] for r in misses).items()))}. This validates an *author synthetic region consistency check*, not an independently human-labelled clipped-text or actual runtime GUI error.

The original changed pixels and mismatched reference source paths are retained outside Git under `/data/var/perceptual-independent-20261024/`. No third-party images were committed, and original files, earlier research receipts, gallery, models, real production services and all historical uncertain submissions remained untouched.

## Compute and boundaries

The completed source-correct clean-state run took **{clean['cost']['CPU_wall_seconds']:.3f}s** low-priority Nitro CPU, while the source-region pixel and bbox audit took **{gold['cost']['CPU_seconds_to_analyze_native_delta_inside_original_annotations']:.3f}s**; GPU calls were zero. An earlier redundant-SHA version of this *new study*, not the previous research evidence, was terminated after **512.764s** because it produced no finalized predictions; it was optimized, and the frozen case list was preserved byte-for-byte. That aborted CPU work is reported rather than omitted from compute accounting. The dataset author also separately reports 70% RotVL-8B accuracy on 100 developer-confirmed natural rotation bugs, but those raw matching samples and labels were **not available for independent local re-evaluation**. Do not substitute their published model scores for a new local benchmark.

**Operational decision:** verify source app, exact activity/state, viewport dimensions/orientation, original source revision, and content SHA before applying immutable screenshot checks. Auto-fail only the verified exact property when reference and final output are both verified. If identity is wrong or unverified, return REVIEW; no positive semantic defect verdict. GUI OCR and vision outputs are proposals. Diagram arrow direction, industrial parts/contact and video temporal defects still require separate independently validated property-specific analyzers. Real owner final 3D photorealism, independently certified GUI true positives/legitimate same-activity dynamic negatives, real timed video defect gold and live SWAAG integration remain **unproved**. Status: PARTIAL.

## Reproduction

`python3 tests/research/rotbench_same_app_state_negatives_20261026.py freeze|pixels|score` and `python3 tests/research/rotbench_author_region_miss_audit_20261026.py inventory|evaluate`, followed by `python3 tests/research/publish_rotbench_same_app_and_region_audit_20261026.py`. Frozen stages refuse overwrites. The paired source image list, author classification source, raw per-image SHA and full ROI results remain isolated; this repo contains only code, a compact per-case summary, public facts and unit tests. Do not repeat the earlier aborted command, uncertain calendar sends or denied old retrievals.
''')
 print('PUBLISHED',DOC,'JSON',PUBLIC,'SHA',sha(PUBLIC))

if __name__=='__main__':main()
