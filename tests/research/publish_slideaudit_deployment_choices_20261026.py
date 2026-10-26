#!/usr/bin/env python3
"""Publish honest same-case cost/false-alert/recall deployment comparison.

Original images, OCR text, complete metadata, model checkpoint remain outside
SWAAG Git. Human gold joins only after same-input model outputs frozen.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
LOCAL=Path('/data/var/perceptual-independent-20261024/slideaudit_unseen_triple_neg_20261026')
PUBLIC=ROOT/'docs/research/perceptual-human-unanimous-negative-deployment-2026-10-26.json'
REPORT=ROOT/'docs/research/perceptual-human-unanimous-negative-deployment-2026-10-26.md'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 if PUBLIC.exists() or REPORT.exists():raise RuntimeError('Research already published; refusing overwrite')
 raw=LOCAL/'deployment_policy_tradeoffs.json';d=json.loads(raw.read_text());source=json.loads((LOCAL/'scored.json').read_text())
 assert source['predeclared_selection_SHA256']==sha(LOCAL/'selection.json')
 assert raw.exists() and d['source_revision']=='642d490b7c1d2e78a50a631bfd359433397f3ecf'
 for k,p in d['frozen_source_SHA'].items():
  paths={'new_gold':LOCAL/'selection.json','new_cv':LOCAL/'pixel_features.json','new_clip':LOCAL/'clip_features.json','new_source_geometry':LOCAL/'source_features.json','old_gold':LOCAL.parent/'slideaudit_selection_20261025.json','old_cv':LOCAL.parent/'slideaudit_343_image_only_ocr_cv_frozen_20261025.json','old_clip':LOCAL.parent/'slideaudit_343_clip_image_only_20261025.json','old_geometry':LOCAL.parent/'slideaudit_343_source_geometry_frozen_20261025.json'}
  assert sha(paths[k])==p,(k,'SHA mismatch')
 neg=d['heldout_72_unanimous_negative_screen_any_alerts'];by=json.loads(json.dumps(d['heldout_72_unanimous_negative_per_category']));old=d['old_343_human_positive_and_matched_negative_by_task'];cost=d['computation']
 # Correct a derived SUMMARY-only aggregation bug without changing frozen
 # source/model predictions or overwriting the original scored JSON receipt.
 # Geometry has no contrast detector; None is ineligible, NOT a measured zero.
 assert by['Insufficient Color Contrast for Readability']['geometry_only']==0
 assert old['Insufficient Color Contrast for Readability']['geometry_only']['abstained']==40
 by['Insufficient Color Contrast for Readability']['geometry_only']=None
 assert (neg['OCR_only'],neg['CLIP_only'],neg['geometry_only'],neg['OCR_or_CLIP'],neg['all_three_union'])==(26,66,26,70,70)
 assert old['Content Overflow/Cut-off']['OCR_only']['TP']==58
 assert old['Occluded Content']['geometry_only']['TP']==93
 assert old['Insufficient Color Contrast for Readability']['geometry_only']['abstained']==40
 assert source['ocr_timeout_count']==0
 assert len(d['cases'])==72
 out={
 'date':'2026-10-26','status':'PARTIAL','full_raw_matrix_source_sha256':sha(raw),
 'origin':'SlideAudit UIST 2025, 600 original and 1800 deliberately altered slides, Prolific trained human annotators, 3 people each; strong agreement = unanimous 3/3. CC BY 4.0 original repository pinned 642d490b7c1d2e78a50a631bfd359433397f3ecf',
 'original_dataset_URL':'https://github.com/zhuohaouw/SlideAudit',
 'primary_paper_URL':'https://pmc.ncbi.nlm.nih.gov/articles/PMC13227769/',
 'gold':'72 previously unseen original image files (not in earlier 343), 6 per each of 12 source/alteration strata; all three target defect categories independently unanimously absent. Other defects and actual interactive GUI correctness not certified.',
 'revision':d['source_revision'], 'original_frozen_SHA256':d['frozen_source_SHA'],
 'all_72_unanimously_negative':True,'unseen_72_source_stratified':True,
 'false_alarms_on_unanimous_negatives_per_category':by,
 'scoring_erratum':{'original_derived_deployment_policy_tradeoffs_json_SHA256':sha(raw),'source_geometry_contrast_reported_zero_is_invalid':True,'corrected_source_geometry_contrast':'NOT APPLICABLE (null), no source contrast detector; all source-only contrast positive/negative observations abstain','immutable_original_model_predictions_unchanged':True},
 'false_review_proposals_per_72_human_unanimous_negatives':neg,
 'false_review_proposal_rate_bootstrapped95_stratified':d['heldout_72_strata_bootstrap_95pct_false_proposal_rate'],
 'earlier_343_source_category_strong_human_positive_and_matched_negative_confusions':old,
 'latency_and_resource':{
  'actually_measured_separate_method_72_wall_s':{k:cost['measured_batch_wall_72_by_rule_s'][k] for k in ('OCR_only','CLIP_only','geometry_only')},
  'actual_72_average_CPU_time_per_image_s':{k:cost['measured_average_cpu_wall_per_image_by_rule_s'][k] for k in ('OCR_only','CLIP_only','geometry_only')},
  'CLIP_model_loading_s':cost['CLIP_cold_load_wall_s'],
  'CLIP_CPU_peak_RSS_KB':source['cost']['CLIP_peak_RSS_KB'],
  'OCR_CPU_peak_RSS_KB':source['cost']['pixel_OCR_CV_peak_RSS_KB'],
  'source_only_geometry_RSS_not_separately_isolated':True,
  'combined_OCR_plus_CLIP_arithmetic_sequential_sum_72_s':cost['measured_batch_wall_72_by_rule_s']['OCR_or_CLIP'],
  'combined_model_end_to_end_wall_time_unmeasured':True,
  'total_new_72_original_source_download_s':source['cost']['download_wall_s'],
  'total_new_72_source_download_bytes_MB':source['cost']['download_MB'],
  'GPU_used':False,
  'human_review_time_per_case':'NOT MEASURED',
 },
 'source_authority_vs_result_reconciliation':{
  'authoritative_for_broad_human_visible_GUI_visual_defects':'NONE of the new tools or data makes a universal authoritative final GUI judge',
  'human_unanimous_negative':'3 trained annotators agreed no specifically named flaw on each new rendered slide, NOT native app or mechanical reality',
  'original_layout_source_geometry':'Source rectangles can prove authored-coordinate overlap/offslide under documented source contract, but this dataset source may precede deliberate raster modifications. Such rectangles cannot auto-FAIL human-visible occlusion or final screenshot clipping without source binding and intent.',
  'CLIP_confidence':'relative zero-shot prompt preference, never an independently calibrated defect probability',
  'OCR_approx_contrast':'grayscale within word box proxy, NOT verified WCAG foreground/background contrast and cannot auto-reject',
  'owner_encoded_MP4':'prior original final warehouse 210-frame full decode and Firefox playback passed narrowly; no audio, photorealism or mechanical truth established',
  'independent_human_vs_source_claim':'No contradiction: geometry of original source or matched prior author release can be correct while human-visible pixels have a different defect; these measure different properties',
 },
 'deployment_decision':{
  'pixel_only_automatic_fail':'NOT AUTHORIZED by evidence; would falsely reject visually human-unanimous category negatives',
  'source_only_auto_fail':'Only when authoritative source AND final rendered state identity verified and narrowly specified actual source property failed; not blanket user-visible outcome',
  'review_proposals':'Use named category OCR/CV/source geometry with exact region/context and uncertainty, prioritize human review; do not skip unflagged on final release unless previously qualified separate property passes',
  'cost_error_tradeoff':'OCR flags 26/72 human unanimous all-negative slides at 171 s CPU; CLIP flags 66/72 at 61 s; union flags 70/72 at ARITHMETIC summed approx 232.3 s, while on older positives union improves target recall but greatly increases control alerts.',
  'no_method_passes_universal_release':'True',
 },
 'case_hash_flags':[{'id_hash':hashlib.sha256(str(x['id']).encode()).hexdigest(),'source_stratum':x['stratum'],'flag_OCR':bool(x['proposals']['OCR_only']['any']),'flag_CLIP':bool(x['proposals']['CLIP_only']['any']),'flag_source_geometry':bool(x['proposals']['geometry_only']['any']),'flag_union':bool(x['proposals']['OCR_or_CLIP']['any'])} for x in d['cases']],
 'limitations':d['warnings']+['No expert-human final owner GUI labels, actor/3D mesh semantics, real defect audiovisual timing or live SWAAG analyzer integration evaluated in this specific new 72-slide test.','Previous 343 positive/control cases and new 72 triple-negative cases have DIFFERENT source-selection schemes. Do not combine their TP/FP counts to claim an unbiased overall deployment accuracy.'],
 }
 PUBLIC.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 bypos=lambda t,m:old[t][m]
 fmt=lambda t,m:f"{bypos(t,m)['TP']}/{bypos(t,m)['TP']+bypos(t,m)['FN']} detected, {bypos(t,m)['FP']}/{bypos(t,m)['FP']+bypos(t,m)['TN']} flags on controls"
 clip='Content Overflow/Cut-off';occ='Occluded Content';con='Insufficient Color Contrast for Readability'
 REPORT.write_text(f'''# Human-unanimous visual negatives: exact deployment error/cost choices — 26 October 2026

**Research status PARTIAL.** Original source <https://github.com/zhuohaouw/SlideAudit> (UIST 2025, pinned `642d490b7c1d2e78a50a631bfd359433397f3ecf`, CC BY 4.0); study <https://pmc.ncbi.nlm.nih.gov/articles/PMC13227769/>. The authors collected 2,400 presentation slides, 1,800 deliberately altered, and had three trained Prolific raters per slide. Their *majority* flaw labels require at least 2/3, while their dataset's `has_strong_agreement` requires **unanimous 3/3**. Earlier project reports sometimes use the generic phrase “strong agreement” without distinguishing consensus thresholds; the original paper establishes the exact distinction. They are independently human annotated, **NOT independent expert visual certification of native GUIs, machines or videos.**

## Completely new heldout negative source, frozen before model outputs

Among 1,102 original slides never used in the earlier 343-source pixel study, exactly 72 were sampled by fixed SHA-256 order: six each for the 12 source-type/alteration strata. All 72 have **unanimous human negatives in all three** categories: clipping, occlusion, insufficient contrast. The selection was frozen SHA `{source['predeclared_selection_SHA256']}` before new actual images were downloaded. Original 72 source image bytes, original human annotations, descriptions and model weights remain isolated under `/data/var/perceptual-independent-20261024/`; project Git contains no third-party images. Each original 3-property label was rechecked after independent pixel/CLIP/source predictions froze, with exact input SHA binding and **no OCR timeouts**.

| Same 72 actual rendered images, original frozen thresholds | False clipping flag | False occlusion flag | False contrast flag | Screens falsely proposed for ≥1 flaw |
|---|---:|---:|---:|---:|
| Tesseract/OpenCV original rule | {by[clip]['OCR_only']} | {by[occ]['OCR_only']} | {by[con]['OCR_only']} | **{neg['OCR_only']}/72** |
| Original pretrained CLIP ViT-B/32 same six prompts | {by[clip]['CLIP_only']} | {by[occ]['CLIP_only']} | {by[con]['CLIP_only']} | **{neg['CLIP_only']}/72** |
| Authored source rectangle rule | {by[clip]['geometry_only']} | {by[occ]['geometry_only']} | **UNKNOWN**, no visual contrast measurement | **{neg['geometry_only']}/72** |
| OCR or CLIP review-proposal union (posthoc) | {by[clip]['OCR_or_CLIP']} | {by[occ]['OCR_or_CLIP']} | {by[con]['OCR_or_CLIP']} | **{neg['OCR_or_CLIP']}/72** |
| All three union (posthoc) | {by[clip]['all_three_union']} | {by[occ]['all_three_union']} | {by[con]['all_three_union']} | **{neg['all_three_union']}/72** |
| Never propose anything (null) | 0 | 0 | 0 | **0/72**, but misses every true defect |

Source/alteration-stratified 4,000-replicate bootstrap 95% descriptive ranges for screen-level *candidate false review proposal rate* are OCR {d['heldout_72_strata_bootstrap_95pct_false_proposal_rate']['OCR_only']}, CLIP {d['heldout_72_strata_bootstrap_95pct_false_proposal_rate']['CLIP_only']}, source geometry {d['heldout_72_strata_bootstrap_95pct_false_proposal_rate']['geometry_only']}, OCR+CLIP union {d['heldout_72_strata_bootstrap_95pct_false_proposal_rate']['OCR_or_CLIP']}. Correlation and purposive selection rule out an iid population-error guarantee. All are categories on *slide renderings*, not production GUIs. Other design defects on these slides are not excluded.

## Miss rates on a different, previously scored positive/control selection

These correspond to precisely the same **old 343 raster images** for each analyzer, independently human annotated before model outputs, but with a different enriched target-positive selection. The following numbers were rederived from frozen per-case predictions and original category-specific human source labels. They are **not** an unbiased combined new+old accuracy statistic.

| Source human category | OCR pixel analyzer | CLIP pixel model | Source geometry | OCR + CLIP review union |
|---|---|---|---|---|
| Clipping, 82 positive and 82 category-negative controls | {fmt(clip,'OCR_only')} | {fmt(clip,'CLIP_only')} | {fmt(clip,'geometry_only')} | **{fmt(clip,'OCR_or_CLIP')}** |
| Occlusion, 105 positive and 105 controls | {fmt(occ,'OCR_only')} | {fmt(occ,'CLIP_only')} | {fmt(occ,'geometry_only')} | **{fmt(occ,'OCR_or_CLIP')}** |
| Insufficient contrast, 20 positive and 20 controls | {fmt(con,'OCR_only')} | {fmt(con,'CLIP_only')} | INELIGIBLE | **{fmt(con,'OCR_or_CLIP')}** |

The additional union improves true-defect recall but has unacceptable human-negative accusation rates. If used to reject automatically, the OCR+CLIP union would falsely reject 68/82 clipping-negative old controls, 81/105 occlusion-negative controls and 16/20 low-contrast-negative controls. If used *only* for proposing selective human review, it still misses 7/82, 26/105 and 5/20 of the original human-positive cases. “Review flagged only and approve everything else” is therefore not qualified.

## Real measured CPU cost and source-authority decisions

| Candidate on exact same 72 source images | Actual batch CPU wall time | Actual CPU seconds/image | Peak measured process RSS |
|---|---:|---:|---:|
| Source object rectangle geometry (extra structured metadata input) | **{cost['measured_batch_wall_72_by_rule_s']['geometry_only']:.3f} s** | {cost['measured_average_cpu_wall_per_image_by_rule_s']['geometry_only']:.4f} | Not independently isolated |
| CLIP ViT-B/32 | **{cost['measured_batch_wall_72_by_rule_s']['CLIP_only']:.3f} s** | {cost['measured_average_cpu_wall_per_image_by_rule_s']['CLIP_only']:.4f} | {source['cost']['CLIP_peak_RSS_KB']:,} KiB |
| OCR/OpenCV | **{cost['measured_batch_wall_72_by_rule_s']['OCR_only']:.3f} s** | {cost['measured_average_cpu_wall_per_image_by_rule_s']['OCR_only']:.4f} | {source['cost']['pixel_OCR_CV_peak_RSS_KB']:,} KiB |

CLIP's total includes **{cost['CLIP_cold_load_wall_s']:.3f} s** cold model load. Simple serial addition of independently measured OCR and CLIP stages is **{cost['measured_batch_wall_72_by_rule_s']['OCR_or_CLIP']:.3f} s /72**, but **that is NOT a separately measured end-to-end simultaneous/serial deployment runtime**. OCR and CLIP share identical image inputs; source geometry uses additionally the original authored object JSON and therefore is not a fair equal-input standalone vision comparison. GPU use was **zero**. Download time **{source['cost']['download_wall_s']:.3f} s**, {source['cost']['download_MB']:.2f} MB of source material, excluded from analyzer timings. Human review cost/latency is not measured and must not be fabricated.

**Operational decisions:** (1) A verified exact property failure (e.g. actual rendered-frame SHA/PTS integrity, source-authoritative mesh contacts or an instrumented DOM clipping ancestor) can automatically fail **that property only**, after independently confirming final output, source revision, unit/viewport and contract. (2) Generic CLIP, OCR word boxes, source geometry without verified rendered state, visual contrast proxies and optical flow are **proposals for review**, not automatic reject or approve. (3) Once intent, provenance or human-readable content becomes uncertain, report REVIEW, not green. (4) Distinguish author source graph from actually visible arrowheads, static mesh source contacts from photorealism/engineering and 15 fps final silent warehouse MP4 frame correspondence from natural video/audio visual acceptability.

**Contradictions resolved:** a positive source rectangle overlap may be absolutely correct about authored rectangles yet a false human-*visible occlusion* claim because overlap can be intentional or absent from altered final pixels. Conversely a human-visible cutoff can occur inside an authored onscreen text box. The three SlideAudit source categories have real independent human labels, but the source shows the authors' 600 original/1800 altered distribution, so the results are not native UI defect rates. Earlier final owner warehouse MP4 fidelity passed all 210 rendered-frame comparisons and reached HTML5 playback end; it remains silent and has no independently human-rated photorealism or unintended motion labels. None of these source facts contradicts the observed model misses and false alerts.

**Unproved:** real owner final GUI, diagram arrowhead image-versus-source direction, machine semantic part completeness/engineering/photorealism, natural event-time-coded video/audio faults, qualified full SWAAG integration and human review time. Overall status **PARTIAL**.

Reproduction of this new frozen study: `python3 tests/research/slideaudit_unseen_joint_negative_20261026.py freeze|download|pixel|geometry|clip|score`; `python3 tests/research/slideaudit_deployment_policy_tradeoffs_20261026.py`; `python3 tests/research/publish_slideaudit_deployment_choices_20261026.py`. All stages refuse rewriting frozen predictions. Source SHA receipts and exact per-case predictions are retained on Nitro; no GPU services or earlier denied effects were used. The CLI stage sequence is for a separate explicitly fresh replication, **not** rerunning this already complete experiment.
''')
 print('PUBLISHED_HUMAN_DEPLOYMENT_REPORT',PUBLIC,sha(PUBLIC))

if __name__=='__main__':main()
