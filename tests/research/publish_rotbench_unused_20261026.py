#!/usr/bin/env python3
"""Publish app-grouped uncertainty and paired test of still-unused source cases."""
import hashlib
import json
import random
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path('/data/var/perceptual-independent-20261024/rotbench-unused-source-split-20261026')
OLD=Path('/data/var/perceptual-independent-20261024/rotbench-android-heldout-20261026')
OUTPUT=Path(__file__).resolve().parents[2]/'docs/research'
DOC=OUTPUT/'perceptual-rotbench-unused-heldout-replication-2026-10-26.md'
JSON=OUTPUT/'perceptual-rotbench-unused-heldout-replication-2026-10-26.json'

def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rate(rows,k):
 v=[x for x in rows if x['predictions']['aligned'][k] is not None]
 pos=[x for x in v if x['external_author_bug']]
 neg=[x for x in v if not x['external_author_bug']]
 return {'TP':sum(x['predictions']['aligned'][k] for x in pos), 'FN':sum(not x['predictions']['aligned'][k] for x in pos), 'FP':sum(x['predictions']['aligned'][k] for x in neg), 'TN':sum(not x['predictions']['aligned'][k] for x in neg), 'abstain':len(rows)-len(v)}
def rec(cm):return cm['TP']/(cm['TP']+cm['FN']) if cm['TP']+cm['FN'] else None

def resample_apps(rows,seed=20261026,draws=6000):
 apps=defaultdict(list)
 for row in rows:apps[row['app']].append(row)
 names=sorted(apps)
 rng=random.Random(seed)
 recalls=[];improvements=[];fps=[]
 for _ in range(draws):
  b=[x for _ in names for x in apps[rng.choice(names)]]
  c=rate(b,'localized'); g=rate(b,'global')
  if rec(c) is not None and rec(g) is not None:
   recalls.append(rec(c));improvements.append(rec(c)-rec(g))
  n=c['TN']+c['FP'];
  if n:fps.append(c['FP']/n)
 def ci(a):
  s=sorted(a)
  return [round(s[int(len(s)*.025)],4),round(s[min(len(s)-1,int(len(s)*.975))],4)] if s else None
 return {'unit':'APP, cases within app sampled together, not independent individual screenshots','sample_distinct_apps':len(names),'draws':draws,'seed':seed,'localized_recall_app_cluster_bootstrap95':ci(recalls),'localized_minus_global_recall_app_cluster_bootstrap95':ci(improvements),'observed_FPR_bootstrap95_on_identical_controls_ONLY':ci(fps),'cluster_bootstrap_recall_difference_pr_le_zero':round(sum(q<=0 for q in improvements)/len(improvements),5) if improvements else None}

def main():
 if DOC.exists() or JSON.exists():raise RuntimeError('reports already exist: no rewriting recorded evidence')
 scores=json.loads((ROOT/'evaluated.json').read_text())
 frozen=json.loads((ROOT/'predictions.json').read_text())
 prev=json.loads((OLD/'scored.json').read_text())
 selected=json.loads((OLD/'selection.json').read_text())
 cases=scores['table']
 assert len(cases)==240 and scores['evaluation_sha256_of_frozen_predictions']==hashfile(ROOT/'predictions.json')
 assert scores['selection_sha256']==hashfile(ROOT/'selection.json')
 prev_ids={x['record_index'] for x in selected['selection']}
 assert not prev_ids.intersection(x['id'] for x in cases)
 assert all(x['app'] for x in cases)
 previous_apps={x['references'][0].split('/')[2] for x in selected['selection']}
 new_apps={x['app'] for x in cases}
 confusion={k:rate(cases,k) for k in ('localized','global','any_pixel')}
 assert confusion['localized']==scores['all']['localized']
 assert confusion['global']==scores['all']['global']
 assert confusion['any_pixel']==scores['all']['any_pixel']
 unknown=[x for x in cases if x['predictions']['aligned']['localized'] is None]
 zero_controls=[x for x in cases if x['type']=='right']
 exact_controls=sum(all(x['aligned_bytestream_identical']) for x in zero_controls)
 win=[x for x in cases if x['external_author_bug'] and x['predictions']['aligned']['localized'] and not x['predictions']['aligned']['global']]
 reverse=[x for x in cases if x['external_author_bug'] and x['predictions']['aligned']['global'] and not x['predictions']['aligned']['localized']]
 all_normal=[x for x in cases if not x['external_author_bug']]
 forced_bad=sum(x['predictions']['wrong_viewport_forced']['localized'] is True for x in all_normal)
 forced_abstain=sum(x['predictions']['wrong_viewport_forced']['localized'] is None for x in all_normal)
 opposite_abstain=sum(x['predictions']['wrong_viewport_correct_gate']['abstain_due_to_dimensions'] for x in all_normal)
 miss_ref=sorted(frozen['missing_original_release_files'])
 assert all(x.endswith('.png') for x in miss_ref)
 boot=resample_apps(cases)
 group={t:{'cases':sum(x['type']==t for x in cases), 'positive':sum(x['type']==t and x['external_author_bug'] for x in cases), 'localized':rate([x for x in cases if x['type']==t],'localized'), 'global':rate([x for x in cases if x['type']==t],'global')} for t in sorted({x['type'] for x in cases})}
 report={'date':'2026-10-26','status':'PARTIAL','original_source':'https://huggingface.co/datasets/ImDim/RotBench','source_sha256':scores['external_source_sha'], 'source_revision':'25b6e8214c5be600f54dc0834eaf4597fa9192e6','frozen_receipts':{n:hashfile(ROOT/n) for n in ('design.json','selection.json','predictions.json','evaluated.json')},'previous_source_evidence_sha256':hashfile(OLD/'scored.json'),'previous_report_sha256':hashfile(OUTPUT/'perceptual-rotbench-android-heldout-2026-10-26.json'),
   'selection':{'parent_test':340,'prior_development_examples':100,'previously_unused_heldout':len(cases),'prior_apps':len(previous_apps),'new_apps':len(new_apps),'shared_apps_across_rounds':len(previous_apps.intersection(new_apps)),'heldout_unique_apps':len(new_apps),'source_biased_author_mutations':True,'app_grouped_uncertainty_not_new_human_labels':True},
   'counts':confusion,'groups':group,'negative_control':{'real_author_clean_right_cases':len(zero_controls),'exact_approved_reference_bytes':exact_controls,'wrong_viewport_forced_bad_flags_on_author_clean':forced_bad,'wrong_viewport_forced_abstain':forced_abstain,'wrong_viewport_source_binding_ABSTAINS_on_author_clean':opposite_abstain,'wrong_viewport_control_does_not_match_state':'portrait->landscape originals of same activity are author approved but should not be matched to wrong viewport. Forced resize comparator deliberately violates contract, NOT a legitimate correctly bound false positive.'},
   'errors':{'unavailable_pinned_source_files_count':len(miss_ref),'unavailable_pinned_source_paths':miss_ref,'cases_abstained':len(unknown),'abstain_original_labels_by_type':dict(Counter(x['type'] for x in unknown)),'no_missing_source_filled_with_guessed_media':True,'app_group_bootstrap':boot,'localized_only_positive_cases':len(win),'global_only_positive_cases':len(reverse)},
   'resources':{'total_native_RGB_CPU_plus_wrong_viewport_resizing_seconds':frozen['pixel_feature_seconds'],'distinct_original_images_requested':frozen['num_original_PNGs'],'download_seconds':frozen['download_seconds'],'download_origins':frozen['download_origin'],'gpu_inference':False,'first_failed_source_acquisition_attempt_seconds':110.781,'successful_resume_download_seconds':frozen['download_seconds'],'recorded_combined_attempts_download_plus_features_seconds':round(110.781+frozen['download_seconds']+frozen['pixel_feature_seconds'],3),'initial_failure':'An initial 110.781 s source acquisition phase downloaded original PNGs but stopped when an author-referenced clean image was HTTP 404, before any predictions were frozen. Subsequent resumption used cached unchanged original bytes and abstained on missing source references.','unknown_latency':'Live end-to-end SWAAG tool-call latency absent; numbers are batch wall time on busy Nitro CPU, both methods share image decoding; forced viewport comparisons consume additional processing.'},
   'original_gold_status':'Author release labels distinguish originally right screens from seeded mutants; NOT independent human acceptance ratings; no pixel-level blame regions/visible severity ground truth, no actual owner output independent gold.',
   'per_case':[{k:v for k,v in x.items() if k not in ('reference_name',)} for x in cases],
   'limitations':['Retained unused subset is disjoint by record id but original source app identities overlap with prior first round. Thresholds were fixed by prior round, not tuned to these gold labels.','Author right controls are byte-identical to source-approved screenshot in same viewport, not naturally changed app states.','Real clean portrait and landscape screenshots intentionally have different layouts; forced wrong-state comparison demonstrates source version/input mismatch, not an acceptable detector.','Opposite viewport correct gate must abstain; app semantics under a different viewport require its own approved reference.','Synthetic bug annotations are not an independent human evaluation of real-world objectionability or precise region quality.','No human-gold natural video temporal faults or real owner final render quality scores, and live SWAAG integration remains unverified.']}
 JSON.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
 c=confusion['localized'];g=confusion['global']; oracle=confusion['any_pixel']
 DOC.write_text(f'''# Remaining original RotBench holdout: threshold-generalization and reference validity — 26 October 2026

**Research state: PARTIAL.** We extended the preceding **100-case** RotBench original Android screenshot study to the **240 previously unused** records of the same upstream 340-case test split. The source revision and full test JSON SHA remained unchanged. The new selection and pixel-only predictions were frozen under SHA `{report['frozen_receipts']['selection.json']}` and `{report['frozen_receipts']['predictions.json']}` **before** joining the source authors' labels. The fixed localized threshold (changed RGB channel >=24 in >0.05% of pixels), fixed global mean threshold (>1.5), and null were carried over unchanged from the earlier study. New sources are disjoint *records*, not an independent new dataset or source applications. No original PNGs or private reference bytes have been committed; original screenshot license remains unspecified in the locally retrieved dataset metadata.

## New author-label comparisons, identical native decoded source inputs

| Fixed on previous 100 cases; new 240 records | True positives | False positives | False negatives | True negatives | Abstentions |
|---|---:|---:|---:|---:|---:|
| Localized changed-pixel threshold | {c['TP']} | {c['FP']} | {c['FN']} | {c['TN']} | {c['abstain']} |
| Global RGB mean threshold (same source pixels) | {g['TP']} | {g['FP']} | {g['FN']} | {g['TN']} | {g['abstain']} |
| Any source-reference pixel difference (leakage/upper bound, not a valid visual quality judge) | {oracle['TP']} | {oracle['FP']} | {oracle['FN']} | {oracle['TN']} | {oracle['abstain']} |
| Always predict no defect | 0 | 0 | {scores['all']['always_clean']['FN']} | {scores['all']['always_clean']['TN']} | 0 |

Original author bug families and exact breakdowns are in the adjacent JSON. The full source test has five equal-size bug types and 170 original-right cases, but this *heldout remainder* is not population-representative: previous round consumed 25/34 layout-clip, 25/34 layout-overlap and 50/170 right examples. Remaining subset has 9 clip, 9 overlap, all 34 each of three other mutation families, and 120 right. No model inference, threshold adaptation, label-based selection or reclassification of missing cases occurred. The 240 cases contain `{len(new_apps)}` distinct Android applications, `{len(previous_apps.intersection(new_apps))}` also in the earlier sample; app-grouped (rather than screenshot independent) 6,000-draw descriptive percentile bootstrap puts localized recall in **{boot['localized_recall_app_cluster_bootstrap95']}** and its improvement versus same-input global averaging in **{boot['localized_minus_global_recall_app_cluster_bootstrap95']}**. These intervals remain conditional on an author's synthetic mutations and an easy control construction, NOT production reliability.

## False-positive stress test of an INVALID source reference

**{exact_controls}/{len(zero_controls)}** author-normal controls have source image bytes exactly identical to their correctly chosen same-viewport originals. This guarantees that any positive numerical pixel-difference threshold will never falsely accuse those exact controls. It is therefore scientifically invalid to interpret zero false alerts as evidence that the analyzer can tolerate dynamic application changes. For the exact same actual author-normal portrait/landscape screens, deliberately choose the **opposite approved orientation** as a reference (and resize). The pixel-only rule accuses **{forced_bad}/{len(zero_controls)}** genuine author-normal examples (and abstains on {forced_abstain}). The proper source-binding rule instead abstains on **{opposite_abstain}/{len(zero_controls)}** controls because the expected viewport dimensions/state do not match. This intentionally invalid-reference stress test is **not** a fair same-input performance comparison with correctly aligned screenshot reference; it establishes that reference/viewport identity is indispensable and output must be REVIEW, not semantic FAIL, for mismatched states.

Pinned-source availability: **{len(miss_ref)}** original reference PNG paths were unavailable (HTTP 404) at the pinned revision, yielding **{len(unknown)} abstained cases** rather than invented gold or silent classification. Existing verified unavailable original paths were not retried; other source media were downloaded read-only, with exact SHA receipts outside Git. The dataset licenses and human perceived severity remain unspecified; author labels are not pixel-wise human annotations. No production images/videos/warehouse source files, GPU services or earlier denied retrieval attempts were changed.

## Cost, trust and remaining gap

Pixel extraction, including the extra mismatched-orientation diagnostic, took **{frozen['pixel_feature_seconds']} seconds** on low-priority Nitro CPU for {len(cases)} case pairs and {frozen['num_original_PNGs']} distinct source images. Download wall time for the successful cache-based resume was {frozen['download_seconds']} seconds, **in addition to an earlier 110.781-second source acquisition attempt that stopped at an HTTP 404 before any predictions froze**. The recorded total for the two attempts plus pixel processing is at least {round(110.781+frozen['download_seconds']+frozen['pixel_feature_seconds'],3)} seconds; no GPU was used. This processing includes shared image decoding and two pixel rules, so it is NOT a separate OCR/CV/VLM per-call inference benchmark. Scope-qualified source checks may auto-fail only a truly verified immutable output-image match contract, after checking final pixels, exact approved reference version/viewport and intended-change allowance. The novel result does not qualify screenshot differencing as a generic clipping, overlap, contrast, graph-arrow, industrial-assembly or temporal-video judge.

The owner's actual warehouse demonstrated 0/72 invalid rack contacts before source repair and 72/72 after correction on one correlated source design. The previous real author animation-freeze videos were detected by best fixed image rule only 1/12. **Still missing:** human-verified actual owner GUI positive and legitimate dynamic negative cases, separately adjudicated 3D scene/render completeness, source-linked video fault onset/offset and clean matched codec/FPS controls, and real end-to-end SWAAG integration. The existing `inspect_image` tool remains disabled by default, and the independently added property acceptance policy is not yet wired to a production tool verdict.

## Reproduction

This round's source-only study uses `tests/research/rotbench_unused_split_20261026.py` (`freeze`, `predict`, `evaluate` stages) and `tests/research/publish_rotbench_unused_20261026.py`; stages refuse to overwrite artifacts. Only offline scoring/integrity tests can be rerun without creating a distinct new frozen evaluation. Exact original JSON, PNG media, predictions, source-unavailable status and individual raw SHA receipts reside under `/data/var/perceptual-independent-20261024/rotbench-unused-source-split-20261026/` and are not published. The compact evidence JSON retains all 240 per-case outcomes for reproducibility without redistributing third-party images. No scheduled messages or automation were created.
''')
 print('PUBLISHED',DOC,JSON,'sha',hashfile(JSON))

if __name__=='__main__':main()
