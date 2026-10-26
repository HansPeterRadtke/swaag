#!/usr/bin/env python3
"""Publish compact reproducible evidence; leave original author PNGs outside Git."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from scipy.stats import beta,binomtest

ROOT=Path('/data/var/perceptual-independent-20261024/rotbench-android-heldout-20261026')
OUT=Path(__file__).resolve().parents[2]/'docs/research'
J=OUT/'perceptual-rotbench-android-heldout-2026-10-26.json'
M=OUT/'perceptual-rotbench-android-heldout-2026-10-26.md'
FILES=('design.json','selection.json','source_test.json','predictions.json','sham_reference_predictions.json','scored.json')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ci(k,n):
    return [round(0 if k==0 else float(beta.ppf(.025,k,n-k+1)),4),round(1 if k==n else float(beta.ppf(.975,k+1,n-k)),4)]

def main():
    if J.exists() or M.exists():raise RuntimeError('EXISTING_PUBLICATION: refuse rewrite without inspection')
    receipts={name:digest(ROOT/name) for name in FILES}
    p=json.loads((ROOT/'predictions.json').read_text())
    sham=json.loads((ROOT/'sham_reference_predictions.json').read_text())
    scored=json.loads((ROOT/'scored.json').read_text())
    select=json.loads((ROOT/'selection.json').read_text())
    assert p['selection_sha256']==receipts['selection.json']==sham['selection_sha256']
    assert scored['frozen_prediction_sha256']==receipts['predictions.json']
    assert scored['source_sha256']==receipts['source_test.json']
    assert p['source_sha256']==receipts['source_test.json']
    a={x['record_index']:x for x in p['cases']}
    b={x['record_index']:x for x in sham['cases']}
    source={x['record_index']:x for x in select['selection']}
    rows=[]
    for c in scored['selected_cases']:
        idx=c['record_index']; orig=a[idx];n=b[idx]
        assert len(orig['images_sha256'])==2 and source[idx]['record_index']==idx
        props=[z['changed_frac24'] for z in orig['orientation_pixel_stats']]
        rows.append({'id':idx,'source_activity':source[idx]['references'][0].split('/')[3], 'app':source[idx]['references'][0].split('/')[2], 'group':c['group'], 'source_author_bug':c['gold_bug'], 'changed_fraction_max':round(max(props),9), 'max_global_rgb_mae':round(max(z['mean_abs_rgb'] for z in orig['orientation_pixel_stats']),6),'reference_image_identical':[bool(x) for x in orig['images_reference_equal_sha256']], 'aligned_localized':orig['predictions']['changed_area_0p05pct'], 'aligned_global':orig['predictions']['global_meanabs_gt1p5'],'mismatched_localized':n['predictions']['changed_area_0p05pct_sham'],'mismatched_global':n['predictions']['global_meanabs_gt1p5_sham']})
    assert len(rows)==100 and len(set(x['id'] for x in rows))==100
    assert sum(x['source_author_bug'] for x in rows)==50
    def cm(key):
        known=[x for x in rows if x[key] is not None]
        return {'TP':sum(x['source_author_bug'] and x[key] for x in known),'FP':sum(not x['source_author_bug'] and x[key] for x in known),'FN':sum(x['source_author_bug'] and not x[key] for x in known),'TN':sum(not x['source_author_bug'] and not x[key] for x in known),'abstained':len(rows)-len(known)}
    counts={key:cm(key) for key in ('aligned_localized','aligned_global','mismatched_localized','mismatched_global')}
    assert counts['aligned_localized']=={'TP':39,'FP':0,'FN':11,'TN':50,'abstained':0}
    assert counts['aligned_global']=={'TP':3,'FP':0,'FN':47,'TN':50,'abstained':0}
    assert counts['mismatched_localized']=={'TP':50,'FP':50,'FN':0,'TN':0,'abstained':0}
    assert counts['mismatched_global']==counts['mismatched_localized']
    misses=[x for x in rows if x['source_author_bug'] and not x['aligned_localized']]
    assert len(misses)==11 and all(x['group']=='mutated-layout-clip' for x in misses)
    assert all(0<x['changed_fraction_max']<=.0005 for x in misses)
    assert all(all(x['reference_image_identical']) for x in rows if x['group']=='original-right')
    pos=[x for x in rows if x['source_author_bug']]
    discordant=(sum(x['aligned_localized'] and not x['aligned_global'] for x in pos),sum(not x['aligned_localized'] and x['aligned_global'] for x in pos))
    report={
       'date':'2026-10-26','status':'PARTIAL','source':'https://huggingface.co/datasets/ImDim/RotBench','source_pinned_revision':json.loads((ROOT/'design.json').read_text())['revision'], 'source_and_predictions_SHA256':receipts,
       'source_gold_scope':'Author manually curated original right Android screens then independently injected synthetic clipping and overlap mutations. These are not new human-adjudicated screenshots or production user-visible severity labels.',
       'copyright_and_license':'Original image files / 47 MB isolated under /data/var and NOT in Git; no explicit screenshot license in retrieved dataset card.',
       'selection':'Externally released test split only, hash ranked 25 clip +25 overlap +50 right; image extraction before author-label join; names encode classes but prediction functions never inspect class names.',
       'sample':{'image_pairs':100,'downloaded_distinct_original_PNGs':p['num_distinct_pngs'],'unique_android_apps':len(set(x['app'] for x in rows)),'unique_activities':len(set(x['source_activity'] for x in rows)),'synthetic_clip':25,'synthetic_overlap':25,'right_controls':50},
       'method':{'main':'fraction of native RGB pixels whose max-channel abs difference from matched right reference >=24; flag if >0.0005','same_input_same_cost_baseline':'global mean RGB absolute delta >1.5 on exact same decoded image/reference arrays','negative_control':'shuffled other-activity CLEAN reference, exact same native orientation dimensions and pixel calculation; abstain if not available','null':'always no defect','gold_join':'after source-blind prediction and sham prediction SHA frozen','no_GPU_models_loaded':True},
       'confusion':counts,
       'subsets':scored['confusion_by_group'],
       'confidence_descriptive_exact_binomial':{'localized_recall_39of50':ci(39,50),'clip_recall_14of25':ci(14,25),'overlap_recall_25of25':ci(25,25),'localized_FPR_0of50':ci(0,50)},
       'paired_same_positive_global_vs_localized':{'localized_only':discordant[0],'global_only':discordant[1],'exact_discordance_p_conditional':float(binomtest(discordant[0],sum(discordant),.5).pvalue),'caution':'posthoc paired description with app correlated cases; not a preregistered model selection or extrapolation'},
       'false_negative_pixel_scope':{'n':len(misses),'all_have_different_PNG_bytes':True,'all_are_clip_mutations':True,'changed_area_fraction_min':min(x['changed_fraction_max'] for x in misses),'changed_area_fraction_max':max(x['changed_fraction_max'] for x in misses),'human_visible_severity_unknown':True},
       'cost':{'primary_and_global_shared_native_pixel_compute_seconds':p['source_check_pixel_feature_seconds'],'sham_identical_operations_pixel_compute_seconds':sham['pixel_feature_seconds'],'download_seconds':p['download_seconds'],'download_bytes_total':p['download_bytes_total'],'GPU_used':False,'resource_note':'busy Nitro, low-priority CPU; time per proposal cannot be separated when source-aligned features share decoded pixels'},
       'scientific_limits':['Controls are EXACT clean reference bytes, so zero false positives is a construction shortcut and not robust to legitimate changing native application content','Sources include 39 Android apps for 100 cases; samples are correlated, so independent-binomial confidence is illustrative only','Authors synthetic mutation label is not independent human judgment of actual visually objectionable clipping/overlap','An exact approved screenshot mismatch is a narrow source-defined property only, not sufficient to auto-reject semantic rendering flaws in dynamic screens','The source alignment method has extra clean reference input; sham controls isolate reference identity but cannot substitute for a same-input generic image-only model','No actual new owner render/audio/video, no natural frame-level fault timing gold, no certified engineering machine reference, no live SWAAG hook or production GPU inference evaluated'],
       'rows':rows,
    }
    J.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    M.write_text(f'''# Real Android GUI heldout source-reference perceptual check — 26 October 2026

**Status PARTIAL.** This is a new source-external actual Android screenshot test, separate from earlier SlideAudit/Firefox source and synthetic manual tests. Public source: [RotBench Android screenshot bug benchmark](https://huggingface.co/datasets/ImDim/RotBench), pinned revision `{report['source_pinned_revision']}`, test JSON SHA-256 `{receipts['source_test.json']}`. The source has original captured Android states and independently author-injected mutation labels. The positive mutations are synthetic and were **not independently human-reviewed for perceptual severity**. Only research artifacts and code were saved in SWAAG; original screenshot media stays in isolated `/data/var`, license unconfirmed, not distributed.

## Independent external labels and fixed matched negative control

The pre-gold frozen manifest SHA `{receipts['selection.json']}` selected 25 clip and 25 overlap mutants and 50 source normal cases by hash rank among test records, from **{report['sample']['unique_android_apps']} apps and {report['sample']['unique_activities']} activities**. The source JSON includes label-leaking filenames; paths were used for stratification, NOT for prediction. All 100 case predictions were computed using actual native-resolution decoded screenshots and the author-captured matching correct orientation image, then frozen under SHA `{receipts['predictions.json']}` **before** the original author bug labels were joined. The predeclared negative control used the same threshold, native image dimensions and operations but a clean reference screenshot from an *unrelated* activity, frozen separately with SHA `{receipts['sham_reference_predictions.json']}`. This directly tests a source identity shortcut; it does not test benign real UI state changes. Raw metadata and exact source files are private isolated research storage, not Git.

| Detector on same 100 screenshots | True defects found / 50 | False alerts / 50 controls | Defects missed / 50 |
|---|---:|---:|---:|
| Correct reference + localized pixel difference (changed area >0.05%, RGB channel difference >=24) | **39** | **0** | **11** |
| Same aligned actual source images, **global** mean RGB difference >1.5 | **3** | **0** | **47** |
| Same localized difference but **wrong app reference** | 50 | **50** | 0 |
| Null always predicts normal | 0 | 0 | 50 |

The localized method found **25/25 overlap** and **14/25 clip** author mutations. All eleven misses were clipping mutations that demonstrably changed native screenshot pixels by only **{100*min(x['changed_fraction_max'] for x in misses):.4f}% to {100*max(x['changed_fraction_max'] for x in misses):.4f}%** screen area; no missed mutation was a byte-identical no-op. Sensitivity 39/50 has descriptive exact-binomial 95% interval **{report['confidence_descriptive_exact_binomial']['localized_recall_39of50']}**; this is optimistic as independent inference because samples share 39 apps. A 0/50 false alert rate has an independent-binomial upper 95% bound of about **7.11%** even under these easy conditions; real benign state changes were not sampled at all.

The localized versus global method compared exactly the same decoded pixels, original correct references and low-priority CPU processing; 36 positive mutants were detected only by localized pixels, 3 by both, 11 by neither. This directly demonstrates that global visual averages conceal small clipped content even when correct reference frames are available. The source-aligned feature extraction took **{p['source_check_pixel_feature_seconds']} s**, the matched wrong-reference negative control **{sham['pixel_feature_seconds']} s** on busy Nitro CPU, no GPU. The wrong-reference negative produces 50/50 clean false alarms: successful detection utterly depends on **correct reference identity**. These timings include both shared localized/global measures in each pass, so are not separate model latencies. Frozen input features, SHA receipts and every per-case prediction and author label are in the companion JSON.

## Owner production decision

**Auto-fail only** the property *“the actual final screenshot deviates from an approved immutable original by more than the defined tolerance”* when both reference identity/version and final rendered screenshot are independently verified and the workflow really forbids changes. On a live changing Android/web UI, this test must **abstain or propose REVIEW**, not label a developer's legitimate changes as faults. OCR, UI VLM, region overlap, contrast proxies and a general high-confidence caption remain evidence proposals unless separately verified against appropriate actual final pixels and semantic requirements. The new external Android sample does not certify native GUI release, independent human-accepted low-contrast, natural valid screen changes, diagram arrows, industrial CAD photorealism, final owner warehouse rendering or temporal/audio sync. Existing actual owner source mesh contact data prove the narrow 72/72 corrected support-chain contract; existing TempGlitch benchmark had at most 1/12 localized character freezes. **Single decisive gap still unresolved: independent human/authoritative gold for actual final owner outputs with truly legitimate dynamic negative variations and time-coded physical/video defects.** No production service, GPU model, previous uncertain Enter, denied source-result retrieval or live asset was touched.

## Reproducibility

`nice -n 15 python3 tests/research/rotbench_frozen_actual_android_pixels_20261026.py freeze` then `extract`; `nice -n 15 python3 tests/research/rotbench_mismatched_reference_control_20261026.py`; then original script `score` and `python3 tests/research/publish_rotbench_android_20261026.py`. Each stage refuses to overwrite a frozen artifact; remove/repoint the isolated data directory only for a genuinely separately declared replication, not this recorded run. No external media is committed. In a new environment this requires permission to download the externally hosted public screenshots; the original author's test labels are only read at `score` time. The committed JSON supports offline integrity and confusion-count tests without the source media.
''')
    print('PUBLISHED',J,M,'evidence sha',digest(J))

if __name__=='__main__':main()
