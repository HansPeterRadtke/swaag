# Human-unanimous visual negatives: exact deployment error/cost choices — 26 October 2026

**Research status PARTIAL.** Original source <https://github.com/zhuohaouw/SlideAudit> (UIST 2025, pinned `642d490b7c1d2e78a50a631bfd359433397f3ecf`, CC BY 4.0); study <https://pmc.ncbi.nlm.nih.gov/articles/PMC13227769/>. The authors collected 2,400 presentation slides, 1,800 deliberately altered, and had three trained Prolific raters per slide. Their *majority* flaw labels require at least 2/3, while their dataset's `has_strong_agreement` requires **unanimous 3/3**. Earlier project reports sometimes use the generic phrase “strong agreement” without distinguishing consensus thresholds; the original paper establishes the exact distinction. They are independently human annotated, **NOT independent expert visual certification of native GUIs, machines or videos.**

## Completely new heldout negative source, frozen before model outputs

Among 1,102 original slides never used in the earlier 343-source pixel study, exactly 72 were sampled by fixed SHA-256 order: six each for the 12 source-type/alteration strata. All 72 have **unanimous human negatives in all three** categories: clipping, occlusion, insufficient contrast. The selection was frozen SHA `0f277303e8213e08b0ec0ff55beee821ba06c322c7d32a9100ed2a6ba70ad3b6` before new actual images were downloaded. Original 72 source image bytes, original human annotations, descriptions and model weights remain isolated under `/data/var/perceptual-independent-20261024/`; project Git contains no third-party images. Each original 3-property label was rechecked after independent pixel/CLIP/source predictions froze, with exact input SHA binding and **no OCR timeouts**.

| Same 72 actual rendered images, original frozen thresholds | False clipping flag | False occlusion flag | False contrast flag | Screens falsely proposed for ≥1 flaw |
|---|---:|---:|---:|---:|
| Tesseract/OpenCV original rule | 19 | 8 | 2 | **26/72** |
| Original pretrained CLIP ViT-B/32 same six prompts | 49 | 47 | 46 | **66/72** |
| Authored source rectangle rule | 2 | 26 | **UNKNOWN**, no visual contrast measurement | **26/72** |
| OCR or CLIP review-proposal union (posthoc) | 61 | 48 | 48 | **70/72** |
| All three union (posthoc) | 61 | 57 | 48 | **70/72** |
| Never propose anything (null) | 0 | 0 | 0 | **0/72**, but misses every true defect |

**Scoring erratum:** The original post-hoc aggregate counted null source-geometry contrast outputs as zero flags when summing booleans. That is an ineligible detector, not an observed true negative; this report and compact JSON correct it to NOT APPLICABLE. The original derived raw summary and all source/model prediction files are retained unchanged with SHA-256 receipts.

Source/alteration-stratified 4,000-replicate bootstrap 95% descriptive ranges for screen-level *candidate false review proposal rate* are OCR [0.2778, 0.4583], CLIP [0.8611, 0.9722], source geometry [0.2639, 0.4583], OCR+CLIP union [0.9444, 1.0]. Correlation and purposive selection rule out an iid population-error guarantee. All are categories on *slide renderings*, not production GUIs. Other design defects on these slides are not excluded.

## Miss rates on a different, previously scored positive/control selection

These correspond to precisely the same **old 343 raster images** for each analyzer, independently human annotated before model outputs, but with a different enriched target-positive selection. The following numbers were rederived from frozen per-case predictions and original category-specific human source labels. They are **not** an unbiased combined new+old accuracy statistic.

| Source human category | OCR pixel analyzer | CLIP pixel model | Source geometry | OCR + CLIP review union |
|---|---|---|---|---|
| Clipping, 82 positive and 82 category-negative controls | 58/82 detected, 26/82 flags on controls | 52/82 detected, 62/82 flags on controls | 9/82 detected, 7/82 flags on controls | **75/82 detected, 68/82 flags on controls** |
| Occlusion, 105 positive and 105 controls | 24/105 detected, 12/105 flags on controls | 74/105 detected, 78/105 flags on controls | 93/105 detected, 33/105 flags on controls | **79/105 detected, 81/105 flags on controls** |
| Insufficient contrast, 20 positive and 20 controls | 8/20 detected, 0/20 flags on controls | 12/20 detected, 16/20 flags on controls | INELIGIBLE | **15/20 detected, 16/20 flags on controls** |

The additional union improves true-defect recall but has unacceptable human-negative accusation rates. If used to reject automatically, the OCR+CLIP union would falsely reject 68/82 clipping-negative old controls, 81/105 occlusion-negative controls and 16/20 low-contrast-negative controls. If used *only* for proposing selective human review, it still misses 7/82, 26/105 and 5/20 of the original human-positive cases. “Review flagged only and approve everything else” is therefore not qualified.

## Real measured CPU cost and source-authority decisions

| Candidate on exact same 72 source images | Actual batch CPU wall time | Actual CPU seconds/image | Peak measured process RSS |
|---|---:|---:|---:|
| Source object rectangle geometry (extra structured metadata input) | **0.050 s** | 0.0007 | Not independently isolated |
| CLIP ViT-B/32 | **61.341 s** | 0.8520 | 1,193,516 KiB |
| OCR/OpenCV | **170.992 s** | 2.3749 | 332,552 KiB |

CLIP's total includes **2.733 s** cold model load. Simple serial addition of independently measured OCR and CLIP stages is **232.333 s /72**, but **that is NOT a separately measured end-to-end simultaneous/serial deployment runtime**. OCR and CLIP share identical image inputs; source geometry uses additionally the original authored object JSON and therefore is not a fair equal-input standalone vision comparison. GPU use was **zero**. Download time **14.331 s**, 34.67 MB of source material, excluded from analyzer timings. Human review cost/latency is not measured and must not be fabricated.

**Operational decisions:** (1) A verified exact property failure (e.g. actual rendered-frame SHA/PTS integrity, source-authoritative mesh contacts or an instrumented DOM clipping ancestor) can automatically fail **that property only**, after independently confirming final output, source revision, unit/viewport and contract. (2) Generic CLIP, OCR word boxes, source geometry without verified rendered state, visual contrast proxies and optical flow are **proposals for review**, not automatic reject or approve. (3) Once intent, provenance or human-readable content becomes uncertain, report REVIEW, not green. (4) Distinguish author source graph from actually visible arrowheads, static mesh source contacts from photorealism/engineering and 15 fps final silent warehouse MP4 frame correspondence from natural video/audio visual acceptability.

**Contradictions resolved:** a positive source rectangle overlap may be absolutely correct about authored rectangles yet a false human-*visible occlusion* claim because overlap can be intentional or absent from altered final pixels. Conversely a human-visible cutoff can occur inside an authored onscreen text box. The three SlideAudit source categories have real independent human labels, but the source shows the authors' 600 original/1800 altered distribution, so the results are not native UI defect rates. Earlier final owner warehouse MP4 fidelity passed all 210 rendered-frame comparisons and reached HTML5 playback end; it remains silent and has no independently human-rated photorealism or unintended motion labels. None of these source facts contradicts the observed model misses and false alerts.

**Unproved:** real owner final GUI, diagram arrowhead image-versus-source direction, machine semantic part completeness/engineering/photorealism, natural event-time-coded video/audio faults, qualified full SWAAG integration and human review time. Overall status **PARTIAL**.

Reproduction of this new frozen study: `python3 tests/research/slideaudit_unseen_joint_negative_20261026.py freeze|download|pixel|geometry|clip|score`; `python3 tests/research/slideaudit_deployment_policy_tradeoffs_20261026.py`; `python3 tests/research/publish_slideaudit_deployment_choices_20261026.py`. All stages refuse rewriting frozen predictions. Source SHA receipts and exact per-case predictions are retained on Nitro; no GPU services or earlier denied effects were used. The CLI stage sequence is for a separate explicitly fresh replication, **not** rerunning this already complete experiment.
