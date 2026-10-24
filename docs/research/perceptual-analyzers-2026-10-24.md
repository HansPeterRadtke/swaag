# Perceptual analyzer research — 2026-10-24 (partial)

## Recorded question and provenance

The original testing question is in Infra `docs/guidlines/original-recordings/general/testing/Recording 2026-09-23 08-47-18-577.txt`; selected prior summary is `docs/guidlines/appendix/research-and-alternatives.md` § Perceptual analyzer toolkit research and § Live perceptual testing. Related GUI acceptance criteria: `docs/guidlines/testing.md`, `gui/README.md`, `gui/recipes/tests_and_verification.md`, `gui/bidirectional_ai_ui_and_live_design.md`. Do not infer that this pointer is proof those full files have been read in this pass.

## Actual measurements, with failure retained

Thor benchmark file `/data/var/swaag/perception-benchmark/acceptance-20261009.json`, independently re-read 2026-10-24: Qwen3-VL 8B (as recorded; exact backend build not rechecked), `clean` none 3.43s correct; `overlap` overlap 3.57s correct; `clipped` predicted none instead of clipped_text 3.30s **FAIL**; `low_contrast_error` 3.77s correct. Total 3/4, not accepted. Harness: `scripts/perceptual_fixture_probe.py`. This is four examples, not defensible statistical accuracy.

A new CPU-only synthetic experiment `tests/research/perceptual_cpu_fixtures.py` writes fixed images and gold geometry/contrast metadata to `docs/research/perceptual-fixtures-20261024/`. On Nitro, the *metadata-based* geometry/contrast rule passed 6/6 (none 3, clipping 1, overlapping 1, low contrast 1). Initial flawed near-touch fixture extended past the canvas and produced 5/6; adjusted to stay inside the canvas and reran 6/6. All counts and per-case timings in `cpu_results.json`. Labels are specified by fixture construction; no image-only detector is being evaluated. This test is useful for browser/DOM or source-bounds checks, **not** an image-only vision baseline, and its artificially easy gold labels cannot establish production recall. New tests used no GPU, did not restart a service and did not submit a model call.

## Working recommendations, not measured model rankings

For instrumented GUIs: acquire DOM/accessibility bounding boxes and computed styles; use rectangle intersection with intentional-overlap exclusions, text element scrollWidth/clientWidth checks, canvas visibility checks, WCAG relative luminance contrast ratios, font size and scale assertions, and multi-viewport screenshots. False positives include legal overlays and intentional text overflow; the test suite must explicitly label these. For screenshot-only UI: compare OCR bounding boxes/text with reference, compute clipped-glyph edge evidence and pixel-level contrast, then run a semantic VLM only on unresolved cases. OCR cannot prove completeness of text unseen beyond a crop. Text legibility needs actual font-resolution/contrast thresholds and user-validated samples.

For diagrams: check graph edge crossings/node bbox overlap, topology/connectivity, annotation legibility, proportions and rendering at delivery size; ask VLMs to verify meaning only after exact textual/structural comparisons. For rendered 3D machinery: physically validate dimensions, topology/normals, intersections, ground contact, material/lighting plausibility and multi-view silhouettes using scene geometry/depth/normal passes before appearance-only VLM scoring. Photorealism ranking requires side-by-side independently rated image holdouts. For moving video: compute camera transforms, optical flow, feature trajectories, object identity/occlusion consistency, exposure changes and cut-aware frame differences. Optical-flow spikes must distinguish intentional edits/cuts from temporal artifacts. No measured comparison on real 3D/video assets is claimed here.

Candidate models include the existing Thor Qwen3-VL 8B (failed UI gate), smaller image-language models suitable for six GiB Nitro memory only after checking actual load/budget, and task-specific OCR/UI parsers. Their image-based precision/recall, model build, VRAM, latency and benchmark samples remain **unmeasured** in this iteration. Thor production services and Nitro 3D rendering take priority.

## Scope and coverage ledger

- First instruction read: `/data/infra/AGENTS.md`, in full, before further operations.
- Repository inventory: `git ls-files` counted 817 tracked paths including service and secret-related paths; this is **inventory only**, not complete human-readable coverage.
- Inspected Git status before writing: Infra main ahead six commits and has substantial other-agent modifications. No Infra files modified.
- SWAAG main was initially clean and ahead 50 commits. Existing `scripts/perceptual_fixture_probe.py` inspected.
- Selected Infra appendix paragraphs and original testing/GUI recording search hits inspected; full bounded reading of every tracked human-readable Infra file and all pertinent recording content remains incomplete.
- Original Thor JSON record read directly, not inferred from secondary text.
- New six-fixture local metadata experiment completed; all fixture images, predictions and per-case data retained.
- External research citations are preserved in the existing Infra appendix. No independent current primary-source paper review completed in this pass.
- No Nitro/Thor vision-model comparative run, OCR accuracy test, true screenshot-only precision/recall, human-rated realism set, video continuity evaluation, or cross-model calibration performed. No general analyzer registry is validated.

## Reproduction

`python3 tests/research/perceptual_cpu_fixtures.py`

Check `docs/research/perceptual-fixtures-20261024/cpu_results.json`. This script expects PIL and DejaVu Sans on Nitro and does not contact any model endpoint.
