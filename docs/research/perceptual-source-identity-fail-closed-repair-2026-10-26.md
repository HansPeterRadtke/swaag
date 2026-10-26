# Exact source binding repair on real Android captured screens — October 26, 2026

**PARTIAL; concrete source-identity failure corrected for an opt-in exact immutable snapshot contract.** This round reread the complete original owner testing recording and Nitro Infra/testing/GUI guidelines, checked Git state clean at `a47f174`, inspected the original 100+240 RotBench results, 352 different-Activity clean-control receipts, the earlier 72 human-slide test and previous denied operations. No production GPU model/renderer/service was changed. Original data and uncertainty/denial receipts remain untouched.

## The measured real cause

A source-frame lookup can be **wrong even if the app/package, viewport size, orientation and source revision all match**. The already frozen external RotBench author-clean negative set contains 352 correct screenshots compared to a clean reference from a **different activity in the same app**; the image-only detector would accuse **352/352 pairs** of a defect. They represent only **137 distinct right/source cases across 28 Android apps**, not 352 independent applications. The original lower-level `decide_property` consumes caller-asserted verification flags but does not verify source identifiers, metadata binding or bytes itself. This is an integration vulnerability, not proof that the low-level function was actually called wrong in production.

Added `RenderSourceBinding` and `decide_bound_immutable_snapshot(...)` to `src/swaag/perception/acceptance_policy.py`. It requires exact source kind/revision/subject/activity or render-state/orientation/dimensions, independently attested source manifest and rendered state, prior SHA-256 receipts recomputed from **both actual original files**, and an explicit *immutable byte-identical snapshot* contract. Unknown binding, other activity, stale output/reference SHA, absent actual file, different source version, unknown source authority or permitted changes => **REVIEW**, never PASS/FAIL. When all preconditions are verified, it returns PASS for identical bytes and FAIL for changed bytes **only for that exact contract**. Attestation booleans must be established by a trusted runtime; this helper does not cryptographically prove a caller's metadata.

## Real source controls replayed, not a synthetic model test

| Original author data case | Previous unqualified interpretation | Strict new exact-source-bound outcome |
|---|---|---|
| 352 correct screens against **wrong activity** source, two orientations each | Previous pixel model flags 352/352 pairs | **0 auto-fails; 704/704 REVIEW** |
| Same 352 right screens compared to their own correct exact bytes, both orientations | Not a visual acceptance question | **704/704 PASS** for byte-equality ONLY |
| 50 author-injected source screenshot modifications | Previous localized pixel screen anomaly method 39/50, missed 11 | **50/50** changed-byte snapshot contract FAIL; NOT fifty human clipping recognitions |
| 50 unmodified source-right samples | Previous detector no discrepancy | **50/50** unchanged-byte PASS |
| Two source tests with missing reference PNGs at pinned original release | Previous correctly abstained | **2/2 REVIEW** with no HTTP request or invented bytes |

Original source SHA-256 `0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7` and the separately frozen replay protocol SHA-256 `50a5c26c4018579c9beabdbe7f9bc98f38db715bbfaa76714b398ce22a5825d0` bind the tested data; 408 distinct actual PNGs rehashed and matched their historical receipts. Stale metadata version, wrong reference or observed byte checksum, missing source proof, false activity attestation, non-immutable contract and unavailable reference each returned **REVIEW**; a correct self-reference alone returned PASS. No original file was modified and the old wrongly bound classifier outputs were preserved for audit.

**Costs:** 5.167 seconds low-priority CPU, peak resident memory 350,152 KiB, no GPU. Concurrent live Nitro sample load average 34.62, GPU 4735 MiB/6144 MiB occupied; trading and Chromium/3D processes remained untouched. This is not a clean single-request latency guarantee.

## Limits and next operational path

The 50 injected changes were authored on real Android screenshots; an immutable-byte check catches edits because they change bytes. It cannot tell whether text is clipped, contrast is inadequate or a redraw is legitimate. Use this gate only after a trusted source and **actual app/UI state** have been attested and the exact-byte property is warranted. General source/provenance checkers on previously approved *lossy* H264 movies require their own PSNR/timebase contract, not this byte-comparison function. `decide_property` is retained as the low-level assertion-only primitive for backward compatibility; live SWAAG `inspect_image` remains an evidence-only, default-disabled proposal tool. No live registered decision tool has been qualified by this offline replay.

**Previously observed errors and denials:** old storyboard transfer and source-resolution v52 retrieval denied, not retried; two original absent author references were not fetched again; eight source annotation rectangles remain outside native bounds and no coordinate remapping was invented. Two local *new* replay-harness schema mistakes (352 pairs versus 137 underlying source cases, and `id` versus `record_index`) were repaired before the frozen final result, without touching originals. Prior human-rated SlideAudit OCR/CLIP false alerts, TopoBench arrow-direction failures and industrial/temporal model misses remain unchanged.

**Remaining priority:** independent truly clean dynamic GUI and calibrated semantic defects; visible diagram arrowheads, engineering-rated machinery appearance, naturally occurring time-coded video/audiovisual errors, and authenticated source identity captured within a live SWAAG tool. Status **PARTIAL**.

Reproduction scripts: `tests/research/rotbench_source_binding_repair_replay_20261026.py freeze|evaluate`; results/referenced media under `/data/var/perceptual-independent-20261024/rotbench_source_binding_replay_20261026/` and earlier original pinned studies. Stages refuse to overwrite receipts. Compact report contains derived SHA and case-level hashed IDs; no third-party images, private assets or secrets were committed. Do not reissue already attempted denied tool operations.
