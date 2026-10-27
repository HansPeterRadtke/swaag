# Independently replicated PNG source/pixel/end-boundary acceptance — October 27, 2026

**Status PARTIAL:** the exact immutable snapshot contract now validates *both independently host-attested artifact hashes and the actual final PNG pixels and end boundary*. Before this patch, the registered tool only inspected the first 24 bytes (PNG signature and IHDR width/height); it could treat a non-renderable/truncated final image as a valid snapshot if an attachment hash matched itself. The host attestor also provided no independent expected approved-image SHA or captured-target SHA, so a mismatched publisher source could be mistaken for authoritative input if its attachment content was otherwise internally consistent.

## Source-frozen real-image replication

The independent protocol SHA is `59036905c5269d29784a6ddb70d2705557c57f8a331d0f2ab4e626ed920042e5`; publisher test release SHA `0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7`. It selects 8 different author-approved right screens, 8 author-injected original pixel changes (no later selection by detector result), 8 real same-app/different-Activity clean screens across 8 Android applications, plus 8 **script-truncated** final PNG copies of the correct real source (only the terminal 12-byte `IEND` removed, IHDR retained). Source/capture artifact hashes were frozen **before** their SWAAG tool calls. No other media or original study receipts were changed.

| Actual SWAAG tool category | Source/script expected verdict | Result | Incorrect automatic decisions |
|---|---|---:|---:|
| 8 original clean PNG reference->unchanged output | PASS exact immutable bytes | **8/8 PASS** | 0 |
| 8 real author-injected different output screenshots | FAIL exact immutable bytes | **8/8 FAIL** | 0 |
| 8 correct alternate Activity screens, wrong source identity | REVIEW | **8/8 REVIEW** | 0 |
| 8 original final PNGs truncated at 12-byte end marker | REVIEW, not a decodable screenshot | **8/8 REVIEW** | 0 |

This is **32 correct outcomes in 32 instances under these defined source-property/script labels**, not 32 human-rated semantic GUI judgments. The source-only positive/no-change outcomes remain separate from the error/abstention controls. Eight source negatives and eight truncated negatives are correlated with the existing RotBench source and cannot justify independent population error rates. For an artificial iid assumption, 8/8 has a 95% exact two-sided lower bound of **63.1%** and 0/16 unjustified decisions an upper bound of **20.6%**; the iid assumption is not defensible for real-world reliability here.

## Verified software repair and actual cost

The registered pure SWAAG `verify_immutable_snapshot` tool now requires `HostAttestedSnapshot.approved_source_sha256` and `.actual_capture_sha256`, which are checked against both persisted SHA-addressed attachment bytes. The trusted host must independently pin these values to the authoritative source manifest and captured final output, not simply echo the model's input. The tool also requires an optional **Pillow local image decoder** (`swaag[perceptual-snapshots]`), checks the final PNG `IEND` boundary, verifies PNG chunk integrity, and fully loads both images' pixels. If dependencies or data are unavailable/corrupt, it returns **REVIEW**, not PASS/FAIL. Source identity, app, orientation and viewport remain mandatory. Existing visual-model proposals never gain authority from this change.

The **complete 32-case run** through SWAAG's attachment store, registered tool, and durable history took **33.716s** on busy Nitro CPU, with no model/GPU call. Median full dispatch time: clean 0.499s, changed 0.598s, wrong Activity 0.719s, invalid final PNG 0.813s. The earlier header-only 3-case study took about 0.15-0.19s per tool call, but differed in data, source and system load, so this does **not** establish an experimentally matched per-image decoding overhead. The optional library is specified in `pyproject.toml`/`uv.lock` and installed only in the isolated research venv; production defaults remain off.

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
