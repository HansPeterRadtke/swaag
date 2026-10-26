# Final encoded owner warehouse video — actual output playback verification (26 October 2026)

**Result: PASS for the narrowly specified encoded-source identity, frame fidelity, presentation timestamps and headless Firefox HTML5 playback. Overall visual research remains PARTIAL.** Inspected and SHA-verified all 210 original 960×540 Cycles PNG files from source scene `6609f09ac0637dd232539f7d5de49ce3e59dc477a5bf25beb125ddcbf73849b6`. Verified the final approved 14-second 15-fps 210-frame warehouse MP4 by SHA `e2ef866ec79318a6b3812865a51922141ad02be66b555900da311171f49b4adb`; compared all decoded RGB video frames to corresponding original PNG. This is actual final encoded media, not just metadata/JSON. The source manifests and immutable videos were never modified.

| Measured production-adjacent property | Actual result |
|---|---|
| Total decoded final frames | 210/210 |
| PTS nonmonotonic frames | 0; maximum error 0.00000033 seconds |
| Source-aligned native decoded PSNR | min **33.28 dB**, median **35.70 dB** |
| Aligned SSIM grayscale / 240px | median **0.99660** |
| Deliberate different approved dock video against wrong warehouse frames | median **8.65 dB** |
| Wrong warehouse source shifted 15 encoded frames (1 s) | median **14.25 dB** |
| Source camera final 3-second stop, output adjacent-pixel motion proxy | early median **13.12353**, tail median **0.00071** |
| Real Firefox headless native HTML5 player | video loaded at 960×540, play() successful, currentTime **14.0 seconds**, `ended=true`, `error=null` |

Two external-to-video negative reference controls confirm that the high fidelity is not merely a generic visually plausible stream. They are valid independent videos/frames that fail the *wrong reference match*, not semantic negative examples of bad video quality. Browser test ran an isolated Firefox instance, not a production browser profile, and saved a screenshot during playback at media time 5.446 s. Both decoded contact sheet and browser screenshot remain in isolated `/data/var/perceptual-independent-20261024/final_encoded_owner_mp4_20261026/` storage. The operation to transfer the contact sheet into this chat was denied, and was not retried or routed around. No original media, publishing target, source blend, renderer, active GPU, gallery, earlier uncertain effects or other workers' state was changed.

**Acceptance boundary:** The warehouse's scene/source-specific static mesh support contact has separate prior 72/72 evidence, but the encoded movie does not independently certify mechanical contact, lighting realism or aesthetic quality. The source intentionally includes a stationary three-second camera hold; this is not a video freeze defect. The actual MP4 has no audio stream, so audio event synchronization is untestable and REVIEW. Firefox playback is a real final-output smoke test, not a human perceptual approval. The old high-resolution v52 retrieval denial was not revisited. General GUI, arrow topology, full 3D semantic completeness and unexpected natural video fault classification remain unqualified. **Overall PARTIAL.**

Reproduction: `python3 tests/research/owner_final_encoded_mp4_acceptance_20261026.py freeze|decode` then `python3 tests/research/owner_final_html5_playback_20261026.py`, then `python3 tests/research/publish_owner_final_mp4_acceptance_20261026.py`; stages refuse overwriting their preserved original receipts. Offline unit checks test the committed compact summary. Native source PNG SHA and all decoded per-frame measures remain local, not Git.
