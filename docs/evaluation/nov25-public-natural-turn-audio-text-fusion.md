# November 25: actual public conversational audio and aligned-text turn detection

**Decision: no-go for automatic turn completion or command commitment.** The new lightweight acoustic graph is computationally fast on Nitro's CPU, but fails the predeclared reliability screen in both English and German. Combining it with Namo's text classifier using ideal word-aligned text does not close the gap. No runtime configuration, live microphone, speaker authorization, agent action, or deployed classifier was changed.

## Fixed inputs and provenance

Baseline SWAAG main: `6b7cfcbbf9cf9d48a8594b47cae19a8d53c62d0a`. Inputs frozen before inference:

- [LiveKit EoT Benchmark data](https://huggingface.co/datasets/livekit/eot-bench-data/tree/ae834af9e9bddf73d1f550c190aaeb99bf31f523), CC BY 4.0 per publisher dataset card. Use the 400 English general rows in the first English shard and 400 German general rows, not the newer accented/short-turn cohorts. The original dataset supplies 16 kHz mono WAV, annotated hold/eot silence intervals and word timings. This is a bounded public conversational recording corpus, not the owner's audio. No source audio or raw transcripts are published in Git.
- [p99lab turn-1-mini](https://huggingface.co/p99lab/turn-1-mini/tree/d548240397d5449901559eb09f87cdfcad70420e), Apache-2.0 with accompanying NOTICE retained. Exact FP32 `onnx/turn-1-mini.step.onnx`, 21,825,125 bytes, SHA-256 `e747b182d685c7db489a3ea2e12bdcd65bac506fcddec00ed3cd0bf9f209fd11`. This is the v1 single-channel EOT graph documented in the v2 repository; not the v2 multichannel interruption model. German is exploratory, not publisher-supported evidence.
- Namo turn detector, exact previously verified local model revision `59245aac1f0a0f170277ebecc32ddaea418083b4`, model SHA-256 `67be31dac7c49abd43a277a105222c9031a0e22ac37f0e22dab395f68b318fc9`; same immutable model/tokenizer files as the November 24 multilingual CPU comparison. The inputs here are entirely new, aligned prefixes, not a replay of the earlier synthetic oracle.
- Original German Parquet SHA-256 `82c6d47d218107023fecc175c92535b39c8f6d69f5d77715e9e6b27c1b2364e9`; first English shard SHA-256 `52bf7ff71e83d0214d744693839f8a6b14e78e0f26db20dfdb7f589453a50103`. Both original shards and license/source metadata are retained.
- Acoustic corpus manifest SHA-256 `f4e217a4c14287c3b1d4eaa8c872a2273d5a37ff7da38c9d114d639ed804892e`; aligned-text manifest SHA-256 `c25e8871ab61930edbad6a385b2cef21096077f994e56088604d7d4e38ca5169`.

## Actual experiment

Run each graph in isolated ONNX Runtime CPUExecutionProvider, one intra-op and one inter-op thread, sequential execution. No GPU, remote inference or agent actions. Acoustic state starts at zero for each row; only complete native 160 ms chunks are passed, state outputs returned unchanged. Use the last score at each native step boundary, at least 200 ms after the annotated pause starts and no later than its end. No padding, future audio, or earlier score from a chunk containing future audio. Evaluate all eligible hold intervals separately; count recall over all true ends, including ineligible ones. There are 1,557 hold intervals and 800 ends, with 937 eligible holds and 793 eligible ends. This local metric is deliberately not the official benchmark leaderboard metric.

Before inference the primary acoustic threshold was fixed at 0.36, with a narrow research screen of zero premature positives on eligible hold spans, at least 95% end recall in each language and 95th-percentile compute time at most 25 ms per native step. Additional fixed thresholds: 0.06, 0.16, 0.50, 0.80 and 0.95.

For the complementary text test, construct the contiguous prefix from publisher word alignments ending by the pause start, stopping at the first invalid or unavailable word. No prior assistant text, later words, or live ASR. 1,730 eligible spans; 1,703 scored, 27 lacking a prefix. Missing scores cannot authorize a conjunction. Text thresholds 0.5 and 0.8 and conjunction thresholds (0.36, 0.5), (0.36, 0.8), (0.8, 0.8) were recorded before aligned-prefix inference. Their latency is optimistic because speech recognition is not performed.

Actual p99 model: 800 rows, 58,059 native CPU steps, median 6.74 ms, p95 13.13 ms, maximum 90.82 ms, elapsed 464.61 s, ONNX Runtime 1.28.0, NumPy 2.4.6. Actual Namo run: 1,703 scored prefixes, median 119.23 ms, p95 507.72 ms, maximum 1,188.02 ms, elapsed 302.92 s. The two jobs overlapped on separate brokered CPU allowances; these are observed compute timings, not microphone-to-response latency.

| Fixed policy | English true ends / 400 | English premature holds / 528 | German true ends / 400 | German premature holds / 409 |
|---|---:|---:|---:|---:|
| Acoustic 0.36 | 347 | 30 | 253 | 39 |
| Acoustic 0.8 | 119 | 1 | 32 | 0 |
| Text 0.5 | 254 | 254 | 234 | 266 |
| Text 0.8 | 70 | 80 | 35 | 67 |
| AND 0.36 / 0.5 | 218 | 14 | 152 | 23 |
| AND 0.36 / 0.8 | 61 | 6 | 27 | 6 |
| AND 0.8 / 0.8 | 23 | 0 | 5 | 0 |

At the primary threshold English end recall is 86.75%, German 63.25%. False hold proportions among eligible intervals are 5.68% and 9.54%, respectively; 26 English and 36 German rows have at least one premature positive. The median detected-end delay is 0.36 s and 0.34 s, excluding actual ASR/response time and missed ends. None of the fixed policies meets the joint zero-false / 95% recall screen. The strict conjunction detects only 28 of 800 ends.

A separate descriptive, **in-sample optimistic upper bound**, not calibration or independent acceptance: search every relevant observed hold-score breakpoint for a static threshold or conjunctive pair with zero observed premature holds. Across the 800 rows, the best single acoustic threshold detects 108 ends (13.5%), text threshold 4 (0.5%), conjunction 126 (15.75%). Even fitting thresholds separately per language cannot reach 95% recall: best English conjunction 95/400; German 77/400. These thresholds must not be deployed: they were chosen using the same labels, and do not establish generalization or safety.

## Independent checks and reproducibility

The original experiment is immutable; never repeat it just to verify. The independent source verifier reconstructs all 800 WAV hashes, all 2,357 labeled spans, all 1,730 causal prefixes from the pinned original Parquet bytes and independently checked the extracted copies. The independent model-output verifiers check the exact executed script hashes, every row, complete step counts, causal endpoints, threshold decisions, raw result hashes and recomputed metrics. A separate diagnostic computes the optimistic threshold frontier with a small brute-force equivalence test.

On Nitro, evidence root `/data/var/swaag-benchmarks/owner-serial-000103-20261125`:

- `download-attempt.json`, `download-result.json`, `download_public_pins.py`, original Parquet shards and model license files.
- `corpus/manifest.json`, `causal-text-manifest.json`, original public-source verification `independent-public-source-verdict.json`.
- `p99-fp32-single-channel-800/{attempt.json,rows.jsonl,result.json}`, result SHA-256 `b0a87b3f94decc3b98c5c3b3350c613d84b18052e6347837f45722b10fccee56`, rows SHA-256 `a4e666d6ee23941eb576f7045922c774db8277787195bee696bdcd808bd5ea2d`.
- `namo-aligned-text-once/{attempt.json,cases.jsonl,result.json}`, result SHA-256 `2128f5ad3edef421825aa76c3cc77c13ef4fe74cc12aa0b3d898ed5a02cfd159`, cases SHA-256 `5cc5c3d16b2a6b9d1788f399555ae37146a5d68eb653dbbf67fa31d8bcb16589`.
- `independent-p99-verdict.json`, `independent-fusion-verdict.json`, `in-sample-zero-false-frontier.json`; latter SHA-256 `fa3c73e152cc62837aa6cb3c66ca88bb8f228b2ca06227208f4cd8546e39ada5`.
- `final-natural-turn-tests.xml`: 111 selected regression tests passed, including Python 3.11 syntax compatibility, provenance, causal boundaries, fusion, previous classifier acceptance, voice operation and speaker similarity checks. Not a claim that the entire repository passed.

Recheck existing evidence from a clean checkout of this source:

```bash
D=/data/var/swaag-benchmarks/owner-serial-000103-20261125
P=/data/venv/bin/python
"$P" scripts/demos/verify_public_natural_turn_source.py --root "$D"
"$P" scripts/demos/verify_p99_natural_turn_evidence.py --root "$D" --run-evidence "$D/p99-fp32-single-channel-800" --as-run-script scripts/demos/run_p99_natural_turn_cpu_once.py
"$P" scripts/demos/verify_natural_turn_fusion_evidence.py --root "$D" --audio-run "$D/p99-fp32-single-channel-800" --text-run "$D/namo-aligned-text-once" --audio-script scripts/demos/run_p99_natural_turn_cpu_once.py --text-script scripts/demos/run_namo_natural_turn_text_once.py
"$P" scripts/demos/diagnose_natural_turn_zero_false_frontier.py --root "$D" --audio-run "$D/p99-fp32-single-channel-800" --text-run "$D/namo-aligned-text-once" --audio-script scripts/demos/run_p99_natural_turn_cpu_once.py --text-script scripts/demos/run_namo_natural_turn_text_once.py
```

The original runners are guarded by an exact fresh broker lease, an explicit one-use opt-in, hashes and an unused private output root. They are not verification commands. Corpus preparation uses the original shards to recreate the WAVs; the independent verifier does not need extracted copies unless explicitly requested.

## Remaining limitations and direction

These are public recorded general-cohort pauses, not a physical phone test, not verified owner speaker identification, not live ASR, not actual barging/overlap and not official leaderboard evaluation. Natural turn completion cannot establish authority to commit a command. Keep manual review and current speaker/intent safeguards; retain all failed candidate evidence. Do not reactivate the previously denied GPU operation. Investigate a genuinely stronger bilingual context-aware turn detector and independently verify interruption and owner-speaker behavior. The November 25 real manual-voice/reconnect and durable stop evidence remains valid for its tested source, not evidence that the latest canonical source is installed in production.
