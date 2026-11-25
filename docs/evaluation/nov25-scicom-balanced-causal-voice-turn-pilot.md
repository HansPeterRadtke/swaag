# November 25: Scicom Whisper acoustic turn-completion comparison

**Decision: no-go.** None of the three tested CPU ONNX graphs meets the
predeclared joint reliability and compute-time screen in English and German.
A new independent 160-recording pilot was completed after the original full
attempt was deliberately stopped. This is a narrow diagnostic sample, not
a representative holdout or the publisher's official leaderboard score.
The publisher already used the public benchmark corpus; German is exploratory
and is not a claim of publisher support. No live service was changed.

## Fixed inputs and execution

SWAAG baseline `9b5b7989d5b5be45a9a8a9a839a110fa3d52c4ce`.
The public source is the previously pinned [LiveKit EoT Benchmark](https://huggingface.co/datasets/livekit/eot-bench-data/tree/ae834af9e9bddf73d1f550c190aaeb99bf31f523),
CC BY 4.0. The 800-row English/German general-cohort source manifest from
the November 25 prior evaluation was retained and reused exactly.
For each of its 1,730 eligible pauses, define only the first complete
native 160 ms chunk boundary at least 200 ms after pause onset and no later
than its end. Scicom receives only samples at or before that point, with
an eight-second trailing window left-padded with zeros as necessary.
Feature extraction is the documented Whisper 80-mel, 16 kHz configuration
without normalization. No actual live VAD or ASR was tested.

Exact anonymous public model revisions and INT8 ONNX weights:

- [Tiny](https://huggingface.co/Scicom-intl/semantic-vad-eot-whisper-tiny/tree/86dddafbb85ee3ca19c6efc74ed6f69b27fd1d9c):
  `86dddafbb85ee3ca19c6efc74ed6f69b27fd1d9c`, 10,118,687 bytes,
  SHA-256 `7312bd478f3a32fa272c056718cb454048fc82df6dce77f02bead40849bbae9d`.
- [Base](https://huggingface.co/Scicom-intl/semantic-vad-eot-whisper-base/tree/b16033db5f46856684572f668012e721dd199e16):
  `b16033db5f46856684572f668012e721dd199e16`, 23,776,923 bytes,
  SHA-256 `d0ef84bbcb490b154295fd494819af82c562c3bf1c45036604f9e3e7cf0232b0`.
- [Small](https://huggingface.co/Scicom-intl/semantic-vad-eot-whisper-small/tree/bed671b2981e69ba7f019a32f3ba1ab212f3be7a):
  `bed671b2981e69ba7f019a32f3ba1ab212f3be7a`, 94,937,548 bytes,
  SHA-256 `460d9367967a079ccbb9ae19683ea1b7b5998735fa7d02d6f58bcee35aef9d45`.

Each model card declares Apache-2.0; original cards, preprocessing
configuration, export reports, source download manifest and model bytes
remain in private evidence. No weights or source recordings enter Git.
The public data input manifest is SHA-256
`6f6decf1532c30ef1b04e38526de743afcecdee8b98b589e5568ae0729456cdd`.

### Original full attempt (incomplete, never replay)

One-thread CPU inference was started once. After 22 complete pause rows,
observed median model-only compute times were about 193 ms, 455 ms and
1,867 ms for tiny/base/small. The projected full run exceeded the admitted
CPU lease window. The exact owned process was stopped. Its original
attempt marker and raw output remain preserved without a completed
result. Eight affected recordings, including the next uncertain input,
were excluded from the subsequent experiment. No completed or possibly
in-flight input was replayed.

### Completed separate balanced pilot

The new sample was determined by SHA-256 rank and a fixed seed, excluding
the eight earlier recordings. For each language, choose 40 hold and
40 end-of-turn cases, one eligible pause per unique recording. Interleave
all 160 cases by a second deterministic hash. The selection was frozen
before inference, without using the earlier model scores. This is a
diagnostic balanced sample, not a statistical representation of all
conversations. The scoring point differs from the earlier W103
streaming model's any-eligible-point metric, so compare only with caution.

ONNX Runtime 1.28.0, CPUExecutionProvider, two intra-op threads and one
inter-op thread, sequential execution. Each of the three graphs was
called once per selected input, 480 scored predictions total, plus a
separate non-oracle warm-up per graph. No remote inference or agent
execution. The actual elapsed time of the complete pilot was 302.60 s.

Fixed score thresholds were 0.5 (primary), 0.55, 0.8 and 0.95. The
primary research gate requires zero false holds, at least 95% detection
of selected ends in each language and p95 preprocessing plus inference
of at most 250 ms.

| Model | EN ends / 40 | EN false holds / 40 | DE ends / 40 | DE false holds / 40 | EN/DE p95 compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | 39 | 10 | 37 | 16 | 349 / 313 ms |
| Base | 37 | 7 | 35 | 12 | 572 / 623 ms |
| Small | 38 | 6 | 38 | 14 | 1,747 / 2,091 ms |

At threshold 0.95 every model has zero false holds on this sample, but
detects very few true ends: tiny 3 EN and 2 DE; base 7 EN and 2 DE;
small 4 EN and 3 DE. At threshold 0.8 all models still make false
hold decisions. Neither higher capacity nor the tested stricter
thresholds solve both premature cutoff and missed ends. Even these
compute times omit live capture, recognition, model scheduling and
response delivery.

## Evidence and recheck

Nitro private root:
`/data/var/swaag-benchmarks/owner-serial-000104-20261125`.
Retain the download scripts and receipts, model pins/metadata, input
manifest, balanced selection, original stopped attempt and completed
pilot. The original W103 Parquet shards remain in the prior evidence
root. No private user recording was used.

The independent verification script reconstructs all causal points from
the earlier verified public manifest, reproduces the deterministic
selection without inference, checks the excluded interrupted rows,
selected waveform hashes, all model bytes and as-run source fingerprints,
validates all 480 score and timing records and recomputes the result.
It returned `independently_verified_scicom_balanced_pilot` with no model
meeting the gate in both languages. It does not independently rerun
inference or prove generalization.

Exact completed evidence:

- `balanced-selection.json` SHA-256 `e6e191b02d1e035b78a28ef942b8acff786ece1250779cb132765493d453f8b7`.
- `scicom-balanced-distinct-once/attempt.json` SHA-256 `e4ca285ef2a42b041afabeb6234a36d2c1e05cb8d7a14fc69b4583c995672b4c`.
- `scicom-balanced-distinct-once/decisions.jsonl` SHA-256 `a474737586c2cce6aa7ecfa6be0b6ddbf4786f905760236a21e461976965cdba`.
- `scicom-balanced-distinct-once/result.json` SHA-256 `d27f5270b4dfbe7d588d5659b4f2721c7bf9c8203f6a99b38fa97adf1429b9af`.
- `full-attempt-stop-receipt.json` SHA-256 `7ec17f5e12dd255d5898a122b31b3cd63b0e21a4a3f1e35bc2755846b7f8e0f9`.
- `w106-independent-scicom-verdict.json` and
  `w106-scicom-evidence-tests.log` (four focused deterministic tests passed).
  All four Scicom scripts parsed under the Python 3.11 grammar.
  Not a claim that the full repository test suite passed.

Read-only verification of original receipts (no inference):

```bash
D=/data/var/swaag-benchmarks/owner-serial-000104-20261125
python3 scripts/demos/verify_scicom_balanced_pilot_evidence.py --root "$D" \
  --old /data/var/swaag-benchmarks/owner-serial-000103-20261125 \
  --scripts scripts/demos
```

The evidence verifier needs the selected derived WAV copies, retained
for this exact audit. They are reproducible from the original public
shards, not owner audio. The run scripts are immutable as-executed
sources, not instructions to repeat these completed or uncertain
experiments. A fresh experiment requires new input identity, scope,
resource admission, and evidence root.

No actual speaker verification, liveness, barge-in, physical phone,
automatic command authority or production deployment was established.
Maintain the guarded manual-review path until stronger independent
endpointing evidence exists.
