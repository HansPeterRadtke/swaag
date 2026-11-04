# Reproducible real-speech and synthetic-noise ASR acceptance, November 4

## Purpose and scope

Voice readiness cannot be inferred from healthy STT process ports or a few
clean read-aloud recordings. A continuous assistant must distinguish real
words from environmental noise, avoid hallucinated commands in silence,
preserve user corrections, and hold consequential actions until a semantically
finished utterance is established. The governing Infra rules are
docs/guidlines/agents/voice_turn_taking_and_priority.md,
docs/guidlines/gui/voice_interfaces.md and docs/guidlines/testing.md.

The new optional script swaag.benchmark.asr_noise_acceptance accepts a private
manifest of genuine consented or lawfully licensed human speech recordings,
each with a reference transcript, exact SHA256 source bytes, unique ID,
language en or de, and full absolute local path. It validates file identity
and 16 kHz mono signed PCM16 format before the first model request, and
refuses WAVs outside bounded 1-45 second / 1.5MB limits. No raw audio,
reference labels, private credentials, WAV derivatives or test receipts are
included in Git. Inputs and logs are left in /data/var.

For each real original, the script adds seeded, deterministically scaled white
noise at preselected 20, 10 and 0 dB signal-to-noise ratios. Original speech
remains genuine but the simulated noise is artificial; the test is NOT a
substitute for real Blackview microphone, background speech, echo, wind,
vehicle, room or telephone channel recordings. The tool records real source
and derived SHA256 hashes, original RMS, clipped-sample percentage,
final transcript, reference-token Levenshtein edits, reference word count,
wall latency, and weighted WER per language/SNR. No clipping occurred in the
recorded full development batch. It additionally submits six 3-second
synthetic-noise-only negative examples over three RMS amplitudes and two
languages to detect false speech recognition.

Every run requires a NEW unique output receipt; an existing or uncertain
result cannot be overwritten. It only calls the already-running Thor resident
STT on http://127.0.0.1:15703/transcribe, with no SWAAG tool calls,
worker controls, live service restarts or voice submissions. The source tests
use an injected, local fake transcriber, test reference-word normalization,
deterministic noise, hash mismatch, duplicate IDs, oversized media, very
quiet legitimate speech, negative hallucinations, unique receipt and endpoint
allowlisting. Do not treat any test output as a production model change.

## Independent real-speech evidence

Source speech: the public Google FLEURS English (en_us) and German (de_de)
validation samples accessed via HuggingFace dataset viewer, originally
IEEE-float WAV and converted into separate hash-verified PCM16 recordings.
The data uses CC BY 4.0. Original WAVs, conversion receipts and transcripts
are retained under
/data/var/swaag-benchmarks/voice-corpus/fleurs-en-20261104/
/data/var/swaag-benchmarks/voice-corpus/fleurs-de/

The fixed development set contains eight clips and 141 official words in
English, and eight clips and 160 official words in German. Clean English
FLEURS measured 7.09% WER; German 1.25%. The original English recording
RMS amplitudes varied widely, from approximately 0.0005 to 0.089, and one
initial offline test incorrectly assumed all real speech should have
RMS above 0.002. That erroneous validation rejected a legitimate recording,
and the corrected test admits low but nonzero recorded audio. No source
recording had its gain modified in the stored baseline.

The complete original standalone seeded Python laboratory run on all sixteen
real recordings completed 48 successful ASR requests, with these
word-count-weighted error rates:

                              20 dB    10 dB     0 dB
English (eight clips)         7.09%   12.06%   26.24%
German (eight clips)          1.88%   10.63%   41.88%

The six non-speech noise trials at RMS amplitudes 0.01, 0.05 and 0.15
for both languages all produced empty final transcripts. These are six
observations on synthetic white noise, not proof that ambient speech or
other environmental sounds cannot hallucinate user commands.

The **new source-controlled reusable benchmark** was independently exercised
on four English and four German source-verified WAVs, at all three SNR levels:
24 speech transcriptions and six noise-only controls, all successful with zero
false noise-only transcripts. Its distinct seeded sequence produced WERs
10.13%, 13.92%, 24.05% in English at 20/10/0 dB and 3.33%, 7.78%, 34.44%
in German. The exact manifest, generated WAV hashes, per-case outputs and
timings remain under
/data/var/swaag-benchmarks/voice-corpus/noise-stress-20261104-v2/
as real-source-8case-manifest.json and reusable-source-8case-result.json.
The earlier independent full batch and a superseded failed RMS-validation
run are separately retained as negative evidence.

These are deliberately small, read-aloud development data, with unequal
clip lengths, source levels, accents and gender distribution. There is no
statistical claim about real microphone performance, worst-case hallucination
rates or model quality across all English/German dialects, speakers and rooms.
Results were used for evaluation and implementation checks; they cannot
serve as an untouched final outer holdout after iterative choices.

## Reproduction

Use Thor with the installed SWAAG venv, the canonical main-source PYTHONPATH,
the private 8-case development manifest, and a brand new output filename:

PYTHONPATH=/data/src/github/swaag/src /data/var/swaag/venv/bin/python -m swaag.benchmark.asr_noise_acceptance --manifest /data/var/swaag-benchmarks/voice-corpus/noise-stress-20261104-v2/real-source-8case-manifest.json --output /data/var/swaag-benchmarks/voice-corpus/noise-stress-20261104-v2/NEW-UNUSED-RECEIPT.json --snr 20 --snr 10 --snr 0

For offline validation run pytest on
tests/test_asr_noise_acceptance.py without contacting the deployed service.

The next meaningful work is a licensed, diverse, real-world audio set with
independent annotations and explicit temporal boundaries; test practical
causal AGC/VAD and partial/final ASR revision separately. Use the still-sealed
LibriSpeech holdout only for a clearly pre-registered final ASR selection check.
Do not release an automatic voice-action policy merely because these clean
or synthetic-noise WER averages meet an invented threshold.
