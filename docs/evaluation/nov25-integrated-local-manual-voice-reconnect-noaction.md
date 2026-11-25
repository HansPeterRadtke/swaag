# Integrated private manual speech-to-response and reconnect test — November 25, 2026

**Observed result:** a single isolated, source-pinned SWAAG voice WebSocket process
received the exact nine-second public German FLEURS recording, restored the
unsubmitted speech draft after reconnect, accepted one explicit test-client manual
confirmation, obtained a fresh answer from the actual resident Qwen Coder model,
synthesized the exact reply with the actual resident Phonon Easton service, and
recovered the complete generated audio after disconnecting during the initial
media delivery. All three model requests completed exactly once. The final
ordinary playback acknowledgement was durably recorded. This is a
**private test, not an installed production feature**.

The previous November 24 separately linked ASR, text and speech component
receipts remain unchanged. This test uses one new continuous private session.
No actual SWAAG agent operation, phone microphone, physical speaker, external
model provider, account, worker control, website change or tool action occurred.
The dedicated semantic classifier is still unavailable; neither its accuracy nor
speaker identity is established. The intentionally disconnected worker-event
channel emitted expected stream-error events. Manual confirmation is only
performed by the local test client; it is not a biometric authorization.

## Exact scope and observed timings

The deployed voice process remained active with its original PID 3413413.
The private child used the existing seven-file guarded W71 candidate,
server SHA-256 `873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf`.
The exact German public fixture is
`fleurs_de_validation_002.wav` (16 kHz mono PCM16, 288,000 audio bytes,
SHA-256 `ad33bba729df7624e8e0c97ffb6a1f06c81af96f59a202b2699f7953343cf4fd`),
previously documented in the November 4 multilingual and public-source
speech research. The public source WAV was unchanged and not committed.
The actual model text acknowledges/paraphrases the recorded subject; the
recorded factual statement was not independently fact-checked in this test.

| Observed stage | Seconds |
| --- | ---: |
| Resident faster-whisper-large-v3 HTTP recognition | 2.644 |
| Resident Qwen3-Coder-30B-A3B Q4 tool-free reply | 2.701 |
| Resident Phonon Easton framed PCM generation | 1.749 |
| Whole private process scenario | 14.867 |
| Generated reply media duration | 5.44 |

The test relayed the actual model response through the native voice SSE parser
using an isolated, deliberately tool-free HTTP adaptor. That adaptor buffered
the whole Coder result and the real framed Phonon response before delivering
them to the native voice server. These figures are individual observed wall
times, not conversational latency distributions or proof of native production
streaming-model responsiveness. The input was uploaded faster than real time
through the manual-recording protocol. The receiving socket disconnected
after initial media bytes, before audio-end; it then received the entire
stored reply from the beginning on a new stream ID. No physical playback
position, partial-media resume offset or lost-ACK recovery was tested.

The final receipt has exactly one speech turn with the original recognized and
submitted text, one assistant answer, `requires_review=true`,
`status=completed`, and matching final stream identity with
`audio_transmitted=true` and `audio_completed=true`. The actual 174,080
recovered PCM bytes match both the original synthesis bytes and the stored
reply WAV. The two reconnects did not duplicate the ASR, Coder or TTS requests.
This was a normal playback acknowledgement, **not** the previously denied
physical lost-playback-ACK experiment.

## Reproduction and read-only independent audit

The as-executed source is
`scripts/demos/verify_integrated_voice_chain_manual_once.py`, SHA-256
`e32371ecebe9133456f4468c167648e0b15562c705b4256abf45a3538d027ff4`,
written against SWAAG remote main
`0cf3736d3933909e204dd49df6fb0a57d16493f1`.
Its default invocation only checks exact file fingerprints. The original
one-use opt-in invocation admitted three short CPU/RAM resource leases
(Jetson, Nitro, Thor); all were explicitly released. The test runner is
retained exactly as executed, **not** a general-purpose service or an
authorization to re-run the fixed experiment. A separate new experiment
requires fresh approval, source admission, immutable unique evidence and
independent resource checks. The original test must not be replayed.

On Jetson the evidence is retained under
`/data/var/swaag-benchmarks/owner-serial-000099-20261125/`.
The `real-integrated-once` directory contains original attempt markers,
raw local ASR/model receipts, model prompt, framed TTS and recovered PCM,
server log, WebSocket event journal, exact saved WAV files, and the private
durable dialogue. No raw audio, model replies or private runtime config
is stored in Git. The original result SHA-256 is
`d778c12e7e7f75961b1dfaa39773b995222d9fbcc88d1846d222e0b265a149b4`.

The independent, offline source verifier
`scripts/demos/verify_integrated_manual_voice_evidence.py` checks the
as-run source and original evidence fingerprints; exact speech, answer,
media and turn identity; order of all three connections, one manual submit,
two speech-final events and original/recovered media; and the saved normal
acknowledgement. It returned
`verified_original_integrated_private_noaction_evidence`. It performs no
new model or network call. The separately preserved original verdict is
`independent-verification.json` beside the private run.
To verify existing evidence (no replay):

```
python3 scripts/demos/verify_integrated_manual_voice_evidence.py \
  --evidence /data/var/swaag-benchmarks/owner-serial-000099-20261125/real-integrated-once \
  --as-run-script /data/var/swaag-benchmarks/owner-serial-000099-20261125/verify_integrated_voice_chain_manual_once.py
```

The new synthetic tests exercise strict PCM framing, manual-only relay
admission, duplicate-attempt refusal and independent event/evidence
tamper rejection. They are not a substitute for the real run.
The separate November 23 first-pass read-only multi-tool demonstration
remains the evidence for actual agent tools; this test does not replace
it. The live automatic voice gateway, physical phone, actual authenticated
user conversation and speaker-replay protection remain unqualified.
