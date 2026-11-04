# Short spoken questions, German/English clarification and voice acceptance

On November 4, 2026, the actual SWAAG foreground Coder was found to answer a
context-free frustrated utterance, `What the fuck?`, with the unrelated
self-introduction `I am SWAAG, your local voice assistant.` The model also
answered German bare expletives such as `Was zur Hölle?` in English. These
are genuine semantic and language-selection failures, not forbidden content
and not reasons to invoke tools. They violate the Infra answer-first and
user-utterance-context voice guidelines.

The source foreground_fast_system.txt now distinguishes a direct question
about SWAAG's identity from a context-free exclamation and specifies the
appropriate short clarification in the language of the utterance. The
installed foreground system template remains under its compact 2100-character
budget and continues to disclose the actual foreground Coder model only when
explicitly requested. An earlier, longer candidate fixed German clarification
but sometimes appended unsolicited model architecture to a rude German
identity question; that weaker approach was rejected in favor of a smaller
single-sentence substitution.

Five independently declared source-catalog regression cases were added:
English `What the hell?`, English `WTF?`, German `Was zur Hölle?`, German
`Was zum Teufel?`, and rude German identity `Wer bist du verdammt noch mal?`.
The existing `What the fuck?` and other user identity/problem-history cases
remain. Every case runs in an isolated durable session with no tools, no
workers and independently verified one-pass foreground responses. The
canonical spoken-conversation corpus now contains 30 distinct cases, not a
claim that 30 cases exhaust user language or represent a statistical sample.

After the minimal prompt change, **11 of 11** selected production-config
canonical live Coder cases passed, with exactly one foreground model call
per case, no workers, no tools, and approximate end-to-end text response
times of 0.4 to 1.5 seconds. The actual generated responses included:

- `What the fuck?`, `What the hell?`, `WTF?` -> `What happened?`
- `Was zur Hölle?`, `Was zum Teufel?` -> `Was ist passiert?`
- `Who the fuck are you?` -> `I am SWAAG, your local voice assistant.`
- `Wer bist du verdammt noch mal?` -> `Ich bin SWAAG, dein lokaler Sprachassistent.`
- An explicit model question -> the actual `qwen3-coder-30b-a3b-instruct-q4-k-m` foreground model, not the default Qwen3.8 background worker.

All original model answers, exact case checks, context and operation receipts
are retained privately under
/data/var/swaag-benchmarks/conversation-20261103/night5/
including multilingual-frustration-canonical-02/summary.json.

An extra predeclared set of 11 short paraphrases was tested with no tool
permission. Nine met that particular test's provisional answer-mode oracle,
including abbreviation and language variations. Two context-free questions
about a supposedly failed recording were delegated for missing evidence;
that action is not necessarily wrong, because the requested external
recording status was absent. The initial provisional scoring expected a
clarification from the foreground mode, so its two recorded failures are
not valid evidence of harmful tool execution or fabricated facts. Do not
change product behavior merely to optimize that oversimplified oracle.

## Real spoken-output integration

The deployed authenticated Jetson WSS -> Thor SWAAG -> Nitro LuxTTS ->
Jetson PCM output path was separately exercised with two new unique
harmless typed test conversations and strict language-aware checks.
`What the fuck?` produced the exact answer `What happened?`, first spoken
audio after 8.1926 seconds and final answer after 8.8926 seconds, with
24,918 received PCM bytes. `Was zur Hölle?` produced `Was ist passiert?`,
first audio after 4.131 seconds and final after 5.0794 seconds, with
32,768 PCM bytes. Each call had one original accepted durable user turn,
one matching final answer, one audio start/end and no additional tools.
Both samples went through the existing Nitro LuxTTS service as signed
16 kHz mono PCM16 audio; there was no real physical speaker or phone.

The exact server-written answer WAVs were independently SHA256-verified
and copied to the private Thor evidence directory. They measured 0.779
seconds (English) and 1.024 seconds (German). Thor's resident large-v3
speech recognizer was separately run on each produced waveform with
both the explicit language setting and automatic language detection.
All four transcripts contained the exact corresponding words with zero
word errors: English `what happened`, German `was ist passiert`.
This is a useful independent intelligibility and signal-integrity
cross-check, not a subjective voice-naturalness/phonetic-accent measurement.
The source waveforms and transcript/latency receipts remain outside Git:
/data/var/swaag-benchmarks/conversation-20261103/night5/voice_clarification_tts_stt_loopback.json.

Known gaps remain: the older contextual-reference case `What the fuck
happened to it?` after an upload-failure history still sometimes delegates
rather than resolving the unique reported referent; this specific failure
was not closed by the context-free expletive correction. Thor's dedicated
semantic voice-turn classifier remains unavailable and Jetson still
reports agent=false. G3 continues to detect the Blackview through USB but
ADB reports it unauthorized. No test was claimed to be Android microphone
capture, physical audio playback, a browser Stop control test, or a
healthy-classifier natural barge-in.

Final acceptance also requires full local SWAAG unit/regression coverage
before a source change can be committed to main. No running production
services were restarted for these isolated tests; the currently loaded
production prompt and the canonical source file are tested independently.

## Complete repository regression and source integrity

The exact current source state, including the 30-case canonical conversation
catalog, passed the entire SWAAG test suite on November 4:
`1760 passed, 3 skipped in 354.35s`, return code zero. A separate targeted
catalog/foreground suite passed 57 tests, and an independent real-model
canonical run passed all 11 short-utterance/language cases, each with one
model call and no tools or workers. The exact full pytest log remains under
`/data/var/swaag-benchmarks/conversation-20261103/night5/full-pytest-after-multilingual-prompt.log`.

Only the prompt, five added conversation scenarios, their independent unit
oracles and this evidence report belong to the corresponding SWAAG main
commit. The unrelated unfinished semantic turn-classifier implementation
and tests are owned by a separate concurrent agent and must not be staged,
reformatted or overwritten. No running models, Android device or voice
WebSocket service needed to be restarted for this source validation.
