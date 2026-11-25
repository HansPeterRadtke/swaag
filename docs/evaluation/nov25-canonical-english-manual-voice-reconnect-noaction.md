# November 25: canonical English voice chain with forged-auto-send refusal

**Verified on the isolated latest canonical DevTests voice source, not the installed production service.** One new public English speech recording went through actual local Whisper recognition, durable manual-review draft, reconnect, forged owner identity and automatic commit refusal, explicit manual submission, actual local Coder response, actual Phonon synthesis and a normal reconnect before playback completion. The recovered PCM exactly matched the original synthesized PCM and saved WAV. No production agent or tool action occurred. Physical phone playback remains unverified.

## Source and fixed input

DevTests remote master was independently observed at `a490b00155afabd7b5d4cb28f32024f5a86e8fbf`. The seven runtime modules were extracted from that exact commit into a private Jetson source directory; no dirty checkout or deployed service was changed. Server SHA-256 `85c70555e4ba5c2730b91557d43de0571e8e3da3735224d654559ef31a35c23b`; exact seven hashes recorded in both runner and independent verifier. Baseline template SHA-256 `f6cec4a73e96521cbedc085e44ccd298d0707c630ccd44869b5dce1d9aca07de`, equal to the read-only installed configuration bytes. A new private configuration binds a loopback ephemeral port, a test-only bearer, isolated storage and an explicit one-shot model relay. Real worker-event and command routes remain disabled. Local Whisper, Coder and Phonon model services are resident, not started, stopped or reconfigured.

The new 5.4 s English recording is `en__606_102__user_turn_005` from the [LiveKit EoT Benchmark public data](https://huggingface.co/datasets/livekit/eot-bench-data/tree/ae834af9e9bddf73d1f550c190aaeb99bf31f523), CC BY 4.0 per dataset card. The original publisher word alignment was checked before the new test. No names, account information, owner recordings or instructions to perform a real action were selected. WAV SHA-256 `32319459d2f24f4c983a6986f685a125414d47bfbe436881bf449158a294dcba`, 172,800 PCM bytes. It is distinct from the earlier German and recorded-ASR-only tests. Original public shards and the provenance manifest remain in the W103 Nitro evidence root. Audio is not committed.

## Actual results

- Three authenticated WebSocket generations. Generation 1 captures and recognizes the exact public PCM, retains one unsubmitted draft. Generation 2 restores that exact draft. The client enables auto-send with a forged owner/verified-speaker claim and then sends a forged automatic commit. Both receive explicit unknown-speaker review decisions. A fresh history read still shows the same single draft and no assistant answer; the real relay has not invoked either Coder or TTS.
- The test then explicitly manually submits the recognized text. The tool-free resident Coder receives only the public transcript as quoted data, without a tools or tool-choice field. One final response from the model is spoken by resident Phonon. This is deliberately a private relay replacing the actual action-capable orchestration endpoint, not a successful production SWAAG action.
- Ordinary disconnect after the first audio frame and before end-of-stream (not a lost playback ACK). Generation 3 recovers the same turn and audio, with a new stream identity. All 112,640 recovered PCM bytes equal the original Phonon bytes and retained output WAV. A matching ordinary playback-completed acknowledgement is durably saved. Exactly one ASR, one Coder and one TTS request, no relay errors. Private child stopped; production Jetson service PID observed unchanged (3413413 before and after).
- Actual elapsed 10.513 s for the entire test including accelerated fixture upload, three connections, two guard challenges and replay. Resident ASR 1.825 s; tool-free Coder 1.921 s; Phonon 1.183 s for 3.52 s of output. These are per-call observations, not live phone turn-to-response latency or a percentile estimate.

Independent verifier checks the as-run source, canonical seven module hashes, attempt chronology, actual raw model responses, recognized text and request consistency, exactly one explicit manual submit, guard events and server log stages, operation identity, recovered stream identity, durable receipt, framed output PCM, actual input/output WAV and unchanged private isolation. It returned `verified_original_integrated_private_noaction_evidence`. Original result SHA-256 `7225cab773ba8e42b1ee9fdca569d7dde4eda81ab1554ca5f68cdd25e2b19355`. Actual run script SHA-256 `f9615cb19cacc8556273fc0e1efc182058309d48ad728c58a49300630ac57ecb`.

52 selected deterministic tests passed on Nitro, including new negative evidence and relay cases, previous integrated/manual and durable-stop/pending media evidence tests and Python 3.11 syntax compatibility. This is separate from the 111 selected natural-turn tests; overlap is not counted as additional distinct tests or proof the whole repository passed.

## Recheck, do not replay

Private Jetson evidence root: `/data/var/swaag-benchmarks/owner-serial-000103-20261125`. Original `canonical-real-integrated-once` contains attempt receipt, all three raw responses and model-call attempt records, actual input WAV, request records, framed/decoded/recovered audio, exact WebSocket event journals and durable conversation/receipt, private configuration and server log. `canonical-source` preserves the seven original Git blob copies. Independent verdict is `independent-canonical-real-verdict.json`. Nitro has `canonical-integrated-tests.xml`.

Existing evidence verification on Jetson, no inference:

```bash
D=/data/var/swaag-benchmarks/owner-serial-000103-20261125
P=/data/src/github/devtests/voice_agent_live_runtime/native_streaming_stt/.venv/bin/python
"$P" "$D/verify_canonical_english_voice_evidence.py" --evidence "$D/canonical-real-integrated-once" --as-run-script "$D/verify_canonical_english_voice_chain_once.py" --canonical-source "$D/canonical-source"
```

The original runner requires fresh exact CPU-only allowances on Jetson, Thor and Nitro, explicit one-use opt-in, fixed source/config/WAV pins and unused evidence root. It refuses duplicate upstream requests, nonlocal child model endpoints and redirects. It is retained for provenance, not a command to repeat completed inference.

## Decision

The current canonical source works for this private manual-review no-action model chain. The separately measured natural conversational endpointing candidates still failed the acceptance screen. Do not deploy automatic commitment on those scores. The live gateway's automatic classifier, owner speaker identity/anti-replay in a real environment, actual phone microphone and physical audio output, production endpoint and release acceptance remain outstanding. Do not describe the private relay's response as an actual agent-completed computer task; retain the earlier independently verified genuine multi-step read-only agent demonstration as separate evidence.
