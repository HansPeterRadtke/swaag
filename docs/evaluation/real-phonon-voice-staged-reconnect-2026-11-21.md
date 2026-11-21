# Verified existing SWAAG answer -> Phonon TTS -> original recorded speech / four-socket reconnect

**Date:** November 21, 2026. **Status:** genuine, checksum-grounded and independently rerun *staged* no-action voice integration, **not** a continuous production assistant turn, human speaker verification, or a physical Android release.

## Fixed evidence, source and effect boundaries

An actual prior canonical SWAAG foreground turn was verified independently on Nitro, using a deliberately empty tool registry, no stateful/side-effect permissions, and a durable completed answer. That source event journal has SHA-256 `1e7fd9076c7c0364d32494dbbfdd95d280ac9071e11daf5c59e268e99b28aaa4`; the answer text has SHA-256 `78c4dd833853f3b881806d7ec76dd572ffa153bcd09bdc72fe7334da7731dbab`. An exact private source-proof manifest, derived from that real journal, is supplied to the acceptance script. The script checks both exact hashes and the no-tools/completed/not-a-fresh-turn flags. The previous SWAAG model call is **not re-run** by this experiment. No SWAAG user/worker operation is authorized by that old answer.

The unchanged original **real** public-domain source recording on Jetson has SHA-256 `2b29bead44440ee10117f8dc6f2f20be552aa2ffbb39b36f10c854dc165376ee`. The already tested Jetson/devtests `voice_agent_live_runtime` implementation and no-action WebSocket acceptance are pinned to worktree commit `b868ed1a4e9f07ca696d03c54e9bb934bf2a8750`; the script verifies the exact source SHA-256 of its voice server and two relevant test modules **before** running. The matching sources must exist locally on Jetson and must not be modified by this acceptance. These dependencies are not silently replaced with unverified current master or the older production voice service.

The script calls only `http://127.0.0.1:15305/synthesize` on Jetson, which is the already-running authenticated private SSH forward to Nitro's warm CPU Phonon 31M Easton voice, not a network-public TTS listener. It rejects every other destination, checks the HTTP media type and exact bounded sixteen-kilohertz mono PCM16 WAV properties, and refuses empty or truncated results. The script does not turn on microphones, connect to a physical phone, restart a service, call an external provider, or interact with any live SWAAG agent tool. The new generated WAV and logs remain private with restricted file permissions.

## Reproduced actual test

The independently reviewed prior canonical SWAAG answer was converted to a five-point-three-six-second WAV, with actual real Phonon synthesis request duration of **two-point-five-nine seconds** in the first local run. The original on-disk complete-PCM format is mono, sixteen kilohertz, sixteen-bit PCM. The WAV checksum was `59ae90f298e95f42662bf75e4bef0640e7ea91261dddd29ecf358eff4115352a`.

That **genuinely synthesized** WAV was supplied to the existing isolated Jetson test's intercepted **non-action** assistant response. The user input remained the actual original WAV, processed by resident Thor Whisper, yielding one manually reviewable draft, not an automatic command. Explicit test-only local confirmation was followed by four genuine authenticated localhost WebSocket generations. Audio partially delivered to the disconnected second socket, then the full exact generated PCM was recovered on the third. The normal playback completion acknowledgment was persisted, and the fourth socket correctly did not replay the response. The first manual test finished in **nine-point-zero-four seconds**. It included exactly one real Whisper call, the same original user WAV checksum, exact generated reply WAV checksum, one durable reviewed turn, and zero real SWAAG action/TTS calls *inside the voice fixture*. This last metric does **not** deny the one genuine Phonon synthesis performed **before** the fixture.

The new owning SWAAG script `scripts/demos/verify_real_voice_phonon_reconnect.py` reproduces the same staged experiment with explicit source/endpoint/provenance checks and private outputs, and includes `tests/test_real_voice_phonon_reconnect.py` with **twenty-seven passing** no-action source-validation, missing-authority, malformed-WAV, endpoint-denial, HTTP-redirect rejection, wrong installed-engine identity and opt-in regressions. The script disables HTTP proxies and redirects, accepts only the existing Jetson loopback port, checks the installed warm service identity and uses mode-six-hundred private outputs. Its first pinned-source live rerun finished successfully in **thirteen wall seconds**, including **two-point-nine-nine seconds** of actual Phonon synthesis and four valid WebSocket generations, with exact replay and a normal ACK. The improved guarded-source live rerun passed in **ten wall seconds**, including **one-point-nine-eight seconds** of actual synthesis and four generations, with exact replay, durable ACK and private result mode 0600. The last generated WAV had a different checksum from the first: independent TTS generations need **not** produce identical PCM, but every replay must exactly match the WAV generated during **that** run. The initial shell invocation failed *before any execution* because `/usr/bin/time` was absent on Jetson; rerunning the exact new test via `timeout` with Bash timing succeeded. This was a launch dependency error, **not** a SWAAG/TTS failure.

## Reproduction and acceptance guards

After checking the project exchange and Raspi resource broker, first prepare one **new**, unused private evidence folder parent with mode 0700, and provide the private Nitro-derived proof JSON. From a clean, pinned SWAAG repository source checkout on Jetson, execute the script **without** `--run` first, supplying `--reply-proof` (the private proof), `--voice-source` (the exact Jetson `voice_agent_live_runtime` directory), and `--evidence` (a previously unused output subdirectory). This checks all source hashes and replies `source_preflight_only`, with **no model call, no output directory, no TTS and no voice activity**. A real run additionally requires **both** `--run` and environment variable `SWAAG_PHONON_VOICE_NOACTION=1`. If any source changes or any claim cannot be verified, it fails closed. Failure evidence is retained in the private output directory; never treat a plausible model text or existing fixture file as a successful live reproduction.

The stable private first-run WAV and logs are on Jetson under `/data/var/swaag-benchmarks/real-canonical-reply-phonon-playback-20261121/`. The independently rerun initial script evidence is the **new subfolder** `second-live-run-20261121`, containing its own generated WAV, `voice-test.log` and `result.json`, with top-level private `verified-source-proof.json` and `pinned-version-real-run-console.log`. The separately rerun guarded script uses `third-live-run-20261121`, top-level `versioned-acceptance-script-guarded.py` and `guarded-live-console.log`; compare the exact source script hash to the eventual published canonical SWAAG revision. The script is only versioned after committing/pushing and independently verifying the remote source: local test artifacts do not constitute publication. Original SWAAG session/source proof remains on Nitro in the existing restricted `canonical-voice-noaction` benchmark folder. These outputs are *not* published to source control or the Multiverse website. Reproducer must check actual source revisions and the service health again at execution; the previous observation is not a guarantee of future availability.

## What is still not proven

This staged integration does **not** use a fresh canonical SWAAG orchestrator response generated and routed directly from the same WebSocket turn. The real user-recorded audio becomes one **manually reviewed test-only** draft, while the genuine canonical reply was obtained **from a distinct earlier, independently verified no-tools turn**. The **first** localhost voice fixture intercepts the assistant action, uses the previously generated TTS WAV and replays it through real WebSockets. The **second** test below exercises `VoiceSession.answer` and real live `stream_tts_reply` using the previously verified no-tools canonical reply through a test-only intercepted external request. Neither proves **fresh same-turn** live SWAAG-orchestrator-to-Phonon streaming, on-the-fly cancellation during TTS, owner speaker authentication, semantic turn completion, actual phone microphone/playback, physically lost ACK recovery, percentile latency, or release readiness. No denied lost-ACK test was rerun, and no production services were changed. A next justified acceptance increment would bind a *fresh* isolated verified canonical response to the **same explicitly reviewed** WebSocket turn and stream newly synthesized audio without giving it command authority.

## Historical source-verifier compatibility fixed during this change

The new SWAAG unit run also revealed that the older *agent source-audit*
verifier had counted a **newly added** current test instead of the exact
historical test source. That was unrelated to the real Phonon/reconnect
behavior and did not invalidate the original stored event journal. The
evaluator now reads the original SHA-256-verified Git blobs, never the
working tree, and the unmodified actual journal passed its twelve grounded
checks again. The separate regression suite confirms rejection of the wrong
source revision; see `real-readonly-agent-demonstration-2026-11-20.md`.

## New actual streaming TTS / resumed playback acceptance

A second, distinct, source-pinned owner-project script,
`scripts/demos/verify_real_voice_phonon_streaming_reconnect.py`, runs the
**actual Jetson `VoiceSession.answer()` and `stream_tts_reply()` methods** on
an authenticated, isolated localhost WebSocket. The original recorded audio
passes through real resident Thor Whisper and remains an editable **manual
review** draft. After explicit **test-only** manual confirmation, the script
intercepts only the *external* `receive_voice_run` call and supplies the
previously verified, exact canonical no-tools reply. It makes **no new SWAAG
model/worker/tool request**; the local answer stub merely permits the real
production-path **Phonon service-streaming code** to run against the already
running Nitro CPU Easton voice through Jetson's existing loopback tunnel.

The first exploratory actual test succeeded: after streaming the initial
**one thousand nine hundred twenty bytes**, the second WebSocket was closed.
`VoiceSession.answer` finished the real streaming synthesis and persisted
**one hundred seventy-one thousand five hundred twenty PCM bytes** in its
private WAV. The third connection resumed **exactly those** generated bytes,
recorded a normal playback-completed ACK and persisted completion. A fourth
connection did not replay. The full test ran about **twelve-point-five seconds**,
including a measured **four-point-six seconds** of live TTS streaming.

The new **versioned** script was copied byte-for-byte from its SWAAG source
into private Jetson test storage and independently executed after a source
preflight that performed **no real model call**. A companion stage-script
source file is required for immutable source-manifest checks; both are located
in the same `scripts/demos` directory in an actual SWAAG checkout. The
versioned script's real-run evidence is under the private Jetson
`fourth-streamed-versioned-20261121` folder. That separate run also **passed**:
actual resident Whisper once, actual Phonon stream around **one-point-eight
seconds**, one thousand nine hundred twenty bytes sent before disconnection,
one hundred seventy-one thousand five hundred twenty bytes retained, exact
recovery, normal ACK, no fourth-generation replay and **zero SWAAG actions**.
Its full test took about **ten-point-three seconds**. The two newly synthesized
WAV hashes differ; PCM equality is asserted **within each resumed stream**,
not between independent model generations.

For a source-only preflight, invoke the streaming script on Jetson with
`--reply-proof` pointing to the restricted source-proof JSON, `--voice-source`
pointing to the pinned `voice_agent_live_runtime` directory, and `--evidence`
pointing to an unused output subfolder, **without** `--run`. Actual TTS and ASR
execution needs **both** `--run` and
`SWAAG_STREAMED_PHONON_NOACTION=1` after a fresh resource-broker/service check.
Use only the pinned source checkout, not the dirty devtests master or the
separately installed older production WebSocket source. As with the staged
script, do not retry a non-idempotent external action or physical lost-ACK
scenario. The private source/proof and real WAVs remain **outside Git**.

This closes the previously untested isolated **live TTS streaming and
reconnect** subpath. It still does **not** prove that a **fresh** canonical
SWAAG reply is generated from the **same** reviewed turn, that the server
identifies the true owner, or that the released Android app plays the full
answer on a physical phone. The original owner speaker/semantic-turn and
production deployment gates remain open.

## November 21: fresh same-turn canonical SWAAG → live Phonon → reconnect

A **further** isolated test finally used the **current recorded speech from the same authenticated localhost WebSocket session**, passed it through a real resident Thor Whisper Large-v3 request, preserved the resulting draft for **test-only manual review**, and then called a **fresh actual canonical SWAAG foreground orchestrator** (Nitro) through an intercepted test-only external gateway. The latter ran with an independently verified explicit **zero-tool** configuration, no side-effect or stateful permission and no tool loader; user speech was transmitted on a fixed SSH subprocess's **standard input**, never interpolated into a shell. The model answer was validated **before any audio synthesis**. The real Jetson `VoiceSession.answer` and Phonon stream generated audio after that response, while the test disconnected on the second WebSocket generation after one thousand nine hundred twenty PCM bytes. The third socket recovered the **exact** stored PCM, acknowledged normal playback, and the fourth socket did **not** replay. This is a successful **isolated same-turn text/voice/TTS transport integration**, not a production app or genuine owner microphone authentication.

The **first** fresh same-turn test proved the mechanical pipeline but **failed voice-text quality**. SWAAG responded with a two-hundred-forty-three-character self-description of absent tool permissions and **repeated** the same clarification question. The remote canonical history confirms **one completed model request after seventy-five-point-two seconds, zero tools and zero workers**. The audio/reconnect transport passed, but this was **not** accepted as a satisfactory voice response. Original evidence remains in private Jetson `fifth-fresh-canonical-20261121` and Nitro `fresh-reviewed-voice-canonical-20261121` folders; neither session has been overwritten or replayed.

A strict acceptance-only validator, `validate_short_voice_clarification`, was added to the owning staged utility. It requires exactly **one** short natural completed question and rejects repeated questions, internal tool/policy disclaimers and malformed/verbose text; it **never rewrites the model response**. The new source-pinned script `scripts/demos/verify_real_voice_fresh_canonical_phonon.py` uses that gate in a **distinct** fresh no-tools canonical session. Its exact source SHA-256 is `577ce1be8cab0137196798080035838125a6438972fd471750907bbab157dea4`, matching the complete real Jetson tested script byte-for-byte. The separate Python unit suite checks this source hash explicitly, in addition to the quality gate. The complete test-only original WAV and independent historical no-tools source proof remain mandatory prerequisites; a run without the explicit `--run` and `SWAAG_STREAMED_PHONON_NOACTION=1` does not perform speech work. The fresh Nitro config is fixed and independently checked as entirely tool-free before admission.

**Second fresh same-turn acceptance (verified):** The actual model produced a concise, correct, natural single question, **“What would you like to know about the passage?”** The fresh canonical durable journal contained exactly **one completed model request, zero tool calls, zero worker starts**, model elapsed time **sixty-eight-point-two seconds**, first-token latency **twenty-six-point-four seconds**. Independent Nitro evidence is in private `fresh-reviewed-voice-canonical-20261121-quality2/independent-fresh-response-evidence-20261121.json`, original journal SHA-256 `854bd60927d5e025eea62706610bc273457b62b50584bd78a1e023636e1adeea`. The Jetson result of **the same fresh reviewed voice turn** confirmed one real Whisper, one fresh SWAAG response, one real Phonon stream, four WebSocket generations, **seventy-six thousand eight hundred PCM bytes** of newly synthesized speech, first **one thousand nine hundred twenty bytes** delivered before disconnect, **exact full PCM recovery**, normal ACK, **no fourth-socket replay**, and **zero executed agent actions**. That complete isolated run took about **eighty-two-point-nine seconds**, including **seventy-five-point-seven seconds** for the SSH and canonical response and **one-point-three seconds** TTS streaming. Independent Jetson source hashes, emitted reply hash and PCM/ACK checks are in private `sixth-fresh-quality2-20261121/result.json` and `fresh-quality2-crosshost-verified-20261121.json`. The independent Nitro and Jetson reply hashes agree.

**Still unverified:** persistent real owner speaker enrollment and identification, semantic automatic end-of-turn acceptance, **production** WebSocket server-to-SWAAG/TTS deployment, genuine physical Android microphone/speaker playback and background reconnection, the previously **denied physical lost-ACK scenario**, multi-sample speech recognition/latency percentiles and normal conversational response time. This test uses a previously approved public-domain recording and an explicitly test-only manual confirmation. It must not be described as authentic human owner approval, or as a production service release. The model's sixty-eight-second response alone is far too slow for a practical voice agent. It is not appropriate to optimize away authorization or contextual grounding just to improve speed.

## November 21 afternoon: source-pinned quality-three run and single-use session admission

The third genuinely fresh, no-action same-turn test used a **new** independent
Nitro session rooted under `fresh-reviewed-voice-canonical-20261121-quality3`
with exact source revision and config SHA verified on-host before execution.
Its independent original journal records **one completed canonical model call,
zero tools and zero workers**. The accepted natural answer was **“What aspect
of the passage would you like to know about?”**. The model-to-Jetson turn took
about seventy-one-point-six seconds, followed by a real Phonon Easton stream
of about one-point-five seconds. Actual resident Whisper processed the
original unchanged recording, first audio was delivered, and one thousand
nine hundred twenty bytes were transmitted before disconnection. The third
WebSocket delivered exactly the entire newly synthesized ninety-four thousand
seven hundred twenty PCM bytes; a normal ACK persisted; no fourth-connection
replay occurred. The isolated full test ran approximately seventy-nine-point-nine
seconds. Exact private receipts remain under Jetson
`real-canonical-reply-phonon-playback-20261121/seventh-fresh-quality3-20261121`
and Nitro `fresh-reviewed-voice-canonical-20261121-quality3`. **The source
actually used by that completed live run had SHA-256
`7aab8bfc6db5cf4d19b56e6b7748c669ab867ef36bec03f29567276322e1af04`.**
It is retained in the private Jetson `quality3_scripts` directory as original
source evidence.

A subsequent independently tested hardening removes the remaining risk of
**silently reusing an older canonical test conversation**. The fresh runner
now requires explicit `--canonical-source`, `--canonical-config`,
`--expected-source-commit` and `--expected-config-sha256`. Before a live
request, its remote read-only check verifies the exact source Git revision,
the private config bytes and empty enabled-tool registry. It then calls
`scripts/demos/claim_fresh_voice_noaction_session.py` on Nitro, which
**atomically creates one mode-0600, single-use private marker** and refuses
existing `session_*` directories or an already-used claim. This prevents
concurrent and repeated execution under the same private session root,
including after an uncertain model or socket outcome. The original and newer
private voice session journals remain untouched.

The claimed-session admission had **eight passing source tests** and a
separate actual local private-filesystem test: its first claim succeeded, the
marker matched the recorded admission with mode 0600, and a second attempted
claim returned a nonzero `FileExistsError` without reopening the session.
Additional source tests validate exact path grammar and fail-closed Git/config
digests. **This added one-use guard is a follow-on deterministic improvement;
the historical quality-three live run above predates the guard and must not be
misrepresented as running the newer source.** Use a new private session and
new proof for the next real acceptance; never call the same accepted `--run`
again merely because its terminal output was lost.

## Final November 21 same-turn run: published one-use guard, actual resident voices

The **published** canonical SWAAG source revision
`243c12c8fd3707490477fd27c1cf8c5bb6831d2b` was independently verified
on remote `main` before the next private real run. On Nitro, the new
`fresh-reviewed-voice-canonical-20261121-quality4` root used a previously
nonexistent session directory, mode-0600 private no-action configuration
SHA-256 `48b7bd7c8e23f81e75cda4f370e1fa5dab6d45cf9d018334944b5905a8e2f57f`,
exact source Git revision and explicit zero-tool registry. The Jetson runner
source copied byte-for-byte from that published revision was checked again
against the stored original voice source and private historical proof. A
source-only dry run created **no** session or model request. On the actual
`--run`, the new Nitro single-use claim was durably admitted, recorded **zero
prior session directories**, and blocked any duplicate claim for that same
configuration.

The complete original real recorded speech then passed through one actual
resident Whisper request into **manual test-only review**, then one **new**
canonical SWAAG tool-free model turn, then live Nitro CPU Phonon Easton
streaming through the existing Jetson private tunnel. The no-action reply
was the natural, accepted single question, **“What would you like to know
about the passage?”**. The model's independent Nitro append-only journal
contains **one model request, one response, one finished turn, zero tools,
zero tool results, zero workers**. The model portion took about
**sixty-seven-point-five seconds** with time to first token around
**twenty-six-point-six seconds**. Jetson's SSH/orchestrator boundary took
about seventy-four seconds in that run. The actual Phonon streamed response
completed in about **zero-point-eight seconds**. A total of
**seventy-six thousand eight hundred PCM bytes** were synthesized, the first
**one thousand nine hundred twenty** delivered before intentional disconnect,
and the **exact entire freshly generated audio** was recovered on the third
WebSocket connection. The normal ACK was durably recorded; a fourth socket
correctly did not replay. The complete isolated run finished in about
**eighty-one seconds**, exit zero, with **no agent tools/actions**.

Independent cross-host proof is private, not copied into Git. On Nitro see
`/data/var/swaag-benchmarks/fresh-reviewed-voice-canonical-20261121-quality4/independent-oneuse-live-canonical-checkpoint.json`
plus the same directory's preserved exact event journal and single-use
claim. On Jetson see
`/data/var/swaag-benchmarks/real-canonical-reply-phonon-playback-20261121/eighth-fresh-quality4-20261121/result.json`
and `quality4-independent-crosshost-evidence.json`. The two independent
checks record the same emitted reply SHA-256
`97f61b088c8c4308493a85f6c89ff462997ea39b79881883d53eeb80ee7fa57b1`.
The source-provenance checksum of the Jetson-copied **final guarded** test is
`e70fd5a908b795cc362d310d287cf88b7fa13e293cafab14b437844ab384f669`.
The single-use guard also has independently observed first-claim success and
second-claim refusal on a separate private Nitro test root, plus a combined
**two-hundred-eighty-one-test regression pass** and follow-on focused checks.
Those test sets overlap and are not the entire SWAAG repository-wide suite.

For **independent spoken-content quality**, an earlier generated real Phonon
Easton WAV (SHA-256 `59ae90f298e95f42662bf75e4bef0640e7ea91261dddd29ecf358eff4115352a`)
was sent back through resident Whisper Large-v3. Its transcription matched
**all nineteen words** of the intended SWAAG answer, **zero word errors in
that single test**. Exact private evidence is on Jetson in
`/data/var/swaag-benchmarks/voice-output-recognition-20261121/original-reply-backtranscription.json`.
This result **does not** overturn the separately recorded error on the
**input** public-domain speech where Whisper changed “soft hearted” into
“thought-tortured” (two errors out of twenty words): different recorded
sources, input/output directions and conditions. Neither single sample is
a population-level model accuracy claim.

**Unresolved production gates remain unchanged:** A test-only local manual
confirmation is not real human owner/speaker authentication. These isolated
servers are not the live phone's permanent service or SWAAG owner session.
Automatic semantic end-of-turn, phone microphone and audible playback, owner
identity, real-world interruption policies, physical lost-ACK behavior and
production service deployment remain unverified or outside this test scope.
The canonical response duration remains far too slow for routine interactive
voice use. No resident service was restarted or reconfigured, no physical
phone touched, and no external user-command operation was executed.

## Fast-foreground full voice acceptance candidate — source-only admission

The real fast-foreground model evaluation identified a substantial SWAAG
latency opportunity without removing context or tool authority. A new
**separate** same-turn no-action voice acceptance consists of
`scripts/demos/run_fast_noaction_voice_once.py` (Nitro) and
`scripts/demos/verify_real_voice_fast_foreground_reconnect.py` (Jetson).
The Jetson side uses the same **real** source WAV, resident Whisper and
manual test-only review, then passes the **exact ASR text over stdin** to a
newly admitted private canonical SWAAG fast-foreground call. Nitro requires
a verified source revision, configuration fingerprint, zero tool inventory,
pre-existing fresh one-use claim and a second exclusive inference marker.
The `generate_fast_foreground_response` path may **answer** with one real
model call, or it may **delegate**. In the latter case this acceptance
**fails closed** and never starts the full controller, a worker or any
action. Only a short natural question and one completed zero-tool durable
turn can proceed into real Phonon streaming and four-WebSocket media recovery.

Source and unit tests are not a successful live acceptance; the real result
requires an independently checked actual Nitro model journal, a real Jetson
Whisper request, a Phonon output stream, exact saved PCM replay, the normal
ACK and no final replay from a **new** unused private run. Preserve private
logs and do not reuse any previous source session. No new resident service,
Android release or owner identity/authorization is introduced by this
experiment.
