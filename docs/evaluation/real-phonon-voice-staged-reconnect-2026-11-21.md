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

### First verified fast-foreground real recorded speech and reconnect (November 21)

**The real fast version passed**, beyond the source-only admission above. The
exact original public-domain recording (SHA-256
`2b29bead44440ee10117f8dc6f2f20be552aa2ffbb39b36f10c854dc165376ee`)
was actually recognized by resident Whisper and held as a **manual test-only
reviewed draft**. The **exact same ASR bytes** (SHA-256
`78d06ad3eb01b5470b3742a96b0d5d089f3ad0b73988078200020d6dd33297e2`)
were passed over authenticated private SSH **stdin**, not shell-interpolated,
into a previously unused, single-use-claimed Nitro SWAAG session on published
revision `f1c78f491d325c5f85bc9f333e5a0f459e98b283`. Source and config
SHA, zero enabled tools and empty prior history were checked before the call.
The real Qwen3.8 twenty-seven-billion Q8 model chose one **direct answer**:
“I received your message. What would you like to know about that quote?”
No deep fallback was invoked. The durable Nitro journal independently confirms
one model request and response, one finished turn, **zero tool calls, zero
workers**, prompt **six hundred ninety-three tokens**, first token around
**one-point-zero seconds**, model elapsed **five-point-two seconds**, and
fast-stage wall time **eight-point-seven seconds**. The Jetson-to-Nitro
call took about **nine-point-nine seconds**.

The accepted question went through the real installed **Phonon Easton**
streaming service in about **one-point-eight seconds** and the original
Jetson WebSocket media path. After exactly **one thousand nine hundred twenty
PCM bytes** were delivered the socket was disconnected. The complete **one
hundred seventeen thousand seven hundred sixty PCM bytes** were recovered
byte-for-byte on the third connection, with a durable **normal** playback
acknowledgment and no replay on the fourth. The whole isolated test
completed in **seventeen-point-six seconds**, exit zero. It never invoked
SWAAG user tools, edited live session state, restarted speech/model services,
or touched a real Android microphone or speaker. The earlier full-orchestrator
acceptance took about eighty-one seconds in another complete run, but the
model prompts and response lengths differ; **the observed time difference is
not a guaranteed speedup**.

Cross-host source-backed evidence is private: Nitro
`/data/var/swaag-benchmarks/fast-foreground-voice-e2e-20261121/independent_fast_model_live_verification.json`
with original append-only source journal SHA-256
`7b2b7f2de9fb6ecc9133620c0815f6822234989e67d32807a3cdbf5c79659213`,
and Jetson
`/data/var/swaag-benchmarks/fast-voice-e2e-20261121/independent_fast_voice_endtoend_evidence.json`.
The independent model and Jetson reply SHA-256 match exactly:
`9e08cff60ac1b53e673fafbbf2121ffa9bc8b246dd821608bf7010186e305f08`.
The test's source code and follow-up speech rules are published on the
**canonical SWAAG main branch**; live services remain unchanged.

A follow-on optional source-level improvement adds
`fast_voice_audio_evidence.py`: it saves the **actual newly generated
Phonon WAV** in the run's own private directory before the ephemeral
VoiceCore test fixture can clean it. The helper checks bounded, non-silent
sixteen-kilohertz mono PCM16, writes mode-0600, refuses overwrite, fsyncs
and verifies a byte-exact SHA-256 copy. Unit tests check the copy, replay
refusal and invalid WAV conditions. **This WAV retention is not part of the
first completed real run above**; a separate source-pinned new session is
required to verify it with actual audio and independent ASR back-transcription.

The user speaker and automatic turn boundary still need real production
validation. The earlier original audio had a meaning-changing recognition
error and required manual review. A standalone quoted statement is not an
automatically authorized command. Responses from one resident model under
shared GPU contention have variable latency; production phone experience
and percentile response times remain unverified.

### Second fast real same-turn voice acceptance: immutable TTS and spoken output

A separate **new** single-use Nitro fast SWAAG session and isolated Jetson
recorded-speech test verified the same real original WAV, resident Whisper,
one actual Qwen3.8 Q8 **fast direct** model response, and Phonon Easton
streaming with exact four-WebSocket recovery. The second attempt used canonical
source revision `20d27feef91bdce97eb2d862ed005fbd8e8660e9` and a distinct
no-action config/session and never replayed the first. Its private append-only
Nitro model journal confirms one model request and response, one finished
turn, **zero tools and zero workers**, one-use claim and extra inference
marker, with model elapsed **nine-point-three-five seconds**, first token
**one-point-four-six seconds**, fast stage **thirteen-point-zero seconds**.
The actual Jetson test took about **twenty-five-point-two seconds**, with
Phonon streaming around **two-point-eight-eight seconds**. The generated
**one hundred seventeen thousand seven hundred sixty** PCM bytes were
recovered byte-for-byte after one thousand nine hundred twenty were sent
before deliberate disconnection; the normal ACK was persisted and the
fourth connection did not replay. The fresh assistant reply was **“I
received your message. What would you like to know about that quote?”**.

The NEW versioned helper `fast_voice_audio_evidence.py` saved the **actual
newly generated** private WAV before temporary test fixture cleanup: exact
SHA-256 `56e81fe7050641490a634e064705673a1f6c61ee2cf46d6b09f33d8270e7fae7`,
mode 0600, non-silent mono sixteen-kilohertz PCM, no overwrite. Jetson then
passed that **same exact WAV**, not a separately synthesized substitute,
through resident Whisper Large-v3: it transcribed the exact **thirteen of
thirteen** intended words (one sample word edit distance zero). The PCM RMS
was approximately two thousand two hundred forty-nine, peak amplitude
fourteen thousand three hundred twenty-two, with **zero clipped samples**
at the configured threshold. This automated audio-output check is **not** a
human perceptual review or a statistical model-quality estimate. The
unchanged input-side original source still has the previously documented
meaning-changing two-of-twenty-word ASR error and must remain manually
reviewable.

Private cross-host evidence: Nitro
`/data/var/swaag-benchmarks/fast-foreground-voice-e2e-retained-20261121/independent_retained_fast_model_verification.json`
with actual journal SHA-256
`3800e7e5b727c9b6a41347b57a73245e36f1a93f8d1338803384090f0dd45d4f`;
Jetson
`/data/var/swaag-benchmarks/fast-voice-e2e-retained-20261121/independent_retained_fast_voice_evidence.json`
and `retained-fast-reply-actual-Whisper-backtranscription.json`, plus the
restricted original WAV in `real-once-20261121/fast-generated-reply.wav`.
The corresponding private source-output and canonical reply hashes agree.
The live source/recorder was not deployed to the physical Android phone.

### Independent live negative boundary: model delegation cannot execute tools

A distinct private one-use no-action Nitro test submitted the **synthetic
test-only**, non-owner text “Use the calculator to compute six times seven.”
The actual resident Qwen model returned fast **`mode=delegate`** after about
**seven-point-nine-seven seconds**, and the fast-only acceptance **exited
with a controlled nonzero refusal**. The model proposed an intention to use
the calculator, but the source and independent journal show **one real model
request and response, one `assistant_progress` proposal, zero actual tool
calls, zero tool receipts, zero workers, no control consumption and no
finished turn**. The deep controller was not invoked. Its one-use claim and
model invocation marker remain, so this pending simulated control must never
be replayed as if it were an already completed external effect. Private
reproducible evidence is at
`/data/var/swaag-benchmarks/fast-foreground-delegation-noaction-20261121/independent_real_negative_delegation_evidence.json`.

The negative test also exposed **a temporary supervisor bookkeeping entry**
when the fast-stage method was invoked directly outside the full controller.
The normal production controller already releases that observer in its
`finally` block. The isolated fast-only runner now uses the same scoped
`supervisor.finish_session(session_id)` API on every exit, without altering
the durable control queue or falsely marking the task complete. A new
source-backed regression tests the negative decision, retained pending
control, zero tool calls and observer release, with **twenty-four** focused
runner tests passing. The original model evidence is unchanged; the cleanup
source correction is later than this live negative test, so no retrospective
claim is made that the model ran new code.

### Remaining decision gates and measurement limits

Across the individually observed real, separate Qwen3.8 fast turns, model
latency varies even on the same short transcript. A four-observation
experimental set includes model times of about five, seven, nine and twenty
seconds; the earlier full-controller live response took about sixty-seven
seconds. They have different prompt lengths and shared GPU loads, so they
**do not establish a general median, tail percentile, controlled speedup or
production service-level objective**. Reproducible low-latency operation
needs cross-load acceptance with complete authority/history. Genuine owner
speaker identity, semantic automatic turn completion, physical Android
mic/playback, live production SWAAG server deployment and actual owner
commands remain outside these no-action tests.


### November 22: reusable, fail-closed verification of the words actually spoken

The earlier real same-turn test saved a byte-exact Phonon reply and independently
transcribed one output through resident Whisper. That one passing
thirteen-word sample **does not** establish every reply is correct or that
a physical phone emitted audible words. A separate later stored three-second
Phonon output was independently submitted to the resident Whisper verifier
without calling SWAAG again. All **eleven of its eleven intended words**
matched, with zero normalized word edits. The one-use private evidence is on
Jetson in `/data/var/swaag-benchmarks/owner-serial-000058-20261122/retained-phonon-audio-asr-verify.json`.

The versioned `scripts/demos/verify_retained_voice_spoken_content.py` now
provides a reproducible source-backed acceptance gate. Supply a private
completed one-use no-action result receipt, its **same** generated Phonon WAV,
and a **new** private evidence directory. Without `--run`, it validates hashes,
completed SWAAG response, zero actions, original disconnect/replay/ACK
receipts and bounded speech audio without contacting a service. Only the
explicit `--run` flag **and** environment opt-in
`SWAAG_RETAINED_TTS_ASR_NOACTION=1` permit a single real HTTP request to the
fixed resident Thor Whisper endpoint; it never calls SWAAG, TTS, shell, a
third-party provider or the production WebSocket. The unused destination is
claimed before the request and intentionally cannot be reused after a failed
or uncertain attempt. Complete expected and observed words are kept in
mode-0600 private JSON outside Git; ordinary stdout omits them. Differences
are recorded as nonzero word edit distance and a nonzero exit status, not
silently accepted.

The script and its deterministic tests require only standard Python and
pytest. Fake-transport tests validate error handling without a model or
network request. This test confirms **agreement between a second ASR model
and one particular generated voice output**, not intelligibility to all
human listeners, natural prosody, acoustic robustness, owner identity,
semantic accuracy of the original transcription, or phone playback. A real
physical-device hearing and microphone acceptance remains mandatory before
calling the assistant production-ready.

### Live Jetson legacy automatic-send authority is not yet protected — November 22

A fresh **read-only inspection of the actual running Jetson service process**
identified `/data/src/worktrees/voice-sse-durable-fastpath-20261022/voice_agent_live_runtime/cloudflare_ws_voice_server.py`
as the loaded script (SHA-256 `561183f64ae76a9aa92e18b8f2a8c1192b7f9fecc7597233bb363bb64e85415f`). Its
real private `/etc/voice-agent/ws.json` sets `auto_send_default=true`.
That production source's `VoiceSession.client_state` accepts a legacy client
claim `build="Voice Agent 1.7.16"` without the newer client commit protocol.
When a turn-completion classifier returns positive high-confidence semantics,
the legacy automatic path can call `submit_turn(..., automatic=True)` after
ASR review, **without a trusted server-side determination of the speaker**.
Existing private WebSocket bearer authentication is device/application
access, not verification of the human speaking. Do not interpret the
built-in test-only `Voice Agent 1.7.16` string as proof that a real authorized
phone currently uses that path.

The actual **production-source unit tests were rerun independently** in an
isolated temporary voice configuration, with `submit_turn` mocked and zero
SWAAG/model/physical-phone calls. Three of three passed: the legacy path
would attempt automatic submission after positive model-free decisions,
manual mode does not auto-submit, and a new/unknown client cannot bypass
the newer commit protocol. Their private source-hash/test receipts are on
Jetson under
`/data/var/swaag-benchmarks/owner-serial-000058-20261122/live-release-source-legacy-auto-commit-result.json`.
This confirms a **source/release safety gap**, not an observed production
submission or a proof that the live phone uses the older build.

A **different, not-deployed candidate** under
`/data/src/worktrees/swaag-local-llm-noaction-20261119/voice_agent_live_runtime/`
contains `require_manual_review_for_unknown_speaker`, including an
`automatic=True` `submit_turn` guard and a default-on
`require_verified_speaker_for_auto_send` flag. Five of five separately
rerun isolated no-action tests passed for spoofed client speaker claims,
automatic submission, explicit manual send, conservative default settings
and stale I/O revision fencing. These tests do **not** demonstrate actual
owner-speaker recognition: they intentionally preserve manual review.
Their private source-hash/test receipts are in the same November twenty-second
Jetson evidence directory.

A safe future release must first source-pin and independently test the
actual intended Jetson service bundle and configuration; preserve the
older installed service and reversible rollback; positively establish
manual review and disabled automatic action for any unverified speaker
including legacy clients; then validate real device microphone, human
hearing/playback, reconnect, and trusted owner-only enrollment/anti-replay.
Do not hot-restart the active owner service or deploy the candidate merely
because these isolated tests passed. Unreviewed automatically submitted
commands remain an explicit production acceptance blocker.

### November 23: actual replayed assistant speech cannot authorize itself

A distinct, deliberately hostile no-action acceptance exercised the exact
archived **unreleased** default-on speaker-review patch for the *currently
running older Jetson WebSocket source*, not the unrelated later voice branch.
The original exact source revision was `f8f89b788f4fdf83d8115c7d4586df880f5427eb`.
The private candidate source was restored without modification from the
November 22 SHA-256-verified four-file archive. The patched server SHA-256 is
`cbcc77fb36ce8cf72cc6442ec0f35d84526aec4868a27b810b2c38bde52bb302`,
and the patched configuration parser SHA-256 is
`b3052253b3d3a22f0f3f69bf6cd264768d6dc0bab99fcd94fbb874b1a98c69ba`.
The production service was **not** restarted or altered.

The fixed input was a **previously genuinely generated** three-point-seven
second Phonon Easton WAV from an actually completed SWAAG no-tools foreground
answer, not synthetic bytes or a newly synthesized substitute. Its SHA-256
is `56e81fe7050641490a634e064705673a1f6c61ee2cf46d6b09f33d8270e7fae7`.
The script first checked the exact original answer receipt, previously
successful byte-identical reconnect and persisted normal playback ACK, and
source and fixture hashes. A separately verified archive contained precisely
the expected four regular files; no arbitrary tar paths were extracted.

After explicit one-use admission, `scripts/demos/verify_voice_tts_replay_noaction.py`
started an **isolated authenticated localhost WebSocket**, supplied this actual
recorded assistant reply as microphone PCM, and made **one real** request to
resident Thor Whisper large-v3. It recognized all **thirteen original words**.
The fake legacy mobile client advertised automatic sending and falsely claimed
trusted owner speaker identity; the candidate kept the speech as one durable
**unsubmitted manual-review draft** without calling the semantic completion
classifier. After disconnect and reconnect, the same draft was restored. A
second fake client with the newer protocol and a forged `commit_turn`
message could not submit it. There were **zero automatic proposals, zero
actual SWAAG/tool/TTS/cancellation calls, and zero automatic submissions**.
All assertions and one-use receipt evidence are stored privately on Jetson
under
`/data/var/swaag-benchmarks/owner-serial-000060-20261123/replay-test-01/`.
The observed run took about three seconds. The original audio source and
prior SWAAG journal were not replayed to a model or service.

The **versioned** script is usable as a negative acceptance check for exactly
this source archive. Run without `--run` for a pure local preflight. Running a
real isolated WebSocket/ASR requires both `--run` and
`SWAAG_TTS_REPLAY_SPEAKER_NOACTION=1`, a pristine private output directory,
the exact archived candidate source, the unmodified matching original fixture,
the original pinned answer receipt and WAV, current broker/service admission,
and no still-running conflicting test. It writes a unique mode-0600 attempt
marker before any resident-ASR request and never retries an unknown outcome.
Fifteen new deterministic tests verify source/WAV/receipt tampering rejection,
negative receipt semantics, no implicit network attempt, failure retention,
exclusive evidence creation, and patched non-action operation boundaries.

**Limitation:** This verifies a conservative unverified-speaker hold even for
replayed assistant audio. It does **not** implement or validate actual owner
voice recognition, playback device acoustics, immunity to a recorded owner's
voice, semantic turn completion accuracy, real Android operation, or any
production deployment. A full owner voice identity and anti-replay mechanism
would require separate protected enrollment and independent real-device
adversarial testing before enabling automatic actions. Do not describe this
as a complete anti-spoofing or production-ready solution.

### November 23: independent actual-process startup of the same patched release candidate

The same exact unreleased patch was also staged as a separate, seven-module
source-pinned private release candidate, with unchanged dependencies copied
from the actual installed Jetson source. A new versioned tester,
`scripts/demos/verify_isolated_voice_release_server.py`, validates all seven
module hashes and the exact original production-config hash before allowing
any process or network operation. It creates a **distinct ephemeral
localhost-only WebSocket process** from the candidate source and a one-use
configuration. Its authentication token is a harmless fake, its event/history
storage is inside its private evidence folder, and **every** ASR, SWAAG,
LLM, Phonon, worker-events and turn-classifier network route is changed to an
unavailable private loopback destination. No actual speech, model or tool
request can leave this test instance. The original production configuration,
service and user phone are never modified.

The first attempted private process startup **failed** before opening a
socket because the test launcher incorrectly resolved its virtualenv's
symlinked Python executable to the system interpreter, where `websockets`
was not installed. The original one-use failure receipt and error log are
retained under `private-process-smoke-01`; **that attempt did not pass**. A
regression was added to preserve the interpreter's original virtualenv
entrypoint, and all thirteen focused admission tests passed. Only then was
an entirely new exclusive `private-process-smoke-02` test admitted. It
**passed**: actual candidate child bound its own localhost socket, HTTP
health responded live, an authenticated WebSocket returned its real hello
with `unverified_speaker_requires_review=true` while `auto_send_default=true`,
empty durable history remained empty, and an unauthenticated WebSocket was
rejected with HTTP 401. Its process was terminated and reaped by its owning
test harness, not by any systemd operation; the production server continued
running with its original unchanged process identity.

The observed second private-process test took about **seven-tenths of a
second** and retains source/config hashes and independent JSON/log proof on
Jetson under
`/data/var/swaag-benchmarks/owner-serial-000060-20261123/private-process-smoke-02/`.
The test is reproducible only with the exact private candidate source,
original production template, the opt-in `SWAAG_PRIVATE_RELEASE_SMOKE_NOACTION=1`
and `--run`, a free private output destination and no competing instance.
Its default mode is a no-write, no-network source preflight. It never
updates or restarts `jetson-voice-ws.service` and cannot be treated as a
production deployment or real Android/owner-speaker acceptance.

Both this independent process smoke and the separate real replayed-TTS ASR
experiment support a conservative fail-closed candidate release. Neither
establishes a trustworthy biometric owner-speaker verifier, semantic-action
quality, live-phone playback, full cross-load voice latency or resilience
against an actual owner's recorded voice. A reversible, idle-safe production
release with actual device acceptance remains separate, unperformed work.

### November 23: one actual independent server process processes real microphone-like PCM

A third separate, **new one-use** Jetson acceptance closed the gap between
previous in-process real Whisper tests and a server merely starting as a
private process. `scripts/demos/verify_isolated_voice_real_audio_reconnect.py`
uses exact SHA-pinned files from the **unreleased patch** for the actual
Jetson service source, the existing immutable genuine SWAAG/Phonon audio
and answer receipt, and an isolated private bearer and history directory.
It sends raw sixteen-kilohertz microphone-like PCM to a **real separately
running** patched WebSocket server, with automatic sending enabled and a
forged old-Android owner claim. A fixed localhost-only ASR relay checks that
the original captured speech occurs in the actual incoming WAV and forwards
**one** request to the existing resident Thor Whisper endpoint. Every other
SWAAG/LLM/TTS/ASR/worker/classifier endpoint in the candidate configuration
remains unreachable at an inert private loopback address.

**Observed true end-to-end result:** Whisper correctly transcribed all
**thirteen of thirteen** intended words from the original generated
three-point-seven-second Phonon reply. The one real ASR inference took about
**two-point-six seconds**; the isolated complete process/voice/reconnect
acceptance took about **three-point-nine seconds**. Despite auto-send and a
forged legacy device/owner identity, the patched server emitted an explicit
`speaker_identity_unverified` review state and stored one unsubmitted draft.
A second WebSocket connection to the **same still-running child** restored
that exact speech draft and refused a forged newer-client owner commit.
There were **zero automatic submissions, zero semantic auto-completion
calls, zero SWAAG model/tool/TTS calls, and zero external side effects**.
The isolated child was terminated after the test; the real Jetson systemd
voice worker retained its original process and source. This distinct stage adds **six** offline tests,
including single-use ASR-relay failures, audio-source pinning, isolated route
admission and permanent refusal of an uncertain duplicate test. Full actual
private evidence is on Jetson under
`/data/var/swaag-benchmarks/owner-serial-000061-20261123/actual-real-WS-audio-once/`.

### November 23: a new server process restores the real review draft after termination

That one-use real-audio server left a valid atomic `conversations/*.json`
containing exactly **one** review-only draft, its privately saved captured
WAV and an orphaned `*.tmp` history file. The test child was terminated
while a later atomic save was in flight. The fully written authoritative
history and the temp file both contained **unsubmitted draft** state; no
finished or submitted action was lost or incorrectly invented. This is a
single observed process-termination case, not evidence of universal power-
loss or filesystem fault tolerance.

The versioned independent
`scripts/demos/verify_isolated_voice_process_restart.py` first reads and
checks the original private answer receipt, recorded WAV, full durable
history, event record and any interrupted temp saves **without any network
requests or writes**. With a distinct explicit opt-in and output folder it
copies the actual review-only state, captured audio and orphan temporary
save to a **new private sandbox**, replacing original absolute audio paths
in that copy only. It then starts a **new real patched server process** with
all model, Whisper, agent, TTS and worker routes disabled, and authenticates
a test-only socket. The new process restored the exact **thirteen-word**
speech draft from durable history without sending it through Whisper or
SWAAG again. An allegedly owner-verified new client and forged `commit_turn`
were rejected. Its private result confirms **zero automatic or manual
commands, zero new ASR/model calls, a correctly restored unsent draft, and
unchanged original files**. It took about **nine-tenths of a second**. Thirteen
new negative and recovery regression tests exercise corrupt/committed
history, unsafe audio paths, malicious temp saves, exclusive attempt
admission, and preservation after a test failure.

The real restart evidence is on Jetson at
`/data/var/swaag-benchmarks/owner-serial-000061-20261123/actual-process-restart-once/`.
The original first-run journal and captured speech remain preserved. All
new testing was limited to private test-only credentials and isolated
server processes; the production voice service, physical phone and owner
recordings were never modified.

These source-backed observations support **in-process voice-to-Whisper,
real independent subprocess voice-to-Whisper, a safe review decision, socket
reconnection, and real child-process restart recovery for one fixed no-action
example**. They do not establish a trusted owner-speaker identity, immunity
to the owner's recorded voice, microphone hardware input, actual Android
playback, a fully automated tool-using voice session or a production-safe
release. A controlled real-device acceptance and separate authority-tested
reversible deployment remain required. The existing historical multi-step
SWAAG tool demonstration is still separate and only succeeded after a
corrective supervisor turn. Do not silently combine these into a claim that
a fresh voice session executed real tools autonomously.
