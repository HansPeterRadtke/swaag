# Voice and concurrent communication

Voice integration is intentionally outside the SWAAG reasoning core. A voice client converts speech to text, determines that the semantic user turn is complete, sends the authoritative finalized turn to the persistent user-facing orchestrator, receives the orchestrator response or durable worker results, and performs speech synthesis locally or in an external voice service.

## Orchestrator-first interaction

There is exactly one official SWAAG entry point for ordinary voice/chat user messages: orchestrator.message. This is not one option among several equivalent APIs. A voice frontend must not send ordinary speech to Task API worker creation/message operations, AG-UI worker submission, A2A task submission, or a worker session. Those are lower-level developer/integration interfaces.


Every finalized ordinary user utterance goes to orchestrator.message. The orchestrator is the conversation endpoint. It does not create a worker merely because the user spoke.

Every finalized message reaches the full action controller with current system state, ordered user authority and the configured completion review. The controller can answer lightweight conversation without creating a worker, or issue orchestration tools when needed. An action can contain both user-facing text and validated tool calls.

The earlier one-call semantic gate remains available for experiments but is not the ordinary dispatch boundary. Live mixed prohibitions exposed lost instructions in that split, and the lexical unperformed-work guard was removed. A separate conversational reader is a hypothesis requiring end-to-end qualification, not an installed correctness shortcut. See the [current guideline comparison](guideline-comparison-2026-10-15.md) and [foreground study](foreground-routing-and-retries-2026-10-15.md).

An explicitly worker-addressed control remains available through the Task API when a client intentionally targets an existing worker. That is a lower-level control operation, not the default route for ordinary human conversation.

## Configurable model topology

The model configuration selects background worker inference. Setting communication.model_base_url selects a separate runtime for the user-facing orchestrator and lightweight communication operations. That orchestrator also performs full planning; this setting alone does not create a fast conversational reader handing off to a different, stronger planner. Model size and process count are not requirements. Qualify correctness, actual concurrent capacity, latency and resources before selecting either shared inference or a separate model. The orchestrator owns user interaction and global task planning; workers own delegated task execution.

True background conversation requires independent inference capacity. Two URLs backed by one single model slot do not create concurrency. Use separate servers or devices or enough parallel backend slots. When the orchestrator shares a backend with a worker, user interaction has control priority and SWAAG uses cancellation plus exact replay rather than pretending transformer generation can always be suspended and resumed.

The specialized communication-status operation still exists for evidence-grounded interpretation of a specifically targeted worker snapshot. It is not the ordinary voice conversation endpoint.

## Android flow

Android should own microphone capture, VAD, STT, TTS, playback, local barge-in, reconnect behavior, and optional wake-word or push-to-talk UX. Send only semantically completed, authoritative user turns to orchestrator.message. An STT-final segment or a pause is not necessarily the end of the user's thought. Retain transcript revisions and turn identity; partial hypotheses may be displayed or used for reversible provisional work but must not authorize consequential actions.

Keep the persistent orchestrator conversation identity across reconnects. When the orchestrator delegates work, retain the resulting plan and worker identities and event cursors for progress and final-result delivery. Do not create a worker for every voice utterance.

During TTS, distinguish intentional interruption from backchannels, hesitation, unrelated speech and noise. Speech detection may trigger reversible local attenuation or a pause, but must not itself cancel an authorized task or finalize a new command. Use semantic turn/interrupt handling, allow correction and resume playback where appropriate. Submit the authoritative completed new turn to the orchestrator. SWAAG can preempt shared-backend inference when necessary, reject stale actions and reconstruct exact context; already committed effects must be checked rather than replayed blindly.

A literal user request to cancel or change ongoing work is still sent to the orchestrator first. The orchestrator applies the requested plan or worker change. Direct worker cancellation and control APIs remain available for clients that intentionally expose lower-level controls.

## Spoken output

Immediate orchestrator replies are already user-facing conversational text and should be spoken directly. For final background-task responses that require a retained audio-specific transformation, request an audio presentation; the canonical worker result remains authoritative and the audio variant is separately compiled and evaluated.

Do not narrate every mechanical event. Keep ordinary heartbeat, tool execution, queue state, and internal progress visual or silent. Speak blocking questions, important failures, explicitly requested status answers, direct orchestrator conversation, and final answers.

## Current implementation boundary

The loopback listener exposes orchestrator.message as the ordinary user-conversation operation. Finalized authority is persisted before execution/session admission. The iterator and JSONL stream expose accepted, committed assistant_progress and completed events with durable receipt identity and event cursors; the ordinary CLI consumes the stream. The synchronous Python method still returns the completed result.

The execution job runs independently of the stream consumer, so slow or paused speech playback does not hold orchestration. The first ordinary action requires a nonempty message; progress from a tool-bearing action is published after its tool receipts succeed and only if a newer user control has not superseded it. A receipt proves the operation it actually records: queued or consumed worker instructions do not prove the requested external effect. Worker-launch/control receipts can return while background work continues.

Session mutation remains serialized to preserve ordering, and model inference can still dominate latency. Current snapshots and stale-action checks protect transitions; these mechanisms do not promise a twenty-second semantic reply or physical voice-device acceptance. Measure receipt, new conversational delivery, dispatch, effects, final answer and playback separately, excluding historical progress replay from new-response latency.

### Durable voice-operation compatibility

The voice gateway supplies an `operation_id` instead of a native `control_id`. The JSONL request accepts `stream: true` and `after_sequence` with this identity. Acceptance includes the operation identity, persistent orchestrator session and derived receipt identity; progress contains exact committed text and increasing history sequences; completion contains the canonical answer. Supplying both identity types is rejected. This path runs the same current full action controller, never the earlier experimental semantic gate.

The operation ledger persists acceptance before execution. Concurrent streams attach to the same in-process execution; completed duplicate requests reuse the canonical result. A lost process or uncertain effect is recovered through `orchestrator.operation_status`, not automatic replay. `orchestrator.interrupt` targets this operation, requests cancellation of its active run and covers the interval before a run ID exists. Cancellation acknowledgement requires the terminal canceled state; it does not undo committed effects. An unconsumed canceled receipt is removed from pending authority so another turn cannot execute it later.

Operation-ID JSONL streams also emit `keepalive` events during quiet inference or context reduction. `communication.stream_heartbeat_seconds` defaults to five seconds and must be positive and at most thirty seconds. Gateways must keep their idle read timeout above this interval. A keepalive proves only connection/service liveness; it carries no assistant text and does not prove model progress or task completion. Native control-ID stream behavior is unchanged.

Correction ingress: both operation-ID and native control-ID requests persist their canonical user receipt before the session execution lock. Operation-ID streaming also persists before executor admission. Regression coverage verifies that an arriving correction rejects a running action's stale tool intent, that queued withdrawal requires no execution slot, and that canceled consumed receipts recover committed answers without replay. The earlier documentation claiming operation-ID ingress still held the execution lock was stale. The voice server now admits finalized follow-ups without waiting for prior inference, while preserving ordered audio delivery. Held-inference regressions verify exact correction ingress, no premature audio, and eventual ordered replies. These tests establish the software boundary; physical-phone interruption acceptance remains unverified.

TaskApi remains the transport-neutral lower-level worker lifecycle boundary: create and start, message or redirect, cancellation, resume, event cursors, status evidence, attachments, and terminal response presentations. The separate status operation remains available for targeted worker status or history questions. These interfaces share durable state but have different responsibilities.

## Minimal deployment

Run the main worker model under model.base_url. Use communication.model_base_url only for an independently qualified configuration; the persistent orchestrator then uses that endpoint while workers use the main endpoint. Keep existing production workloads and Infra admission reserves intact. Enable the communication service and keep its listener on loopback behind an authenticated gateway when Android is remote.

The Android or voice gateway sends every finalized ordinary utterance to orchestrator.message, not to a freshly created worker and not directly into an arbitrary active worker. Background worker and event APIs are used after delegation for durable progress, cancellation, and final results.

Benchmark actual simultaneous latency with the deployed servers before relying on the topology. A second orchestrator or communication model improves responsiveness only when the hardware or backend can actually execute it independently of worker inference.

## Retained-state foreground context (October 16, 2026)

The normal action controller and independent completion checker remain the conversation boundary. The communication tool set includes the read-only `runtime_snapshot_lookup`: the first view retains active workers/plans, all outstanding questions and revisions, current user controls and live-worker controls. Terminal task details and terminal workers' queued controls remain in the exact timestamped snapshot, with counts and JSON-pointer retrieval. Terminal status is not evidence of an effect. Inspect the exact details before reporting historical outcomes or resuming terminal work. Snapshot freshness fences use the canonical full snapshot, not the display view.

Earlier assistant machine-action envelopes are disclosed on demand; exact user wording, visible replies, delivered progress and current-turn actions remain available to the action controller. Consecutive identical earlier user messages can use lossless repetition notation with all source sequences. The checker receives visible wording once, with exact metadata and duplicated turn envelopes available through integrity-checked history sources. Tokenization, inference transport and internal generation rejection records are recoverable telemetry; actual tool failures, current authority and task state are not converted into success or deleted. Cryptographic bookkeeping stays in durable records instead of being repeated in every prompt component.

The foreground stage uses compact default instructions and capability descriptions while retaining scoped durable instruction content and precedence. Custom system/action templates are preserved. Full tool schemas load through the existing discovery path. Removing `runtime_snapshot_lookup` from the communication tool configuration restores full snapshot rendering and the ordinary templates; canonical history is unchanged either way. Whole-request provider token counts determine whether input accounting is exact, even when individual component estimates are approximate. Failed identical runtime-context projections are not retried merely because another source was reduced. Completion reduction measures capacity with its structural output minimum rather than its desired maximum.

Production-state reproduction used a consistent copy of `/data/var/swaag/voice_sessions`, including thirty-seven terminal workers and the retained conversation. It exposed both oversized terminal-state input and more than one thousand legacy per-fragment tokenization events entering completion evidence. Evidence is retained under `/data/tmp/voice-guideline-repair-20261016` on Thor, including `snapshot-disclosure-regressions-v9.log` (234 passing checks), `snapshot-disclosure-regressions-v10.log` (54 passing follow-up checks) and the explicitly versioned production-state probes. Interrupted superseded probes are failures/interrupted trials, not successful acceptance. These changes do not by themselves establish physical Android playback or a latency guarantee.

The final retained-state regression matrix passed 237 checks (`snapshot-disclosure-final-tests.log`). Both BF16 and the existing Q3 endpoint completed a copied-production greeting with one action and one independent completion check; the observed Q3 run was slower and was not selected for production. The public authenticated endpoint subsequently returned `Hello.` and 13,312 PCM bytes with an audio-end event, but first audio took 160.84 seconds. This was transport consumption, not a physical speaker/playback test.

Profiling then identified an inventory-wide repeated scan: each worker's independent-status lookup enumerated every session and loaded its status history. Independent status observations now share one fresh batch per history owner per inventory, and worker action statuses use indexed event-type reads. The batch never survives into another snapshot, so newly rejected observations and external journal updates are checked again. The relevant status, snapshot, question/revision and history regressions passed 41 checks, including an external writer invalidating one of four observations between inventories. A same-state non-profiled comparison measured warm snapshot collection at 3.46 seconds before batching and 0.33 seconds afterward. These are specific host measurements, not a latency SLA.

Two isolated BF16 semantic holdouts also completed against the compact foreground templates. A mixed status question plus publication restriction queued exactly one control with the original user authority (229.79 seconds). A request to translate the same quoted prohibition left the worker unchanged and returned the German translation with zero controls (159.36 seconds). The latter initially omitted the translation; the independent completion checker rejected that draft and the controller corrected it before delivery. These limited holdouts support the tested authority boundary, while their latency and the rejected draft remain part of the evidence; they are not a claim of general semantic or voice-quality acceptance. Results are in `foreground-quality-bf16/results.json` and its client log.

The final production test on code `2df026e` again returned `Hello.` and 13,312 PCM bytes through the authenticated public endpoint. First PCM arrived at 100.676 seconds and audio-end/final completion at 101.180 seconds (`production-smoke-status-batch.json` on Nitro). This improves the earlier 160.84-second first-audio measurement but still fails conversational responsiveness. No physical playback acknowledgement was supplied. Required voice services remained active and enabled; the Android delivery stays version 1.7.16 because the subsequent changes are server-side.

A further foreground delay came from separate live HistoryStore instances treating each other's validated local appends as external journal edits. The verified file-observation cache is now shared by live stores for the same canonical root within one process; file identity, timestamps, size and SQLite cursor checks still invalidate genuinely external changes. Weak ownership releases unused roots. A regression reproduced five full scans after five peer appends before the fix and none afterward; same-size external content tampering is still rejected. Ninety-nine history/status/snapshot/control/stream/operation checks passed. On the retained-state copy, snapshot collection immediately after a peer append fell from about 2.49 seconds to 0.29 seconds. This cache is derived acceleration only; durable authority and recovery remain in the journal and database.

Foreground actions now have a separate initial output reservation, `communication.initial_foreground_output_tokens` (default zero). Zero selects the schema's structural minimum; the compiler still allocates all safe available headroom up to the desired output size. Worker reservations are unchanged. Actual output exhaustion expands the reservation and recompiles before any returned action can execute, including a complete-looking JSON tool call accompanied by a length finish reason. The new setting, recovery and context checks passed 174 tests (`foreground-budget-tests.log`). A fresh retained-production measurement after sequence 2851 fit at 7903 input tokens, 225 output tokens and 64 safety tokens, with no context projection required.

The preceding real Android production-chain run failed its unchanged 150-second playback deadline (`live/android-chain-production16.log` on Nitro). Two avoidable context projections consumed 55.994 and 64.54 seconds because the generic initial output minimum exceeded available headroom by 39 tokens. Its server turn eventually completed; that late completion is not an Android test pass. The reservation repair targets this reproduced delay without clearing conversation history, disabling completion verification, or changing model services.

The next Android run still failed (`live/android-chain-budget16.log`, 172.254 seconds). Its live action prompt was 48 tokens larger than the copied-state measurement, so the reservation change alone did not eliminate compression. Worse, the completion reducer measured a placeholder using the full evidence catalogue even when the actual foreground prompt used a compact catalogue. The resulting false capacity deficit prevented recovery and repeatedly regenerated the same answer. That owned test operation was explicitly canceled and its terminal canceled state verified. A synthetic retained-history regression reproduced the catalogue failure before the fix and passes afterward. Both the real prompt and sizing placeholder now use the same displayed catalogue.

History role and source headers now share one line. Durable-instruction JSON uses compact separators without modifying string content. Ordinary foreground completion conversation rows use an explicit sequence/role/content column map; JSON escaping, exact order, extended metadata and unknown evidence objects are preserved. Round-trip tests recover the original rows exactly. Ninety-six completion/history/instruction/provenance checks passed (`foreground-format-final-tests.log`), following 132 action/provenance checks (`compact-header-tests.log`). A fresh copied-state measurement fits the next action at 7693 input tokens; its completion context still requires semantic reduction at 9309 input tokens. This is not yet a live playback pass or conversational-latency acceptance.

On deployed `008c856`, Android again missed the unchanged playback deadline (`live/android-chain-catalog16.log`, 180.806 seconds). The action fit directly and took 27.691 seconds; completion reduction now worked, but its first 256-token output was truncated after 63.95 seconds and its retry took another 58.638 seconds. The independent checker then completed in 7.213 seconds and the server finished; the late server success does not turn the Android timeout into a pass.

Completion reduction now names the current execution boundary, distinguishes the immediately preceding finalized user request from older turns, and states that the candidate answer is supplied separately to the verifier. It asks for the shortest sufficient purpose-specific evidence while preserving applicable constraints and uncertainty. A direct replay of the failing projection with explicit boundary/candidate context completed in one attempt in 44.393 seconds using 165 tokens (`completion-projection-boundary-probe.json`). The final boundary wording and history/completion suite passed 52 tests (`completion-boundary-tests-final.log`). These measurements preserve the original source history and normal independent checker; physical playback and latency acceptance still require their own observed results.

The next production-connected Android run on code `512f057` passed (`live/android-chain-boundary16.log` on Nitro): 151.571 seconds for the whole instrumentation case, with its unchanged 150-second playback wait. Recognition was exactly `Can you hear me? Please say hello.`, the answer was `Yes, I can hear you. Hello.`, and all 1738 milliseconds rendered through emulator AudioTrack. The acknowledged outbox was cleared. Canonical Jetson conversation `android-chain-8eaf227c-715b-4fc4-ad86-6ce2be122090` has completed state, transmitted audio and `audio_completed=true`. ASR measured 1.445 seconds, agent handling 114.780 seconds and TTS 2.518 seconds. Model calls took 27.585 seconds for action, 57.621 seconds for one successful history projection and 9.200 seconds for independent verification. Evidence is `live/android-result-boundary16.json` on Nitro and `live/android-server-boundary16.json` on Jetson. Temporary test TLS/SSH forwarding was then removed and the isolated test app stopped; production services stayed active.

This closes that functional Android deadline failure for the observed run. It does not close the fast-conversation requirement: the retained-history response still takes nearly two minutes. Model throughput, history reduction and processing overhead remain material. No physical phone was connected, and no physical microphone/speaker, OEM or accessibility claim follows from emulator rendering. The normal APK remains version 1.7.16 in Nitro Downloads; these server-only changes require no additional APK build.


## Phone failure reproduction and correction (October 16, 2026, evening)

The actual phone request at 17:32 UTC reached Jetson and SWAAG. Its reply eventually completed and the client acknowledged playback, but agent processing took 634.224 seconds. A second request queued behind it and was explicitly canceled during this repair, with terminal cancellation verified. Active services and the earlier emulator pass did not establish usable phone responsiveness.

The retained phone request exposed recursive history projection, duplicated current request text, and a completion evaluator demanding delivery of the very candidate it was supposed to approve for delivery. Fresh runtime state is now a separate, exact completion component; history reduction cannot replace its current controls or receipts. The immediately preceding current request is supplied verbatim once, with unknown message metadata conservatively retained. Identical typed copies inside the action snapshot refer to that canonical user-message sequence; different pending messages and all control identities/authority stay exact. Canonical snapshots and history are unchanged.

When the foreground completion prompt overflows, a model sees every earlier row and selects constrained row IDs relevant to the current request/candidate. The harness copies those rows exactly, preserves disclosure notices, and leaves the full integrity-checked source catalogue available to the independent verifier. There is no keyword routing, recency filter or sampled selection. Oversized, invalid or unavailable selections fall back to existing reduction instead of truncating evidence. Selection returns IDs without a verbose explanatory essay. The verifier is explicitly at the pre-delivery boundary: accepting permits delivery; it cannot require that candidate's prior delivery. Claims about earlier playback, fixes and external effects still need evidence.

The focused completion/history/snapshot suite passed 80 checks (`phone-final-tests.log`); the Infra documentation gate passed 12. Earlier replay variants remained failures: one generated repeated 256-token summaries; another demanded prior playback and recursively expanded snapshot evidence. The final saved-phone replay completed its selection and independent verdict in 21.963 seconds, with one 9.998-second selection and one 8.320-second verdict (`phone-selection-final-live.log`). A separate real-model negative replay rejected the unsupported claim that earlier playback succeeded and the sending issue was fixed (`phone-false-playback-verdict.json`, 35.303 seconds). These are backend replay measurements, not yet a deployed Android-chain result or a conversational-latency guarantee.

APK delivery correction: the requested Nitro web Explorer uses `/data/var/web_portal/uploads`, not desktop Downloads. The normal package is available there as `android03-voice-agent-1.7.16.apk`, verified against the original APK. The user confirmed its visibility. These changes are server-side; no additional APK build is required.


The first deployed Android run on `be96544` failed its unchanged playback deadline (`live/android-chain-phone-repair16.log`, 179.026 seconds overall). The input had grown to 121031 tokens because the canceled earlier phone turn had performed three full read-only snapshot lookups, each retaining a 99041-character payload as a tool message. Those past reads bypassed the older internal-action disclosure rule and re-entered every new action prompt. The owned verification operation was canceled after the timeout; it is not a pass.

Earlier `runtime_snapshot_lookup` results now carry exact tool-event/snapshot/pointer references instead of replaying the entire old read payload. Current-turn reads, effect receipts, visible conversation, unknown result extensions and results with context updates remain unchanged. The original tool records and snapshot values remain immutable and available through history lookup. The action/history/snapshot regression matrix passed 138 tests (`phone-snapshot-read-tests.log`).


Measured overflow in the initial foreground action now uses the same constrained semantic selection before attempting expensive projections. The model chooses exact earlier visible conversation messages; the harness preserves the current request, every message from the active turn, tool/effect receipts, extended unknown authority metadata and independently observed runtime state. Selected earlier messages keep their original roles, bytes and provenance. No selection runs when the original action context fits, and unavailable/invalid selection keeps the existing recovery path.

A reproduction from retained production state through sequence 3863, including the three prior full snapshot reads, reduced the initial action from 121031 tokens (original deployed failure) to 8660 with read references alone, then to 5838 with semantic selection. That selection took 3.809 seconds; the full pending action compiled within the unchanged 8192-token model limit without executing the action (`phone-action-current-selection.log`). The original source history was neither reset nor truncated.

The combined action, snapshot, history-selection and completion regression suite passed 181 tests in 59.47 seconds (`phone-action-selection-tests.log`).


### Phone failure follow-up: bounded exact history selection (2026-10-16)

The Android run against `5db2bd2` generated the correct greeting, but did not finish playback within the unchanged test deadline. Its independent completion history required 17,918 tokens; selecting that history also overflowed at 14,248 tokens. The prior fallback entered slow recursive summaries. This failed run is retained in `android-chain-history-selection16.log`; it is not a playback pass.

Exact-row selection now splits only after a measured capacity failure, on whole-record boundaries, processes every fragment in source order, and unions selected identifiers before copying original records. No sampling, fixed recency window, canonical history deletion, or completion-check bypass is introduced. Invalid output, a single oversized record, or exhausted existing call budget still falls back conservatively. Current input and fresh state remain separate. Sixty focused completion/history/recovery checks passed in 11.60 seconds (`phone-batched-selection-tests.log`); live Android validation is pending.


The next Android run (`android-chain-batched-selection16.log`, conversation `android-chain-b0914b48-bb1e-4c55-9b9a-684bbf8a329c`) also failed its unchanged playback deadline. Bounded source selection worked, but one selector generated repeated whitespace inside a JSON array until its output budget exhausted. No phone/audio success is inferred from that failed run. The selector now explicitly requests a minified JSON line and its request/candidate components have clear boundaries. Replaying the exact failing model request with the formatting instruction returned a valid empty selection in 4.674 seconds and seven output tokens, versus the original 58.442-second whitespace exhaustion. The sixty focused checks passed again in 11.59 seconds (`phone-minified-selection-tests.log`). Model service configuration and resource allocations remain unchanged.


## October 16 verified playback after the phone-failure repair

The new Android run against SWAAG code `8016d9b` passed `ConnectedVoiceWorkflowTest` in 139.673 seconds overall with the original 150-second playback wait unchanged. Its fresh conversation is `android-chain-e963f647-76ff-4b57-8417-05e0d93f6d17`. The app restored synthetic speech, obtained the editable text `Can you hear me? Please say hello.`, submitted it, displayed `Yes, I can hear you. Hello.`, cleared its acknowledged outbox and rendered all 1,738 milliseconds through emulator AudioTrack. Android measured 133,668 milliseconds for its transaction. The matching canonical server turn is completed with audio transmitted and the client's playback acknowledgement recorded. Final ASR took 1.320 seconds, agent processing 104.043 seconds and TTS 2.223 seconds.

The four exact-history selector calls completed in 4.183, 6.046, 4.489 and 5.574 seconds; the normal action call took 25.315 seconds and independent verification 10.357 seconds. Earlier failed runs remain failures. This is a bounded emulator functional pass. The remaining delay is still a failed fast-response requirement; physical-phone acoustics, full accessibility and upstream correction priority remain unqualified. One internal model status also still misinterpreted earlier unavailable activity telemetry, although the tested user-facing greeting was correct. No overall guideline-compliance claim is made.

Nitro evidence: `/data/tmp/voice-guideline-repair-20261016/live/android-chain-minified-selection16.log`, `android-result-minified-selection16.json` and `public-health-phone-repair16.json`. Matching Jetson evidence is `live/android-server-minified-selection16.json` under the same evidence root. Temporary loopback TLS/SSH helpers and ADB forwarding were removed, and the isolated test app was stopped. All public readiness components remained true afterward. Production services remain running. The normal phone APK remains verified in Nitro web Explorer's upload folder; these server repairs need no replacement for app version 1.7.16.


## October 16 actual-phone automatic-send and telemetry follow-up

The subsequent real phone complaint reproduced two defects that a manual greeting did not cover: Jetson abandoned semantic turn classification after two seconds on the shared main-model route, and the controller copied old delivered progress claiming an unavailable backend. The backend had no activity adapter; missing activity telemetry did not establish service failure.

Code 1b58622 reports unsupported/failed activity observations separately from service availability, retains unknown availability, and lets the existing semantic history selector choose earlier delivered progress. Exact originals and active-turn authority remain recoverable and unchanged. Fifty-one focused supervision/history/recovery checks and twenty progress-selection/history checks passed. Replaying the actual failing phone request showed that telemetry wording alone was insufficient and that removing stale progress alone still prompted an irrelevant snapshot read. The final capability-aware prompt returned a direct acknowledgement with no tools or continued loop in 44.754 seconds. This is a semantic replay pass, not conversational responsiveness or phone acoustic proof. Evidence remains in `/data/tmp/voice-guideline-repair-20261016/phone-progress-capability-replay16.log` and the earlier failed replay logs.

The separate voice gateway now uses its qualified dedicated Qwen3-VL-8B-Instruct classifier for turn completion and interruption only; SWAAG still uses Qwen3.8-27B BF16 with independent completion verification. Gateway configuration and deployment details are documented in Infra `docs/voice-agent-live.md`. Automatic Android validation follows separately; the earlier manual tests do not establish automatic-send behavior.


## October 16 automatic Android acceptance and remaining latency

The fresh automatic run passed `ConnectedVoiceWorkflowTest` with `voiceChainAutomatic=1`: `OK (1 test)`, 120.687 seconds overall. Conversation `android-chain-ab38d884-7606-4e72-8697-5c28cb1c8b42` recognized and automatically committed `Can you hear me? Please say hello.`, cleared the editing field after acknowledgement, displayed `Yes, I can hear you. Hello.`, and rendered all 1,738 milliseconds through emulator AudioTrack. The unchanged playback deadline was 150 seconds. Android measured 111,497 milliseconds total and observed acknowledged history with an empty editor after 21,389 milliseconds, including connection setup and real-time fixture upload. Jetson recorded `audio_transmitted=true` and `audio_completed=true`; final ASR took 1.438 seconds, the semantic turn decision about 0.720 seconds, main agent processing 81.824 seconds and TTS 2.756 seconds. The first automatic instrumentation failure remains recorded separately; this fresh pass does not overwrite it.

This establishes the tested automatic PCM/STT/classifier/proposal/client-commit/ACK/UI/agent/TTS/playback-ACK path. Only the microphone source was synthetic; model replies, server events and playback were real. It does not establish physical-phone acoustic playback, visual acceptance on the Power Armor screen, all accessibility behavior, correction priority under concurrent work, or acceptable conversational latency. The main agent delay remains a failed responsiveness requirement. A faster main-agent candidate using the already-running eight-billion-parameter model was tested on the exact failing phone context: selection took 2.874 seconds and action 10.414 seconds, but it invented an editing-state diagnosis and continued the loop. That candidate failed qualification and was not deployed as the agent. It remains qualified only for the separately tested narrow turn/interruption classifier task. Main agent and workers retain Qwen3.8-27B BF16; model services/quotas were unchanged.

Thor gateway readiness now includes the dedicated classifier, so a failed classifier produces HTTP 503 even when SWAAG is healthy. The final gateway/configuration suite passed 45 tests; Jetson fallback suite passed 49; the Infra documentation gate passed 12. The deployed implementation is SWAAG code 1b58622, Thor voice code 59d823d7 and Jetson voice code 25fe17b. Private evidence under `/data/tmp/voice-guideline-repair-20261016/`: Nitro `live/android-chain-automatic16b.log`, `live/android-result-automatic16b.json`, `live/public-health-automatic16b.json`; Jetson `live/android-server-automatic16b.json`; Thor `phone-progress-replay-fast16.log` and `phone-classifier-health-tests16.log`. Public readiness and all voice dependency services were verified active. Owned test TLS/SSH forwarding and ADB reverse were removed; the isolated test app was stopped. Normal package 1.7.16 remains unchanged in Nitro web Explorer uploads, with its original verified checksum. Actual phone client-state telemetry already identifies version 1.7.16; these server repairs need no new APK.


## October 17 evidence-schema factoring

The independent completion contract now groups exact evidence source IDs by source kind. This is an algebraic factoring of identical object-schema branches: it preserves precisely the same allowed (source_kind, source_id) pairs, closed fields and required properties, without permitting cross-kind IDs. No history, source catalogue, authority or independent completion check is removed.

The completion/preemption/context suite passed 78 tests and the authority/stream/voice-operation suite passed 40. The original preemption test had a stale assertion for an old human-readable snapshot heading; it failed on the unchanged baseline too. It now checks that actual active-worker state is present. An initial baseline probe accidentally pre-imported the installed package; that result was discarded and the source-pinned baseline failure was retained.

A replay of the exact production completion request retained the same successful verdict. Schema size fell from 10,478 to 2,618 characters. The sequential original/factored calls took 18.157 and 7.478 seconds, respectively, with identical 4,477 prompt tokens. These single ordered measurements include cache/load effects and are not a causal speedup claim or end-to-end latency acceptance. Main agent remains Qwen3.8-27B BF16 and the independent checker stays enabled. Evidence: schema-live-replay17.json, schema-tests17c.log, authority-tests17b.log under /data/tmp/voice-guideline-repair-20261017 on Thor.

The October 17 natural-pause Android run reached the actual classifier/proposal/commit path and completed AudioTrack playback with a matching server playback acknowledgement. It took 145.633 seconds in-app, including connection/setup and real-time fixture upload; server agent time was 101.576 seconds. This failed responsiveness despite functional delivery. Its trace identified old context-preparation audits still repeated in completion evidence. Selection output counts, exhausted context-reduction output, derived tool-result projections and projection reuse now join the exact recoverable runtime audit source. Canonical user authority and original tool effects remain separate and unchanged; older source references retain their frozen event types. The evidence/preemption/context matrix passes 79 checks (audit-tests17b.log). The first new test run failed because its fixture omitted required tool-result input fields; that failure is retained. The independent checker also rejected the false phone-playback/fixed-sending claim in schema-false-playback17.json.

After the audit-view correction, one background trial measured 72.923 seconds of agent processing but failed Android playback because its synthetic-uplink fixture had omitted the real Mic entry point's foreground-service start. That trial is not an Android acceptance pass. The corrected fixture invokes the real service-start path, checks foreground ownership, and substitutes only the microphone PCM. Its fresh run passed background AudioTrack playback with the screen and test bindings removed, microphone off, and restored acknowledged history when the screen returned. In-app elapsed time was 123.416 seconds, server agent time 89.348 seconds, ASR 1.496 seconds and TTS 3.764 seconds. Both sides recorded complete playback. These distinct timings include load/cache/state differences and do not establish a causal performance gain. Conversational responsiveness remains failed. Android evidence: android-live-background-fixed17.log and android-result-background-fixed17.json on Nitro; matching android-server-background-fixed17.json on Jetson.


## October 18 foreground payload and stream-reader repair

The optional `communication.compact_foreground_actions` setting removes separately generated status and continuation fields from the ordinary foreground action contract. It retains the answer, exact tool calls, questions and response constraints. The runtime derives continuation from nonempty tool calls, applies the existing authority/snapshot fences and independent completion check, and leaves worker, silent and custom-template contracts unchanged. The setting defaults to false; deployment opt-in is recorded separately. No model server or history-retention setting changes.

A copied-production comparison through sequence 4669 completed the greeting with five model calls in both variants: 71.237 seconds with the existing contract and 60.434 seconds with the compact contract. The action calls took 22.876 and 11.136 seconds. These sequential measurements include cache/load effects and do not establish acceptable conversational latency. A direct replay against the existing Q3 endpoint was slower than BF16 and was not selected. Main agent and workers retain Qwen3.8-27B BF16; the dedicated turn classifier is unchanged.

The mixed-command qualification exposed an independent verifier defect: after a correct translation, the checker treated the quoted restriction as a new worker instruction. The checker now judges the whole authoritative utterance and distinguishes quoted/translated/hypothetical data from separately requested actions. The final real-model probes durably queued the genuine mixed restriction exactly once with its original authority, and translated the quoted restriction with zero worker controls. They took 73.731 and 22.438 seconds, respectively. An earlier bad quoted-command probe was interrupted in its isolated copy and remains a recorded failure. Final receipt/playback probes no longer claimed unobserved phone state; the editor probe still made an unnecessary offer of further work, so complete style/autonomy compliance is not claimed.

Transport history cursors now filter delivery-event types in SQLite before decoding payloads while preserving the existing journal integrity fence, exact sequence advancement, append races and recovery behavior. The same 38-page captured history scan took approximately 1.31 seconds before and 0.071 seconds afterward, with the same final sequence and visible events. Canonical history is not deleted or truncated. The final combined regression matrix passed 141 tests in 25.56 seconds, including malformed compact actions, actual tool results before completion, false playback rejection, correction fencing, quoted intent, history corruption, empty pages, racing appends and operation recovery.

Evidence is retained under `/data/tmp/voice-guideline-repair-20261018` on Thor: `full-comparison.log`, `full-baseline/result.json`, `full-compact/result.json`, `compact-quality-final.log`, `cursor-filtered.json`, `compact-replay-final-budgeted.log` and `final-regressions.log`. Earlier failed fixtures and probe errors remain separate. These are backend and isolated model results; deployment and a fresh Android playback test are still pending. No Android source changed, so these server changes require no APK replacement beyond the already delivered normal version 1.7.17.


The compact-action/stream repair was enabled on the idle production communication service at 09:02 UTC on October 18 (code 707942a). A fresh natural-pause automatic Android run passed in 113.908 seconds overall, 107.635 seconds in-app; acknowledged editor/history state was observed after 38.702 seconds, including emulator connection/setup and real-time fixture upload. The exact recognized request and reply were correct and all 1,738 ms rendered through AudioTrack. The matching Jetson turn is completed with audio_transmitted and audio_completed true. ASR took 1.433 seconds, agent processing 63.070 seconds and TTS 3.840 seconds. This is functional emulator evidence, not physical-phone validation or acceptable conversational latency. Nitro and Jetson evidence is `live/android-result-compact18.json`, `live/android-chain-compact18.log` and `live/android-server-compact18.json` under the October 18 evidence root. The current APK credential authenticated on public WSS in 0.938 seconds; an earlier probe used an old copied fixture credential and received 401. A generic urllib health probe received 403, while the Android-agent request returned 200. No authentication/TLS/WAF setting changed.

### Independent direct-conversation verification candidate

The optional `communication.direct_reply_verification` setting defaults to false and requires compact foreground actions. Only the first tool-free candidate, with no questions or resumed-control recovery, is eligible. An independent model checks the entire current request, current authoritative interventions, the exact conversation view already admitted before candidate generation, completion-scoped instructions and a newly observed runtime snapshot. It may approve only a sufficient direct conversational reply. It cannot execute tools, accept a handoff or finalize a turn. The existing final control/freshness fences remain in force.

Any non-approval, missing evidence, invalid/truncated result or context overflow falls back to the full evidence-capable completion evaluator. Tool actions, worker turns, questions, resumed turns and custom completion templates keep their existing full path. A genuine restriction embedded in a question is action, while translation/quotation is data. No action or prior external effect is proven by an assistant statement. Explicitly unverified playback can be a truthful answer to a receipt question; an explicit inspection/repair request still needs real work.

A copied-production greeting through sequence 4737 took 31.818 seconds with three model calls: one history selector, the action and the independent direct check. The verdict took 4.025 seconds and nine output tokens. This is a single isolated measurement, not a percentile or deployment acceptance. The first real-model matrix conservatively rejected a truthful receipt/uncertainty answer; its failure is retained. Refined scope wording passed all nine cases, including false playback, unperformed cancellation, mixed restrictions, prior constraints, a follow-up reference, quoted translation, honest uncertainty and explicit inspection. Full-controller mixed/quoted probes passed with exactly one/zero worker controls and exact original authority; the quoted reply conservatively fell back to the full verifier. Their times were 73.004 and 25.345 seconds.

The regression matrix passed 198 tests; ten final edge tests additionally cover complete-looking truncated verdicts, stale runtime snapshots, late corrections and invalid output. The initial truncation-test wrapper had a wrong fake-client signature; its fixture failure is retained separately and the corrected case passes. Evidence: `direct-full.log`, `direct-verifier-quality.log` (initial failed case), `direct-verifier-quality-v2.log`, `direct-workflow-quality.log`, `direct-final-regressions.log` and `direct-final-edge-tests-fixed.log`. Production activation and a fresh Android run for this second setting are pending.


The second setting was enabled on the idle production communication service at 09:20 UTC (implementation 2b1935c). The fresh natural-pause automatic background Android test passed: 76.361 seconds overall, 72.112 seconds in-app, acknowledged UI state after 26.314 seconds, and all 1,738 ms rendered with both the screen and test binding removed. The microphone stayed off and returning UI restored the acknowledged history. Matching Jetson state records completed with transmitted and acknowledged playback. ASR was 1.496 seconds, agent processing 41.865 seconds and TTS 2.410 seconds. The independent direct verdict was present in canonical history before terminal delivery; no worker/tool action was created. The initial post-restart preparation gap was 17.059 seconds, so the copied-state 31.818-second result must not be represented as the deployed result.

A subsequent warm authenticated public text-to-agent/TTS check returned the same correct greeting, began receiving PCM after 36.322 seconds and received the complete stream after 37.945 seconds. Its pre-snapshot preparation gap was 8.712 seconds. This separate probe consumed PCM without rendering it and deliberately sent no playback acknowledgement. It establishes public transport/agent/TTS output, not Android or physical acoustic playback. No percentile or general latency acceptance follows from either single trial. Conversational response time, cold preparation overhead, full two-stage handoff and physical-device qualification remain open. The actual speech-recognition/automatic-send/background-playback path passed in the emulator; no claim of complete guideline compliance is made.

Nitro evidence is `live/android-chain-direct18.log`, `live/android-result-direct18.json`, `live/warm-public-voice18.log` and `live/warm-public-voice18.json`; Jetson evidence is `live/android-server-direct18.json`, `live/warm-public-server18.json` and `live/final-health18.json`. The production flags are also retained in the Infra host configuration. Test forwarding and ADB reverse were removed and the isolated test app was stopped. The normal phone APK remains 1.7.17 in Nitro WEB Explorer uploads; these server-only repairs require no newer app build.

## October 18: spurious word-count metadata caused a multi-minute voice stall

The first live test after the Qwen3-Coder model switch confirmed that Jetson
received the audio, resident Whisper large-v3 transcribed the exact greeting,
the turn classifier accepted the completed utterance, and SWAAG durably
accepted it. SWAAG's action model produced a correct concise reply but also
invented response_constraints.exact_word_count=13, despite no active user
word-count requirement. The runtime correctly treated this model field as a
proposal rather than authority; however, verifying the proposal replayed all
historical messages, including three unrelated prior tool results of roughly
one hundred thousand characters each. Its exact_word_count_repair model call
assembled about one hundred and twenty thousand input tokens, spent minutes
prefilling them and did not return before the isolated smoke-test timeout.
The synthetic operation was canceled by its own durable ID; existing live
voice services and other sessions were not canceled.

Repair: send the complete original user-message history and only the most
recent assistant dialogue needed to interpret references to the exact-count
verifier. Do not include unrelated historical tool results as if they were
word-count authority. The original messages and tool evidence remain
recoverable in SWAAG's append-only history. Active user corrections and
current request text are still included; genuine exact numeric word-count
requirements still use the existing structured one-word-per-field enforcement.
This addresses the large-prompt stall without silently enforcing a count
invented by the action model. The fix has regressions covering massive stored
tool results, newer corrections, invented counts, and a valid current numeric
requirement. Passing backend tests are not proof of acceptable end-to-end
latency or physical-phone operation; a fresh real transport test is needed
after deployment.

## October eighteenth: cooperative foreground priority across Thor model processes

The user-facing Coder model and Qwen3.8 Q8 worker model occupy different
llama.cpp servers but share Thor's GPU. SWAAG previously prioritized model
requests only within an individual model URL. The communication service now
installs an optional inference-admission guard on each configured worker
backend *different from the foreground backend*. The guard checks pending
accepted foreground operations and waits before starting the next worker
model call. It does not cancel running inference, stop model servers, change
worker objectives, or replay tools. Existing queued inference identity,
priority, cancellation, queue-time accounting and worker durability remain.
The foreground Coder coordinator is never guarded by worker status.

The guard reads process-local accepted foreground stream futures and the
active ordinary foreground operation. This prevents an abandoned persisted
operation left by an older process from permanently freezing workers. A
foreground text/voice turn that is accepted before a worker's next model
call fences that new call; an already-running Qwen3.8 inference is not
preempted by this change. The final concurrency race before admission is
not a cross-process atomic transaction. Actual cross-backend cancellation
and replay remain a separate, measured requirement.

Regressions cover wait/release, queue retention, cancellation while waiting,
finite timeout, foreground bypass, and same-backend backward compatibility.
An isolated real-backend smoke test on Thor held an accepted foreground
future while a genuine Qwen3.8 SWAAG worker queued. No worker model slot
started until the foreground future completed; the same worker then
finished a five-answer arithmetic request successfully, around twenty-eight
seconds after release. The user-facing inference server was never stopped.
This is verified cooperative admission, *not* yet maximum-speed real-time
preemption or end-to-end physical-phone acceptance.

## October 19: fuse independent direct reply and proposed word-count verification

Production voice logs repeatedly showed the Coder model attaching a fabricated
numerical reply-word constraint to a simple "Can you hear me?" turn. The runtime
correctly refused to treat that claim as user authority, but resolving it
required an extra expensive LLM call before the independent direct-conversation
completion check. This change merges the two independent semantic judgments
into one strictly validated, structured direct-conversation verdict *only*
for the first accepted, tool-free, question-free foreground action. The direct
checker receives the model-proposed number as explicitly **untrusted** data
and returns both `direct_reply_sufficient` and
`declared_word_count_supported`. The checker may approve the candidate only
when no real numeric count was required and the entire reply is supported.

A real numeric requirement, an incorrect or incomplete candidate, unsupported
claims, invalid JSON/schema, missing live state, concurrent user correction,
missing evidence, or a checker refusal instead uses the original independent
exact-count repair and full completion evaluator; an action with tool calls is
never eligible. Every proposed constraint and its disposition remains in the
durable audit, and the canonical original user request is untouched. No
regex-based intent router or text-as-shell-command mechanism was introduced.

The focused direct-reply tests passed (13), including fabricated constraints,
explicit genuine word counts and invalid combined verdicts. Broader action,
communication, inference and question tests passed (194). A fresh isolated
real-model Coder greeting test completed with an independently verified
answer in about eight seconds, but that particular model output contained no
invented numerical word count, so it did not exercise the fused-count branch
in live inference. That branch has fake-model regression coverage and still
requires a full production-history smoke test before latency improvements can
be claimed. This change is server-side; no Android APK rebuild is required.

### October 19 actual production-history audio acceptance

After deploying the integrated source, an isolated synthetic microphone
utterance was streamed to the live Jetson WebSocket, passed the resident
Whisper large-v3 final transcription, the natural silence detector,
semantic turn-completion classifier and automatic user-turn commit. The
retained production SWAAG orchestrator replied through Qwen3-Coder and
LuxTTS returned actual PCM frames, the correct final answer and terminal
audio events. Total session-to-audio-complete time was **53.633 seconds**;
first returned PCM began at **52.046 seconds** and the exact final
transcription arrived at **4.9 seconds**. The result is in
`/data/tmp/voice-basic-check-20261018/result.json` on Jetson. This
shows functioning live server communication, not acceptable conversational
latency or physical-phone microphone/speaker acceptance.

The real production-history trace is especially important: the Coder again
invented exact_word_count=13 for “Can you hear me? Please say hello.”
The new independent combined checker responded with
`direct_reply_sufficient=true` and `declared_word_count_supported=false`.
The durable event `response_constraint_repaired` rejected this unsupported
model-authored count without an additional model call. The first action
prompt still contained about 13,268 estimated tokens and took 13.377
seconds of model time. The independent combined check used about 8,381
input tokens and took 9.051 seconds. Preparation and transport remained
substantial; the goal of genuinely responsive speech is not met.

A separate isolated test of the previously dormant one-call
`generate_orchestrator_interaction` gate demonstrated a promising
1.69-second simple greeting and 0.82-second explicit worker-start routing,
but it **incorrectly returned route=respond** for the mixed request
“Can you hear me, and also create a worker to fix the Android voice app?”
It falsely promised to initiate that work without performing a handoff.
A hypothetical request also exhausted its output budget. Therefore this
gate is **not safe to enable as a bypass** for the authoritative full
action controller without stronger semantic acceptance tests, reliable
operation receipts, and independent fallbacks. No router bypass was deployed.

### October 19 full Android emulator chain against deployed SWAAG

A real Android instrumentation test was repeated after deployment, using the
existing version 1.7.17 application, a synthetic PCM microphone fixture,
the actual started foreground service and backgrounded Activity, an
isolated local TLS/SSH test transport, Jetson's production ASR and WebSocket
server, the live Coder foreground orchestrator, and Nitro's LuxTTS output
played through emulator AudioTrack. The test passed: transcription matched
"Can you hear me? Please say hello.", the automatic edit/commit/ACK path
cleared the composer, background playback completed, and the returned
assistant said "Yes, I can hear you. Hello." All generated emulator
transport helpers were shut down and the ADB reverse mapping removed.
The test report is on Nitro at
`/data/tmp/voice-guideline-repair-20261019/live-emulator/result.json`.

This was **not** a fast interaction. Android reported 56.035 seconds to its
acknowledged-submission screen condition and 118.986 seconds from its test
starting point until successful background playback and UI confirmation.
The server's own logged session (identified by its privacy-preserving
conversation hash) reached ASR final after about ten seconds from WebSocket
session opening, and recorded about 56.326 seconds of SWAAG/model wall time
before LuxTTS. Android's system ActivityTaskManager logged approximately
24.646 seconds to display the test app's MainActivity on this emulator,
compared with shorter but still multi-second display times on previous test
runs. The UI timing is an emulator observation, not evidence of equivalent
latency on the user's physical phone. No new phone APK is necessary for the
server-side verifier repair; the physical phone still needs acceptance.

The live SWAAG durable trace also showed the model inventing
`exact_word_count=10`, and the fused independent verifier correctly
returned `direct_reply_sufficient=true` and
`declared_word_count_supported=false`. There was **no third word-count model
call**. Nevertheless the foreground model's request preparation and
output-verification remained slow. This establishes the safe optimization in
both a direct WebSocket test and an Android UI/service/audio test; it does
not establish production-ready real-time latency.

## October 19: move deterministic worker-state assembly off the voice hot path

Production-backed profiling identified expensive deterministic work in
CommunicationService._orchestrator_runtime_snapshot. It rebuilt the full
worker/question inventory on every foreground turn. Cold profiling took
39.334 seconds with 27.6 million Python calls: question_inventory used
38.745 seconds, latest_communication_statuses 34.637 seconds and
_ensure_session_indexed 31.720 seconds while 21,169 journal events were
reverified. Warm profiler observations took 3.522 and 0.718 seconds.
Those numbers include profiler overhead. A separate uninstrumented
full refresh of the real production-shaped 39-worker store took 5.0 seconds.

The new foreground architecture creates an exact, full read-only worker,
plan, question, control and supervision snapshot on an independent service
task. Initial history prewarm completes before serving user requests.
The user-facing thread copies the already prepared snapshot, preserving
authoritative original source records.

Each cache hit performs small SQLite index checks to detect worker/plan
status changes, exact per-worker history cursors (progress and questions),
and exact worker control receipt state including pending versus processed.
Foreground active-run state and pending user controls are overlaid from
their small own canonical store. A revision change or expiry causes a fresh
validated snapshot rather than unverified stale status assertions.
No model call or semantic intent decision is introduced by this code.

Unprofiled real-store verification: a full worker snapshot cost about
five seconds on this run. Twenty-five subsequent foreground snapshot
reads took 0.0586 seconds total, median 2.33 milliseconds each INCLUDING
the canonical revision check. Copy-only median for a 100 KB prepared
snapshot was 0.617 milliseconds over two hundred reads. These are
component timings, not complete voice-turn latency.

The previously attempted persistent filesystem timestamp shortcut for
history validation remains rejected: Thor files can show identical
mtime/ctime after immediate same-size alterations, so trusting those
timestamps allowed tamper tests to fail. This version retains complete
integrity checks and moves costly work off the user request path. The
foregoing numbers require offline regression and live deployed validation
before the entire voice app can be called responsive. Model choice and
quantization are unchanged.
