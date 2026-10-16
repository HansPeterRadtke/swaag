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

Compatibility limitation: this operation path admits its canonical user receipt under the session execution lock. It does not yet provide the native control-ID path's pre-lock authority ingress. The Android/server queue and semantic cancellation remain separate controls; do not claim that a queued voice correction already superseded an in-flight action. This boundary still needs qualification before full priority-guideline acceptance.

TaskApi remains the transport-neutral lower-level worker lifecycle boundary: create and start, message or redirect, cancellation, resume, event cursors, status evidence, attachments, and terminal response presentations. The separate status operation remains available for targeted worker status or history questions. These interfaces share durable state but have different responsibilities.

## Minimal deployment

Run the main worker model under model.base_url. Use communication.model_base_url only for an independently qualified configuration; the persistent orchestrator then uses that endpoint while workers use the main endpoint. Keep existing production workloads and Infra admission reserves intact. Enable the communication service and keep its listener on loopback behind an authenticated gateway when Android is remote.

The Android or voice gateway sends every finalized ordinary utterance to orchestrator.message, not to a freshly created worker and not directly into an arbitrary active worker. Background worker and event APIs are used after delegation for durable progress, cancellation, and final results.

Benchmark actual simultaneous latency with the deployed servers before relying on the topology. A second orchestrator or communication model improves responsiveness only when the hardware or backend can actually execute it independently of worker inference.
