# Voice and concurrent communication

Voice integration is intentionally outside the SWAAG reasoning core. A voice client converts speech to finalized text, sends that text to the persistent user-facing orchestrator, receives the orchestrator response or durable worker results, and performs speech synthesis locally or in an external voice service.

## Orchestrator-first interaction

There is exactly one official SWAAG entry point for ordinary voice/chat user messages: orchestrator.message. This is not one option among several equivalent APIs. A voice frontend must not send ordinary speech to Task API worker creation/message operations, AG-UI worker submission, A2A task submission, or a worker session. Those are lower-level developer/integration interfaces.


Every finalized ordinary user utterance goes to orchestrator.message. The orchestrator is the conversation endpoint. It does not create a worker merely because the user spoke.

The first orchestrator stage is deliberately small and constrained. In one model call it semantically chooses between an immediate orchestrator reply and full orchestration. Greetings, acknowledgements, can-you-hear-me checks, and other lightweight interaction that needs no external tool, worker-state mutation, or substantial delegated work are answered immediately by the orchestrator itself. That path creates no worker and does not run the generic independent completion-evaluation stage.

Only an utterance that actually requires substantive execution, a task or worker state change, live worker inspection, or external or tool work enters the full orchestration planner. The distinction is model-owned semantic judgment, not a keyword router.

An explicitly worker-addressed control remains available through the Task API when a client intentionally targets an existing worker. That is a lower-level control operation, not the default route for ordinary human conversation.

## Recommended two-model topology

Use the model configuration for background worker execution and communication.model_base_url for the user-facing orchestrator and other lightweight communication operations when a second, normally smaller and faster model is available. The orchestrator owns user interaction and global task planning; workers own delegated task execution.

True background conversation requires independent inference capacity. Two URLs backed by one single model slot do not create concurrency. Use separate servers or devices or enough parallel backend slots. When the orchestrator shares a backend with a worker, user interaction has control priority and SWAAG uses cancellation plus exact replay rather than pretending transformer generation can always be suspended and resumed.

The specialized communication-status operation still exists for evidence-grounded interpretation of a specifically targeted worker snapshot. It is not the ordinary voice conversation endpoint.

## Android flow

Android should own microphone capture, VAD, STT, TTS, playback, local barge-in, reconnect behavior, and optional wake-word or push-to-talk UX. Send only finalized STT utterances to orchestrator.message. Partial hypotheses may be displayed locally but should not become authoritative SWAAG turns.

Keep the persistent orchestrator conversation identity across reconnects. When the orchestrator delegates work, retain the resulting plan and worker identities and event cursors for progress and final-result delivery. Do not create a worker for every voice utterance.

When TTS is playing and the user begins speaking, stop playback locally immediately. After STT finalizes the new utterance, send it to the orchestrator. SWAAG can preempt active shared-backend inference when necessary, reject stale actions, reconstruct exact context, and continue workers from durable history.

A literal user request to cancel or change ongoing work is still sent to the orchestrator first. The orchestrator applies the requested plan or worker change. Direct worker cancellation and control APIs remain available for clients that intentionally expose lower-level controls.

## Spoken output

Immediate orchestrator replies are already user-facing conversational text and should be spoken directly. For final background-task responses that require a retained audio-specific transformation, request an audio presentation; the canonical worker result remains authoritative and the audio variant is separately compiled and evaluated.

Do not narrate every mechanical event. Keep ordinary heartbeat, tool execution, queue state, and internal progress visual or silent. Speak blocking questions, important failures, explicitly requested status answers, direct orchestrator conversation, and final answers.

## Current implementation boundary

The loopback communication listener exposes orchestrator.message as the ordinary user-conversation operation. Its fast semantic gate answers lightweight interaction in one constrained model call or routes substantive work into the full orchestration planner.

TaskApi remains the transport-neutral lower-level worker lifecycle boundary: create and start, message or redirect, cancellation, resume, event cursors, status evidence, attachments, and terminal response presentations. The separate status operation remains available for targeted worker status or history questions. These interfaces share durable state but have different responsibilities.

## Minimal deployment

Run the main worker model under model.base_url. When possible, run a second fast model and set communication.model_base_url; the persistent orchestrator then uses that endpoint while workers use the main endpoint. Enable the communication service and keep its listener on loopback behind an authenticated gateway when Android is remote.

The Android or voice gateway sends every finalized ordinary utterance to orchestrator.message, not to a freshly created worker and not directly into an arbitrary active worker. Background worker and event APIs are used after delegation for durable progress, cancellation, and final results.

Benchmark actual simultaneous latency with the deployed servers before relying on the topology. A second orchestrator or communication model improves responsiveness only when the hardware or backend can actually execute it independently of worker inference.
