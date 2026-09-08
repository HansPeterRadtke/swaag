# Voice and concurrent communication

Voice integration is intentionally outside the SWAAG agent core. A voice client converts speech to finalized text, sends that text into durable task/communication interfaces, receives canonical results plus optional verified audio presentations, and performs speech synthesis locally or in an external voice service.

## Recommended two-model topology

Use `[model]` for worker execution and `[communication].model_base_url` for a second, normally smaller and faster model. The main worker remains the sole owner of task execution. The communication model reads separately budgeted durable worker evidence and deterministic runtime state for cheap status/history conversation and can semantically escalate unchanged evidence to the main model when stronger reasoning is needed.

True background conversation requires independent inference capacity. Two URLs backed by one single model slot do not create concurrency. Use separate servers/devices or enough parallel backend slots. The scheduler gives communication/control traffic higher mechanical priority and uses aging to prevent worker starvation.

The communication model is not a second task-mutating agent. Task-changing instructions are persisted as worker controls/messages. The worker reconciles them semantically with its objective and evidence. Cancellation remains a separate mechanical operation.

## Android flow

Android should own microphone capture, VAD, STT, TTS, playback, local barge-in, reconnect behavior, and optional wake-word/push-to-talk UX. Send only finalized STT utterances into durable SWAAG history. Partial hypotheses may be displayed locally but should not become authoritative turns.

Keep worker/session identity and the event cursor across reconnects. Use Task API, AG-UI, or another durable adapter for worker lifecycle/events. Use the separate communication path for status questions that should not interrupt background reasoning. Forward task-changing speech as a worker message/control. Use cancellation for a literal stop request.

When TTS is playing and the user begins speaking, stop playback locally immediately. After STT finalizes the new utterance, send it to SWAAG. SWAAG can preempt active inference when necessary, reject stale actions, reconstruct exact context, and continue from durable history.

## Spoken output

Request `presentation_modes=["audio"]` for final task responses that will be spoken. The canonical worker `result` remains authoritative; `audio` is a separately compiled and independently evaluated listenable rendering with context-sensitive numeric precision.

Do not narrate every mechanical event. Keep ordinary heartbeat, tool execution, queue state, and internal progress visual or silent. Speak blocking questions, important failures, explicitly requested status answers, and final answers.

## Current implementation boundary

The worker API already supports create/start, message/redirect, cancellation, resume, event cursors, status evidence, attachments, and terminal response presentations. The communication service already supports a separately configured model for status interpretation and semantic escalation. It is disabled by default and its raw listener is loopback-only.

The simpler communication submit/status path returns ordinary communication text and is not the presentation-aware task response surface. A production Android gateway should therefore use the worker/task interface for task turns and verified audio presentations, while using the communication model for concurrent status conversation. A future unified voice endpoint should remain a thin adapter over these primitives rather than create another state authority or agent core.

## Minimal deployment

Run the main model server and configure `[model].base_url`. Run a second fast model server and set `[communication].model_base_url`. Set `[communication].enabled=true` and start `python -m swaag communication serve`. Keep the listener on loopback and place an authenticated TLS gateway or supported protected adapter between Android and the host. Create one durable worker/session, retain IDs and cursor on Android, request audio presentation for spoken final answers, and route status questions through the communication model.

Benchmark actual simultaneous latency with the deployed servers before relying on the topology. A second communication model improves responsiveness only when the hardware/backend can execute it while the worker model is busy.
