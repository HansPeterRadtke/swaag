# SWAAG manual

SWAAG is a local-first autonomous agent runtime. Append-only session history and durable worker/event stores are authoritative; user interfaces and protocol adapters are projections over that state.

## Architecture

SWAAG has three capability layers. Layer 1 is the agent harness: model calls, exact context compilation, durable history, inference admission, cancellation/replay, worker lifecycle, completion evaluation, and transport-independent task state. Layer 2 contains repository-owned system capabilities such as history, notes, prompt instructions, filesystem/workspace access, shell/process control, attachments, artifacts, wakeups, and shared state. Layer 3 is open-ended external capability such as MCP servers, browser automation, OCR, databases, proprietary APIs, and document converters. Layer 3 is optional and never becomes core merely because it is useful.

A worker remains a sequential agent loop. Multiple workers may run independently while inference capacity is centrally scheduled. Communication is a separate control path so a busy worker does not have to stop merely because the user asks a status question.

## Installation and validation

For development, create a virtual environment and install the package in editable mode. Production service environments should use `scripts/install-runtime-env.sh`, which installs the pinned runtime under `/data/var/swaag` rather than depending on a user home directory.

Use `python -m swaag --help` for the main CLI, `python -m swaag.benchmark --help` for benchmarks, and `python -m swaag doctor` to check the configured model endpoint and constrained-output path. The deterministic regression suite is `python -m pytest -q` from an environment containing the project test dependencies.

## Main model

Configure the worker model under `[model]`. `base_url` selects the live endpoint. SWAAG supports llama.cpp and capability-discovered OpenAI-compatible backends through a neutral model-client interface. Context capacity, serialization, structured-output support, tokenizer behavior, and model identity are discovered where available; explicit fallbacks are recorded rather than silently guessed.

The packaged endpoint is a development default. Override it in deployment configuration. Keep `model_identity` empty when the backend can be fingerprinted. Set a remote context fallback only when the deployed limit is actually known.

## System and external tools

`[tools].enabled` controls Layer 2 capability exposure. Disabling a system capability also disables its automatic context contribution; disabled notes or prompt instructions are not silently injected by Layer 1. Filesystem roots, stateful behavior, side effects, and shell privileges remain mechanical policy.

External tools are configured separately. MCP and other external catalogs are schema-discovered and semantically selected, but missing optional Layer 3 systems do not make the core unhealthy. Shell can be a host-dependent fallback when permitted without turning invoked programs into SWAAG dependencies.

## Conversation and sessions

`python -m swaag ask` runs one agent turn and `python -m swaag chat` provides an interactive shell. Sessions are durable and can be inspected with the session, state, history, notes, and reader commands. Controls sent to active work are appended durably and reconciled semantically by the next model call rather than interpreted by keyword rules.

## Durable workers and Task API

`TaskApi` is the transport-neutral programmatic worker boundary. Clients can create/start a worker, inspect it, send a message or redirect, cancel it, resume an input-required worker, archive terminal work, add/list attachments, and consume durable events with bounded cursors. The worker `result` remains the universal conversational output. Structured output and response presentations are optional augmentations.

Long work is resumable because state does not live in a socket. Retain worker/session identity and the latest event cursor. `events.wait` provides bounded long polling; AG-UI, A2A, and other adapters project the same durable lifecycle.

## Response presentations

A task may request `visual`, `audio`, or both presentation modes. The canonical verified worker result remains authoritative. Visual relevance selection and audio rendering are separate model operations with independent constrained evaluation. Audio rendering produces listenable prose and uses context-sensitive numeric precision: irrelevant machine precision may be simplified, while thresholds, identifiers, paths, code, timestamps, versions, and other meaningful exact values remain exact.

When TTS is used, prefer the verified `audio` presentation over speaking the raw worker result. Keep the canonical result for history and inspection.

## Two models: background worker plus communication model

SWAAG already supports a separately configured communication model. The main worker uses `[model]`; set `[communication].model_base_url` to a second, normally smaller and faster endpoint. The communication runtime uses a restricted communication-tool set and separately budgeted durable worker evidence plus deterministic runtime state. It can answer cheap status/history questions without making the busy worker service them synchronously. If stronger interpretation is required, it can semantically request escalation; SWAAG sends the unchanged evidence to the main model and records the escalation provenance.

This is not two independent task-mutating agents. The worker remains the task owner. The communication path observes durable state and handles communication. A user instruction that changes the task is persisted into the worker control path, where the worker incorporates it at a safe execution boundary.

True simultaneous inference requires actual backend capacity. Two URLs that ultimately share one single model slot do not create concurrency. Use separate model servers/devices or a backend with enough parallel slots. SWAAG's scheduler gives communication/control traffic higher mechanical priority and uses aging to avoid worker starvation.

The communication service is disabled by default. Enable `[communication].enabled`, set `model_base_url` when using a second model, and start `python -m swaag communication serve`. The raw listener is loopback-only by design. Non-local clients should use an authenticated TLS gateway or supported protected adapter rather than exposing it directly.

## Voice and Android

Voice is a client/communication layer, not another SWAAG core. Keep microphone capture, VAD, speech-to-text, wake-word or push-to-talk behavior, text-to-speech, playback, and immediate local barge-in in the Android application or another external voice service.

Send finalized recognized utterances to SWAAG as authoritative user text. Do not continuously append unstable partial STT hypotheses to durable history. Keep one durable worker/session identity and event cursor across reconnects. Status questions can use the separate communication model while the main worker continues background work. Task-changing speech must be forwarded as a durable worker message/control. A literal stop request should use cancellation.

When the user starts speaking over TTS, stop playback immediately on the device; do not wait for a server round trip. Once STT finalizes the utterance, send it to SWAAG. SWAAG already supports inference preemption, stale-action invalidation, exact context reconstruction/replay, and durable continuation.

For final spoken answers, request `presentation_modes=["audio"]` and send the verified audio text to TTS. Ordinary heartbeat, tool execution, queue state, and internal progress should normally stay visual or silent. Speak blocking questions, important failures, requested status answers, and final user-facing answers.

The worker/Task API already exposes terminal presentations. The simpler communication submit/status path is primarily control/status and returns ordinary communication text; it is not the presentation-aware task response surface. A production voice gateway can therefore use Task API or AG-UI for durable task turns and event streaming while using the communication model for concurrent status conversation.

## Interruption, cancellation, and status

A new worker message is persisted before inference preemption. A model action compiled from an older control snapshot cannot execute after a newer control arrives. Completed tool evidence is retained and unstarted calls are abandoned. Controls arriving during completion evaluation, response presentation, structured output, or terminal commit keep the candidate provisional and start another run cycle.

Cancellation is a distinct mechanical operation. Backends that cannot truly suspend transformer state are represented honestly: SWAAG cancels the active request, reconstructs exact context, and replays when appropriate rather than claiming bit-for-bit suspension/resume.

Status combines deterministic liveness with semantic interpretation. Mechanical state, heartbeat, active operation, processes, waits, controls, and wakeups remain readable even when semantic status generation fails.

## Protocol adapters

`python -m swaag communication serve` hosts the loopback communication/control service. AG-UI projects rich run/event/state/client-tool behavior. A2A projects durable external task interoperability. The Open WebUI Pipe maps stable chat/message identities onto workers and cursor-based events. MCP is a capability protocol, not the task protocol. See `docs/task-api.md` for the precise supported subsets and limitations.

## Files and attachments

Attachments enter as raw content-addressed bytes plus metadata and provenance. SWAAG does not automatically OCR, transcribe, convert, or semantically inspect every upload. Interpretation belongs to selected Layer 3 capabilities or explicitly enabled shell tooling. This preserves original evidence and avoids hidden preprocessing becoming another authority.

## Persistence and recovery

Session history is append-only and hash-linked. Worker lifecycle and protocol projections retain integrity-linked references to canonical events. Context compaction never replaces authoritative history. Exact sources remain recoverable after projections, repeated compaction, and process restart. Wakeups, controls, questions, failures, shared state, tool results, attachments, and prompt-instruction provenance survive restart through their owning stores and event links.

## Observability

Operational telemetry is not semantic history. OpenTelemetry traces and metrics cover model operations, inference admission, tools, context budgets, reductions, protocol requests, workers, and durable correlation IDs without replacing append-only SWAAG history. The communication service can export OTLP when standard environment configuration is present.

## Security and deployment

The raw communication listener and MCP HTTP binding are loopback-oriented. Do not expose them directly to an untrusted network. Use TLS termination and authentication at the deployment boundary. A2A has optional bearer protection and requires an HTTPS public base when enabled. Tool permissions, filesystem roots, side effects, shell privileges, external credentials, and attachment fetch behavior must be explicit.

## Troubleshooting

If the model is unreachable, use `doctor` and verify endpoint, health, identity, context capacity, and structured-output support. If a worker appears inactive, inspect mechanical status and event cursors before relying on semantic status text. If work is waiting for input, resume the exact open interrupt/worker. If an external capability is absent, verify its catalog/MCP configuration; optional Layer 3 failure is not core failure. If a voice client cannot converse while the worker is busy, verify that the communication model really uses an independent endpoint or that the shared backend has spare inference capacity.

## Reference map

Read `docs/design-principles.md` for the semantic/mechanical boundary, `docs/tool-architecture.md` for the three capability layers, `docs/context-management.md` for context budgeting and projection, `docs/task-api.md` for programmatic worker/protocol behavior, `docs/voice-and-communication.md` for voice and two-model deployment, and `docs/security-boundaries.md` for security. `docs/benchmark-methodology.md` and `docs/live-benchmark-results.md` describe validated evidence. `docs/TODO.md` records remaining partial or unvalidated work and must not be read as completed functionality.
