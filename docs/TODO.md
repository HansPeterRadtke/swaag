# Swaag implementation TODO

This is the gap list between the current repository and the intended architecture in `docs/design-principles.md` and `docs/context-management.md`. It is an implementation checklist, not a list of ideas. Items marked **partial** already have useful foundations in the repository but still differ materially from the target semantics.

## P0 - context compilation and semantic ownership

- [x] **Replace message-count-triggered history compression with budget-driven context pressure.** Compression is now triggered by actual next-call overflow rather than message count.
- [x] **Remove fixed character sizing as an architectural control for runtime summaries.** Runtime summary targets are token-budget-driven. Benchmark-only character checks may remain only where they measure an output property rather than model context capacity.
- [x] **Create an explicit context-compiler abstraction for production worker, summary, tool-result-projection, and model-backed capability calls.** Future semantic operations must use the same compiler as they are added; direct capability-owned model clients are forbidden.
- [ ] **Make the context compiler account for every serialized component explicitly.** **Partial.** Current budgeting handles output reserve, safety, fixed overhead, schemas, prompt token counts, individual completion-evidence results, and attachment references as named components. Extend/prove accounting for retrieved/web material and every future protocol or semantic-operation component with provenance.
- [x] **Persist per-call context accounting/provenance for current production LLM operations.** Extend the same record to every future semantic operation and benchmark export.
- [ ] **Make all semantic context inclusion/exclusion decisions LLM-owned.** **Partial.** Current history compression is model-authored, but deterministic orchestration still decides some inclusion based on fixed policies. Audit every place that drops, clips, selects, prioritizes, or summarizes semantic information and move meaning-dependent choices behind an LLM decision with hard numerical constraints enforced afterward.
- [ ] **Never silently truncate semantic material to recover context space.** Audit browser output trimming, tool-output clipping, history windows, prompt assembly, error text, file reads, benchmark adapters, and model/provider wrappers. Where truncation is only a transport/display safeguard, preserve a durable raw reference; where it affects model knowledge, use semantic reduction or explicit bounded retrieval.
- [x] **Make overflow recovery iterative and semantic for current production operations.** Action and independent completion-evaluation calls calculate measured overflow, semantically project the largest reducible raw tool result, rebuild, and re-tokenize for bounded rounds while retaining raw provenance. History analysis, history compaction, and ordinary action tool-result projection hierarchically reduce an exact source that cannot fit a reducer call: deterministic segmentation drops no fragment, while every reduction and recombination is model-authored. Every future semantic operation must use the same measured, iterative pattern.
- [x] **Make output reserve operation-specific and semantically appropriate.** The compiler separates per-call operation minimums from per-call desired headroom, preserves full-fidelity input that fits the minimum, and reconstructs every production structured semantic call after backend-reported output starvation. Dynamic reducer targets set desired output directly instead of inheriting a static ratio; if an expanded hard minimum causes overflow, evidence-bearing operations return to bounded semantic reduction.
- [ ] **Benchmark context ordering and positional retrieval per model and operation.** The August 26 recording explicitly requires experiments for lost-in-the-middle and ordering effects. Compare system/task instructions, current request, history, tool schemas, tool results, and retrieved evidence at different positions and context utilizations; do not hard-code a universal order until measured.
- [ ] **Eliminate any mismatch between model-reported context capacity and actual request serialization.** **Partial.** Live llama.cpp compilation now takes authoritative per-slot `n_ctx` from `/props`, while fakes/offline replay use an explicit configured fallback, and exact counting uses fixed rather than proportional safety. Add OpenAI-compatible discovery and backend/version verification for grammar/chat-template overhead and model switches.

## P0 - durable history, projections, and recoverability

- [ ] **Treat append-only history as authoritative and every prompt history as a derived projection.** **Partial.** Swaag already has append-only event history, replay, archive/search, and embedding machinery. Make this separation explicit in the context compiler and prevent compacted summaries from becoming accidental authoritative state.
- [ ] **Track provenance from every summary/projection back to exact source events.** **Partial.** History summaries and tool-result projections now retain exact event hashes, session-aware ranges, transitive source references, and their own projection-event identity across replay. Apply the same lineage contract to future derived views such as research and attachment projections.
- [ ] **Support dynamically sized history projections per call.** **Partial.** `history_analyze` and independent communication-status calls compile exact evidence for their specific question and compute projection targets only from that call's measured overflow, preserving exact source hashes. General routing, worker, and older completion evidence still need independent semantic selection and sizing; do not reuse one summary merely because it already exists.
- [x] **Support re-expansion from raw history after compaction.** Summary prompts identify exact session-aware source ranges and direct the worker to history-window retrieval. Search/window results preserve exact event hashes, including immutable archive shards, and their lineage survives subsequent tool-result projection and summary compaction.
- [ ] **Benchmark hierarchical summaries versus targeted retrieval versus recent-verbatim-plus-summary projections.** No single strategy should be assumed universally best.
- [ ] **Add compaction-preservation evals for dates, identifiers, user constraints, negative constraints, causality, unresolved questions, promises, file paths/references, tool outcomes, and task completion state.**
- [ ] **Test repeated compaction over very long tasks for semantic drift.** Include cases where an early detail becomes important much later.

## P0 - tool definitions and tool results as context-budgeted data

- [x] **Stop injecting the complete enabled-tool registry into every worker context.** Production defaults now expose a compact capability index plus semantic staged schema loading.
- [x] **Add a compact capability index separate from full tool schemas.** The LLM loads exact schemas only for semantically selected capabilities.
- [ ] **Keep tool selection semantic.** Do not create deterministic keyword-to-tool routing or file-type-to-tool routing as a replacement for LLM judgment.
- [x] **Store raw tool results durably and semantically project oversized observations under measured context pressure.** Raw canonical events remain authoritative, durable projections can be reused only when their source hash and token target match, and exact older/archive retrieval references survive re-projection.
- [x] **Give tool-result reducers overarching task context.** The projection operation receives the original user goal, raw result, source provenance, and a token target.
- [x] **Add explicit token accounting per loaded tool schema and individual tool-result prompt component.** Export these measurements in benchmark reporting next.
- [ ] **Benchmark generic shell versus bespoke structured tools per model.** **Partial.** A reproducible live harness now alternates strategy order over identical inspection/calculation and exact-edit tasks, disables replay, checkpoints every case, retains full sessions/workspaces, and verifies exact external effects. Run and record it against the Jetson model after the currently active context-order experiment releases the single inference slot; do not infer a general tool policy from one model or a tiny sample.
- [ ] **Keep MCP as a replaceable capability adapter, not the core agent protocol.** **Partial.** `mcp.py` follows the 2026-07-28 request metadata, discovery, list-cache, deterministic-listing, validation-error, and execution-error contracts without leaking MCP assumptions into context/history/lifecycle architecture. Add upstream SDK/schema conformance coverage as the draft evolves.

## P0 - research, uncertainty, and autonomous continuation

- [ ] **Make research/tool-based uncertainty resolution a first-class worker behavior.** **Partial.** Browsing capability exists. The system prompt/runtime should explicitly encourage the LLM to resolve missing or stale information via local search, documentation, web research, experiments, or other tools when useful instead of immediately asking the user.
- [ ] **Keep the decision to research semantic.** Do not trigger web search from keywords, fixed retry counts, or deterministic topic classes.
- [ ] **Record research provenance and preserve raw evidence.** Later calls should receive budgeted semantic projections while retaining links/references to original evidence.
- [x] **Implement semantic question criticality.** Every model-authored question carries `blocking` or `optional`, a semantic reason, and an explicit provisional assumption; deterministic code only enforces the declared lifecycle effect.
- [x] **Allow optional questions to remain pending while useful work continues.** Optional questions are durable events, do not force input-required state, and may accompany useful tool work. A later answer can resume/redirect even a provisionally completed worker without losing history.
- [ ] **Benchmark ambiguity handling and unnecessary clarification.** Include AskBench-like cases, recoverable uncertainty, wrong-premise cases, and cases where tools can answer the question without user interruption.
- [ ] **Make persistence an explicit evaluated behavior.** The worker should inspect, test, verify, research, and improve rather than stopping at the first plausible output.
- [ ] **Support intentionally open-ended/improvement-until-interrupted tasks.** These need explicit lifecycle state rather than pretending every task has a natural final answer.

## P0 - completion semantics

- [x] **Do not equate a model `final` action with verified task completion.** A separate constrained semantic evaluation decides whether a final candidate satisfies the original objective; rejection returns the worker to the action loop.
- [ ] **Add an independent completion-evaluation LLM operation for substantial tasks.** **Partial.** The operation receives the objective, final candidate, semantic status, and current-turn raw tool/error evidence with exact event provenance. It preserves full fidelity when the request fits and iteratively projects only measured overflow. Add dynamically selected older-history, artifact, research, and attachment evidence rather than assuming the current turn is sufficient.
- [x] **Allow completion evaluation to request more work rather than merely score the result.** Its schema returns actionable remaining work, which is durably recorded and fed back into the worker loop.
- [x] **Use deterministic verifiers as evidence, not semantic substitutes.** Tool outcomes and test evidence constrain the independent LLM decision; deterministic runtime code does not synthesize semantic completion.
- [ ] **Benchmark premature completion and over-working separately.** Add long-horizon tasks where the first plausible answer is intentionally insufficient and tasks where continuing after success is wasteful.

## P1 - worker lifecycle, multi-tasking, and communication

- [x] **Implement multiple independently addressable worker instances.** Each durable worker has its own task identity, session/history/projection state, mechanical lifecycle, event cursor, and control/cancellation path while retaining the simple sequential inner agent loop.
- [x] **Add worker lifecycle operations: create/start, inspect, message/redirect, cancel/stop, resume/reconstruct, and archive.** Cancellation is durable and stops active inference; archive retains the actual terminal state rather than inventing a new execution state. A true suspended backend state is intentionally not claimed as pause.
- [ ] **Separate user-facing communication from busy worker execution.** **Partial.** The long-running communication service owns a bounded multi-worker executor and transport-neutral task API, can redirect/cancel active inference, and uses an independent compiled operation session for status interpretation. A configured separate communication model avoids preempting the worker; same-model operation uses explicit cancel-and-exact-replay. Benchmark priority/latency under real long inference and tools and extend this separation beyond status.
- [ ] **Implement a communication agent with explicit escalation.** A smaller/faster model may handle cheap status or routing only after benchmark evidence; difficult semantic requests must escalate to a stronger model.
- [x] **Let the communication agent read durable worker history and deterministic runtime state without receiving entire worker contexts.** Status questions use a dedicated operation session and contract, exact target-event snapshot plus deterministic mechanical state, central context compilation, measured overflow, hierarchical semantic projection, exact source references, and bounded validation repair. The operation never writes target worker history, avoiding stale-writer races.
- [ ] **Support user interruption at arbitrary worker stages.** New instructions should enter durable history/control state and influence the next semantically valid continuation.
- [ ] **Define conflict semantics for instructions arriving while inference/tool execution is active.** Decide with LLM assistance where semantic reconciliation is required; keep delivery/order mechanics deterministic.

## P1 - inference scheduling, cancellation, and preemption

- [ ] **Turn cooperative model preemption into a backend-independent request lifecycle.** **Partial.** Worker/task states and cancellation requests are durable; active inference observes cancellation and records terminal preemption evidence, while redirects invalidate stale requests. Add a backend-neutral inference request table for queued/completed/failed/superseded calls and queue priority.
- [ ] **Do not claim true suspend/resume unless a backend proves it.** For llama.cpp/OpenAI-compatible backends, distinguish cancellation plus replay/reconstruction from actual continuation of the same generation state.
- [ ] **Benchmark live llama.cpp cancellation behavior on the deployed Jetson version.** The previous direct experiment could not run because the endpoint was unavailable.
- [ ] **Benchmark llama.cpp parallel slots/continuous batching versus vLLM scheduling for the actual Jetson workload.** Include latency to interactive communication requests while long worker generations are active.
- [ ] **Add priority scheduling for communication/control traffic without starving worker progress.**
- [ ] **Account for cancelled/replayed inference in history, token/cost metrics, and completion reasoning.**

## P1 - status, questions, heartbeat, and failure reporting

- [ ] **Define a deterministic runtime state machine for worker activity.** **Partial.** Durable worker states now distinguish created, queued, working, input-required, cancellation-requested, canceled, completed, and failed; heartbeat phases cover the inner run. Scheduled wait and every tool/evaluation substate still need unified task-state projection.
- [ ] **Generate semantic status with an LLM from a separately budgeted status context.** **Partial.** The dedicated status operation now preserves the complete event/runtime-semantic snapshot when it fits, projects only after measured overflow, cites mechanically validated event sequences, and records its own durable lifecycle. Benchmark prompt/context alternatives and hallucinated-completion behavior before closing this item.
- [x] **Support status importance/criticality as semantic output.** The LLM emits the semantic level under a closed contract; deterministic code validates it and assigns the sortable rank while timestamps and runtime state remain mechanical.
- [x] **Implement heartbeat independent of semantic status generation.** Status calls have their own operation session and heartbeat; failure tests prove the target worker active-run heartbeat is unchanged, and status failures remain durable in the operation history.
- [x] **Integrate external process supervision/watchdog for production on Jetson.** The repo-backed communication service emits READY/WATCHDOG/STOPPING notifications and the deployed infra unit uses systemd watchdog supervision.
- [ ] **Guarantee durable critical failure delivery.** Rich side-channel events are optional; a critical failure must also have a representation every client can retrieve later.
- [x] **Guarantee durable critical-question delivery.** Blocking questions are append-only `agent_question` events, place workers in durable `input_required`, and remain in the normal conversational result even without a rich live UI.
- [x] **Make silent inactivity diagnosable.** Worker inspection exposes the last durable mechanical transition, active run/request/tool identity, start/update/heartbeat timestamps and heartbeat age, run-process liveness, and local executor supervision state without an LLM. A periodic mechanical pulse keeps liveness current even while a backend is still evaluating a prompt and has not streamed its first token.

## P1 - interfaces and structured output

- [x] **Define a transport-independent internal task/event API.** `TaskApi`, `WorkerStore`, and `WorkerManager` own the internal command/query/event model; the communication TCP service is one transport and MCP remains a capability boundary.
- [ ] **Add an AG-UI adapter for rich streaming UI events.** **Partial.** The communication transport exposes cursor-paginated `ag_ui.events` over durable worker events and maps run/result/failure/cancellation/input-required events to current AG-UI run, text, activity, custom, stable run-ID, success-outcome, and typed interrupt shapes. Add push subscription and map canonical inner tool/state events rather than only worker lifecycle events.
- [ ] **Evaluate/add an A2A adapter for durable external task semantics.** **Partial.** The communication transport exposes `a2a.get`; its A2A 1.0 projection maps submitted/working/input-required/completed/failed/canceled and artifacts without letting A2A own internal state. Add send/cancel protocol operations, subscription, authentication, and upstream conformance tests before claiming an A2A server.
- [ ] **Add an Open WebUI adapter.** **Partial.** The communication transport exposes `open_webui.get`, using persistence-safe `status` events and the normal return channel for final or critical input-required text. Add the actual Pipe/tool integration plus file/source mapping.
- [x] **Keep durable fallback semantics independent of live WebSocket UI connections.** All protocol views are rebuilt from canonical worker state/events using a cursor; final results and critical input remain in durable task state and the common conversational result.
- [x] **Support caller-defined structured output schemas.** `TaskApi.create` accepts a portable closed JSON schema and explicit top-level mechanical bindings. Semantic fields are generated in a separately compiled, schema-constrained, cancellable LLM call; IDs, lifecycle state, timestamps, objective, and run count bindings are filled and validated deterministically. The merged output is durable in the terminal worker event and returned by worker inspection.
- [x] **Keep normal conversation as the universal common-denominator output.** Worker `result` remains the complete conversational answer; caller-defined structured output is an optional durable augmentation rather than a replacement channel.

## P1 - attachment and file handling

- [x] **Add first-class raw attachment references to session/task state.** Content-addressed raw bytes survive disposable session projections and archival; exact attachment events carry stable ID/name/type/size/hash/source lineage through replay, worker inspection, and the task API.
- [x] **Do only cheap deterministic file facts automatically.** Upload records bytes, size, name, hash, and a MIME guess but never reads or converts content into the model context.
- [x] **Let an LLM decide whether a task requires content analysis and which capability to use.** Context contains references only; `read_attachment` and `extract_attachment` are staged capabilities selected through the normal semantic tool loop.
- [ ] **Integrate all2text/Docling/image/OCR/speech/specialized readers as selectable capabilities rather than mandatory ingestion stages.** **Partial.** The configurable `extract_attachment` adapter invokes all2text only when selected, retains its exact auditable manifest and complete output as separate integrity-checked artifacts, and was exercised against the real local all2text command. all2text can route optional providers, but dedicated multimodal/Docling/speech capabilities and availability reporting still need implementation and evaluation.
- [ ] **Budget extracted content like any other large tool result.** **Partial.** Bounded previews retain exact full artifacts and raw-attachment lineage, normal measured tool-result overflow projection applies, and extraction text/manifests remain readable from read-only artifact storage after session archival. Add an attachment-specific semantic projection/re-expansion benchmark before closing this item.
- [ ] **Add attachment tasks that require no inspection, partial inspection, and deep multimodal inspection to benchmarks.**

## P1 - observability and logging

- [ ] **Map operational traces/metrics to OpenTelemetry GenAI semantic conventions.** **Partial.** In-process agent invocation, logical model calls across retries, agent-side tools, backend/exact token usage, duration, errors, and cancellation/preemption now use current GenAI spans/metrics without capturing semantic content. Add backend queue/slot measurements and exporter deployment before closing this item.
- [x] **Keep OpenTelemetry separate from authoritative execution history.** OpenTelemetry uses only the standard API and can be sampled or disabled independently. Append-only history remains the replay authority; correlation uses shared session/run/model-call/tool-call identifiers rather than treating spans as state.
- [x] **Expose context-budget metrics.** Every compilation records context capacity/source, exactness, fit, input/output reserve, safety, required/overflow tokens, and bounded component categories as OTel metrics plus a correlated span event. Semantic-reduction attempts expose call kind, hierarchical use, and dynamic target size; successful durable history compactions expose frequency and hierarchical strategy without semantic content.
- [ ] **Expose inference scheduler metrics.** Queue depth, queue wait, active generation count, priority, cancellation latency, and backend slot utilization.
- [ ] **Add correlation identifiers connecting UI/task events, worker history, model calls, tool calls, and traces.** **Partial.** Spans carry durable conversation/session and active run IDs; model/tool spans and their canonical history events share explicit call IDs; worker/task projections already expose the session relationship. Propagate standard trace context through external protocol adapters and inference backends.

## P1 - prompts and evaluation

- [x] **Version every behaviorally important prompt as an implementation artifact.** Every prompt assembly carries SHA-256 identities for the prompt protocol, rendered system instruction, and each canonical template file it uses. Durable `prompt_built` events store those artifact identities plus the exact rendered-prompt hash, covering worker/action, inline semantic capabilities, summary, projection, completion, caller-schema, and communication/status operations without relying on package version alone.
- [ ] **Add systematic prompt-variant evaluation with held-out tasks.** Do not merge prompt changes based only on qualitative inspection.
- [ ] **Import/adapt instruction-following evals inspired by FollowBench and AgentIF.** Cover many simultaneous user constraints and long tool-heavy prompts.
- [ ] **Add over-refusal evals inspired by OR-Bench.** Measure unnecessary refusal separately from unsafe compliance.
- [ ] **Add research-behavior evals.** Test stale facts, missing local information, cases requiring web research, and cases where browsing would be wasteful.
- [ ] **Add context-engineering evals.** Measure whether the correct facts survive projection under tight budgets and whether irrelevant material is excluded without semantic loss.
- [ ] **Add communication/status evals.** Test whether a separate context can correctly explain worker progress without hallucinating completion.
- [ ] **Add model-routing evals before relying on a small communication model.** Measure escalation recall, end-to-end trajectory success, latency, and cost rather than only one-step classification accuracy.

## P2 - architecture cleanup after P0/P1 semantics exist

- [ ] **Remove obsolete configuration knobs whose semantics are superseded by per-call context compilation.** Do this only after migration/tests; candidates include fixed message-count compression and fixed summary-character controls.
- [ ] **Document each remaining deterministic policy and why it is non-semantic.** Anything that decides meaning belongs under suspicion and should be justified or moved to an LLM.
- [ ] **Ensure benchmark adapters use the same production context compiler and semantic policies.** Avoid benchmark-only shortcuts that hide real overflow/compaction behavior.
- [ ] **Make context/accounting replayable for recorded model tests.** A recorded trajectory should show exactly why each call saw its context and be usable for regression testing without live inference.
- [ ] **Add migration/versioning for durable event/history schemas before multi-worker and richer event protocols expand them.**
- [ ] **Review security/permission boundaries after dynamic tool discovery and attachments.** Semantic tool choice must never bypass deterministic authorization and workspace/path controls.

## Already present foundations that should not be reimplemented

These are not TODOs themselves; future work should extend them rather than creating parallel replacements:

- Exact/tokenizer-backed budgeting primitives and structured-output token floors in `budgeting.py`.
- Model-authored summary generation and summary budget checks in the runtime.
- Append-only event history, replay/integrity checks, archive/search, history tools, and embedding-index infrastructure.
- Central tool registry and constrained tool/action schemas.
- Browser automation capability.
- Communication store/service primitives.
- Cooperative model-call preemption primitives.
- Durable scheduled wakeups.
- Benchmark and verifier infrastructure.
- Existing deterministic permission/path/tool execution boundaries.

## Definition of done for this TODO

This file is complete only when Swaag's normal execution path embodies the design principles rather than merely documenting them: every LLM call is context-compiled with exact accounting and explicit output reserve; every semantic inclusion/reduction/routing/completion decision is LLM-owned; raw state remains recoverable; workers remain controllable and observable during long work; research and persistence are evaluated behaviors; and adapters/telemetry do not become the internal architecture.
