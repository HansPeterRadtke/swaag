# Swaag implementation TODO

This is the gap list between the current repository and the intended architecture in `docs/design-principles.md` and `docs/context-management.md`. It is an implementation checklist, not a list of ideas. Items marked **partial** already have useful foundations in the repository but still differ materially from the target semantics.

## P0 - context compilation and semantic ownership

- [x] **Replace message-count-triggered history compression with budget-driven context pressure.** Compression is now triggered by actual next-call overflow rather than message count.
- [x] **Remove fixed character sizing as an architectural control for runtime summaries.** Runtime summary targets are token-budget-driven. Benchmark-only character checks may remain only where they measure an output property rather than model context capacity.
- [x] **Create an explicit context-compiler abstraction for production worker, summary, and tool-result-projection LLM calls.** Future semantic operations must use the same compiler as they are added.
- [ ] **Make the context compiler account for every serialized component explicitly.** **Partial.** Current budgeting handles output reserve, safety, fixed overhead, schemas, and prompt token counts. Extend/prove accounting for selected history, summaries, tool definitions, individual tool results, retrieved/web material, pending user/control input, attachments, and protocol framing as named components with provenance.
- [x] **Persist per-call context accounting/provenance for current production LLM operations.** Extend the same record to every future semantic operation and benchmark export.
- [ ] **Make all semantic context inclusion/exclusion decisions LLM-owned.** **Partial.** Current history compression is model-authored, but deterministic orchestration still decides some inclusion based on fixed policies. Audit every place that drops, clips, selects, prioritizes, or summarizes semantic information and move meaning-dependent choices behind an LLM decision with hard numerical constraints enforced afterward.
- [ ] **Never silently truncate semantic material to recover context space.** Audit browser output trimming, tool-output clipping, history windows, prompt assembly, error text, file reads, benchmark adapters, and model/provider wrappers. Where truncation is only a transport/display safeguard, preserve a durable raw reference; where it affects model knowledge, use semantic reduction or explicit bounded retrieval.
- [ ] **Make overflow recovery iterative and semantic.** If compilation still exceeds budget after one reduction, calculate the exact required reduction and invoke another semantic projection/reduction pass rather than failing early or applying a blind fallback.
- [ ] **Make output reserve operation-specific and semantically appropriate.** **Partial.** The compiler now separates operation minimums from desired headroom, preserves full-fidelity input that fits the minimum, and reconstructs action calls after backend-reported output starvation. Extend adaptive retry to every semantic operation and benchmark policy rather than freezing arbitrary ratios.
- [ ] **Benchmark context ordering and positional retrieval per model and operation.** The August 26 recording explicitly requires experiments for lost-in-the-middle and ordering effects. Compare system/task instructions, current request, history, tool schemas, tool results, and retrieved evidence at different positions and context utilizations; do not hard-code a universal order until measured.
- [ ] **Eliminate any mismatch between model-reported context capacity and actual request serialization.** **Partial.** Live llama.cpp compilation now takes authoritative per-slot `n_ctx` from `/props`, while fakes/offline replay use an explicit configured fallback, and exact counting uses fixed rather than proportional safety. Add OpenAI-compatible discovery and backend/version verification for grammar/chat-template overhead and model switches.

## P0 - durable history, projections, and recoverability

- [ ] **Treat append-only history as authoritative and every prompt history as a derived projection.** **Partial.** Swaag already has append-only event history, replay, archive/search, and embedding machinery. Make this separation explicit in the context compiler and prevent compacted summaries from becoming accidental authoritative state.
- [ ] **Track provenance from every summary/projection back to exact source events.** **Partial.** History summaries and tool-result projections now retain exact event hashes, session-aware ranges, transitive source references, and their own projection-event identity across replay. Apply the same lineage contract to future derived views such as research and attachment projections.
- [ ] **Support dynamically sized history projections per call.** The same session may need a tiny history projection for routing, a large one for worker reasoning, and a different one for status or completion evaluation. Do not reuse one summary merely because it already exists.
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
- [ ] **Benchmark generic shell versus bespoke structured tools per model.** The recordings explicitly require empirical testing of whether the local model can discover and use normal Linux commands reliably. The live Jetson llama.cpp test was blocked when the endpoint went down; perform and record it when the service is normally available rather than restarting infrastructure just for the test.
- [ ] **Keep MCP as a replaceable capability adapter, not the core agent protocol.** **Partial.** `mcp.py` exists. Verify it follows current MCP direction and does not leak MCP assumptions into context/history/lifecycle architecture.

## P0 - research, uncertainty, and autonomous continuation

- [ ] **Make research/tool-based uncertainty resolution a first-class worker behavior.** **Partial.** Browsing capability exists. The system prompt/runtime should explicitly encourage the LLM to resolve missing or stale information via local search, documentation, web research, experiments, or other tools when useful instead of immediately asking the user.
- [ ] **Keep the decision to research semantic.** Do not trigger web search from keywords, fixed retry counts, or deterministic topic classes.
- [ ] **Record research provenance and preserve raw evidence.** Later calls should receive budgeted semantic projections while retaining links/references to original evidence.
- [ ] **Implement semantic question criticality.** Distinguish blocking questions from optional questions where the worker can proceed with a provisional assumption.
- [ ] **Allow optional questions to remain pending while useful work continues.** A later user answer must be able to redirect/revise the worker without losing history.
- [ ] **Benchmark ambiguity handling and unnecessary clarification.** Include AskBench-like cases, recoverable uncertainty, wrong-premise cases, and cases where tools can answer the question without user interruption.
- [ ] **Make persistence an explicit evaluated behavior.** The worker should inspect, test, verify, research, and improve rather than stopping at the first plausible output.
- [ ] **Support intentionally open-ended/improvement-until-interrupted tasks.** These need explicit lifecycle state rather than pretending every task has a natural final answer.

## P0 - completion semantics

- [ ] **Do not equate a model `final` action with verified task completion.** **Current mismatch/partial.** The constrained action loop has a final action and deterministic validation, but target completion is semantic against the user's actual objective.
- [ ] **Add an independent completion-evaluation LLM operation for substantial tasks.** Feed it user requirements, relevant history projection, produced artifacts/results, deterministic test evidence, and remaining known issues.
- [ ] **Allow completion evaluation to request more work rather than merely score the result.** The worker should resume with explicit deficiencies.
- [ ] **Use deterministic verifiers as evidence, not semantic substitutes.** Tests, exit codes, schemas, file state, benchmarks, and measurements should constrain/evidence completion but cannot alone decide whether the user's semantic objective is fulfilled.
- [ ] **Benchmark premature completion and over-working separately.** Add long-horizon tasks where the first plausible answer is intentionally insufficient and tasks where continuing after success is wasteful.

## P1 - worker lifecycle, multi-tasking, and communication

- [x] **Implement multiple independently addressable worker instances.** Each durable worker has its own task identity, session/history/projection state, mechanical lifecycle, event cursor, and control/cancellation path while retaining the simple sequential inner agent loop.
- [x] **Add worker lifecycle operations: create/start, inspect, message/redirect, cancel/stop, resume/reconstruct, and archive.** Cancellation is durable and stops active inference; archive retains the actual terminal state rather than inventing a new execution state. A true suspended backend state is intentionally not claimed as pause.
- [ ] **Separate user-facing communication from busy worker execution.** **Partial.** The long-running communication service now owns a bounded multi-worker executor and transport-neutral task API, can redirect/cancel active inference, and remains available while worker threads run. Benchmark priority/latency under real long inference and tools, and separate the communication model/context fully.
- [ ] **Implement a communication agent with explicit escalation.** A smaller/faster model may handle cheap status or routing only after benchmark evidence; difficult semantic requests must escalate to a stronger model.
- [ ] **Let the communication agent read durable worker history and deterministic runtime state without receiving entire worker contexts.** Its own context must be compiled and budgeted like every other LLM call.
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
- [ ] **Generate semantic status with an LLM from a separately budgeted status context.** Include enough goal/current-step/recent-event context to explain meaning; benchmark alternatives rather than hard-coding one prompt.
- [ ] **Support status importance/criticality as semantic output.** Keep elapsed time and mechanical state deterministic.
- [ ] **Implement heartbeat independent of semantic status generation.** A wedged LLM/status call must not suppress mechanical liveness evidence.
- [ ] **Integrate external process supervision/watchdog for production on Jetson.** Use systemd watchdog semantics rather than only an in-process timer.
- [ ] **Guarantee durable critical failure delivery.** Rich side-channel events are optional; a critical failure must also have a representation every client can retrieve later.
- [ ] **Guarantee durable critical-question delivery.** Interactive UI prompts may be transient, so blocking questions need persisted task state/main-channel fallback.
- [ ] **Make silent inactivity diagnosable.** Record the last mechanical transition, active request/tool identifiers, start time, and supervisor health without needing an LLM to be alive.

## P1 - interfaces and structured output

- [x] **Define a transport-independent internal task/event API.** `TaskApi`, `WorkerStore`, and `WorkerManager` own the internal command/query/event model; the communication TCP service is one transport and MCP remains a capability boundary.
- [ ] **Add an AG-UI adapter for rich streaming UI events.** **Partial.** A tested projection maps durable run/result/failure/cancellation/input-required events to current AG-UI run, text, activity, and custom shapes. Add a streaming endpoint and map canonical tool/state events rather than only worker lifecycle events.
- [ ] **Evaluate/add an A2A adapter for durable external task semantics.** **Partial.** A tested A2A 1.0 projection maps submitted/working/input-required/completed/failed/canceled and artifacts without letting A2A own internal state. Add protocol operations, pagination/subscription, authentication, and conformance tests before claiming an A2A server.
- [ ] **Add an Open WebUI adapter.** **Partial.** A tested projection uses persistence-safe `status` events and the normal return channel for final or critical input-required text. Add the actual Pipe/tool integration plus file/source mapping.
- [ ] **Keep durable fallback semantics independent of live WebSocket UI connections.**
- [ ] **Support caller-defined structured output schemas.** LLM-generated semantic fields should use schema-constrained generation; deterministic runtime fields such as timings/state/IDs should be filled mechanically.
- [ ] **Keep normal conversation as the universal common-denominator output.** Rich structured fields/events augment it rather than making basic clients impossible.

## P1 - attachment and file handling

- [ ] **Add first-class raw attachment references to session/task state.** Preserve original bytes and stable metadata before semantic inspection.
- [ ] **Do only cheap deterministic file facts automatically.** Size, name, storage reference, and safely detected type are mechanical; what the file means or whether it should be inspected is semantic.
- [ ] **Let an LLM decide whether a task requires content analysis and which capability to use.** Do not automatically convert every file to text.
- [ ] **Integrate all2text/Docling/image/OCR/speech/specialized readers as selectable capabilities rather than mandatory ingestion stages.**
- [ ] **Budget extracted content like any other large tool result.** Preserve raw source and extracted representation outside context and project only what a call needs.
- [ ] **Add attachment tasks that require no inspection, partial inspection, and deep multimodal inspection to benchmarks.**

## P1 - observability and logging

- [ ] **Map operational traces/metrics to OpenTelemetry GenAI semantic conventions.** Cover agent invocation, model calls, tools, token usage, latency, errors, cancellation/preemption, and relevant queue measurements.
- [ ] **Keep OpenTelemetry separate from authoritative execution history.** Telemetry may be sampled/expired; semantic history must remain replayable according to Swaag retention rules.
- [ ] **Expose context-budget metrics.** Record requested context capacity, input/output reserve, component sizes, reduction count, compaction frequency, and overflow failures.
- [ ] **Expose inference scheduler metrics.** Queue depth, queue wait, active generation count, priority, cancellation latency, and backend slot utilization.
- [ ] **Add correlation identifiers connecting UI/task events, worker history, model calls, tool calls, and traces.**

## P1 - prompts and evaluation

- [ ] **Version every behaviorally important prompt as an implementation artifact.** Include worker/system, context-selection, summary, tool-result reduction, research, question-criticality, status, communication, and completion-evaluator prompts.
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
