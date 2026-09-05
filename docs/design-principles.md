# Swaag design principles

Swaag is an LLM harness. Its central responsibility is constructing the right bounded context for every LLM call. The LLM is the semantic engine; deterministic code surrounds it with exact resource accounting, execution, persistence, and verification.

## Fundamental boundary

Deterministic code owns facts that can be computed exactly or conservatively: model context limits, token and byte counts, timestamps, durations, process state, storage references, tool execution, transport, permissions, output reservations, queue state, and hard validation.

LLMs own decisions that depend on meaning: relevance, importance, interpretation, user intent, what history matters, what part of a tool result matters, redundancy, what should be summarized, which tool is appropriate, whether research is useful, whether an ambiguity warrants asking, whether work is complete, and what a status means to the user.

Do not replace semantic judgment with arbitrary age cutoffs, fixed message counts, keyword rules, MIME-based intent rules, or blind truncation. Give the LLM the semantic problem and numerical constraints; verify its product deterministically.

## Context budgeting is the core algorithm

### Preserve full fidelity whenever it fits

Context reduction is not a default optimization. First construct and measure the richest semantically relevant request available for the operation. If the complete candidate information fits together with the required output reserve and safety margin, include it without summarization, selection, or lossy projection. Reduction begins only because the actual serialized next call does not fit.

When reduction is necessary, it is specific to the semantic operation being constructed. There is no universal permanently compressed history or permanently preferred ordering that should replace richer source state merely because it is smaller.

Every LLM call needs an explicit budget before it is sent:

`fixed input + selected history + tool definitions + tool results + retrieved data + other dynamic input + output reserve + safety margin <= model context capacity`

Tokenize the actual serialized request with the tokenizer and chat/tool format appropriate to the model. Output space is first-class, with an operation-specific minimum and desired headroom. Preserve the richest input whenever it fits with the minimum and safety; a desired output ratio is never a reason to discard valid input. If generation actually exhausts its output limit, increase the minimum for a bounded retry and reconstruct/re-tokenize the call.

After mandatory input and output reserve are accounted for, history, summaries, tool schemas, tool results, retrieved documents, attachments, and other material compete for the remaining budget. Recalculate this for every operation; there is no single permanent agent context.

If a request is too large, deterministic code knows how many tokens must be removed but not which meaning should be removed. Ask an LLM for semantic reduction against a concrete target, then re-tokenize and verify. Never silently truncate semantic data just to fit the window.

## History, summaries, and tool results

The complete execution history is external durable state, not the current context. Preserve messages, LLM requests and responses, tool calls and results, statuses, questions, timestamps, errors, and relevant model/prompt metadata outside the window.

Each call receives a semantic projection sized for its current budget. Summaries are lossy derived views, never ground truth. Their target size follows from context calculation and can change from call to call. Summary prompts must be benchmarked because summaries can silently lose dates, identifiers, constraints, causality, or other later-relevant information.

Keep raw tool results recoverable, including failed external-process output and provider error bodies. A bounded error or preview must cite an integrity-checked exact artifact committed before the failure event. Include small relevant results verbatim; semantically select or summarize large ones against an explicit target. Give reduction calls enough task context to know what matters and retain references to raw results.

## Tools and research

Tools are capabilities, not the architecture. Ordinary OS work may be best served by a generic shell when the model reliably discovers standard commands; specialized capabilities benefit from structured tools. MCP is useful at the capability boundary but should not own Swaag's context construction, scheduling, history, or lifecycle.

Tool discovery should allow detailed schemas to be loaded only when relevant. Selecting a capability is semantic and belongs to an LLM; permission enforcement and execution are deterministic.

Research is normal agent work. When information is missing, stale, or uncertain, the LLM should consider local search, documentation, web research, experiments, or verification before asking the user to do the work. Whether research is worthwhile is semantic and should depend on uncertainty, freshness, risk, available sources, and expected value, not fixed retry counts.

## Save human attention

The objective is not the earliest plausible answer. The worker should normally inspect its result, validate it, try alternatives after recoverable failures, research missing information, test its work, and improve deficiencies. Some tasks may intentionally continue until interrupted.

Not every ambiguity should stop work. An LLM should distinguish genuinely blocking questions from questions where a provisional assumption is reasonable. Noncritical questions may be exposed while work continues; critical uncertainty can require immediate input. Semantic criticality belongs to an LLM; hard safety, permission, and resource constraints remain deterministic.

Completion is semantic. Do not declare completion because a model emitted a final-looking response. Evaluate against the user's objective, requirements, evidence, deterministic tests, and remaining deficiencies. For substantial tasks, an independent LLM evaluation call may be preferable to asking the producing context to grade itself. Supply deterministic evidence such as tests, exit status, file state, schema validation, and measured values.

Persistence and stopping need dedicated benchmarks. Final-answer correctness alone does not measure premature stopping or pointless continued work.

## Workers, communication, and inference

A base worker can remain sequential: construct context, call an LLM, execute an action, record the result, repeat. Independent tasks can use independent workers with independent histories. Expensive inference may still be shared and centrally scheduled.

The user must remain able to communicate while workers wait for long inference or tools. The target architecture separates worker execution from a responsive communication/control path. A small communication model may handle cheap operations only after benchmarking and must have escalation to stronger reasoning. Escalation need is a semantic model decision, not a keyword router: when requested, deterministic code gives the strong model the same exact question, mechanical snapshot, and source events, records integrity-linked request/resolution/failure provenance, and uses the existing cancellation-plus-exact-replay path if the main model is busy. The communication layer should inspect durable history and runtime state, answer questions about active work, forward instructions, start or stop work, and expose failures without requiring a busy worker to service everything synchronously.

Long inference must not become an untouchable global critical section. Swaag already contains cooperative preemption primitives. Backend cancellation capabilities vary and must be represented honestly. If a backend cannot truly suspend and resume transformer generation, cancellation followed by context reconstruction/replay is not bit-for-bit resume.

Every logical model call enters a durable backend-neutral inference lifecycle before transport execution. Cross-process admission uses the discovered backend slot capacity, records queued/running/completed/failed/cancelled/superseded states, and gives communication/control calls deterministic priority. Queue age increases effective priority so sustained control traffic cannot starve ordinary workers. A preempted request releases its admission before the communication operation runs; an unchanged request is then admitted again and exactly replayed, while changed target state permanently supersedes it.

User controls remain ordered durable source data at every worker stage. Deterministic code may invalidate a model decision made from an older control snapshot, stop unstarted calls after an execution boundary, or keep a provisional result from becoming terminal. It must retain already completed tool evidence and must not classify two instructions as compatible, conflicting, overriding, or more important. The next centrally compiled worker call receives the exact pending controls in delivery order and makes that semantic reconciliation. Completion evaluation, response presentation, and caller-defined output are provisional model operations too: a newer control discards their stale candidate and continues the worker rather than failing or silently committing it.

## Status, heartbeat, and failures

Mechanical observability and semantic explanation are different. Runtime code can know that inference has run for a measured duration, a tool process is active, a worker is waiting, or a process stopped reporting. An LLM can explain what that means relative to the user's goal and its importance.

Liveness must not depend on the worker explaining its own health. Active runs therefore expose a closed mechanical phase/substate contract, semantic operation kind, monotonic transition sequence, stage-entry timestamps, and a separately refreshed heartbeat. Context instruction resolution, serialization, token measurement, fit/overflow, inference admission/dispatch/streaming, contract validation, tool execution/effect verification, evidence reduction, waiting, and terminal stages remain distinguishable even when no model-authored status is available. Use an external supervisor in production; systemd watchdog supervision is appropriate on Linux.

Semantic status should receive enough context to be meaningful, potentially the overarching goal, current step, recent events, and deterministic state without the entire worker context. Prompt and context variants should be benchmarked. Critical questions and failures need a durable fallback every client can represent. Silent inactivity is not error reporting.

The current status path implements that separation as its own constrained, context-compiled operation. It starts with the complete exact status evidence snapshot, reduces only after measured overflow, validates source-event citations, and records results or failures in an operation history without writing into the concurrently active worker history. An optional separately configured communication runtime can answer without preemption or semantically escalate the unchanged snapshot to the main runtime; escalation lifecycle events preserve both operation identities, exact evidence lineage, and whether the trigger was semantic or mechanical assistant failure. Mechanical heartbeat state remains independently readable even if semantic interpretation fails.

## Interfaces and files

Swaag should not be architected around voice, one browser UI, CLI text, or one proprietary client. Attachments are raw files or durable references plus user intent. The LLM decides whether content inspection is necessary. Copying images does not require image understanding; analyzing them does.

Do not convert every upload to text automatically. Preserve raw files and cheaply determine mechanical facts such as size and likely type. Raw attachment identity/storage belongs to SWAAG; content interpretation does not. Browser automation, OCR, document conversion, speech transcription, image understanding, databases, proprietary APIs, and similar domain capabilities belong to the unbounded external-tool layer and should enter through generic connectors such as MCP or through explicitly enabled shell access. The LLM decides whether a supplied capability is relevant; Python enforces only schema, authorization, resource, and state mechanics. See `tool-architecture.md`.

Output should support a universal conversational channel and richer structured events or fields where available. Avoid inventing protocol layers unnecessarily. AG-UI is relevant for rich UI events, A2A for durable task-oriented interoperability, Open WebUI through an adapter, and MCP for capabilities. They solve different layers. Keep the internal event/state model transport-independent.

## Prompts, benchmarks, and observability

System prompts, context-selection prompts, summarization prompts, status prompts, completion evaluators, tool descriptions, and other LLM-facing instructions are implementation. Do not promote a prompt because it sounds better. Establish a baseline, test variants on representative trajectories, inspect failures, and evaluate held-out cases. Prompt behavior is model-dependent.

Durable learned instructions are different from ordinary working notes. The agent may semantically author, revise, consolidate, or remove a correction; declare broad model-call scopes and optional free-form step categories; and choose session-only or cross-session user persistence. Deterministic runtime code matches only broad declared call kinds. For categorized candidates, a separately compiled LLM sees the exact upcoming semantic context and selects the applicable exact source IDs; uncategorized entries deliberately apply to every call in their broad scope. Selection failure includes all candidates. This avoids a handwritten category taxonomy while preventing programming, research, reporting, tool-use, or other step-specific guidance from becoming one giant universal prompt. The selected exact instructions are injected into the system role, context-accounted, and retained with integrity-linked selection provenance. The runtime must not infer semantic relevance, category, scope, or persistence from keywords, tool names, or task fixtures. User corrections, tool workarounds, and instruction distillation can feed this mechanism, but automatic distillation, consolidation, and category isolation require dedicated held-out evaluations so a weak model cannot silently bloat or corrupt future calls.

Durable working notes use the same semantic ownership without pretending they are universal instructions. The model authors their free-form categories and lifecycle operations. A separately compiled selector sees every exact note and the specific upcoming action context, then chooses relevant IDs; labels never map deterministically to handlers or tools. Selection failure includes all notes. Selected notes are preserved exactly when they fit and projected only after measured overflow, while add/replace/remove/compaction events and exact source lineage remain recoverable. This keeps useful programming, research, evidence, reporting, and future-work state durable without placing every note into every action forever.

User-facing relevance selection and audio presentation are separate semantic operations. A final answer should omit operational noise that is not meaningful to the user, while preserving requested evidence. Optional audio-style conversion may rewrite tables, lists, numbers, and visual structure into listenable prose, potentially with a smaller model or another device, but it must preserve the complete selected information. Do not burden a reasoning-heavy worker call with presentation work without comparing the one-call and staged alternatives. Benchmark information preservation, spam removal, latency, and small-model suitability before enabling either transformation by default.

Benchmark instruction following, simultaneous constraints, unnecessary refusal, tool choice, history retrieval, preservation under compaction, overflow behavior, research behavior, clarification, persistence, premature completion, status quality, and end-to-end success.

Operational telemetry should map onto OpenTelemetry GenAI conventions where applicable. Operational telemetry, durable semantic history, and current model context are separate layers.

## Current implementation versus target

Swaag keeps a sequential inner model/tool loop but now surrounds it with independently addressable durable workers, a transport-neutral task/event API, backend-neutral inference admission, ordered control and cancellation, worker-aware wakeup dispatch, and a supervised communication service. Every current production semantic operation uses central full-fidelity-first context compilation. Raw event history and artifacts remain authoritative while overflow projections retain exact lineage. Independent completion evaluation, staged semantic tool discovery, caller-defined output, response-presentation stages, protocol adapters, and OpenTelemetry instrumentation are implemented and covered by deterministic tests.

This is still not the completed target. Live model experiments and replay catalogs remain required for context layout, compaction, tool strategy, research, autonomy, prompt instructions, presentation, cancellation, and long-horizon behavior. Communication model routing lacks a genuinely distinct small/strong deployment comparison. AG-UI, A2A, MCP, and Open WebUI adapters still have the conformance and exposure gaps recorded in `TODO.md`; external-tool integration coverage and the host Collector/live telemetry path remain partial. Do not promote those partial areas to complete based only on unit tests or interface shape.

## Benchmark/runtime separation

Production runtime enforces only mechanically knowable safety, protocol, authorization, resource, and state-consistency constraints. Benchmark task definitions own evaluation-only action/tool/repetition budgets because the benchmark owns the task oracle and finite test boundary. Benchmark limits must never be injected into production AgentConfig.

The benchmark runner persists an atomic configuration-signed checkpoint after each completed task. Compatible restarts restore completed task results and skip them; incompatible settings are rejected rather than silently mixing results. `--clean` is the explicit destructive opt-in.

## Evaluation dimensions must remain separable

Benchmarks must not collapse mechanically different claims into one pass rate. Generation-time schema validity is separate from semantic correctness. Exact long-horizon fact preservation is separate from provenance/recoverability, semantic retrieval, resistance to later conflicting material, and measured overflow handling. A benchmark may aggregate these dimensions for convenience only if it also reports every constituent dimension independently. See `docs/benchmark-methodology.md`.

## Durable instruction authority

Durable semantic memory has an explicit trust boundary. Model-authored rules are learned operating preferences and may never assign themselves project/user authority. Trusted project or user instructions require a trusted ingestion path with source provenance. Trusted entries are never removed from a call by model relevance selection; conflicts within the durable layer resolve deterministically by authority, specificity, then recency. The current user request remains the governing authority for the current turn.

## Context reduction requires measured need and executable headroom

Exact context that fits should be admitted before semantic reduction. Durable notes are therefore included exactly on the first measured action candidate; semantic note selection is an overflow-recovery step, not a mandatory pre-filter. History compression likewise must not select material merely because it is old: semantic span selection determines what may be reduced, while deterministic code only enforces measured budgets, provenance, replay, and protected-span boundaries. For local backends, advertised context capacity is not proof that a near-limit reduction request is executable; an explicit semantic-reduction working-set cap may force hierarchical source fragmentation before inference without shrinking the ordinary agent context window.
