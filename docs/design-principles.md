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

Keep raw tool results recoverable. Include small relevant results verbatim; semantically select or summarize large ones against an explicit target. Give reduction calls enough task context to know what matters and retain references to raw results.

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

## Status, heartbeat, and failures

Mechanical observability and semantic explanation are different. Runtime code can know that inference has run for a measured duration, a tool process is active, a worker is waiting, or a process stopped reporting. An LLM can explain what that means relative to the user's goal and its importance.

Liveness must not depend on the worker explaining its own health. Expose deterministic heartbeat/state and use an external supervisor in production; systemd watchdog supervision is appropriate on Linux.

Semantic status should receive enough context to be meaningful, potentially the overarching goal, current step, recent events, and deterministic state without the entire worker context. Prompt and context variants should be benchmarked. Critical questions and failures need a durable fallback every client can represent. Silent inactivity is not error reporting.

The current status path implements that separation as its own constrained, context-compiled operation. It starts with the complete exact status evidence snapshot, reduces only after measured overflow, validates source-event citations, and records results or failures in an operation history without writing into the concurrently active worker history. An optional separately configured communication runtime can answer without preemption or semantically escalate the unchanged snapshot to the main runtime; escalation lifecycle events preserve both operation identities and exact evidence lineage. Mechanical heartbeat state remains independently readable even if semantic interpretation fails.

## Interfaces and files

Swaag should not be architected around voice, one browser UI, CLI text, or one proprietary client. Attachments are raw files or durable references plus user intent. The LLM decides whether content inspection is necessary. Copying images does not require image understanding; analyzing them does.

Do not convert every upload to text automatically. Preserve raw files and cheaply determine mechanical facts such as size and likely type. Broad converters such as all2text or Docling are capabilities, not semantic routers. OCR, speech transcription, image understanding, structured extraction, shell operations, or no inspection can all be correct for the same file depending on intent.

Output should support a universal conversational channel and richer structured events or fields where available. Avoid inventing protocol layers unnecessarily. AG-UI is relevant for rich UI events, A2A for durable task-oriented interoperability, Open WebUI through an adapter, and MCP for capabilities. They solve different layers. Keep the internal event/state model transport-independent.

## Prompts, benchmarks, and observability

System prompts, context-selection prompts, summarization prompts, status prompts, completion evaluators, tool descriptions, and other LLM-facing instructions are implementation. Do not promote a prompt because it sounds better. Establish a baseline, test variants on representative trajectories, inspect failures, and evaluate held-out cases. Prompt behavior is model-dependent.

Durable learned instructions are different from ordinary working notes. The agent may semantically author, revise, or remove a correction, declare the model-call kinds where it applies, and choose session-only or cross-session user persistence. Deterministic runtime code may then match only those declared scopes, inject every matching instruction from both stores into the system role, account for its exact serialized cost, and retain integrity-chained event provenance. The runtime must not infer instruction scope or persistence from keywords, tool names, or task fixtures. User corrections, tool workarounds, and instruction distillation can feed this mechanism, but automatic distillation and de-duplication require dedicated evaluations so a weak model cannot silently bloat or corrupt every future prompt.

User-facing relevance selection and audio presentation are separate semantic operations. A final answer should omit operational noise that is not meaningful to the user, while preserving requested evidence. Optional audio-style conversion may rewrite tables, lists, numbers, and visual structure into listenable prose, potentially with a smaller model or another device, but it must preserve the complete selected information. Do not burden a reasoning-heavy worker call with presentation work without comparing the one-call and staged alternatives. Benchmark information preservation, spam removal, latency, and small-model suitability before enabling either transformation by default.

Benchmark instruction following, simultaneous constraints, unnecessary refusal, tool choice, history retrieval, preservation under compaction, overflow behavior, research behavior, clarification, persistence, premature completion, status quality, and end-to-end success.

Operational telemetry should map onto OpenTelemetry GenAI conventions where applicable. Operational telemetry, durable semantic history, and current model context are separate layers.

## Current implementation versus target

Swaag currently has a small sequential agent loop, tool execution, model abstraction, communication abstractions, events, benchmarking utilities, and cooperative inference preemption. These are foundations, not the completed target architecture.

Do not document the following as completed until they exist and are tested: exact per-call context compilation across dynamic components; lossless durable event history with semantic projections; dynamically sized history and tool-result compaction; semantic tool discovery; research planning as first-class behavior; independent completion evaluation; multi-worker lifecycle management; a dedicated communication agent; AG-UI/A2A/Open WebUI adapters; OpenTelemetry integration; or external heartbeat supervision owned by Swaag.
