# SWAAG Target System Definition

## Purpose

SWAAG is a sequential semantic agent runtime whose primary objective is to carry out the user's actual instruction as faithfully and competently as possible. The model performs semantic judgment; deterministic code provides reliable execution, accounting, integrity, isolation, persistence, and hard limits.

This document resolves the alternatives and open questions in `user_definition_from_recordings.md`. When this target conflicts with a recoverable explicit recording requirement, the recording wins and this target must be corrected. Brainstorming in the recordings is intentionally resolved here rather than implemented literally by default.

## Non-negotiable invariants

The effective user instruction is the highest-level semantic objective. The runtime must preserve the complete authoritative history. Model context must never exceed the model's context window. Every semantic control call must use generation-time constrained decoding plus local validation. Deterministic code must not pretend to know semantic relevance, satisfaction, importance, or correctness merely from mechanical heuristics. Terminal completion that requires a user answer must contain a non-empty user-facing answer. User interruption and new instructions must be accepted independently of a long-running main operation. The core agent remains sequential: one semantic main-agent action is resolved at a time unless a future explicit design change says otherwise.

## Resolved architecture

The core consists of a sequential main-agent loop, an append-only authoritative event history, deterministic prompt-budget accounting, a semantic context-selection layer, a constrained structured-action model interface, a capable execution/tool layer, and an independent control plane for interrupt/status/new instructions.

There is no model-call fan-out inside the main-agent decision loop. Tool subprocesses may be long-lived or asynchronous when their semantics require it, but their completion events feed back into the sequential semantic loop. Independent external services may run concurrently; that does not make the agent's semantic decision stream parallel.

## Effective user instruction

The runtime must carry forward the effective instruction across turns. A new user message may refine, replace, interrupt, or correct the previous objective. User dissatisfaction is a high-priority semantic event. The next decision after explicit dissatisfaction or repeated failure must include recovery analysis rather than silently repeat the previous strategy.

For underspecified instructions, the default is the smallest reasonable, reversible action that literally satisfies the request and does not invent a large hidden project. Ask a clarification only when a materially important choice cannot reasonably be inferred or when proceeding would likely violate the user's actual intent. Do not ask merely to avoid work that can be performed safely and reversibly.

## Canonical history and projections

`complete_history.jsonl` remains the canonical session history. It is append-only, hash chained, and sufficient to rebuild runtime state. Derived state files, indexes, embeddings, summaries, databases, and caches are projections only.

The history must include user messages, assistant messages, effective instruction changes, prompt assembly metadata and token budgets, exact model request metadata, structured model responses, accepted/rejected actions, status fields, tool calls and arguments, tool outputs or durable references to large outputs, errors, retries, file mutations, summaries, interruption/control events, and terminal state.

Large output is stored completely outside the active prompt when necessary and represented by a bounded preview plus a durable history/source reference. No information is discarded solely because it does not fit in context.

## Context construction

Context construction has two layers. The deterministic budget layer tokenizes or conservatively accounts for every candidate and guarantees `input + reserved_output <= model_context_limit` with an explicit safety margin. The semantic selection layer decides which history, summaries, source excerpts, and task state are most relevant.

The deterministic layer may drop or truncate candidates only according to explicit mechanical policy after the semantic layer has supplied priorities, and it must record what was omitted. It may never infer that old equals irrelevant, recent equals important, large equals spam, or tool failure equals low value.

The semantic selector should have access to the effective user objective and metadata for all candidate history ranges. It can request full detail for ranges it judges important. Recent detailed history is a default convenience, not a fixed invariant. Long-running tasks may retain large contiguous detailed spans when necessary.

Summaries are derived compression and must contain source-range provenance. When a summary becomes insufficient or questionable, the agent can retrieve the source events again. Summary creation is constrained and recorded.

## Model interface and structured action

Every semantic model call uses generation-time JSON-schema or grammar constraints supported by the provider, followed by local schema validation. There is no silent unconstrained-text fallback for control decisions.

The main action schema contains at least: a user-facing `assistant_message`; `continue_loop`; semantic status consisting of situation, next action, reason, and importance; zero or more permitted tool calls; and explicit recovery/uncertainty fields where needed. The schema must allow a useful answer with no tool call and tool calls with continued work.

If `continue_loop=false` on a turn for which the user expects an answer, `assistant_message` must be non-empty after trimming. Empty terminal messages are rejected and reprompted as a contract violation. Silent completion is allowed only when the user explicitly requested no response or the enclosing protocol defines a non-text terminal result.

The model, not deterministic code, decides whether the task is semantically complete. Deterministic validators may reject a terminal action when mechanical contract requirements are unmet, such as empty required output, invalid schema, pending mandatory tool result, or known unfinished transaction.

## Recovery loop

Recovery is a semantic mode, not a pile of benchmark-specific heuristics. It is triggered by explicit user correction/dissatisfaction, repeated tool or verification failure, repeated rejected actions, or a model-declared need to reconsider. Recovery receives the effective user objective, relevant detailed history, previous strategies, failures, and evidence. It must produce a root-cause hypothesis, what should change, and the next strategy.

A separate history-analysis sub-agent may later implement this mode, but the initial target keeps it inside the same model/runtime with a dedicated constrained recovery schema. This avoids extra topology while satisfying the required behavior. A dedicated offline analysis service can be added later without changing the history contract.

## Tool architecture

The primary general-purpose execution primitive is a robust command/process tool. It supports non-interactive commands, persistent interactive sessions when required, stdin, environment/workdir, timeouts, cancellation, bounded live output, durable full-output capture, process state, and clear exit/signal metadata.

Specialized tools are justified when they provide capabilities that ordinary CLI programs cannot easily provide with equivalent correctness, history integration, or safety. Existing deterministic file/edit primitives may remain as implementation mechanisms and optional agent tools, but the runtime must not require the model to use a custom text-edit tool when a normal command is appropriate.

A wait tool supports relative durations in milliseconds, seconds, minutes, hours, days, weeks, and months, and absolute timestamps. It persists a wake-up record and does not hold an LLM request open. Years are intentionally not a normal supported unit. Operators may configure a maximum wait horizon.

Tool state and shared resources are thread/process safe. Tools may be exposed through MCP where cross-process interoperability is useful. The target does not require one MCP server process per primitive; in-process tools are acceptable when they obey the same schema, isolation, and concurrency guarantees.

## Large tool output

Every tool result has a durable complete representation plus a bounded prompt representation. The prompt representation includes size/count metadata and a retrieval handle. The model can request more ranges, search, summarize, or semantically inspect the output. The runtime must never inject arbitrarily large command output into context merely because the command returned it.

The semantic model decides whether content is important. Deterministic code decides how many bytes/tokens can be safely shown at once.

## Reading and source traversal

Readers support bounded sequential traversal and targeted retrieval. Sequential reading state is persistent so the agent can continue through very large files without losing place. The agent can also search or jump when that is more appropriate. No rule requires all large sources to be read linearly if semantic search or a targeted query can reliably answer the task.

## Status and HMI

The core remains text-in/text-out. Speech, GUI, phone, TTS, and STT are separate HMI systems.

Each accepted semantic action records structured status fields in history: situation, action/next step, reason, and importance. Importance uses a small text enum such as `minor`, `normal`, `major`, `critical`, because textual categories are model-friendly and more meaningful than arbitrary numbers. The UI may suppress minor statuses or all statuses; suppression never deletes history.

Status is generated as part of the same main constrained action, not through an extra status-only LLM call. If a deliberately isolated semantic call lacks enough global context to produce meaningful status, the runtime marks status as `isolated_call` and records the literal purpose supplied to that call rather than inventing global reasoning.

## Independent communication/control

The required capability is a control plane that remains responsive while the main agent is working. It can query recent status/history, stop/cancel current work, enqueue a new user instruction, and request redirection.

A separate small natural-language assistant model is not required for the core implementation. The initial target exposes these controls directly through CLI/API/HMI. A small assistant can later be layered on top as a consumer of the same history/control APIs if measurements show it improves usability. This resolves the recording's assistant-agent proposal without making an additional model a core dependency.

## Memory and semantic retrieval

Authoritative memory is history. Working memory, notes, event memory, embeddings, and indexes are derived views. No derived memory can silently overwrite or replace source history.

Semantic history retrieval is performed by the model or a model-assisted retrieval step. Embedding similarity may be used as a candidate generator, particularly over structured status fields, but embeddings are not the final judge of relevance. Deterministic recency is a fallback ordering only when no semantic scoring is available; it must not be described as the desired relevance policy.

## Configuration

All operational values with reasonable deployment variability live in configuration with explicit precedence. This includes model endpoint/identity, context/output reserves, timeouts, command limits, wait horizon, status display, history paths, archival thresholds, semantic retrieval budgets, schema mode, and tool enablement. CLI/environment overrides are normalized into one effective configuration and recorded in session metadata.

## Testing and acceptance

Tests must verify invariants rather than bake in task answers. Required coverage includes context-never-overflows property tests, structured-decoding enforcement, empty-terminal-answer rejection, history replay/integrity, large-output retrieval without context flooding, interruption during long-running work, user-correction recovery, sequential semantic execution, tool cancellation, wait persistence, and model-driven relevance selection.

Benchmarks may evaluate semantic quality, but benchmark-specific expected answers or task-family routing must not leak into runtime behavior.

## Resolved decisions from the recordings

The main agent is sequential. Semantic relevance is model-driven while token accounting is deterministic. Complete history remains canonical; summaries/indexes/embeddings are projections. Status is structured in the same constrained action and optional to display. Independent interrupt/status/new-instruction control is mandatory, but a second assistant model is optional and deferred. Recovery uses an explicit semantic recovery phase in the main runtime before adding a separate analysis service. MCP is supported as an interoperability boundary, not mandated per tool. The shell/process tool is the principal general execution tool; specialized tools exist only when they add real capability or reliability. Long waits are persistent scheduled events rather than sleeping model calls. All control-model calls use constrained decoding. Terminal answers that should communicate with the user cannot be empty.

## Known source limitation

The source transcript for `Recording 2026-08-11 12-06-27-811` is corrupted after the initial assistant/history-tool discussion. The original audio is present and untouched. The target definition must be revisited after that audio is retranscribed; until then no design claim may be attributed to the unavailable portion.
