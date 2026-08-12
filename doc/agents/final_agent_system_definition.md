# Final SWAAG Agent-System Definition

## Status and provenance

This document is the resolved target architecture for SWAAG. It is deliberately separate from `user_definition_from_voice_recordings.md`.

The user-definition document preserves the recordings, including tentative ideas and unresolved alternatives. This document makes explicit decisions for those alternatives after evaluating the recordings as a whole, the existing SWAAG implementation and benchmark failures encountered during development, current primary documentation, relevant research, and a small local experiment against the currently used Qwen2.5-14B model on Thor.

This is a target definition, not a claim that the current code already implements every item below.

## The central rule

SWAAG is an agent runtime whose primary job is to execute the user's actual request as faithfully as possible. The model owns semantic interpretation and task strategy. Deterministic code owns mechanical truth: persistence, exact event ordering, token budgets, schema validation, concurrency, process execution, clocks, file boundaries, checksums, and other facts that do not require semantic judgment.

Python must not silently become a second semantic agent. There is no hidden planner, reviewer, policy engine, semantic completion gate, or deterministic relevance classifier that overrides the model's interpretation of the task. If the runtime needs semantic judgment, that judgment belongs in a model call with the relevant evidence available.

At the same time, the model must not be asked to manufacture mechanical facts. File existence, tool handles, process state, byte counts, token counts, timestamps, event sequence numbers, artifact offsets, test exit codes, database state, and similar facts come from deterministic mechanisms and are supplied to the model as authoritative evidence.

## Resolved architecture in one view

SWAAG consists of a durable event/history core, a context assembler, a constrained action loop, an extensible tool layer, a long-running-work scheduler, and a parallel communication/control service. The main agent is not a chain of hardcoded planning stages. Each action is one constrained model decision made from the user's request, the current authoritative state, the selected history/evidence, and the available tool schemas.

Every significant action and result becomes a durable event. Exact historical evidence remains recoverable even after it is no longer in the immediate model context. Large outputs remain exact in artifacts rather than being destroyed by truncation. The context window contains only what can fit safely, selected through a combination of deterministic budget accounting and model-authored semantic memory management.

The HMI is outside the core agent. The core speaks text plus structured events. CLI, web, mobile, speech-to-text, text-to-speech, and other interfaces subscribe to those events and submit user messages/control commands without redefining the agent runtime.

# Decisions for every recorded open question

## Context construction and history selection

### Decision

Use a tiered virtual-context architecture. Never define relevance as a fixed number of recent messages.

The canonical tiers are:

1. **Mandatory exact context.** The current user request and refinements, hard runtime/tool instructions, the action schema, current tool registry, unresolved direct tool errors, and other mechanically required current evidence.
2. **Active exact history.** The recent and causally active events that the model has explicitly kept or that are directly connected to the current action.
3. **Model-authored working summary.** A compact summary of older context that includes source event-sequence references. The summary is navigation and compression, not authority.
4. **Searchable exact history.** Older canonical events remain available through structured history search and exact history-window retrieval.
5. **Large exact artifacts.** Oversized raw outputs live outside the prompt and are retrieved in bounded slices or searched by offset/query.

The context assembler first computes the exact admission budget, including the rendered prompt, schemas/tool definitions, an output reservation, and a configurable safety margin. It may compact only material that is semantically replaceable. It must never silently truncate the current user request or mechanically required error/evidence state.

When compaction is necessary, the model chooses what older information is semantically important to preserve. The runtime validates only mechanical limits. Exact source sequence references are retained so the model can page old evidence back into context.

### Rationale

The recordings identify context filling as the main quality problem. Research supports not equating a larger context with better access to information: *Lost in the Middle* shows substantial position-dependent degradation in long contexts, while MemGPT demonstrates a virtual-memory-style approach in which the model retrieves and evicts information across memory tiers. The design therefore treats the context window as working memory, not as the history database.

## Canonical history storage

### Decision

Use SQLite as the canonical local history/event database, in WAL mode, on local storage only. Use one serialized write path and independent read connections. Require a SQLite version containing the 2026 WAL-reset fix: version 3.51.3 or newer, or an explicitly patched/backported version.

The database schema is append-oriented. Core tables include sessions, events, messages, tool calls/results, status records, wakeups, artifact metadata, summaries, and control/inbox messages. Every canonical event has a monotonically increasing sequence ID within its session, UTC timestamp, event type, and structured payload. Derived search indexes and summaries may be rebuilt from canonical events.

Use `PRAGMA journal_mode=WAL` and `synchronous=FULL` for canonical agent history. A dedicated writer/transaction coordinator serializes writes. Readers use separate connections and short transactions. Checkpointing is performed periodically without holding long-lived read transactions.

Do not place a WAL database on a network filesystem. Remote processes and other machines access history through the history service/API, not by mounting the database file.

### Search

Use ordinary indexed SQL for session, sequence, time-range, event-type, tool-name, and status-field queries. Use SQLite FTS5 for lexical full-text search over user text, assistant/status text, summaries, and selected tool metadata.

Embeddings are optional derived indexes, never canonical history. They are generated asynchronously and keyed back to exact source events. Store separate searchable embeddings for the semantic fields that the recordings repeatedly identify as useful: situation, chosen action, and reason. Similarity results only identify candidate events; the model must retrieve the exact source event/window before relying on it.

### Archival

No information is deleted by default. Closed sessions may be moved to immutable read-only archive shards when configurable age/size thresholds are exceeded. The active history service searches active storage plus archive indexes transparently. Archival is a storage optimization, not semantic forgetting.

## Large tool output

### Decision

Do not run an automatic semantic classifier on every tool result. Instead, separate exact persistence from prompt admission.

Every tool has a bounded immediate result. If stdout, stderr, test output, terminal output, file content, or another stream exceeds the immediate bound, persist the complete exact content as an artifact with artifact ID, total length, checksum, encoding metadata, and creation event. The immediate result exposes the handle and mechanically chosen bounded preview metadata. The model can then use `read_artifact` and artifact search to inspect the exact portions it needs.

Artifact search is lexical/regex/mechanical and returns matching offsets plus bounded context. A model-authored summary may be generated on demand if it is useful, but the exact artifact remains authoritative.

This avoids both bad extremes described in the recordings: flooding the next prompt with enormous output and having deterministic code guess which semantic part matters.

## Which operations are first-class tools versus ordinary CLI

### Decision

Do not wrap every existing operating-system capability in a bespoke SWAAG tool. A first-class tool is justified when it adds agent-specific semantics that the ordinary CLI does not conveniently provide.

The required native tool families are:

- one-shot command execution with bounded output, timeout/background semantics, exact exit status, and artifacts;
- persistent interactive terminal sessions with create/send/read/close and exact opaque handles;
- history search/window/retrieval;
- artifact read/search;
- durable wait/scheduling and wakeups;
- agent control/inbox operations for pause, stop, redirect, and communication;
- notes/working memory where useful;
- deterministic calculator/time helpers;
- structured verification/test execution where it adds reliable environment/result metadata.

Ordinary text transformations, common filesystem inspection, package commands, compilers, version-control commands, and other standard OS utilities should normally be invoked through the command/terminal layer instead of duplicated merely because the agent could have another tool.

Specialized read/edit tools may exist when they demonstrably add bounded-output handling, exact patch validation, syntax checks, portability, or other mechanical guarantees. Their justification is those semantics, not the mere existence of `cat`, `sed`, or another equivalent command.

## Terminal/process API

### Decision

Support two complementary execution modes.

**One-shot command:** run a non-interactive command with explicit cwd/environment, timeout, output limits, background option, exit status, and artifact capture.

**Persistent terminal:** `create`, `send`, `read`, and `close`, returning a durable opaque `terminal_id`. Output reads are offset-based and bounded. A terminal created in one action is used by its exact returned handle in a later action; calls in one model action cannot assume outputs from other calls in that same action.

Linux uses a PTY. Windows support uses the appropriate Windows pseudo-console/process primitives behind the same logical interface. Platform differences stay below the model-facing schema whenever possible.

Background processes have process IDs/handles, bounded polling, explicit cancellation, and durable state events. The runtime never blocks an agent turn indefinitely on an unbounded process read.

## Durable waiting and scheduling

### Decision

Support both relative durations and absolute target times.

Relative human-readable units are milliseconds, seconds, minutes, hours, days, weeks, and calendar months. A calendar month is not converted to an arbitrary fixed number of seconds; it is resolved to an absolute target date using calendar arithmetic. Absolute dates use ISO-8601-compatible date/time input. Ambiguous times use the configured local timezone and are normalized to UTC in storage.

The default maximum scheduling horizon is 366 days. It is configuration, not a semantic prohibition; deployments can raise it deliberately. This satisfies the recording's desire to support months without casually leaving forgotten multi-year wakeups.

A durable wakeup stores the target UTC time, creation time, session ID, reason/message, status, and source event. On restart the scheduler reloads pending wakeups and immediately dispatches overdue ones exactly once.

For an active relative wait, monotonic/boottime clocks measure elapsed duration so wall-clock corrections do not shorten or lengthen the interval. The persistent target UTC time provides restart recovery. The wait lifecycle emits `wait_entered`, `wait_resumed`, and `wait_completed` events.

## Status messages

### Decision

Generate status information in the ordinary constrained action call, not through an additional LLM call by default. Extra calls add latency/cost and can disagree with the action that actually executes.

Every model action may include a structured status object:

- `situation`: short statement of the relevant current state;
- `action`: what the agent is doing or about to do;
- `reason`: why that action follows from the task/evidence;
- `importance`: `minor`, `normal`, or `major`.

Three levels are sufficient. They are stored as human-readable text because the data is consumed by both humans and models. The runtime also derives an integer rank mechanically (`minor=1`, `normal=2`, `major=3`) for filtering/sorting, so no numeric mapping has to be inferred later.

A small Thor Qwen2.5-14B test compared 15 synthetic classifications with the same three semantic levels. Text labels scored 11/15 and numeric 1/2/3 labels scored 13/15. This is too small and synthetic to establish a model-quality advantage for either representation. The final choice is therefore based on schema clarity and durable readability, not the unconfirmed hypothesis that text is intrinsically easier.

Status events are persisted regardless of whether they are displayed. Live display is configurable and enabled by default when the interface can carry it. The HMI may locally filter by importance without changing what is stored in history.

## Isolated model calls

### Decision

The default is to include the overarching task context needed to understand why a call exists. Do not let deterministic code guess that a semantic call should be contextless.

An isolated helper call is allowed only when its contract intentionally defines a self-contained transformation and isolation has a concrete reason, such as reducing unwanted bias or processing independent chunks. The isolation is explicit in the call type/tool contract or requested by the model, is recorded in history, and must be benchmarked against the context-aware version before becoming a default for a task family.

An isolated call does not fabricate semantic status. The system may expose the literal helper prompt as mechanical observability data, clearly labeled as such.

## Parallel assistant / communication layer

### Decision

Implement the communication layer as a separate service/process using the same generic SWAAG runtime and event/tool abstractions with a restricted configuration. Do not build a second unrelated agent framework.

The communication service has its own model endpoint when resources permit. Model size is configurable rather than fixed to 4B or 8B; the deployment uses the smallest model that passes a dedicated communication/history-control benchmark. If no independent model is available, the interface still provides deterministic status/history views and can enqueue control/user messages without waiting for the main model.

The communication agent's default tools are history search/window, current time and time arithmetic, task/status lookup, and agent-control/inbox operations. General arbitrary shell access is not required for the communication role. Optional read-only file access can be enabled by configuration.

The communication service reads canonical history through the history API/service. It submits new instructions through a durable prioritized inbox. `stop` and `pause` have higher priority than ordinary redirection/messages. The main runtime checks the inbox between model actions and at safe tool boundaries.

Requests directed to the main agent receive correlation IDs. The communication service can immediately tell the user that a request is queued, then wait asynchronously for a correlated main-agent response without blocking other communication requests.

Use async concurrency inside the communication service for multiple user connections, but keep model-generation concurrency bounded by the actual model server capacity.

## Thread/process safety

### Decision

Canonical mutable state is coordinated through process-safe mechanisms, not ad hoc shared Python objects. SQLite transactions protect durable history. Durable inbox/wakeup tables coordinate cross-process work. In-memory thread queues are used only inside one process. Cross-process queues use process-safe IPC or the durable database/service interface.

Each SQLite connection has clear ownership. The history writer is serialized; readers use independent connections. Tools that expose independent services must be safe for concurrent clients or must serialize their own mutable resource.

## MCP boundary

### Decision

MCP is supported as an interoperability and isolation boundary, not mandated as SWAAG's internal hot-path dispatch mechanism.

SWAAG defines one canonical tool schema/implementation interface. A tool can be mounted directly in-process or exposed through an MCP adapter. Use MCP when the tool is external, separately deployed, on another machine, supplied by another project, or intentionally isolated behind a service boundary. Keep latency-sensitive local primitives in-process unless isolation/interoperability justifies the protocol hop.

This decision is strengthened by the current MCP 2026-07-28 architecture, whose core is explicitly stateless and request/response oriented. That is excellent for scalable integration boundaries but does not replace SWAAG's own durable session/history/event model.

## Root-cause/history analysis

### Decision

Provide a read-only `history_analyze` capability implemented as an agent/sub-agent using the same history retrieval primitives. It may use a larger context/model or run on another machine. It does not directly mutate the workspace.

Input includes the analysis question and authoritative session reference. The analyzer reconstructs the user goal and refinements, searches exact history, inspects relevant model input/tool evidence, and returns a constrained result containing:

- reconstructed goal/constraints;
- observed failure or dissatisfaction evidence;
- candidate root causes;
- exact source event references for each claim;
- what previous strategy was wrong or incomplete;
- recommended materially different next strategy;
- unresolved uncertainties.

The main model decides when semantic evidence warrants calling it. Python may expose mechanical signals such as repeated identical failures, failed test counts, or explicit new user messages, but it does not classify sarcasm or dissatisfaction itself.

Offline analyzers may continuously scan closed/active sessions for quality research, but their findings are annotations, not retroactive edits to canonical history.

## Ambiguous user requests, corrections, sarcasm, and initiative

### Decision

The permanent behavioral prompt stays small. Its core meaning is:

> Execute the user's actual request and constraints. Ground claims in authoritative evidence. Do not substitute a different goal. When details are unspecified, choose a reasonable minimal interpretation that advances the literal request. Use history to resolve references and refinements. Treat user corrections as evidence that the current interpretation/strategy may be wrong and re-evaluate before continuing.

Do not hardcode a sarcasm detector. Sarcasm, insults, corrections, and dissatisfaction are semantic language phenomena for the model to interpret in context. The important behavior is that a correction causes reassessment rather than repetition.

For underspecified requests, default to the smallest useful action consistent with the literal request rather than launching an enormous project. If two interpretations would produce materially different irreversible outcomes and the user has supplied no basis to choose, the model should avoid the irreversible choice and choose a reversible/minimal path or request the missing information when genuinely necessary.

Creative requests necessarily permit the model to invent unspecified creative details. The permanent instruction must therefore not say “never invent anything”; it must distinguish unsupported factual claims from creative choices required to complete the user's task.

## Constraint decoding and structured actions

### Decision

Every model-to-runtime control decision uses constrained structured output. A model backend that cannot provide a tested constrained-output mechanism is not a supported autonomous-control backend.

JSON Schema 2020-12 is the canonical schema language at SWAAG boundaries, but each model adapter compiles the canonical contract to a deliberately portable subset supported by that backend. For llama.cpp, avoid known-problematic constructs such as nested `$ref` chains and unsupported keywords; prefer explicit flat object schemas, primitive types, arrays, enums, required fields, bounded values, and `additionalProperties: false` where appropriate.

The expected structure is also explained in the prompt when the backend does not expose schema semantics directly to the model. This follows llama.cpp's own documentation, which notes that a JSON schema used for grammar enforcement is not automatically visible to the model outside tool-calling contexts.

Constrained generation is followed by independent JSON parsing and schema validation. Invalid output is never silently accepted. A bounded retry includes the exact validation/parsing error and the same contract. Repeated malformed output is classified as a model/backend structured-output failure with the raw response durably recorded. SWAAG does not fall back to unconstrained autonomous action generation.

Schema-portability tests run in CI against every supported model backend/version.

## Tool-call sequencing

### Decision

Tool calls inside one action are parallel declarations with respect to model knowledge: the model chooses all arguments before any result exists. Therefore one call may depend only on evidence that existed before the action. If call B needs an artifact ID, terminal ID, process ID, offset, filename, or other value returned by call A, B belongs in the next model action.

The runtime exposes the latest mechanically authoritative handles/cursors in context. It may resolve narrowly defined symbolic handle references such as the most recent stdout artifact, but it never invents a semantic dependency graph on behalf of the model.

## Failure recovery

### Decision

A failed tool/test/verification result is evidence, not a universal Python-owned semantic gate. The model interprets what it means for the user task. However, if the user explicitly requested a side effect or verification, a final chat message cannot substitute for performing that operation.

Immediate exact duplicate actions are rejected before re-executing pure duplicate work. The model receives the previous evidence and must choose a materially different action or explicitly justify why no further action is required.

Coding recovery follows module responsibility: fix the defect in the component that owns the violated contract instead of masking it downstream merely to satisfy a test. The action prompt should state this as a general software-engineering principle, not as benchmark-specific knowledge.

## Configuration

### Decision

Every operational parameter that can reasonably vary by deployment belongs in configuration. Defaults live in the repository. Deployment/host overrides, environment variables, and CLI overrides may change them with a documented precedence order. Configuration includes model endpoints and limits, context budgets, output bounds, scheduler horizon, status display/filtering, history paths/retention, tool enablement, assistant model/worker limits, timezones, and backend-specific schema settings.

Configuration changes that affect generated model output participate in model/cache identity so cached responses are never reused across output-affecting differences.

# Research and experimental evidence behind the decisions

## Long context is not equivalent to reliable memory

Nelson Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023), show that models can perform substantially worse when relevant material appears in the middle of long contexts. This supports SWAAG's use of explicit retrieval and context prioritization rather than simply maximizing prompt history.

Charles Packer et al., *MemGPT: Towards LLMs as Operating Systems* (2023), demonstrate an LLM-managed virtual-memory approach in which relevant historical data can be retrieved into limited context and less relevant material evicted. SWAAG's resolved history tiers use the same high-level insight while retaining an exact append-only event authority.

## SQLite for same-host history concurrency

SQLite's WAL documentation states that readers and a writer can proceed concurrently, while only one writer can exist at a time, and that all WAL processes must be on the same host. This directly motivates a serialized local writer plus multiple local readers and a service/API boundary for remote access. SQLite's documentation also reports a WAL-reset corruption bug fixed in 3.51.3 and selected backports, so the minimum supported runtime must contain that fix.

SQLite FTS5 provides built-in full-text indexing suitable for exact-history candidate retrieval. It is the primary text search layer; semantic vectors remain optional derived indexes.

## MCP is an integration protocol, not SWAAG's memory model

The Model Context Protocol specification defines a standard host/client/server interface for tools and contextual data. The latest 2026-07-28 revision changes the core to stateless, self-describing request/response operation. SWAAG therefore uses MCP where an interoperability/service boundary is valuable while retaining its own durable state model internally.

## llama.cpp structured-output constraints are real but not universal JSON Schema

The llama.cpp grammar documentation supports JSON-schema-constrained generation but explicitly documents unsupported/broken schema features including nested `$ref` behavior and other keyword limitations. It also states that grammar schema is not inherently shown to the model. This justifies SWAAG's portable schema subset, explicit prompt description, post-validation, backend-version tests, and refusal to silently degrade to unconstrained control output.

## Timing semantics

POSIX/Linux clock documentation distinguishes wall-clock time from monotonic clocks and notes that monotonic time is not affected by discontinuous wall-clock changes. SWAAG therefore uses monotonic/boottime clocks for active elapsed waits and persisted UTC targets for durable restart recovery.

## Communication concurrency

Python's standard documentation distinguishes thread queues, async queues, and process-safe multiprocessing queues. SWAAG therefore does not treat an `asyncio.Queue` as multiprocess synchronization; durable database/inbox state or process-safe IPC is required across processes.

## Thor Qwen2.5-14B status-label experiment

A small controlled local experiment ran 15 status-importance examples twice under strict JSON-schema generation. The textual `minor/normal/major` version classified 11/15 according to the hand-authored expected labels; the numeric `1/2/3` version classified 13/15. The sample is too small and subjective to establish superiority. The architecture therefore uses textual labels for durable semantics and derives numeric rank mechanically, avoiding reliance on the unverified assumption that either encoding is inherently easier for the model.

# Primary sources consulted

- Model Context Protocol Specification, revision 2026-07-28, modelcontextprotocol.io/specification/2026-07-28
- Model Context Protocol 2026-07-28 release notes/blog, blog.modelcontextprotocol.io/posts/2026-07-28/
- SQLite Write-Ahead Logging documentation, sqlite.org/wal.html
- SQLite FTS5 documentation, sqlite.org/fts5.html
- SQLite thread-safety documentation, sqlite.org/threadsafe.html
- llama.cpp grammar / JSON-schema documentation, github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- JSON Schema Draft 2020-12 core/validation documentation, json-schema.org/draft/2020-12/
- Python 3.14 standard-library documentation for `asyncio`, `queue`, and `multiprocessing`
- POSIX/Linux `clock_gettime` / monotonic-clock documentation
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, arXiv:2307.03172
- Packer et al., *MemGPT: Towards LLMs as Operating Systems*, arXiv:2310.08560

# Final resolved definition

The final SWAAG architecture is therefore not “every brainstorm implemented literally.” Every piece of information from the recordings is preserved in the user-definition document; every unresolved architectural choice is resolved here into one coherent system. Where the recordings supplied a strong invariant, this design keeps it. Where they supplied alternatives, this design chooses one and explains why. Where an idea remains useful only as an optional extension, the architecture preserves the extension point without hardcoding the example as the only possible implementation.

The implementation is correct only when the code, tests, and full live benchmark conform to this final definition while preserving exact traceability back to the source recordings.
