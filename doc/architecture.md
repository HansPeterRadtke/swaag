# Architecture

## Package map

- `swaag.config`
  - typed config loading and validation
- `swaag.tokens`
  - exact and conservative token counters
  - budget report construction
- `swaag.prompts`
  - deterministic prompt assembly from explicit components
- `swaag.grammar`
  - plain, GBNF, and JSON-schema contract specs
- `swaag.model`
  - direct llama.cpp client for `/health`, `/tokenize`, and `/completion`
- `swaag.events`
  - strict event factory, allowed event types, required payload keys, hash generation
- `swaag.history`
  - canonical append-only history
  - replay engine
  - hash-chain verification
  - derived projections and index
  - latest-session resolution, human-readable session names, control inbox, and exact-detail history queries
- `swaag.planner`
  - model-driven task planning, strict validation, and plan-step transitions
- `swaag.prompt_analyzer`
  - prompt classification and completeness analysis
- `swaag.decision`
  - explicit execution decisions derived from prompt analysis
- `swaag.expander`
  - deterministic task expansion for vague or incomplete goals
- `swaag.strategy`
  - strategy selection for exploration, assumption validation, and fallback behavior
- `swaag.working_memory`
  - derived short-term task state
- `swaag.memory_semantic`
  - semantic and procedural memory extraction and retrieval
- `swaag.project_state`
  - derived multi-file project-state model
- `swaag.context_builder`
  - deterministic context compiler with scoring, ranking, and selection traces
- `swaag.evaluator`
  - explicit step-result evaluation
- `swaag.security`
  - trust classification and semantic-memory promotion policy
- `swaag.compression`
  - history-summary decisions and summary payload helpers
- `swaag.notes`
  - bounded working notes
- `swaag.reader`
  - bounded sequential reading
- `swaag.editing`
  - pure text editing engine and previews
- `swaag.environment`
  - persistent shell state
  - workspace snapshots
  - tracked foreground/background processes
  - browser bridge and filesystem/process managers
- `swaag.tools`
  - tool definitions, validation, policy gating, isolated execution contexts
- `swaag.runtime`
  - bounded hierarchical runtime orchestration
  - background-process polling, waiting, and resume handling
  - active-session control-plane classification/application
  - deferred task queue handling
  - code checkpoint create/restore
- `swaag.subsystems`
  - planning, reasoning, tool, and file execution subsystems
- `swaag.cli`
  - shell commands and inspection surface

## Core invariants

- every persisted event goes through `HistoryStore.record_event(...)`
- every persisted event is created through `events.create_event(...)`
- only `history.py` writes session files and runtime-controlled file outputs
- replay from `complete_history.jsonl` alone rebuilds state
- runtime model calls go through explicit budget checks
- tool execution is bounded and isolated from the live in-memory session object

## Runtime flow

1. rebuild or create session
2. record the incoming user message
3. analyze the prompt and create an explicit execution decision
4. expand the task if needed and select a strategy
5. create or resume an explicit plan
6. update working memory and project state
7. build context from selected recent history, semantic memory, plan, strategy, notes, and active entities
8. run a bounded subsystem-driven reasoning loop
9. on each step:
   - poll tracked background jobs and resolve any newly completed steps
   - select the responsible subsystem
   - build prompt and budget report
   - call model when needed
   - evaluate the result against the machine-checkable done condition
   - replan or recover on failure, inconsistency, or drift
   - if only background work remains, enter explicit waiting instead of faking completion
   - if control messages are queued, classify and apply them between model calls
10. record final answer and turn completion

## Evaluation architecture

Evaluation exposes two user-facing categories:

- deterministic correctness tests
- agent behavior tests

Agent behavior tests run in two modes:

- cached mode
- no-cache validation mode

- deterministic correctness tests
  - deterministic health checks for the repository and runtime
  - imports, smoke coverage, harness coverage, scheduler/runtime/history checks
  - expected to remain at `100%`
- agent behavior tests (cached mode)
  - runs the real runtime/orchestrator/tool loop with cassette-backed record/replay by default
  - keeps reruns fast once the cache exists
  - still allows tightly controlled scripted fixtures for focused support-check families
- agent behavior tests (no-cache validation mode)
  - tasks run through the real agent runtime with direct model calls enabled
  - scored by difficulty tier with per-task rubric breakdowns

Validation difficulty tiers:

- `extremely_easy`
- `easy`
- `normal`
- `hard`
- `extremely_hard`

The curated validation subset is intentionally balanced: each tier must keep at
least `10` distinct tasks, and validation fails if that floor regresses.

The category evaluator in `swaag.benchmark.evaluation_runner` writes:

- deterministic correctness JSON and markdown reports
- agent behavior tests (cached mode) JSON and markdown reports
- agent behavior tests (no-cache validation mode) JSON and markdown reports
- one combined report with:
  - deterministic correctness percent
  - agent behavior tests (cached mode) percent
  - agent behavior tests (no-cache validation mode) percent
  - no-cache validation per-tier percents
  - no-cache validation per-task percentages
  - no-cache validation rubric excerpts for the weakest tasks
  - one final overall percent

## Memory model

- episodic memory
  - the canonical append-only history file
- semantic memory
  - trusted or derived facts extracted from events
- procedural memory
  - strategy-like summaries extracted from plans
- working memory
  - current goal, current step, recent results, active entities
  - always derived from history; never authoritative

## History vs projections

Authoritative:
- `complete_history.jsonl`

Derived:
- `current_state.json`
- `notes.json`
- `reader_state.json`
- `history_index.json`

The system is correct if replay works from history alone. Projections exist only for faster inspection.

## Background job model

Background execution is deliberately narrow and explicit.

- only tools that declare `background=true` start detached work
- the runtime binds the resulting `process_id` to the owning running plan step
- that step is not marked complete until a later poll yields a terminal process state and verification passes
- dependent steps remain blocked until the owning step completes

Process lifecycle is persisted through history events:

- `process_started`
- `process_polled`
- `process_completed`
- `process_timed_out`
- `process_killed`
- `wait_entered`
- `wait_resumed`

Replay restores:

- tracked process records
- waiting state and waiting reason
- the relationship between background work and plan progress

This keeps the agent single-threaded at the semantic layer while still letting
one turn keep doing useful foreground work during long-running shell activity.

## Session and control model

Sessions now have two identifiers:

- stable internal `session_id`
- human-readable `session_name`

Resolution rules:

- no session argument => resume latest session, or create a new one
- explicit name => resume-or-create by that name
- rename updates the session index and history without changing the internal id

Active-session control is separate from the normal work plane:

- control messages are persisted to the session inbox immediately
- classification uses a dedicated structured control prompt
- the current task keeps running unless the control action is explicitly stop,
  cancel, replace, or conflicting enough to require clarification
- deferred follow-up work is stored as explicit session tasks instead of being
  silently merged into the current goal

## Exact-detail history queries

Generic retrieval is not enough for questions like:

- what exact command was run
- what path was written
- where a file was copied

`HistoryStore.query_history_details(...)` provides a dedicated path for these
questions:

- it resolves the target session explicitly
- scans canonical history as the source of truth
- ranks matches from full original event payloads, not just previews
- returns event payloads that the CLI can expose directly

Scoring constants (`token_score`, `exact_score`, `type_bonus`, `preview_chars`) are
config-backed via `[selection_policy]` and overridable per-deployment.

## Config system

All policy is in `src/swaag/assets/defaults.toml`. No magic numbers remain in
touched code for what a user would reasonably want to tune.

Override precedence (lowest to highest):
1. `defaults.toml` — packaged defaults, defines every key
2. `config/local.toml` — local override file, deep-merged at startup
3. `SWAAG_CONFIG` env var — explicit path to an additional override file
4. `SWAAG__SECTION__KEY=value` env vars — per-key overrides, highest priority

Config sections are ordered by practical tuning likelihood and risk
(most user-tunable near the top, most advanced/risky near the bottom):
- `[model]` — server/model facts that MUST match the running server
- `[runtime]` — operational limits: tool budget, step count, timeouts
- `[tools]` — which tools to enable
- `[planner]` — max plan steps and replan count
- `[context]` — token reserve policy
- `[sessions]`, `[environment]`, `[logging]` — session and shell config
- `[notes]`, `[reader]`, `[editor]`, `[memory]`, `[compression]` — subsystem limits
- `[security]`, `[retrieval]`, `[guidance]`, `[skills]`, `[prompts]` — system config
- `[budget_policy]` — advanced scale-free output/safety/section ratios
- `[context_policy]` — advanced context assembly priorities and token hints
- `[selection_policy]` — advanced retrieval weights, scoring text limits, skill delta

Three types of values — the distinction is explicit:
- **Server/model facts** (`model.context_limit`, endpoints): must match the running
  llama.cpp server; never treated as tunable
- **User-tunable policy** (ratios, weights, limits): safe to experiment with;
  all live in `defaults.toml`; none hardcoded in Python
- **Derived allocations** (`per-call token budgets`, `per-section budgets`):
  computed at runtime from `context_limit × policy`; never set directly

Key policy constants moved out of code:
- `budget_policy.safe_input_floor_tokens` (was `128` in budgeting.py)
- `selection_policy.retrieval_scoring_text_chars` (was `280` in embeddings.py)

## Artifact tracking

Plan steps are the unit of artifact expectation. `build_project_state` derives:
- `expected_artifacts` — all planned outputs (one per non-trivial step)
- `pending_artifacts` — expected artifacts from steps still pending or running
- `completed_artifacts` — expected artifacts from successfully completed steps

This state appears in the context bundle's `project_state` component and is
visible to the LLM during decision and answer calls. `_should_force_not_done_answer`
uses plan step statuses (the authoritative source) to block premature finalization
when non-respond steps are still pending, running, or failed.

## Retrieval structural weights

Structural weights in `[selection_policy]` control the shortlisting bias before
the LLM ranker applies semantic relevance scores:

- `retrieval_structural_tool_message` — advantage for tool-result messages (high signal)
- `retrieval_structural_user_message` — mild advantage for user messages
- `retrieval_structural_failed_event` — advantage for failure/error events
- `retrieval_structural_summary_event` — advantage for plan/summary events
- `retrieval_structural_modified_file` — advantage for recently modified files
- `retrieval_structural_procedural_memory` — advantage for procedural memory items
- `retrieval_trust_untrusted_memory` — trust discount for untrusted semantic items

All weights are zero-to-one and additive with the LLM's semantic score. They exist
to give the shortlister a signal about source quality without replacing the ranker.
