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
  - portable JSON-schema contract factories for constrained generation
- `swaag.schema_portability`
  - portable schema validation for production generation contracts
- `swaag.model`
  - llama.cpp and OpenAI-compatible transport adapters with generation-time
    JSON-schema constrained decoding
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
  - model-returned prompt-analysis contract parsing and validation
- `swaag.decision`
  - model-returned task-decision contract parsing and validation
- `swaag.expander`
  - model-returned task-expansion contract parsing and validation
- `swaag.strategy`
  - model-returned strategy settings without fixed workflow constraints
- `swaag.working_memory`
  - derived short-term task state
- `swaag.memory_semantic`
  - raw event snapshot storage and recency retrieval
- `swaag.project_state`
  - derived multi-file project-state model
- `swaag.context_builder`
  - deterministic context compiler plus optional model-scored retrieval traces
- `swaag.evaluator`
  - explicit step-result evaluation
- `swaag.security`
  - trust metadata and raw event-memory promotion policy
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
  - active-session control-plane model decisions and mechanical application
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
- every semantic decision is produced by the model and then mechanically
  parsed, validated, executed, recorded, and verified
- every live model call is constrained at generation time with a strict closed
  JSON-schema contract; semantic text returns `{"text": string}` and is unwrapped
  only after validation
- OpenRouter and OpenAI-compatible providers use Chat Completions Structured
  Outputs; llama.cpp uses top-level `json_schema`
- every generation schema must pass the portable-schema validator
- deterministic code must not infer intent, classify prompts, select semantic
  file roles, install fixed workflows, choose recovery approaches, or generate
  final semantic content
- planning, tool-choice, recovery, and semantic delegation prompts receive the
  complete enabled tool registry, including registered names, descriptions,
  full input schemas, and usage guidance
- planning, tool-choice, tool-input, recovery, and response prompts receive
  relevant observed evidence from prior history and tool results within the
  context budget; deterministic code may budget and mark trust boundaries, but
  must not reduce needed file text, command output, verifier evidence, or tool
  errors to metadata-only summaries
- recovery prompts and selected-tool argument prompts mechanically expose
  recent failed tool/verification evidence and latest observed file snapshots
  when available. If a failed source snippet, range, pattern, or other target is
  stale against the current artifact, Python reports the evidence and enforces
  execution validation; the model chooses the repair, reread, clarification, or
  blocker.
- task-decision `execution_mode=single_tool` never creates a deterministic
  direct-tool plan; it is planner context only, and objective verification must
  come from the model-returned plan
- schema-valid but locally invalid planner output is retried by the model with
  validation evidence; Python does not repair plan semantics or verification
  conditions, and the bounded repair loop is generous enough for multiple
  distinct structural corrections before failing closed
- generated plans use `verification_type="composite"` for every step; every
  check declares its own `condition="required"` or `condition="optional"` in the
  constrained model wire schema. Semantic answer or reasoning verification still
  requires at least one model-declared `criterion` check marked required, unless
  the model declares a required exact/string match against `assistant_text`;
  optional-only criteria are rejected because they cannot make semantic answer
  verification fail closed
- the parser mechanically converts each check's local status into the internal
  required and optional check-name lists used by execution and history. Legacy
  saved payloads with the old lists remain readable, but live constrained output
  cannot create a mismatched cross-reference. Python must not invent
  task-specific checks, expected content, tools, condition importance, or
  success criteria.
- plan `input_text` is model-authored instruction context, not executable tool
  arguments; selected-tool arguments are generated only by the later
  constrained tool-input call
- tool identity is exact. If a plan step declares `expected_tool`, the runtime
  must reject any different registered tool name and return mismatch evidence
  to the model; Python must not define equivalence tables or special cases for
  apparently similar tools.
- verification may expose the latest tool result under the current step's
  model-declared `expected_output`, `expected_outputs`, and `output_refs`
  labels; this is structural aliasing only, not content synthesis
- tools may register required objective verification check types. Runtime plan
  review requires the model plan to both declare and require a matching check;
  missing, optional-only, or inconsistent checks are rejected with validation
  evidence. Python does not promote optional checks, rewrite condition
  importance, or invent the expected path, text, command, or criterion.
- `edit_text` prefers portable `replace_exact` edits over raw offsets when the
  model has observed the current text: `old_text` must match exactly once,
  `new_text` is applied atomically, and zero or multiple matches fail closed.
  `replace_range` remains a low-level operation protected by `expected_text`.
- file-mutating tools require a constrained semantic result review after the
  mechanical objective checks pass. The model receives the declared step goal,
  success criteria, deterministic verification evidence, tool output, diff, and
  current file text, then decides under the `verification` JSON-schema contract
  whether `result_satisfies_step` is true. Python only enforces the gate,
  records rejection evidence, and retries or replans.
- required semantic review fails closed when its backend is unavailable,
  degraded, or violates the structured schema contract. Only an exact literal
  assistant-text match declared by the model can be checked without semantic
  scoring; non-literal relevance requires a constrained model verdict.
- final response completion requires a constrained model-owned
  `final_objective_satisfied` verdict over the original request, candidate
  answer, active plan, recent observations, and current workspace evidence.
  Failure is fed back through normal verification/replanning rather than
  reported as success from a narrowed later plan.
- if an earlier step-level objective check failed but the current workspace
  evidence may already satisfy the original request, the model may replan to
  observe the current state and answer. Runtime may mechanically record
  `unresolved_objective_verification_deferred`, but the final response still
  cannot complete without the model-owned `final_objective_satisfied` proof.
- unresolved artifact placeholders in actual side-effect tool arguments are
  rejected as malformed executable data; Python does not substitute semantic
  content for them
- skills add instructions and metadata only; they must not hide, remove, or
  preselect tools
- the runtime must keep working through actions, observations, history,
  recovery, and verification until verified success, explicit clarification, or
  a genuine blocker/watchdog stop; a reasoning phase ending is not completion
- unresolved watchdog or safety exits return a mechanical incomplete status
  instead of a semantic final answer
- benchmark success is not architectural success; see
  `doc/model_semantic_authority.md`

## Runtime flow

1. rebuild or create session
2. record the incoming user message
3. ask the model for prompt analysis and task decision contracts
4. ask the model for task expansion if the model requested expansion, then ask
   the model for strategy settings
5. create or resume an explicit plan
6. update working memory and project state
7. build context from selected recent history, raw memory snapshots, plan,
   strategy, notes, and active entities
8. run a bounded subsystem-driven reasoning loop
9. on each step:
   - poll tracked background jobs and resolve any newly completed steps
   - select the responsible subsystem
   - build prompt and budget report
   - call model when needed through constrained JSON-schema contracts
   - for tool work, ask the model to select the tool from the complete enabled
     registry, then ask the model for selected-tool arguments with that tool's
     registered documentation and schema
  - evaluate the result against the machine-checkable done condition
  - for registered file-mutation tools, run constrained semantic result review
    before the step can complete
  - replan or recover on failure, inconsistency, or drift
   - if only background work remains, enter explicit waiting instead of faking completion
   - if control messages are queued, ask the model for the control action and
     mechanically apply the selected state transition
10. record a verified final answer, or record a transparent incomplete status
    when a safety bound stops unresolved work

## Evaluation architecture

Evaluation exposes two user-facing test categories:

- `code_correctness`: deterministic software-correctness checks.
- `agent_test`: cached agent behavior checks.

Manual validation / real usage is not a test category. It is cache-first by
default; explicit uncached execution requires `--uncached-live`. The category evaluator writes separate JSON and markdown reports for
`code_correctness`, `agent_test`, and the combined fail-fast result. The
correctness category is binary; the agent category reports real benchmark
quality directly instead of a pytest wrapper status.

Model-output caching is a runtime invariant, not just a test fixture. All semantic subsystems share the same constrained model client. The cache hash includes the complete output-affecting request payload plus model/server identity and caller scope: prompt or messages, strict JSON Schema, seed, temperature and other supplied sampling controls, token limit, stop sequences, endpoint, profile, and model fingerprint. Cache writes are atomic and process-safe, and identical concurrent misses are deduplicated. An offline replay may reuse a recorded model fingerprint only when all non-probed identity fields match exactly.

The authoritative cached benchmark catalog behind `agent_test` is full-catalog
and record-replay backed. It currently contains 59 realistic tasks across six
families (`coding`, `file_edit`, `reading`, `multi_step`, `failure`, `quality`)
and all five difficulty tiers, with intentionally asymmetric tier counts so the
hardest tiers can carry richer scenarios instead of symmetrical filler. Verifiers are
programmatic: executable test commands, exact file expectations, structured JSON
checks, allowed-modified-file locks, and anti-tamper guards carry the benchmark
instead of hidden scripted model answers.

The benchmark content is intentionally heterogeneous. Hard and extremely hard
tasks are not just larger copies of easy tasks: they add multi-file
dependencies, shell/environment workflows, contradiction handling, stale-source
rejection, unsafe-plan refusal, repeated-action traps, and iterative
correction loops.

`agent_test` result reporting is score-based. It surfaces:

- total tasks / successes / failures / false positives
- full-task success percentage
- difficulty-group scores and difficulty-group average
- family-group scores and family-group average
- top-level group-average score
- average task score

Manual validation keeps the five difficulty tiers for real-model task scoring:

- `extremely_easy`
- `easy`
- `normal`
- `hard`
- `extremely_hard`


## Memory model

- episodic memory
  - the canonical append-only history file
- event memory
  - raw, bounded event snapshots retained for context and audit
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
- action selection uses a dedicated structured control prompt
- the current task keeps running unless the control action is explicitly stop,
  cancel, replace, or conflicting enough to require clarification
- deferred follow-up work is stored as explicit session tasks instead of being
  silently merged into the current goal

## Exact-detail history queries

Generic retrieval is not enough for questions like:

- what exact command was run
- what path was written
- where a file was copied

`HistoryStore.query_history_details(...)` provides a mechanical path for these
queries:

- it resolves the target session explicitly
- scans canonical history as the source of truth
- ranks matches from full original event payloads using configured lexical
  detail-query weights
- returns event payloads that the CLI can expose directly

This exact-detail query is an inspection utility over persisted machine events.
It is not used to infer user intent, select strategy, choose tools, or answer
semantic task questions.

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

Artifact labels are bookkeeping and context for the LLM. They are not a runtime
substitution language. If model output puts an unresolved placeholder such as
`{{artifact_name}}` where concrete executable input is required, validation
rejects it and recovery returns to the model.

This state appears in the context bundle's `project_state` component and is
visible to the LLM during decision and answer calls. `_should_force_not_done_answer`
uses plan step statuses (the authoritative source) to block premature finalization
when non-respond steps are still pending, running, or failed.

## Retrieval Boundaries

Retrieval shortlisting may use mechanical source-quality signals and bounded
text excerpts, then optional model scoring provides semantic relevance. Offline
`unavailable` mode returns neutral scores; it does not fall back to keyword,
regex, TF-IDF, filename, or language-specific scoring.

If the model-scored retrieval backend violates its structured relevance schema
after bounded attempts, the runtime records `semantic_retrieval_degraded` and
rebuilds context with `unavailable` retrieval. This is a mechanical continuity
fallback, not a semantic scoring replacement.

Structural weights in `[selection_policy]` control source-quality bias before
the model ranker applies semantic relevance scores:


- `retrieval_structural_tool_message` — advantage for tool-result messages (high signal)
- `retrieval_structural_user_message` — mild advantage for user messages
- `retrieval_structural_failed_event` — advantage for failure/error events
- `retrieval_structural_summary_event` — advantage for plan/summary events
- `retrieval_structural_modified_file` — advantage for recently modified files
- `retrieval_structural_procedural_memory` — legacy weight for stored event-memory items
- `retrieval_trust_untrusted_memory` — legacy trust discount for stored event-memory items

All weights are zero-to-one and additive with the model's semantic score. They
exist to give the shortlister a source-quality signal without replacing the
model ranker.


## Test Categories

SWAAG exposes exactly two authoritative test categories:

- `code_correctness`: deterministic mechanical checks with no model-server dependency.
- `agent_test`: cached agent behavior checks using real model responses through the record/replay cache.

Manual validation / real usage is not a test category. `python3 -m swaag.benchmark manual-validation ...` replays exact cache hits and records misses by default; `--uncached-live` explicitly bypasses that cache.
