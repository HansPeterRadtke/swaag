# Runtime Loop

## High-level loop

Each user turn is handled as:

1. `message_added` for the user input
2. `turn_started`
3. `prompt_analyzed` from a model contract
4. `decision_made` from a model contract
5. optional model-returned `task_expanded`
6. model-returned `strategy_selected`
7. model-returned `plan_created` or `plan_updated`
8. `working_memory_updated`
9. optional `project_state_updated`
10. `reasoning_started`
11. repeated subsystem-driven reasoning steps, bounded by config
12. optional background-process polling / wait / resume cycles
13. `reasoning_completed`
14. final assistant answer
15. `turn_finished`

If a session is already active, new user input first enters the control plane:

- the message is queued immediately
- model control-action selection waits until the current model call finishes
- deterministic state transitions then mechanically apply the model-selected
  action to:
  - answer a status/session-summary query
  - add a non-destructive note or constraint
  - queue a deferred follow-up task
  - stop or cancel the current task
  - replace the current task
  - ask for clarification if the control intent is ambiguous

The control contract may choose to preserve or change the current task.
Deterministic code records and applies that selected action; it does not infer
control intent from wording.

## Reasoning steps

For each reasoning step:
- poll tracked background jobs first
- if any background job completed:
  - record process lifecycle events
  - verify the owning step against the final tool result
  - complete or fail the owning step
  - unblock dependent work
- find the next executable plan step
- transition it through `step_started`
- dispatch to one explicit subsystem:
  - planning subsystem
  - reasoning subsystem
  - tool subsystem
  - file subsystem
- build context, build a budget report, and call the model only through the guarded runtime path
- if model-scored retrieval violates its structured contract, record
  `semantic_retrieval_degraded` and rebuild the context with neutral
  `unavailable` retrieval instead of applying keyword or filename relevance
- include relevant observations from prior tool results and generated events in
  later planning, tool-choice, tool-input, recovery, and response prompts within
  the configured context budget. Read-file text, command output, verifier
  evidence, and tool errors are semantic evidence for the model; they must not
  be replaced by metadata-only summaries when the next model decision depends
  on their content.
- on recovery after failed or incomplete mutations, expose recent failed tool
  inputs, validation errors, verification evidence, and latest observed file
  snapshots as mechanical evidence. If current observations show that an older
  source snippet, range, pattern, or path state is stale, the runtime may reject
  repeated invalid execution and ask the model to continue; it must not infer
  the repair or downgrade the exact requested final state.
- include the complete enabled tool registry, with registered names,
  descriptions, full input schemas, and usage guidance, in every semantic call
  that can select, preselect, rank, constrain, or delegate tool-capable work,
  including task decision, planning, direct/tool routing, replanning, recovery,
  and subagent selection.
- for tool/file work:
  - record `tool_chain_started`
  - ask the model to choose the tool from the complete enabled registry
  - ask the model for the selected tool's arguments using that tool's registered
    description, complete input schema, and usage guidance
  - reject a non-tool action or a tool that does not match the
    model-authored step's `expected_tool`, record the rejection as
    model-visible execution evidence, and continue within watchdog bounds so
    the model can choose a valid next action. Matching is by exact registered
    tool name; the runtime does not treat tools as interchangeable.
  - detect exact repeated structured actions within the current step using the
    tool name, closed input object, response field, step id, and expected-tool
    binding; this guard ignores mutable edit/note counters so an identical
    side-effect write cannot loop merely because it changed history state
  - run one or more tool calls in an isolated copied session context
  - validate outputs
  - record generated events such as reads, notes, previews, edits, and writes
- evaluate the result with `evaluation_performed` / `evaluation_failed`
- before accepting a response step as final success, run constrained
  model-owned final objective verification against the original request,
  candidate answer, active plan, recent observations, and current workspace
  evidence
- if a tool starts explicit background work:
  - bind the process id to the running step
  - leave that step `running`
  - keep the process in environment state until completion
- if the done condition is satisfied:
  - record `step_completed`
- otherwise:
  - record `step_failed`
  - optionally record `replan_triggered`

## Background execution and waiting

Background execution is explicit, not automatic. A foreground `run_tests`
result may intentionally be diagnostic: coherent failing output is structurally
valid unless the model explicitly requires a `command_success` objective check.

- today the practical backgroundable tools are:
  - `shell_command`
  - `run_tests`
- the tool input must request `background=true`
- the owning plan step stays `running` until the background process is polled to a terminal state
- dependent steps stay blocked until that verification succeeds

When no foreground work is ready but background work is still running:

- the orchestrator selects `wait`
- the runtime records `wait_entered`
- the loop sleeps for `runtime.background_poll_seconds`
- polling continues without busy looping
- when the background job changes state or disappears from the running set:
  - the runtime records `wait_resumed`
  - dependent work becomes eligible again if verification passes

The waiting state is persisted in history and rebuilt into `EnvironmentState` so
replay can explain why the runtime was blocked.

## Control messages and deferred tasks

Active-session control uses a dedicated structured control contract.

- control action selection is model-based
- legality and state transitions are deterministic
- `continue_with_note` records a note without stopping the current task
- `queue_after_current` appends a deferred task to session state
- `replace_task` installs a replacement goal and replans
- `stop` / `cancel` stop the current task explicitly

Deferred tasks are session-scoped. `python3 -m swaag ask` with no prompt
consumes the oldest deferred task if one exists; otherwise it resumes the
latest session and expects normal user input.

## Stop conditions

The reasoning loop stops when one of these happens:
- the active response step returns a response, that response verifies, and
  final objective verification accepts the whole requested result
- background work is still running and the loop is in an explicit wait cycle
- `runtime.max_tool_steps` is reached inside a tool or file subsystem
- `runtime.max_reasoning_steps` is reached as a watchdog
- the same model-returned structured decision repeats too often without state
  progress
- evaluation fails and replanning/retry bounds are exhausted
- consistency or drift recovery fails to restore a valid state
- a decision call fails or returns malformed output and the runtime reaches a
  fatal structured-call error or a transparent incomplete state
- a budget failure prevents further decision calls

## Fallback Behavior

The runtime does not spin indefinitely, but safety bounds are not success
conditions. If the loop stops without a verified response step, SWAAG returns a
mechanical incomplete-status message that names the stop reason and states that
verified success was not reached. It does not perform a fresh semantic answer
call for unresolved work, because that can convert failed execution into a
normal-looking final answer.

Semantic final answers are generated only as response/reasoning steps and must
pass their step verification plus whole-goal final objective verification before
they are exposed as completion.

`execution_mode=single_tool` from task decision is not a stop condition and not
a direct execution shortcut. It only adds planner context; the planner model
must still return an explicit plan with objective verification checks.

Planner validation failures are retryable semantic failures until the bounded
planner repair budget is exhausted. If the planner returns schema-valid JSON
that fails local plan validation, the runtime records `plan_validation` evidence
and asks the planner model for a corrected plan under the same closed schema.
Python must not repair the plan, fill missing required conditions, choose
verification checks, or rewrite semantic steps. The retry bound is a generous
deadlock guard; distinct contract-shape corrections must not be cut off merely
because planning has taken wall-clock time.
For semantic response or reasoning verification, the model must declare and
require at least one `criterion` check under composite verification, unless it
declares a required exact/string match against `assistant_text`. Optional-only
criteria are rejected during planning so the runtime does not execute a
semantic answer step that cannot fail closed.
When a required semantic reviewer backend is unavailable, degraded, or returns
output that violates its constrained schema, verification fails and the evidence
returns to the loop for model-owned recovery. Exact literal assistant-text
matches declared by the model are the only reviewer path that can pass without
semantic scoring.

The live constrained plan schema puts `condition="required"` or
`condition="optional"` directly on every model-authored verification check and
derives step completion conditions from kind and exact expected tool. It omits
the former dedicated objective slot and does not allow the model to emit
`tool_effect_verified`. The parser mechanically converts local statuses into
internal required and optional check-name lists. Runtime then installs any exact
automatic mechanical check registered by the selected tool before plan review.
This does not change model-authored importance or create task-specific semantic
content. Exact duplicate checks collapse; conflicting same-name checks fail.
Legacy saved payloads with separate lists and objective slots remain readable.

Planner `input_text` is instruction context, not an executable argument object.
A model may name dependencies through `input_refs`/`output_refs`, and may refer
to placeholder-looking labels in step instructions. The later selected-tool
input call is the only place actual tool arguments are generated. Side-effect
tool inputs must contain concrete values; if the model returns
`{{artifact_name}}` as an actual side-effect argument value, the runtime records
validation evidence and lets the model replan or retry. Python does not
resolve semantic placeholders or treat them as content.

Visible editor backup files are disabled by default. File mutation tools record
original text, diffs, and write provenance in history; they must not create
surprise workspace files unless an explicit operator policy enables backups.
substitute the artifact content.
Plan dependency edges are model-authored and mechanically validated. In an
initial plan, every `depends_on` entry must name an earlier step in that plan.
During replacement planning, a dependency may also name a completed prior step
listed in replan evidence; completed identifiers are reconstructed from canonical
`step_completed` events for the current turn rather than only from the latest
replacement plan. Runtime validates such an edge as already satisfied and strips
it before topological sorting and execution. Unknown external dependencies,
self-dependencies, and cycles among replacement-plan steps remain invalid and
are returned to the planner as validation evidence.
For verification only, the latest tool result is also available through the
current step's model-declared `expected_output`, `expected_outputs`, and
`output_refs` labels. This lets the model's `artifact_present` checks reference
its own labels without Python choosing the label meaning.

Plan review enforces tool-declared logical output cardinality. `read_file` reads
one file per call, so one step cannot claim multiple independent file-output
references; multi-file inspection uses separate ordered steps. Respond and
reasoning steps reject tool-result checks because their evidence source is
assistant text. A `run_tests` step whose success criteria require passing tests
must require `tool_result_success`, which checks its actual structured result without rerunning the command; `tool_output_nonempty` without command success is
reserved for an explicitly diagnostic baseline whose failure is acceptable.
Plan semantic review receives compact recent event projections. Prior
`review_completed` events contribute only review kind, target, role, pass/fail,
and reason; their embedded review evidence is never reinserted into a later
review prompt. This prevents recursive evidence growth across repeated replans
while preserving the outcome needed for recovery.
Some tools register an automatic mechanical objective-verification type.
Runtime installs that exact registered check after parsing and makes it required;
the model does not emit or name it. The live plan wire omits
`tool_effect_verified` and `file_contains`. `edit_text` and `write_file` register
persisted-hash checks that prove the current file matches the tool result and a
real mutation occurred. The model uses `tool_result_success` for the actual `run_tests` call and adds `command_success` only for a distinct
independent executable correctness test; constrained mutation review and whole-goal review
retain semantic authority. Python does not promote optional checks, rewrite
condition importance, or invent expected content. Legacy saved containment
checks remain readable, resolve matching relative paths from the latest tool
result, and reject empty targets mechanically.
For observed text edits, the selected-tool argument call should use
`edit_text` operation `replace_exact`: the model supplies `old_text` and
`new_text`, the editor requires exactly one literal match, dry-run previews
report objective match evidence, and zero or multiple matches are explicit
errors. `replace_range` remains available only as a guarded low-level operation
with `expected_text`.
For edit execution, an absent replacement pattern is also a mechanical failure,
not an "already applied" success inferred from replacement text appearing
elsewhere in the file. The tool error includes current file text so the model
can choose the recovery edit.
After a registered file-mutating tool passes those mechanical checks, runtime
must still run constrained semantic result review before completing the step.
The review prompt includes the model-declared goal and criteria, deterministic
verification evidence, the tool result and diff, and the current file text. A
failed `result_satisfies_step` verdict is recorded as a model-visible
observation and the loop retries or replans under the normal recovery policy.
This prevents broad substring checks from becoming deterministic semantic
success while keeping the actual semantic judgment with the model.

After the final response text is generated and step verification passes, runtime
asks the verifier model for a `final_objective_satisfied` verdict through the
same portable `verification` contract. The evidence is assembled mechanically:
original request, candidate answer, active plan, recent tool/verification
events, and current workspace state. If the verdict is false, the response step
fails and normal recovery/replanning continues while budgets allow.
When a previous step-level objective check remains unresolved, a recovery plan
does not have to repeat the mutation if the model decides current observations
already satisfy the request. Runtime may defer that unresolved proof to the
mandatory final objective verifier and records
`unresolved_objective_verification_deferred`; a final answer still fails closed
unless `final_objective_satisfied` passes.

A response returned during a plan step that requires a tool is not a successful
answer and does not complete the step. Likewise, a tool choice that conflicts
with the model-authored `expected_tool` is not executed. The runtime records a
`tool_mismatch_rejected` event with the selected action/tool and expected tool,
then lets the model choose again under the existing repeated-action and
tool-step watchdogs. This enforces the model-authored plan without Python
selecting a recovery tool or inventing answer content.

The repeated-action guard is mechanical. It compares exact structured actions,
not intent. If a preview has already failed and the model repeats the same
side-effect action for the same step, the subsystem hands off to normal step
verification instead of executing the same mutation until the tool-step
watchdog expires. Different model-authored arguments are still allowed as
legitimate iterative refinement, and any failed verification is returned to the
model for recovery or replanning.

## Recorded reasoning events

The loop records:
- `prompt_analyzed`
- `decision_made`
- `task_expanded`
- `strategy_selected`
- `plan_created`
- `plan_updated`
- `plan_completed`
- `subsystem_started`
- `subsystem_progress`
- `subsystem_completed`
- `tool_chain_started`
- `decision_parsed`
- `tool_input_parsed`
- `tool_chain_step`
- `tool_chain_completed`
- `process_started`
- `process_polled`
- `process_completed`
- `process_timed_out`
- `process_killed`
- `wait_entered`
- `wait_resumed`
- `control_message_processed`
- `control_action_applied`
- `deferred_task_queued`
- `deferred_task_consumed`
- `code_checkpoint_created`
- `code_checkpoint_restored`
- `working_memory_updated`
- `project_state_updated`
- `context_built`
- `semantic_retrieval_degraded`
- `reasoning_started`
- `step_executed`
- `step_completed`
- `step_failed`
- `evaluation_performed`
- `evaluation_failed`
- `replan_triggered`
- `drift_detected`
- `recovery_triggered`
- `consistency_checked`
- `consistency_failed`
- `reasoning_completed`
- `error` when non-semantic runtime failures force fallback
- `fatal_system_error` when a core constrained semantic model call violates its
  enforced JSON-schema contract

## What is bounded

The runtime is bounded by:
- `runtime.max_reasoning_steps`
- `runtime.max_tool_steps`
- `runtime.background_poll_seconds`
- `runtime.max_repeated_action_occurrences`
- prompt admission checks
- summary compaction limits
When failed-test recovery exhausts model-authored plan retries, runtime compiles
a bounded fail-closed recovery plan from observed facts. It reads up to three
recently changed files, requests one corrective edit per file using the exact
failed stdout and stderr, reruns the exact failed command, and permits a final
response only after `tool_result_success`. This fallback is structurally
validated without another semantic planning call.
