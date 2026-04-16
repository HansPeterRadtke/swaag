# Runtime Loop

## High-level loop

Each user turn is handled as:

1. `message_added` for the user input
2. `turn_started`
3. `prompt_analyzed`
4. `decision_made`
5. optional `task_expanded`
6. `strategy_selected`
7. `plan_created` or `plan_updated`
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
- semantic classification waits until the current model call finishes
- deterministic state transitions then decide whether to:
  - answer a status/session-summary query
  - add a non-destructive note or constraint
  - queue a deferred follow-up task
  - stop or cancel the current task
  - replace the current task
  - ask for clarification if the control intent is ambiguous

The default is conservative: side questions, progress questions, added
constraints, and clarifications do not stop the current task.

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
- for tool/file work:
  - record `tool_chain_started`
  - run one or more tool calls in an isolated copied session context
  - validate outputs
  - record generated events such as reads, notes, previews, edits, and writes
- evaluate the result with `evaluation_performed` / `evaluation_failed`
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

Background execution is explicit, not automatic.

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

- semantic classification is LLM-based
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
- the active response step returns a direct response
- background work is still running and the loop is in an explicit wait cycle
- `runtime.max_tool_steps` is reached inside a tool or file subsystem
- `runtime.max_reasoning_steps` is reached
- the same structured decision repeats too often
- evaluation fails and replanning is exhausted
- consistency or drift recovery fails to restore a valid state
- a decision call fails or returns malformed output, then the runtime falls back to a final answer call
- a budget failure prevents further decision calls

## Fallback behavior

The runtime does not spin indefinitely.
When the reasoning loop stops without a final answer from a response step, it performs one final plain-text answer call.
If that final call itself cannot fit the budget, the runtime raises `BudgetExceededError`.

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
- `error` when malformed model output or other runtime failures force fallback

## What is bounded

The runtime is bounded by:
- `runtime.max_reasoning_steps`
- `runtime.max_tool_steps`
- `runtime.background_poll_seconds`
- `runtime.max_repeated_action_occurrences`
- prompt admission checks
- summary compaction limits
