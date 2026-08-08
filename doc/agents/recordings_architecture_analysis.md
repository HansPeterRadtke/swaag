# Agent Architecture from the Voice Recordings

This document consolidates the agent-system design described in the voice recordings under `/data/var/voice_agent_live_runtime/recordings/text/agents` and compares that design with the current SWAAG implementation. The recordings are the source for the intended behavior; the code is evaluated against them rather than used to reinterpret them.

The current text set contains four recordings. `Recording 2026-08-04 10-27-01.txt` contains only “Eh.” and has no architectural content. `Recording 2026-08-04 10-36-47.txt`, `Recording 2026-08-06 17-47-36-666.txt`, and `Recording 2026-08-07 12-43-50-515.txt` contain the substantive design. A rerun backup of the August 7 recording exists under `text_rerun_backup`; its wording and punctuation differ slightly because of transcription, but its architecture and intent are materially the same as the current transcript.

## Core design principle: context construction is the central problem

The recordings repeatedly identify context filling as the main technical problem in the entire agent system. If context were unlimited, the ideal implementation would be simple: retain the complete detailed history forever and derive every future model call from it. Since context is finite, the system has to decide what information stays verbatim, what can be summarized, what can be omitted from the immediate prompt while remaining retrievable, and how much space must remain for the model’s output.

This is not described as a simple “recent messages detailed, older messages summarized” rule. “Recent” can mean one message, a handful of tool interactions, or a very long chain of exact actions and results. The correct amount depends on the task. Some work requires the exact wording of the user, exact code fragments, exact tool parameters, or a long causal chain of previous steps. Other work does not. The recordings therefore reject a fixed deterministic definition of relevance.

The intended split of authority is clear. Python should own mechanical invariants such as exact token accounting, context ceilings, output reservation, persistence, schema validation, and execution. The model should own semantic judgments such as what is important, which old details matter, what can be summarized safely, and which tool output is relevant to the next decision. Deterministic code cannot reliably infer semantic importance from size, recency, event type, or other shallow features.

### Current SWAAG status

This principle is substantially implemented. SWAAG performs exact or conservative token accounting for each call, reserves output tokens, applies a safety margin, includes the structured-output schema in admission accounting, and refuses a call when the exact prompt cannot fit. Context overflow triggers bounded compaction rather than silent truncation. This is strongly aligned with the recordings’ highest mechanical rule: never overflow the model context.

Compaction is model-authored. The summary prompt explicitly asks the model to preserve goals, constraints, facts, decisions, file paths, tool results, unresolved work, and exact wording when correctness depends on it. The model can also request a bounded number of recent source messages to remain verbatim through `preserve_recent_messages`. Python validates the requested count and remains authoritative over the hard token limit.

The model can now recover arbitrary older exact events through first-class `history_search` and `history_window` tools. Search returns bounded ranked previews and authoritative sequence references; the window tool then returns exact durable event payloads around a selected sequence. This preserves the recordings’ intended division of responsibility: Python bounds and indexes retrieval mechanically while the model decides which old detail matters. Summaries remain navigation aids rather than authorities, and an old exact tool result or user statement can be promoted back into the active reasoning context on demand.

## Large tool output and semantic triage

The August 6 recording identifies a particularly difficult version of the context problem: a command can unexpectedly produce an enormous amount of output. The system cannot simply put arbitrary stdout into the model context forever, but deterministic code also cannot know which portion of that output matters. A huge directory listing may be useless noise in one situation and crucial evidence in another.

The recording proposes model-mediated classification or summarization of tool output as a possible solution. The important point is not the exact category scheme. The important point is that semantic importance should not be guessed by deterministic code.

### Current SWAAG status

SWAAG partially mitigates output growth through bounded file-reading tools, structured tools, history compression, exact prompt admission, and background-process polling. However, generic shell output itself does not have a general semantic triage stage before it becomes part of the agent transcript. There is also no general shell-output byte or token admission layer that asks the model what to preserve before the output can dominate the next action call.

This is now implemented through durable text artifacts. Shell, structured test, and completed background-process output is kept bounded in immediate tool/history payloads; whenever output exceeds the capture limit, the exact full stream is persisted under the session runtime tree with its SHA-256, total character count, and artifact ID. The model can use `read_artifact` to retrieve exact bounded slices by offset. Python therefore decides only how much fits immediately, while the model decides whether and where to inspect the raw output.

## History must be authoritative and always recoverable

The recordings treat history as the core state of the agent. Every user message, model decision, tool call, argument, result, and important system event should be recorded. Recent history should often remain detailed, older history may become compressed for prompt efficiency, but the complete history must continue to exist as the source of truth.

The ideal is append-only durable history with derived projections. Summaries and working notes are secondary structures for navigation and context construction, not authorities that replace exact events.

### Current SWAAG status

This is strongly implemented. SWAAG uses append-only event history and can rebuild session state from it. Exact user messages and tool results are explicitly described in the prompt as authoritative. Durable notes are explicitly labeled as navigation aids rather than authorities. Session projections are derived from history rather than treated as primary state.

Model autonomy over historical retrieval is now implemented. `history_search` exposes ranked bounded discovery and `history_window` exposes exact event windows, both against the same append-only history used for rebuilding session state.

## The original user request is the highest semantic authority

The August 7 recording places unusually strong emphasis on instruction fidelity. The agent should do what the user actually asked, not what a hidden planner, reviewer, policy heuristic, or deterministic task classifier decides the user should have asked. Vague prompts necessarily require the model to choose reasonable concrete details, but those details must remain subordinate to the user’s request rather than becoming a substitute objective.

The recording also warns against apparently safe generic rules that accidentally block legitimate work. A rule such as “never make anything up” sounds sensible until the user says “make a game,” which necessarily requires creative choices. The correct top-level principle is therefore not a long set of rigid semantic rules. It is fidelity to the user’s actual request, with the model using judgment where the request leaves freedom.

### Current SWAAG status

The current action architecture matches this direction well. SWAAG no longer has the deleted planner/subagent architecture. It uses one constrained action loop in which the model selects the next action. The action prompt states that the original request, later user messages, and exact tool results are authoritative, and explicitly says that the runtime does not classify, plan, review, or decide the task for the model. The prompt assembly also includes the original user request verbatim and labels it authoritative on every action call. New user interventions are likewise included verbatim and authoritative.

There is, however, a prompt-layer inconsistency. The always-present system prompts are still minimal. The standard system prompt says that SWAAG is a local Python agent, should be concise and factual, should use tools when useful, and should never invent tool results. The lean prompt is even shorter. Neither system prompt currently states the recording’s central semantic rule that the user’s request is the highest task authority and that hidden system planning should not replace it. That rule exists in the action-template layer, not at the strongest always-present prompt layer.

The recordings specifically suggest that a very small number of foundational instructions may need to appear in every context. The current implementation should therefore be considered partially aligned, not complete. A concise always-present semantic charter should likely state that the user’s actual request and later corrections are authoritative, exact observed results outrank guesses, and the model should make its own semantic task decisions within the mechanical constraints of the runtime. This should remain compact so it does not itself become context noise.

## No fixed planner: constrained general action selection

The August 6 recording rejects a fixed plan as a central architecture. If the model receives a new machine and does not know the operating system, it should be able to discover that by taking appropriate actions. The system should not need a deterministic planner that has already classified the task and produced a workflow. Instead, constrained decoding should expose a generalized action space with enough capability for the model to decide what happens next.

### Current SWAAG status

This requirement is implemented well. The old planner, strategy, subagent, review, and related hidden semantic subsystems have been removed from the current architecture. The active contract is a simple agent action containing an assistant message, zero or more constrained tool calls, and a continue/stop decision. The model receives the complete enabled tool registry and chooses what to do next.

Python still performs mechanical validation and, in a few places, enforces verification workflow constraints. Those constraints should remain mechanical and evidence-driven. They should not evolve back into task-specific planning logic.

## Tool philosophy: broad capability, but avoid unnecessary bespoke tools

The August 6 recording says that an agent needs a very broad set of possible actions and that there are never truly “enough” tools. At the same time, it argues against reimplementing every ordinary operating-system capability as a bespoke agent tool. Linux already has mature commands, help text, and man pages. A general shell can expose enormous capability without putting hundreds of custom schemas into every prompt.

The intended custom-tool boundary is therefore not “everything becomes a tool.” A custom tool is most justified when it provides something that is difficult, unsafe, stateful, or awkward to express through a normal command line, or when the system needs structured execution semantics that shell output alone cannot provide.

### Current SWAAG status

SWAAG currently exposes a fairly large bespoke tool set: file listing and reading, repository search, text editing, file writing, diff inspection, workspace snapshots, structured test execution, notes, browser actions, process polling and killing, waiting, and wakeup management, in addition to the shell.

This is only partially aligned with the recordings’ shell-first philosophy. Some of the custom tools have strong justification. `run_tests` gives structured pass/fail evidence. Durable notes and wakeups are agent-state functions rather than ordinary shell operations. Process tracking integrates with persistent agent state. Bounded readers and editing tools provide exact event generation and deterministic workspace evidence. Those benefits are real.

Other tools overlap capabilities already available from the shell. That duplication costs prompt space because the complete enabled tool registry is included in every action call. The architecture should continually justify bespoke tools against this cost. A useful rule is: prefer the shell for ordinary operating-system functionality; keep custom tools when they materially improve persistence, structured verification, bounded retrieval, browser integration, agent memory, or another runtime invariant.

The current benchmark-driven requirement that test runners use `run_tests` rather than generic shell execution is an example of structured verification being deliberately separated from ordinary shell use. That is mechanically useful, but it should not expand into a general prohibition against normal command-line autonomy.

## The shell must eventually behave like a real terminal

The recording explicitly calls interactive tools a major topic and says the command-line tool deserves substantial engineering effort. A genuinely general agent terminal eventually needs more than one-shot command execution: persistent state, long-lived processes, polling, stdin, terminal or pseudo-terminal behavior when required, and a way to discover command documentation such as `man` or `--help` without special system knowledge.

### Current SWAAG status

The shell implementation supports non-interactive commands through a persistent logical shell state, background execution, process polling, killing, workspace snapshots, and tracked environment/cwd changes. This is useful and much stronger than a stateless subprocess wrapper.

SWAAG now also exposes a separate persistent `terminal` tool for cases that genuinely require interactive state rather than ordinary one-shot shell execution. It is backed by a detached PTY worker, supports terminal IDs and names, persistent shell cwd/environment across tool calls, incremental bounded output reads, later stdin delivery to interactive child processes, listing, and process-group close semantics. `shell_command` remains the preferred lower-cost interface for ordinary non-interactive work.

The shell also does not need a dedicated man-page tool: the model can already invoke `man`, `--help`, `info`, or other local documentation through `shell_command` when those commands are available. That matches the recording’s preference for using native operating-system documentation rather than duplicating it inside agent schemas.

## Waiting and durable wakeups

The August 4 recording asks for both short waiting and long durable wakeups. Short waits may simply block for a period. Long waits should be expressed in human-readable units and may target either a relative duration or a specific date and time. The recording explicitly considers seconds, minutes, hours, days, weeks, and months, and discusses the possibility of long-running background agents. It is skeptical about surprising multi-year wakeups but treats long duration support as conceptually possible.

### Current SWAAG status

Durable wakeup parsing supports seconds, minutes, hours, days, weeks, months, and years, as well as timezone-aware absolute ISO-8601 times. Wakeups are persisted outside the process, can be listed and cancelled, and survive process restart. Due wakeups become control messages and are recorded into authoritative history when claimed.

Synchronous waiting is less expressive. `wait_seconds` accepts only a numeric number of seconds and is bounded by the tool timeout. This is appropriate for short waits, but it does not implement the human-readable-unit interface described in the recording. Milliseconds are not supported by the durable duration parser either.

Durable wakeups now have an autonomous dispatcher. The wakeup store uses file locking and atomic replacement, separates scheduled/claimed/delivered states, supports reclaiming stale claims after a crash lease, and queues a deterministic control ID before marking delivery. The dispatcher discovers due sessions and resumes the existing session through a control-only runtime turn; scheduler controls are not persisted as fake user messages, so the original user request remains the semantic authority. A systemd deployment unit is included for continuous dispatch.

## Semantic authority versus deterministic enforcement

The recordings repeatedly return to one dividing line: deterministic code should not make semantic decisions it cannot actually justify. Importance, relevance, task meaning, and whether a piece of output can safely be summarized are semantic questions. Context size, file paths, timestamps, process status, exact token count, JSON validity, and whether a command returned zero are mechanical questions.

### Current SWAAG status

The present architecture is much closer to this boundary than the old planner-era implementation. The model chooses actions and authors summaries. Python validates schemas, tracks the workspace, executes tools, stores events, enforces context limits, and verifies objective test outcomes.

There are still places where mechanical recovery policy can become overly prescriptive. For example, the runtime can require diagnostic actions after failed structured verification. That rule is defensible as workflow safety because it is triggered by an objective failure and does not decide what the semantic diagnosis should be. Nevertheless, such rules should remain minimal. They must not grow into hidden task-specific planning or semantic classification.

The benchmark’s deterministic verifier should likewise provide evidence back to the same general action loop rather than act as an external semantic planner. Verification feedback should say what objectively failed and let the model decide the repair.

## Recommended architecture direction

The recordings imply an architecture with a small deterministic kernel and a model-controlled semantic loop. The durable kernel should own append-only history, exact model-call accounting, tool schemas, execution, persistence, process state, model/server identity, wakeup storage, and hard context ceilings. The model should own task interpretation, action choice, relevance, history/detail retrieval decisions, output triage, and summary content.

The highest-value architecture work remains context quality rather than adding another planner. The foundational pieces identified in the recordings are now present: exact history retrieval, durable raw-output artifacts with model-controlled inspection, a persistent interactive PTY terminal, an autonomous crash-safe wakeup dispatcher, human-readable waits, and the compact user-authority charter in every system prompt. Future changes should concentrate on measuring and improving semantic context selection while keeping these interfaces small and mechanically bounded.

The tool registry should also be reviewed continuously for prompt cost. Custom tools should remain when they provide runtime invariants or structured evidence that the shell cannot provide cleanly. Ordinary operating-system behavior should default to the shell rather than being duplicated automatically as more schemas.

## Current implementation verdict

The current SWAAG architecture is directionally close to the recordings and substantially closer than the removed planner-era design. Its strongest matches are the single model-controlled action loop, verbatim authoritative user input, exact durable history, exact context admission, model-authored compression, bounded adaptive recent-message preservation, durable notes, broad shell/tool capability, and persistent wakeups.

The current implementation now covers the concrete missing capabilities identified by the recordings: model-facing exact history retrieval, recovery of older exact details, bounded durable handling of unexpectedly huge output, persistent interactive terminal state and stdin, autonomous crash-safe wake-and-resume dispatch, human-readable synchronous waits, and an always-present system-level statement of user-request authority. The remaining architectural discipline is continuous rather than a discrete missing subsystem: keep the bespoke-tool surface small, prefer the shell for ordinary operating-system work, and keep semantic relevance decisions with the model while Python enforces only mechanical bounds and persistence invariants.

The central design test remains the one stated in the recordings: for every model call, does the context contain the information necessary to make the correct next decision, in the most useful form available, without ever exceeding the context window and while reserving enough space for the output? Every future subsystem should be judged by whether it improves that property without stealing semantic authority from the model.
