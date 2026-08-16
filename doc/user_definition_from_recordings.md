# User Definition From Agent Recordings

## Authority and synchronization rule

This document is a one-way, cleaned written representation of the user's agent recordings. The recordings are the source of truth. This document may be corrected when the recordings contradict it; the recordings must never be edited to match this document.

The goal is information preservation, not verbatim transcription. Filler words, speech disfluencies, accidental repetitions, and transcription noise are removed, but substantive information, alternatives, uncertainty, and open questions are retained. Statements that the user explicitly described as examples, possibilities, ideas, or brainstorming are marked as such and are not silently promoted into requirements.

The source set is the eight text transcripts under the agent recordings tree, backed by the corresponding original audio where present. The transcript `Recording 2026-08-11 12-06-27-811.txt` is visibly corrupted after its meaningful opening: one sentence is repeated 380 times. Its original MP3 still exists. This document therefore preserves every recoverable statement from the text but does not invent the missing remainder. That source span remains a transcription-recovery item.

## Highest-level definition

The agent exists to do what the user actually wants. Correctly following the user's instruction is the highest-level quality criterion. The system should not insert an independent semantic agenda between user and agent. If the user's instruction is broad, the agent should still remain inside the literal scope of that instruction rather than substitute a different task. If the user says only "do something", almost any actual action can satisfy the literal request; an enormous ten-day project would usually be an unreasonable interpretation, while doing nothing would fail it.

User dissatisfaction is evidence that the previous interpretation or execution is wrong. Sarcasm, anger, blunt correction, or a direct statement that the agent is wrong should all have the same operational meaning: do not continue unchanged. Reconsider the history, the user's actual instruction, what happened, and what must change. Sarcasm detection itself does not need a special deterministic subsystem; the model can interpret it semantically.

The user explicitly does not want brittle deterministic code making semantic judgments that it cannot actually know. Importance, relevance, meaning, whether output is useful or spam, whether previous behavior satisfies the user, and what history is needed are semantic questions. The model should make those judgments. Deterministic code should handle things it can know exactly: byte/token budgets, schemas, file boundaries, process state, timestamps, integrity, limits, and execution mechanics.

## Context is the central quality problem

The central technical problem is filling the model context correctly. If context were unlimited, the simplest high-quality system would keep the complete detailed history available and derive everything from it. Because context is finite, the system must select what to include while preserving everything outside the active prompt in authoritative history.

The absolute rule is that the active request must never overflow the model context. The system must calculate input size before the model call and reserve enough space for the expected output. At the same time, it must provide all information necessary for the model to solve the current task.

"Recent history" is not a fixed number of messages or tool calls. Depending on the task, the model may need only the last user message and one step, or a hundred detailed steps including every tool call, parameter, and result. Long-ago history is not automatically unimportant, and recent history is not automatically important. Summarization is useful, but the decision about what detail to keep must be task-dependent.

The system should therefore preserve complete history and create derived summaries, indexes, semantic relevance judgments, or selected views for prompt construction. Summaries and indexes must never replace the authoritative history.

## History

The agent always needs access to history. History should contain enough information to reconstruct what the agent saw, what it decided, why it decided it, what it did, and what happened. This includes user instructions, prompts/model inputs where practical, model outputs, semantic actions, tool calls, parameters, tool results, status information, errors, retries, summaries, and control events.

History can become very large. Large histories must be handled rather than destructively truncated. Old material may eventually be archived, indexed, summarized, or moved into scalable storage, but the authoritative information must remain recoverable. A database or other scalable index is an implementation possibility; it is not itself the source of truth.

History analysis is important for diagnosing mistakes. A history-analysis capability should be able to find the effective user instruction even when it evolved over several user messages, inspect the actual model context and decisions, identify why behavior diverged, and produce a root cause plus a changed plan. The recording describes a dedicated history-analysis sub-agent/tool as one possible implementation, not as the only allowed architecture.

Semantic search over history is also a possible derived capability. The recording suggests embeddings for fields such as situation, chosen action, and reason so that a user or analysis tool could ask whether a similar situation occurred before. This is explicitly an idea, not a requirement that embeddings themselves be authoritative.

## Tools

The agent should have broad practical capability. The tool set can always grow, but specialized agent tools should exist where a normal command-line program cannot do the job easily, safely, statefully, or with the required agent-history integration.

The command-line/environment tool is especially important and should be extremely capable. On Linux, ordinary mature CLI programs already provide file inspection, text transformation, process control, search, editing, networking, documentation, and system introspection. The model can use `--help` and man pages rather than requiring the agent harness to reimplement every basic Unix operation as a dedicated semantic tool. Windows support is also desired, although Linux can come first.

Interactive commands and long-running processes require first-class handling. Command output must not blindly flood the active model context. A deterministic layer can enforce size limits and store the complete result externally/history-side, but it cannot decide semantic importance from size alone. The model should be able to inspect, summarize, classify, or retrieve the relevant result.

A sleep/wait capability is explicitly wanted. It should support relative durations with human-readable units such as milliseconds, seconds, minutes, hours, days, weeks, and potentially months, and absolute wake-up date/time. Years were explicitly considered undesirable. Long waits must not require keeping an active model call open.

All tools that may be used from multiple threads or processes must be concurrency-safe. The recording mentions MCP servers as an attractive standard implementation. MCP is an implementation option, not a requirement that every local primitive become its own process.

## Structured model actions and constrained decoding

Constrained decoding is a core requirement. The user explicitly said it must always be used and tested rather than relying on unconstrained model text and hoping to parse it afterward. Model calls that control the agent should return structured, schema-constrained data. Local schema validation remains useful even when generation-time constraints are active.

The structured action should give the model the practical choices it can need: communicate with the user, continue or finish, invoke one or more tools where allowed, record status/reasoning information appropriate for history, request analysis/recovery, and otherwise drive the next loop step. The structure should be general rather than encode benchmark-specific or task-family-specific answers.

A terminal action must produce the user-facing result required by the task. The system must not accept an accidentally empty terminal answer when a user-facing answer is required.

## User interface and control plane

The core agent is text-in/text-out. Speech-to-text, text-to-speech, GUI, phone interfaces, and other HMI layers sit outside the core agent and may be mounted on top of it.

A task may run for seconds, hours, months, or longer. Therefore the user must be able to interact with the system independently of the main agent's current long-running operation. The control plane must support interruption, stopping, redirection/new instructions, and status inspection without waiting for the current main-agent loop to finish naturally.

Intermediate status messages are useful but optional for display. If generated, they should be retained in history even when the UI hides them. A useful semantic status contains the current situation, what is being done next, and why. The recording explores adding an importance/criticality category, preferably semantic text labels rather than arbitrary numeric levels. It also explores generating status as part of the same constrained model action rather than spending a separate model call.

The recording discusses a small separate assistant model/thread that can read history, answer status questions quickly, calculate times, and pass stop/redirect/questions into the main agent. This is a proposed architecture rather than a mandatory implementation. The underlying requirement is independent communication/control while the main agent is busy.

## Configuration

Anything meaningfully configurable should be represented in configuration rather than hard-coded: model/provider endpoints, context budgets, timeouts, tool limits, status display, verbosity, storage paths, runtime behavior, and similar operational choices. CLI overrides or environment overrides are acceptable, but they should resolve into one explicit effective configuration.

## Recovery and dissatisfaction

When repeated failures occur, or when the user says the agent is wrong or dissatisfied, the agent should not continue the same strategy blindly. It should analyze the relevant history and evidence, verify that it understands the actual effective user task, identify what failed and why, and change its plan. A dedicated analysis tool/sub-agent is one possible implementation; a semantic recovery phase inside the main loop is another.

The user explicitly described the detailed history-analysis sub-agent architecture as brainstorming. The requirement is the behavior: deep evidence-based correction, not that exact internal topology.

## Explicit alternatives and brainstorming preserved from the recordings

The following were proposed as possibilities rather than fixed requirements: a dedicated history-analysis LLM tool; classifying every history/output item into semantic importance categories; embeddings over situation/action/reason for similarity search; a separate small communication assistant model; representing status importance with two, three, five, or ten levels; isolated model calls that intentionally omit overarching task context when testing proves isolation improves quality; an MCP server per tool; and a database as the canonical physical history storage.

The recordings repeatedly warn not to take these examples at face value merely because they were spoken. The system should choose the architecture that best satisfies the underlying requirements and measured quality.

## Source-specific notes

`Recording 2026-08-04 10-27-01.txt` contains essentially no usable substantive transcript text. Its audio exists, but the current text transcript does not provide recoverable requirements.

`Recording 2026-08-04 10-36-47.txt` defines sleep/wait as an example of a necessary practical tool and establishes context filling, adaptive history detail, output headroom, and never overflowing context as the central quality problem.

`Recording 2026-08-06 17-47-36-666.txt` emphasizes broad command-line capability, avoiding redundant harness tools where mature CLI programs already exist, output flooding, and the distinction between deterministic mechanics and semantic model judgments.

`Recording 2026-08-07 12-43-50-515.txt` establishes exact user-intent adherence as the highest priority and discusses literal/minimal handling of underspecified tasks.

`Recording 2026-08-09 13-35-10-245.txt` develops dissatisfaction/recovery behavior and proposes, explicitly as brainstorming, a history-analysis sub-agent/tool.

`Recording 2026-08-10 12-20-47-854.txt` separates HMI from the text core, requires interruption/status control for long-running work, strongly requires constrained decoding, and brainstorms structured status messages and importance levels.

`Recording 2026-08-11 11-49-55-272.txt` clarifies that status display is optional but useful in history and proposes a decoupled small communication assistant as a possible architecture.

`Recording 2026-08-11 12-06-27-811.txt` continues that assistant/history-tool idea, requires thread/process-safe tools, mentions MCP as attractive, distinguishes a simple history reader from a deeper history-analysis capability, and proposes semantic similarity search. The transcript then becomes corrupt through repeated text; the original audio remains authoritative and must be retranscribed before any unrepresented later statements can be claimed as synchronized.
