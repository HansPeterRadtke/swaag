# Context management implementation contract

Context management is Swaag's primary harness responsibility. Every LLM operation should eventually pass through one context-compilation path.

## Full-fidelity-first invariant

Before invoking any semantic reducer, compile the richest available candidate context. If that real serialized request fits with its output reserve and safety margin, send it unchanged. Do not summarize history, project tool results, omit candidate information, or load a smaller representation merely to save tokens. Compression is an overflow-recovery mechanism, not a standing preference.

When overflow occurs, measure which components consume the failed request and use semantic reduction targeted to that operation. Rebuild and re-tokenize after every reduction.

## Compilation pipeline

1. Identify the semantic operation and model.
2. Read model context capacity and tokenizer/chat-format behavior.
3. Identify the operation's minimum useful output and desired output headroom.
4. Account for mandatory input: system and operation instructions, current request, required tool/protocol framing, and other nonoptional material.
5. Calculate the remaining dynamic-input budget.
6. Present candidate history, tools, tool results, retrieved sources, files, summaries, and state to semantic selection/reduction operations as needed.
7. Assemble the candidate request.
8. Tokenize the actual serialized request.
9. If it exceeds budget, calculate the required reduction and perform another semantic reduction pass. Never silently truncate semantic data.
10. Send only after the hard invariant passes.
11. Record request, response, token accounting, selected projections, and source references in durable history.

## Accounting and allocation

For every call, the target implementation should record model identifier, context capacity, output reserve, safety margin if any, mandatory-input tokens, each dynamic component's tokens, final input tokens, actual output tokens, and provenance identifying which source records produced projections or summaries.

Avoid a universal percentage allocation. A document-extraction call may devote almost all input to a document; a communication call may need status and recent history; a tool-selection call may need capability descriptions and little history. Deterministic code calculates capacities. An LLM decides semantic allocation within them.

Output budgeting has two distinct values. The operation minimum is a hard validity/usefulness requirement; desired headroom is a soft maximum. Both can be supplied per call, so a dynamically sized projection is not forced into a static call-class output ratio. Compile and measure the richest candidate against the minimum first. If it fits, reserve as much desired headroom as remains without dropping that candidate. A desired amount or percentage must never cause otherwise-valid semantic input to be reduced. If the backend later reports output-limit exhaustion, raise the minimum, reconstruct and re-tokenize the call under the new hard constraint, and record the evidence. This recovery applies to actions, summaries, completion evaluation, caller-defined structured output, model-backed capabilities, tool-result projections, and health probes. When the larger minimum makes a reducible evidence call overflow, its normal bounded semantic reduction loop receives that measured overflow rather than silently shrinking output or input.

For live llama.cpp calls, `GET /props` `default_generation_settings.n_ctx` is the authoritative per-slot capacity. Packaged `model.context_limit` is an explicit fallback for offline replay, fakes, and clients without a capacity probe. Probe failures in live operation are errors rather than silent fallback. Exact `/tokenize` accounting uses a fixed safety allowance for transport/template details; estimator fallback records its strategy and uses a proportional conservative margin.

## History and results

Raw history and raw tool results remain durable. Context compilation selects projections. Candidate strategies include recent exact events plus older summaries, hierarchical summaries, semantic retrieval, and targeted re-reading of raw events. These are strategies to benchmark, not universal rules.

Every derived history summary records session-aware exact source-event references/ranges and its own projection event. Prompt rendering labels it as derived and points to `history_window` for re-expansion. History search/window responses carry exact hashes, including for immutable archive shards; if those responses are later summarized or projected, their original cross-session lineage is propagated transitively. A projection may be reused only when its source event/hash matches and its measured size satisfies the new call's target.

Model-backed capabilities use the runtime semantic-call service rather than constructing a model client of their own. This keeps live capacity discovery, prompt-component accounting, output-starvation recovery, preemption, and durable request evidence on the same path as worker calls. History analysis first submits complete exact candidate events as individually measured components. Only a measured overflow starts bounded semantic projection; an oversized reducer input is mechanically segmented without dropping fragments, each fragment is projected semantically, and the projections are recombined semantically. The output cites exact source event sequences and hashes, while raw append-only events remain authoritative.

If a summary or projection must fit a target token count, give the semantic operation that concrete per-call target and enough task context to preserve relevant information. Re-tokenize the result and retry semantic reduction if necessary.

Tool schemas also consume context. Support capability discovery or staged schema loading. The LLM selects semantically relevant capabilities; deterministic code calculates serialized cost and enforces permissions. Never destroy the only copy of source information because a prompt needs to become smaller.

Attachments enter context as a separate reference component containing only mechanical metadata: stable attachment ID, original name, likely media type, byte size, content hash, source, and event lineage. Raw content remains in content-addressed storage outside disposable session projections. Uploading never triggers extraction. If inspection is semantically useful, the LLM loads and calls a direct text reader, all2text, or a future specialist capability. Bounded previews must retain an exact raw or derived artifact reference and flow through the same measured tool-result projection path as other observations.

## Failure modes to test

Test mandatory input exceeding capacity, minimum output exceeding available capacity, desired headroom under input pressure, output-limit finish reasons and reconstruction, oversized tool schemas, enormous tool results, long histories, summaries exceeding targets, summaries losing critical facts, multilingual tokenization differences, model switches with different capacities, tool-call serialization overhead, context expansion after user interruption, and repeated compaction over long-running tasks.

## API direction

Converge on an explicit context compiler rather than distributing token arithmetic across the agent loop. It should accept an operation description, model capabilities, mandatory material, candidate semantic sources, and output requirements, and return a verified model request plus accounting/provenance.

The compiler is not the semantic authority. It invokes semantic LLM operations when selection or compression requires understanding and mechanically verifies their products.
