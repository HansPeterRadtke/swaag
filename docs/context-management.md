# Context management implementation contract

Context management is Swaag's primary harness responsibility. Every LLM operation should eventually pass through one context-compilation path.

## Full-fidelity-first invariant

Before invoking any semantic reducer, compile the richest available candidate context. If that real serialized request fits with its output reserve and safety margin, send it unchanged. Do not summarize history, project tool results, omit candidate information, or load a smaller representation merely to save tokens. Compression is an overflow-recovery mechanism, not a standing preference.

When overflow occurs, measure which components consume the failed request and use semantic reduction targeted to that operation. Rebuild and re-tokenize after every reduction.

History compaction receives the exact deficit from the failed next call. It considers the smallest prefix that can mechanically recover that deficit, derives the summary target from the measured source and replacement-provenance cost, and asks the model what meaning to preserve. It records requested, estimated, and actual recovery and refuses to replace exact prompt history with a derived view that recovers no tokens. Tool-result, runtime-context, completion-evidence, communication-evidence, and history-analysis projections likewise reduce only the current measured deficit plus fixed serialization slack; they do not halve content or cap targets with a permanent context percentage.

## Compilation pipeline

1. Identify the semantic operation and model.
2. Read model context capacity and the active server/model chat-template identity.
3. Identify the operation's minimum useful output and desired output headroom.
4. Account for mandatory input: system and operation instructions, current request, required tool/protocol framing, and other nonoptional material.
5. Calculate the remaining dynamic-input budget.
6. Present candidate history, tools, tool results, retrieved sources, files, summaries, and state to semantic selection/reduction operations as needed.
7. Assemble role-preserving messages, ask the connected backend to serialize them with its active model template, and account for every returned wrapper segment.
8. Tokenize the exact serialized request that will be sent.
9. If it exceeds budget, calculate the required reduction and perform another semantic reduction pass. Never silently truncate semantic data.
10. Send only after the hard invariant passes.
11. Record request, response, token accounting, selected projections, and source references in durable history.

## Accounting and allocation

For every call, the implementation records model identifier, context capacity, output reserve, safety margin if any, every actual serialized component, final input tokens, actual output tokens, and provenance identifying which source records produced projections or summaries. There is no unmeasured fixed-overhead or safe-input reservation: exact serialized framing is counted directly, while estimator uncertainty belongs only to the disclosed conservative safety strategy. Prompt builders retain model-neutral system/user component ranges. For live llama.cpp, `/apply-template` supplies the active model's exact prefix, inter-message framing, and generation suffix; Swaag proves that semantic message bytes are unchanged, reconstructs named accounting components whose concatenation equals the returned prompt, and rejects a model/template identity change during compilation or before inference. Each prompt assembly also carries content hashes for that server protocol identity, rendered system instruction, and canonical template artifacts; `prompt_built` persists those versions with the complete rendered-prompt hash so behavior can be correlated and replayed without guessing from a package version.

Clients without server-side template rendering use an explicit model-neutral text envelope only as a disclosed offline/fake fallback. It is not used for live llama.cpp. Record/replay stores exact server renderings and exact tokenizer counts keyed by the complete role/content or text plus model/request identity. Replay never probes the model server and fails on a missing rendering or token count instead of falling back to a hardcoded model-family template, estimating silently, or performing a network call.

Avoid a universal percentage allocation. A document-extraction call may devote almost all input to a document; a communication call may need status and recent history; a tool-selection call may need capability descriptions and little history. Deterministic code calculates capacities. An LLM decides semantic allocation within them.

Output budgeting has two distinct values. The operation minimum is a hard validity/usefulness requirement; desired headroom is a soft maximum. Both can be supplied per call, so a dynamically sized projection is not forced into a static call-class output ratio. Compile and measure the richest candidate against the minimum first. If it fits, reserve as much desired headroom as remains without dropping that candidate. A desired amount or percentage must never cause otherwise-valid semantic input to be reduced. If the backend later reports output-limit exhaustion, raise the minimum, reconstruct and re-tokenize the call under the new hard constraint, and record the evidence. This recovery applies to actions, summaries, completion evaluation, caller-defined structured output, model-backed capabilities, tool-result projections, and health probes. When the larger minimum makes a reducible evidence call overflow, its normal bounded semantic reduction loop receives that measured overflow rather than silently shrinking output or input.

For live llama.cpp calls, `GET /props` `default_generation_settings.n_ctx` is the authoritative per-slot capacity, while `chat_template` plus model/server identity define the prompt protocol. Packaged `model.context_limit` is an explicit fallback for offline replay, fakes, and clients without a capacity probe. Probe failures in live operation are errors rather than silent fallback. Exact `/tokenize` accounting measures the `/apply-template` result and uses only the configured fixed safety allowance; estimator fallback records its strategy and uses a proportional conservative margin.

Text stop sequences are model-specific decoding configuration, not universal prompt framing. The package default supplies none and relies on the connected model's EOS behavior. A deployment may configure extra stops explicitly, in which case they are preserved in the exact request/cache identity.

## History and results

Raw history and raw tool results remain durable. Context compilation selects projections. Candidate strategies include recent exact events plus older summaries, hierarchical summaries, semantic retrieval, and targeted re-reading of raw events. These are strategies to benchmark, not universal rules.

Every derived history summary records session-aware exact source-event references/ranges and its own projection event. Prompt rendering labels it as derived and points to `history_window` for re-expansion. History search/window responses carry exact hashes, including for immutable archive shards; if those responses are later summarized or projected, their original cross-session lineage is propagated transitively. A projection may be reused only when its source event/hash matches and its measured size satisfies the new call's target.

Model-backed capabilities use the runtime semantic-call service rather than constructing a model client of their own. This keeps live capacity discovery, prompt-component accounting, output-starvation recovery, preemption, and durable request evidence on the same path as worker calls. History analysis first submits every exact event in the requested durable session up to the current-action boundary as one named, hashed source component. Lexical search ranking, recency, and the legacy `max_events` request hint do not decide what the analyzer may see. Only a measured overflow starts bounded question-specific semantic projection of that complete source; an oversized reducer input is mechanically segmented without dropping fragments, each fragment is projected semantically, and the projections are recombined semantically. Projection provenance records the exact session, event range and count, complete-source hash, target, and measured overflow. The output cites mechanically validated exact source event sequences and hashes, while raw append-only events remain authoritative and recoverable with `history_window`.

Communication status is an independent semantic operation, not a worker-action side effect. Its candidate context contains the verbatim question, deterministic mechanical snapshot, exact runtime-semantic fields, and every exact target event available at the snapshot. Unbounded semantic strings duplicated in mechanical projections are represented by hash/size there and retained exactly in the semantic evidence. If the complete request overflows, the runtime calculates a per-call projection target, semantically reduces the complete authoritative source (hierarchically when necessary), rebuilds, and re-tokenizes. Cited event sequences are mechanically checked against the snapshot. The resulting status and any failure are stored in a separate operation history so reading status never races with or mutates an active worker's append-only history.

If a summary or projection must fit a target token count, give the semantic operation that concrete per-call target and enough task context to preserve relevant information. Re-tokenize the result and retry semantic reduction if necessary. When even one exact history message or raw tool result cannot fit its reducer call, segment it mechanically without dropping bytes, reduce every fragment semantically under measured budgets, and semantically combine those derived fragments. The final derived view still cites the original exact event; intermediate summaries and projections never replace raw history.

Durable working notes and the live workspace manifest follow the same rule. Every action candidate starts with every exact note and the complete current workspace file list; neither a fixed note-token cap nor a fixed file-count prefix controls admission. The file list is rebuilt from the workspace root for each call rather than frozen into initial session state. Only measured overflow permits an LLM-authored, objective-specific projection. Each projection records the complete source hash, objective hash, measured sizes, exact recovery locator, and model-call accounting. A durable projection is reused only for the same objective while its source hash still matches, so a changed request, filesystem, or note restores the full source instead of silently applying a stale view. Exact notes remain recoverable through the notes capability and the live filesystem remains authoritative through `list_files`.

Note storage limits still fail closed instead of clipping content or evicting older notes. Explicit note compaction is a context-compiled semantic operation over every exact current note, uses hierarchical no-drop segmentation on measured overflow, and leaves the source note events authoritative.

Tool schemas also consume context. Support capability discovery or staged schema loading. The LLM selects semantically relevant capabilities; deterministic code calculates serialized cost and enforces permissions. Never destroy the only copy of source information because a prompt needs to become smaller.

Attachments enter context as a separate reference component containing only mechanical metadata: stable attachment ID, original name, likely media type, byte size, content hash, source, and event lineage. Raw content remains in content-addressed storage outside disposable session projections. Uploading never triggers extraction. If inspection is semantically useful, the LLM loads and calls a direct text reader, all2text, or a future specialist capability. Direct UTF-8 reads are exact bounded slices with start/end/next offsets and an honest completion flag, allowing model-selected sequential re-expansion without injecting or silently dropping the rest. Bounded extraction previews retain exact raw or derived artifact references and flow through the same measured tool-result projection path as other observations. Complete extraction text and the exact conversion manifest are separate integrity-checked artifacts; session archival moves those artifacts into read-only archive storage before deleting disposable session files.

## Failure modes to test

Test mandatory input exceeding capacity, minimum output exceeding available capacity, desired headroom under input pressure, output-limit finish reasons and reconstruction, oversized tool schemas, enormous tool results, long histories, summaries exceeding targets, summaries losing critical facts, multilingual tokenization differences, model switches with different capacities, tool-call serialization overhead, context expansion after user interruption, and repeated compaction over long-running tasks.

## API direction

Converge on an explicit context compiler rather than distributing token arithmetic across the agent loop. It should accept an operation description, model capabilities, mandatory material, candidate semantic sources, and output requirements, and return a verified model request plus accounting/provenance.

The compiler is not the semantic authority. It invokes semantic LLM operations when selection or compression requires understanding and mechanically verifies their products.
