# Memory, Reading, And Editing Policies

## Authority

History is authoritative. Working memory, event memory, notes, summaries, embeddings, indexes, logical file views, and plans are derived working projections. None may silently replace or rewrite the source history.

## Working memory

Working memory is short-term structured state derived from history. It can track the active goal, current step, recent evidence, active entities, and other bounded task state. Changes are history events so state can be rebuilt.

## Semantic retrieval

The desired relevance policy is semantic, not recency-first. A model or model-assisted retrieval phase chooses which older/recent history ranges are relevant to the current effective user objective. Embeddings, lexical search, timestamps, event types, and recency may generate candidates, but deterministic mechanics do not make the final semantic relevance judgment.

If a semantic scorer is temporarily unavailable, deterministic retrieval may use recency/search as a conservative fallback. That fallback must be recorded and must not be described as equivalent quality.

Untrusted source/tool content keeps trust/provenance metadata. Blocking untrusted text from being promoted into authoritative semantic facts does not mean hiding it from the model when the user's task requires inspecting it; it means preserving provenance and preventing derived memory from treating it as trusted instruction.

## Context safety

Every prompt path must account for the entire request and reserve output headroom before calling the model. Context overflow is forbidden. Large tool/source results are stored completely and exposed through bounded previews/retrieval handles rather than dumped blindly into the active prompt.

## Reading

The sequential reader supports bounded traversal of large sources and persistent offsets. Targeted search/jump retrieval is also allowed. The agent decides semantically whether sequential reading, search, or another inspection method is appropriate.

## Editing

The deterministic edit engine can provide exact range/pattern operations, previews, and auditable writes. It is an implementation capability, not a requirement that the agent always use a custom edit tool. The general command/process tool remains a valid way to use mature system editing utilities when appropriate.

All file mutations are recorded with enough information to audit what changed. Tool execution receives isolated structured session context rather than unrestricted mutable runtime internals.
