# Deterministic policy audit

Swaag uses deterministic code only for exact facts, hard constraints, bounded
execution, and verification. A numerical limit may reject, defer, or require a
new semantic call; it must not decide relevance, intent, priority of meaning, or
task completion. This document records the remaining production policies and
why they stay mechanical.

## Context and model transport

- Model/server context capacity, tokenizer counts, serialized prompt ranges,
  structured-schema floors, output minima, fixed exact-count safety, and
  conservative estimator safety are measurable admission facts.
- Desired output budgets are maxima, not input quotas. A backend-reported
  output-limit finish reason raises the operation minimum and reconstructs the
  call. Only measured overflow may trigger semantic reduction.
- Reduction rounds and semantic-call budgets bound resource use. Python computes
  the token deficit and validates the result; an LLM decides what meaning can be
  reduced. Exhaustion fails explicitly instead of dropping source material.
- Retry counts, connect/token timeouts, and backoff limit transport failure.
  They never convert a failed call into a semantic answer.
- Live backend identity, chat-template identity, cache keys, and exact replay
  matches are integrity facts. Missing exact replay data blocks a cached run
  rather than selecting a substitute answer.

## History and storage

- Event sequence, hashes, timestamps, source ranges, content hashes, archive
  boundaries, and projection locators are append-only provenance.
- Storage, upload, and read-chunk limits protect resources. Oversized writes fail
  closed; bounded reads return offsets and completion flags; bounded tool/browser
  previews and external-process/provider errors retain integrity-linked raw
  artifacts before their reduced display text is recorded.
- History compaction starts with the smallest oldest prefix that can mechanically
  recover the measured deficit while leaving at least one exact message outside
  the candidate. This chooses a replaceable storage region, not which facts to
  retain: the summary LLM receives every exact candidate message and owns semantic
  retention, including how many candidate-tail messages remain verbatim.
- Search result counts and protocol cursors are caller-visible retrieval bounds,
  not context-selection policy. Raw events remain addressable through exact
  windows, archive retrieval, artifacts, or another model-selected search.

## Tools, permissions, and verification

- Enabled-tool lists, side-effect/stateful switches, path roots, write allowlists,
  schemas, preflight hashes, process ownership, and command timeouts are hard
  authorization or execution constraints.
- Tool availability is a measured capability fact. The LLM selects capabilities
  semantically from the compact index; Python does not route by keywords, MIME
  type, task ID, or fixture shape.
- Action/tool-call totals and repeated-identical-action limits prevent runaway
  resource consumption. Hitting a limit is an explicit failure, never evidence
  that the task is complete.
- Exit codes, persisted file hashes, diffs, schema validation, and executable
  tests are evidence. They constrain independent semantic completion evaluation
  but do not synthesize a final answer.

## Workers and inference

- Worker states, event order, control snapshots, run ownership, wakeup due times,
  process liveness, and heartbeat age are mechanical facts.
- A control arriving after an LLM snapshot invalidates stale uncommitted work;
  completed tool evidence remains durable. The next LLM call interprets and
  reconciles controls in exact arrival order.
- Inference admission uses discovered slot capacity. Control/status traffic has
  a mechanical latency priority while queue aging prevents starvation; neither
  value represents semantic task importance.
- Cancellation releases transport work. Unsupported true suspension is never
  claimed: an unchanged request is reconstructed and replayed exactly, while a
  changed source snapshot supersedes it.
- Continuous completion mode is an explicit caller contract. Normal completion,
  blocking-question criticality, status meaning, and escalation remain LLM
  decisions under closed schemas.

## Protocols and presentation

- JSON/schema validation, protocol versions, cache TTLs, cursor bounds, IDs,
  transport locality, and deterministic lifecycle mappings enforce wire
  contracts. Adapters project canonical task/history state and do not own it.
- Caller-declared structured-output bindings may fill only mechanical fields such
  as IDs, timestamps, and lifecycle state. Semantic fields come from a centrally
  compiled LLM operation.
- Visual relevance and optional audio rendering are separate LLM operations.
  Deterministic code validates source preservation and exposes the raw verified
  worker result; it does not remove details because of field names or formatting.

## Audit rule

Any new deterministic branch that inspects natural-language content, domain field
names, task metadata, expected answers, benchmark fixtures, or source-string
patterns is presumed semantic and must be removed or justified by a hard protocol,
security, integrity, or resource invariant. Tests and benchmark traces should prove
the boundary rather than encode fixture-specific repairs.
