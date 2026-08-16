# History And Projections

## Canonical history

Each session has one authoritative append-only `complete_history.jsonl`. Past entries are never modified or deleted. Events are sequence ordered, uniquely identified, schema validated, and hash chained. History corruption is a hard error rather than something silently repaired from a projection.

The canonical history must retain the complete semantic and mechanical audit trail needed to reconstruct what the agent saw and did: user instructions, effective-goal changes, prompt assembly and budget metadata, model requests/responses, accepted and rejected actions, structured status, tool calls/results, file mutations, summaries, recovery/control events, and terminal output. Very large tool/source content may be stored by durable reference, but the information must remain recoverable.

## Projections

`current_state.json`, `notes.json`, `reader_state.json`, `history_index.json`, working memory, event memory, summaries, embeddings, and future database indexes are derived projections only. They are rebuildable from canonical history plus durable referenced artifacts and are never an independent source of truth.

## Replay

`HistoryStore.rebuild_from_history(session_id)` and `replay_history(path)` replay canonical events in order and rebuild in-memory state deterministically. `history diff` compares replayed state with current projections; disagreement means the projection is wrong or history is corrupt, never that the projection overrides history.

## Prompt selection

History retention and prompt selection are separate problems. The system keeps complete history but only sends a bounded task-relevant view to the model. Detail is not selected by a fixed "last N" rule. Recent events are often useful, but old events may be essential and recent output may be irrelevant. Semantic relevance is decided by the model/model-assisted selector. Deterministic code enforces token budgets and provenance but must not pretend recency is semantic importance.

Summaries retain provenance to the source event ranges they compress. If the summary is insufficient, the source events can be retrieved again.

## Scale and archival

Long-lived histories may be indexed, archived, partitioned, or represented in a database for performance. Archival is lossless. It must remain possible to reconstruct or inspect the original authoritative event content.
