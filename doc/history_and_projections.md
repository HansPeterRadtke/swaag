# History And Projections

## Canonical history

Each session has one authoritative file:
- `complete_history.jsonl`

Each event contains:
- `id`
- `sequence`
- `session_id`
- `timestamp`
- `type`
- `version`
- `payload`
- `metadata`
- `prev_hash`
- `hash`

The file is append-only.
Past entries are never modified or deleted.

## Integrity checks

History reads verify:
- strict sequence order
- unique event IDs
- valid event schema
- correct per-event hash chain

If any check fails, replay raises `HistoryCorruptionError`.

## Projections and index

Derived files may exist:
- `current_state.json`
- `notes.json`
- `reader_state.json`
- `history_index.json`

These are caches only.
They are not required for recovery.

## Replay

`HistoryStore.rebuild_from_history(session_id)` and `replay_history(path)`:
- read only `complete_history.jsonl`
- replay events in order
- rebuild in-memory state deterministically

Rebuilt state currently includes:
- messages
- notes
- reader state
- logical file views
- pending writes
- active plan
- working memory
- semantic memory
- counters and last event hash

## Event classes in active use

History currently records events for:
- session lifecycle
- user and assistant messages
- prompt assembly and budget checks
- model requests, responses, retries, and tokenization
- planning and reasoning
- working memory and semantic memory
- tool execution and tool results
- file reads, edit previews, edit applications, and file writes
- history summaries and compression
- rebuilds and doctor checks

## CLI inspection

Useful commands:

```bash
python3 -m swaag history show <session_id> --tail 20
python3 -m swaag history replay <session_id>
python3 -m swaag history diff <session_id>
```

`history diff` compares replayed state against the current state projection. It is a projection check, not a second source of truth.
