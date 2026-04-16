# Memory, Reading, And Editing Policies

## Working memory

Working memory is short-term, structured, and derived.
It currently tracks:
- active goal
- current step id and title
- recent tool results
- active entities such as files, notes, and plan id

It is updated through `working_memory_updated` events and rebuilt from history.
It is never authoritative.

## Semantic and procedural memory

Semantic memory stores trusted or derived facts extracted from history.
Procedural memory stores plan-strategy summaries.

Current promotion rules:
- trusted calculator and time results may be promoted
- plan events may produce derived procedural memory
- untrusted tool outputs such as file reads are not promoted when `security.block_untrusted_semantic = true`

Relevant retrieval is explicit and recorded with `memory_retrieved`.

## Sequential reading

The sequential reader reads bounded segments only.
Each reader state tracks:
- source kind
- source reference
- current offset
- chunk size
- overlap size
- finished flag

Reading events include:
- `reader_opened`
- `reader_chunk_read`
- `file_chunk_read` or `buffer_chunk_read`

The runtime and tools use this instead of dumping whole sources into context.

## Editing

The edit engine is pure and deterministic.
Supported operations:
- `replace_range`
- `insert_at`
- `delete_range`
- `replace_pattern_once`
- `replace_pattern_all`

Editing behavior:
- dry-run preview records `edit_previewed`
- real writes record `edit_applied` and then `file_write_applied`
- file writes happen only through `history.py`
- backups are optional and policy-controlled

## Tool isolation

Tools run with an isolated copied session context.
They can inspect structured session data, but they do not receive the live mutable in-memory session object.
Only structured tool outputs and generated events are allowed back into the runtime.
