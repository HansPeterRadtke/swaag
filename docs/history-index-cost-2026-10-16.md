# History search-index reconstruction cost — 2026-10-16

The installed long-history investigation separated model generation from local
work. Its frozen 5,570-event journal is 24,539,248 bytes. Creating a fresh SQLite
history/search index and reading the exact conversational messages took 104.380
seconds on Jetson. Warm exact-message reads took a median 0.009 seconds; reading
all events took 0.528 seconds, and full replay without a checkpoint 0.451 seconds.
Production already had its index. The cold cost therefore does not explain all
whole-turn latency, and reducing it is not a claim of faster model inference.

`HistoryStore._sqlite_insert_event` queried the FTS table's unindexed session and
sequence columns before every insertion. That scans existing search rows for each
new event and makes bulk reconstruction quadratic. The correction uses the
canonical `INSERT OR IGNORE` result: a newly inserted event receives its search
row directly in the same transaction. An existing event retains the old existence
check, including repair of a missing search row. There is no schema migration,
history deletion, relevance filtering, semantic change, or cache invalidation
shortcut. This invariant assumes a valid transactional index; it does not add
recovery for externally fabricated orphan FTS rows.

On the exact same frozen journal, the candidate's fresh index plus exact read takes
4.715 seconds. Both indexes contain 5,570 byte-equivalent canonical rows and 5,570
byte-equivalent search rows; both pass SQLite integrity checks. The authoritative
journal hashes match. Warm reads remain essentially unchanged. This is one
controlled before/after reconstruction pair with raw results, not a population
latency estimate or whole-conversation speed claim. SQLite is recorded in the
equality artifact rather than inferred from the system Python.

All 28 focused history-tool tests pass in 8.44 seconds, including idempotent replay,
repair of a missing search row, transactional rollback after an injected search
write failure, and exact retry. The complete immutable 95a022a suite passes 1,332 tests with three optional skips
in 230.61 seconds on Thor. Installation remains gated on the active probe ending
and the combined release checks. The active installed model probe keeps its previous package until
it stops; no running environment is replaced.

Evidence is retained under
`/data/var/swaag-benchmarks/foreground-routing-20261015`:
`frozen-history-read-profile/result.json`,
`frozen-history-read-profile-candidate/result.json`,
`history-index-equivalence.json`, `history-index-focused.txt`, and the two runnable
profile scripts. Frozen source copies preserve the exact failed-turn history for
reproduction, including earlier failures and the inaccurate derived summary.

## October 26 follow-up: canonical JSONL append visibility under concurrent readers

The frozen round-four SWAAG agent-reliability investigation reproduced a
`HistoryCorruptionError` in a **temporary, simulated** streaming session while
another thread was appending a JSONL event. The original error's temporary
history file was no longer retained, so the precise cause of that spontaneous
failure is not independently proven. An additional deterministic split-append
fault test verified the mechanism against the **pre-fix** `HistoryStore` source:
a reader could open a still-incomplete final event line during the writer's
SQLite transaction and raise corruption even though the same writer eventually
completed a valid hash chain. The original pre-fix SHA-256 was
`74ede3f4b58bf387ffc6a9478c8fdc0b56f6b48cc042f0f69a5bfd6e2486516a`.

The research-branch fix adds a **per-session, cross-process `history_journal.lock`**
(using the existing `session_lock.py` ownership mechanism). A writer holds it
across canonical SQLite `BEGIN IMMEDIATE`, the entire append, the commit and
local verified-stamp update. A peer reader with a changed file stamp acquires
the same journal lock before it validates JSONL or updates its SQLite index.
After waiting, the peer checks the current index cursor and file identity
again; a previously verified unchanged file still returns on the existing
fast path without rescanning its history. The expensive source scan does **not**
hold a global SQLite write transaction: the per-session journal lock permits
unrelated sessions to append concurrently. A short SQLite transaction is taken
only when adding previously unindexed events. All parties using the journal
must use the updated writer and reader code; mixed old/new cooperating writers
are not certified safe by this research.

Source-authentic isolated regressions cover whole-history read, `has_history_at_or_after`,
`OrchestratorEventCursor.caught_up`, and a **separate OS process** racing a
partially written event. All readers wait until the writer finishes, then see
the committed, hash-verified event. Another regression forces a valid but slow
large-session revalidation and proves a writer to an unrelated session can
complete while the scan is still held. Historical verified-tail appends do not
cause repeated full JSONL scans. Existing tamper tests still reject changed
same-size bytes; a permanently incomplete external tail still raises
`HistoryCorruptionError` instead of being skipped/retried as a successful
source record. These are targeted causality tests, not proof every production
crash or externally uncooperative append source is repaired.

**Status:** The source changes are only in the dedicated research Git branch;
no installed SWAAG service, model process, Qwen GPU backend, state directory or
archived owner task was modified or restarted. The complete research record,
pre-fix journal evidence and tests are preserved in
`research/agent-model-reliability-2026-10-24/ROUND8_HISTORY_APPEND_VISIBILITY_2026-10-26.md`
and `tests/test_history_partial_append_visibility.py`. No independent real
Qwen direct-versus-specialist final-effect or GPU inference data are implied.
