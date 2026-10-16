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
