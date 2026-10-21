# Fast worker lifecycle revision: timestamp collision fix (2026-10-22)

## Observed failure

Thor SWAAG's fast snapshot fence used a worker count and the global maximum updated_at timestamp. The worker store creates each transition's timestamp before obtaining the SQLite write lock. Two independently running workers can therefore commit in an order different from their generated timestamps. The last stored update can belong to a worker whose timestamp is earlier than another worker's already-stored timestamp; COUNT and MAX(updated_at) stay unchanged despite a real change from created to working or another state transition. WorkerStore itself persisted the correct transition, but the quick response could publish a stale cached worker-state answer because the worker/plan portion of its revision tuple did not change.

A negative test on the original source constructed two workers with distinct timestamps, transitioned the earlier-created worker to working using a timestamp between those original timestamps, verified the worker database was changed, and **observed equal fast-status revision tuples before and after**. This was a real deterministic test failure, not a hypothetical stale prompt.

## Correction

CommunicationService._foreground_task_revision now reads individually identified worker lifecycle fields in the existing worker-history grouping query, once in ordered form from the worker store. It retains the original count and maximum timestamp slots for backward compatibility, then includes the ordered per-worker status, last update, archival mark, run count, completion mode and model route as an additional revision tuple component. The existing _foreground_status_revision includes this component automatically, while continuing to exclude unrelated worker history token/accounting cursors. It therefore catches a meaningful worker state transition even if global MAX(updated_at) is unchanged.

No worker database migration, status mutation, model endpoint restart, notification, or voice-device change is needed. The deterministic fence still cannot prove the semantic quality of the worker report; this is state-change detection only.

## Regression coverage

A first targeted negative test failed on the unchanged source. After the correction, it passes. Additional tests verify a model response is rejected and delegated to the full orchestration controller when a real worker status changes while first speech is being generated, even when the background world refresher has not run; and that an extra worker run with an unchanged synthetic timestamp/status is distinguished by its run_count. Existing fast worker state, explicit question freshness, status-admission and token-accounting regressions are retained.

## Synthetic timing sanity check

Thor Python SQLite in-memory test, forty samples per case:

- 128 workers: prior two-query method 0.075 ms average, current one-query plus exact lifecycle fingerprint 0.245 ms average.
- 1,024 workers: prior 0.495 ms, current 1.604 ms average.
- 4,096 workers: prior 2.302 ms, current 7.346 ms average.

These results measure only the particular SQLite worker-row query and Python revision materialization, not end-to-end user latency, disk-cold filesystem costs, concurrent worker writes, or total snapshot refresh time. The number of terminal/archived workers retained by a deployment affects the cost. The current design prioritizes current-state correctness over a small additional query cost, and does not reread complete worker semantic histories.

## Release boundary

This is a source/test-level correction on a separate worktree until independent regression and integration checks complete. Before production acceptance, verify a clean source revision, all relevant offline regressions, installed-package byte parity, and a stable idempotent runtime update when parallel voice-agent work permits it. Active Thor SWAAG communication/voice model services have not been restarted, and source tests are not evidence of real phone audio or real-model concurrency behavior.

## Completed offline and installed-candidate acceptance

The complete isolated Thor source regression suite finished with **1,706 passed, three optional skipped, zero failed in 366.96 seconds**. The new two-worker stale-revision reproduction failed on original source before implementation and passed afterward. The unchanged installed-package candidate was built with the project's runtime installer under a dedicated test-only runtime directory, not the active service path. A read-only byte comparison reported **212 of 212 packaged source files matching**, no added/changed files and SQLite **3.53.1**. A separate focused suite imported from this **installed candidate**, not the source tree, and passed **33 tests**.

A point-in-time read-only Thor live inspection showed both configured model health endpoints responding HTTP 200, each server advertising one inference slot with no in-flight inference at that instant, and no nonterminal worker in either inspected SWAAG worker database. Those observations were insufficient to authorize or certify production service replacement, Android device playback or actual code in the already-running gateway. Thor's installed production environment was left unchanged and remains a different source revision.
