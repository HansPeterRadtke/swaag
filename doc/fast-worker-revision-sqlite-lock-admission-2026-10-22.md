# Foreground worker revision under SQLite writer contention — October 22, 2026

## Original defect reproduced on Thor

The final user-facing fast-response freshness fence reads worker, plan and worker-history revisions before accepting a generated status answer. Before this change, it used the normal writable WorkerStore, OrchestrationStore and HistoryStore connection helpers. Those helpers open SQLite with a thirty-second busy timeout and journal-mode initialization, suitable for durable writes but inappropriate for optional fast state observations.

A new deterministic test created an isolated disposable worker database and held a genuine SQLite exclusive writer lock while a separate thread requested the foreground task revision. On the unmodified source it remained blocked after two hundred and fifty milliseconds and completed only after the test released the writer, measuring about three hundred and thirty milliseconds in that controlled reproduction. This defect can delay the spoken response with no extra evidence returned; it is not an LLM inference problem. No production database was locked or changed.

## Corrective implementation

Foreground revision checks now open **strictly read-only** SQLite URI connections with an explicit **forty-millisecond busy timeout** and no journal-mode or schema mutation. All three revision sources — workers, plan database and worker history — use the same short-lived helper. Connection ownership is explicit and every handle closes even if query execution raises. Missing or locked state raises a SQLite exception instead of silently substituting an empty or old revision; the existing optional fast stage catches such unavailability and proceeds without certifying the missing snapshot. The full orchestrator remains the authority for state inspection or user-requested work.

These short reads still use the durable worker lifecycle fields and the existing covering history index for semantic status/question events. The correction does not remove the final receipt-lock stale revision comparison or use token activity as task progress. It does not invent a percentage or ETA or relax an authorization rule. Because the stores are independent, no cross-database transaction or absolute global wall-clock guarantee is claimed.

## Verification and limits

Focused regression checks use **disposable** real SQLite databases, including separate exclusive writer locks on the worker store, plan store and history index. All three cases require the foreground revision to report unavailability before a quarter-second deadline rather than block until the writer releases. The test then verifies successful reading once the lock is released. Another check uses a filename containing spaces, hash and question mark, proves URI read-only enforcement and ensures a missing database is not created.

The first lock contention test was observed failing against the original code before implementation and passing after it. The expanded focused source suite passed **thirty-one tests** including the earlier last-moment freshness and worker lifecycle regressions. The later isolated non-editable installed-package candidate matched **212 of 212** source/package files with SQLite **3.53.1** and passed **56 focused tests** while importing SWAAG from the installed package (not the source-tree Python path). The full isolated Thor source regression suite subsequently finished with **1,729 passed, three optional skips and zero failures in 346.87 seconds**, with the test runner exiting successfully. This is offline/source acceptance, not production microphone-to-speaker or real Qwen contention acceptance.

In an additional read-only/disposable SQLite benchmark, eight repeated attempts to read while an exclusive writer lock was held returned SQLite busy errors in **40.6–40.7 milliseconds** each (mean **40.6 ms**) instead of continuing to wait for a writer release. This measures the lock response of the SQLite metadata helper, not model inference or end-to-end audio output.

The test reproduces an exclusive SQLite writer lock, not a production-longitudinal measurement of typical WAL write contention or model response time. During a true database unavailability, a direct factual worker claim without a fresh inventory cannot be certified; the system may need the full controller or explain that observations are temporarily unavailable. Real Qwen server inference concurrency and Android microphone-to-speaker acceptance remain separate gates.

## Deployment boundary

This change is confined to an isolated SWAAG worktree. No live Thor SWAAG package or communication/voice gateway was replaced. Deployment must preserve concurrent voice-agent code and history and verify source-package parity, idle-safe service transition, backend health and real user-facing behavior.
