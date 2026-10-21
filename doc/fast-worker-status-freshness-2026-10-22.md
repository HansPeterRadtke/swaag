# Fast worker status freshness — reproducible correction, October 22, 2026

## Problem and original failure

The fast spoken stage uses a precomputed world view instead of scanning every worker journal while a user waits. This is correct for latency, but its final response fence previously compared a narrow meaningful_worker_state_revision that tracked only the **number** of open questions, not their identities/content or completeness. Two different pending worker questions with one item each yielded an equal revision. The same comparator omitted worker model route, latest reported activity/next action and continuous-work cycle progress. The new negative tests reproduced these exact equal-revision failures before changes.

There was a second, independent race: _fast_foreground_state_revision retained only the first five worker/plan store revision fields, ignoring durable worker event cursors. A worker could append a new progress or question event after the background snapshot was captured and before first speech, without the background refresher running again or a worker-store timestamp changing. A negative end-to-end fake-model test reproduced delivery of an old direct answer in that window before the fix.

## Changes

fast_worker_state.meaningful_worker_state_revision now fences the identity/content/criticality of sampled pending questions, inventory completeness, individual worker model routes, their model-authored reported actions, finite/continuous explicit assessment values and already existing plan/terminal lifecycle facts. It continues to ignore snapshot capture clocks, heartbeat age and transient inference phases, which are not completed-work evidence.

CommunicationService._foreground_task_revision adds a **covering-index-only** read of maximum sequence numbers for material worker event types: agent status/progress, question creation/revision/resolution and legacy worker input-required. The quick-response fence uses durable worker/plan revisions, control-receipt revisions and these narrow semantic cursors. It intentionally **excludes the broad every-event cursor** so token accounting, ordinary tool telemetry and write-heavy unrelated observations do not continually invalidate quick speech. Full task snapshot invalidation continues to use the broad cursor for its existing deeper synchronization guarantee.

The optional quick model never gains a tool/worker authority. On a detected change, its stale answer is withheld and the original message goes through the existing full orchestrator path; no model-derived percentage or ETA is invented.

## Tests and limitations

The new regressions reproduce a question replacement with an unchanged count, inventory becoming incomplete, worker route switching, explicit continuous-cycle assessment changing, and model-authored next action changing. They check that ordinary snapshot clock/heartbeat/inference-phase changes are not treated as completion state. Integration tests mutate a world snapshot during quick-model inference, and independently simulate a new durable semantic cursor **without background refresh**, requiring safe fallback. An additional test uses actual SQLite-backed worker events to verify semantic-cursor changes and non-semantic token telemetry suppression.

The focused worker freshness/status suite passed **30 tests** after the fix. The complete source regression on this isolated candidate passed **1,703 tests, three optional skips and zero failures** in **334.31 seconds**, without starting a real model inference. This is a deterministic code-level correction, **not** real Qwen/Android model acceptance, perfect semantic relevance for long history, or proof of a deployed voice response. The isolated source test/merge gate and an actual runtime deployment are separate; no live service was restarted or worker stopped by this correction.

## Isolated indexed-query performance evidence

An in-memory Thor SQLite experiment populated **60,000 durable-like history rows** across **50 worker sessions**, with a covering (session_id,event_type,sequence) index and periodic agent progress entries. Thirty executions of the semantic cursor query returned fifty rows, averaged approximately **0.96 milliseconds**, and peaked at approximately **2.71 milliseconds** in that run. EXPLAIN QUERY PLAN selected the covering index by session and event type. This is a synthetic *query* measurement, not a complete foreground reply latency or a promise that production storage/cold pages exhibit the same cost.

A separate read-only inspection of two Thor SWAAG runtime worker databases observed no nonterminal worker rows at that instant; one orchestration-plan database listed one active plan. This is not a verified plan completion state, and these database counts cannot establish global current activity across every owner/host.

## Unresolved production checks

Before an idle-safe live deployment, compare installed source/package bytes and active-process revision separately. Under real concurrent worker/foreground Qwen load, measure fast-turn admission, index query latency, stale-reply rejection frequency, fallback latency, active worker status fidelity and actual slot preemption. Such tests must not invent worker completion percentages or treat an HTTP health check as end-to-end acceptance.

## Installed candidate acceptance after main integration

The complete regression was followed by a clean fast-forward of the isolated correction into Thor main. The repository status remained clean, locally ahead of remote tracking. A **second, separate** installation candidate was built with the repository's normal runtime installer under an explicitly overridden runtime-data root, not over the running SWAAG environment. The read-only preflight reported **212 source files and 212 exact installed file matches**, with zero missing/extra/changed package contents and SQLite **3.53.1**. **Eighty focused regression tests passed while importing from this installed candidate package with the source-tree Python path disabled**. The preexisting live SWAAG package still differs from current source; voice gateway, communication daemon and model servers were not restarted or replaced. No remote push or Nitro merge is asserted.
