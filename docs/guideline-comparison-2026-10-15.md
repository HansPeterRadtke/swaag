# Complete current guideline reread and SWAAG comparison — 2026-10-15

## Reading coverage and source identity

The complete current /data/infra/docs/guidlines corpus was reread, including all
167 original recordings. Infra revision:
867200e2c493f0c38c06e1121db3ad620f585b79. The latest integrity check at
02:01:33 UTC on October 16 accounts for 246 physical files, 234 unique whole-file contents,
12 byte-identical aliases, and all 108 reading pages. No new unread files,
missing files or changed source hashes were found. Infra's canonical preservation
checker reports originals=167 coverage=167 preserved=167 errors=0.

Exact consecutive repetitions were represented with their text and multiplicity
in losslessly verified reading copies. Source originals were not edited. The
preservation appendix duplicates the separately read originals and was checked
against them; its surrounding documentation was read. Truncated display portions
were reread with overlapping excerpts. Byte verification is preservation evidence;
the reading ledger separately records semantic findings and their comparison.

Evidence is under
/data/var/swaag-benchmarks/foreground-routing-20261015/full-reread, with the
whole-file source inventory in the parent directory. The semantic ledger retains
observations made during the reread, so intermediate test states there remain
chronological evidence rather than latest release claims. The repository ledger is
[guideline-comparison-2026-10-15-files.json](guideline-comparison-2026-10-15-files.json).

This updates the **current comparison**, not the historical contents of the
[October 14 baseline](guideline-comparison-2026-10-14.md) or the
[October 14 correction record](guideline-corrections-2026-10-14.md).
The current implementation and experimental trajectory are documented in
[foreground-routing-and-retries-2026-10-15.md](foreground-routing-and-retries-2026-10-15.md).

## Current comparison

Identifiers refer to the requirement families in the October 14 comparison;
G1 adds the latest goal-aware communication requirement. A family mapping is
not a claim that every sentence is mandatory or every condition has passed.

| Requirements | Current implementation and evidence | Remaining qualification |
| --- | --- | --- |
| A1, A2: exact authority and semantic decisions | Finalized messages persist before execution admission; ordered control identity and stale-action fences remain. Retried handoffs reuse a receipt without collapsing later corrections. Quoted text and completion-review feedback remain subordinate to actual user authority. The former lexical fast-reply override was removed in the earlier correction. | Real-model interpretation still has counterexamples. Valid schema, storage and tool permission are not proof of correct interpretation. |
| A3, A4, A7: conversation, immediate delivery, safe concurrency | Ordinary controller actions can contain user text and tool calls. Durable receipt and committed progress stream independently of the consumer; pausing speech consumption does not hold execution. The current hold fixture accepts in 0.15 seconds, dispatches its exact instruction, and observes real worker effects. | The hold trajectory takes 73.5 seconds to first semantic speech and 453.9 seconds to its final reply. Later fresh actual-state status takes about 89 seconds on production BF16 and 199–212 seconds on Q3; the accumulated-history continuation exceeds 1,800 seconds. Initial wording still mixes continuing checks with created/not-started state. Fast acknowledgement is not fast semantic completion. |
| A5, A6: current state on each relevant model call | Whole-system worker/plan/result/question snapshots and stale-state checks are implemented. Installed early refresh rejects stale prepared work before HTTP admission and during context reduction. The installed correction preserves exact current receipts outside lossy projection and refreshes question inventory from the same observation. | Sources are explicitly observed separately, not falsely atomic. Long-history exact receipt context has a real budget cost; no arbitrary top-N deletion was introduced. Ongoing churn can require retries. |
| A8, A9, A12, A13: scheduling, priorities, background work, supervision | Durable dependencies, configurable priority/preemption, weighted admission, authorized backlog, finite/continuous work, independent supervision and recovery already exist. Session mutation locking remains to preserve ordering. | Broader mixed-task/load and semantic supervision acceptance remains bounded. No heartbeat-based automatic kill or unrequested autonomous work was added. |
| A10, A11: completion and questions | Exact question evidence, revisions, ordering and durable input-required state exist. A live counterexample showed the evaluator accepting unfinished approval-gated work as completed. Candidate lifecycle context corrects the exact replay; the subsequent live worker enters input_required and the final answer is accurate. | Explicit approval now resumes the same worker and produces/verifies both authorized file effects. The installed mechanical durable-question gate corrects a model verdict that equated a resolved duplicate with the still-open original. The real worker closure now consumes original user-approval evidence and leaves no open worker question. The hold case contains a worker question and a foreground question; it does not alone establish their later reconciliation. |
| B1–B3: accounting, exact history and reduction | Whole serialized request accounting, exact evidence/recovery and semantic compaction remain. A reproduced impossible projection target is corrected to the actual generation allowance; its exact replay succeeds. Installed receipt preservation fixes an observed stale-context loss. The installed release adds source-backed, authority-keyed history reuse and measured preservation batches; a real eight-fact/decoy case retains exact evidence while reducing four calls to one and 47.6 seconds to 24.3 seconds. | Long-horizon compaction quality and availability under repeated budget pressure remain open. Model time still dominates many traces. |
| B4, B7, B8: large sources, reusable workspace and forge | Earlier installed corrections preserve exact streamed workspace output with recovery artifacts, pagination, mutation coordination, versioned/CAS catalogue, pinned versions, rollback/retirement and capability discovery. | Catalogue correctness does not prove all OS-level aggregate memory, disk, subprocess, network and service quotas or learned helper quality. Conditional reuse is not mandatory overhead for every simple task. |
| B5, B6, G1: memory, Diane and user goals | Source-aware notes, editable scoped prompt policy and Diane's evidence-based interface exist. The latest recording requires current-task-first answers and separate scoped, revisable, explicit/inferred goals with provenance, recency, priority/conflicts and exclusions. | Generic notes and free metadata do not establish the complete goal lifecycle or selective retrieval. Dedicated held-out tests are needed for supersession, fictional goals, unrelated-task exclusions, privacy and answer-first rendering. No personal-data collection or inferred-goal promotion was enabled. |
| B9–B11: attachments, external capabilities, model/tool experiments | Raw sources stay recoverable; capability schemas and external provider bindings exist. Shell versus structured tools, small readers, separate renderers, multimodal analyzers and model topology are experimentally evaluated options. | Small-model trials already exist and contain semantic failures. A Q3 twenty-seven-billion model is not a genuinely small reader. No new model or rendering pass is qualified by these fixes. |
| C1–C4: configuration, bounded resources, logging and measured performance | Earlier corrections add meaningful parameter metadata and policy. Queues reject overflow rather than silently discard accepted commands. Durable authority is separate from bounded operational logs. Timing distinguishes model generation, admission, receipt, effect and final delivery. | Model-call count, long context, repeated reduction and queue delays remain material. Overlapping accumulated admission waits are not additive wall-time phases. All-domain OOM/load safety is not inferred from local tests. |
| C5–C7: infrastructure, networks, security | Guarded model admission and reserve policy remain authoritative. No production model was started, evicted, resized or restarted in this study. Existing authenticated/loopback transport boundaries, tool policies and source trust remain. | Voice-device lifecycle, long network partitions and protected external-provider deployments require their actual environments. Prior resource incidents remain failures in historical evidence, not authorization to bypass guards. |
| C8–C11: engineering, traceability and truthful release claims | Red reproductions precede source correction; immutable candidates, focused tests, actual effects and frozen negative/held-out cases are separately recorded. The complete original corpus distinguishes requirements, experiments, examples and research. | Historical failures remain recorded: the early lean-capacity regression and later snapshot-order regression are reproduced and corrected without relaxing their tests. The combined history-reuse/current-status candidate passes 1,307 tests with three optional skips; the installed measured-selection source passes 1,312 tests with three optional skips on Thor, after isolating a CLI test from the host replay cache. The preceding 525ef45 release had installed byte equality and passing smoke tests on both hosts. The further partial-history recovery correction passes 17 focused checks and the complete 1,314-test suite with three optional skips. Native selector-output counting then passes 1,323 tests with three optional skips and actual tokenizer checks on both deployed models; its whole-turn trial is distinct. Preparation-scoped identical-selection reuse then passes 1,329 tests with three optional skips; whole-turn outcomes and installed certification remain separately recorded. No failed run is relabeled successful. |
| D1–D5: audio, GUI and voice | Core audio-style policy, question/status streams, ordered finalized text and speech-independent execution are implemented. VAD, transcription, semantic turn finality and permission to execute remain distinct. | Actual model wording and frontend playback are behavioral gates. No microphone, Android layout, physical TTS, editable transcript or semantic barge-in acceptance is claimed by these text fixtures. |
| D6–D9: games, trading, robotics, KMD and source support | All domain and original-source material was read. General requirements for causal evidence, independent oracles, failure testing, recoverable state and resource ownership apply to SWAAG engineering. | Domain architectures, physics/game controllers, market hypotheses, predictive models and physical robot acceptance are conditional project work, not missing mandatory orchestrator features or authority for unrelated side effects. |

## Architecture conclusion and next acceptance

The logical worker, orchestration and user-conversation responsibilities are
required. Their model/process count is not prescribed. Existing shared-backend
and independent-backend concurrency evidence is retained with actual routing
identity; the shared preemption benchmark is not relabeled independent. Current
repairs keep these boundaries and the session ordering lock intact.

The latest held-out completion-scope matrix was frozen before requests and passes
all ten calls: temporary hold is incomplete; permanent cancellation can complete;
a foreground status answer or verified queue handoff can complete while background
work remains pending; an unobserved requested effect is incomplete. Both detailed
and verdict-only completion modes were tested on the already-running Q3 endpoint.
This is bounded model-evaluation evidence, not an end-to-end execution test or a
population accuracy estimate.

The saved worker approval/resume trajectory now passes with actual effects and
question closure. The installed correction release is byte-verified on both hosts. The measured-selection release is installed after source, bounded live, equality and smoke checks.
Its accumulated-history continuation and the new partial-progress candidate remain
explicit acceptance scopes; the latter has not been installed. First-response
truthfulness, long-turn responsiveness and goal-memory workflow need distinct
acceptance evidence. Further architectural change must answer those measured
limitations rather than assume that a separate small model solves them.
