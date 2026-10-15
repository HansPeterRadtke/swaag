# Foreground routing and retry delivery — October 15, 2026

This continues the installed correction release in
[guideline-corrections-2026-10-14.md](guideline-corrections-2026-10-14.md).
The purpose is to test the actual ordinary foreground model and durable control
delivery, then correct observed implementation defects without a model split.

## Environment and evidence

Baseline source/install: cae1421 (runtime code 4689f39). Both existing Thor model
services were active with their original process IDs and zero restarts. The owned
nine-billion experiment remained inactive. Thor reported about 19 GiB available.
No model service was started, stopped, resized or restarted.

The actual installed BF16 host profile was used: qwen38-bf16-best at
10.8.0.7:14833/v1, vLLM, 8192-token context, thinking disabled. Client replay cache
was disabled; server prefix caches were not purged. Tests ran serially. Other host
activity was not suppressed, so these are warm-service observations, not a
controlled throughput ranking.

Evidence on Jetson:
`/data/var/swaag-benchmarks/foreground-routing-20261015`.
The probe, manifests, exact inputs, durable histories, streamed events, supervision
outcomes and compact summaries are retained. Each foreground case has a
420-second external deadline. Workers are genuinely recorded in isolated durable
state but deliberately not started; these tests measure foreground model routing
and actual queue effects, not worker execution. The preceding day's real worker
file-effect probes remain separate evidence for consumption and execution.

## Predefined foreground cases and baseline result

All four baseline cases completed and met the control oracle: one control per
affected worker, none to an unaffected worker, exact original user authority and
a completed reply. This is four specific cases, not an estimated general accuracy.

| Case | Meaning | Actual worker controls | First spoken progress | Final reply |
| --- | --- | --- | --- | --- |
| Mixed English restriction | Hold publication and answer status | Alpha: one | 59.5 s | 148.5 s |
| Quotation | Translate the prohibition; leave work unchanged | None | Final only | 79.9 s |
| German restriction | Continue checks, hold publication until explicit approval | Alpha: one | 66.9 s | 165.5 s |
| Scoped restriction | Hold Alpha publication; leave Beta unchanged; report both | Alpha: one; Beta: none | Final only | 150.6 s |

Durable input acceptance took 0.15–0.23 seconds. The answers correctly reported
that the created workers had not started their checks. The German progress wording
initially described a hold as already imposed before qualifying it as queued;
the final response clearly distinguished queue delivery. Queue acceptance is not
worker acknowledgement or completed effects.

Model first-token times ranged from approximately four to nine seconds. Complete
action generation took 37–64 seconds, and completion evaluation took 30–39 seconds.
A first token does not yet contain a validated executable action. The scoped case
generated its real control in the first action but left assistant_message empty,
so spoken delivery waited for the later final answer.

## Reproduced duplicate delivery and correction

A deterministic real-runtime reproduction called agent_control twice from the
same original foreground event, using a worker ID and its resolved session ID and
different model routing paraphrases. The old implementation created two control
IDs. The failing assertion and original output are retained.

Controls carrying trusted foreground authority now derive their durable identity
from the exact original authority records and resolved target session. Repeating
a model tool call, changing its routing paraphrase, or addressing the same target
through another alias returns the first receipt. A new user event, an additional
correction within the turn or another target remains a separate delivery. Direct
controls without foreground provenance retain their distinct-message behavior.
The existing transactional enqueue enforces first-payload preservation, including
concurrent attempts and retries after processing. Receipt status now reports the
observed pending/processed state; it is not proof of completed worker effects.

The scoped-case delay is addressed in the foreground control policy: a background
handoff should include a useful short acknowledgement and already-observed status
in that action's assistant_message. Unobserved actions remain intentions. Runtime
delivery still occurs only after successful receipts, and failed or superseded
drafts remain suppressed. Existing sessions upgrade the old core policy once,
preserving other instructions. This is model guidance; live adherence is tested
below rather than assumed from the text.

## Software verification

The duplicate-delivery regression failed before the correction. After correction,
68 control/history/stream/queue/scheduler tests passed. An additional 45 worker,
snapshot and architecture tests passed. Six authority tests passed including
concurrent receipt reuse, reopening durable state, a new correction, a new turn,
another target and direct-control compatibility. The final policy/control/stream
group passed 22 tests, including one-time policy upgrade and preservation of an
unrelated durable instruction.

Two mistakes in the new concurrency test fixture were corrected before it ran:
AgentEnvironment is constructed per session and imported from its implementation
module, rather than a runtime attribute or package-level export. Those initial
fixture failures remain in the evidence directory. They were not runtime defects.

## Prompt-only installed rerun

Revision e66e401 was installed on both hosts with all 195 package files matching
source. Installed smoke tests passed on both hosts. Four serial live cases preserved
the exact original authority and correct control scope. Mixed English delivered
progress at 61.9 seconds and finished at 151.4 seconds; quotation finished at
72.1 seconds without controls. However, scoped and German instructions still
omitted early progress, finishing at 194.4 and 250.4 seconds respectively.
The scoped final answer also used headings/bullets rather than the expected
spoken format. The explicit acknowledgement policy alone was insufficient.

## Enforced ordinary foreground acknowledgement

The ordinary first foreground action now carries an explicit, budget-accounted
prompt component requiring a useful nonempty assistant message in the same action
as authorized tool calls. Empty messages fail action validation before any tools
run and enter the existing bounded repair loop. Receipts still precede delivery;
superseded and failed drafts remain suppressed. Unobserved effects must be
described as intentions. Explicit user formatting requests take precedence over
the default spoken presentation.

This applies to the ordinary joint foreground action contract. Worker turns,
explicit silent turns and the separate experimental single-responsibility mode
retain their existing contracts. It does not add unsupported constraints to the
portable response schema. The runtime enforces presence, not semantic truth or
presentation quality; these still require live inspection.

The final focused regression run passed 36 tests across stream delivery,
intervening controls, fresh snapshots and trusted authority. A new regression
proves that an empty draft executes no tools, its repair executes exactly once,
and delivery follows the actual tool receipt. Four older cancellation fixtures
were adjusted to produce a valid nonempty initial draft; all cancellation
assertions remain intact. An initial implementation that changed the experimental
tool-only mode failed its compatibility regression and was removed. Both initial
failures and the final passing output remain in the evidence directory.

## Enforced installed acceptance

Revision e844850 was installed on both hosts, with all 195 package files matching
source and both installed smoke tests passing on each. The ordinary
acknowledgement/cancellation/snapshot/authority group passed 36 tests; a further
27 context/action compatibility tests passed, with 82 unrelated tests deselected.
The latter includes the unchanged experimental single-responsibility contract.

The scoped case delivered progress at 77.9 seconds and completed at 201.7 seconds,
with one Alpha control and no Beta control. Mixed English delivered progress at
66.9 seconds and completed at 164.7 seconds, with one Alpha control. Independent
durable-history inspection confirms successful queue receipts preceded progress,
target sessions matched, and exact original authority and receipt identities
matched the worker queues. No repair was needed for either initial message.
The scoped final answer still used a heading and bullets, failing spoken-format
adherence despite passing the finite routing/delivery oracle.

The quotation case FAILED: its first action correctly returned the translation
without a tool call. Completion evaluation then incorrectly extracted the
embedded prohibition as a real command and demanded agent_control. The following
action queued that unwanted control. Later evaluations alternated between
criticizing the unsolicited control and demanding acknowledgement of it. The
420-second supervisor deadline expired without a final response. Its exact
original foreground authority remained attached, but correct provenance alone
does not make unnecessary routing acceptable. Workers were not started, so no
publication or file effects occurred.

The deadline is included as a failure rather than silently omitted because the
external supervisor killed the process before result.json could be written.
A separate timeout-observation.json reconstructs the retained queue/events and
identifies itself explicitly as an observer record. The original trace is intact.

## Completion-review authority correction

The failed quotation trace exposed two problems in the surrounding instructions.
Completion feedback was introduced as a finding and required a materially new
action, discouraging a correct unchanged answer when the checker itself was
wrong. The receipt policy also expressed its final-status requirement without
explicitly conditioning that sentence on a requested, actually queued control.

Reviewer feedback is now explicitly fallible model output, subordinate to the
complete user request and observed evidence. Genuine omissions still require
correction. A model that identifies a mistaken review can explain the discrepancy
and submit a correct candidate to evaluation again; it cannot bypass evaluation
or manufacture an unauthorized effect to appease it. Ordinary foreground
completion receives an accounted scope component distinguishing the current user
turn from completion of all background work, preserving quotation/discussion
scope and requiring actual delivery of genuine restrictions.

Core control policy version four makes receipt reporting conditional on a control
actually requested and queued. Old core versions are upgraded once; unrelated
user rules remain. Current runtime snapshots are no longer incorrectly headed as
evidence exclusively from before the user turn. The complete original request,
exact evidence and freshness fences remain present.

The missing foreground-scope regression failed before correction. The first
completion/snapshot/control/stream group passed 51 tests. The final policy/label group passed 42 tests, including scoped completion context,
fallible review feedback, exact history/projection behavior and one-time policy
upgrade. Installed opposing live reruns are recorded below when complete.



## Failed scope-only rerun and role correction

Revision b6f7839 matched all 195 installed files on each host and both installed
smoke tests passed on each. Nevertheless its quotation rerun reproduced the wrong
review and unwanted control. The advisory-feedback wording and additional scope
text were insufficient. The owned experiment was stopped after 347.5 seconds
once the failure was retained, before the remaining scope-variant cases ran.
The supervisor received SIGINT; model services were untouched. This result is
recorded separately from the earlier deadline.

Inspection of the exact compiled checker prompt found both core actor policies
in its system instructions, because their scope was all model operations. Thus
the independent evaluator was also told to act as the user-facing orchestrator,
manage workers and issue receipt-stage responses. A focused regression reproduced
that role leakage while proving an ordinary all-scope user rule was present.

The two core actor policies now explicitly apply only to action and experimental
orchestrator-interaction calls. Role policy version four and control policy
version five upgrade their prior core versions once. Independent completion keeps
its evaluator role, exact user requirements, full historical evidence, current
runtime snapshot, foreground scope and freshness checks. User-owned all-scope
instructions remain system authority; historical policy events remain exact
recoverable evidence. This is explicit operation-role scoping, not automatic
semantic filtering of the user's message or a new routing gate.

The role correction passed 42 focused completion, authority and snapshot tests,
including preserved action-role instructions, retained all-scope user authority
and one-time core-policy upgrades.

The next live matrix retains the four predefined cases and adds an opposing
holdout: an explicit request to tell Alpha a quoted prohibition. That case must
forward the instruction, distinguishing authorized quotation delivery from a
translation that explicitly leaves work unchanged. No model-wide quality claim
follows from either outcome.
