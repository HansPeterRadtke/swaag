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
420-second external deadline. Workers in the foreground matrix are genuinely recorded in isolated durable
state but deliberately not started; that matrix measures model routing and actual
queue effects. The separately labeled real-worker trajectory below also tests
worker execution and harmless file effects. The preceding day's real worker
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


## Audit-evidence ablation and recoverable context correction

The role-scoped installed quotation case still failed and queued an unwanted
control. It was stopped after 235.4 seconds; the remaining role cases and proposed
holdout were not run. The checker cited actor-policy text that was now present
only in historical audit evidence. Source inspection confirmed that historical
prompt-instruction events duplicated the current actor policy, while prior-turn
model request/response envelopes can duplicate entire earlier prompts and
unaccepted drafts.

A raw constrained checker experiment reused the exact failed request, with the
same endpoint, response schema, generation parameters and recorded state. It also
constructed a clearly labelled counterfactual with a real publication restriction
and a status-only candidate, for which completion must be false. No tools ran.
Repeating the complete request/candidate with explicit scope at the end still
failed the translation (69.9 seconds); it correctly rejected the genuinely
undelivered restriction (41.9 seconds). This prompt-order change was not adopted.

Replacing only the two core-policy audit text bodies with explicit exact-source
references passed both opposing raw checker cases: translation accepted in 27.3
seconds, undelivered real restriction rejected in 27.2 seconds. Other task data,
current state and original wording were retained. This is a two-case raw input
ablation, not installed end-to-end qualification. The negative is a synthetic
counterfactual, and its retained source identifiers are fixture metadata rather
than newly verified live history.

Completion context now distinguishes harness configuration/model envelopes from
canonical task evidence by their explicit event schemas. Core project-policy audit
records retain identity, scope, hash and exact recovery metadata rather than
reactivating their text as task instructions. Saved prompt/model-request/model-
response envelopes likewise have explicit audit references. Canonical user and
assistant messages, selected actions, tool observations and state remain exact.
No utterance keywords or task meanings decide this boundary. Current applicable
instructions still enter through the scoped instruction contributor.

The runtime_audit evidence provider advertises referenced audit events only, pins
the operation session, verifies the event integrity and exact source hash, and
allows the completion model to request the complete original event. All raw
records remain unchanged. Tests cover retained all-scope user rules, prior model
prompt/draft leakage, exact source recovery, changed hashes and cross-session
rejection. The core-policy-only group passed 44 tests; the generalized audit run passed 43 checks and exposed one new fixture missing
mandatory event-envelope fields. After supplying attempt and prompt_mode without
weakening assertions, both targeted scope/feedback tests passed. The final broader
run and installed fresh/persistent results are recorded below after observation.


## Installed audit-reference candidate

Code revision 1766a6d is installed on Jetson and Thor. All 195 package files match
source byte-for-byte on each, with per-file hashes retained. Both installed smoke
tests passed on both hosts. The final broader focused suite passed 86 tests in
151.4 seconds, covering completion sources, snapshots, authority, stream ordering,
intervening controls, durable instructions and context compilation. The preceding
invalid new event fixture remains recorded separately; its missing mandatory
attempt/prompt_mode fields were supplied without changing any assertion.

The first actual installed quotation rerun passed: zero worker controls, correct
translation, no failed completion/re-expansion loop, final response in 106.3
seconds. The completion reviewer explicitly recognized the quoted text as the
translation subject and the existing worker assignment as unchanged. Its action
call took 61.5 seconds and completion check 40.5 seconds. This is a functional
repair of the reproduced case, not a twenty-second interaction or proof that
all future quotations will be interpreted correctly.

The mixed English restriction also passed: its actual receipt preceded progress
at 68.0 seconds, final response at 155.3 seconds, exactly one Alpha control, no
output repair and one successful completion review. The scoped case preserved
Alpha-only delivery and produced progress at 56.3 seconds. It finished at 240.8
seconds, but its first reviewer incorrectly required worker acknowledgement and
check initiation before releasing the conversation. After requesting the exact
control-policy audit source, the second reviewer recognized the receipt-stage
contract and accepted the same candidate. This extra 54.5-second review is part
of the result, not omitted from the latency.

The German case FAILED the 420-second deadline despite correct delivery of one
control and progress at 82.4 seconds. Its second action exhausted a 288-token
output budget; the existing bounded repair completed a valid action. The candidate
then overstated the hold as already applied while also saying it was pending.
The reviewer demanded worker acknowledgement and check initiation and requested
two policy audit sources instead of finishing the conversation. Both returned
verdicts were incomplete; another call was in flight at the external deadline.
Exact authority, target identity and receipt-before-progress checks still pass,
but neither final-reply completion nor truthful presentation is certified by that.

This fixture intentionally leaves workers created and enables only control/status
tools. Therefore it cannot establish that the model would correctly start a real
worker, nor can it satisfy a demand to observe actual worker execution. It remains
a failed conversational case. A separate real-worker probe and opposing raw
completion-scope counterfactuals distinguish that fixture boundary from the
receipt-versus-application reporting problem.

For the three completed initial audit cases, recorded model calls consume
101.9/106.3 seconds (quotation), 146.9/155.3 seconds (mixed), and 229.1/240.8 seconds
(scoped). The remainders include preparation, scheduling, persistence and Python
work; they are not a measured pure scheduling phase. Effective output rate across
these four cases' returned calls is approximately 4.6 tokens/second at the median,
computed as output tokens divided by elapsed time after first output. This is not
provider-internal decoding throughput: prefix caching, queueing and host load are
not isolated. No twenty-second complete-action guarantee follows from a short
first-token time.


## Explicit quoted-delivery holdout

The actual foreground call routed the explicit quoted instruction to Alpha once,
with exact full original authority and progress at 60.5 seconds. It finished in
266.9 seconds. This passes the predefined routing/queue/completed-reply oracle,
but FAILS truthful wording: its final answer says both "Its checks remain running"
and "its checks haven't started." The first reviewer identified the unsupported
claim; after expanding the receipt policy, the second rationalized the contradiction
and accepted it. The same exact recorded candidate and evidence are therefore a
negative case in the next checker experiment. It must not be described as an
unqualified successful conversational response.

## Foreground handoff scope and persistent telemetry follow-up

Three raw checker probes added an explicit receipt-stage completion scope. The
truthful pending-status counterfactual passed in 28.7 seconds; an unsupported
applied/started counterfactual was rejected in 56.3 seconds; the exact contradictory
quoted-delivery candidate was rejected in 56.1 seconds. The latter preserves the
recorded candidate and evidence. The two other candidates are explicitly synthetic.
These are first checker verdicts, not completed runtime recovery loops: both
negative verdicts also requested further audit evidence. The raw probe appended
the scope to the system message; the installed candidate keeps it in the existing
budget-accounted foreground scope component. Live acceptance therefore remains
necessary and is not inferred from identical placement.

The completion scope now explicitly permits an accurate pending receipt response
while acknowledgement, initiation and execution remain background work. A user
who explicitly requests waiting for or verifying an effect still requires the
actual effect. An undelivered instruction remains incomplete, and a claim of
started/applied action cannot be inferred from a pending receipt. Worker completion
semantics are unchanged.

The persistent-session translation FAILED the original 420-second supervisor
limit, including its 76.4-second successful status priming turn. No worker control
was sent. Its target action was generated, but completion history then required
hierarchical evidence projection. The first attempted completion context carried
about 61,700 historical-evidence characters after just one ordinary preceding
exchange. Token/accounting/scheduling records dominated much of that volume.

A narrow explicit event-schema boundary now collects budget/accounting reports,
token progress, inference queue/start/finish events and instruction-selection
telemetry into one exact recoverable range source. It does not select by utterance
meaning, truncate task data or delete any raw event. User/assistant messages,
selected actions, tool results, questions and system snapshots remain exact.
Recovery pins the session and event range, verifies each event's integrity and the
bundle's hash, size, count, boundary references and event-type counts. An evaluator
requesting historical runtime measurements can recover all exact records through
the existing semantic evidence-request loop.

An offline reconstruction of the retained persistent case reduces historical
layout from 61,567 to 31,796 characters; source inventory grows from 4,140 to 4,972
characters. The combined representation falls by about 44 percent. The slight
baseline difference from the original compiled record reflects reconstruction
against the retained observation set; the exact original compiled accounting is
also retained. This is representation size evidence, not a latency result.

The existing completion/snapshot group passed 37 tests. Two new recovery tests
pass, including canonical interleaved user-event preservation, exact recovery,
changed hash/range and cross-session rejection, and model-requested expansion.
Initial new fixtures omitted a message timestamp and then compared an in-memory
message enriched with replay provenance against its persisted canonical form;
those fixture errors were corrected without weakening the preservation assertion.
The original failure logs remain alongside the passing results. The broader regression run passed 88 tests in 147.9 seconds. Installed rerun
outcomes follow below after observation.

## Installed handoff/telemetry candidate

Revision e1e38c5 is installed on both hosts. All 195 package files match source
byte-for-byte on each host, and both installed smoke tests pass on each. The
unchanged service configuration and existing model workloads are retained.

The first fresh translation rerun passes in 87.3 seconds, with zero worker
controls and a correct translated answer that leaves background work unchanged.
The German request finishes in 206.4 seconds, with receipt-stage progress at
99.1 seconds and one exact-authority control to Alpha. The explicit quoted-delivery
holdout finishes in 170.1 seconds, with progress at 61.3 seconds and one control.
Both pass the queue-target, exact-authority, identity and receipt-before-progress
oracles. Both FAIL truthful final wording: they say checks are continuing or are
left running while also saying that the created worker has performed no steps and
checks have not begun. The checker accepts these contradictions. Structural
routing success is therefore not conversational acceptance.

The persistent-session rerun successfully primes a status exchange in 77.0 seconds.
Its next translation action takes 46.9 seconds, but completion still overflows:
31,934 historical characters and 5,111 source-inventory characters produce a
14,652-token conservative input estimate against an 8,192-token backend window.
Hierarchical projection starts; its first result exhausts the 256-token output
budget after 60.5 seconds and enters bounded output repair. The telemetry change
reduces representation size but does not eliminate semantic projection.

The persistent turn again FAILED the 420-second external deadline, including its
77.0-second priming exchange. No worker instruction was sent. Three projection
responses exhausted their output limits before one projection completed; another
projection was in flight at the deadline. This is still a functional failure to
return the conversational answer, not a routing failure. The telemetry correction
alone does not qualify persistent-conversation latency.

A subsequent raw consistency ablation added an explicit clause-by-clause factual
check at the exact existing foreground-scope location, retaining the recorded
German request, candidate, evidence, schema and backend settings. It FAILED: the
contradictory candidate was accepted in 38.5 seconds. The explicitly synthetic
truthful pending candidate was accepted in 40.0 seconds. Neither requested extra
evidence. The unsuccessful checker addition was removed before deployment; this
experiment does not justify making the checker prompt longer.

A narrower actor candidate now distinguishes requested future work from observed
execution and instructs it to correct unsupported premises instead of echoing them.
It upgrades only the harness-owned user-control policy to version six, retaining
user rules. It is tested from the source package using the installed interpreter
and dependencies, with package identity and exact candidate diff recorded. It is
not yet an installed or qualified release. The first source candidate failed
before inference because its policy text exceeded the configured 2,000-character
storage limit; this is retained as `consistent-german_hold`. Its focused test run
also failed on the same guard and was interrupted after diagnosis. The policy
was shortened to 1,825 characters without raising the storage limit or removing
the exact-authority, conditional receipt or unrelated-work restrictions. The
bounded candidate uses the separate `grounded` label. Its 27 control, snapshot and
streaming tests passed in 75.8 seconds, but its first real German reply FAILED:
at 89.6 seconds it claimed publication had been halted and checks left running,
while the observed worker was created with no steps and a pending control. The
owned experiment supervisor was interrupted at 166.1 seconds after that decisive
spoken-stage failure. The remaining planned candidate cases did not run. The
candidate and migration change were withdrawn; their exact diff and failure
history remain in the evidence directory. Source/installed equality was rechecked
at all 195 package files. This failed prompt candidate was never deployed.

The installed-code real-worker experiment then started separately. Its supervisor
starts the existing worker only after observing the actual foreground control
receipt. It permits the local checks marker and forbids a publication marker
until user approval. This tests real model routing, exact authority consumption,
background execution and effects, not autonomous selection of a worker-start tool.

## First installed real-worker trajectory

The full trajectory FAILED its 900-second external deadline. It did establish
actual authorized effects and exact authority consumption: the check marker
contains exactly PASS, the publication marker is absent, and one worker control
with the complete original German message is processed. The worker was admitted
85.1 seconds after ingress and first spoken progress arrived at 85.2 seconds,
while foreground processing continued. The experiment supervisor initiated that
worker start only after the durable control, so autonomous start-tool selection
is not certified.

The model later retried the same agent_control. Both actual receipts have the
same durable control ID; the first says pending and the second processed. This
verifies replay deduplication after consumption. The second progress message
incorrectly says pending despite that processed receipt, so spoken truthfulness
fails. A stale foreground action was rejected before effects, and a later
structurally invalid action was also rejected.

The worker repeatedly returned a terminal message saying it was waiting for
approval without a blocking question, and its independent completion check
correctly rejected overall completion. It had not reached input_required when
the deadline terminated the owned experiment process. The foreground was also
stuck in evidence projection/output repair and never delivered its final reply.
The read-only `timeout-observation.json` reconstructs this outcome from durable
history, queue rows and actual files; it explicitly labels the remaining working
row as stale after process termination. Effects/authority pass, overall completion
fails. Production services were untouched.

## Reproduced stale-state recovery framing defect

The installed real-worker trajectory exposed a separate implementation defect.
After the snapshot fence rejects a candidate because worker state changed, the
next serialized prompt labels the feedback a mechanical error and says to
"Correct only that structural error." The next line explicitly requires
reconsidering the fresh task state. The contradictory wrapper also receives
remaining-work and tool-recovery feedback through the same input parameter.
The exact live excerpt is retained in `stale-recovery-prompt-reproduction.json`;
it proves the mismatch, not that this is the sole cause of every model failure.

The existing real-runtime stale-state regression was extended to inspect the
actual second request as well as proving fresh evidence and zero stale tool
execution. It fails against the old wrapper. The first assertion was initially
inside the model double, where the retry machinery obscured it; it was moved
after the completed fake trajectory so the retained red result directly shows
the contradictory serialized instruction. No runtime acceptance assertion was
weakened.

A shared feedback component now tells the model to use current authoritative
instructions and observations, repair structural errors where reported, and
reconcile changed state, controls, failed tools or remaining work instead of
preserving stale decisions. Feedback still grants no new authority and cannot
override the user. Joint action, capability selection and decomposed terminal
response use the same component; empty-feedback prompts and validators remain
unchanged.

The source-candidate real-model recovery probe PASSED in 122.2 seconds. The
experiment created a real second durable worker after the first action request
captured the one-worker state. The stale action was rejected without delivery or
effects; the next action received both the fresh two-worker state and corrected
recovery framing. The final sentence accurately listed Alpha and Beta as created.
Neither worker started, neither received a control, and completion evaluation
remained enabled. This is actual BF16 generation with a controlled concurrent
state change, not a model double or a latency comparison. The broader action, snapshot, control, streaming, completion and compiler run
passed 160 tests in 504.9 seconds. Installation and the repeated real-worker
trajectory follow after observation.

## Timing and comparison with Infra measurements

For the three completed installed handoff cases, model-call elapsed time accounts
for 82.7/87.3 seconds (translation), 197.9/206.4 (German instruction) and
163.2/170.1 (quoted delivery), about 95–96 percent of end-to-end time. Total local
admission waits are 0.110, 0.148 and 0.196 seconds respectively; no individual local
wait exceeds 0.081 seconds. The remainder is combined preparation, compilation,
persistence, scheduling and Python work, not a measured pure orchestration phase.
The retained `handoff-timing-decomposition.json` records every returned call.

The first action sees roughly 6,400–6,500 input tokens and first content after
8.5–8.9 seconds. Its complete 182/404/231-token structured output takes
47.9/96.4/58.5 seconds. Thus this Thor BF16 trajectory is chiefly delayed after
first output, unlike the earlier Jetson Q4 prefill-dominated trajectory. First
content is not a complete validated message-plus-tool action. Warm server caching,
provider-side queueing, transport and other host activity are not isolated; these
are not native prompt-evaluation or decoding counters.

Infra's `measurements.md` and model matrix have zero instrumented latency calls
for the BF16-capacity row. Its phase report supplies no equivalent current BF16
vLLM/8192-window service baseline. Q3's reported ten-token/second result uses a
different quantization/configuration, and the Q8 aggregates mix contexts/phases.
They cannot substantiate a like-for-like speed regression for this service.
Infra's methodology explicitly warns against treating heterogeneous rollups as
controlled throughput comparisons. Existing service settings and load were kept,
so no raw-speed ranking or throughput guarantee is claimed here.

## Refresh against the latest guideline additions

The complete earlier audit covered 242 files and 165 original recordings. The
current tree contains 246 files and 167 originals; four files were added and
12 existing files changed, with none removed. `guideline-delta.json` records the
exact current hashes and reading provenance. Both new recordings and both new
normalized documents were read in full, with all changed normalized/index/research
sections compared against the audited baseline. The canonical source-preservation
check reports 167 originals, 167 coverage entries, 167 preserved, zero errors.

The new answer-first/goal-relevance recording reinforces this study's distinction
between received, queued, started and completed, and the current user's task
outranking unrelated background goals. Current notes, editable prompt instructions,
Diane evidence and exact authority provide foundations; they do not certify
selective goal retrieval, explicit-versus-inferred goal provenance, supersession,
project exclusions or personalization quality. Those require their own held-out
acceptance. In particular, the generic Note schema and free metadata are not proof
of a complete goal workflow, and rendering all notes that fit is not proof of
selecting only applicable goals. No goal extraction or separate rendering model
was silently enabled by this study. The new learned-habit game guideline governs
a game task; it does not require embedding a game controller in SWAAG.

## Remaining acceptance boundaries

The deterministic regressions establish the runtime's authority, receipt identity,
message-presence, freshness and recoverability contracts. They do not certify
arbitrary model-authored wording. The created-worker matrix separately checks
queue effects, completed replies and receipt-stage streaming; it does not score
physical worker execution or all spoken-format requirements. In particular,
"queued", "acknowledged" and "applied" remain distinct states, and a reply that
uses stronger language than the observed state fails truthful reporting even if
its queue-target oracle passes.

The runtime has not been split into additional conversational models or processes.
Existing model services, memory reservations and optional communication daemon
lifecycle remain unchanged. The previous small-model and independent-backend
results keep their original limited scopes; this study does not qualify a new
model, remove the session ordering lock, or bypass completion evaluation.
