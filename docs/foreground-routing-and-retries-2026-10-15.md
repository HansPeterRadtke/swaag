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
passed 160 tests in 504.9 seconds. Revision c532164 is installed on both hosts;
all 195 source package files match installed bytes on each, and both installed
smoke tests pass on each. The repeated installed real-worker trajectory uses the
same configuration, effect/state oracle and 900-second deadline under the
`recovery-german-worker-effects` label. Its manifest now also records the prompt
builder hash and exact probe script; those additions do not change the test
semantics.

The installed rerun completes its foreground reply in 432.6 seconds and the
worker in 641.8 seconds. Allowed checks, withheld publication, exact original
user authority and a single processed control all pass. The predefined overall
oracle still FAILS: the final reply calls the already-consumed control pending
and reports no progress; the worker eventually becomes completed instead of
input_required, despite explicitly requesting approval in ordinary prose. Two
completion reviews reject that unfinished objective; a third accepts the same
blocked state as complete. This is not an approval-wait success or a fully
truthful status response. `recovery-german-worker-effects/trajectory-proof.json`
retains the exact effects, receipts, model calls and final states.

## Current receipt and worker-report snapshot correction

The rerun exposed two concrete context omissions. A consumed worker control
vanished from the snapshot, while its earlier pending tool receipt remained in
conversation evidence. The worker's latest semantic progress was also omitted
unless it had used explicit progress reporting or reached a terminal state.
Thus a new agent_status alone did not change the freshness fingerprint.

Two regressions reproduce these faults before the correction: a processed
receipt is absent, and a first-stage-pending foreground draft is delivered even
though the worker recorded first-stage-done during generation. The initial
status test omitted required event fields; that harness error was corrected
before retaining the clean red reproduction. No acceptance criterion changed.

The snapshot now derives pending controls and compact delivery receipts from
one canonical query after legacy inbox import. Receipts retain identity,
current pending/processed status and timestamps without copying processed
instruction text into the current authority lane. All receipts are retained;
there is no arbitrary recent-N cutoff. Existing context-budget projection and
exact source recovery apply if the snapshot does not fit. The prompt explicitly
distinguishes consumed delivery from verified execution. The already-collected
worker inventory also includes its latest model status with source sequence and
hash, labeled as a report rather than independently verified effects. Material
report changes now invalidate stale foreground drafts.

The actor template also contained an ambiguous terminal instruction to finish
and wait for the next message. A source candidate now states the actual runtime
protocol: unfinished work blocked on a user answer or authorization needs a
structured blocking question, which preserves input_required; waiting only in
prose does not create that state. This changes guidance, not semantic routing
rules, the action schema or the completion gate. Its acceptance is the repeated
real-worker trajectory, not a string check.

Snapshot/control/question regressions pass 42 tests in 87.5 seconds. Existing
question-schema, streaming and control regressions pass 25 tests in 67.9 seconds.
The `observed-german-worker-effects` source-candidate rerun retains the same
900-second deadline and predefined effect, authority, wait-state and foreground
completion oracle. Exact source hashes, candidate diff and probe are retained.
The worker reaches input_required with a real structured blocking question,
checks.txt containing PASS, and no publication.txt. Its first spoken foreground
reply still mixes continuing work with a created/not-started observation, so
that wording is not certified. The foreground fails its 900-second deadline after 15 returned model calls,
including repeated evidence-projection exhaustion. The complete predefined
oracle therefore still fails despite the corrected worker wait. This process
imported the reducer before the following allowance correction; it is not a
live test of that later fix. Read-only timeout and trajectory records preserve
that distinction. The single control is processed once and the one blocking
question is durable.


## Reproduced projection target/output allowance mismatch

The persistent translation trace has five exact evidence-projection requests.
The first pair asks for approximately 1,727 content tokens while allowing only
256, then 384 generation tokens. The next fragments ask for 864 while allowing
256/384. `projection-allowance-reproduction.json` retains the actual serialized
instruction and allowance. The reducer bounded the generation request by the
configured reserve but passed the larger caller target unchanged into its prompt.
That inconsistent contract caused observed output exhaustion, retry and exact
source subdivision; it does not establish that every projection failure has this
cause.

The reducer now bounds the instructed content target by its generation allowance
with the existing JSON framing slack. If measured input pressure reduces output
headroom further, it rebuilds that instruction against the admitted budget before
making a model call. The local compilation target only decreases, so it converges;
raw input is retained, no keyword relevance filter or blind source truncation is
introduced, and output validation/recovery remains enabled. Tiny targets keep
their existing target-aware allowance.

The regression fails against the old requested target of 1,727 with only 256
admitted tokens. The near-full-context fixture was adjusted from 1,800 source
words (which triggered subdivision) to 1,600 to exercise the intended distinct
headroom-contraction path. With that valid fixture, both the large-target and
small-target budget checks pass. The broader context-engineering, completion and compiler run passes 46 tests in
193.3 seconds; notes, workspace and communication-status consumers pass 34 in
44.3 seconds; tool-result and prompt-instruction projection consumers pass 13
in 61.1 seconds. Together with the snapshot/control/question and streaming
sets above, these are 160 focused passing tests. The real repeated conversation
is queued sequentially after the worker run, preserving exclusive study load;
its result determines the live acceptance and performance claim.



The repeated source conversation still FAILS the unchanged 420-second deadline,
but follows a different trajectory. Priming takes 344.7 seconds: five action
generations include four identical successful status lookups before an accurate
answer and completion review. Exact lookup results are present in subsequent
serialized requests; the runtime did not lose the observations. The requested
translation then starts with about 75 seconds of the total deadline remaining
and times out while projecting repeated prior tool results. No worker control
is sent. The initial intermediate observation that priming was still unfinished
is retained separately; the final timeout record includes its later completion.

This is a model decision loop and a remaining persistent-context cost, not a
controlled comparison with the earlier completion-evidence projection path.
The test's priming time is not subtracted from its predefined deadline, and no
translation success is inferred from zero controls. A separate installed replay
of the exact previously truncated evidence fragment, purpose and caller target
isolates the generation-allowance correction after the installed worker run.

The receipt/progress/wait correction is committed as 2f6891d and the projection
allowance correction as 1d3a0fe. Both hosts are installed at 1d3a0fe; each has
195 source-package files byte-identical to the installed package and two passing
installed smoke tests. Installation did not restart either Thor model service.
The installed real-worker acceptance (`current-german-worker-effects`) is queued
after the persistent source study so those trajectories do not compete for model
admission. Source versus installed execution is retained explicitly in manifests.


The installed BF16 full-worker rerun PASSES the predefined effect/authority/
wait-state/foreground-completion oracle. First progress arrives at 88.2 seconds;
the accurate final answer arrives at 773.3 seconds, and the supervisor exits zero
at 776.0 seconds. The exact original instruction is consumed once, checks.txt
contains PASS, publication.txt is absent and the worker is input_required. The
final reply reports those effects accurately and asks for publication approval.
There are two blocking question records, one worker and one foreground; duplicate
question reconciliation and a later approval/resume cycle are not tested here.
The initial progress wording still mixes continuing work with a created/not-started
observation and remains an explicit presentation failure. Eighteen model calls
return and one stale action is rejected. This is functional success for the
predefined fixture, not interactive responsiveness or universal model qualification.

The exact projection replay initially fails before inference because its harness
configured an empty enabled-tools list, which configuration validation forbids.
That failed script/error is preserved. The corrected reducer-only probe enables
a read-only status tool (without executing it), uses a new evidence directory,
and is queued after the isolated freshness candidate to avoid overlapping load.

## Earlier task-state refresh candidate

The installed trace shows repeated context reduction of an observation captured
before the worker finishes, followed by a stale action rejection after generation.
Two additional regressions reproduce avoidable stale work: a reduction round
returns a projection of the old snapshot after a worker transition, and a prepared
request is sent even when that transition occurred while waiting for admission.
The existing post-generation fence prevents stale effects but does not save that
inference call.

A detached worktree at the same installed base isolates this candidate from the
running backend comparison. It refreshes the whole-system source and discards
only that source's derived projection between reduction rounds. It also compares
task facts after inference admission and before transport, releasing the admission
and preemption records as superseded if stale. The existing post-generation
check remains. All three sites share one comparison helper, and newer pending
user controls remain under the existing ordered authority fence rather than being
consumed by this task-state refresh. Provider errors at the newly added admission
check also release the acquired slot.

Both new regressions fail against the base and pass with the candidate. The
broader snapshot, preemption, instruction-control, stream and inference run
passes 94 tests in 205.0 seconds. A mistaken test filename first caused no tests
to run; the corrected invocation and its complete successful output are retained.
A real BF16 probe is queued after Q3: Beta is created after Alpha-only context
obtains inference capacity, and the stale request must be superseded without any
HTTP model request, then answered from the fresh two-worker state. No workers
start or receive controls. This candidate is not yet merged or installed.

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


## Installed same-workflow backend comparison

The existing Q3 endpoint advertises qwen38-q3-sched2, the Qwen3.8-27B
UD-Q3_K_XL GGUF, and an 8192-token maximum. The installed full-worker fixture is
queued on that already-running endpoint after the BF16 worker and exact reducer
replay. The only fixture configuration changes are base URL, profile name and
model identity. The same isolated files, original German instruction, exact
control authority, approval-wait oracle, completion evaluation and 900-second
deadline apply. This is a shared controller/worker backend comparison, not a
small-model or independent-assistant experiment. Server quantization and slot
configuration differ; it is not a controlled raw-throughput ranking. No model
is started, evicted, resized or restarted and production selection is unchanged.

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


## Completed comparisons and freshness promotion — 20:56 UTC

Q3 completes the same installed worker fixture in 505.7 seconds (first spoken
progress 55.8 seconds), with PASS checks, no publication, one exact processed
control, and durable input_required. Its predefined functional oracle passes.
Manual communication review FAILS: the final reply repeats an obsolete pending
receipt despite current processed state, calls the hold a release instruction,
and speaks a raw checksum and internal receipt syntax. Initial progress retains
the running/created contradiction. Fifteen model calls return; local admission
waits are 64.6 seconds for the worker and 200.7 seconds for the foreground. These
are overlapping accumulated waits, not additive wall-time phases. Two blocking
questions again remain; later approval/resume is outside this fixture.

The early-freshness source candidate PASSES actual BF16 inference: the Alpha-only
prepared call is superseded at admission before HTTP transport; exactly one fresh
action request is sent. The answer correctly reports Alpha and Beta both created
in 64.7 seconds. Worker starts, controls and file effects remain zero. There are
also completion-evaluation calls; one action request does not mean one total LLM
call. This is a controlled state-change fixture, not a universal latency claim.
The change is committed/pushed as bd3042c and installed on both hosts. Each has
195 byte-identical source/installed package files and two passing smoke tests.
No production model service was restarted or reconfigured.

The corrected exact-fragment reducer replay PASSES in 54.6 seconds with one model
request and no output exhaustion. Its prompt asks for 232 content tokens within
256 generation tokens instead of 1,727 within 256. The returned projection reports
that this *pre-target* history contains no translation and preserves the created
worker state. This is isolated evidence for the inconsistent-budget correction,
not proof of persistent-conversation completion or every projection's accuracy.
The failed empty-tool harness made no model calls and remains separately recorded.

The full current guideline reread is now in progress in addition to the earlier
baseline/delta comparison. The exact source manifest covers 246 files (234 unique
whole-file contents plus 12 exact aliases). A reading display preserves exact
consecutive repetition with explicit multiplicities and verifies lossless source
reconstruction; originals are unmodified. The source appendix duplicates the
originals and is integrity-checked separately. Page generation is not recorded as
semantic reading. The semantic reading ledger records only inspected portions.


## Current facts lost during projection: reproduced candidate correction

The Q3 final request at sequence 155 retained the earlier exact pending receipt but omitted the processed receipt observed in snapshot sequence 111. The semantic snapshot projection at sequence 147 dropped its lifecycle facts. The independently rendered question inventory also remained from an older outer-loop observation. Evidence: `q3-german-worker-effects/receipt-context-diagnosis.json`.

An isolated candidate at `current-facts-candidate` now renders current exact control receipt identities, statuses and lifecycle timestamps as mandatory, nonprojectable context, with explicit consumption-versus-execution semantics. It refreshes the separate durable question inventory from the same frozen snapshot before assembling context, and avoids repeating that inventory or those receipts in the projectable snapshot body. Historical source events remain untouched and recoverable. Exact receipt context has an explicit token-budget cost and is not silently truncated; large receipt histories remain a capacity consideration.

Both regressions fail for the observed reasons on the prior implementation (the first draft of the inventory test used an invalid event name; the corrected red run uses the actual `agent_status` event). Candidate tests: 21 snapshot tests passed in 96.16 seconds; 65 context/authority/question/control tests passed in 80.71 seconds. The candidate is not yet installed. Uncached Q3 worker-effect fixture `current-facts-q3-german-worker-effects` is underway; delivery, effects, waiting, first wording, final wording and latency remain separate acceptance checks.

### Current-facts Q3 live result and separate completion defect

The candidate `741bf61` passed 86 focused/related deterministic tests. Its Q3 worker-effects fixture completed the foreground response in 428.17 seconds (first progress 85.37 seconds), with one exact authoritative control consumed, checks marker `PASS`, and no publication. The final answer accurately reports checks and withheld publication, unlike the previous Q3 stale-pending answer. However, the worker recorded `completed` instead of `input_required`, with no blocking question, so the predefined overall oracle **fails**. This is not a deployable claim of end-to-end success.

Worker sequence 69 selected a terminal action with empty questions while its own status acknowledged pending approval. At sequence 81 the independent completion evaluator returned complete=true because nothing more was authorized until approval. That confuses a temporary hold with finishing the retained objective. The actor prompt already describes the durable-wait protocol; the evaluator lacks that lifecycle distinction. A separate isolated candidate tests explicit evaluator context for completed versus paused, cancellation versus temporary hold, and foreground versus background scope. No automatic execution is authorized by that context, and no production model service is changed.

## Completed full guideline reread and approval-wait correction

The current whole-corpus reread is complete: 246 files, 234 unique contents plus
12 exact aliases, all 167 originals, all 108 reading pages. Exact preserved
repetition displays and source hashes were checked; the canonical appendix
checker reports zero errors. The complete current comparison and file mapping
are in [guideline-comparison-2026-10-15.md](guideline-comparison-2026-10-15.md).
Historical reading-ledger findings retain the intermediate state at observation.

An exact replay of the incorrect current-facts completion request reproduced
complete=true in 20.660 seconds. Adding explicit completion-versus-durable-wait
semantics changed that same development case to complete=false in 13.316 seconds.
The timing is not a speed ranking: prefix cache, output length and load differ.
The actual candidate places the shared lifecycle instruction before the candidate
answer; the isolated replay appended it to the request. Both completion modes
include the same lifecycle component. Temporary hold, permanent cancellation,
foreground status scope and verified handoff are explicitly distinguished.

The actual Q3 wait-state candidate fixture then passes the predefined effect,
authority, wait and foreground-completion oracle. Receipt arrives at 0.152
seconds; first speech at 73.531 seconds; accurate final answer at 453.910 seconds.
The worker consumes one exact authoritative control, writes checks.txt=PASS,
leaves publication.txt absent, and records input_required. Ten model calls
return; three stale preparations/results are rejected. Accumulated local
admission waits are 69.0 seconds for the worker and 128.8 seconds for foreground,
overlapping rather than additive wall-time phases. Manual final-wording review
passes; first progress still mixes continuing checks with created/not-started
state and fails its separate truthfulness gate. Worker and foreground each
retain a blocking question. The initial worker is deliberately started by the
fixture supervisor after the control receipt; this is not autonomous start
selection evidence.

A separately frozen synthetic completion-scope holdout passes ten of ten actual
Q3 requests, covering five cases in full and verdict-only modes: temporary
approval hold remains incomplete; explicit permanent cancellation can complete;
a foreground status answer and verified pending handoff can complete without
waiting for a background worker; a claimed unobserved file effect is incomplete.
These cases were not used to tune the tested candidate. They qualify this bounded
reviewer distinction, not arbitrary model semantics or actual worker execution.

Focused completion/question/action tests pass 133 checks in 694.65 seconds.
Earlier two invocations named nonexistent test paths and ran zero tests; they
are retained as harness errors, not counted. The complete installed-baseline
suite found one real lean-prompt budget regression: 1,227 passed, three skipped,
one failed in 1,519.57 seconds. Added wait guidance made the fixed small-context
request 28 tokens too large. Concise equivalent guidance reduces the lean rule
from 54 to 23 whitespace words while preserving the durable-wait distinction;
the unchanged original fallback test now passes. The combined candidate passes
53 focused tests in 203.66 seconds. Its immutable full run is separately retained.
The earlier current-facts full run was explicitly superseded, not called a pass.

## Explicit approval exposed a disconnected resume path

The next experiment reopens the actual persisted waiting task and delivers a
new explicit German approval through the ordinary foreground API. The standard
questions capability is added to the earlier deliberately restricted fixture,
so question resolution is possible. No supervisor manually starts or resumes
the worker in this continuation. Before-state copies, the exact message, source
hashes and acceptance oracle are frozen in approval-resume-q3.

The foreground emits a real agent_control call and retains the exact approval
at 131.3 seconds, but the worker remains input_required with the control pending.
Source inspection explains this: agent_control only enqueues. TaskApi.message
and explicit resume can wake a worker, but the ordinary controller does not call
them. The scheduled-wakeup dispatcher considers due wakeups, not arbitrary
queued answers; the ordinary orchestration tick did not recover this queue.
This is a dispatch defect, not evidence that the model cannot interpret approval.

Three focused regressions reproduce the gap: worker-ID delivery, session-ID
delivery, and recovery after enqueue-before-dispatch interruption. Seven negative
checks pass on the old implementation. The first red invocation used the wrong
working directory and ran zero tests; the corrected red run is the evidence.

Candidate 0dc71b3 connects pending controls to the existing worker lifecycle.
It awakens only input_required workers with pending agent_control messages.
Created, completed, failed, canceled, cancellation-requested and archived states
are not implicitly started or revived. Delegated-tool waits remain blocked.
Awakening means semantic reconciliation of original authority, not approval by
the dispatcher or automatic question resolution. Periodic service reconciliation
recovers the enqueue/submission interruption window. The input-required commit
now rechecks pending controls under the same transition lock, and targeted
delivery enters that lock even if it observes working just before the wait.

Fourteen direct/negative/race/capacity/owner-routing tests pass in 27.42 seconds.
They verify exact authority, one submission across retries, no revival after a
consumed receipt, separate-backend ownership, canceled/archived/delegated guards,
and eventual dispatch after capacity clears without losing the accepted answer.
The broader first focused run passes 69 tests in 137.50 seconds; its early import
preceded the final targeted-race refinement, which is covered by the fourteen-test
run and the new immutable full-suite gate. No production installation is claimed
for this candidate yet. A live recovery test is scheduled after the preceding
observer ends and will use the already accepted explicit approval, not invent a
new authorization or rewrite history.

## Approval recovery, question provenance and projection follow-up

The first approval continuation expires at the unchanged 900-second deadline.
The exact instruction is queued, but the worker remains input_required and the
publication marker is absent. Fifty-six model calls returned before termination:
16 anchor, eight verbatim-selection, 24 evidence-projection, three action, four
history-selection and one prompt-instruction projection calls. Their returned
model wall times sum to about 761 seconds; outstanding/other phases explain why
this is not the entire wall time. Three distinct serialized selector requests
were each repeated identically seven times. The immutable before-recovery copies
under approval-dispatch-recovery-q3 preserve this failed observation.

Actual dispatch recovery then starts the saved worker without a new user message
or a manual resume. It consumes the already accepted exact approval, writes the
publication marker and reads both exact file values back. However, overall
closure fails after 209.32 seconds: the model copies an answered question into
the new-question array while attempting a resolution tool call, then follows
misleading repair feedback by dropping the resolution and asking again. The
worker remains input_required with two open copies. Successful effects and
dispatch are therefore reported separately from failed durable closure.

The question-resolution tool also rejects the legitimate forwarded original-user
approval as evidence: it only accepts local user-message events, while this
worker receives original authority through a processed control. Two focused
red tests reproduce this missing evidence path and absent local citation;
three negative provenance checks pass. Candidate 2aefc0b accepts only processed
agent_control evidence backed by the existing durable original-user authority,
never a routing note, scheduler control, bare tool result or invented receipt.
The current authority context supplies the local consumed-control event sequence
for citation without promoting the routing paraphrase. Action guidance clarifies
that the questions array opens new unanswered questions; resolution uses the tool,
an empty new-question array and continued execution. The new wording is shorter
than the former lean guidance. Thirty-eight focused provenance/lifecycle/authority/
criticality/small-context tests pass in 16.03 seconds. A mistaken test filename
in the first invocation ran zero tests and is not counted.

A separate source counterexample showed goal-specific tool projections being
reused without checking their objective. An Alpha-only projection was returned
for a later Beta question, even with explicit different-goal metadata. Candidate
2f91350 binds active and archived projection reuse to the exact objective,
including later user interventions in action reduction. Legacy projections without
that identity remain preserved but are not silently reused for a new selection.
Within one frozen action preparation, an unchanged declined source/goal/target
selection is not retried merely because another context source was reduced.
Changed target, goal or source remain eligible; no global inference replay is
enabled and no authority is dropped. Three red tests reproduce the old behavior;
14 focused/edge checks pass in 90.85 seconds, including archive and changed-target
reuse boundaries.

The immutable pre-dispatch candidate 26b6282 passes its complete suite:
1,231 passed, three optional skips, 1,583.08 seconds. The intermediate dispatch
full run is explicitly superseded, not called a pass. The final combined candidate
2aefc0b has its own immutable full-suite run and a real saved-worker closure
experiment. That phase resumes the incorrectly reblocked worker without adding
new authority and requires both questions resolved, completed status, unchanged
verified files and no repeated write call. It is not relabeled a clean original
end-to-end timing trial. Production remains on bd3042c until final certification.


## Completion context recovery after the approved worker's duplicate question

The 2aefc0b saved-worker continuation resolved the duplicate question through
processed original-user control evidence at local sequence 95. This verifies the
new provenance path with the real Q3 backend. It did not resolve the earlier
question. The same worker remained working after 850.544 seconds; its supervisor
terminated the test process at 900.023 seconds. The only tool call in this phase
was questions; both previously verified file hashes were unchanged. This is an
overall failure, not task closure. The dependent foreground status fixture correctly
refused to run after the failed precondition.

The completion reviewer never reached inference. Its measured input was 39,719
tokens against an 8,192-token context, including 29,549 historical-evidence tokens
and a 3,226-token source catalogue. Nested budget_reports in prior turn_finished
events duplicated thousands of diagnostic characters. Individually catalogued
model envelopes also grew the mandatory catalogue. With other components already
over budget, history reduction had no available target; reducing a tool result to
64 tokens failed and repeated terminal attempts made no progress.

Candidate 67414da preserves accepted turn answers and outcome fields exactly,
replacing only nested mechanical budget reports with integrity-checked recovery
references. Model prompt/request/raw-response envelopes join the existing exact
telemetry range; canonical user messages, selected actions and actual tool results
remain independently available. The range stores its event-type selection, so old
references recover their original exact events even after additional audit types
become supported. Old individual model-audit references remain readable. No raw
history is removed or rewritten. Completion also receives current unresolved
questions directly from authoritative state instead of inferring their current
status solely from historical question events.

Fifty focused completion/snapshot tests pass in 143.09 seconds. Six dedicated
recovery regressions pass in 7.95 seconds, including exact re-expansion, unchanged
canonical events, legacy range/individual references and fail-closed malformed
metadata. Early test fixture errors are retained in the artifacts and are not
counted as runtime defects; commit 65fb5c6 corrects the new fixtures' mandatory
model-event attempt field without changing runtime code. The premature full run
was superseded, not counted as a pass. The immutable corrected candidate's full
suite and saved-worker recovery are now running. The latter uses ordinary orphan
reconciliation and resume after the recorded timeout, with no new user authority,
no permitted repeat file writes, and unchanged file-hash oracles.


### Failure-history growth and the final audit bundle

The 65fb5c6 continuation can reduce history, but the failed earlier run had also
accumulated sixteen action-mode fallback events containing two full budget reports
each and fourteen unavailable-review events containing another report. An exact
same-prefix comparison at sequence 792 therefore measures the whole failure
history, not only the first failing completion call. The original canonical prefix
is unchanged at 4,758,671 characters, SHA-256
3d9739f998a9a5da4b94f1af31a83b6a49caeb17464c13df744a498383964894.

| View | Historical characters | Catalogue entries | Catalogue characters |
| --- | ---: | ---: | ---: |
| Original 2aefc0b | 414,440 | 153 | 76,469 |
| Turn/model audit 65fb5c6 | 290,320 | 3 | 1,810 |
| Nested budget references, intermediate | 130,334 | 17 | 9,025 |
| Final shared audit bundle 2e9b79b | 130,488 | 1 | 955 |

These are exact character counts for one fixed source prefix, not measured tokens,
latency or model-quality scores. The final bundle retains accepted turn answers,
turn outcomes and review failure reasons as separate visible semantic rows. Only
their nested diagnostics use the shared exact source reference; full original
events remain hash-verified and recoverable. Runtime fallback accounting is also
bundled. A dedicated eighty-failure regression reproduces catalogue growth before
the fix and checks exact recovery and every visible failure reason afterward.

The intermediate field treatment passed 54 related checks in 184.75 seconds.
The final refinement has its own focused and complete gates; the incomplete
65fb5c6 full run is explicitly superseded. The combined pre-audit candidate
2aefc0b completed its full suite with 1,256 passing tests and three optional skips
in 1,629.69 seconds. That result does not certify the newer audit change by itself.
The final immutable candidate is 2e9b79b, still not installed at this checkpoint.
A fresh guideline-integrity check sees all 246 reread files unchanged, with no new
or missing source files.


## Durable blocking-question gate and recursive review catalogues

The turn-audit continuation finished in 781.799 seconds (supervisor 785.225),
without another file write, but failed its frozen oracle: the worker became
completed while the original blocking question remained open. The reviewer at
sequence 1127 cited the resolved duplicate question as if it settled the original.
The authoritative unresolved-question component was present; valid model output
and a true completion verdict were therefore insufficient. WorkerManager also
looked only for question/revision events in the current run. That produced both
wrong directions: an older unresolved question did not prevent completed, while
a newly created question resolved in the same run still forced input_required.

Candidate eebb7e8 reads the current replayed question state when assigning worker
status. Managed worker sessions also receive a mechanical pre-completion gate:
existing blocking questions must be explicitly reconciled through the question
tool before a terminal completion candidate is reviewed. This gate applies even
when optional model completion evaluation is disabled. Foreground status replies
are a separate scope and may finish while a local/background question is pending.
A new blocking question still ends a turn in a durable wait. The gate supplies
exact unresolved IDs and requires resolving answered questions without copying
them into the new-question array. It does not infer answers or create user authority.

Three deterministic red cases reproduce the two worker-state errors and the
missing terminal guard; a foreground-status negative passes before the change.
All four pass after correction in 11.64 seconds. Seventy-two related tests pass in
123.55 seconds, including actual question-tool resolution, waiting-control routing,
provenance and revision handling. The legacy fake worker answer fixture now
records an explicit user-answer event and performs the real resolution operation;
it still asserts that the original task request is not duplicated.

The prior successful model verdict had itself embedded two complete source
reference arrays of 135,931 and 134,961 characters. Reinjecting this diagnostic
catalogue into the next review caused another large recursive context cost.
The audit view now keeps verdict, reason, remaining work and semantic projection
exact, while the full reference arrays and projection budget remain in the
integrity-checked shared recovery bundle. A dedicated red regression and exact
recovery assertion cover this path.

The superseded 2e9b79b live phase was deliberately terminated after 638.169
seconds, with no pass claimed; its partial full suite is likewise superseded.
The dependent status precondition failed before model interaction. The current
candidate resumes the same authorized worker after normal orphan reconciliation,
with unchanged file-hash/no-repeat-write/no-new-control oracles. Its full suite
and conditional final foreground status are running. The installed source is
still bd3042c until the final gates are satisfied.


## Verified worker closure and remaining foreground reduction defect

The immutable eebb7e8 candidate completes the full deterministic suite with
1,270 passing tests and three optional skips in 1,345.25 seconds. The saved
worker continuation passes its frozen acceptance oracle in 718.057 seconds:
completed, no open questions, unchanged result-file hashes, no new worker
controls, and only the questions tool called. Its first premature terminal action
is mechanically rejected; the subsequent resolution consumes local processed
control evidence at sequence 95 and closes the actual original question. Twenty
historical evidence projections still make this expensive. This phase closes
state; the authorized file effects were produced and independently read back in
the earlier dispatch phase, and were not repeated.

The ordinary foreground status probe still fails its 900-second deadline. Its
first progress sentence arrives at 300.065 seconds after redundant status and
question lookups; there is no final reply. The status tool returns its documented
full durable status history, about 18,000 characters. Exact projection reduces
that result to 479 characters and the snapshot to 391 characters. At sequence
1098 the request is only 398 tokens over budget, with 1,967 estimated tokens of
durable instructions remaining. The existing instruction projection can cover
that overflow, but action preparation tries raw history reconstruction first.
That rebuild expands the already reduced tool/history view into a 9,792-character
summary, increasing the measured request to 9,994 required tokens. Further
history-selection calls consume the deadline. This is an ordering defect, not
merely slow decoding. Canonical source history is preserved throughout.

An isolated red regression proves the unnecessary history-compaction call even
when the existing instruction projection subsequently fits. The candidate moves
that existing guarded recovery before history reconstruction. It neither removes
instructions nor replaces semantic reduction with keyword routing. Other budget
recovery remains available when instruction reduction cannot fit the request.
The same saved foreground conversation is the next live acceptance target.

Separately, completion projection target calculation had mixed a standalone
historical-token estimate with overflow measured on the whole serialized request,
forcing a 32-token target despite unused context room. Candidate 7dd3815 instead
measures the room after all required context using the same detailed/verdict-only
prompt builder, authority components and current questions. Two red regressions
and 58 related checks pass after correction in 118.18 seconds. A first prototype
omitted a still-used reduction-call bound variable; that NameError test run is
retained and was fixed before commit. This accounting correction is not by itself
a latency improvement claim.


## Process-loss recovery for already-consumed foreground receipts

The first instruction-order status retry did not reach inference: after 0.497
seconds it raised because the original receipt was processed but had no final
turn result. The previous implementation only supported retries before receipt
consumption or after completed response commit. The live timeout landed between
those boundaries. Reusing the same control ID is deliberate; creating a new ID
would conceal the interrupted-request recovery gap.

A deterministic reproduction executes one tool, injects connection failure on the
next action, recreates the service, and retries the same exact message/identity.
The old source rejects it. Candidate 32fbbb7 continues the original objective under
the existing cross-process execution lock, preserving original user messages and
committed evidence. It does not reissue the saved tool batch. Explicit resumed
control identities link each continuation in durable turn-start events; streaming
cursors and completed-answer replay follow these links. An unrelated later turn
cannot become this receipt's answer, and a canceled turn cannot be resumed by a
transport retry. Newer pending user corrections still reach the continuation.
The focused reproduction passes in 8.95 seconds with one user message, one
processed control, one actual tool call and successful subsequent answer replay.

The initial broader run passes 33 cases but fails the new cancellation negative:
the newly emitted cancellation event was missing from the event schema. This is
an undeployed candidate defect; the final candidate registers the event with its
required run identity and reason and reruns cancellation, streaming, concurrent
control and waiting-worker checks. All failed artifacts are retained.

The live continuation reuses the original receipt and cached exact tool-result
projection. Its first stream item is historical progress at sequence 1001, not a
newly generated low-latency answer. With corrected reduction ordering the action
prompt now fits at 6,575 input tokens without another history reconstruction.
Final model output, durable foreground-question reconciliation and unchanged-file
oracles remain separate live gates at this checkpoint.


## Final preparation refinements and acceptance boundaries

The recovery-order candidate passes 106 action/projection tests in 454.62 seconds.
After registering the cancellation event, the receipt-recovery candidate passes
34 streaming, control and waiting-worker tests in 96.49 seconds. Its immutable
combined full suite is running at 5c887ed. The broader eebb7e8 suite remains a
separate completed result: 1,270 passed and three optional skips.

The 32fbbb7 live status continuation fails its 900-second deadline. It makes
37 model requests: fourteen instruction projections, six tool-result anchors,
ten verbatim selections, five evidence projections and two actor calls. Thirty-six
return before termination. No new history compaction occurs, so the original
ordering defect is bypassed, but the overall response still does not complete.
The model attempts one question resolution with a tool-result citation at sequence
225. The validator correctly rejects that citation: the actual user message is at
sequence 153. The rejected draft is not delivered as successful progress. There
are no question resolutions, repeated file effects or new worker controls.

Inspection shows that canonical user-message source sequences already exist in
message metadata but are absent from rendered action context, while tool-result
sequences are visible. Candidate e413087 exposes those exact user-message
references for standard, lean, capability-selection and terminal-response prompts.
Four red regressions cover historical and current user sources. Missing metadata
and tool quotations do not acquire invented user references. An existing verbatim
boundary check rejects the first marker placement; f9a4446 moves metadata before
the unchanged Current user message heading. The test is not relaxed. The failed
prototype's remaining test process is explicitly superseded and retained.

The same live trace also exposes excessive early instruction-reduction attempts.
f9a4446 puts instruction recovery after available tool/snapshot views but before
raw history reconstruction. Fifteen focused checks pass in 41.95 seconds before
the final boundary adjustment; the wider unchanged prompt/action checks cover
that adjustment. No production source was changed between these candidates.

Candidate 364c869 measures instruction-projection room with the whole required
request and the projection header, using the required output minimum. It skips
model reduction when that remaining request already exceeds the budget. It also
tracks the target actually passed into the inner reducer: repeated outer targets
must not collapse to identical model requests after the generation allowance caps
them. An observed run issued three equivalent 232-token targets despite nominal
outer shrinkage. The deterministic red case repeats a nominal target of 1,168
after an effective target of 64; the corrected retry is smaller than 64. Two red
regressions now pass; sixteen focused tests pass in 47.04 seconds, the target
regression passes in 0.46 seconds, and 26 related instruction/completion tests pass
in 30.18 seconds. The first capacity prototype incorrectly retained optional desired
output reserve, consuming otherwise usable context; its failing check is retained
and the measurement now fixes only the required minimum reserve. The actual
model call still receives its normal configured output policy.

The next same-receipt continuation uses the combined 364c869 source and a longer
1,800-second observation deadline. This deliberate deadline change means any
later completion is not by itself a like-for-like latency improvement. It preserves
the frozen no-repeat-effect, no-new-control, actual completion and question-state
oracles. A separate queued fresh-conversation probe will copy the completed
worker's actual fixture after this run stops, preserve the previous foreground
history under another name in the copy, and invoke the ordinary entry point with
a new foreground conversation. That measures actual-state snapshot availability
without relabeling fresh context as long-history recovery. Model calls are
sequential between the two probes. The original fixture remains authoritative.


## Installed correction release and final full-suite boundary

The 5c887ed full run finishes with 1,276 passed, three optional skips and one
failure in `test_action_refreshes_snapshot_after_context_reduction` (1,680.50 s).
That unchanged test reproduces on 5c887ed and passes on 364c869 after the f9a4446
recovery-order correction. The whole final snapshot suite passes 21 cases in
82.42 s. The final source also has 134 prompt/action checks on f9a4446 and the
separately recorded capacity/target checks on 364c869. These are scoped results,
not an invented full-suite pass on 364c869.

364c869 is committed, pushed and installed on both Jetson and Thor. Each installed
package matches all 195 tracked package files; each installed smoke suite passes
two tests. Private rollback copies retain the prior bd3042c package. The actual
Jetson production configuration selects Thor BF16 at port 14833, not the old
failed local Jetson llama unit. Both Thor model units remain active with their
original process IDs and zero restarts. No model eviction, startup, resize or
restart was used. Exact certification and installation artifacts are retained in
`release-certification.json`, `installed-release-ready.json`, both host equality
reports and the final installed smoke logs.

## Long-history deadline versus fresh actual-state status

The combined 364c869 continuation reaches its extended 1,800-second deadline
without a final answer. It resumes the original receipt, makes 82 requests and
receives 81 responses: six tool anchors, eleven tool verbatim selections, six
evidence projections, two instruction projections, one actor, five history span
selections and fifty history verbatim selections. It commits no history reduction
and resolves no question. Its one actor repeats status/question lookups despite
the current snapshot. The new progress message arrives at 317.029 s; replayed
sequence 1001 is not new response latency. The exact breakdown is retained in
`final-budget-status-analysis.json`. This remains an acceptance failure, not a
successful result after increasing the deadline.

Two sequential fresh-conversation trials copy the real completed-worker fixture
and preserve its previous foreground conversation under another name in the copy.
They invoke the ordinary installed entry point with the same German status-only
request, unchanged result-file hashes and no new controls. This is actual-state
integration evidence; it neither substitutes synthetic state nor qualifies the
accumulated original foreground history and its remaining question.

| Installed backend | First progress | Final answer | Returned model calls | Result |
| --- | ---: | ---: | ---: | --- |
| Thor Q3, server capacity 2; SWAAG admits 1 | 70.777 s | 203.927 s | 4 | PASS, accurate completion, exact file values, no new work/effects |
| Production Thor BF16, 1 admitted sequence | No intermediate progress | 88.686 s | 2 | PASS, accurate completion and no new work/effects |

Q3 asks for redundant worker status. Its action calls take 64.139 and 63.572 s,
with 6,824 and 7,000 input tokens and first tokens at 45.671 and 40.764 s. A tool
anchor adds 17.208 s and completion review adds 37.209 s. Total recorded admission
wait is only 0.394 s. These timings distinguish client model-call time from SWAAG admission
queueing; server-internal queue/prefill phases are not separately exposed; they do not imply a hardware/quantization throughput ranking from two
different model trajectories. BF16 answers directly from the snapshot and then
undergoes completion review. Neither result meets a twenty-second end-to-end
conversational target.

## Bounded reuse and current-status lookup candidates

8037b1f reproduces an unnecessary second history reduction after the same task's
first successful reduction. The old source fails the new regression. The candidate
still compiles authoritative history first; only measured overflow permits reuse
of summaries with exact source/projection provenance and a matching objective
fingerprint. Request, ordered pending instructions, validation feedback, latest
original user-event identity, system/selected durable policy, tools and contract
participate in that fingerprint. Unknown legacy summaries and changed authority
reproject from exact sources. New observations and the current runtime snapshot
are rebuilt normally; a changed instruction selection while compiling the reused
view rejects that candidate. A later larger context restores full exact history.
Thirteen focused cases pass, and eight source/provenance/unchanged legacy tests
pass in 51.90 s. A replay correction also attaches projection provenance at the
actual nonzero replacement index, rather than only at index zero.

The real long run above commits no successful history reduction, so this reuse
correction alone is not evidence that its initial large reconstruction will finish.
It removes repeat work after a successful reduction; the hard first-reduction and
semantic-selection limitations remain explicit.

a47fba4 changes `agent_status_lookup` to return the exact latest status plus source
count/range/hash and an explicit full-history recovery instruction. The caller can
request every exact status using `include_history=true`; no old event is deleted,
truncated or semantically declared irrelevant. Omitted/null/false selects current
status; invalid types fail. The tool guidance first directs the model to the
already-supplied snapshot when sufficient. Twenty-nine status/control/discovery
checks pass in 19.03 s. On the same saved actual worker, default JSON shrinks from
19,281 to 2,022 characters; all 25 prior statuses remain available and the canonical
history hash is unchanged. Character counts are not a model-latency result.

The earlier partial 8037b1f full run is explicitly superseded by a full run on the
combined immutable a47fba4 candidate; its partial output is retained, not scored as
passing. The first live candidate comparison exposes a fixture issue: copied
worker history still names original marker paths while read roots allow only the
clone. Q3 chooses direct file verification and is correctly denied. A subsequent
paired protocol allows reads of the original fixture, restricts writes to the two
clone marker paths, checks both original and clone hashes, and rejects any attempted
write/control. Results from those protocols must remain separate.


## Current-status comparison outcomes and preservation batching

The first a47fba4 Q3 trial returns an honest worker-reported answer after 247.373 s,
including the failed direct-read limitation; it makes no writes or controls.
Its extra timestamp is an audio-style defect, separate from factual/effect checks.
The same candidate on the actual production BF16 configuration returns the same
accurate concise answer as baseline in 87.948 s, compared with 88.686 s before.
Those two single observations do not establish a speed difference.

With original marker paths readable and writes restricted to the clone, Q3
baseline and candidate both pass the frozen state/effect checks. Final latency is
199.464 versus 212.259 s; first progress is 69.951 versus 70.124 s. Both actually
invoke status lookup. Its anchor call drops from 17.215 to 6.905 s, but completion
review rises from 36.116 to 58.653 s. Thus the smaller status payload does not
improve overall observed latency in this pair. A second, explicitly requested
status-lookup pair also passes: 201.013 versus 212.192 s, one lookup each, no direct
file reads, writes or new controls. Its anchor improves from 16.897 to 6.940 s
while completion review rises from 36.925 to 62.271 s. No repeated trial was used
to replace these slower candidate outcomes. Exact responses and phase timing are
in `fresh-status-comparison.json` and each fixture's `timing-analysis.json`.
Question-absence wording in fresh conversations covers the worker/new conversation;
it cannot close or qualify the retained old foreground question. Both explicit
lookup answers also render a numeric percentage, so factual checks are not an
all-pass audio-style certification.

The canonical Infra performance reports were checked again. Their archived Q3
row contains one decode observation around 10 tok/s and lacks TTFT/prefill data;
BF16 has no instrumented timing row. There is no matching phase row for these
current 8K vLLM deployments. Current stream-derived Q3 decoding is around 10 tok/s,
but the historical configuration/context and incomplete timing fields do not
support a direct latency ranking. Current client timing retains first token,
request completion, token counts, SWAAG queue time and remaining wall time.

Inspection of the 50 history-preservation calls identifies fixed batches of 16
source units despite ample remaining context. Candidate e99d983 begins with the
complete exact candidate set and splits only after measured context overflow.
It also measures desired response space for all selectable IDs, avoiding the old
fixed allowance when many IDs are needed. A second red case shows that treating
this desired maximum as a hard minimum needlessly splits otherwise fitting input;
66d3a65 keeps it soft and lets observed output exhaustion raise the minimum through
the existing recovery path. Exact units, semantic preservation instructions,
source mapping, failure-preserves-source behavior and cancellation remain intact.
The focused final batch/compiler checks pass 18 tests in 35.05 s. Broader history
and full-baseline results remain separately recorded.

A predefined 64-unit live Q3 preservation comparison includes eight authoritative
constraints/facts across the source, routine progress and explicitly untrusted
contradictory decoys. Baseline and 66d3a65 both retain all eight required lines,
select no extra lines, and leave canonical source text unchanged. Baseline takes
four calls and 47.573 s; the candidate takes one call and 24.324 s. This is a
measured improvement for that bounded preservation operation, not proof that a
whole long conversation is now responsive. A gated retry of the original
foreground receipt follows only after both preservation oracles pass, with the
original no-repeat-effect/no-new-control/question-closure checks unchanged and a
900-second deadline. Its accumulated history is not reset.

Harness mistakes are retained separately: one nonexistent test-path invocation,
a missing import during test construction, an incorrectly quoted probe description,
and an empty enabled-tool list rejected before inference. None is a model score
or a passing software run. The first batching full-source candidate's hard-output
reservation regression and its correction are also retained rather than hidden.


The immutable a47fba4 combined full suite completes with **1,307 passed and three
optional skips in 1,576.87 seconds**. The e99d983 batching candidate passes 118
broader action/history/provenance/long-horizon tests in 592.92 seconds; 66d3a65's
soft-output refinement passes the separately recorded 18 batch/compiler/output
recovery checks. An exact-final-source full run also starts on Thor in an isolated
66d3a65 worktree. Its retained pytest configuration identity proves the candidate
root and source Python path; the production checkout and installed package are
not substituted for the test source.

## Final cross-host regression and interrupted-history recovery

The exact 66d3a65 full run on Thor finishes with 1,310 passed, three optional
skips and one failure in `test_cli_budget_demo` (232.87 seconds). Repeating that
test from the isolated worktree still fails, disproving the initial working-directory
configuration hypothesis: neither checkout has a local config. Directly replaying
the patched CLI exposes the actual error, `Model cache byte capacity reached;
existing replay evidence is unchanged`. This test was using the host replay cache.
33fd178 supplies its existing isolated `make_config` fixture to the CLI, retaining
the rendering assertions and leaving the real cache and its capacity policy intact.
All seven CLI tests then pass in 6.38 seconds. A full 33fd178 run follows from the
isolated candidate working directory; the earlier failed run is retained.

The 66d3a65 original-receipt trial reaches its 900-second deadline. It returns
32 model calls, commits two history reductions and reaches an actor call plus
read-only file verification, but produces no new user-facing reply before the
deadline. The saved reductions replace 11,357 tokens in the first span and retain
exact source provenance. File hashes and worker control receipts remain unchanged;
the foreground approval question remains open. These are partial recovery results,
not a completed conversation. `history-batch-status-analysis.json` retains the
full operation counts and phase times. The 900-second run cannot be compared as a
paired speed measurement with the prior 1,800-second run on different accumulated
history.

A further continuation keeps the same original receipt, message, fixture and
66d3a65 source, testing recovery after the persisted reductions. Its initial
compilation explicitly records `reused_goal_history_projection`, with nine derived
messages for eighteen authoritative messages. Current context is rebuilt; it still
requires additional budget recovery. No duplicate user instruction is inserted.
The final outcome is recorded separately rather than conflated with the failed
first deadline.

The complete final 33fd178 source suite on Thor passes **1,311 tests with three optional skips in 221.63 seconds**. This includes the unchanged production code from 66d3a65 plus the isolated CLI regression test. The old cache is preserved, and no production model lifecycle operation is performed.
