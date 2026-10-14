# Guideline corrections — October 14, 2026

This is the implementation follow-up to `guideline-comparison-2026-10-14.md`.
The comparison records the installed baseline, not the corrected deployment.
Work is in progress. Source changes, test evidence and deployed acceptance are
separate states; this document does not certify every guideline as complete.

## Changes and invariants

- Ordinary finalized user messages now enter the full action controller directly.
  The separate routing gate and English claim-detection regular expressions no
  longer decide whether ordinary messages reach orchestration. The experimental
  gate remains available for comparative studies. This is a correctness baseline,
  not a claim that full planning is the fastest possible conversational design.
- The trusted orchestrator and control instructions remain active when the prompt
  instruction **editing tool** is disabled. Previously disabling that capability
  accidentally removed instruction enforcement from actual model requests.
- Every action preparation receives a newly observed global state snapshot,
  including workers, terminal results, plans, questions and supervision. The
  exact observation is retained in history. A changed task fingerprint rejects
  stale output before effects. New user controls use the existing outer ordered
  authority fence. The decomposed single-responsibility path has the same fence.
  Sources remain explicitly individually consistent, not a fictitious global
  atomic transaction. Semantic projections retain exact recovery references.
- Foreground JSONL and CLI delivery now supports accepted/progress/completed
  events with durable control IDs and event cursors. The execution job is
  independent of the speech/transport consumer. Reconnection replays committed
  evidence rather than rerunning accepted work. Final-only clients remain valid.
  Dispatch text is emitted after successful tool receipts; failed/superseded
  drafts are suppressed. A receipt does not prove arbitrary claims in draft text.
  A long synchronous foreground tool can delay this progress; substantial work
  should use background worker dispatch. No Android audio implementation changed.
- Worker control accepts an exact worker ID as well as a session ID/name, resolving
  the correct owning runtime. Server-derived original user authority accompanies
  the model's routing note in the durable control record. SQLite migration,
  idempotent first-payload preservation and replay retain the original wording.
  Workers receive original instructions in order with routing notes subordinate.
- Queue acceptance, worker acknowledgement and completed effects are distinguished
  in the control policy. A queued control permits an honest pending reply; the
  foreground must not repeatedly send the same control while waiting for a queued
  worker. This is necessary but model adherence still requires live acceptance.
- Shell/Python/pip scratch output is streamed with bounded memory into exact,
  integrity-checked artifacts, including CRLF and invalid UTF-8 bytes. Output and
  time limits stop the owned command and explicitly qualify capture completeness.
  Failed publication retains raw recovery evidence. Text/directory reads have
  continuation cursors; writes support revision checks and atomic replacement.
- Shared virtual-environment initialization/mutation uses cross-process locks and
  a durable readiness marker. Normal commands hold a shared version lock and see
  the virtual environment read-only; package installation is exclusive. Sandbox
  commands cannot write catalogue/lock/ready files through a broad root mount.
- `tool_forge` and `diane` provide versioned shared helper/experience catalogues,
  compare-and-swap publication, immutable pinned revisions, rollback, retirement,
  scoped discovery and compact prompt pointers. Forge promotion requires an exact
  tested file hash and retained validation artifact. Evidence remains agent
  reported and must be interpreted; it does not automatically trust executable
  content. Helpers do not expand tool permissions. Payload/revision admission is
  bounded, but this is not a complete OS disk/RAM/process quota or archival policy.
- Configuration now describes the previously generic parameters, identifies
  sensitive permission controls, reports checked ranges/units, and exposes
  foreground/status inference priority and shared-model preemption policy.
  Metadata coverage is not certification of every conceivable combination.
- Explicit `model.thinking_mode` applies consistently to generation and exact
  template/token counting on supported llama.cpp/vLLM adapters. The default
  preserves server behavior. Normal action instructions forbid invented exact
  word-count constraints; live traces showed those causing unnecessary repair
  calls on ordinary questions.

## Evidence location and verification status

Raw evidence: `/data/var/swaag-benchmarks/guideline-corrections-20261014`.
Long live runs use frozen source trees with per-file hashes. Never combine their
latencies as if prompt versions, load and hardware were identical.

The first broad diagnostic run collected 1,197 tests: 1,190 passed, three optional
skips and four failures. It loaded an earlier source version before the ordinary
routing/progress changes and is not final certification. The failures were:

1. An old status fixture omitted its required operation-session reference.
2. Two logging fixtures read only the current file even though expanded metadata
   rotated startup evidence into retained backups; assertions now inspect all
   retained files and still verify exact ordering/redaction.
3. A test required instruction enforcement to disappear with the editing tool;
   the corrected contract requires enforcement to remain active.

The focused corrected routing/instruction/logging/snapshot run passed 71 tests.
Earlier focused evidence includes 99 status/action tests, 89 workspace/tools
checks, 56 shared-knowledge/config/capture checks, 79 model/context/config checks,
and seven native thinking/template tests. Later authority/stream and full-suite
results must be recorded before deployment certification.

## Live investigations and rejected assumptions

The earlier compact-reader and semantic-review studies produced syntactically
valid output while dropping mixed status/prohibition messages. Exact forwarding
in the messages actually forwarded does not establish routing correctness.
Those reviewers were not adopted as an authority gate.

Full-action experiments exposed missing prompt-policy enforcement, which was
fixed in the contributor. Some earlier synthetic worker objectives already
withheld deployment authorization; absence of another hold in those cases is not
unambiguous evidence of a lost restriction. The later objective explicitly
allowed deployment immediately after passing tests, making the new restriction
observable. Earlier variants disabled completion evaluation for timing; later
variants enabled it. Their results must not be conflated.

A BF16 run actually queued a worker restriction and finished with a verified
answer. Another French run addressed a worker ID through a session-only control
tool and failed. A nine-billion run first missed the hold, then queued it after
completion checks, then repeatedly demanded acknowledgement from a deliberately
queued worker. Those traces are retained. The experiment clients were stopped
after capturing the repeated loop, without stopping model services. They do not
qualify those versions as reliable foreground controllers.

The revised receipt matrix uses unmodified ordinary foreground dispatch, exact
user-authority provenance, corrected target resolution, completion evaluation,
and a per-case 240-second external experimental deadline. It contains English
and French mixed restrictions, quotation-only non-action, status and a real
calculator invocation. A deadline is a failed experiment, not a successful reply.
Results are retained under `unified-receipts-v5-*` and the supervisor JSONL.

## Guarded small-model admission

The canonical nine-billion unit could not be admitted with its existing prompt
cache: estimated demand plus the four-GiB reserve exceeded available memory.
No existing workload was stopped, evicted or resized. A separate experimental
unit with its own port, smaller context/cache and one-hour lifetime was prepared
in Infra and started through the canonical memory guard. The guard estimated
10.2 GiB; observed available memory after load was about 12 GiB. Existing BF16
and Q3 service PIDs/restart counters were unchanged. Exact unit configuration,
service observations and experiment source are retained. The experimental unit
must be stopped after the study, or its bounded lifetime will expire.

## Remaining acceptance work

Finish the revised live matrix and investigate semantic failures. Verify a real
independent communication backend with the specifically named independent-model
benchmark; supplying an assistant URL to the preemption benchmark is insufficient.
Complete relevant resource-policy and cancellation tests, final coherent full
suite, source/installed equality and installed smoke tests. Reconcile the Infra
command wrapper and Thor endpoint/dependency drift without enabling an unwanted
boot daemon. Preserve unrelated host configuration changes and concurrent Infra
commits. Update this document and the comparison with actual deployed results.

A general learned-memory quality claim, arbitrary long-horizon reliability,
physical voice-device acceptance, all external perception providers, and full
OS enforcement of shared-workspace resources remain separate acceptance scopes.


## Additional evidence during correction

The completion verifier now receives a fresh authoritative whole-system snapshot
including pending worker controls, retains its source reference, and invalidates
completion if relevant state changes during its call. Previously action context
was current while verifier evidence could lag. A dedicated transition test passes;
the surrounding completion/control/stream group passed 46 tests, with one newly
added fixture corrected to obey the actual closed response schema and then passing.
The single-responsibility path now records rejected stale/invalid actions as well
as rejecting them. Three focused snapshot/preemption-policy checks passed.

The first five revised nine-billion cases produced two completed interactions:
status and a real calculator result (323), taking about 62 and 70 seconds. English
hold, French hold and quotation-only reached the 240-second experimental deadline.
A separate six-case verdict probe confirmed that native boolean constraints can
emit true and false. Reordering decision/reason fields did not solve instruction
semantics: the reason-first missed-hold probe falsely approved an undelivered
restriction. No schema-order workaround was adopted. Original traces remain the
basis for distinguishing correct actions, unjustified claims and timeouts.

After 16:41:54 UTC the fresh-state/decomposed nine-billion probe overlaps the BF16
and later Q3 matrix. This explicitly measures loaded behavior and is not an idle
model-speed comparison. The frozen-source manifests and concurrency note identify
that change. Deadline failures under this load do not alone prove semantic failure.

A host resource-enforcement doability probe found that Jetson's user systemd
scope did not acquire separate memory/pids controllers. An in-sandbox address-space
limit was enforced, but a low process limit counted the host user's processes and
prevented even the first child. These are not acceptable aggregate per-job resource
limits; neither was silently enabled as if it solved workspace resource isolation.
Network/GPU device isolation, command/output bounds, catalogue admission and safe
model-memory admission are distinct from this remaining OS integration requirement.

Verified SQLite rollback copies and installed-package archives are prepared before
schema migration/deployment. The Jetson backup contains eight checked databases.
The optional communication daemon remains deliberately inactive; package repair
must not convert that existing lifecycle choice into automatic boot activation.


## Final source regression and package preparation

The corrected broad diagnostic run passed 1,200 tests with three skips and two
fixture failures: a canceled-ingress test raced durable commit against its later
capacity-release finally block, and the SQLite inventory still expected history
schema version one. The ingress fixture now explicitly holds the receipt method
after durable commit, proves capacity is retained, releases it, and waits for its
return. The inventory now expects the actual version-two authority migration.
The corrected control/migration/benchmark group passed 47 tests. A final broad
run is still being collected; do not describe the earlier run as fully green.

The live decomposed nine-billion probe exposed two additional implementation
failures. A harness-generated newline in the completion-verdict instruction was
stripped by the native Qwen chat template, violating exact accounting. Removing
only that generated trailing newline fixed the native reproduction; original
user text and the accounting guard remain exact. Seven native/verdict checks
passed. The decomposed action paths also bypassed the normal measured context
recovery, and a real tool error emitted an unregistered event. Those paths now
share ordinary action preparation and semantic overflow recovery, preserving
full authoritative history and recovery references. The missing event is
registered. All 111 action/snapshot checks passed, including real failed-tool
recovery and overflow/recovery for capability, tool-call and terminal decisions.
A source-wide literal event-registration scan found no remaining unknown types.

The current wheel was built offline and passed both package smoke tests in an
isolated environment with declared dependencies installed. The first temporary
venv mistakenly assumed it inherited another venv's dependencies and failed to
import requests; that environment setup error was corrected without changing
package dependency declarations. Production installation is a separate gate.

## Completed receipt matrix and limits

The frozen V5 matrix completed all 15 supervised attempts. Seven returned a final
answer within the 240-second deadline, and those seven met the finite routing
expectation. Twelve of fifteen had the expected presence/absence of a worker
control, regardless of terminal reply. These are distinct measurements: a
correctly queued control followed by a deadline is not a completed interaction.
Nine-billion English hold omitted delivery; nine-billion and BF16 quotation-only
cases queued an unwanted worker control. Exact original user authority remained
separate from model notes, but that alone does not qualify semantic routing.
The Q3 French case delivered a hold and replied, while the English case repeated
controls and timed out. Status and calculator cases succeeded on nine-billion
and BF16; Q3 status/quotation/French completed while calculator timed out under
load. The complete scored JSON and original SQLite histories retain the details.

The later native-fixed decomposed nine-billion English hold returned a status
answer with zero worker controls despite approving completion. Its conditional
case exposed the missing context recovery; its quotation case exposed the
missing event registration. These mechanical defects are now fixed and tested,
but the semantic missed hold remains negative model-qualification evidence.
The smaller/decomposed controller is not adopted as the production default.

The owned experimental nine-billion service was stopped after the study. Thor's
existing BF16 and Q3 services retained their process IDs and zero restart counts.
The exact guarded unit is committed in Infra, inactive and not boot-enabled.
The canonical launch wrapper and host profiles select the observed BF16 VPN
endpoint with an 8192-token context and explicit thinking disabled, without
changing tool permissions. The optional communication profile no longer points
to the unavailable coder endpoint or implicitly starts model services. No daemon was enabled or started by this correction. Jetson remains disabled;
Thor is inactive with its pre-existing boot enablement unchanged. Its lifecycle
is an explicit choice.


## Live worker consumption acceptance

An actual ordinary BF16 foreground interaction queued the changed instruction
exactly once, preserving its full original wording and source reference. The
receipt appeared after about 134 seconds; the verified foreground response after
339 seconds. The first model token arrived after 20 seconds, which is not the
time of a validated executable tool call. The first request used 6,749 prompt
tokens and 215 generated tokens, taking 131 seconds under the concurrent Q3 load.
The subsequent action and completion evaluation took 120 and 79 seconds.

The real worker then consumed the durable control, read the fixture README,
created the still-authorized tests-passed.txt containing exactly PASS, and did
not create the now-prohibited deployed.txt. Its control became processed and
its durable status became completed. Only harmless isolated fixture files were
involved. The original harness called the active status running rather than
working and saved an early observation; an independent observer waited for the
actual terminal state and verified the final files/control without rerunning
the instruction. Both records and the corrected harness are retained. The final
observation passes. This proves this end-to-end case, not every model utterance.
The model's receipt/result prose also contained unnecessary technical identifiers;
functional authority handling succeeded while presentation quality remains a
separate model-adherence limitation.


## Full-suite result and independent backend correction

The final broad run passed 1,205 tests with three optional skips in 1,385 seconds.
It loaded source before the later decomposed recovery/event-registration delta;
the subsequent 111 passing action/snapshot tests cover that delta. Coverage must
be reported with this distinction rather than claiming one full run loaded all
later source changes.

The actual capability_communication_independent_model task used BF16 on port
14833 and a separate Q3 runtime on 14834. The uncached main turn completed in 86
seconds without preemption/replay. Its status call took 59 seconds and overlapped
main generation. The benchmark's mechanical oracle passed, but manual evidence
review found the answer stale: it said the main agent was waiting for capacity
although inference had begun immediately after the snapshot. Its empty semantic
quality oracle does not certify that answer. Original evidence is retained under
independent-acceptance.

The status freshness fingerprint had deliberately excluded scheduling telemetry
to avoid rejecting shared-model preemption's own changes. Independent readers now
also fence the main operation's phase, kind and call identity, while still ignoring
clocks/token counters/streaming substates. Shared-model preemption and escalation
retain their different handling. A queued-to-inference regression and the live
independent rerun verify this correction; record their final outcomes below.


The independent execution-phase change passed 59 status/admission/preemption
regressions. Its first live rerun correctly withheld both stale drafts when the
main work advanced during each attempt; it reported inability to verify a current
answer instead of publishing the old queued claim. This preserves bounded
recovery rather than relaxing the freshness fence for a green benchmark.
The benchmark also began its status probe when a preemption handle was merely
registered, before backend dispatch. The independent probe now waits for a
matching model_request_sent event; shared preemption's distinct observation is
unchanged. This corrects experiment construction, not application routing. The
additional focused probe and fresh live result are retained separately.


## Installed release verification

The corrections are committed and pushed. Both Jetson and Thor now run the
current installed package; all 195 package files match source byte-for-byte on
each host. The canonical offline installer completed on both. History migrated
to schema version two with identical counts for every existing table and passing
SQLite integrity checks: one production history on Jetson and two on Thor.
Private pre-deployment database/package/configuration backups remain available.

Both real installed-package smoke tests passed on each host. The lean production
environment did not include the optional pytest dependency; it was installed for
these checks. Thor required its normal package index because that optional test
package was not cached. This changes no runtime dependency declaration.

The canonical Jetson command used the installed host profile and a fresh isolated
session, invoked the real calculator once for 23 * 29, observed 667 and returned
that result. Action inference took 36 and 39 seconds, with a 14-second completion
check. Client replay caching was disabled; server-side prefix caches were not
purged, so these are warm-service observations, not cold-load timings.

Four targeted benchmark-observation tests passed. The corrected independent
benchmark completed with a current answer after 58 seconds while the main action
continued. Evidence proves the assistant used Q3 on 14834 and the main action
used BF16 on 14833: distinct session backend URLs, actual request model aliases,
model-request/response timestamps and zero preemption/replay events. The main
action ran 17:23:12–17:24:20 UTC; assistant generation ran 17:23:14–17:24:10.
The main turn's later completion check finished at 17:24:43. This verifies genuine
backend overlap and this status answer, not semantic reliability for arbitrary
instructions. Independent ordinary-controller adoption remains rejected.

The Infra runtime/voice profiles and optional Thor unit were installed from their
committed source and verified exactly. The Thor unit remains inactive and retains
its pre-existing enabled state; Jetson communication remains disabled/inactive.
No model service was restarted. The Jetson host configuration unrelated to this
work remains byte-identical to the original recorded hash. Concurrent Infra
changes were merged normally and both Infra checkouts are synchronized.


## Adversarial installed-worker correction

A further installed BF16 worker probe reproduced the quotation-routing risk with
an exact original user instruction to leave the worker unchanged and an
intentionally contradictory model routing note. The worker incorrectly withheld
the second authorized file and repeated the first write. The exact original was
stored, but the runtime combined the generated note and original wording in one
section labelled as authoritative user interventions. Labelling the note
subordinate did not prevent its semantic promotion. The owned experiment client
was stopped after this concrete failure was retained; model servers were untouched.

The runtime now places only the original user text in the authoritative action
and completion channels when provenance is available. The generated routing note
remains exact in the durable control record for audit/recovery; it is not injected
as user authority. Legacy/direct controls without attached original authority
retain their prior behavior. A focused regression checks both action/verification
authority and durable note preservation. The installed adversarial rerun below
must verify both originally authorized files, not merely absence of an error.
