# History reduction levels and current small-model qualification — 2026-10-16

## Reproduced history availability failure

The installed BF16 original-receipt trial at source 07f8b5d/package 275dce1 ends
with BudgetExceededError after 1,151.63 seconds, before any new ordinary action.
Its last action compilation needs 19,363 tokens against the deployed 8,192 limit.
The thirteen returned model calls take 1,082.65 seconds. Worker inventory, run
counts, authoritative controls and file hashes remain unchanged; the foreground
question remains open. The diagnostic allowance is twelve reductions, not the
production default four.

The old selector explicitly tells the model to protect all windows containing
task facts and does not explain the supported important/ordinary paths. All four
real windows receive protect/unresolved_work. The correction explains that those
paths use summaries with selected exact source passages, while protect prevents
reduction and compressible is reserved for verified redundant progress. Full
literal retention requirements still select protect. No schema, history, guard,
source authority, output reserve, or production context limit is relaxed. Whole
windows and oversized fragments share one instruction definition.

The frozen precursor replay passes eight candidate cases. The final source
removes the misleading word independently: a normal summary selects its own
verbatim_spans field; this is not an independent semantic verifier. The exact
final prompt passes eleven live BF16 classification checks (eight windows plus
three fragment cases), including exact complete retention, unresolved restrictions
and supersession, redundant progress, and untrusted source instructions. These
are path-selection checks, not proof that every resulting summary is faithful.

Immutable source fffc2d6 passes 1,349 tests with three optional skips in 236.46
seconds on Thor. Focused history/action checks pass 113 cases in 525.93 seconds
on Jetson. An initial accidental test invocation used baseline main rather than
the candidate; its 111 passes in 518.25 seconds are retained separately and are
not counted as a candidate gate. The source is pushed on fix/semantic-selector-levels.
The original-receipt continuation reaches its 1,800-second deadline at sequence
6963, starting at 6504. All four windows now select important and the run enters
hierarchical summarization, but no projection or new actor action completes.
Worker inventory/run counts, controls, question state and file hashes remain
unchanged. No fresh user instruction or manual question closure is injected.
The nine-billion GPU load and control matrix overlap this trial, so its durations
are not an isolated latency comparison.

## Rejected tool-selector prompt

A separately frozen six-case, two-prompt Q3 study tests a more precise relevance
instruction. Both prompts truncate on the real 243-unit old status record under
the captured request allowance, after 228.02 and 352.01 seconds respectively. The
candidate improves one small checksum-state preservation case, but does not fix
the observed large-record failure and is not adopted. Generation competes with
other Thor requests; these are descriptive durations, not an isolated speed test.

## Current nine-billion-model setup and limits

Earlier October twelfth small-model and independent-backend results already exist;
this follow-up tests the corrected ordinary controller rather than repeating that
removed immediate gate. The installed nine-billion service is inactive. Its current
default eight-GiB host prompt cache causes guarded admission refusal: 18.1 GiB
estimated footprint plus four-GiB reserve exceeds 18.6 GiB available. A separate
owned benchmark process explicitly disables host prompt caching, retains a 32,768
context, Q5_K_M weights, q8_0 KV, Flash Attention and one slot, and is admitted at
10.1 GiB estimated footprint. No service or Infra configuration changes are made.

The first owned launch falls back to CPU because the command runner omits the
account's normal video group. It is stopped cleanly without any inference
measurement. Authorized privileged execution through runuser restores the
account's existing groups without changing account/device permissions. The GPU
launch becomes ready after 120.25 seconds; NVIDIA accounting positively identifies
the owned process with 6,242 MiB. A one-second memory monitor stops only this owned
process at five GiB available, before Infra's four-GiB reserve. Existing model
services retain their original process identities. This temporary model must be
cleaned up after the probes.

The current-controller matrix retains five existing exact-authority/scope oracles,
with isolated created-but-not-started workers and real durable control queues.
The mixed prohibition/status case times out at 300 seconds. The quotation case
passes the queue/no-effects oracle but fails manual answer review: it translates
the surrounding meta-instruction as part of the quoted sentence. The scoped case
queues controls to both Alpha and Beta despite the instruction to leave Beta
unchanged, then repeats progress. Structural/queue success is not relabeled as
full semantic success. Both the German restriction and quoted-delivery cases queue one correct Alpha
control with exact original authority, but each reaches its 300-second deadline
without completing the reply. The mixed case fails to enqueue its restriction.
Repeated scoped actions reuse the same control IDs: idempotency works, although
target selection fails. These queues are observed in the authoritative inbox,
not inferred from absence of worker conversation events.

Two isolated real-shell tests pass. The successful command writes exactly the
expected marker and the expected-failure command reports exit seven and the
observed diagnostic without retrying. Each first action combines user text with
the real shell call. First progress takes 18.865 and 17.661 seconds respectively;
completed replies take about 43 seconds each. These tests start after the BF16
trial has stopped, despite the original manifest anticipating overlap. They are
not evidence of nineteen-second command latency under BF16 background load.

The probes inherited the compatible transport and vLLM provider settings while
using llama.cpp. Their observed effects remain real, but context-accounting
qualification must be repeated with the correct adapter. The subsequent long
status trial exposes the [tokenizer admission defect](tokenizer-admission-correction-2026-10-16.md).
No small-model orchestrator promotion follows from these observations.

## Evidence

All raw manifests, requests, outputs, source snapshots and timings are retained
under /data/var/swaag-benchmarks/foreground-routing-20261015. Relevant directories
are history-selector-levels-study, history-selector-final-study,
selector-relevance-study, candidate-history-levels-long-status-bf16,
nine-billion-current-controls and nine-billion-uncached-gpu-runtime.
Full-suite evidence is mirrored under thor-evidence. The combined release, including the subsequent tokenizer and native-boundary
corrections, is installed on both hosts after 1,378 full-suite passes and three
optional skips, exact 195-file package equality and two installed smoke passes
per host. The separate installed long-history diagnostic and temporary-runtime
cleanup are tracked in the tokenizer study.
