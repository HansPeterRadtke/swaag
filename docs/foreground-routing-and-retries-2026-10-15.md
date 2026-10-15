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

## Corrected installed acceptance

The source correction is ready for the serial installed rerun. Results and final
source/install equality will be recorded here after observation. No universal
model-reliability or fixed twenty-second dispatch claim is made.

