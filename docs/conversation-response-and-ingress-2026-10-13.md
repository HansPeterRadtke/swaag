# Conversation response cost and finalized-input ingress — 2026-10-13

## Outcome and scope

This continues [the control/freshness study](conversation-control-and-freshness-2026-10-13.md) from source commit `8976970`. It completes another 135 synthetic response trials, a real TCP cancellation trajectory, an independent compact-reader trajectory, and a replay profile of the production context/inference machinery. No tested conversational model receives production authority to discard, apply, or route user instructions.

The verified implementation corrections are narrower than a new architecture: receive finalized ordinary protocol messages before both request admission and the shared execution pool; keep accepted receipt jobs bounded and durable across transport cancellation; close persistent clients during service shutdown; supply the orchestrator's own active objective and outstanding user receipts in the current-state snapshot. The single-writer planner lock remains.

Evidence is retained on Jetson under `/data/var/swaag-benchmarks/conversation-response-study-20261013`. The Thor model service journal and loaded service configuration are under the same root on Thor. Exact requests, streamed responses, native timings, manifests, scripts, canonical histories, failed reproductions and later verification are preserved. `analyze_results.py` rebuilds `analysis.json`; the final integrity index identifies retained files.

## Models, admission, and comparability

The Qwen3.5-9B Q5 service was initially inactive. The canonical guarded start admitted it with 18.3 GiB available, a 10.1 GiB estimate, a 4 GiB reserve and 4.2 GiB remaining headroom. It used Thor llama.cpp, context 32,768, one slot, q8_0 K/V, Flash Attention, batch 1,024, microbatch 256 and fourteen threads. Its observed build was `b9842-6f4f53f2b`. It was stopped only after verifying its owned process and idle slot.

The existing Qwen2.5-14B-Instruct Q4_K_M weights were then admitted into an isolated temporary Thor service. The guard observed 18.5 GiB available, a 12.0 GiB estimate, the same 4 GiB reserve and 2.5 GiB headroom. This service used the same observed llama.cpp build, context 8,192, one slot, q8_0 K/V, Flash Attention, batch 1,024, microbatch 256, fourteen threads and the canonical CUDA MPS environment. Sampled MemAvailable was about 9.1 GiB while loaded. This establishes operation for the exercised requests, not full-context peak-memory certification.

The existing Qwen3.8-27B BF16 vLLM endpoint provided a larger-model comparison. It remained at one admitted sequence and context 8,192. Its props endpoint is unsupported; the saved props response is a 404, not capacity evidence. The already-running Q3 vLLM endpoint with two server sequences was the planner in the live trajectories; SWAAG's chat adapter truthfully recorded its conservative local capacity of one.

SmolVLM2-2.2B-Instruct Q4_K_M ran text-only on Jetson through an admitted temporary service, using the pinned SM72 build `b10535-8a832e4bf`, context 8,192, q8_0 K/V, Flash Attention, six threads and one slot. Admission observed about 18.1 GiB available and estimated 3.9 GiB with the 4 GiB reserve. The service's process had CUDA mappings and GPU device handles; startup logging did not expose a per-layer offload count, so no complete offload or native-context memory qualification is inferred from the command flags. No vision projector or multimodal capability was exercised.

All temporary services created or activated for this study were stopped after their slots were verified idle. The two pre-existing Thor model services retained their original processes and zero restarts. No production service was evicted, resized, restarted or substituted to admit a candidate. The unrelated Infra host configuration edit was retained. Jetson's ordinary production llama.cpp service remained inactive; these trials do not repair its existing admission/configuration situation.

The authoritative Infra methodology and phase reports do not contain a matching controlled comparison for these exact host, model, quantization, build, prompt, context, cache and load combinations. Their mixed-phase aggregate speeds are not a valid baseline for declaring a regression here. The archived nine-billion-parameter Q8 figures are a different quantization and workload; the prior twenty-seven-billion-parameter Jetson prompt-processing measurements are different hardware and model scale. No universal performance or quality ranking is claimed. No new Infra quality score is published without that campaign's template/parser gates.

## Synthetic response experiments

The first matrix randomized twelve cases across three prompt/output contracts with seed 73, temperature zero and a 192-token output cap. Its state distinguished active tests from a previous revision's successful tests, completed lint, unauthorized/unstarted deployment, failures, questions and missing history. The production variant used the actual extracted routing instruction. The other variants were experimental read-only replies, with receipt stipulated synthetically and no action dispatch.

| Qwen3.5-9B Q5 variant | Cases | Median total | Median first content | Median output tokens |
|---|---:|---:|---:|---:|
| Actual production routing prompt/schema | 12 | 4.579 s | 0.851 s | 113 |
| Compact read-only JSON | 12 | 2.191 s | 0.861 s | 40.5 |
| Compact read-only plain text | 12 | 1.672 s | 0.551 s | 35 |

First content means the first nonempty streamed content token, including JSON framing; it is not time to a validated, speakable answer. The shorter replies mainly reduce decoding, not just prefill. Median native decoding was 3.864 s for the production contract, 1.530 s for compact JSON and 1.065 s for plain text. Native prefix caching remained enabled; these are fresh inference calls, not cold-prefix measurements or saved-response replays.

A stricter read-only prompt explicitly distinguished unknown history, observed execution, negation, quotation, all outstanding questions, authorization and audio style. Twelve new held-out cases were fixed before execution and combined with the original twelve, randomized with seed 79. The exact same twenty-four payloads and order were then run against the other candidates.

| Strict reply candidate | Cases | Valid answer-only JSON | Median total | Median first content | Median output tokens |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Q5, Thor llama.cpp | 24 | 24 | 1.912 s | 0.729 s | 35 |
| Qwen2.5-14B Q4, Thor llama.cpp | 24 | 24 | 1.806 s | 0.421 s | 27.5 |
| Qwen3.8-27B BF16, Thor vLLM | 24 | 24 | 7.907 s | 0.598 s | 35 |
| SmolVLM2-2.2B Q4, Jetson llama.cpp | 24 | 13 | 4.163 s | 1.311 s | 173.5 |

The Q14 median is not proof that it computes faster than Q5: Q14 had a median 503.5 cached input tokens versus Q5's 336, while aggregate native decode was about 21 versus 28 tokens/s. Smol's eleven malformed responses ended at the 192-token cap. Increasing the cap could change structural validity, but its already-complete answers also fail semantically.

Observed failures, retained without retrying to improve scores:

- Q5 answered an unknown-history question with an unsupported assertion that the user had not prohibited deployment. It omitted the nonblocking question in the two-question inventory even under the stricter instruction.
- Q5 asserted that a restart had not taken effect because deployment was unauthorized and tests were running. Neither fact establishes the restart's execution history.
- Q14 handled missing approval history more cautiously and returned both outstanding questions, but still claimed the restart had not executed. It asserted loss of test progress without a recorded restart/resumption contract and emitted percentages/internal IDs despite audio instructions.
- BF16 correctly expressed uncertainty for the negative deployment hypothetical and prior prohibition, but another held-out history question still received an unsupported denial of earlier release approval based on current authorization.
- Smol confused earlier successful tests with the current unfinished run, echoed questions instead of answering, reversed a negated deployment hypothetical, and failed exact-output instructions.
- The compact production-gate comparison still treated an implicit deployment restriction as a direct conversational answer. A fluent acknowledgement is not evidence that a restriction reached a worker.

Three additional Q5 probes removed the potentially ambiguous empty history array and explicitly marked previous approvals unavailable. One response acknowledged uncertainty, but unsupported denials and invented causal explanations remained. All three original outputs remain in `reader-absence-probe.jsonl`.

These are counterexamples and feasibility observations, not calibrated semantic accuracy estimates. The held-out set is small; there is no independent blinded semantic scoring. Structurally valid JSON does not qualify a model for instruction routing or factual status reporting.

## Protocol receipt, backpressure, and cancellation

The prior ordinary method persisted input before the session execution lock, but the real JSON-line dispatcher still acquired the global request semaphore first. A correction therefore remained outside the durable inbox while a request occupied that slot. Moving receipt before that semaphore alone was insufficient: `asyncio.to_thread` shared the executor occupied by model processing.

The final implementation gives finalized-message receipt a dedicated one-thread executor and a bounded admission semaphore. The durable inbox retains its own capacity limit. Receipt jobs do no model inference; execution still passes through normal request and session admission. Transport cancellation cannot cancel an already submitted receipt or release its ingress capacity early. Overflow explicitly reports that the request was not accepted. Already accepted jobs drain on service close.

Focused reproductions cover the real dispatcher with its normal executor and a one-thread saturated executor, exact Unicode/whitespace preservation, ordering, superseding controls, failure/restart recovery and bounded transport cancellation.

The uncached live test used actual TCP connections, the real protocol dispatcher, a single global request slot, a one-thread default executor, the Q3 planner and normal completion evaluation. Only the calculator was enabled.

- The first planner action request began at 13.723 s.
- Cancellation was sent at 13.735 s and observed in the durable inbox at 13.762 s: **26.36 ms**.
- The original request and its admission slot were still occupied.
- One stale action was rejected. **No tool executed.**
- Both callers received the reconciled answer at about 103.8 s.

The fast receipt result is not a fast conversational-response result. Five serial model calls consumed 95.451 s of 103.850 s overall; summed local admission waits were about 0.24 s. The remainder includes setup, serialization/tokenization, durable I/O, transport and verification, not a measured pure orchestration cost. Prompt budgets on the chat backend were explicitly inexact estimates.

One 11.228 s repair call was caused by the model inventing an exact twenty-five-word requirement. The separate repair correctly rejected that nonexistent requirement. The action prompt already says this field must be null unless explicitly required; removing the validator because the call is expensive would conceal the error. Likewise, completion evaluation was enabled and measured, not silently omitted to improve the timing.

## Snapshot completeness and freshness

A real snapshot taken during planner inference had no workers or plans yet. Supervision reported model activity but did not supply the exact objective or outstanding user receipts. The snapshot now includes the orchestrator session, active run/objective and exact pending finalized user messages, alongside the existing worker, plan, result, question and supervision evidence.

A task-fact fingerprint covers active-run identity/objective, pending user messages, worker/plan state and questions. Heartbeat, inference phase/substate/detail and polling clocks remain timestamped observations in the prompt, but do not alone invalidate a reply. This distinction was established by a live counterexample: dispatching became awaiting-result while the task and all supplied execution facts were unchanged. The original failure is retained; re-evaluating those same saved snapshots under the corrected comparison passes without pretending a new inference run occurred.

This remains optimistic per-source observation, not a cross-store transaction. State can change after the final read. Mechanically changed task facts trigger the existing bounded refresh/rejection behavior. Detailed telemetry is observed at capture time rather than guaranteed unchanged at speech delivery.

## Actual compact-reader overlap and surrounding cost

An experimental Q14 read-only call used SWAAG's normal semantic-call compiler, schema validation, inference admission, exact token accounting and full current snapshot. The exact status question independently entered the ordinary protocol's durable inbox. The experiment did not replace the production response route.

The status receipt took about 25 ms while the planner was busy. Reader execution began at 13.510 s and returned at 25.647 s: **12.138 s**. Both ordinary callers still waited. The reader correctly stated that the requested calculation had not yet run; the captured canonical execution inventory was explicitly complete and empty. Its answer still used numeric digits, so this is not an audio-style pass. The original freshness comparison rejected the draft on the inference substate transition described above.

The reader's exact prompt was 1,414 tokens. Native inference took 7.336 s, including 3.236 s to evaluate 1,077 uncached prompt tokens and 3.879 s to decode thirty-six tokens. Decode speed was about 9.3 tokens/s while the separate Q3 planner was active, compared with about 21 tokens/s in the standalone synthetic Q14 matrix. This is consistent with shared-hardware contention, but prompt differences and uncontrolled ambient load preclude a pure causal throughput estimate.

The authorized calculation eventually executed exactly once and returned 30,003. The status question did not cancel the task, but its arrival conservatively invalidated an in-flight draft. Ordinary responses arrived at about 150.5 s, roughly 137 s after the status request. A separate endpoint allows overlap; it does not remove shared GPU contention, control reconciliation or history-delivery requirements.

A subsequent instrumented replay used the exact archived reader input after the planner finished. It is explicitly not a new current-state conversation trial. The first runtime operation took 6.797 s, with 2.196 s model time; twelve tokenizer requests took 2.262 s, five props requests 0.934 s, and template application 0.199 s. Repeating in the same runtime took 3.667 s with 2.130 s model time and no tokenizer network calls, because exact token counts were cached. Four props requests still took 0.759 s. Native prefix reuse covered 1,413 of 1,414 prompt tokens in both replays.

Source inspection explains why those props reads are not automatically redundant: serialization identity, capacity discovery and pre-dispatch model/template verification use them to prevent inconsistent or stale admission. A future optimization should batch/cache work within a verified server-generation boundary and preserve exact accounting and change detection. It must not delete these checks or discard context merely to lower latency.

## Shutdown correction and release verification

A persistent idle TCP client reproduced a ten-second SIGTERM timeout. On the installed Python runtime, cancelling `serve_forever` waited for connections before the service's cleanup could close them. The service now owns listener cancellation, tracks client writers, closes clients before awaiting server closure, and drains accepted ingress jobs. Python 3.13's native client-close API also covers accepted transports whose handlers have not yet started; tracked writers retain compatibility with earlier supported Python versions.

The targeted ingress/shutdown suite passed 68 tests. The final snapshot/control suite passed 19 tests, including the new inference-substate reproduction. Earlier focused failures are retained as evidence. Release evidence and the full-suite fixture correction are recorded below.

## Architecture decision

No new model split or prompt is promoted. The measurements support independent read-only response generation as a candidate and disqualify relying on these tested small-model decisions to discard instructions. They do not establish complete factual or audio quality even for the larger BF16 reader.

The next acceptance boundary is a reliable ordinary response path with durable exact instruction receipt independent of conversational interpretation, reconciled conversation history, current-state validation, bounded load/latency measurements and verified worker control. Same-backend capacity discovery remains separate from the server's ability to run two sequences. Voice finalization, TTS overlap, partial-answer safety and long-run cancellation/tail latency remain unmeasured here. Receipt durability and shutdown are implemented corrections; interactive conversation remains an open requirement.


## Release evidence

Implementation commit: `0860786a349278ef42bc2ac0d628f098f194e155`.

The broad deterministic run completed in 1,074.99 seconds: **1,147 passed, three optional skips, one failure**. That run began before the final inference-substate fingerprint refinement. Its failure was an existing concurrency test double that chose its answer by model-call ordinal: a legitimate snapshot refresh for the first user was therefore returned as the second user's answer. The fixture now answers the exact current utterance, waits for verified durable receipt instead of a timing assumption, and asserts one refresh plus unchanged user/assistant history ordering. Production reply routing was not changed to fit the fixture.

The final source checks cover preemption, concurrent ordinary controls, snapshots, communication service lifecycle and queue capacity: **111 passed in 133.12 seconds**. The complete broad suite was not redundantly repeated after that localized fixture/fingerprint correction. Its original failure remains in `full-suite.txt`; the final result is in `final-regression.txt`. This is not represented as a clean full-suite rerun.

The canonical installation script rebuilt the production package, passed import and CLI smoke checks, and source-byte comparison verified all **190 tracked package files**, with no mismatch. A further **63 tests passed in 62.25 seconds** while importing the installed package explicitly, including the finalized-input, snapshot and persistent-client shutdown regressions.

A real uncached request then ran through the installed interpreter and actual TCP dispatcher using the existing Q3 backend. It returned exactly `ready` in **11.900 seconds**, preserved the exact input once, executed zero tools and left zero pending receipts. This certifies the exercised installed path, not the inactive default Jetson model endpoint or interactive latency.

Infra retained its pre-existing `hosts/jetson/config.json` edit. The communication service remained inactive, consistent with the existing foreground-use arrangement; the installation did not restart active user work or enable an automatic service.
