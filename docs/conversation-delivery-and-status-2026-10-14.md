# Conversation delivery, status admission and reasoning — 2026-10-14

## Material resource failure

This experiment failed resource qualification. Despite guarded admission, the added Q5 server's unreserved host prompt cache filled and available memory repeatedly fell below the 4 GiB reserve. At 06:24:18 UTC the host safety guard terminated an unrelated Blender render under combined pressure. The study recorded memory but did not enforce the reserve while running; the later cleanup check was too late. The owned Q5 service is now stopped. Its original unit is unchanged, and the two pre-existing model services remain healthy with unchanged PIDs and no restarts.

Infra's admission estimator omitted the installed llama.cpp default 8 GiB saved-prompt cache. The correction is committed and pushed as 806fd60 and applied to both Jetson and Thor without restarting any running model. Four regressions failed before the correction; the 17-test suite passes on both machines. A read-only check against the actual inactive Thor unit now refuses admission: 18.5 GiB available versus an 18.1 GiB estimate plus 4 GiB reserve. No model was started by that verification. Full incident evidence and remaining allocation limits are documented in Infra docs/llms/swaag-memory-admission-2026-10-14.md. The interrupted rendering work was not blindly restarted.

## Scope and evidence

This continues the [scheduling/accounting investigation](conversation-scheduling-and-accounting-2026-10-14.md) from source baseline 88d5903. Evidence is retained under /data/var/swaag-benchmarks/conversation-delivery-study-20261014 on Jetson; Thor holds service provenance and memory samples under the same path. Requests, native timing frames, complete streamed events, isolated canonical histories, durable delivery outboxes, exact inputs, failed fixtures and corrected reproductions are retained. The analyze.py script reconstructs analysis.json.

The ordinary orchestrator.message interface retains durable exact finalized receipt and a single canonical planner writer. This study does not give a small reader authority to drop, dispatch or reinterpret instructions. The two-stage delivery harness is experimental, not the ordinary CLI/voice implementation.

## Reproduced corrections in the existing status API

A dispatcher reproduction occupied both the global request semaphore and a one-thread default executor. The ask_status operation could not start even though its independent model could run. The original test timed out; a dedicated bounded status executor now admits this work independently of the planner pool. Inference coordination and shared-model preemption remain authoritative. The max_concurrent_requests setting bounds status executor width, and max_pending_requests bounds its running-plus-queued jobs separately from durable communication requests. A disconnected waiter cannot cancel an admitted job or release capacity while it is still running. Backend/submit errors release the slot; shutdown drains admitted work according to the service's wait policy.

Separate reproductions showed the status API returning an obsolete answer after a tool result or finalized control arrived during generation. It now rechecks durable task facts, refreshes once, and rejects a second changed draft. The current snapshot includes exact pending control receipts, observation time and per-source consistency qualification. Heartbeat, token accounting and inference scheduling alone do not force a refresh; effect start/result/error, progress, questions, task/run identity and pending controls do. Stronger-model escalation recaptures state after its preemption wait.

A rejected draft remains historical evidence in its isolated operation session but is excluded from the latest-status projection. Small-model drafts that requested escalation are also excluded, including when the stronger call fails. This does not turn optimistic observation into a cross-store transaction or promise that state cannot change after the final read.

The first freshness fixtures used incomplete event schemas and exercised fallback instead of the intended race. Those errors are preserved separately. Corrected baseline reproductions fail on actual stale replies, then pass after the correction.

Focused source checks: 64 passed across status, preemption, history replay/projection and event compatibility; 76 passed across protocol lifecycle, ordinary finalized controls, orchestrator snapshots and bounded queues. The earlier 58-test run certified admission before the freshness refinement. The full deterministic baseline from the preceding study remains 1,157 passed and three optional skips; this is not a new full-suite run for the present change.

## Actual two-stage delivery

The experiment uses real receipt and the unchanged canonical planner, an isolated read-only semantic runtime, normal context compiler/token accounting/schema validation, and a separate durable SQLite delivery outbox. It sends distinct accepted, observed, unavailable and completed events over actual TCP. Acceptance means durable receipt only. A reply is checked against both current task facts and canonical execution/progress/completion inventory. Observation failure does not cancel planner work.

| Trajectory | Follow-up receipt | Follow-up observation at client | Follow-up final result | Actual tools |
|---|---:|---:|---:|---:|
| Q3 planner, BF16 reader, FIFO observations | 21 ms | 61.116 s | 167.764 s | one calculator |
| Q3 planner, Q5 9B reader, obsolete-observation cancellation enabled | 23 ms | 10.551 s | 136.663 s | one calculator |
| Q3 planner, Q5 9B reader, cancellation plus disconnect/resume | 32 ms | 8.359 s | 112.505 s | zero |

These are individual trajectories, not latency percentiles or a controlled model-speed ranking. Reader model, output length, prefix-cache reuse and scheduling differ. Existing production services were not resized or restarted.

The BF16 first observation changed twice and was withheld. It occupied the reader ahead of the status question. A later status answer reached the client while planning continued. A second model alone therefore does not guarantee responsiveness.

The Q5 reader used an experimental constant enable_thinking=false template argument. Native completion and token accounting consumed that exact closed-thinking prompt; production model configuration was not changed. Its initial reply arrived in 11.158 seconds but falsely described SWAAG as incapable of using the calculator because the reader itself was read-only. The later answer correctly stated that calculation had not run, but exposed internal dispatch terminology and numeric digits. The authorized calculator eventually executed exactly once and returned 30,003. Delivery speed is not a quality pass.

In the cancellation trajectory, observation was deliberately started during planner action inference; the finalized correction followed reader inference entry. The old pure observation was cancelled in about 0.816 seconds. The client disconnected after receipt and resumed by receipt ID and event cursor. Both exact user messages remained in canonical history, the stale planner proposal was rejected, zero tools executed, and the outbox retained the final answer. Both receipt results refer to the same canonical completion event, allowing clients to deduplicate that terminal answer by identity rather than text.

Limits: reconnect was tested within the same process, not crash recovery of in-flight delivery jobs. Prototype delivery history is not reconciled into canonical conversation context, so subsequent references to an early spoken answer remain an acceptance gap. Bounds, multi-client ownership, backpressure, TTS playback/interruption and service-crash recovery need explicit contracts before promotion. No TTS latency or Android acceptance is claimed.

## Qwen3.5-9B reasoning comparison

Thor's existing inactive Q5 service was admitted through the canonical guard: 18.5 GiB available, 10.1 GiB estimated footprint, 4 GiB reserve and 4.4 GiB extra headroom. That startup estimate was incomplete, as detailed above. It retained llama.cpp build b9842, context 32,768, one slot, q8 K/V, Flash Attention, fourteen threads, batch 1,024 and microbatch 256. Previously running BF16 and Q3 services stayed in place.

Sixteen cases include twelve previous cases and four newly fixed holdouts. Pair order was randomized with seed 83, temperature zero and the same strict read-only prompt/state/schema. Native prefix caching remained enabled; saved-response caching was absent.

| Policy | Valid answer JSON | Median total | Median first content | Median native prefill | Median native decode | Median output tokens |
|---|---:|---:|---:|---:|---:|---:|
| thinking disabled, total cap 1,024 | 16/16 | 3.678 s | 2.557 s | 0.340 s | 1.492 s | 43 |
| unrestricted thinking, total cap 1,024 | 0/16 | 33.903 s | none | 0.274 s | 32.207 s | 1,024 |
| thinking budget 128, total cap 512 | 16/16 | 6.657 s | 5.074 s | 0.278 s | 6.123 s | 184 |

Unrestricted failures all reached the total output cap while still producing reasoning; none emitted answer content. This establishes starvation under that policy, not inability to answer with unlimited tokens. A generic arithmetic sentinel succeeded with default and explicitly enabled thinking, so the parser/template path itself was not simply broken.

Installed llama.cpp source maps the chat parameter thinking_budget_tokens to its reasoning sampler. An eight-token-budget sentinel returned a valid answer in 1.469 seconds with twenty-one completion tokens. The follow-up then tested all sixteen cases with budget 128. Requests and source references are retained. The total cap also changed, so this is a different bounded policy, not an isolated single-parameter causal claim.

Bounded thinking fixes output starvation but does not qualify the reader. It still denies an earlier prohibition when history is unavailable, asserts an unrecorded restart did not occur, invents lost progress after restart, and explains restart execution using unrelated deployment authorization. It sometimes mistakes its read-only role for inability of the whole system to cancel work. The two-question inventory improves in this sample, while other failures remain. No model, reasoning setting or prompt is promoted.

These are retained counterexamples and a small feasibility matrix, not a calibrated semantic-accuracy estimate. Native phase times are server counters; wall time also includes transport, streaming and scheduling. Medians of phases need not add to median wall time.

## Acceptance boundary

Evidence supports separating durable finalized input from observation delivery and superseding obsolete pure response computation. It does not support removing planner serialization, letting a small model discard constraints, or treating a provisional reply as proof of an applied instruction.

Implemented changes repair an existing status API's admission and stale-answer handling. They do not make ordinary conversation asynchronously responsive by themselves. Ordinary delivery still needs demonstrated conversational accuracy, canonical visible-history reconciliation, correlated completion ownership, crash recovery and measured voice behavior.


## Live existing status API and prompt-contract correction

The first real TCP ask_status trial used the Q3 planner and separate Q5 status runtime with both the shared request limit and default executor set to one. Admission now worked while the planner was busy. However, two Q5 calls failed local validation because action was empty, and fallback preempted/replayed the main model. The status answer took 70.955 seconds and the planner completed at 214.341 seconds. One calculator executed; zero pending controls remained. The harness assertion of zero preemptions correctly failed and is retained. This is not a successful independent-model trajectory.

Investigation found a prompt/validator contradiction: the prompt permitted empty metadata wherever the portable schema allowed it, while the local validator required nonempty answer, situation, action and reason. The portable schema intentionally excludes minLength, so it was not weakened or bypassed to resolve the mismatch. Frozen-input replay compared the original wording with explicit required-field wording. Original prompts produced empty action in three of four cases. Two aligned formulations eliminated those empty values but each had one truncated response in four raw trials at the original 192-token cap. Raw replay did not apply SWAAG output recovery and is not a new current-state observation.

The final prompt states the existing nonempty requirements, distinguishes model inference from tool invocation/effects, and retains the short metadata and audio-style requirements. With that prompt, the complete live API returned its status in 50.417 seconds while the planner remained active. The first 192-token response was truncated; normal output recovery increased the allowance to 288 and completed with 213 tokens. Both calls used the independent Q5 endpoint. No planner preemption occurred, the calculator executed once, and the planner finished at 120.525 seconds. These sequential trajectories have different backend load and generated tokens; their wall-time difference is not attributed wholly to prompt wording.

The new executor also needed explicit context-variable propagation to retain the request tracing context previously supplied by asyncio.to_thread. A regression exposed the missing propagation before release; copying request context fixes it. Final status/preemption/prompt checks passed 54 tests. Existing whole-system read-only-role confusion and other small-model factual failures remain open; this contract correction does not qualify the model.


## Installed verification and full-path timing

Source correction 140fd921c07313b6bf428d538d74431515419ee7 was installed through scripts/install-runtime-env.sh. All 190 tracked package files match the installed package byte for byte. Nineteen focused installed-runtime tests passed in 24.22 seconds with import-path verification; the final source status/preemption checks passed 54 tests. The prior full-suite result is retained as baseline evidence, not relabeled as a new full-suite run.

A fresh actual TCP trajectory used the installed package with an experimental 288-token status allowance and thinking disabled only in the harness. Its independent Q5 call returned 227 tokens in 22.642 seconds; complete status latency was 32.701 seconds. The canonical Q3 planner continued, finished in 113.652 seconds, and used the calculator exactly once. There were zero preemptions and no pending controls. The shared request semaphore was still occupied when status returned. This establishes the corrected admission path, not interactive latency or small-model semantic qualification. Neither the experimental output allowance nor model/template setting was promoted.

## Measured non-model overhead

Four frozen-input replays used fresh runtime/client instances, identical evidence, the same 288-token allowance and ABBA ordering of ordinary versus compact tokenization-audit logging. Native prefix reuse remained enabled. Every trial emitted the same 227-token answer. Full time ranged from 17.918 to 22.191 seconds, while model time ranged from 8.343 to 8.910 seconds. Each compilation made fourteen sequential tokenizer HTTP requests, taking 6.160 to 8.474 seconds, plus template and other work.

Normal/compact median full times were 20.336/19.542 seconds. Network and repeated-run variation prevent attributing that difference to audit logging; no assistant audit policy was changed. Durable accounting remains intact.

A separate twelve-request ABBA transport probe used identical full-prompt tokenizer bodies. All returned 2,827 tokens. Fresh-connection median was 0.502 seconds, pooled median including first connections was 0.295 seconds, and the four warm pooled requests had median 0.268 seconds. The probe's explicit tokenizer body was consistent within the comparison but differs from the production body's optional flags. This is a candidate transport optimization, not a measured end-to-end SWAAG improvement. No shared Session or new tokenizer cache was introduced: concurrency ownership, cancellation, closure, authentication and record/replay equivalence require focused verification before promotion.

## Final disposition

The existing status API's independent admission, request-context propagation, current-state recheck and prompt/validator contract are corrected and installed. Exact finalized controls and planner serialization remain authoritative. The small reader and two-stage delivery prototype remain unqualified because of retained factual/role errors and unresolved visible-history, crash-recovery, ownership and voice acceptance. The tested Q5 co-residency configuration is also rejected on resource evidence. Independent capacity can improve scheduling, but the measured full path and semantic failures do not justify deploying a separate conversational model.

Thor evidence contains 2,423 one-second memory samples, 1,028 below 4 GiB and a minimum of 1.40165 GiB. Guard attribution and source evidence identify the omitted cache; they do not fully apportion every host fluctuation. No further model start was attempted after the incident review. Future live experiments require active reserve enforcement on the owned experimental process, a complete cache/checkpoint budget and current host-load admission.
