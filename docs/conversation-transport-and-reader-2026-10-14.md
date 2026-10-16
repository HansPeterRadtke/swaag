# Conversation transport and small-reader follow-up — 2026-10-14

## Verified transport correction

Source baseline ff60c1d. The preceding study measured fourteen sequential native tokenizer requests during a full status operation. Each request opened a fresh HTTP session. A real loopback HTTP/1.1 reproduction confirms five separate connections in one context compilation before the change.

Native tokenizer connections are now reused only within one synchronous context compilation. Each client and OS thread owns its session; nested scopes borrow that session. Scope exit closes the pool after success, failure or cancellation. Responses are consumed/closed, server cookies are cleared between requests, and authorization/trace headers are obtained afresh for every request. Model generation, inference cancellation, model identity probes, request bodies, exact accounting and durable audit records are unchanged. Record/replay forwards this optional scope only in live record mode; offline replay never opens the delegate transport. Clients without the capability retain their existing behavior.

The Requests documentation describes session connection pooling, cookie persistence and response-consumption requirements: https://requests.readthedocs.io/en/stable/user/advanced/. Installed Requests is 2.34.2. The implementation deliberately does not share a Session between concurrent threads or retain server cookies across independent requests.

Focused verification: 87 tests pass across real HTTP transport, compilation, record/replay, model identity/context, streaming and tokens. Another 54 tests pass across status admission, freshness, status generation and preemption. A new fixture initially attempted to patch a slotted instance method; that test-construction error and its corrected rerun are retained. An incorrect regression filename selected no tests and is also retained separately.

## Controlled full status-operation comparison

An already-running Thor Qwen3-VL-8B-Instruct Q4_K_M service provided native llama.cpp b9842-6f4f53f2b with 8,192 context tokens and one slot. The service was idle when inspected. It was neither restarted nor reconfigured. Access from Jetson used an owned SSH loopback tunnel, with its host key verified independently through the Thor connector. No host-key check was disabled.

The experiment used the complete frozen status request and durable event sources from the prior installed trajectory. It is a replay of fixed evidence, not a claim about live current worker state. One warmup precedes eight ABBAABBA measured trials, four with the old per-request behavior and four with scoped reuse. Every trial creates a fresh runtime/client and disables SWAAG saved-response caching. Server prefix/cache behavior is unchanged. Template thinking is disabled only in this harness; the experimental output allowance is 288 tokens.

| Measure | Fresh tokenizer connections | Scoped reuse |
|---|---:|---:|
| Trials after warmup | 4 | 4 |
| Median full status time | 11.863 s | 10.276 s |
| Median tokenizer HTTP time | 3.357 s | 1.849 s |
| Median model time | 5.773 s | 5.818 s |

All eight measured trials have byte-equivalent canonical completion payloads, the same 2,815 prompt tokens, the same 191 output tokens and exactly the same complete response text. All fourteen tokenizer body hashes match in order; each trial retains 45 history events including 28 tokenizer audit events. The median full-path reduction is 1.586 seconds (about 13.4%). This is evidence for this warm, fixed-input, SSH-forwarded deployment path, not a universal latency estimate or a small-model quality pass. The first warmup produced a different answer and is excluded explicitly.

## Safely aborted additional-model startup

Before using the existing endpoint, an isolated temporary Q5 9B service was admitted through the corrected Infra guard with host saved-prompt cache explicitly disabled. A separate supervisor sampled host memory every quarter second and would stop only the owned experimental unit below 8 GiB available, above Infra's 4 GiB reserve. Runtime duration was bounded independently by systemd. No production workload was resized or evicted.

The new process remained in model loading for about eight and a half minutes while consuming approximately one CPU core. A hardware query blocked in the NVIDIA driver; an attempted debugger attachment could not complete. The owned debugger was terminated and the experimental model stopped cleanly. The temporary inactive unit was removed. Minimum observed MemAvailable was 13.606 GiB across 2,006 samples; there were no reserve breaches or guardian events. Both existing vLLM endpoints then completed actual short inference requests, and the diagnostic processes exited. Startup delay is not counted as conversational latency.

These observations do not prove a specific CUDA allocation or fragmentation defect. The retained kernel memory errors predate this attempt. No GPU reset, MPS restart, global cache drop or production model restart was attempted. General reports of Jetson allocation delays are not sufficient to diagnose this exact Thor event.

## Installed concurrent status trajectory

The installed package at source fe70bef was exercised through the actual TCP communication listener with main request admission and the default executor each limited to one. The planner used the existing Thor Qwen3.8 Q3 endpoint; status used the existing Qwen3-VL 8B endpoint. The inference registry proves distinct backend routing. The status reply returned in 20.323 seconds while the planner was still running and its main request slot remained occupied. The planner finished in 100.800 seconds, made exactly one calculator call with result 30003, left no pending input and suffered no preemption.

This is a concurrency pass and a semantic failure. The answer said the calculator was actively computing, but the supplied snapshot explicitly reported model inference/dispatching. The calculator tool call occurred 19.123 seconds after status generation. The prompt already required tool events before claiming tool invocation. Thus this failure is unsupported interpretation, not stale evidence or a lost instruction. Numeric output also violated audio style. The detailed evidence and interpretation are retained in live-status-api-installed-qwen-vl/analysis.json.

Status used 2,819 input tokens, 170 output tokens, 0.240 seconds of local inference admission, 5.186 seconds of model time to first token and 15.526 seconds total model time. The remainder of the 20.323-second path includes context construction/accounting, state reads and delivery. The raw answer-only matrix below is not an end-to-end latency estimate. Independent backends still share Thor compute resources; the warm fixed-input transport comparison and this concurrent trajectory have different load conditions.

The planner made four semantic calls: initial interaction, tool selection, final-answer action, and completion evaluation. Their measured model times were 10.224, 39.203, 27.097 and 17.802 seconds respectively, about 94.326 seconds in total. The separate completion check is a substantial measured cost, but this calculator trajectory does not establish that removing a general objective-verification step preserves correctness for other work. No such removal is made.

## Reader quality and larger-model controls

Ninety-six raw inference trials completed: twenty-four fixed cases on each of two 8B prompt policies, followed by the same twenty-four strict-policy cases on each existing 27B backend. Sixteen cases came from the previous diagnostic study; eight new heldouts were fixed before execution. The new cases explicitly vary available and absent history, withdrawn permission, confirmed restart, already-completed cancellation targets, old failed revisions, documented resume semantics, answered questions and worker-log claims of authority. They are synthetic state interpretation tests and execute no orchestrator actions.

The 8B policy pair used randomized order with seed 109. The two 27B profiles used randomized paired order with seed 137. Generation remained temperature zero, seed 109, thinking disabled, maximum 512 output tokens and an answer-only JSON schema. Cross-profile verification confirms identical strict-policy request contents apart from model identifier. No saved-response cache was used; existing server prefix-cache policies and loads were not changed. The native 8B model runs Q4_K_M on llama.cpp; the 27B BF16 and UD-Q3_K_XL models run vLLM 0.28.1rc1.dev233+gcdefd9d49 with one and two admitted sequences respectively. Each matrix request was sequential, so these measurements do not isolate quantization, backend, cache or scheduling effects.

Complete responses were manually reviewed against the fixed expected criteria. This is an unblinded agent review, not an independent oracle or a population accuracy estimate. Supported means all required parts are answered without a material unsupported claim. Partial means incomplete or ambiguous. Failure means a material contradiction, invented authority/action/causality, or confusing absent evidence with a negative fact. Structural validity and clear spoken-style violations are separate. Per-case judgments, qualifications and response hashes are retained in the two review.json artifacts; partial results are not counted as supported.

| Profile | Valid JSON | Supported / partial / failed | Median total | Median first content | New eight heldouts supported | Clear style violations |
|---|---:|---:|---:|---:|---:|---:|
| 8B strict | 24/24 | 12 / 3 / 9 | 1.656 s | 0.595 s | 6/8 | 10/24 |
| 8B compact | 20/24 | 10 / 2 / 12 | 2.450 s | 0.455 s | 4/8 | 16/24 |
| 27B BF16 strict | 24/24 | 18 / 3 / 3 | 9.367 s | 0.591 s | 6/8 | 2/24 |
| 27B Q3 strict | 24/24 | 19 / 1 / 4 | 7.088 s | 2.691 s | 7/8 | 2/24 |

The compact 8B prompt had fewer median input tokens (427.5 versus 551.5), but produced more median output tokens (77 versus 41.5), repeated itself to the output limit four times, and was slower overall. Shortening the policy did not improve this measured quality/latency tradeoff. All four truncations remain in the denominator. An early harness stopped on its first JSON failure; the resumed harness retained every completed row, continued structural failures and verified all forty-eight distinct case/policy pairs. A resume-script syntax error is retained separately and is not an inference trial.

Concrete failures matter more than a small aggregate ranking. Both 8B policies denied an explicit earlier prohibition; the BF16 model also failed that case while Q3 answered correctly. All four profiles asserted current tests were not failing even though their outcome was unknown. Both 27B profiles denied historical production approval despite missing conversation history, even while correctly abstaining on a closely related past-prohibition question. The larger models handled several other missing-history and hypothetical questions better. These inconsistent boundaries prevent promotion of any tested reader as an instruction-routing authority.

The tested strict 8B prompt is useful diagnostic evidence, not a replacement for the production status contract. The raw matrix omits the runtime compilation, durable receipts, delivery and factual recheck costs that appear in the installed trajectory. Conversely, JSON correctness, low first-token latency and a live concurrency pass do not establish truthful conversation.

## Release and operational verification

The transport change is installed in the Jetson runtime. All 190 tracked package files match source fe70bef byte for byte; twenty-five focused tests pass against imports from the installed package, including physical HTTP connection reuse/cleanup, record/replay identity and status admission/freshness. Source verification remains the 87-test and 54-test focused suites above. The earlier complete suite is historical evidence, not a new full-suite claim for this release.

The existing-model comparison and live trajectories were supervised with 3,057 host-memory observations. Minimum available memory was 18.474 GiB; no observation crossed the experimental 8 GiB abort threshold or Infra reserve. The earlier isolated startup phase has its separate 13.606 GiB minimum above. A privileged query of the actual infra-safety-guard.service journal recorded no events since this phase began. The preexisting BF16, Q3 and 8B model processes remained running with their original process identities; both managed vLLM units retained zero restarts. The owned temporary model unit is absent and the owned SSH forwarding and monitoring processes have exited. This does not erase the preceding study's separately documented reserve violation and unrelated-render termination.

Read-only baseline inspection also found that Jetson's default local model service had already been failed since October twelfth and that the ordinary /data/bin/swaag link had a missing Infra target. The installed absolute CLI works, and every inference experiment here uses explicit existing Thor profiles. Neither the default model configuration nor unrelated launcher/configuration state was changed to make a benchmark pass. The unrelated Infra hosts/jetson/config.json modification is preserved.

## Architectural conclusion and unresolved acceptance

Promote only scoped tokenizer transport reuse. Preserve exact user-message receipt independently of model interpretation, canonical planner ownership, evidence checks and current model defaults. The measured independent status path can overlap real planning; it is still too slow and semantically unreliable to certify the requested fast ordinary conversation. The ordinary orchestration session lock is not removed. These results do not certify ordinary early replies, concurrent speech execution, reliable small-model routing or a new process/model split.

Future acceptance must demonstrate reliable interpretation of both present and missing authority/action evidence on fresh heldouts and actual changing worker states, together with end-to-end responsiveness and verified durable instruction ordering. A more concise prompt, larger model, valid schema or one successful demonstration is not an adequate substitute. The current evidence supports targeted transport improvement and continued conservative routing, without choosing an unverified architecture.

## Evidence

Jetson and Thor retain artifacts under /data/var/swaag-benchmarks/conversation-transport-study-20261014. Jetson includes the failing transport reproduction, focused tests, exact full requests/results, HTTP timings, supervised memory trace, and scripts. Thor includes unit provenance, guarded admission, startup and guardian journals, memory trace and diagnostic outcomes. The completed matrices, frozen manifests, full streaming outputs, per-case reviews, installed TCP histories, cross-profile checks and release identity are retained there. Evidence indexes enumerate file hashes. Structural validity and semantic accuracy are evaluated separately. No new reader or prompt is promoted by the transport measurement.
