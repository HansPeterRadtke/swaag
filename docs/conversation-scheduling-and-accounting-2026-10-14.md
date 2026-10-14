# Same-backend conversation scheduling and streaming accounting — 2026-10-14

## Scope and baseline

This continues the [October 13 response and ingress study](conversation-response-and-ingress-2026-10-13.md). The source baseline was clean at 337b3d6, matching origin/main. Infra was at c4cd308 with the existing unrelated hosts/jetson/config.json modification; that file was not changed. All experiments use isolated benchmark state under /data/var/swaag-benchmarks/conversation-scheduling-study-20261014 on Jetson. Thor provenance is also preserved under the same directory on Thor.

The running Thor endpoints were inspected directly. Qwen3.8-27B UD-Q3_K_XL remains on vLLM 0.28.1rc1.dev233+gcdefd9d49, port 14834, max_model_len 8192, max_num_seqs 2 and 1 GiB KV allocation. Its actual process command agrees with the service declaration. The BF16 endpoint on 14833 has max_num_seqs 1. Neither production service was restarted or reconfigured. Thor had about 18.4 GiB MemAvailable; Jetson about 19 GiB. No new model was admitted.

Both endpoints return 404 for /server_info. The model inventory reports the model alias and context length, but not the sequence admission limit. The /metrics cache_info value kv_cache_max_concurrency=1.17647 is a memory-related ratio, not the declared maximum request count; it is identical on these differently configured services. It must not be rounded or interpreted as capacity discovery. SWAAG's conservative chat-transport capacity of one is truthful. Capacity two below is an explicit experiment-only resolver override verified against the actual service launch, never claimed to be automatic API discovery.

Infra's methodology, measurements, phase table and model matrix were cross-checked. Their Q3 entry uses a different context/KV configuration, while the phase-separated table has no matching Q3/vLLM row for this deployment. The historical approximately ten-token-per-second Q3 result cannot be treated as a controlled baseline for these new requests. Today's comparison holds hardware, weights, server and payloads constant.

## Controlled admission experiment

admission.py uses the actual InferenceRequestCoordinator and its durable SQLite lifecycle, followed by raw SSE measurement. Six paired worker/status trials use capacity order 1,2,2,1,1,2. The worker has 1156 input tokens and exactly 192 generated tokens; min_tokens and ignore_eos intentionally hold the decode workload constant. This is an inference scheduling workload, not a completed software task or quality score. The status prompt has 90 input tokens and returns the same valid eleven-token JSON answer in all six trials. It starts only after the worker's first generated content. There are no tools, effects or response-cache hits. Server prefix caching remains enabled, and the first worker has an additional cold-prefix cost.

| Median, three trials each | Local capacity 1 | Local capacity 2 |
|---|---:|---:|
| Status completion from submission | 20.471 s | 3.025 s |
| Status queue delay | 18.344 s | 0.014 s |
| Status first content from submission | 19.515 s | 1.355 s |
| Worker total time | 20.418 s | 22.032 s |
| Worker still generating when status finishes | 0/3 | 3/3 |
| Maximum server running requests | 1 | 2 |

The warm single-capacity worker measurements are 20.399–20.418 s; the first is 23.883 s. Two-capacity workers take 22.016–22.042 s. Thus overlap improves status latency at a measurable worker cost, rather than making both requests uniformly faster. Every coordinator ended with zero queued/running requests. Native metrics independently prove two server requests running during overlap. Raw requests, events, usage and before/during/after metrics are retained per trial. Native cumulative phase counters reconcile exactly with two requests, 1246 prompt tokens and 203 generated tokens in every pair. The cold pair has zero prefix-cache hits and 6.255 s summed prefill time; every later pair has 784 cached tokens. Warm summed prefill is 2.803–2.804 s at capacity one and 2.896–2.898 s at capacity two. Summed decode is 19.276–19.295 s versus 21.637–21.664 s. Server-side queueing is below three milliseconds in every pair, confirming that the approximately eighteen-second serial delay is in SWAAG admission. These are sums of per-request phase durations, not wall-clock makespan or isolated kernel throughput; overlapping durations must not be added as sequential elapsed time. No native preemption occurs in these short-prompt pairs.

## Actual planner, finalized ingress and full-state reader

live_shared_reader.py runs a real TCP CommunicationService with one request admission slot and one default executor worker. A calculator-only planner receives the original authorized task, then a finalized status question after the first action request starts. The independent read-only prototype uses the **same** Q3 endpoint, the same SWAAG inference database, the complete current snapshot, complete conversation in the isolated session, explicit complete tool-execution evidence and the exact durable receipt. It does not route, apply or discard instructions. The ordinary protocol responses still use production FIFO/session locking. Capacity is the only intentional admission change between runs; generated answers and prompt lengths differ slightly, so these are trajectories rather than repeated identical-payload statistics.

| Observed trajectory | Capacity 1 | Capacity 2 |
|---|---:|---:|
| Finalized status receipt while planner is occupied | 14 ms | 13 ms |
| Prototype reader total | 43.102 s | 25.324 s |
| Prototype reader SWAAG queue delay | 31.727 s | 0.039 s |
| Prototype reader model time | 9.334 s | 23.419 s |
| Ordinary status response after question | 147.420 s | 149.912 s |
| Whole trajectory | 160.780 s | 163.115 s |
| Calculator calls / stale action rejections | 1 / 1 | 1 / 1 |
| Maximum server running requests | 1 | 2 |
| Server preemption counter increase | 0 | 1 |

Both readers correctly say that the calculation has not yet executed, return while the planner remains active, and pass the existing snapshot fingerprint recheck. The authoritative calculator then executes 137*219 exactly once, returns 30003, and both ordinary callers receive the committed final answer. The intervening question is preserved exactly and invalidates the old proposed action before execution. This verifies receipt, ordering and single execution for these trajectories, not a universal effect fence.

The capacity-two reader has 1451 prompt tokens and 42 output tokens, with first content at 16.413 s after dispatch. Its concurrent first planner action has 2967 input tokens, 216 output tokens, and takes 42.026 s. Queue elimination does not eliminate prefill, shared execution costs or server memory scheduling. The metrics show one server-internal preemption during overlap. The installed scheduler source requeues running requests when block allocation fails; the observed counter increase is consistent with that mechanism, but the sampled aggregate metrics do not identify an exact victim or allocation event. There is no basis to attribute all slowdown to preemption or promise that two declared sequence slots imply two continuously resident full-context requests.

The ordinary caller gains no corresponding benefit because the read-only prototype is not the production response path and the production session lock remains held. Removing that lock or promoting the prototype would require a separate authority, freshness, failure and delivery design. These measurements do not authorize silently applying a capacity-two override to arbitrary chat backends.

## Confirmed streaming-accounting defect and correction

The running vLLM implementation emits requested aggregate usage after the terminal answer chunk, then [DONE]. SWAAG previously neither requested this standard trailer nor read past the answer's finish_reason. Replaying the exact captured live status stream through the old parser yields prompt_tokens=null and completion_tokens=10, while the server reports 90 and 11. Stream-event count is not token count.

The chat client now requests include_usage, retains validated nonnegative integer prompt/completion counts and their provenance, reads the trailer, and preserves the answer's finish reason independently of the empty-choice usage chunk. Native llama.cpp requests retain their existing protocol. EOF without usage remains a clearly labelled event-count fallback; malformed numeric fields do not become exact counts. The SSE terminator ends consumption, and a streamed error after partial content is a failure rather than a successful partial answer. Native and chat counters are recorded separately from fallback provenance.

Seven initial regression cases failed before the correction. The initial focused suite then passed 48 tests; additional error/EOF regressions are included in the final certification below. An uncached live client request returned the exact Unicode answer in 2.687 s; provider tokenization and completion usage agree at 79 input and 16 output tokens, with stop preserved. This also resolves the tokenization hypothesis: the tested default chat request is counted exactly. No thinking-template mismatch was reproduced. Historical cached records are not retrospectively rewritten; recorded old counts remain historical evidence.

The usage correction improves measurement fidelity, not model speed. The runtime first_token_seconds field still describes first content; these non-reasoning JSON trials do not establish reasoning-token latency. Raw benchmark SSE retains its separate first-generated-token and first-content measurements.

## Extra-call investigation

Both live trajectories incurred an approximately eight-second exact-word-count repair call. The action model invented a forty-word response constraint while preparing the calculator call, despite no user instruction requiring such a count. The existing repair model correctly rejected that invented constraint and preserved the draft. The repair is unnecessary for the real user task, but is currently necessary to correct the model's invalid proposal. Deleting validation would conceal the error.

A predeclared synthetic prompt comparison is retained under constraint-prompt. It appends six finalized utterances to archived actual completed-calculator context, including two held-out variations, and compares the existing action prompt with one general clarification distinguishing a required count from a measurement of the generated answer. No tools are dispatched by this study. All twelve calls returned terminal structured actions with correct declared count metadata, for both prompts. Thus this terminal-context matrix did not reproduce the invented-count failure observed in the preceding tool-selection action and establishes no improvement for the clarification. Both prompts nevertheless failed both exact-length cases: the ten-word request produced eight words, and the held-out seven-word request produced five. Both ten-word answers also omitted the actual calculation result. Exact-string requests were preserved. The clarification is not promoted and the validators remain. These are synthetic appended-turn results, not actual ordinary-API response failures; the production repair/completion path was not bypassed in the earlier real trajectories.

## Higher-precision small-model comparison

The available Qwen3.5-9B Q8_0 weights were tested on the same Thor llama.cpp build b9842-6f4f53f2b as the earlier Q5_K_M reader. The exact prior 24 requests, ordering, schema, seed 79, greedy decoding, reasoning-disabled setting and output bound were reused. Both configurations use context 32768, one slot, q8_0 KV, Flash Attention, 14 threads, batch 1024 and microbatch 256. The historical Q5 sample and new Q8 sample are sequential measurements, not simultaneous randomized trials; prefix-cache occupancy and cold startup differ.

The preexisting nine-billion-parameter service was verified inactive before use. A recorded runtime-only override selected the Q8 weights, a distinct alias and port 14835. The canonical guarded starter admitted an estimated 13.1 GiB footprint against 18.4 GiB available, retaining the 4 GiB reserve and about 1.2 GiB estimated additional headroom. The original admission backstop stayed active. After the test the owned process was confirmed idle, stopped, and only the experiment override was removed. The complete service definition matches its pre-experiment bytes and is inactive again. The two production vLLM processes remain unchanged with zero restarts.

Loading took several minutes rather than constituting a refusal. The ordinary command environment denied NVIDIA device access for a separate version probe; that probe eventually printed the same build and was not the model service. The model service had actual NVIDIA device descriptors, and a short native backtrace during the investigation showed CUDA stream synchronization in inference. It subsequently served all cases normally. This distinction prevents a diagnostic-shell permission error from being misreported as model incompatibility. The backtrace probe overlapped early inference and is retained; latency samples are not claimed to be entirely free of inspection overhead. The memory sampler retained 271 observations, with minimum MemAvailable 7.856 GiB. This establishes this short-request study's reserve, not a fully exercised 32768-token memory-capacity certification.

Both Qwen template/parser sentinels passed: reasoning disabled returned answer 4 with no reasoning in 0.787 s; enabled returned answer 4 with separately parsed reasoning in 10.240 s. Captured template suffixes confirm the respective closed/open thinking prefixes. Neither was truncated.

All 24 Q8 responses were parseable JSON. Median total latency was 2.219 s, median first content 0.717 s, median native decode 23.935 tokens/s and median output length 37 tokens. Historical Q5 median total was 1.912 s with median output length 35; output variation and prefix caching prevent attributing that entire difference to quantization.

Q8 still fails material semantics: it reverses the question about **not** deploying into an answer about deploying now; treats unknown prior instructions/approval as a negative; invents deployment authorization as the cause of a restart not taking effect; invents that a quoted cancellation came from historical logs; and omits nonblocking questions in one all-questions case. The separate held-out three-question inventory is complete. Several answers also expose internal IDs or numeric status despite the requested spoken style. Higher precision therefore does not qualify this small reader or explain away the Q5 failures. No routing authority or production response path is transferred to it.

## Certification and disposition

The final source suite passed 1157 tests with three optional skips and no failures in 1138.20 seconds. Source and tests were unchanged during this run. The correction is committed, pushed and installed through the canonical runtime installer. All 190 tracked package files match the installed package byte for byte. A further 61 focused streaming, tokenization and cache tests pass against the installed package in 2.21 s, with the import path explicitly verified.

An uncached installed-runtime TCP request returned exactly ready in 11.936 s. Its single model call took 10.122 s and retained the server's 1334 input / 43 output token usage and stop reason. No tools ran, no controls remained pending and the exact user text occurred once. This is installed execution evidence, not a claim of interactive latency. The preceding full-state concurrency experiments used the same tested source correction before installation.

The production BF16/Q3 services retained processes 2546/77648 and zero restarts. The temporary Q8 process is stopped, its runtime override removed, and the preexisting inactive service restored exactly. No Infra repository file was modified; the unrelated Jetson configuration edit remains. Raw evidence, scripts, source/installed identities, native phase accounting and test logs are indexed by evidence-index.json in the benchmark root. Thor-owned start, memory, native log and cleanup records reside in its matching benchmark directory.

Current decision: retain the production instruction path, locks, model selection and conservative capacity fallback. The controlled scheduling benefit is real; sufficient production conversational correctness and responsiveness are still unproved. Earlier small-model failures remain unresolved by today's scheduler measurements.

