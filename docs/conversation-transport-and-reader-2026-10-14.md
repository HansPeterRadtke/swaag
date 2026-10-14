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

## Evidence and remaining checks

Jetson and Thor retain artifacts under /data/var/swaag-benchmarks/conversation-transport-study-20261014. Jetson includes the failing transport reproduction, focused tests, exact full requests/results, HTTP timings, supervised memory trace, and scripts. Thor includes unit provenance, guarded admission, startup and guardian journals, memory trace and diagnostic outcomes. The ongoing semantic matrix compares two prompt policies on sixteen earlier diagnostic cases and eight fixed new heldouts, with complete raw failures retained. Structural validity and semantic accuracy are evaluated separately. No new reader or prompt is promoted by the transport measurement.
