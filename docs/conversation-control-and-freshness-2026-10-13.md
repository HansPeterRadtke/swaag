# Conversation control, freshness, and independent reading — 2026-10-13

## Scope and decision

This continues the [October 12 investigation](conversation-model-feasibility-2026-10-12.md), starting from SWAAG `48da5fc`. It fixes demonstrated message-delivery and stale-answer defects and tests an independent read-only conversation candidate. It does **not** approve a small model as the authority deciding which finalized user instructions reach orchestration.

Three production corrections are supported by focused reproductions and live evidence:

1. Finalized ordinary messages enter the existing durable control inbox **before** waiting for the session execution lock. An active planner can reject a proposal compiled before a correction arrived. Pending receipts are drained in durable order, not file-lock acquisition order. A message already reconciled by the active planner reuses that run's committed response, instead of starting the same work again.
2. Before publishing a direct reply, the service checks the durable task facts again. One changed snapshot triggers one fresh generation. A second change fails explicitly rather than returning a known-stale answer. Completed results, failed results, plan revisions, assessed progress, questions and revisions participate; heartbeat ages and history bookkeeping do not.
3. Response-constraint repair and independent completion checking receive current user authority, including corrections. Completion evidence is partitioned at the active **run** boundary, so a correction becoming a user message does not accidentally turn all of the interrupted run's inference logs into historical evidence requiring recursive summarization.

The single-writer execution lock remains. No small-model prompt or new process topology is promoted. A read-only conversation model can overlap orchestration, but every finalized input must remain independently durable and available to orchestration; the small model must not be allowed to discard restrictions.

## Environment and evidence

All task effects were isolated to experiment session roots. The only live action tool enabled in the control/overlap probes was `calculator`. No deployments, shell actions, file-writing tools, or production worker tasks were authorized by these probes. Model response caching was disabled; native KV prefix reuse was not disabled.

Evidence root: `/data/var/swaag-benchmarks/conversation-control-study-20261013` on Jetson. It contains exact scripts, JSONL requests and responses, streamed events, canonical SWAAG histories, configurations, timings, failure logs and an integrity index. The auxiliary experimental model's service journal is on Thor. Infra remained at `c4cd308`, with its unrelated `hosts/jetson/config.json` edit preserved.

Thor's existing Qwen3.8 Q3/two-sequence and BF16/one-sequence services retained their original processes. The temporary Qwen3.5-9B Q5 service was admitted with 18.1 GiB available, a 10.1 GiB estimate and the unchanged 4 GiB reserve. It was stopped only after its experiment ended, its PID was checked and its slot was idle.

A locally installed Qwen2.5-VL-3B Q4 candidate was inspected. The canonical guard refused its unknown `qwen2vl` cache layout and requested a measured cache reservation. No model started, no reservation was invented, and the guard was not changed. Its unrelated game-owned service also has a disabled execution override; that service was not altered.

The Qwen3.5-9B Q8 experiment used the same configured llama.cpp binary path, 32,768 context, one slot, q8_0 KV, Flash Attention on, batch 1,024, microbatch 256 and fourteen threads as the Q5 study. A direct command was admitted but could not initialize CUDA under the command user's execution context and fell back to CPU; that process was stopped without benchmarking it. A temporary systemd service then ran as the normal model user with the canonical memory guard in `ExecStartPre`. Admission observed 18.2 GiB available, estimated 13.1 GiB and preserved the 4 GiB reserve. Sampled available memory remained above the reserve. Q8 `/props` reported build `b9842-6f4f53f2b`; do not substitute the different build identity recorded in the October 12 report. The Q5 build identity was not independently captured in this session, so a strict cross-build performance comparison is not claimed.

## Semantic decomposition experiment — not promoted

`extract_probe.py` tests a different contract: identify operative work instructions first, then whether remaining parts need tools, whether the answer uses runtime facts, and the answer. Routing is mechanically derived from the model's declared instructions/tool requirement. The complete original utterance remains the proposed forwarding payload; extracted clauses are not replacements for it. These are synthetic-state calls and execute no orchestrator actions.

The eighteen old cases were retained from the previous study. Sixteen new cases were written before running the new prompt and include implicit prerequisites, revoked permission, negated hypotheses, quotations, answer-format instructions, priority changes, conditional cancellation and multiple negation. Greedy seed forty-two measures repeatable behavior, not a statistical reliability bound.

| Model | Cases | Correct derived route | Median full response | Median content TTFT |
| --- | ---: | ---: | ---: | ---: |
| 9B Q5, old cases | 18 | 18 | 5.710 s | 0.953 s |
| 27B Q3, old cases | 18 | 17 | 9.470 s | 2.550 s |
| 9B Q5, new cases | 16 | 15 | 3.151 s | 0.595 s |
| 9B Q8, same new cases | 16 | 15 | 6.762 s | 0.930 s |

The Q3 model treated a quoted prohibition as an operative command. Both 9B quantizations missed: “I want the test results before you touch the deployment. Are they ready?” They answered status without declaring the prerequisite. The failure therefore persists in the tested higher-precision deployment; lower-bit quantization alone does not explain it. This is not a general claim about all quantization effects.

Answer quality also fails independently of routing. Examples include inferring past user instructions from missing history, answering a negative hypothetical as a positive one, claiming an instruction was recorded without execution evidence, and violating audio presentation rules. Several forwarded cases proposed a nonempty answer despite the contract; such an answer must be discarded, not spoken as verified fulfillment. Route counts are not quality pass counts.

The old Q5/Q3 matrices overlapped on Thor; the Q8 matrix overlapped the completion experiment, while the new Q5 matrix was less contended. These are bounded latency observations, **not** a controlled quantization throughput ranking. Native timings and complete stream events remain in the JSONL files. Reasoning-mode sentinel evidence is retained separately; reasoning is not assumed from a request flag alone.

## Timing attribution and server capacity

`analyze_phases.py` derives `phase-analysis.json` from the retained streams and canonical histories. The synthetic prompts contain 502–515 input tokens. The median retained KV prefix is 285 tokens; response-cache disablement must not be described as cold-prefill measurement. Native timing aggregates are:

| Matrix | Token-weighted new-prompt throughput | Token-weighted decode throughput | Median prompt processing | Median decoding | Median client-minus-native time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9B Q5, old | 316.64 tok/s | 14.33 tok/s | 0.709 s | 4.739 s | 0.229 s |
| 9B Q5, new | 601.05 tok/s | 31.21 tok/s | 0.366 s | 2.550 s | 0.227 s |
| 9B Q8, new | 335.11 tok/s | 12.37 tok/s | 0.697 s | 5.770 s | 0.231 s |

These differences include load and prompt/output differences, not only quantization. VLLM did not return native phase timings in these streams; do not substitute client TTFT for pure prompt-evaluation time.

The successful completion-enabled cancellation has five serial model responses totaling **117.664 s** of **125.679 s** overall, about 93.6 percent. Local scheduler queue waits total **0.312 s**. The remaining approximately **7.703 s** includes setup, prompt construction/tokenization, durable I/O, other network requests and verification; it is not a measured pure-Python overhead category. The action budget estimates were 2,972 and 3,102 input tokens with TTFT 12.691 and 12.840 s. The repair estimated 647 input tokens and took 9.715 s; the completion check estimated 2,185 and took 17.812 s. Those vLLM prompt budgets are explicitly marked inexact, and native prompt-token counts were absent. The independent reader used an exact 1,879-token prompt; its model request took 9.184 s inside the 13.370 s reader operation. Overlapping model durations must not be added as serial wall-clock phases.

The authoritative Infra Q8 report (`docs/llms/measurements.md`) records historical token-weighted decode throughput of 24.209 tok/s and prompt throughput of 1,024.703 tok/s over heterogeneous phases. Its median prompt was 56 tokens, and the documented benchmark lifecycle used one benchmark model at a time. The matrix records batch 256 versus this study's 1,024, and exact build/load equivalence has not been established. The shared-hardware Q8 results here therefore do not establish a throughput regression relative to that archived campaign. Infra explicitly marks historical Qwen quality scores provisional and requires both reasoning-mode sentinels before a quality ranking. No such ranking is approved here.

A final provenance check established that the Q3 server is genuinely vLLM, advertises the expected model and 8,192-token context, and runs with `--max-num-seqs 2`. Its standard chat API does not expose llama.cpp `/props`. SWAAG intentionally admits **one** request for that transport, but previously labeled this conservative fallback as `server_props:total_slots`. That misleading provenance reproduced with both cached and uncached clients. The model client, cache wrapper and scheduler now propagate the actual source (`conservative_fallback:openai_transport`), preserving the existing admission limit. Native llama discovery remains reported as server properties. Original experiment histories are retained unchanged, with this correction documented; they are not retroactively rewritten.

Consequently, a server supporting two sequences does not prove that ordinary SWAAG calls use both. Independent backends can overlap as observed, while same-backend admission and the conversation session lock remain distinct constraints. Any increase in same-backend admission requires a separately validated capacity contract and workload test, not an invented value inferred from the model name.

### Reasoning-mode sentinel limit

The retained Q8 arithmetic sentinel returned the correct JSON integer with thinking disabled in 1.152 s. With thinking enabled it exposed reasoning but produced no final answer before either the 256-token limit (10.092 s) or 768-token limit (29.474 s). These are observed mode-switch behavior and failed completion checks, not a dual-mode qualification. The native request and stream evidence are in `q8-template-sentinels-retained.jsonl`.

## Finalized-input delivery and recovery

The pre-change deterministic reproduction blocked an actual planner action, submitted a finalized cancellation through `CommunicationService.orchestrator_message`, and found that the correction never reached the planner's pending-control inbox while it waited on the session lock. The correction had no opportunity to invalidate the pending action.

The corrected path reuses the runtime's existing control-generation check. It does not infer user intent using keywords and does not remove locking. Accepted front-door controls become exact user messages in receipt order, with an internal receipt identity. Already-materialized controls are not reinjected into the “later intervention” channel on future turns, where an old restriction could otherwise wrongly supersede a newer instruction. Ordinary worker controls retain their existing semantics.

Tests cover cancellation during a planner call, multiple ordered corrections, a later explicit instruction lifting an earlier restriction, recovery after gate connection loss and after action connection loss, and no duplicate user message/action when a pending receipt is resumed. A processed receipt without a committed response is reported as such; a later unrelated turn is not evidence that the earlier one completed.

This is **not** an exactly-once guarantee for arbitrary external effects or retransmitted HTTP requests. The API still accepts a text message without a caller-supplied idempotency key; a fresh call creates a fresh receipt. Pending receipts survive a failed run and are drained on the next ordinary request. Autonomous startup draining and end-to-end voice/network replay remain separate acceptance work.

### Real model cancellation

`live-control-q3-b` used the existing Q3 vLLM backend through the ordinary service method. The first action request began at 13.991 s; the cancellation arrived at 13.994 s. The runtime rejected one stale action, executed **zero** calculator calls, preserved both user messages and returned the same reconciled response to both callers. Total time was 90.975 s, not an interactive latency pass. Calls included the gate, the discarded action, reconciliation and an unnecessary model-invented word-count repair.

With normal completion evaluation enabled, an intermediate implementation exposed a regression: converting the correction into a user message moved the evaluator's historical boundary into the active run. It recursively projected its own inference logs. Six completed projection calls were observed before the isolated run was deliberately canceled at 402.688 s. Cancellation used SWAAG's durable run-cancellation API. No tool ran. This is an **intermediate implementation failure**, not an original-production timing baseline or a successful completion test.

After partitioning evidence at `turn_started`, `live-control-q3-evaluation-fixed` completed in 125.679 s with the gate, two action calls, one word-count repair and **one** completion evaluation. There were no evidence-projection calls, one stale-action rejection and zero tool calls. The completion checker saw the finalized correction. Concurrent load and different generations preclude subtracting these runs as a precise speedup estimate, but elimination of the recursive projection calls is directly observed.

Word-count metadata hallucination remains a measured cost. Its repair now sees current authority, but required constraint checking was not globally disabled to make a benchmark faster.

## Snapshot freshness

The regression test initially returned “Verification is still running” after the worker had completed during generation. The new guard rejected that draft and regenerated from the completed record. Separate tests cover progress changes without a worker-status transition, continuously changing state, and bookkeeping that must not trigger needless regeneration.

`live-freshness-q5` used a real model with a controlled durable worker transition after prompt construction. The first draft was rejected; the second answered that all twelve checks passed with exit code zero. Total time was 18.088 s, with two model calls. The answer unnecessarily exposed the test marker, a remaining presentation-quality failure.

The check is an optimistic comparison of per-source durable task observations, not an atomic transaction across stores. State can change again after the final read. Supervision remains timestamped sampled evidence. The conservative guard can refresh a conversational answer even when the changed task is irrelevant, and under repeated change it returns an explicit error instead of an unverified answer. A narrower dependency contract requires further semantic reliability evidence.

## Independent read-only conversation prototype

`live-overlap` ran the Q3 planner and a separate 9B Q5 read-only gate, while the same finalized status question also entered the actual durable planner inbox. It did not add a production configuration switch or change ordinary API response delivery.

The planner's first action started at 16.992 s. The status request arrived at 16.996 s; the reader started at 17.035 s and returned at 30.405 s, about **13.37 s** later, while both planner callers remained active. Its answer accurately reported the earlier verifier's twelve passing checks, and the relevant task snapshot was unchanged at the recheck. The ordinary planner response returned at 141.790 s, about **124.79 s** after the status request. One authorized calculator call executed, with result 30,003; the status question did not cancel that original task.

This demonstrates independent response generation alongside real orchestration with durable input delivery. It does not establish production conversation-history reconciliation, speech streaming, TTS timing, long-run tail latency or safety of speaking an unverified partial answer. The status question also invalidated one in-flight planner draft because every newly received input is conservatively treated as potentially relevant authority; responsiveness of the separate reader does not mean free background throughput.

## Remaining architecture gates

The tested small-model semantic gate still cannot decide reliably which restrictions may bypass full orchestration. The existing synchronous ordinary API still waits for the single-writer planner; the independent reader is experimental. An idle planner can still misclassify a restriction while workers operate, and this patch does not provide a global atomic fence around every independent worker or plan admission. Receipt durability and active-planner rejection are specific, verified improvements, not a solution to all background-control routing.

The default Jetson backend remains unavailable under its existing memory-admission refusal. No production workload was evicted, shrunk or restarted to change that. Android/TTS end-to-end behavior was not exercised. Model selection remains a measured research decision, not a mandated split into a particular number of models or processes.

## Validation and deployment record

Functional changes are committed as `93c8857`; the subsequent capacity-provenance correction is `18af5ba`. The canonical noneditable runtime installer completed successfully after both code changes. All **190** tracked source-package files match their installed counterparts byte for byte; `installed-verification.json` records the final code revision and hashes. The installed Python import and installed CLI help smoke checks succeeded.

Validation scope is deliberately explicit:

- Full deterministic suite: **1,140 passed, 3 optional skips** in 1,129.31 s. This run covered receipt/freshness/current-authority changes but started before the final completion evidence-boundary refinement; it is not an exact-final-revision full-suite certification.
- Final completion-boundary/control/worker checks: **50 passed** in 141.61 s after that refinement, including exact Unicode authority and preservation of multiple corrections.
- Installed-package receipt/snapshot/installer checks: **16 passed** in 32.93 s, with imports explicitly verified from the production package rather than repository source.
- Capacity-provenance/model-context/inference/preemption checks: **85 passed** in 73.33 s after the final telemetry correction; both cached and uncached reproductions had failed beforehand.
- Installed-package native/cached/uncached capacity checks: **3 passed** after the final installation. Source identity was verified again.

Earlier focused failures are preserved as reproduction or fixture-development evidence. Live cancellation, completion checking, stale-response regeneration and independent reading are described above with their actual scope; none is converted into a broad architecture acceptance claim.

At cleanup both temporary 9B model services were inactive, both existing Thor production model PIDs were unchanged with zero restarts, and `/proc/meminfo` reported approximately 18.4 GiB available. `final-thor-state.json` retains the service, memory and actual Q3 backend identity evidence. The unrelated Infra modification remains untouched. Raw logs, scripts, manifests and canonical histories are covered by `evidence-index.json`; failed attempts and the intentionally canceled intermediate experiment are retained and not counted as passes.
