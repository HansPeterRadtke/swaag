# User-facing conversation model feasibility — initial live study, 2026-10-12

This document preserves the initial study and the later controlled follow-up. The initial sections describe the earlier no-new-model, no-code-change pilot. The follow-up below includes guarded small-model admission, real orchestration experiments, and a verified snapshot correction; it is not an architecture recommendation or deployment approval.

## Environment and safety

Jetson client queried Thor's already-running vLLM `qwen38-q3-sched2` (Qwen3.8-27B UD-Q3_K_XL GGUF; two admitted sequences; endpoint 10.8.0.7:14834). This is a **quantized large model**, not a genuinely small model, and must not be described as proof that a small assistant is adequate. Thor had about 19 GiB `MemAvailable` before the pilot, with 122 GiB total and 103 GiB used. The GPU memory query returned N/A/permission errors; host memory is unified, and `free` alone does not certify safe additional model admission. Jetson had approximately 18 GiB available. No model was started because Infra requires guarded admission and a reserve.

## Reproducible pilot

Retained exact prompt, state, user text, responses, usage and timings: `/data/var/swaag-benchmarks/conversation-model-study-20261012/pilot.json` and `pilot-compact.jsonl`. The compact script is `run_pilot.py` in that directory. State was a deliberately synthetic authoritative snapshot: worker A running tests at 78%, no exit code, worker B idle, deployment not started, orchestrator managing tasks, no verified completion. No tools or orchestration actions were actually executed.

First unconstrained prompt: status/identity question with hesitations, 186 prompt tokens, capped at 220 completion tokens. Latency 22.533 seconds; output leaked internal reasoning and was truncated without a usable JSON response. Failure.

Compact prompt with `chat_template_kwargs.enable_thinking=false`, 130 completion-token cap: status question returned correct pending-test answer, no forwarding, in 8.005 seconds. Task question returned a sensible forwarding decision in 8.350 seconds but **rewrote the user instruction** instead of preserving its exact original wording, so it fails exact-instruction fidelity. Identity/state question returned correct basic state in 8.412 seconds, but surrounded JSON with markdown fences, so it fails strict machine-JSON parsing. None of these outputs were mechanically schema-constrained; the pilot cannot establish structured-output reliability.

## Findings and unresolved gates

The large quantized model can produce useful brief answers with a compact state prompt in roughly eight seconds on this endpoint under these conditions, but that is not sufficiently fast to assume voice-interactive suitability, and the unconstrained-thinking variant was much slower and failed. The experiment exposes two real correctness issues: exact forwarding fidelity and JSON-format reliability. The next controlled matrix must compare genuinely small candidates and quantizations, prompt variants, server-enforced JSON schema, full state snapshots, ambiguous multi-intent spoken language, authority conflicts, stale/missing status, safety-sensitive actions, tool-call choice and arguments, real orchestration dispatch, simultaneous background worker inference, repeated seeds, first-token and full-answer latency, and host/GPU memory headroom. Record whether TTS can start from safe partial output while the durable user instruction is dispatched, with exact event timestamps. Never claim actual side effects from this pilot. Do not change architecture or promote a candidate until those gates pass.

## Follow-up structured-output and spoken-input probes

A follow-up ran nine retained cases (one repeated with a changed prompt) on the existing Thor Q3 endpoint, with thinking disabled and OpenAI-compatible `response_format: json_object`. The median full-response latency was 8.209 seconds, range 5.042–12.060 seconds, and every response parsed as JSON. The raw results are in `structured-study.jsonl` alongside the initial pilot. This establishes only syntactic JSON validity, not the required field types or correct semantic routing.

Concrete failures: the model failed to forward a conditional deployment instruction; failed to forward a no-deployment prohibition; returned `forward_to_orchestrator` as a descriptive string instead of a boolean on one case; and sometimes paraphrased the user's command in `instruction_exact`. It also claimed to have forwarded a restart without any actual orchestrator invocation. A later correction case preserved the exact text, but no deterministic dispatch occurred. These are disqualifying for production as a trusted instruction router.

No genuinely small model was loaded: Thor `MemAvailable` was about 19 GiB with two production vLLM services active, and the guarded start command for the installed nine-billion-parameter candidate was blocked by the tool safety layer before execution. Infra's reserve and no-eviction rules were respected. The attempted independent communication capability invocation was also blocked before execution. Further tests must use the repo's real constrained schema and an actually available smaller endpoint, with explicit verified forwarding and concurrent worker activity.

## Extended thirty-two-case prompt/state matrix

Retained reproducible script and full request/response/usage records under `/data/var/swaag-benchmarks/conversation-model-study-20261012/extended_matrix.py` and `extended-matrix.jsonl`. The matrix contains two system-prompt variants (minimal and strict), three synthetic state representations (short, contradictory, longer repetitive), and eight natural spoken-style user utterances. Thirty-two completed cases, median full-response latency 8.548 seconds. All thirty-two returned parseable JSON with correct field types. In cases routed to the orchestrator, exact utterance preservation was successful. However, both prompt variants **failed to forward the explicit prohibition** `No, do not deploy anything yet, just tell me how far the tests got.` The apparent exact-forward pass count is misleading for those cases because an empty string is allowed for nonforwarded messages. Route correctness must be scored independently from field typing and exact preservation. This is a demonstrated semantic routing failure despite syntactically valid output.

The synthetic state is supplied with each user request; the experiment does not validate real-world freshness, atomicity, state changes during generation, or worker-action concurrency. Model is Qwen3.8-27B Q3, not a small candidate. No production architecture changes were made.

## Controlled small-model and ordinary-entry investigation — 2026-10-12, 19:05 UTC onward

This follow-up supersedes the earlier inability to start a small model. It does
not approve a new conversational architecture. The source baseline was SWAAG
`41d1db3`. Existing Infra edits were preserved. The current production-default
Jetson model address had no listener during this investigation; historical
Jetson timing is therefore not a current control measurement. Tests used Thor's
observed live endpoints and isolated SWAAG histories, never production tasks.

### Admission and runtime configuration

Thor initially reported approximately 19.3 GiB available out of 122 GiB unified
memory. The canonical guarded start admitted the previously inactive
`llama-cpp-qwen3-5-9b.service`, accounting for its approximately 10.1 GiB requirement
and the 4 GiB reserve. The service loaded successfully, with no restart or
eviction of existing workloads. Available memory after actual model activity
remained above the reserve (8.7 GiB at 19:30 UTC). This is an observed admission,
not permission to bypass future checks.

| Candidate | Runtime and relevant settings | Endpoint |
| --- | --- | --- |
| Qwen3.5-9B Q5_K_M | 8,953,803,264 parameters; 6.13 GiB weights; native llama.cpp commit `86b94708`; 32,768 context; one slot; q8_0 KV; flash attention; batch 1024 / microbatch 256 | Thor 14831 |
| Qwen3.8-27B UD-Q3_K_XL | 12.24 GiB GGUF weights; vLLM `0.28.1rc1.dev233+gcdefd9d49`; 8,192 context; two sequences; 1 GiB KV allocation | Thor 14834 |
| Qwen3.8-27B BF16 | Same observed vLLM build; 8,192 context; one sequence; 1 GiB KV allocation | Thor 14833 |

Infra advanced concurrently from `802c9a5` to `c4cd308` during the study. The new
workspace/forge guidance and related diffs were read; no unrelated Infra content
was changed. The final Infra dirty path remained the pre-existing Jetson host
configuration. Environment observations are retained in `jetson-environment-final.json`
and `thor-environment-final.json`. These are samples, not continuous peak-memory
measurements.

The three logical responsibilities are not equivalent to three configured
models. Currently `communication.model_base_url` selects both the immediate gate
and the full user-facing orchestrator, as well as the optional specialized
status assistant. Merely pointing that setting at a small model also assigns
it full plan construction. Separating only the gate is an unimplemented
candidate, not what the ordinary-entry test below measured.

### Reproducible schema and routing matrix

`scripts/benchmark_conversation_models.py` extracts the actual production gate
prompt and uses the actual `orchestrator_interaction_contract` JSON Schema.
Requests use server-enforced `json_schema`, thinking disabled, temperature zero,
a 192-token output cap, and streaming. Every request, full response, native
timing/usage field, stream event, seed, prompt and state is retained under
`/data/var/swaag-benchmarks/conversation-model-study-20261012`.
The manifest records the source head and exact prompts; an existing evidence
file is never overwritten. These calls do **not** dispatch tasks.

Nine baseline cases cover greeting, current progress, negative instructions,
conditions, correction within an utterance, mixed question/action, quotation,
hypothesis and a pure status question. Nine held-out cases cover a continuing
hold, revoked approval, review before publication, priority, an untrusted quoted
log, an undecided plan change, an exact Unicode filename, a history question,
and a prohibition on further commands. “Held out” means not used to author the
initial compact variants; subsequent analysis does not make them unseen again.

| Model / prompt / state | Calls | Valid schema | Correct route | Median full response | Median TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9B / production / compact baseline | 9 | 9 | 8 | 4.264 s | 1.045 s |
| 9B / appended restriction guidance / compact baseline | 9 | 9 | 8 | 3.977 s | 0.958 s |
| 9B / shortest compact / compact baseline | 9 | 9 | 9 | 2.464 s | 0.736 s |
| 27B Q3 / production / compact baseline | 9 | 9 | 7 | 9.334 s | 4.367 s |
| 27B Q3 / compact2 / compact baseline | 9 | 9 | 9 | 9.094 s | 2.918 s |
| 9B / compact2 / all, two greedy seeds | 36 | 36 | 26 | 3.467 s | 0.775 s |
| 9B / compact2 / pretty JSON held-out | 9 | 9 | 7 | 3.281 s | 0.812 s |
| 27B Q3 / compact2 / compact held-out | 9 | 9 | 8 | 8.750 s | 2.915 s |
| 27B BF16 / production / compact all | 18 | 18 | 13 | 14.908 s | 1.156 s |
| 27B BF16 / compact2 / compact all | 18 | 18 | 17 | 13.571 s | 0.740 s |

A route pass is not an answer-quality pass. The shortest prompt's nine correct
routes conceal a wrong explanation of a quoted instruction. Compact2 still
elicits unsupported assertions about absent history and quoted logs, unnecessary
status details in greetings, and numeric/machine-noise violations of audio
style. The 9B model repeatedly drops five classes of work-control instructions:
the original prohibition, continuing hold, revoked approval, review requirement
and priority change. Its explanation often treats the restriction as already
satisfied by current state, although a new instruction must reach authoritative
task control even when the present state happens to agree.

The Q3 production prompt also returned `respond` while claiming to route a
restart. This violates the execution contract independently of JSON validity.
The current defensive prose matcher does not detect that claim. Expanding
keyword matching is not a demonstrated semantic solution.

Compact2's Q3 held-out extension passed eight of nine routes: it still dropped
the review-before-publication requirement. The BF16 candidate also dropped that requirement with compact2. Its production
prompt missed five control cases and claimed it had kept deployment on hold
without dispatch. No tested prompt has been promoted.

The pretty-state comparison preserved identical facts and changed JSON
whitespace. It corrected two routing errors but retained two; compactness did
not establish a quality advantage. Token counts and KV reuse differ. In the
first production 9B request, 797 prompt tokens took 1.095 s, about 728 tokens/s;
82 output tokens decoded at about 29 tokens/s. This differs fundamentally from
historical Jetson 27B throughput. It is not evidence of a software regression
on Jetson or a pure quantization ranking.

Temperature-zero seed repeats are repeatability checks, not independent
sampling robustness. Native llama.cpp reused prompt prefixes across some
requests. “Uncached” below means no SWAAG response replay; it does not mean
disabled backend KV caching. The sample counts are too small for a production
tail-latency or broad reliability claim. vLLM did not return per-request native
prefill/queue timing in these chat responses; TTFT includes both and transport.
The BF16 candidate has faster TTFT but slower full responses than the Q3
candidate here. Output length, weight format, admitted-sequence limits, backend
implementation and observed load prevent a universal throughput ranking.

### Reasoning and template/parser qualification

The final bounded 9B reasoning-on probe used compact2, the same schema and five
selected failure/authority cases, with the output budget increased from 192 to
768 tokens. It produced only three valid JSON responses and one correct route.
That one route pass still gave a wrong explanation of the quoted log. Two cases
spent all 768 tokens in reasoning, returned no answer content, and ended with
`finish_reason=length` after about 25 seconds. Revoked permission and changed
priority still went to `respond`. For the three completed responses, answer
content first appeared after 8.1–10.6 seconds; the first reasoning token
appeared much earlier. Reasoning therefore did not repair the sampled failures
within this bounded latency budget. Evidence: `q5-thinking.jsonl`.

The methodology's THINKING-ON/OFF requirement was checked explicitly with
`template_sentinels.py` and `template-sentinels.jsonl`. All three endpoints
returned correct constrained JSON with reasoning off. The 9B endpoint also
exposed separate reasoning followed by valid JSON with reasoning on. Both vLLM
endpoints returned valid JSON but did **not** expose the requested reasoning
phase, so those reasoning-on mode sentinels failed.

The observed vLLM launch commands have no reasoning-parser option. The actual
checkpoint template opens a thinking block unless `enable_thinking=false`;
the installed vLLM structured-output manager's `should_fill_bitmask` applies
the grammar immediately when no reasoner is configured. Those observations
support a parser/configuration explanation. No production service was restarted
to test a different parser. Consequently the Q3/BF16 results are bounded
THINKING-OFF deployment observations, not a fully qualified Qwen model-quality
ranking under Infra's dual-mode methodology. The successful JSON parser alone
does not certify reasoning-mode conformance.

### Actual independent-backend benchmark

The correct task, `capability_communication_independent_model`, passed in
`independent-q3-q5-identified/benchmark_results.json`. The worker used the Q3
vLLM endpoint with the exact served-model identifier and OpenAI transport;
the assistant used the 9B llama.cpp endpoint. Earlier setup attempts are
preserved separately: an invalid benchmark mode, missing `/v1` transport,
and a mismatched served-model identifier failed before valid inference.
Those are configuration failures, not model-quality results.

Canonical history proves separate backends and real overlap:

| Event | UTC / elapsed |
| --- | --- |
| Worker action sent / received | 19:15:16.888 / 19:16:03.310 |
| Assistant status sent / received | 19:15:23.347 / 19:15:43.369 |
| Full status operation | 26.717 s |
| Assistant model elapsed / TTFT | 19.720 s / 5.555 s |
| Worker resumed/completed after status | 37.143 s |
| Preemptions / replays | 0 / 0 |

The assistant compiled 2,359 input tokens and generated 190 output tokens.
Native prefill was 5.038 s and decode was 14.185 s, about 13.4 tokens/s. A later
idle replay of the same request generated the same number of tokens at
29.2 tokens/s, taking 7.335 s overall. That idle request reused 2,099 prompt
tokens, so its total-time difference combines KV reuse and contention; the
decode-throughput comparison independently demonstrates contention.

The main worker also made a 1,253-input-token completion-evaluation call taking
14.410 s. This occurred outside the status-answer critical path. The generic
worker completion check is intentional correctness policy; disabling it globally
is not justified by this one benchmark. The ordinary direct-answer gate already
avoids that worker loop.

This benchmark exercises the specialized status interface. It is **not** proof
that ordinary `orchestrator.message` can respond independently.

### Real ordinary-entry trajectory

`ordinary_live.py` and `ordinary-live-q5-configured/` retain the actual
CommunicationService trajectory. Configuration assigns the 9B model to the
gate and full orchestrator, Q3 to workers, disables response caching and external
action tools, restricts the workspace to the experiment, and enables only
orchestration control. Completion evaluation is explicitly disabled here to
match the existing bounded orchestration benchmark; it remains enabled in the
independent-backend benchmark above. An initial script configuration error
occurred before model invocation and is not counted as an experiment result.

The fixed snapshot allowed a correct completed-verification answer in 12.555 s.
The next real user message prohibited further work while asking for status.
The gate answered the question without entering orchestration. Its exact text
was recorded in conversation, but no authoritative control update was applied.

A later explicit request superseded that hold and asked for one harmless worker
returning an exact marker. The full orchestrator eventually created the correct
plan and worker, and the worker returned `ORDINARY DISPATCH VERIFIED`.
That is a real dispatch and durable completion, not a simulated forwarding
claim. Recovery costs were substantial: an action proposed the wrong worker
objective and `continue_loop=false` despite tool calls; the runtime rejected it.
Two actions invented an exact one-hundred-word response constraint, requiring
two independent repairs. A subsequent tool call put `completion_mode` in an
invalid location and was rejected before effects. The model repaired its call,
created/started the plan, then generated a final reply. No invalid draft caused
external effects.

The request took 166.883 s to return its initial plan-start answer. Worker
execution began 127.276 s after that request arrived, while the orchestrator
continued generating its final response. Thus useful work **can** overlap that
response. It does not imply fast initial acknowledgement.

A status question arrived immediately after the worker became active. It waited
39.606 s for the earlier session lock and returned after 59.601 s total. The
worker completed at 19:29:06.185, but the answer returned roughly ten seconds
later still describing it as working. This is observed stale-state delivery
during generation, not merely a hypothetical race. Exact timings, messages,
tool validation errors, inference histories, worker records and completed plan
are preserved.

### Verified snapshot correction and remaining consistency boundary

Before the fix, the immediate gate excluded terminal workers and plans, and
discarded the progress-bearing `workers` portion of the already-computed question
inventory. A regression test failed because the model prompt lacked the actual
completed result and failure evidence. The correction includes terminal records
and the full inventory, exposes collection timestamps and the per-source
consistency limitation, and collects after any preemption wait immediately
before the gate call. It does not truncate history or replace exact user text.

Three focused regression tests establish terminal result/error inclusion,
history-derived progress inclusion, and a completion transition during the
preemption wait. They pass. The existing preemption and question/progress tests
also passed in the forty-four-test focused run. Full release validation is
recorded below.

`lock-and-freshness/result.json` additionally uses controlled inference with
the actual service and stores. A finalized second instruction cannot enter its
model call while the first holds the execution lock. A worker transition during
the first model call is absent from its already-built prompt and present in the
second. Single-writer ordering was preserved; the fix does not claim an atomic
cross-store snapshot or freshness at response delivery.

### Framework overhead and integration limits

Two profiled ordinary status interactions took 10.787 and 9.824 s with about
5.55 and 5.86 s in the model call. The first used ten exact-tokenization HTTP
requests totaling 2.582 s and five metadata requests totaling 1.402 s. The second
used five tokenization requests totaling 1.875 s and four metadata requests
totaling 1.103 s. Backend activity polling overlaps inference and must not simply
be added to wall time. Profiles and HTTP timings are in `latency-profile/`.
The Jetson deterministic regression suite was running during profiling; these
are diagnostic measurements, not uncontended throughput certification.

A paired ten-iteration tokenization transport probe returned identical counts.
Median fresh-connection latency was 0.298 s versus 0.161 s with connection reuse.
This supports investigating scoped connection reuse or batching, with lifecycle
and thread-safety tests. It does not justify stale backend metadata or removal
of exact context accounting, and no transport change was installed.

SWAAG's ordinary handler delegates through a thread and contains no TTS playback
wait. Plan execution can already continue while its response is generated.
Actual microphone, finalized-turn transport, speech playback and interruption
latency were not certified: the observed Thor voice gateway was inactive and
its configured foreground endpoint had no listener. Its historical deployment
configuration is not evidence of current user-visible behavior. No Android
redesign or voice service change was made.

### Release verification

The snapshot correction is committed as `1e77611` and installed using the
repository's standard runtime installer. All 190 tracked package files match
the installed package byte for byte. Both installed smoke tests and three
direct installed-snapshot checks pass; evidence is in
`installed-snapshot-check/verification.json` and `install-snapshot-fix.log`.
No production model process was restarted for package installation.

The deterministic suite completed with **1,129 passed and 3 optional skips** in
1,045.50 seconds. The suite began before the final timestamp-order cleanup
(which places capture time after reading supervision); the three snapshot
regressions were rerun after that cleanup and passed in 4.99 seconds.
`final-regression-suite.log` and `snapshot-final-tests.log` preserve the results.
No experimental prompt, backend configuration, connection pooling, lock
removal, or automatic new work policy was deployed.

### Decision and next acceptance boundary

The small candidate is feasible to load safely and measurably faster on short
conversation requests, but it is not reliable as the sole authority deciding
whether a finalized user instruction reaches task control. Larger models also
show semantic routing failures. Retain every failure and evaluate control
receipt, later-message ordering and pending action invalidation independently
of conversational prose. The existing lower-level worker control queue is a
useful foundation, not proof of an equivalent ordinary-entry guarantee.

The data supports repairing the snapshot omission now. It does not select a
separate-model architecture, remove session locking, promote a compact prompt,
or approve the small model as the full orchestrator. Still-open acceptance
includes durable receipt before long conversation waits, superseded-action
prevention through the ordinary entry point, authoritative state changes during
generation, additional adversarial/state-conflict and stochastic holdouts,
same-weight scheduling controls, and actual finalized-voice/TTS timing.

### Final operational state

The temporary 9B service was returned to its original inactive state after all
experiments. Cleanup verified the exact process started by this study and an idle
inference slot before stopping it. Thor then reported 18.8 GiB available. The
pre-existing BF16, Q3 and vision-model process identities remained unchanged.
The guarded start remains the required mechanism for any future load. Evidence:
`temporary-model-cleanup.txt` and `existing-model-processes.json`.

The Jetson default endpoint outage was traced, rather than assumed to be a
model-performance defect. Its service had failed at 16:52 UTC, before this
investigation, with exit status 75 because the admission guard calculated a
16.5 GiB footprint plus 4.0 GiB reserve against only 18.4 GiB available, a
2.1 GiB shortfall. It explicitly started no model and stopped nothing. Current
Jetson headroom was still below that combined requirement. No guard was bypassed,
no existing workload was evicted, and the unrelated host configuration was
preserved. `jetson-backend-admission-failure.txt` retains the exact journal evidence.
The installed-package verification is therefore not a claim that the default
Jetson model endpoint has been restored.
