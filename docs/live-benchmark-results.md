# Live benchmark results

These measurements are retained as model/server-specific evidence, not universal policy. Machine-readable artifacts remain under `/data/var/swaag/benchmarks` because they contain full runtime traces and host-specific identities.

## Communication routing - 2026-08-27

- Artifact: `/data/var/swaag/benchmarks/communication-routing-2026-08-27-qwen38-27b/communication_routing_results.json`
- Endpoint/model: `http://127.0.0.1:14829`, Qwen3.8-27B UD-Q4_K_XL, llama.cpp `b10535-8a832e4bf`, one slot, authoritative `n_ctx=131072`.
- Pair validity: assistant and strong URLs were identical because no separate small endpoint was reachable. The artifact therefore sets `routing_pair_is_distinct=false` and `routing_policy_selection_supported=false`.
- Answer quality: 4/4 cases returned the required evidence-backed facts with valid durable citations. Strict route-oracle score was 2/4: both routine cases correctly stayed local, while the same capable model answered both complex cases correctly without requesting escalation.
- Cost/latency: four model calls, 7,294 prompt tokens, 1,441 completion tokens, mean 234.115 seconds per case. All calls completed in one generation after communication output headroom moved from a fixed 512-token cap to the central soft per-kind policy; no output-starvation retry occurred.
- Interpretation: this run validates full-fidelity status context, factual grounding, citation checks, and the no-fixed-cap output path on the live model. It cannot validate small-to-strong routing policy. Selecting a communication model still requires a genuinely distinct candidate endpoint and repeated active-worker cases.

An earlier attempt began with `n_ctx=262144` and was correctly invalidated when the managed server restarted with `n_ctx=131072`; the retained result is the subsequent clean, stable-capacity run.

## Context position feasibility probe - 2026-08-27

- Partial artifact: `/data/var/swaag/benchmarks/context-order-2026-08-27-qwen38-27b-v3.json`
- Endpoint/model: the same Qwen3.8-27B endpoint and llama.cpp build, with authoritative `n_ctx=131072`.
- Exact completed case: the early-position marker at 10% of the server window passed. The server-template prompt contained 13,118 tokens, the model returned the exact code in 22 tokens with finish reason `stop`, and the recorded marker token fraction was 0.05496.
- Cost: first-token latency was 2,427.459 seconds and total latency was 2,440.171 seconds while the host also ran the complete Python suite. llama.cpp measured 2,426.152 seconds of prompt evaluation for 13,118 tokens and 12.700 seconds for 22 output tokens.
- Interpretation: desired output headroom of 15,729 tokens did not force input reduction or output bloat; the model stopped naturally. The full 15-case default matrix would nevertheless consume impractical Jetson wall time at the live 131k window, especially at 25-90% input. The run was stopped immediately after its first durable checkpoint rather than pretending an incomplete matrix was evidence about middle/late retrieval.

The v4 harness therefore records both authoritative server capacity and an optional explicit benchmark working window, and can select marker positions or semantic-layout rotations. This permits honest bounded matrices at several absolute working sizes without relabeling the server's actual capacity. Full-window cases remain available and must be reported separately from bounded-window results.
