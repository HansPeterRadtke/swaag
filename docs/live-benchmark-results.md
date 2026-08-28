# Live benchmark results

These measurements are retained as model/server-specific evidence, not universal policy. Machine-readable artifacts remain under `/data/var/swaag/benchmarks` because they contain full runtime traces and host-specific identities.

## AG-UI official SDK conformance - 2026-08-28

- Artifact: `/data/var/swaag/manual-tests/ag-ui-sdk-conformance-repo-20260828T035857/result.json`.
- Method: install exact official `@ag-ui/client`, `@ag-ui/core`, and `@ag-ui/encoder` 0.0.59 packages outside the repository; prepare a completed durable worker and stable external thread/run mapping; launch the source communication service on temporary loopback port 13402; and replay that run through the official `HttpAgent` rather than a custom SSE parser.
- Result: the SDK accepted the real POST shape, decoded `RUN_STARTED`, the complete text-message lifecycle, and successful `RUN_FINISHED`, assembled both the activity and assistant messages, and returned the exact durable result. Swaag exited zero on `SIGTERM` with empty stdout/stderr.
- Scope: this isolates protocol serialization, ordering, replay, and official-client state assembly without requiring model inference. It does not claim official live new-run, interrupt-resume, client-tool/shared-state, binary transport, capability discovery, or WebSocket conformance.

## A2A official SDK conformance - 2026-08-28

- Artifacts: `/data/var/swaag/manual-tests/a2a-sdk-conformance-repo-20260828T033843/result.json` for list/get/subscribe/cancel and `/data/var/swaag/manual-tests/a2a-sdk-new-task-conformance-20260828T095035Z/result.json` for new-task send/stream.
- Method: install exact official `@a2a-js/sdk` 1.1.0 outside the repository. The first probe seeds one non-running durable task through Swaag's transport-neutral API and exercises list, get, subscribe, and cancel. The second launches a queue-only temporary Swaag service whose injected client raises on any model access, then lets the official client create one unary and one streaming task before canceling both.
- Result: the SDK decoded the Swaag identity, card, list/get pagination, exact task/context IDs, initial-task-first streams, intermediate status changes, cancellation, and clean terminal stream closure. For new tasks, both IDs and contexts were server-generated, `SendMessage` returned submitted, and `SendStreamingMessage` yielded a submitted Task followed by terminal cancellation. The second artifact records `inference_allowed=false`, `model_client_accesses=[]`, process exit zero, and empty stderr. The initial probe also exposed and the implementation now fixes a real CLI override defect: the card advertises the effective bound host and port rather than stale configured defaults.
- Scope: this establishes official JavaScript client interoperability for Agent Card discovery plus new and existing task operations on the implemented JSON-RPC/SSE binding without model inference. It does not claim HTTP+JSON, gRPC, extended-card signatures, or non-local authentication.

## MCP official SDK conformance - 2026-08-28

- Artifact: `/data/var/swaag/manual-tests/mcp-sdk-conformance-repo-20260828T051124/result.json`.
- Method: install exact `@modelcontextprotocol/client` and `@modelcontextprotocol/core` 2.0.0 packages outside the repository, pin the official client to protocol `2026-07-28`, spawn the installed Swaag stdio adapter, and let the SDK validate/decode discovery, complete tool listing, and a real `list_files` tool call.
- Result: the SDK negotiated its modern era and exact protocol version, decoded Swaag server identity, all 28 enabled tools, text plus structured call output, and no tool or process error; Swaag stderr was empty. The reusable installer/probe live in `scripts/install-mcp-conformance-env.sh` and `scripts/mcp-sdk-conformance.mjs`.
- Scope: this establishes current official TypeScript client interoperability for the implemented stateless stdio surface. It does not claim Streamable HTTP header routing, multi-round-trip input, subscriptions, or task lifecycle through MCP; Swaag keeps those concerns outside its core task architecture.

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

The first bounded v4 matrix used an explicit 8,192-token working window and completed all nine cases after resuming from its durable checkpoints. Early, middle, and late retrieval each passed at approximately 10%, 50%, and 90% input utilization: 9/9 overall. Exact prompt-token counts were 835-837, 4,101-4,103, and 7,367-7,368 respectively. First-token latency rose from 40.975-44.451 seconds at 10%, through 309.481-314.693 seconds at 50%, to 817.276-845.687 seconds at 90%; total latency was 46.514-50.273, 317.266-322.517, and 826.099-852.570 seconds. Every result stopped naturally after 16 or 22 completion tokens despite much larger desired output headroom. The complete resumable artifact is `/data/var/swaag/benchmarks/context-order-2026-08-27-qwen38-27b-window8192-v4.json`.

This bounded matrix found no positional retrieval failure for its exact-code probe, but it is not evidence that ordering is irrelevant for semantic agent operations.

The balanced semantic-layout matrix then completed all six cyclic layouts at 50% of the same explicit 8,192-token working window. Every layout contained an exact system instruction plus task instruction, current request, conversation history, tool definitions, tool results, and retrieved evidence; each user section occupied every position exactly once. All seven fields were recovered in every case: 6/6 layouts, 42/42 field checks, and 6/6 checks at each user position. Every server-template prompt contained 4,091 tokens and every response stopped naturally after 154 completion tokens. Five uncontended first-token measurements were 307.680-353.184 seconds with total latency of 364.579-414.330 seconds. The fourth request spent 3,909.056 seconds before its first token because an unrelated single-slot model review already occupied the endpoint; it still passed after admission, so that latency is host-contention evidence rather than an ordering effect. The complete artifact is `/data/var/swaag/benchmarks/context-layout-2026-08-27-qwen38-27b-window8192-v1.json`.

Neither bounded matrix showed an accuracy reason to impose a new universal production ordering on this model. They do show that the configured 120/150/240-second defaults were not valid live-experiment timeouts for this deployment: even uncontended 4,091-token calls needed more than 300 seconds before the first token. Bespoke live benchmark commands now apply the documented 900-second final-benchmark timeout by default and expose `--timeout-seconds` plus endpoint overrides instead of relying on hidden environment settings. Selected full-window semantic cases and other models remain separate work.

## llama.cpp stream cancellation - 2026-08-27

- Artifact: `/data/var/swaag/benchmarks/backend-cancellation-2026-08-27-qwen38-27b-v1.json`
- Endpoint/model: Qwen3.8-27B UD-Q4_K_XL on llama.cpp b10535, one slot, `n_ctx=131072`.
- Method: from an idle server, admit a schema-constrained 4,096-output-token stream, request cancellation one second after `/slots` confirms that request, require the client call to raise `ModelCallPreempted`, poll until the slot is idle, check `/health`, and execute a fresh constrained recovery request.
- Result: cancellation returned to the client in 0.167 seconds and the slot was idle after 0.703 seconds. The server remained healthy. The immediate recovery call returned the required value with finish reason `stop`, first token at 5.170 seconds, and total latency 8.424 seconds.
- Repair found by the live trace: the first probe proved that llama.cpp released the slot, but urllib3 surfaced an `AttributeError` when the cancellation watcher closed the response body. The client had translated only selected public transport exception classes, so the runtime would misreport successful preemption as a transport failure. `LlamaCppClient.send_completion` now translates any close-induced ordinary exception after the cancellation flag is observed, while preserving unrelated exceptions. A deterministic regression reproduces that transport-close race.

This validates cancellation and fresh admission for the deployed server/client pair. It does not claim transformer-state suspension, token accounting for discarded prefill, parallel-slot behavior, or vLLM equivalence.

## Attachment capability discovery - 2026-08-28

- Artifact: `/data/var/swaag/manual-tests/attachment-capabilities-20260828T023257`.
- Method: invoke the production `inspect_attachment_capabilities` tool against the existing `/data/venv/bin/all2text` environment without supplying or reading an attachment, then re-read the tool's exact artifact through Swaag's durable store.
- Result: 14 provider statuses, 53 provider-family statuses, 40 optional Python libraries, and 9 external tools were represented in the compact result. The exact 176,306-character JSON response was retained as artifact `artifact_54de7023a923` with SHA-256 `dd88a304e77a054d51dcbcc4a6e30373ceefd69a711a8e5815e28fc61a48f7cf`.
- Interpretation: the model can now inspect actual host capability evidence before choosing an extraction profile, while deterministic code neither reads the raw attachment nor chooses a provider. The probe does not validate extraction quality or imply that every reported contract-only specialist is executable.

## Attachment OCR extraction - 2026-08-28

- Artifact: `/data/var/swaag/manual-tests/attachment-ocr-20260828T033954`.
- Method: create a raw PNG containing `SWAAG OCR MARKER 48291`, add it through the production attachment store, and invoke the normal `extract_attachment` tool with the all2text `tools` profile. The tool itself received only the model-selectable attachment ID and profile; no Swaag file-type route selected OCR.
- Result: all2text selected its image-analysis route and the configured Tesseract provider, reported `ocr_used=true`, and recovered the exact marker at 94% reported confidence. Swaag retained the complete 24,001-character derived result as `artifact_3ae6ad0784cb` with SHA-256 `bb6a6e42b362127b9515f163701985d7a6abbe5f4dde253fbb0dac73ce2cf4cc`, plus the exact manifest and successful-process diagnostics.
- Interpretation: the existing manifest-backed all2text adapter already exposes a working selectable OCR specialist on this host, so a duplicate direct Tesseract adapter would add a specialized branch without new behavior. This verifies the mechanical extraction path, not the live model's semantic decision to inspect or its ability to choose among multiple specialists; those remain benchmark work.

## OpenTelemetry Collector path - 2026-08-28

- Artifact: `/data/var/swaag/manual-tests/otelcol-20260828T041837`.
- Collector: official `otelcol-contrib` 0.159.0 Linux ARM64 archive, verified against release SHA-256 `abb8665cc963e886c2d1286c50b38bcb2e53d968b192c3d8fe4d1ed6b91c3901`.
- Method: validate the repo candidate configuration with the real Collector, launch it as the service user, poll its loopback health endpoint, emit a real Swaag agent span and metrics through the installed OTLP/HTTP protobuf exporters, shut down cleanly, and parse every emitted file record as JSON.
- Result: health on `127.0.0.1:13502` reported available; the receiver on `127.0.0.1:13501` accepted both signals; the 3,600-byte sink contained two valid records with `resourceSpans` and `resourceMetrics`; Collector stderr was empty.
- Service artifact: `/data/var/swaag/manual-tests/otel-communication-20260828T045123`.
- Service method: launch the real Collector and source-tree communication host on temporary loopback port 13402, request its A2A agent card twice, terminate both processes with `SIGTERM`, require zero exits and empty stderr, and parse the independently rotated trace and metric files.
- Service result: one 1,779-byte trace document and one 4,708-byte metric document carried the configured service name and `/.well-known/agent-card.json` route. Both processes exited zero with empty stderr. The probe also proved that the communication host must translate `SIGTERM` into orderly asyncio shutdown and explicitly flush its batched SDK providers.
- Installed-runtime verification: after installing commit `5e49b5a` into `/data/var/swaag/venv`, the same probe passed from the production entry point at `/data/var/swaag/manual-tests/otel-communication-installed-20260828T050115`: one trace document, two metric documents, both route/service identities present, zero exits, and empty stderr.
- Repair evidence: an earlier service probe at `/data/var/swaag/manual-tests/otel-communication-20260828T044604` exposed a Collector nil-writer panic when one alpha file-exporter instance served trace and metric pipelines during shutdown. A second probe at `/data/var/swaag/manual-tests/otel-communication-20260828T044715` verified the independent exporters and clean Collector shutdown, but exposed missing traces because Python does not run `atexit` handlers for default `SIGTERM`. The final probe verifies both general repairs rather than suppressing either failure.
- Scope: this proves the binary, configuration, receiver, exporters, graceful service shutdown, SDK flush, HTTP instrumentation, and metric path as the unprivileged service user. Installing/enabling the repo-backed system units and verifying emissions from the already-running communication daemon still require the root-owned infra sync.
