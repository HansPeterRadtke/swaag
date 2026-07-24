# Live Runtime Profiles

`swaag` runs against the direct local `llama.cpp` server. The profile and output-mode choice is now centralized in `src/swaag/live_runtime_profiles.py`. `finalproof`, cache-first validation, and explicit `--uncached-live` validation read from that same source of truth.

These profiles are transport settings only. They must not encode benchmark success,
task-family routing, prompt shortcuts, or semantic recovery behavior.

## Locally discovered llama.cpp profiles

From `/data/bin/llama.sh` and `/data/etc/hosts/nitro.json`:

- `small_fast`
  - `ctx=2048`
  - `gpu_layers=31`
  - default host profile
- `mid_context`
  - `ctx=20000`
  - `gpu_layers=24`
- `max_context`
  - `ctx=32000`
  - `gpu_layers=31`

These are not equivalent. Larger context profiles give more prompt headroom but they cost more latency, especially for structured calls.

## Recommended runtime choices

- Fast validation checks (cache-first by default; `--uncached-live` for a bypass)
  - profile: `small_fast`
  - structured mode: `server_schema`
  - fixed seeds: `11,23,37`
  - timeout: `120`
  - reason: small local context, bounded latency, and generation-time JSON-schema constraints
- Final validation benchmark catalog
  - profile: `small_fast`
  - structured mode: `server_schema`
  - fixed seeds: `11,23,37`
  - timeout: `180`
  - Local transport profile: uses the default `2048` context profile for repeatable live runs
  - reason: this controls latency, timeout, and schema transport only; it must not add benchmark-specific semantic routing
- Heavy structured / larger-prompt runs
  - profile: `mid_context`
  - structured mode: `server_schema`
  - timeout: `240`
  - use this only when the request no longer fits `small_fast` or when a larger prompt window is worth the latency
- Slow local hardware fallback
  - profile: `small_fast`
  - structured mode: `server_schema`
  - timeout: `240`
  - connect timeout: `15`
  - progress poll: `10.0`

## Structured output modes

- `server_schema`
  - required for every live semantic model call
  - OpenRouter and OpenAI-compatible providers use Chat Completions Structured Outputs
  - llama.cpp uses its top-level `json_schema` request field
  - the runtime still validates parsed JSON locally after generation

## Why the default uses `small_fast` + `server_schema`

This is the default transport choice on the current machine:

- local llama.cpp is configured with a modest default context envelope
- repeatable live runs should not change model profile inside the runtime loop
- `server_schema` keeps generation-time schema enforcement active for every live semantic call

`finalproof` now exports these settings explicitly:

- `SWAAG_LIVE_MODEL_PROFILE=small_fast`
- `SWAAG_LIVE_STRUCTURED_OUTPUT_MODE=server_schema`
- `SWAAG_LIVE_SEEDS=11,23,37`
- `SWAAG_LIVE_TIMEOUT_SECONDS=180`
- `SWAAG_LIVE_CONNECT_TIMEOUT_SECONDS=10`
- `SWAAG_LIVE_PROGRESS_POLL_SECONDS=5.0`

The validation benchmark command in `finalproof` uses the same values explicitly.
The agent does not restart or switch llama.cpp profiles inside the runtime loop. A human may choose a different external server profile before launching the server, but the proof path keeps one fixed profile for the full run.

## Timeouts and observability

The runtime no longer treats a long call as a dead call.

- simple answers use the lower simple timeout
- structured planning / decision calls use the structured timeout
- verification-heavy calls use the verification timeout
- benchmark validation runs use the benchmark timeout where appropriate
- long uncached validation calls emit `model_request_progress` events during polling

The benchmark reports include:

- total wall-clock time
- average task time
- slowest task time
- retry counts
- timeout failure rate
- unconstrained contract violation count
- profile / mode used

## Environment variables

- `SWAAG_RUN_LIVE=1`
- `SWAAG_LIVE_BASE_URL=http://127.0.0.1:14829`
- `SWAAG_LIVE_TIMEOUT_SECONDS=180`
- `SWAAG_LIVE_CONNECT_TIMEOUT_SECONDS=10`
- `SWAAG_LIVE_MODEL_PROFILE=small_fast`
- `SWAAG_LIVE_STRUCTURED_OUTPUT_MODE=server_schema`
- `SWAAG_LIVE_SEEDS=11,23,37`
- `SWAAG_LIVE_PROGRESS_POLL_SECONDS=5.0`

## Proof loops

- Fast dev loop
  - `python3 -m swaag.devcheck`
  - runs the smallest deterministic subset based on changed files
- Final proof loop
  - `python3 -m swaag.finalproof`
  - runs imports, scaled catalog, runtime verification flow, end-to-end tests, full fast suite, packaging checks, cached agent tests, large benchmark, full manual-validation catalog, and archive proof

## Practical guidance

- Use `small_fast` for normal local live runs unless the operator intentionally starts a different server profile.
- Switch to `mid_context` only when prompt assembly or structured output genuinely needs more room.
- Reserve `max_context` for ad hoc heavy debugging or unusually large structured calls.
- Do not downgrade live semantic calls to post-validation-only or unconstrained text output.

## Model-side concurrency benchmark

Measured against the current local server on `small_fast` (`ctx=2048`, `parallel=1`):

- single long request
  - `18.419s`
  - `192` completion tokens
  - `10.424 tok/s`
- two long requests started concurrently
  - combined wall time: `36.041s`
  - combined throughput: `10.655 tok/s`
  - per-request latency: `17.310s` and `36.038s`
- one long plus one small request started concurrently
  - combined wall time: `19.151s`
  - the small request still took `19.146s`
- one long request while background shell work runs
  - `17.822s`
  - `10.773 tok/s`

Decision:

- do **not** implement model-side concurrency in the runtime yet
- keep the scheduler focused on overlapping shell/process work with a single semantic worker path

Reason:

- the server is configured with `parallel=1`
- concurrent requests do not materially improve total useful throughput
- concurrent requests significantly damage tail latency for the queued request
- background shell activity does not meaningfully hurt the single-request baseline
