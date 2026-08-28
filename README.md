# SWAAG

SWAAG is a local-first autonomous agent runtime for llama.cpp and OpenAI-compatible model servers. The current architecture is a single constrained action loop: the model chooses one structured action at a time, Python validates it mechanically, tools execute in an isolated workspace, observations are appended to authoritative history, and the loop continues until the model returns a verified final response or a hard runtime limit stops it.

The model owns semantic choices. Python owns schemas, constrained decoding, path and permission policy, exact context accounting, tool execution, persistence, replay, retries, and deterministic verification. The runtime never silently repairs semantic output with hard-coded planner logic.

## Runtime environment

Swaag is a standard `pyproject.toml` package. Development environments may live in the checkout, but long-running services use a reproducible runtime environment under `/data/var/swaag` so service execution never depends on a user's home directory. `uv.lock` pins the resolved dependency graph, and `scripts/install-runtime-env.sh` installs the required Python under `/data/var/swaag/python`, creates `/data/var/swaag/venv`, and synchronizes only the base runtime dependencies unless extras are explicitly requested.

## Design documentation

The implementation is intentionally small while the intended harness architecture is broader. Read these before architectural changes:

- `docs/design-principles.md` — semantic/deterministic boundary, agent behavior, tools, communication, interfaces, and current-versus-target scope.
- `docs/context-management.md` — primary implementation contract for per-call context calculation, output reservation, history projection, summaries, and tool-result sizing.
- `docs/deterministic-policies.md` — audited hard/mechanical production policies and the semantic branches they must never replace.
- `docs/research-and-standards.md` — external systems, protocols, benchmarks, and research discipline.
- `docs/task-api.md` — durable worker operations and caller-defined structured output.

## Core behavior

Every session has append-only event history and replayable state. Prompt context is assembled from the current request, detailed history, model-authored projections, durable notes, environment state, scheduled wakeups, and semantically selected tool schemas. Live llama.cpp capacity and chat serialization come from the connected server: Swaag reads per-slot properties, applies the active model's chat template, verifies the model/template identity again before inference, and tokenizes the exact resulting prompt. Remote OpenAI-compatible backends are capability-discovered rather than treated as llama.cpp: model metadata and capacity come from provider responses, vLLM-style message tokenization is used when exposed, and opaque serialization or explicit fallback capacity is recorded honestly. Exact tokenizer-backed budgeting preserves the richest input that fits an operation-specific output minimum and fixed safety allowance, then uses any remaining desired output headroom. Measured overflow or observed output starvation triggers model-authored reduction and re-tokenization rather than silent truncation.

Status questions use an independent constrained LLM operation over deterministic liveness state and an exact durable worker-history snapshot. Full evidence is retained whenever it fits; measured overflow triggers purpose-specific hierarchical projection with exact source references. Semantic status importance stays separate from the worker heartbeat, and status reads never append to a concurrently active worker history. An optional separate communication endpoint can answer cheap questions without disturbing main inference and can semantically escalate the unchanged evidence to the main model. A mechanical failure of that optional endpoint also fails over without pretending the decision was semantic; trigger, request, result or failure, and any required main-call replay remain durable.

Tools are registered centrally and exposed with closed JSON schemas. Built-ins cover file reading and editing, shell and test execution, calculations, working notes, durable call-scoped prompt instructions, browsing, raw attachment inspection, selectable all2text extraction, short waits, and durable wakeups. Attachment capability discovery reports the current host's complete provider-family inventory as a compact mechanical index while retaining the exact all2text discovery response as an integrity-checked artifact; it never reads a file or deterministically chooses a specialist. Model-authored prompt instructions carry explicit operation scopes plus a semantic session/user persistence choice; the central compiler injects every matching exact entry from both event-sourced stores into the system role and records its source, IDs, hashes, and token cost. Exact instructions remain full fidelity whenever they fit. Only a measured overflow that cannot be recovered from another reducible source permits a per-call model-authored instruction projection; the raw stores and projection provenance remain authoritative and recoverable. Durable wakeups support human-readable relative durations and timezone-aware absolute times, survive process restarts, and are delivered exactly once as session control messages. The communication service dispatches due wakeups through the owning durable worker lifecycle; the mutually exclusive standalone dispatcher provides the same worker-aware behavior for deployments without that service.

Task callers may opt into separately compiled `visual` and `audio` response presentations. The raw verified worker result remains authoritative; relevance selection and listenable rendering are distinct semantic calls, and independent constrained evaluation rejects information loss or operational spam before a variant is exposed. No extra presentation call runs by default while live strategy and small-model benchmarks remain incomplete.

## Installation

Create the project environment and install the package in editable mode:

```text
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Run the agent CLI with `python -m swaag --help`. Run the benchmark CLI with `python -m swaag.benchmark --help`.

## Validation

The authoritative deterministic regression command is:

```text
.venv/bin/python -m pytest -q
```

The full model benchmark is:

```text
.venv/bin/python -m swaag.benchmark run --clean --output /data/var/swaag/benchmarks/<run>/output --json
```

Live context-position diagnostics are resumable and use the connected server's active chat template:

```text
.venv/bin/python -m swaag.benchmark context-order --output /data/var/swaag/benchmarks/context-order.json
.venv/bin/python -m swaag.benchmark context-layout --output /data/var/swaag/benchmarks/context-layout.json
```

Live tool and attachment-context experiments retain complete run histories and verification evidence:

```text
.venv/bin/python -m swaag.benchmark tool-strategy --output /data/var/swaag/benchmarks/tool-strategy
.venv/bin/python -m swaag.benchmark attachment-context --output /data/var/swaag/benchmarks/attachment-context
```

Repeated live compaction checks are resumable and preserve a checkpoint after every cycle:

```text
.venv/bin/python -m swaag.benchmark compaction-preservation --cycles 3 --output /data/var/swaag/benchmarks/compaction-preservation.json
```

The scoped-instruction benchmarks exercise persistence/self-repair and measure strict simultaneous-constraint following through the production agent loop:

```text
.venv/bin/python -m swaag.benchmark prompt-instructions --output /data/var/swaag/benchmarks/prompt-instructions
.venv/bin/python -m swaag.benchmark note-behavior --output /data/var/swaag/benchmarks/note-behavior
.venv/bin/python -m swaag.benchmark instruction-following --output /data/var/swaag/benchmarks/instruction-following
```

Runtime state belongs under `/data/var`; source code belongs under `/data/src`.

## Documentation

Current supporting documents cover installation, history projections, memory and deterministic editing, and live runtime profiles under `doc/`.

## Observability

Swaag instruments protocol requests, agent invocations, logical model operations, tool executions, token usage, context budgets, semantic-reduction counts/targets, and durable compaction frequency with current OpenTelemetry HTTP and GenAI conventions. AG-UI/A2A HTTP and Open WebUI/local adapter requests extract bounded W3C trace context, executor-backed workers retain that context, and every model-backend request receives the active `traceparent`/`tracestate`. External request/context IDs and durable worker/session/run/model-call/tool-call IDs provide mechanical correlation without copying semantic content or arbitrary baggage into telemetry. Library use remains inert unless its host configures an SDK provider. The long-running `communication serve` host configures batched OTLP/HTTP traces and periodic metrics when standard `OTEL_EXPORTER_OTLP_ENDPOINT` (or a signal-specific endpoint) is present; `OTEL_SERVICE_NAME`, `OTEL_METRIC_EXPORT_INTERVAL`, sampling, headers, certificates, and timeouts retain their standard OpenTelemetry meanings. Durable append-only Swaag history remains the replay authority rather than telemetry.

On the Jetson host, `scripts/install-otelcol-contrib.sh` installs the checksum-pinned official Linux ARM64 Collector under `/data/var/swaag/otelcol-contrib`. The repo-backed Collector configuration in `deploy/otelcol-contrib.yaml` accepts OTLP/HTTP only on `127.0.0.1:13501`, exposes health only on `127.0.0.1:13502`, and writes bounded rotating trace and metric OTLP JSON evidence under `/data/var/swaag/telemetry`. Separate signal exporters prevent one file writer from racing another signal during Collector shutdown. These sinks are operational evidence only; they never replace append-only execution history.

Model calls also pass through a durable backend-neutral admission lifecycle. Live llama.cpp capacity comes from `/props` `total_slots`; communication/control calls receive higher mechanical priority, queue aging prevents worker starvation, and every release/retry/cancellation/supersession remains inspectable independently of the transport backend.

Durable prompt records include the exact rendered-prompt hash and SHA-256 versions for the active server prompt protocol, rendered system instruction, and canonical template files used by each operation. Recorded replay cassettes retain the exact server-rendered prompt separately from model completions, so offline replay neither contacts a server nor guesses a model-family wrapper. This makes prompt behavior inspectable across action, compaction, projection, completion, structured-output, capability, and communication calls without placing semantic prompt content in operational telemetry.

Protocol conformance dependencies remain outside the Python runtime. Repo-backed probes exercise pinned MCP 2026-07-28 discovery/list/call, official A2A 1.0 card/list/get/cancel/subscription decoding, and the official AG-UI `HttpAgent` POST/SSE lifecycle. Their exact SDKs install under `/data/var/swaag/protocol-conformance`; none become Python runtime dependencies. MCP remains a replaceable capability boundary rather than Swaag's task protocol.
