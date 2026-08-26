# SWAAG

SWAAG is a local-first autonomous agent runtime for llama.cpp and OpenAI-compatible model servers. The current architecture is a single constrained action loop: the model chooses one structured action at a time, Python validates it mechanically, tools execute in an isolated workspace, observations are appended to authoritative history, and the loop continues until the model returns a verified final response or a hard runtime limit stops it.

The model owns semantic choices. Python owns schemas, constrained decoding, path and permission policy, exact context accounting, tool execution, persistence, replay, retries, and deterministic verification. The runtime never silently repairs semantic output with hard-coded planner logic.

## Runtime environment

Swaag is a standard `pyproject.toml` package. Development environments may live in the checkout, but long-running services use a reproducible runtime environment under `/data/var/swaag` so service execution never depends on a user's home directory. `uv.lock` pins the resolved dependency graph, and `scripts/install-runtime-env.sh` installs the required Python under `/data/var/swaag/python`, creates `/data/var/swaag/venv`, and synchronizes only the base runtime dependencies unless extras are explicitly requested.

## Design documentation

The implementation is intentionally small while the intended harness architecture is broader. Read these before architectural changes:

- `docs/design-principles.md` — semantic/deterministic boundary, agent behavior, tools, communication, interfaces, and current-versus-target scope.
- `docs/context-management.md` — primary implementation contract for per-call context calculation, output reservation, history projection, summaries, and tool-result sizing.
- `docs/research-and-standards.md` — external systems, protocols, benchmarks, and research discipline.
- `docs/task-api.md` — durable worker operations and caller-defined structured output.

## Core behavior

Every session has append-only event history and replayable state. Prompt context is assembled from the current request, detailed history, model-authored projections, durable notes, environment state, scheduled wakeups, and semantically selected tool schemas. Live llama.cpp capacity comes from the server's per-slot properties. Exact tokenizer-backed budgeting preserves the richest input that fits an operation-specific output minimum and fixed safety allowance, then uses any remaining desired output headroom. Measured overflow or observed output starvation triggers model-authored reduction and re-tokenization rather than silent truncation.

Status questions use an independent constrained LLM operation over deterministic liveness state and an exact durable worker-history snapshot. Full evidence is retained whenever it fits; measured overflow triggers purpose-specific hierarchical projection with exact source references. Semantic status importance stays separate from the worker heartbeat, and status reads never append to a concurrently active worker history.

Tools are registered centrally and exposed with closed JSON schemas. Built-ins cover file reading and editing, shell and test execution, calculations, notes, browsing, short waits, and durable wakeups. Durable wakeups support human-readable relative durations and timezone-aware absolute times, survive process restarts, and are delivered exactly once as session control messages.

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

Runtime state belongs under `/data/var`; source code belongs under `/data/src`.

## Documentation

Current supporting documents cover installation, history projections, memory and deterministic editing, and live runtime profiles under `doc/`.

## Observability

Swaag instruments agent invocations, logical model operations, tool executions, token usage, and context budgets with the current OpenTelemetry GenAI conventions. The base package depends only on `opentelemetry-api`: without a hosting-process SDK/provider it is a standard no-op, while deployments can configure sampling and exporters independently through normal OpenTelemetry mechanisms. Semantic prompts, responses, tool arguments, and tool results are deliberately not captured; durable append-only Swaag history remains the replay authority rather than telemetry.

Durable prompt records include the exact rendered-prompt hash and SHA-256 versions for the protocol wrapper, rendered system instruction, and canonical template files used by each operation. This makes prompt behavior inspectable across action, compaction, projection, completion, structured-output, capability, and communication calls without placing semantic prompt content in operational telemetry.
