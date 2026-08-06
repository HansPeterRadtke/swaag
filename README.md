# SWAAG

SWAAG is a local-first autonomous agent runtime for llama.cpp and OpenAI-compatible model servers. The current architecture is a single constrained action loop: the model chooses one structured action at a time, Python validates it mechanically, tools execute in an isolated workspace, observations are appended to authoritative history, and the loop continues until the model returns a verified final response or a hard runtime limit stops it.

The model owns semantic choices. Python owns schemas, constrained decoding, path and permission policy, exact context accounting, tool execution, persistence, replay, retries, and deterministic verification. The runtime never silently repairs semantic output with hard-coded planner logic.

## Core behavior

Every session has append-only event history and replayable state. Prompt context is assembled from the current request, recent detailed history, model-authored compressed history, durable notes, environment state, scheduled wakeups, and the complete enabled-tool registry. Exact tokenizer-backed budgeting reserves output tokens and a safety margin before any request is admitted. When compaction is required, the model can request bounded verbatim retention of recent messages while Python enforces the hard context limit.

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
