# SWAAG

## (S)elfhosted (W)orking (A)utonomous (AG)ent

SWAAG is a local-first autonomous agent core for a direct `llama.cpp` server.
It is built to stay inspectable, replayable, and reproducible on one machine.

Core properties:
- strict context-budget control on every model call
- append-only canonical history with replayable state rebuild
- explicit runtime loop with planning, tools, verification, and recovery
- self-hosted operation against a local `llama.cpp` HTTP server
- session persistence, control messages, checkpoints, and benchmark plumbing

## What is in this repo

This repository contains the full working agent project, including:
- the main `swaag` package under `src/swaag`
- CLI entrypoints for agent use, devcheck, test profiles, benchmarks, and final proof
- benchmark fixtures and local benchmark wrappers
- full repo-level test suite under `tests/`
- additional package-level smoke tests under `src/swaag/tests`
- detailed documentation under `doc/`

## Install

Clone the repo and install it in editable mode:

```bash
cd /data/src/github/swaag
python3 -m pip install -e .[test]
```

Optional benchmark dependencies:

```bash
python3 -m pip install -e .[official-benchmarks]
```

Optional packaging/publish tooling:

```bash
python3 -m pip install -e .[publish]
```

If your current Python environment is not writable for publish extras, use a
throwaway build venv instead:

```bash
python3 -m venv /tmp/swaag-build
/tmp/swaag-build/bin/python -m pip install build twine
/tmp/swaag-build/bin/python -m build
```

Once the package is published, the intended install command is:

```bash
pip install swaag
```

## Local `llama.cpp` server setup

SWAAG expects a local `llama.cpp` server that exposes at least:
- `/health`
- `/tokenize`
- `/completion`

Official `llama.cpp` repository:
- https://github.com/ggml-org/llama.cpp

A solid general-purpose example model for local testing:
- Qwen2.5-7B-Instruct-GGUF
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF

Example server launch:

```bash
llama-server \
  -m /absolute/path/to/Qwen2.5-7B-Instruct-Q5_K_M.gguf \
  --host 127.0.0.1 \
  --port 14829 \
  -c 2048
```

Default SWAAG config expects the server here:

```toml
[model]
base_url = "http://127.0.0.1:14829"
context_limit = 2048
```

If your server lives elsewhere, override it with either `config/local.toml` or env vars:

```bash
export SWAAG__MODEL__BASE_URL=http://127.0.0.1:14829
export SWAAG__MODEL__CONTEXT_LIMIT=2048
```

## Quickstart

Basic checks:

```bash
cd /data/src/github/swaag
python3 -m swaag doctor --json
python3 -m swaag tools
```

Single-turn ask:

```bash
python3 -m swaag ask "Use the calculator tool to compute 6 * 7. Reply with the numeric result only."
```

Interactive chat:

```bash
python3 -m swaag chat
```

Useful session commands:

```bash
python3 -m swaag sessions
python3 -m swaag rename latest my-session
python3 -m swaag control "Keep the current task running, but also answer with plain digits." --session latest
python3 -m swaag checkpoint create --session latest --label before-edit
python3 -m swaag checkpoint restore --session latest
python3 -m swaag history detail latest "What exact command copied src.txt to dst.txt?"
```

## Config

Main defaults:
- `src/swaag/assets/defaults.toml`

Example local override:
- `config/local.example.toml`

Environment override prefix:
- `SWAAG__...`

Config path override:
- `SWAAG_CONFIG=/path/to/file.toml`

Examples:

```bash
export SWAAG__MODEL__BASE_URL=http://127.0.0.1:14829
export SWAAG__SESSIONS__ROOT=/tmp/swaag-sessions
export SWAAG__TOOLS__READ_ROOTS='["/safe/root"]'
export SWAAG__TOOLS__ALLOW_SIDE_EFFECT_TOOLS=true
```

## Testing

Installed-package smoke test:

```bash
python3 -m swaag.tests
```

Full repo test suite:

```bash
cd /data/src/github/swaag
python3 run_tests.py
pytest -q
```

Changed-area developer loop:

```bash
python3 -m swaag.devcheck
```

Fast tests:

```bash
python3 -m swaag.testprofile fast
```

System tests:

```bash
python3 -m swaag.testprofile system
```

Integration tests:

```bash
python3 -m swaag.testprofile integration
```

No-cache validation against a real local server:

```bash
SWAAG_RUN_LIVE=1 python3 -m swaag.testprofile live
```

Final proof loop:

```bash
python3 -m swaag.finalproof
```

## Evaluation

SWAAG presents testing in two user-facing categories:

- deterministic correctness tests
  - imports, smoke tests, deterministic unit tests, harness checks, runtime plumbing
  - no model traffic
  - expected to stay at `100%`
- agent behavior tests
  - cached mode is the normal fast path
    - uses cassette-backed record/replay by default when model-backed task execution is needed
    - keeps reruns fast once the cache exists
  - no-cache validation mode is the occasional real-model confirmation path
    - uses the same real agent runtime with direct `llama.cpp` calls
    - grouped into five difficulty tiers:
      - `extremely_easy`
      - `easy`
      - `normal`
      - `hard`
      - `extremely_hard`
    - the curated validation subset keeps at least `10` distinct tasks in each tier
    - each task produces a percentage score plus rubric breakdown

Recommended commands:

```bash
python3 -m pytest -q -x
python3 -m swaag.benchmark agent-tests --mode cached --clean --validation-subset --output /tmp/swaag-agent-tests-cached
python3 -m swaag.benchmark agent-tests --mode no-cache-validation --clean --validation-subset --output /tmp/swaag-agent-tests-validation
python3 -m swaag.benchmark test-categories --clean --validation-subset --output /tmp/swaag-test-categories
```

`test-categories` is the authoritative combined score command. It averages:

- deterministic correctness percent
- agent behavior tests (cached mode) percent
- agent behavior tests (no-cache validation mode) percent

Artifacts written by the combined evaluator:
- `test_categories_results.json`
- `test_categories_report.md`
- `deterministic_correctness/functional_correctness_results.json`
- `deterministic_correctness/functional_correctness_report.md`
- `agent_behavior_cached_results.json`
- `agent_behavior_cached_report.md`
- `agent_behavior_cached/agent_behavior_cached_results.json`
- `agent_behavior_cached/agent_behavior_cached_report.md`
- `agent_behavior_cached/replay_cache/`
- `agent_behavior_validation_results.json`
- `agent_behavior_validation_report.md`
- `agent_behavior_validation/agent_behavior_validation_results.json`
- `agent_behavior_validation/agent_behavior_validation_report.md`

The human-readable combined report includes:
- per-category summaries
- per-tier no-cache validation scores
- lowest-scoring no-cache validation tasks with rubric excerpts
- artifact paths for each category and mode
- replay-cache location for cached mode

Focused cached-mode agent behavior support checks:

```bash
python3 -m swaag.benchmark agent-support --all --clean --output /tmp/swaag-agent-support
```

## Build and publish

Build a source distribution and wheel:

```bash
python3 -m build
```

Or use the provided helper:

```bash
./build.sh
```

The helper builds the package, uploads it with `twine`, and cleans local build artifacts.

## Benchmarks

Main benchmark entrypoints:

```bash
python3 -m swaag.benchmark evaluate --clean --output /tmp/swaag-eval --json
python3 -m swaag.benchmark agent-support --all --output /tmp/swaag-agent-support --json
python3 -m swaag.benchmark agent-tests --mode cached --validation-subset --output /tmp/swaag-agent-tests-cached --json
python3 -m swaag.benchmark agent-tests --mode no-cache-validation --validation-subset --output /tmp/swaag-agent-tests-validation --json
python3 -m swaag.benchmark test-categories --validation-subset --output /tmp/swaag-test-categories --json
python3 -m swaag.benchmark run --clean --output /tmp/swaag-benchmark --json
python3 -m swaag.benchmark external list
python3 -m swaag.benchmark external smoke --all --output /tmp/swaag-external-smoke --json
python3 -m swaag.benchmark system --all --output /tmp/swaag-system-bench --json
```

Use `test-categories` for the authoritative combined scoring view:
- deterministic correctness percent
- agent behavior tests (cached mode) percent
- agent behavior tests (no-cache validation mode) percent
- no-cache validation per-tier difficulty percents
- no-cache validation per-task rubric scores
- one final overall percent as a simple arithmetic average of the three category scores

Use `evaluate` when you want a fast non-live combined run. Use `run` when you
want only the built-in full-agent benchmark task set.

Reproducible bounded SWE-bench fixtures live in:
- `src/swaag/benchmark/fixtures/swebench/`

Local Terminal-Bench task fixtures live in:
- `src/swaag/benchmark/terminal_tasks/`

## Documentation

Start here:
- `doc/installation.md`
- `doc/testing.md`
- `doc/architecture.md`
- `doc/runtime_loop.md`
- `doc/context_budgeting.md`
- `doc/history_and_projections.md`
- `doc/memory_and_editing.md`
- `doc/live_runtime_profiles.md`

## License

MIT. See `LICENSE`.
