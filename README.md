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

SWAAG has exactly two authoritative test categories:

- `code_correctness`: deterministic software-correctness checks.
- `agent_test`: cached agent behavior tests, including the full cached benchmark catalog.

Uncached llama.cpp execution is explicit manual validation / real usage, not a test category.

The authoritative agent_test path executes the full cached benchmark catalog, not a reduced representative subset. The current catalog contains 50 realistic tasks across all six task families and all five difficulty tiers, with 10 tasks in each tier including extremely_hard. Coding and multi-step tasks are verified by real workspace edits plus executable checks; reading, failure, and quality tasks use structured-output or anti-tamper contracts instead of benchmark-author hardcoded answers.

Run deterministic code-correctness tests:

```bash
python3 -m swaag.testprofile code-correctness
```

Run cached agent tests, including the full cached benchmark catalog:

```bash
python3 -m swaag.testprofile agent-tests
```

This runs the real cached benchmark, not a pytest wrapper around benchmark-harness
checks. The terminal output shows benchmark progress and benchmark-quality
metrics such as full-task success percentage, difficulty/family group averages,
and average task score.

Run both with fail-fast ordering:

```bash
python3 -m swaag.testprofile combined
```

Generate JSON and markdown reports:

```bash
python3 -m swaag.benchmark test-categories --clean --output /tmp/swaag-test-categories
```

`code_correctness` is reported as a binary correctness result. `agent_test` is
reported as the real benchmark result with task counts, false positives,
full-task success percentage, group-average score, difficulty scores, family
scores, and average task score.

Manual real-model validation, not tests:

```bash
python3 -m swaag.benchmark manual-validation --clean --validation-subset --output /tmp/swaag-manual-validation
```

Report artifacts appear at:

- `/tmp/swaag-test-categories/test_categories_results.json`
- `/tmp/swaag-test-categories/test_categories_report.md`
- `/tmp/swaag-test-categories/code_correctness/code_correctness_results.json`
- `/tmp/swaag-test-categories/code_correctness/code_correctness_report.md`
- `/tmp/swaag-test-categories/agent_test/agent_test_results.json`
- `/tmp/swaag-test-categories/agent_test/agent_test_report.md`
- `/tmp/swaag-manual-validation/manual_validation_results.json`
- `/tmp/swaag-manual-validation/manual_validation_report.md`


## Evaluation

SWAAG exposes exactly two authoritative test categories:

- `code_correctness`: deterministic software-correctness checks with no model traffic.
- `agent_test`: cached agent behavior checks, including the full cached benchmark catalog.

Uncached llama.cpp execution is explicit manual validation / real usage, not a test category.

The authoritative agent_test path executes the full cached benchmark catalog, not a reduced representative subset. The current catalog contains 50 realistic tasks across all six task families and all five difficulty tiers, with 10 tasks in each tier including extremely_hard. Verification is programmatic: test commands, exact file expectations, allowed-modified-file locks, structured JSON checks, and anti-tamper guards carry the benchmark instead of magic final strings.

Recommended commands:

```bash
python3 -m swaag.testprofile code-correctness
python3 -m swaag.testprofile agent-tests
python3 -m swaag.testprofile combined
python3 -m swaag.benchmark test-categories --clean --output /tmp/swaag-test-categories
```

Manual validation, not tests:

```bash
python3 -m swaag.benchmark manual-validation --clean --validation-subset --output /tmp/swaag-manual-validation
```

The `test-categories` command writes JSON and markdown reports and stops before
`agent_test` if `code_correctness` is not 100% green.


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
python3 -m swaag.benchmark agent-tests --output /tmp/swaag-agent-tests --json
python3 -m swaag.benchmark test-categories --output /tmp/swaag-test-categories --json
python3 -m swaag.benchmark manual-validation --validation-subset --output /tmp/swaag-manual-validation --json
python3 -m swaag.benchmark run --clean --output /tmp/swaag-benchmark --json
python3 -m swaag.benchmark external list
python3 -m swaag.benchmark external smoke --all --output /tmp/swaag-external-smoke --json
python3 -m swaag.benchmark system --all --output /tmp/swaag-system-bench --json
```

Use `test-categories` for the authoritative combined test report. Use
`manual-validation` only when you intentionally want uncached real-model usage.

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
