# Testing

## Default inner loop

```bash
cd /data/src/github/swaag
python3 -m swaag.devcheck
```

`devcheck` is the normal developer command.

It uses a two-stage selector:
- deterministic changed-area mapping to build the smallest correct candidate set
- `pytest-testmon` inside that candidate set to run only affected tests when a baseline exists

It prints:
- changed files
- chosen test profile
- candidate tests
- `pytest-testmon` mode
- explicit follow-up profiles when a change requires `live` or `benchmark_heavy` execution

If `pytest-testmon` is installed but no baseline exists yet, `devcheck` uses
`--testmon-noselect` once to populate `.testmondata` while still limiting the
run to the selected candidate tests.

If `pytest-testmon` is unavailable, the fallback is explicit: the same candidate
set runs without incremental deselection.

## Build the pytest-testmon baseline

```bash
python3 -m swaag.devcheck --baseline
```

or explicitly per profile:

```bash
python3 -m swaag.testprofile fast --baseline
python3 -m swaag.testprofile system --baseline
```

A baseline is only needed once per profile unless the `.testmondata*` files are
removed.

## Authoritative two-category commands

There are exactly two test categories. Use these as the authoritative entry points:

### Code-correctness tests

```bash
python3 -m swaag.testprofile code-correctness
```

Runs exactly the code-correctness file list — no marker-based deselection. This is
the authoritative command for the deterministic correctness category. Should stay
at 100%. If it drops, that is a real code bug.

### Agent tests

```bash
python3 -m swaag.testprofile agent-tests
```

Runs exactly the agent-test file list. Tests that require a live model
(`test_live_llamacpp.py`) skip gracefully unless `SWAAG_RUN_LIVE=1` is set.

### Combined (fail-fast)

```bash
python3 -m swaag.testprofile combined
```

Runs code-correctness first. Runs agent-tests only if code-correctness is fully
green. This is the authoritative combined entry point.

## Devcheck inner loop and testprofile incremental profiles

`devcheck` is the normal developer incremental command. It runs the smallest
correct candidate set from the changed area:

```bash
python3 -m swaag.devcheck
```

It uses a two-stage selector:
- deterministic changed-area mapping to build the smallest correct candidate set
- `pytest-testmon` inside that candidate set to run only affected tests when a baseline exists

It prints:
- changed files
- chosen test profile
- candidate tests
- `pytest-testmon` mode
- explicit follow-up profiles when a change requires live or benchmark-heavy execution

If `pytest-testmon` is installed but no baseline exists yet, `devcheck` uses
`--testmon-noselect` once to populate `.testmondata` while still limiting the
run to the selected candidate tests.

If `pytest-testmon` is unavailable, the fallback is explicit: the same candidate
set runs without incremental deselection.

### Build the pytest-testmon baseline

```bash
python3 -m swaag.devcheck --baseline
```

or explicitly per internal profile:

```bash
python3 -m swaag.testprofile fast --baseline
python3 -m swaag.testprofile system --baseline
```

A baseline is only needed once per profile unless the `.testmondata*` files are
removed.

### Internal devcheck profiles

These are used by the devcheck inner loop and are not the authoritative user-facing
test interface. The authoritative commands are `code-correctness`, `agent-tests`,
and `combined` above.

#### Fast profile

```bash
python3 -m swaag.testprofile fast
```

Cheap deterministic tests only:
- no live model
- no clean-room install
- no heavy benchmark execution
- no broad runtime smoke coverage

#### System profile

```bash
python3 -m swaag.testprofile system
```

Broader local system coverage:
- runtime / orchestration
- environment / history / end-to-end smoke
- scheduler / background-process lifecycle
- planner / reasoning / subagents
- benchmark-structure and report logic

## Agent behavior tests

Agent behavior tests run in two modes:

- cached mode (the normal path; uses record/replay cassettes)
- no-cache validation mode (manual real-model validation)

**Cache policy:** All tests run with LLM calls cached. Real uncached execution is
real usage mode or manual validation mode, not a test category.

## Evaluation architecture

SWAAG exposes two user-facing evaluation categories:

- deterministic correctness tests
- agent behavior tests

Agent behavior tests run in two modes:

- cached mode
- no-cache validation mode

### Category 1: deterministic correctness tests

This is the repository health category:

- import hygiene
- smoke tests
- deterministic unit tests
- parser/schema/validator checks
- harness correctness
- packaging and install checks
- benchmark/report plumbing

This category should stay at `100%`. If it drops, that is a real code bug.

### Category 2: agent behavior tests

This category covers the real runtime stack. It does not grade helper functions
in isolation. It runs the real runtime/orchestrator/tool loop.

Cached mode is the normal fast path:

- record/replay cassettes are used by default for model-backed task execution
- cassette keys are based on normalized request payload plus request metadata
- reruns avoid real delegate model calls once a cassette exists
- focused scripted support-check families still exist for tightly controlled failure
  and recovery scenarios
- the cassette hash includes request metadata such as base URL, completion endpoint,
  model profile, structured output mode, and seed
- `timeout_seconds` is stored for observability but excluded from the hash because
  it is a transport setting rather than semantic model input

No-cache validation mode is the occasional real-model confirmation path:

- uses the same real agent runtime
- keeps direct `llama.cpp` calls enabled
- uses the curated validation subset with at least `10` distinct tasks in each
  difficulty tier

Run cached mode with:

```bash
python3 -m swaag.benchmark agent-tests --mode cached --clean --validation-subset --output /tmp/swaag-agent-tests-cached
```

Cached-mode artifacts:

- `agent_behavior_cached_results.json`
- `agent_behavior_cached_report.md`
- `agent_behavior_cached/agent_behavior_cached_results.json`
- `agent_behavior_cached/agent_behavior_cached_report.md`
- `agent_behavior_cached/replay_cache/`

Focused cached-mode agent behavior support checks:

```bash
python3 -m swaag.benchmark agent-support --all --clean --output /tmp/swaag-agent-support
```

Run no-cache validation mode with:

```bash
python3 -m swaag.benchmark agent-tests --mode no-cache-validation --clean --validation-subset --output /tmp/swaag-agent-tests-validation
```

No-cache validation artifacts:

- `agent_behavior_validation_results.json`
- `agent_behavior_validation_report.md`
- `agent_behavior_validation/agent_behavior_validation_results.json`
- `agent_behavior_validation/agent_behavior_validation_report.md`
- `agent_behavior_validation/benchmark_results.json`
- `agent_behavior_validation/benchmark_report.md`

The no-cache validation task catalog is grouped into five ordered difficulty tiers:

- `extremely_easy`
- `easy`
- `normal`
- `hard`
- `extremely_hard`

The curated validation subset keeps at least `10` distinct tasks in each tier so the
category cannot silently collapse into only the easiest cases.

Every no-cache validation task carries:

- a `0-100%` task score
- a rubric breakdown
- machine-readable evidence

Each tier score is the arithmetic mean of its task scores. The no-cache validation score
is the arithmetic mean of all no-cache validation task scores.

### Combined evaluation command

Run the full combined evaluation with:

```bash
python3 -m swaag.benchmark test-categories \
  --clean \
  --validation-subset \
  --output /tmp/swaag-test-categories
```

The final overall score is the arithmetic mean of:

- deterministic correctness tests percent
- agent behavior tests (cached mode) percent
- agent behavior tests (no-cache validation mode) percent

Artifacts:

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

The combined markdown report also includes:

- per-category summaries
- no-cache validation lowest-scoring tasks with rubric excerpts
- explicit artifact locations for each category

The JSON output contains:

- deterministic correctness percent
- agent behavior tests (cached mode) percent
- agent behavior tests (no-cache validation mode) percent
- no-cache validation per-tier difficulty percents
- no-cache validation per-task scores
- no-cache validation per-task rubric breakdowns
- final overall percent

### Fast non-live combined evaluation

The older combined evaluator is still useful during local iteration when you
want deterministic correctness plus the scripted full-agent task catalog:

```bash
python3 -m swaag.benchmark evaluate --clean --output /tmp/swaag-eval
```

## Test categories

There are exactly two test categories:

- **code-correctness tests** — deterministic software-correctness checks with no model dependency.
  Authoritative command: `python3 -m swaag.testprofile code-correctness`
- **agent tests** — agent behavior tests that depend on LLM output (cached or uncached).
  Authoritative command: `python3 -m swaag.testprofile agent-tests`

Both commands use explicit file lists — no marker-based deselection.

**Execution order rule:** Agent tests must not start if code-correctness tests are not 100% green.
Use `python3 -m swaag.testprofile combined` to enforce this automatically.

To run agent tests explicitly:

```bash
# Cached mode (replay, no live model needed)
python3 -m swaag.benchmark agent-tests --mode cached --clean --validation-subset --output /tmp/swaag-agent-tests-cached

# Validation mode (real model, no cache)
SWAAG_RUN_LIVE=1 python3 -m swaag.benchmark agent-tests --mode no-cache-validation --clean --validation-subset --output /tmp/swaag-agent-tests-validation
```

Installation and local server setup are documented in:

- `doc/installation.md`

## Changed-area invalidation rules

The deterministic selector broadens when changes touch:
- `pyproject.toml`, scripts, CLI wrappers, or packaging files -> `integration` profile
- runtime/core prompt/model/orchestrator files -> `system` profile
- environment / retrieval / guidance / skills / subagents / tools -> `system` profile
- benchmark catalog/runner/report files -> benchmark-structure tests in `system` profile
- live runtime profile docs/config -> dedicated live-structure consistency tests

Devcheck always runs only code-correctness tests (filtered via `-m not agent_test`).
Agent tests in the candidate set are filtered out and must be run separately via
`python3 -m swaag.testprofile agent-tests`.

Docs that are not runtime inputs do not trigger unrelated code tests.
Docs with dedicated consistency checks map only to those tests.

## Final proof loop

```bash
python3 -m swaag.finalproof
```

This runs the deliberate proof path:
- imports
- scaled catalog
- runtime verification flow
- end-to-end smoke
- local non-live pytest path
- integration tests
- no-cache validation tests
- large benchmark CLI
- no-cache validation subset CLI
- archive proof

## Benchmark and archive proofs

```bash
python3 -m swaag.benchmark evaluate --clean --output /tmp/swaag-eval --json
python3 -m swaag.benchmark run --clean --output /tmp/swaag-benchmark --json
SWAAG_RUN_LIVE=1 python3 -m swaag.benchmark run --clean --validation-subset --model-profile small_fast --structured-output-mode post_validate --timeout-seconds 180 --seeds 11,23,37 --output /tmp/swaag-live-benchmark --json
python3 scripts/archive_proof.py
```

## Official external benchmark harnesses

The project now includes a project-owned wrapper for official external benchmark
tools. Use it to validate harness availability, capture stderr/stdout, and keep
benchmark reports in one consistent format.

Install the optional toolchain first if you want local official-harness smoke
checks to pass instead of reporting missing commands/modules:

```bash
python3 -m pip install -e .[official-benchmarks]
```

List supported integrations:

```bash
python3 -m swaag.benchmark external list
```

Smoke-check every configured harness:

```bash
python3 -m swaag.benchmark external smoke --all --output /tmp/swaag-external-smoke --json
```

Run the agent-system benchmark families that exercise the real runtime/session
architecture:

```bash
python3 -m swaag.benchmark system --all --output /tmp/swaag-system-bench --json
```

Run one official harness with explicit template variables:

```bash
python3 -m swaag.benchmark external run \
  --benchmark swebench-full \
  --output /tmp/swaag-external-run \
  --var predictions_path=gold \
  --var dataset_name_args='--dataset_name /tmp/full_row.json' \
  --var cache_args='--cache_level instance' \
  --json
```

Run the real local agent-generated SWE-bench path for a bounded text target:

```bash
python3 -m swaag.benchmark external agent-run \
  --benchmark swebench-lite \
  --output /tmp/swaag-swebench-lite-agent \
  --var instance_ids_args='--instance_ids astropy__astropy-12907' \
  --var cache_args='--cache_level instance' \
  --json
```

```bash
python3 -m swaag.benchmark external run \
  --benchmark terminal-bench \
  --output /tmp/swaag-terminal-bench \
  --var dataset_locator_args='--dataset-path /path/to/tasks' \
  --var task_selection_args='--task-id hello-world' \
  --timeout-seconds 30 \
  --json
```

Artifacts:

- JSON report: `external_benchmark_smoke_results.json` or `external_benchmark_full_results.json`
- Markdown report: `external_benchmark_smoke_report.md` or `external_benchmark_full_report.md`
- per-benchmark stdout/stderr capture directories under the chosen output root

Real defaults live in:

- `src/swaag/assets/defaults.toml` under `[external_benchmarks]`

Override precedence is the same as the rest of the agent:

1. packaged defaults in `src/swaag/assets/defaults.toml`
2. `config/local.toml`
3. `SWAAG_CONFIG=/path/to/file.toml`
4. `SWAAG__...` per-key environment overrides

Typical external requirements:

- `swebench_*`: the `swebench` Python package and its official Docker-based evaluator
- `swebench_*` bounded local-first proof runs can use:
  - `dataset_name_args='--dataset_name /tmp/row.json'` to point the official evaluator at a local single-row dataset file
  - `instance_ids_args='--instance_ids <id>'` to target a single official instance id from a hosted dataset
  - `cache_args='--cache_level instance'` to keep the built instance image for follow-up proof runs on the same instance
- `external agent-run` is the real local text SWE-bench proof path. It clones the benchmark repo, runs the real `python3 -m swaag ask` entrypoint inside that workspace, captures the resulting repo diff as `agent_predictions.jsonl`, and then calls the official evaluator on those generated predictions.
- `python3 -m swaag.benchmark system --all` runs the curated agent-system benchmark families against the real runtime/session/history/checkpoint architecture. This is separate from the official public harnesses.
- `predictions_path=gold` remains useful for harness self-checks only. It does not prove agent quality.
- `terminal_bench`: `tb` from the official `terminal-bench` package plus Docker. The project wrapper uses the real-agent adapter and accepts `dataset_locator_args='--dataset-path /path/to/tasks'` for local-first runs against an already checked-out dataset. If `docker compose` is missing, the wrapper can fall back to an existing `docker-compose` binary or a repo-local standalone compose download.
- local text SWE-bench `agent-run` requires the normal agent model endpoint configured in `SWAAG__MODEL__BASE_URL`. If that endpoint is unavailable, the run is externally blocked by the local model-server environment rather than the harness itself.
- local text SWE-bench `agent-run` can also be blocked by external hosted-dataset
  or repository fetch failures. The wrapper now classifies dataset-download and
  evaluator build-network failures as explicit external blockers instead of
  silently reporting a generic pass/fail with stale artifacts.
- when you already have a prepared single-row dataset file from a previous run,
  prefer reusing it via `dataset_name_args='--dataset_name /path/to/row.json'`
  for bounded local reruns. That avoids unnecessary hosted dataset traffic and
  makes multilingual/full reruns much more stable on slower machines.

Use `--timeout-seconds` when you want a bounded real-run attempt on a machine
that may otherwise spend a long time downloading images, datasets, or waiting on
external infrastructure.

The external benchmark wrapper now treats timeouts as `failed`, not
`external_blocked`. A timeout is only evidence that the bounded run did not
finish within the chosen window; it is not automatically treated as an external
blocker.

Benchmark-environment policy keys now live under:

- `[external_benchmarks.model_server]`
  - `preflight_enabled`
  - `healthcheck_timeout_seconds`
  - `retry_attempts`
  - `retry_sleep_seconds`
- `[external_benchmarks.terminal_bench]`
  - `compose_probe_timeout_seconds`
  - `compose_download_timeout_seconds`
  - `allow_compose_download`

Reproducible bounded benchmark inputs live in:

- `src/swaag/benchmark/fixtures/swebench/`
- `src/swaag/benchmark/terminal_tasks/`

Those repo-local fixtures are the supported proof inputs for bounded local runs.
Temporary output can still go to `/tmp`, but the selected cases themselves are
stored in the repository.

## Model-concurrency benchmark

Do not implement concurrent LLM requests blindly. Measure first:

```bash
python3 scripts/benchmark_model_concurrency.py --output /tmp/swaag-model-concurrency.json
```

This benchmark records:

- one long request alone
- two long requests concurrently
- one long plus one small request concurrently
- one long request while background shell work runs

Use the results to decide whether model-side concurrency is worth the extra
complexity on the current hardware.

## Speed report

See `doc/test_execution_report.md` for measured test-subset-selection examples and
inner-loop timing comparisons.
