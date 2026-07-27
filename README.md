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

## Architectural invariant: the model owns semantics

SWAAG is a model-directed agent. Every decision that depends on meaning must be
made by the model: interpreting the user, classifying intent, selecting a
strategy, decomposing work, assigning semantic file roles, choosing tools,
constructing semantic tool inputs, selecting a recovery approach, synthesizing
content, and producing the final answer.

Deterministic code is limited to semantics-independent infrastructure such as
tool implementations, schemas, serialization, history mechanics, permissions,
path safety, budgets, retries, transport, replay, explicit machine-format
parsing, and objective execution verification. Regexes, keyword lists, filename
patterns, fixed workflows, language-specific phrases, benchmark cases, and
hardcoded answer logic must never replace model judgment. Code that works only
for one wording, grammar, abbreviation, domain, culture, or language is wrong,
even when it improves a benchmark.

All contributors and automated coding agents must follow the continuation guide
in [`doc/model_semantic_authority.md`](doc/model_semantic_authority.md). Passing
tests or benchmarks does not establish correctness when this invariant is
violated.

Every live model call uses generation-time JSON-schema constrained decoding.
For OpenRouter and OpenAI-compatible providers, SWAAG uses Chat Completions
Structured Outputs with `response_format.type=json_schema`, strict schemas, and
`provider.require_parameters=true`. For `llama.cpp`, SWAAG uses the server's
top-level `json_schema` request field. Plain semantic text is still constrained
as a closed object such as `{"text": string}` and unwrapped only after
validation.

Semantic tool selection and semantic delegation decisions always receive the
complete enabled tool registry: registered name, meaningful description, full
input schema, and registered usage guidance. This includes task decision,
planning, direct/tool routing, replanning, recovery, and subagent-selection
calls. Skills may add prompt text, examples, expected outputs, and verifier
hints, but skill metadata must never hide, remove, or preselect tools for the
model.
Tool identity is exact. Python must not treat two registered tools as
interchangeable, even when they appear to serve a similar purpose. If a
model-authored step declares `expected_tool`, any later selected tool must have
that exact registered name or the runtime returns mismatch evidence to the
model for correction.

Semantic recovery and replanning must also receive relevant observations from
history. For example, a successful file-read observation must make the observed
text available to later planning, tool-choice, and recovery calls within the
context budget. Deterministic code may enforce budgets and safety boundaries,
but it must not collapse relevant evidence to metadata-only summaries that
force the model to guess.
Failed tool inputs, verifier failures, and latest artifact snapshots are also
model evidence. If an edit range, pattern, source snippet, or other target is
now absent from the current artifact, Python may report that fact mechanically
and reject repeated malformed execution, but the model must decide whether to
repair from current state, re-read, ask for clarification, or declare a genuine
blocker. Recovery prompts must not let broad partial matches replace exact
requested final-state proof.
Exact repeated structured actions inside one step are a mechanical loop-safety
case: the runtime may stop executing byte-equivalent repeats and hand off to
verification, but it must not infer semantic equivalence between different
actions or synthesize replacement arguments.

`execution_mode=single_tool` is only a model-owned planning hint. The runtime
must still ask the planner model for an explicit plan and objective verification
criteria; Python must not synthesize a generic direct-tool plan from the
preferred tool name.

Planner output is validated mechanically. If a schema-valid plan fails local
plan validation, SWAAG retries the planner with validation evidence under the
same constrained schema; it does not fill missing conditions or rewrite the plan
in Python. The repair budget is a correctness guard, not a latency target: it
must allow multiple distinct contract corrections before failing closed.
Generated plans use `verification_type="composite"` for every step. Every
ordinary model-authored verification check declares `condition="required"` or
`condition="optional"` directly on that check; the constrained wire schema has
no duplicate condition-name arrays. Structural completion conditions are derived
from the step kind and exact expected tool rather than repeated by the model.
Semantic answer or reasoning verification
is still model-owned: the plan must include a model-declared `criterion` check
with `condition="required"`, unless the model declares a required exact/string
match against `assistant_text`. An optional-only criterion cannot prove a
semantic answer or reasoning step. The parser translates those local statuses
into internal required and optional check-name lists without changing the
model's choices.

Model-declared artifact references are not executable content. Plan
`input_text` is instruction context; selected-tool arguments are generated only
by the constrained selected-tool input call. If actual side-effect tool input
contains an unresolved artifact placeholder such as `{{artifact_name}}`, SWAAG
rejects it mechanically and returns the failure to the model for recovery.
Tools may register required objective verification check types; when they do,
the model selects a dedicated compact `objective_verification_check`, preserves
its model-authored name, and supplies only the payload fields applicable to that
objective type. This slot is always required and cannot select mechanical-only
checks such as `tool_files_changed`. If the check is missing or malformed,
runtime review rejects the plan with
structured validation evidence and asks the model for a corrected constrained
plan. Python must not promote optional checks, rewrite condition importance, or
synthesize verification semantics. Objective file-content verification must
declare concrete expected text precise enough to reject partial or corrupt
edits; an empty containment target is a failed check, not a successful match.
The `condition` field is only the check's required/optional status; output labels
such as `file_content` belong in dataflow fields, and existence checks such as
`file_exists` are not objective proof for mutating tools unless the model also
declares the tool's registered objective check type as required.
For observed text edits, `edit_text` exposes `replace_exact` as the preferred
operation: the model supplies `old_text` and `new_text`, the editor requires
exactly one literal match by default, and zero or multiple matches fail closed.
`replace_range` remains a low-level operation guarded by `expected_text`, not
the simple path.
File-mutating tools also require a constrained semantic result review after
mechanical verification passes. The reviewer model receives the step goal,
success criteria, deterministic verification evidence, tool result, diff, and
current file text, then returns the strict `verification` JSON-schema verdict
for `result_satisfies_step`. Python may enforce that this review exists and
may retry on rejection, but it must not decide that a partial containment match
is semantically sufficient.
Required semantic review and required semantic answer verification fail closed:
if the semantic backend is unavailable, degraded, or violates its structured
schema contract, SWAAG records that evidence as a verification failure instead
of marking the reviewer perspective as passing.

Before a final answer is exposed as successful completion, SWAAG runs a
constrained model-owned final objective verification against the original user
request, the candidate answer, the active plan, recent observations, and the
current workspace evidence. A failed `final_objective_satisfied` verdict is a
normal verification failure and feeds replanning/recovery; Python must not turn
a weakened later plan or partial current state into success.
If a prior step-level objective check failed but later current evidence appears
to satisfy the original request, the model may choose a recovery plan that
verifies/observes the current state and answers. Python may record
`unresolved_objective_verification_deferred`, but completion is still blocked
until the mandatory final objective verifier accepts the whole request.

The runtime is optimized for correctness and completion, not low latency. A
turn must continue through planning, tool calls, observations, recovery, and
verification until success is verified, a genuine blocker or clarification need
is reached, or a mechanical watchdog prevents an infinite loop. Preparatory
stages, short wall-clock expectations, or exhausted benchmark-shaped reasoning
phases are not success. If a watchdog or safety bound stops unresolved work,
SWAAG returns a mechanical incomplete-status message; it does not ask the model
to produce a normal semantic final answer for unverified work.

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

Manual validation / real usage is not a test category. It is cache-first by default; pass `--uncached-live` only for an intentionally uncached model run.

The authoritative agent_test path executes the full cached benchmark catalog, not a reduced representative subset. The current catalog contains 59 realistic tasks across all six task families and all five difficulty tiers, with intentionally asymmetric difficulty counts so the harder tiers can carry richer scenarios instead of tidy symmetry. Coding and multi-step tasks are verified by real workspace edits plus executable checks; reading, failure, and quality tasks use structured-output or anti-tamper contracts instead of benchmark-author hardcoded answers.

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
# Cache-first: replay exact hits and record missing outputs.
python3 -m swaag.benchmark manual-validation --clean --full-catalog --output /tmp/swaag-manual-validation

# Explicit cache bypass.
python3 -m swaag.benchmark manual-validation --uncached-live --clean --full-catalog --output /tmp/swaag-manual-validation-live
```

The cache lookup hashes the complete output-affecting request identity: model/server fingerprint, endpoint, prompt or messages, strict JSON Schema, seed, temperature, top-p and other supplied sampling fields, token limit, stop sequences, runtime profile, and caller scope. Transport timeout is recorded for diagnostics but is not part of output identity.

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

Manual validation / real usage is not a test category. It is cache-first by default; pass `--uncached-live` only for an intentionally uncached model run.

The authoritative agent_test path executes the full cached benchmark catalog, not a reduced representative subset. The current catalog contains 59 realistic tasks across all six task families and all five difficulty tiers, with intentionally asymmetric difficulty counts so realism wins over neat tier symmetry. Verification is programmatic: test commands, exact file expectations, allowed-modified-file locks, structured JSON checks, and anti-tamper guards carry the benchmark instead of magic final strings.

Recommended commands:

```bash
python3 -m swaag.testprofile code-correctness
python3 -m swaag.testprofile agent-tests
python3 -m swaag.testprofile combined
python3 -m swaag.benchmark test-categories --clean --output /tmp/swaag-test-categories
```

Manual validation, not tests:

```bash
python3 -m swaag.benchmark manual-validation --clean --full-catalog --output /tmp/swaag-manual-validation
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
python3 -m swaag.benchmark manual-validation --full-catalog --output /tmp/swaag-manual-validation --json
python3 -m swaag.benchmark run --clean --output /tmp/swaag-benchmark --json
python3 -m swaag.benchmark external list
python3 -m swaag.benchmark external smoke --all --output /tmp/swaag-external-smoke --json
python3 -m swaag.benchmark system --all --output /tmp/swaag-system-bench --json
```

Use `test-categories` for the authoritative combined test report. Use
`manual-validation` for real-model validation. It reuses or records model outputs by default; add `--uncached-live` only when an uncached run is intentional.

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
