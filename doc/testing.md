# SWAAG Testing

SWAAG has exactly two authoritative test categories.

## 1. code_correctness

Deterministic software-correctness checks. These tests cover imports, unit behavior,
parser/schema/validator logic, command construction, report formatting, harness
plumbing, and other mechanical code behavior. They do not require a model server.

Authoritative command:

```bash
python3 -m swaag.testprofile code-correctness
```

Artifact-producing command:

```bash
python3 -m swaag.benchmark test-categories --clean --output /tmp/swaag-test-categories
```

When code_correctness is not 100% green, the category runner stops immediately and
agent_test is not started.

## 2. agent_test

Cached agent behavior tests. These tests exercise the agent workflow with scripted,
fake, or record/replay model responses. They are the normal day-to-day agent tests.
All LLM-dependent test execution belongs here and must use cache-backed behavior.

The authoritative agent_test path executes the full cached benchmark catalog. It is not a representative subset: current catalog coverage is 196 tasks across coding, file_edit, reading, multi_step, failure, and quality families, with all five difficulty tiers including extremely_hard.

Replay cassettes are keyed by a normalized full request envelope: request payload,
model profile/base URL metadata, structured-output mode, seed, generation parameters
present in the payload, and configured model timeouts. Per-call transport timeout is
recorded for debugging but is not part of the hash because it does not affect model
output semantics.

Authoritative command:

```bash
python3 -m swaag.testprofile agent-tests
```

Artifact-producing command:

```bash
python3 -m swaag.benchmark agent-tests --clean --output /tmp/swaag-agent-tests
```

## Combined fail-fast test command

```bash
python3 -m swaag.testprofile combined
```

or, with JSON/markdown artifacts:

```bash
python3 -m swaag.benchmark test-categories --clean --output /tmp/swaag-test-categories
```

The combined execution is strictly ordered:

1. run code_correctness
2. stop if code_correctness is below 100% or any code-correctness test fails
3. run cached agent_test only after code_correctness is fully green

## Manual validation / real usage

Uncached llama.cpp execution is not a test category. It is explicit manual
validation / real usage.

Manual validation command:

```bash
python3 -m swaag.manual_validation --clean --validation-subset --output /tmp/swaag-manual-validation
```

Manual validation writes:

- `/tmp/swaag-manual-validation/manual_validation_results.json`
- `/tmp/swaag-manual-validation/manual_validation_report.md`
- `/tmp/swaag-manual-validation/manual_validation/`

## Report artifacts

`python3 -m swaag.benchmark test-categories --clean --output /tmp/swaag-test-categories`
writes:

- `/tmp/swaag-test-categories/test_categories_results.json`
- `/tmp/swaag-test-categories/test_categories_report.md`
- `/tmp/swaag-test-categories/code_correctness/code_correctness_results.json`
- `/tmp/swaag-test-categories/code_correctness/code_correctness_report.md`
- `/tmp/swaag-test-categories/agent_test/agent_test_results.json` when code_correctness passes
- `/tmp/swaag-test-categories/agent_test/agent_test_report.md` when code_correctness passes

## Incremental devcheck

`python3 -m swaag.devcheck` is an internal changed-file selector for fast local
feedback. It chooses a focused deterministic profile and can request explicit
follow-up for manual-validation files or expensive agent files. It does not define
additional authoritative test categories.

Common commands:

```bash
python3 -m swaag.devcheck --dry-run
python3 -m swaag.devcheck --changed-file src/swaag/runtime.py --dry-run
```

`pytest-testmon` is used when available to narrow deterministic reruns. If it is
not installed or no baseline exists, devcheck falls back to explicit candidate
files.

To create a pytest-testmon baseline, run the selected deterministic profile once
without forcing affected-test selection; later runs can use the baseline for
faster candidate tests.
