# SWAAG Testing

SWAAG has exactly two authoritative test categories.

## 1. code_correctness

Deterministic software-correctness checks. These tests cover imports, unit behavior,
parser/schema/validator logic, command construction, report formatting, harness
plumbing, and other mechanical code behavior. They do not require a model server.
They also include architecture regressions that prove model ownership of
semantic decisions: skills cannot hide enabled tools, planning and tool-choice
prompts disclose the full enabled registry, selected-tool argument prompts use
registered tool documentation, every live contract is constrained JSON schema,
provider payloads match their adapters, llama.cpp receives top-level
`json_schema`, and production generation schemas pass the portable-schema
validator.

Deterministic product tests must also exercise the assembled public runtime
loop, not only helper functions. The real-loop integration tests use scripted
model responses with the actual runtime, registered tools, temporary workspace,
history recording, observations, replanning/recovery, verification, and final
filesystem/result inspection. They cover all five difficulty tiers and the
coding, file_edit, reading, multi_step, failure, and quality families. Negative
cases assert that incomplete or weakened current states do not claim semantic
success, while recovery plans that observe an already-correct current state
must still pass whole-goal final objective proof before answering.

Authoritative command:

```bash
python3 -m swaag.testprofile code-correctness
```

The output is binary and intentionally concise: total checks, passed, failed,
skipped, percent, and the final pass/fail result. It does not run the real
benchmark catalog.

Artifact-producing command:

```bash
python3 -m swaag.benchmark test-categories --clean --output /tmp/swaag-test-categories
```

When code_correctness is not 100% green, the category runner stops immediately and
agent_test is not started.

## 2. agent_test

Cached agent behavior tests. The authoritative agent_test path exercises the full benchmark catalog through RecordReplayModelClient over the real model client. Existing cassettes are replayed; missing cassettes are recorded from real model responses. No benchmark catalog task embeds fixed model responses.

The authoritative agent_test path executes the full cached benchmark catalog. It is not a representative subset: the current catalog contains 59 realistic tasks across coding, file_edit, reading, multi_step, failure, and quality families, and it intentionally uses an asymmetric difficulty mix so hard and extremely_hard can hold richer scenarios instead of tidy filler.

Benchmark verification is programmatic rather than benchmark-author scripted:

- coding and multi_step tasks require real workspace edits plus executable verifier commands
- file_edit tasks verify exact file contents and allowed-modified-file constraints
- reading tasks verify exact JSON outputs against explicit schemas
- failure tasks preserve protected files and reject unsafe changes
- quality tasks use explicit prompt-understanding oracles plus structured answer expectations

The catalog is intentionally varied, not a string-template exercise:

- coding covers single-file bug fixes, refactor compatibility repairs, spec-driven logic fixes, and multi-file release-flow repairs
- file_edit covers exact config edits, multi-occurrence replacement, no-op detection, and source-to-target synchronization
- reading covers structured extraction, contradiction handling, stale-note rejection, and null-preserving reporting
- multi_step covers release-note synchronization, computed report generation, shell-driven workflows, filesystem selection, note-taking, and iterative refinement
- failure covers unsafe shell requests, bad plans, and repeated-action traps
- quality covers vague prompts, incomplete prompts, already-decomposed prompts, and debug-log interpretation

Replay cassettes are keyed by a normalized full request envelope: request payload,
model profile/base URL metadata, structured-output mode, seed, generation parameters
present in the payload, and configured model timeouts. Per-call transport timeout is
recorded for debugging but is not part of the hash because it does not affect model
output semantics.

Authoritative command:

```bash
python3 -m swaag.testprofile agent-tests
```

This command runs the real cached benchmark catalog through the normal agent
loop. It is not a pytest wrapper. The output shows benchmark progress as tasks
complete and ends with real benchmark-quality metrics:

- total tasks
- successful tasks
- failed tasks
- false positives
- full-task success percentage
- difficulty-group average score
- family-group average score
- overall group-average score
- average task score

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

When architecture defects are known, benchmark execution must stop until those
defects are fixed and the architecture regressions pass. A benchmark pass is not
evidence that deterministic code preserved model semantic authority.

Real-loop integration tests must cover task-decision paths that can name a
preferred tool. In particular, `execution_mode=single_tool` must still produce
a model-returned plan with objective verification, and a first incomplete or
wrong mutation must force continued tool work rather than generic success.
Context-building tests must also prove structured retrieval protocol failures
record `semantic_retrieval_degraded` and fall back only to neutral retrieval.
Regression tests must also cover the boundary between planner instruction text
and executable tool input. Placeholder-looking text in planner `input_text` is
inert context and must not be executed or rejected as tool JSON. Unresolved
artifact placeholders in actual side-effect tool arguments are mechanically
rejected with model-visible validation evidence, no placeholder mutation, and
final verification against concrete file contents. File-content checks must
prove empty containment targets fail rather than passing every existing file.
Tests must prove that model-declared output-ref artifact aliases verify against
real tool results, so successful reads or edits do not fail merely because the
model named the output `file_content` or another arbitrary label.
Tests must prove that registry-declared automatic mechanical objective checks
are installed exactly once before plan review and cannot be supplied through the
live model wire. Mutating tools without an automatic default must not pass with
only artifact-exists or nonempty-output checks; missing, optional, or malformed
model-authored `file_contains` or `command_success` proof must be rejected with
model-visible validation evidence.
Tests must prove exact tool identity. If a model-authored plan declares
`expected_tool`, executing a different registered tool name must fail
verification or be rejected as mismatch evidence; tests must not encode
read-tool or other tool-pair equivalence as a success condition.
Regression tests must cover that every live model-facing verification check has
a local required/optional condition status, that exact duplicate checks collapse
while conflicting same-name checks fail, and that dataflow labels cannot occupy
condition fields. They must also prove `file_exists`, `tool_files_changed`,
`artifact_present`, or `tool_output_nonempty` alone cannot satisfy objective
proof for a mutating tool. Legacy saved plans with internal condition-name lists,
explicit done conditions, and the former objective slot must remain readable,
while live plans derive done conditions and registered automatic mechanical
checks. Diagnostic `run_tests` failures must be accepted as observations unless
the plan explicitly requires `command_success`.
They must also prove that semantic response or reasoning plans are rejected
before execution unless at least one model-declared `criterion` check or
exact/string assistant-text match is required; optional-only semantic criteria
must not allow a final answer step to run and then fail late.
Required semantic reviewer failures must be fail-closed in tests: unavailable,
degraded, or schema-violating semantic backends must produce verification
failure unless the model declared an exact literal assistant-text match that can
be checked mechanically.
They must also cover file-mutation semantic result review: a broad edit that
passes a weak containment check but corrupts the artifact must be rejected by a
constrained model `verification` call, recorded as a model-visible observation,
retried through the real runtime loop, and accepted only after the final file
state is objectively verified.
Real-loop tests must also cover whole-goal final objective verification. A
scripted model may replan to a weaker read-only check after a failed mutation,
but the assembled agent must not expose that as success when current evidence
does not satisfy the original request. The test must prove
`final_objective_satisfied` failed, recovery/replanning continued, a repair
tool actually ran, and a later final objective proof passed after the final
mutation.
They must also cover the opposite recovery case: if an earlier step-level
objective check failed because the check itself was wrong but the current
artifact now satisfies the request, a recovery plan may reread/observe and
answer without repeating the mutation. The runtime must record
`unresolved_objective_verification_deferred`, run the final objective verifier,
and expose the answer only if that proof passes.
Prompt and tool-guidance tests must keep file-edit objective checks and edit
arguments model-owned but explicit enough that the model is told to reject
partial or corrupt edits through its own declared checks.
Real-loop recovery tests must prove that a failed mutation followed by replan
receives the relevant observed tool output, such as read file text, so the
model can choose the recovery action from evidence instead of guessing from
metadata.
They must also cover stale-source recovery: after a partial mutation and
failed retries against an absent source snippet, range, or pattern, the real
runtime loop must expose recent failed tool evidence and latest file snapshots
to the planner and selected-tool argument calls, reject false success, continue
through replan/recovery, and verify the final artifact after the last mutation.
Real-loop tests must also prove that a model response inside a step requiring a
tool is rejected as invalid execution evidence, does not complete the step, and
does not prevent a later model-selected corrective tool call from reaching
verified success.
The same coverage must exist for a model-selected wrong tool inside a step with
`expected_tool`: the wrong tool is not executed, the rejection is recorded, and
a later model-selected matching tool can still reach verified success before
replanning is attempted.
Planner tests must distinguish structural condition normalization from semantic
repair: dependency artifact names already declared in `input_refs` may map to
`dependencies_completed`, an explicit `dependencies_completed` condition may
add that generic structural check object, but empty required lists and unknown
condition names must still fail with model-visible validation evidence. The
runtime must continue through multiple distinct planner-contract repair
attempts before a valid model-authored plan executes real tools and
verification.

`python3 -m swaag.testprofile all` is accepted as an alias for `combined`.

## Manual validation / real usage

Manual validation / real usage is not a test category. It uses cache-first
reuse-or-record behavior by default. Pass `--uncached-live` only for an explicitly
uncached llama.cpp run.

Manual validation command:

```bash
# Cache-first default.
python3 -m swaag.manual_validation --clean --full-catalog --output /tmp/swaag-manual-validation

# Explicit cache bypass.
python3 -m swaag.manual_validation --uncached-live --clean --full-catalog --output /tmp/swaag-manual-validation-live
```

Cache identity covers every supplied output-affecting field, including model/server fingerprint, endpoint, prompt, strict schema, seed, sampling parameters, maximum output tokens, stop sequences, profile, and caller scope. Exact hits are replayed; misses are recorded atomically. Transport timeout does not change a completed model output and is therefore diagnostic metadata rather than part of the hash.

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
- `/tmp/swaag-test-categories/agent_test/agent_test_cached_results.json` when code_correctness passes
- `/tmp/swaag-test-categories/agent_test/agent_test_cached_report.md` when code_correctness passes

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
