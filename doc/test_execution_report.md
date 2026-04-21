# Test execution speed report

Measured on the current tree with all infrastructure in place.

## Baseline measurements

| Path | Wall time | Tests run | Tests deselected |
|---|---|---|---|
| `pytest -q` (default, excludes integration/live/bench) | ~10m 16s | 289 passed | 29 |
| `-m system` only | ~10m 27s | 121 passed | 197 |
| `-m fast` only | ~3.1s | 168 passed | 157 |
| `devcheck --changed-file src/swaag/tokens.py` | ~1.0s pytest / ~7s total | 13 passed, 2 deselected | n/a |
| `devcheck --changed-file src/swaag/runtime.py` | ~29m | 16 candidate files | testmon runs full system |

## Example: tiny local source change

```
changed_files: [src/swaag/tokens.py]
test_profile: fast
candidate_tests: [tests/test_tokens.py, tests/test_budgeting.py, tests/test_imports.py]
testmon mode: forceselect (baseline ready)
final executed: 13 passed, 2 deselected
pytest time: 0.99s
why: tokens.py has a narrow deterministic mapping → stays in the fast test subset.
```

Speedup: **~1s vs ~10m** — roughly **600x** faster than the default `pytest -q` path.

## Example: runtime-core source change

```
changed_files: [src/swaag/runtime.py]
test_profile: system
candidate_tests: 16 files (8 fast + 8 system)
testmon mode: forceselect
why: runtime.py affects orchestration → broadens to system test subset.
```

This is the worst-case inner-loop change. The system test subset is expensive because
runtime tests do full orchestration turns. With a warm testmon baseline and no
actual source changes touching new code paths, testmon will progressively
deselect more tests as the baseline stabilizes.

## Example: test-only change

```
changed_files: [tests/test_tokens.py]
test_profile: fast
candidate_tests: [tests/test_tokens.py]
testmon mode: forceselect
final executed: only affected tests in test_tokens.py
why: changed test file included directly; stays fast.
```

## Example: packaging change

```
changed_files: [pyproject.toml]
test_profile: integration
candidate_tests: 8 files (packaging + CLI + live profile tests)
testmon mode: forceselect
why: pyproject.toml affects packaging wiring → broadens to integration test subset.
```

## Example: docs-only change

```
changed_files: [README.md]
test_profile: fast
candidate_tests: (none)
why: README.md has no dedicated consistency tests; no code tests added.
```

## Example: doc with runtime consistency tests

```
changed_files: [doc/testing.md]
test_profile: fast
candidate_tests: [tests/test_devcheck.py]
why: doc/testing.md has a dedicated consistency check → only that test runs.
```

## Test subset cost summary

| Test subset | Typical wall time | When used |
|---|---|---|
| fast | 3-7s | Narrow source edits, test-only changes, docs |
| system | ~10m | Runtime/orchestration/core changes |
| integration | varies | Packaging, pyproject, CLI, model-client |
| live | requires real server | Explicit `--allow-live` only |
| benchmark_heavy | long | Explicit proof runs only |

## Two-stage selection

1. **Coarse changed-area candidate selection**: deterministic mapping from
   changed files to candidate test families. Broadens test subset when needed.
2. **Fine-grained pytest-testmon deselection**: within the candidate set,
   testmon further reduces to only actually-affected tests when a baseline
   exists.

If testmon is unavailable, the system degrades explicitly: same candidate set
runs without incremental deselection, and the output says why.

## Key invariants

- Every automatic skip is explainable via `devcheck --dry-run`.
- Stale selector entries fail loudly via the internal selector-registry validator.
- Missing testmon degrades explicitly (no silent full-suite fallback).
- Missing baseline uses `--testmon-noselect` once to populate `.testmondata`.
- Changed files touching shared infrastructure broaden the test subset accordingly.
