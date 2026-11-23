# SWAAG current-source first-pass multi-tool audit — actual failed run and error-feedback improvement

**November 23, 2026. Verified first-pass result: FAILED.** This was a distinct, authentic local SWAAG developer model session, with a resident LLM controlling genuine allowlisted read-only tools and durable event receipts. The model was asked to identify and recover from a deliberately obsolete file path, locate two current voice-safety test sources, read their contents, name their exact safety tests and calculate the number of functions. Unlike the earlier historical November 20 test that required a separate supervisor correction, this November 23 test tested whether the model would do it **correctly on the first user turn**. It **did not**. Repeated real file-not-found tool calls were observed instead of directory recovery. The process was terminated for its own repeated errors; no model call or already failed effect was replayed. Do **not** publish this as an autonomous successful tool demonstration, as a working voice assistant, or as performance evidence for a phone.

## Frozen new task and original code

Canonical SWAAG input revision before the attempt: `83a3ff051374409dafaa6491c9c0afa467c7b63f`. The new versioned pure preparer is `scripts/demos/prepare_fresh_voice_safety_agent_demo.py`; it pins distinct source files **not used in the November 20 original demo**:

- `tests/test_voice_classifier_acceptance_benchmark.py`, original SHA-256 `e7d9de675686176e5489a26c4ba196f9234d9a4ff56999dc5e1b123272f09ffa`, four top-level test functions, and the exact safety test `test_invalid_classifier_output_never_passes_as_negative_oracle`.
- `tests/test_isolated_voice_release_server.py`, original SHA-256 `4f55a35efbe8372a116dca5c3feae23ff70e1d1cd25328c208f67513d36ed323`, seven top-level test functions, including `test_private_source_preflight_has_no_process_or_network_or_writes`.

The fixed prompt first requests a **real** `read_file` attempt on the absent `tests/test_voice_classification_legacy.py`, recovery by `list_files` on `tests`, then two separate source `read_file` calls and a **real** `calculator` with the expected expression `4 + 7` and correct answer **11**. A model merely claiming any of this is not evidence. Source counts and function names were computed independently by Python AST and pinned to source SHA-256 before inference; parametrized test functions count as one function each.

The model backend was **already-running Thor Qwen3-Coder-30B-A3B-Instruct Q4** at `http://10.8.0.7:14833`, with its original service and consumer registrations left intact. Developer session permitted exactly `read_file`, `list_files` and `calculator`, with a read root confined to the checked-out SWAAG repo and **no shell, file-write, external MCP/agent tool, provider, broker action, or account effect**. Model cache and retry were disabled. It used a unique private session and output folder; nothing was written to live production history.

Compatible interpreter was Python **3.11.9** under `/data/venvs/riganything-py311/bin/python`, with `PYTHONPATH` including the existing SWAAG Python3.11 dependencies and the isolated source and process-only `LD_PRELOAD=/data/var/swaag/test-deps/sqlite-3510300/libsqlite3.so.0`. That preflight confirmed **SQLite 3.51.3** and available `opentelemetry.context`. The separate Python3.12 SWAAG installation's missing `opentelemetry.context` remains an **environment dependency issue**; the earlier genuine Python3.11 `foreground_fast.py` grammar bug was previously fixed. Neither is evidence that the entire SWAAG test suite passes on arbitrary host Python environments.

## Correctly recorded pre-inference configuration rejection

The first source-only preparation used `runtime.max_validation_recovery_cycles = 0`; the actual `load_config()` correctly raised `ValueError` because the configured value must be positive. This was a **config preflight failure**, not a live model failure, and no provider/model call had been sent. The original private directory `real-firstpass-tool-demo-once` and its `preflight-failed-runtime-config.json` receipt remain untouched. The source preparer was corrected to a compatible positive value `1`, and a **fresh, different** private `real-firstpass-tool-demo-configfixed-once` directory was prepared. Live SWAAG `load_config()` and `swaag tools` independently verified the exact three read-only tools, source paths, and no side effects before the one-use model request.

## Actual model, tools, failure, and targeted cancellation

**Observed, not simulated:** SWAAG's durable `complete_history.jsonl` recorded a real initial `read_file` tool call on the nonexistent old path. The tool returned `FilesystemError: File does not exist`. The model then selected the **same nonexistent read** three more times. Four real `tool_called` events each have a correlated genuine `tool_error` event, **zero** `tool_result` successes, no listing, no second or third source read, no calculator call, and no final turn. Four complete model responses were observed in about **41.86, 43.36, 40.42 and 38.07 seconds** respectively, approximately **5.0 to 5.9 generated tokens/second**, and a fifth model request was in flight when terminated. This fifth request's result remains **unknown**. It was not resent or rerun. The observed tool error was not hidden: all four subsequent recorded **actual model prompts explicitly included** the prior `FilesystemError` and missing path. The model still repeated that failure. No claim of successful recovery or first-pass agent autonomy is warranted.

After independently observing the same exact error at least three times, the test stopped to avoid wasting more shared Coder inference. The first process selection correctly **refused** to signal because it matched both a `timeout` wrapper and actual model-client child; on revised exact matching it signaled **only the unique owned SWAAG test Python process**, pid `1063449`. The wrapper reaped it, and the private process receipt reports exit code **143** (explicit SIGTERM); that is not a model timeout, not a platform denial, and not a successful exit. The shared Coder service (then pid `5468`), actual Jetson voice worker (then pid `3413413`), Thor ASR/Qwen, original user data and other workers were unchanged.

The **real** journal has 226 valid, source-linked SWAAG events, at `sessions/session_802326d857c2/complete_history.jsonl` under the private root `.../real-firstpass-tool-demo-configfixed-once`, SHA-256 `c0f70b01637b77467fe988ed00e55215b7573f6ed7635dea41f2f3f166738f01`. Console SHA-256: `392ca9f4c7986701019b8990874f4983d37e6776379f9611c1f4f7aa781fa79c`. Its one-use admission marker, original prompt, first preflight error, frozen source/config SHA, logs and process-exit receipt remain intact.

## Independent oracle and actual safety evidence

A new independent verifier, `scripts/demos/verify_fresh_voice_safety_agent_demo.py`, checks the **original pinned Git source blobs**, the exact frozen prompt SHA and tool allowlist, every durable event's cryptographic SWAAG hash and payload schema, unique correlated tool calls/errors/results, genuine missing-path failure followed by a later directory listing, **two byte-for-byte exact source reads after that listing**, a genuine calculator result of **11**, model request/response correlation, a single completed user turn with no supervisor correction, a final answer supported by the tool receipts, and successful process exit. Its unit fixtures include independently hash-linked **synthetic** event sequences but are marked as source-only verifier regression checks, **never** as a verified model/tool demonstration. Tampered source reads, forged successful calculator results, altered event hashes, duplicated tool receipts, no model responses, missing terminal event or a second user correction must not receive `verified_first_pass`.

The genuine private verification result is `failed_or_incomplete`: the original source/prompt, event hash chain, tool receipts and real failed path are valid, but **no directory recovery, no two successful file reads, no calculation, no final supported response, and no terminal completion**. It also correctly rejects the unreturned fifth model request and the nonzero process exit. Its **first** private oracle had incorrectly marked the process as successful because the exit check was conditional on other checks passing. That verifier bug was corrected; the first result was preserved unchanged, and a **new** source-only, no-provider-call verifier output `independent-real-firstpass-revalidated-oracle-v2.json` independently confirmed the failed verdict, with all applicable exit and request checks rejecting success. Neither checker reran a model, tool, or Send.

## Actionable real code defect and safe corrective implementation

The actual action prompt **did** receive all preceding filesystem tool errors. However, SWAAG's runtime `_run_model_tool_loop` emitted additional explicit `tool_invocation_failed_recoverably` feedback only under `_single_responsibility_mode()`. The actual **normal combined-action mode** did not get the same explicit next-action error guidance. The source correction in `src/swaag/runtime.py` now records the recoverable error feedback for **both** modes after a real failed tool invocation, explaining that the original error must be inspected and that directory listing can resolve an unknown path. It explicitly says **not** to retry a provider Send, externally effective or other unverified side effect merely because a failure was reported.

Importantly, this is **semantic guidance, not an unconditional runtime rejection or quota on repeating identical tools**. The existing SWAAG policy allows valid attempts after externally changing state; for example, an authorized owner-side file may appear later. New regression cases in `tests/test_agent_action_loop.py` show a genuine failed read followed by a model response that gets the explicit error feedback, a real successful `list_files`, a real `read_file` and terminal answer; and a second case shows an initially missing file that subsequently appears and may then be read using the same invocation. These use deterministic fake *model responses* to verify SWAAG's real tool dispatch and prompt; they **do not** prove a new live Coder model would follow that hint correctly on first attempt. New source preparer and independent oracle have separate pure regression suites.

The first four recovery-focused tests, when accidentally attempted under SWAAG's static Python3.11 venv, failed before any tests executed because `sqlite3.sqlite_version=3.50.4` instead of SWAAG's required `>=3.51.3`. Re-running them under the previously verified Python3.11/SQLite3.51.3 process passed **four focused tests**, covering both execution modes and legitimate retry. The distinct source preparation and independent verifier test suites also passed after correcting a fixture's incorrect event index. Only actual executed and independently verified test counts should be reported. No historical evidence, original files, live service or current model session was modified to turn a failure into success.

## Reproducible exact no-action commands and limits

Run only against a newly checked-out known SWAAG `main` and a fresh unused private evidence path; do not use a source checkout with unrelated dirty work. First validate the compatible Python/SQLite process and the resource broker's model consumers and headroom, then prepare a **different new** source-audit task with:

```
P=/data/venvs/riganything-py311/bin/python
W=/data/src/worktrees/OWN_FRESH_SWAAG_SOURCE
export PYTHONPATH=/data/src/github/swaag/.venv/lib/python3.11/site-packages:$W/src:$W
export LD_PRELOAD=/data/var/swaag/test-deps/sqlite-3510300/libsqlite3.so.0
$P "$W/scripts/demos/check_swaag_agent_runtime.py" --repo "$W"
$P "$W/scripts/demos/prepare_fresh_voice_safety_agent_demo.py" --repo "$W" --evidence "/data/var/swaag-benchmarks/OWN_NEW_UNUSED_RUN"
```

The generated task/config are **source-pinned** to the original files and defined model alias. Only after a separate one-use admission, safe model-consumer lease check and explicit no-action authorization could a model call be sent. Observe the actual journal and do not repeat an unknown request. The independent read-only oracle checks the saved original outcome using:

```
$P "$W/scripts/demos/verify_fresh_voice_safety_agent_demo.py" \
  --repo "$W" \
  --journal "/data/var/swaag-benchmarks/ACTUAL_RUN/sessions/ACTUAL_SESSION_ID/complete_history.jsonl" \
  --manifest "/data/var/swaag-benchmarks/ACTUAL_RUN/manifest.json" \
  --task "/data/var/swaag-benchmarks/ACTUAL_RUN/fixed-new-task.txt" \
  --process-receipt "/data/var/swaag-benchmarks/ACTUAL_RUN/firstpass-process-exit-and-evidence.json" \
  --out "/data/var/swaag-benchmarks/ACTUAL_RUN/unused-distinct-verifier-result.json"
```

These are command **shapes**, not a claim that repeating the observed original prompt/model configuration would succeed. Each attempted real model request and source output is immutable one-use evidence. Actual voice input/answer/playback, phone reconnect, owner-speaker authentication, anti-replay, the old live Jetson automatic-send source and the missing dedicated turn classifier remain separate release blockers. The historical November 20 multi-step test remains only `verified_after_supervisor_correction` and must not be conflated with the failed new first-pass attempt.

### Cross-revision input pinning and expanded source verification

For future **distinct** one-use developer evaluations, the preparer now
separately freezes the **exact original source-blob revision**
`83a3ff051374409dafaa6491c9c0afa467c7b63f` and the actual current
**runtime code HEAD**. The two test-file SHA-256 values must still match the
original pinned blobs **before** any model request. The independent verifier
accepts a later documented runtime revision only when that Git commit exists
and the original source-blob revision and byte hashes still match. The
historical live manifest, created when both revisions were the same and
before that additional field existed, remains verifiable unchanged. The
verifier now also correlates real model requests to responses by unique
request identifier; requires actual model-derived action selections matching
each tool's **exact arguments**; and requires a fresh completed model response
after the genuine filesystem error, containing the error evidence, before it
can count directory listing as **model-driven recovery**. A preplanned listing
issued in the same stale action as the failed tool does not count as such a
recovery. These rules do not add external provider requests or reproduce a
failed one.

The independent verifier's updated regression suite contains synthetic hash-
linked counterexamples for tampered source reads, mismatched calculator
results, corrupted event hashes, duplicated tool receipts, omitted/forged
model responses, unmatched tool calls, invented recovery, altered source
manifests, second-turn supervisor corrections, and nonzero exit codes.
They only test the verifier. The **genuine**, unrerun Coder attempt was
independently regraded under additional private, separately named oracle
receipts, with the same **failed** outcome. Its tool and hash provenance was
valid, but the fifth model request did not receive a response, the process
exited 143 after owner-owned test cancellation, and neither the listing, two
source reads, arithmetic, nor final terminal output occurred.

A broad, separately executed source regression under the qualified Python3.11
and SQLite3.51.3 interpreter passed **166 selected cases**, covering the
production agent action loop, original filesystem error handling, permission
and worker-control restrictions, storage and history replay, original
historical verified-but-supervised demonstration, new bounded source
preparation, and source-level independent grading. Additional oracle tests
were extended after that first broad run; the final combined postpublication
suite is the authoritative count for the precise published source revision.
No full repository suite pass is asserted. No TTS, phone speaker or actual
incoming Android microphone was part of this developer model/tool attempt.
