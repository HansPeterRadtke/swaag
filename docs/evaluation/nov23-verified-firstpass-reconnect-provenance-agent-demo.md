# SWAAG: genuinely completed first-pass multi-step read-only agent task

**Date:** 23 November 2026. **Outcome: `verified_first_pass`.** A self-hosted Coder model, controlling actual SWAAG source-reading, directory-listing and calculator tools, completed a new security-source investigation with a deliberate real filesystem failure, correct autonomous recovery, two exact genuine file inspections, a real calculation, and a completed answer in a **single original user turn**. This is a real locally hosted model execution with genuine durable source-backed receipts, **not** a synthetic or self-reported demo. The original unsuccessful November 23 source audit is preserved independently and remains a failure; the different November 20 tool demonstration remains only `verified_after_supervisor_correction`.

This success **does not** establish real-time voice responsiveness, physical Android phone operation, live human speaker identification, microphone/replay protection, or automatic tool authorization. The normal Jetson voice WebSocket server still uses its older live code; the dedicated Thor turn classifier is not available. No production services were restarted, phone apps installed, external messages sent or genuine user tasks performed.

## Exact real task, code and tools

The original frozen SWAAG Git source revision and model-program base was `c2a9e55008c8872abed290ac4e34e08c473083dc`, the canonical remote `main` at the time of the test. The clean isolated Nitro worktree was `/data/src/worktrees/swaag-real-agent-firstpass-wave66-20261123`. Separate reproducible, independently authored, versioned entrypoints:

- `scripts/demos/prepare_fresh_reconnect_provenance_agent_demo.py` exclusively prepares fixed private files and freezes source/prompt/config SHA-256, without a model call.
- `scripts/demos/run_fresh_reconnect_provenance_agent_once.py` permits **one new** private model/tool session, verifies the actual resident model alias and config, and stops only its own Python process if the precise missing-file error repeats three times or a developer-only ceiling is reached. It does not terminate or restart the shared Thor Coder service. It never repeats an uncertain earlier attempt.
- `scripts/demos/verify_fresh_reconnect_provenance_agent_demo.py` is a distinct independent evaluator. It reads the actual durable SWAAG event history, the frozen original source Git blobs, and the process-exit receipt. It verifies model request/response IDs, selected action/tool arguments, call/result/error correlation, all cryptographic event hashes, actual error-to-recovery causality, real file contents, AST test counts, real calculator output and a supported terminal response. A plausible answer or a test-file existence alone never counts.
- Corresponding `tests/test_prepare_fresh_reconnect_provenance_agent_demo.py`, `tests/test_run_fresh_reconnect_provenance_agent_once.py`, and `tests/test_verify_fresh_reconnect_provenance_agent_demo.py` test admission, source integrity, one-use refusal, and the independent verifier. Synthetic verifier fixtures are **clearly separate from the real model run**.

The frozen original source inputs, completely distinct from previous first-pass and corrected demos, were:

| File | Original SHA-256 | Top-level test functions | Required named test |
|---|---|---:|---|
| `tests/test_orchestrator_indexed_receipt_replay.py` | `675779990610a4b56d581e626ed64ea1a908987eb96d4696bcdbb93f43c1f9ff` | 2 | `test_reconnect_replays_exact_answer_without_materializing_giant_journal` |
| `tests/test_tool_result_provenance.py` | `e44c724b7cf102f1c22a88775e41c6f7e8b1962df2f4ee1c7cbbab8a55b79aed` | 3 | `test_tool_message_is_separate_accounting_component_with_source_event` |

The fixed first user task directed the model to deliberately attempt `read_file` on a **different obsolete path**, `tests/test_replay_connection_legacy.py`, **once only**, observe the genuine error, use `list_files` on `tests` to recover, read the two actual current files separately, identify the named tests, count top-level Python test declarations, **call** `calculator` on the observed counts, and answer using the real results. Counting parametrized Python `def test_...` declarations as one was part of the original oracle.

The already-running model was Thor `qwen3-coder-30b-a3b-instruct-q4-k-m` at an existing private Coder inference endpoint. The SWAAG model cache was disabled, retries set to zero, no private provider or other LLM account was used, and **only** the three tool names `read_file`, `list_files`, and `calculator` were enabled. Filesystem reads were limited to the isolated SWAAG source checkout; no shell, OS-write, broker, external MCP, buyer interaction, phone control, provider Send, purchase or destructive tool was available. The real Python was 3.11.9 with process-only SQLite 3.51.3 preload and the established isolated SWAAG dependencies. The first historical Python 3.11 syntax bug in `foreground_fast.py` was addressed previously; the separate Python 3.12 `opentelemetry.context` package gap is a different environment issue and is not fixed by this result.

## Source-backed observed execution

All steps below were extracted from actual SWAAG event receipts after the single one-use process exited normally. This is the order recorded, not a desired plan:

1. `read_file({"path":"tests/test_replay_connection_legacy.py"})` returned one real `FilesystemError`, demonstrating that an obsolete file was not silently invented or treated as a success.
2. The *next completed model call* included this real error in its actual prompt, plus SWAAG's newly published explicit tool-error recovery guidance. The model selected `list_files({"path":"tests"})`. That real tool returned the test directory contents.
3. The model separately selected `read_file` on `tests/test_orchestrator_indexed_receipt_replay.py`, then on `tests/test_tool_result_provenance.py`. Their actual tool-result `text` fields exactly match the SHA-pinned original Git file bytes. Independent AST parsing counted **2** and **3** top-level test functions, and the required exact function names are present.
4. The model selected the real `calculator({"expression":"2 + 3"})`, whose actual tool result was **5**. It then produced an accurate final answer referring to both source paths and named functions, the missing-path error and directory recovery, and the observed arithmetic.

**Independent genuine oracle result:** `verified_first_pass`, **17/17 causal/integrity checks passed**. Exact recorded totals: **320 durable events, 6 model requests and 6 completed responses, 5 actual tool calls, 4 successful tool results, 1 deliberate tool error, 1 completed original user turn, 0 corrective user turns, 0 real side-effect tools**. Real process exit code **0** with no cancellation or retry. The model did not select the failed path a second time.

Measured model-response durations were **127.073, 43.104, 46.732, 45.825, 43.747 and 33.302 seconds**. The unusually slow first call took **109.439 seconds to first model token**; later calls had first-token delays of about 15–24 seconds. Individual actual tool-dispatch elapsed times were **0.0476s** (failed read), **0.1574s** (directory listing), **0.1003s** and **0.0937s** (two reads) and **0.0558s** (calculator). The complete local test harness wall time was **376.437 seconds**, or roughly **6 minutes 16 seconds**, including model, prompt/context compilation, tool execution and orchestration. The model is therefore not yet fast enough for acceptable natural speech-to-response delay, despite this verified agent result. These are sample timings on shared live hardware, not latency guarantees or a population speed benchmark.

## Immutable private provenance and exact commands

All raw model request payloads, complete original model output, full session history, prompts, private environment details and one-use marker are retained **privately** on Nitro at:

`/data/var/swaag-benchmarks/owner-serial-000066-20261123/fresh-reconnect-provenance-agent-once/`

The genuine journal is in `sessions/ACTUAL_SESSION_ID/complete_history.jsonl`, **not** a synthetic unit-test transcript. The private source-only `manifest.json`, original `fixed-new-reconnect-task.txt`, `read-only-config.toml`, `real-cli-console.log`, `result.json`, and the independently produced `independent-firstpass-reconnect-provenance-oracle.json` remain unchanged. The original independent verification JSON SHA-256 is `ed601a6f6cb534cf375b4bffe36b1d927ce8c00d6b721f46f7b2f15aad2c4efe`, and the exact journal SHA-256 is `f79b108198b96ff9419638cf407aa9ff85e95c320b18ae8f634338af1fdd8c28`.

The precise executable runner source as it existed **during** the one-use model/test run was also preserved privately as `../private_real_agent_runner_exact_source_at_execution.py`, SHA-256 `13682458afd3744dfbcf1560e3a197c681a4646f9010cd89bfd3e5349deced48`. Immediately afterward the published copy changed only an inaccurate initial-result field name, from `tools_executed_claimed=False` to `tool_executions_not_assumed_before_history_verification=True`, to avoid confusing an admission-time declaration with the four subsequently observed successful actual tool results. That source-only labeling correction did **not** change the model request, tools, execution logic, generated response or immutable run evidence.

Reproduction uses a **different, unused** private evidence path and the exact frozen original Git blobs, while preserving all recorded earlier one-use attempts. These commands require a current SWAAG checkout with the source and additional local runtime dependencies and do not themselves authorize new model requests unless the designated scope and resource broker permit one:

```
W=/data/src/worktrees/UNUSED_FRESH_SWAAG_CHECKOUT
NEW=/data/var/swaag-benchmarks/UNUSED_NEW_ONE_USE_RUN
P=/data/venvs/riganything-py311/bin/python
export PYTHONPATH=/data/src/github/swaag/.venv/lib/python3.11/site-packages:$W/src:$W
export LD_PRELOAD=/data/var/swaag/test-deps/sqlite-3510300/libsqlite3.so.0
$P "$W/scripts/demos/check_swaag_agent_runtime.py" --repo "$W"
$P "$W/scripts/demos/prepare_fresh_reconnect_provenance_agent_demo.py" --repo "$W" --evidence "$NEW"
$P "$W/scripts/demos/run_fresh_reconnect_provenance_agent_once.py" --repo "$W" --evidence "$NEW"
# Only after an actual independent authorization, ONE USE, and broker check:
SWAAG_FRESH_RECONNECT_SOURCE_AGENT_NOACTION_ONCE=1 $P "$W/scripts/demos/run_fresh_reconnect_provenance_agent_once.py" --repo "$W" --evidence "$NEW" --run
$P "$W/scripts/demos/verify_fresh_reconnect_provenance_agent_demo.py" --repo "$W" \
 --journal "$NEW/sessions/ACTUAL_SESSION_ID/complete_history.jsonl" \
 --manifest "$NEW/manifest.json" \
 --task "$NEW/fixed-new-reconnect-task.txt" \
 --process-receipt "$NEW/result.json" \
 --out "$NEW/UNUSED_INDEPENDENT_ORACLE_RESULT.json"
```

The `prepare` command explicitly refuses existing sessions and changed source blobs; the `run` command records a one-use marker before any model request and bounds only its owned child; the evaluator refuses output overwrites, external tool calls, invented file reads, unsupported calculations, unreturned model requests, fabricated final content and nonzero process exit. Running the same previously completed task again **does not** provide independent evidence, is not needed, and could duplicate a provider effect; it is prohibited under the one-use protocol.

## Limitations and next independent SWAAG release steps

This evidence qualifies **one fixed, modest, source-audit task on one resident Coder model configuration**. It does not establish generalized autonomous competence on other tasks, real speaker enrollment, anti-spoofing or replay exclusion, physical Android input/output, live voice turn detection, AI reasoning speed, buyer-ready release, user data isolation across all configurations, or repeatable low latency. The original source itself belongs to the tool test and was not mutated by the agent. Separate previous tests already covered genuine ASR and generated Phonon speech under guarded no-action stubs; no new audio was replayed for this exercise. The separate earlier November 20 tool demo remains only `verified_after_supervisor_correction`, and the Wave65 Coder demo remains an independent documented failure.

Priorities still outstanding: bring the dedicated turn classifier online with an actual admitted resource budget and acceptable accuracy/latency; coordinate a reversible guarded Jetson voice-server release with all active consumers; authenticate the true owner with robust protection against actual voice recordings; and test the Android client on the real Blackview microphone, playback and reconnection without automatic actions. Keep automatic voice authorization off until those checks pass. Do not treat an LLM's voiceprint similarity score as a live-person authentication credential.

### Verified durable restart and exact state replay, after the successful turn

The original model process had already exited successfully. A **separate new
Python 3.11 SWAAG process** was launched using only the same private
source-pinned configuration and the read-only CLI command:

```
$P -m swaag --config "$NEW/read-only-config.toml" history replay session_e516d73aacfe
```

This command invokes `HistoryStore.rebuild_from_history`, **not** `ask`, `chat`,
`run_turn` or any model provider. It reloaded all **320** genuine historical
events and reconstructed a state with exactly **one** completed turn, the
original session ID, the exact final `last_event_hash`
`f4a76f529d5b6ba18f8675453fe327950ad0340eb693f0027ed454b84e45f337`,
and a byte-for-byte-equivalent final assistant answer (SHA-256
`5e7a0bf714370409980af247b508c3a63a6db292dec447c65939b0f5151cf9ab`).
The original `complete_history.jsonl` SHA-256 remained
`f79b108198b96ff9419638cf407aa9ff85e95c320b18ae8f634338af1fdd8c28`
before and after. The independently observed new process exited code **0**
after **1.268 seconds**. This replay issued **zero** new model requests or
tool calls and did not resend any original user message or reuse an uncertain
provider effect. The one-use private receipt is
`../fresh-reconnect-provenance-agent-once/independent-restart-replay-result.json`,
SHA-256 `b9666afd8c0dd2aa6bce637614380a683925f97a77e45a08508300f2d627273b`.

This is a real durable SWAAG **developer-session process-restart/replay**
acceptance, not a physical phone reconnection, fresh speech playback or
active model/task resumption. It does not prove that the Android client can
recover from network loss or that an interrupted *in-flight tool effect* is
safe to repeat; such requests must be verified against receipts before any
subsequent action.

### Focused source acceptance

Before source publication, a targeted regression group covering the new
source preparation, new one-use guarded runner, hash-linked independent event
oracle, original two read-only source test files, model-authority controls,
history and storage boundaries, historical voice replay and the guarded
speaker-review unit tests passed **133 tests, zero failures, zero errors and
zero skipped**, using the qualified Python 3.11 + SQLite 3.51.3 process.
The private result is
`../wave66-new-reconnect-agent-and-safety-prepublication-junit.xml`,
SHA-256 `0252c5af2c9fcbf62b22c7d7231d869702a8271d98fb46a52b5a7a296a79b84b`.
The historical 173-case Wave65 suite separately passed for the unchanged
original `src/swaag/runtime.py` recovery code, but these sets should **not**
be summed as disjoint new tests or misrepresented as the entire repository.
The final source increment is independently checked again after the Git push.
