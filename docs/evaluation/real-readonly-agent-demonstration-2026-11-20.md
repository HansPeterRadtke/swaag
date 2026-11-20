# SWAAG authentic read-only multi-step agent audit, November 20 2026

**Real model, real SWAAG tools, durable replay. Independently verified AFTER
a supervisor-directed corrective turn. NOT first-attempt autonomous success.**
This is a developer-session demonstration, not a normal user-facing voice
turn, customer success, production release or paid product.

## Fixed task and execution boundary

Source revision at baseline: SWAAG main fdfd745. Test branch:
fix/agent-buyer-real-demo-py311-20261120. The original task explicitly
requested read_file on a deliberately obsolete path,
tests/test_history_integrity.py, then recovery with list_files when the
file did not exist; real separate reads of
tests/test_history_replay.py and tests/test_storage_boundaries.py;
identification of their history-tamper and symlink-escape test functions;
count their top-level test functions and combine the counts using
calculator rather than mental arithmetic.

The independent source AST oracle found seven history-replay tests and
three storage-boundary tests, totaling ten. Relevant exact source test
names are test_external_history_tampering_is_still_detected_with_file_stat_cache
and test_session_storage_rejects_symlink_escape. The files were matched
against their source SHA-256 by an independent verifier. Input prompt
and source hashes are retained in the private run manifest and logs.

The actual SWAAG model was already resident Qwen3.8-27B Q8 on Thor's
private VPN, with cache disabled. The original developer test configured
ONLY list_files, read_file and calculator tools, with the read root
confined to the SWAAG source worktree. Side effects, shell, writes,
external MCP, worker controls, purchases and all outside actions were
mechanically unavailable. The corrective turn allowed ONLY calculator.
No other shared model was stopped or restarted.

## Reproduced environment failures and supported repair

SWAAG declares requires-python >=3.11. Python 3.11.14 failed to parse
the multiline f-string expression at foreground_fast.py line 66, a
source compatibility defect. The exact same source parses under 3.12.
A separate Python 3.12.3 environment failed importing
opentelemetry.context, which is a dependency-environment failure,
NOT a software assertion failure.

An already-tested isolated SWAAG branch had an appropriate Python 3.11
syntax fix. That existing commit was cherry-picked, preserving origin,
into this development worktree. Source now compiles under Python 3.11.

The first Python 3.11 test attempt still had 35 test failures and
77 passes because its statically linked SQLite was version 3.50.4:
the actual durable HistoryStore requires at least 3.51.3. No database
minimum or guard was weakened. Nitro already had a suitable
SQLite 3.51.3 shared library and a separate Python 3.11.9 Conda
interpreter dynamically linked to SQLite. One process-only
LD_PRELOAD plus the existing dependency site-packages resolved the
environment without changing any production service. Focused actual
SWAAG tests then PASSED all 112 cases: source parse, history and
tampering, enabled-tool restrictions, filesystem/symlink boundaries,
cross-worker user-source authority and error/artifact retention.
The new versioned preflight distinguishes source grammar failures,
missing dependencies and actual runtime SQLite incompatibility.

## Real V1 execution and controlled error

In the first real model-generated action, SWAAG called five actual tools.
The obsolete read_file path failed with a recorded FilesystemError.
The model recovered with list_files on tests, listing 167 entries,
then read both requested test files separately. Actual tool results
contained 8,179 and 2,164 characters, respectively; the independent
verifier matched both result texts to the precise on-disk files.

However, the model then called calculator with 0 + 0 and received zero,
not the requested 7 + 3. The first model action took 142.473 seconds
for 252 completion tokens. Its subsequent model request exceeded
its hard 160-second generation deadline, recorded model_unavailable
and ReadTimeout and NEVER returned a completed answer. First process
exited nonzero after 318.51 seconds. This run was NOT successful.

## Real durable replay and explicit correction

A fresh SWAAG process reopened the same durable session, successfully
restoring the original user task, five tool calls, the missing-file
error, both real reads, the incorrect calculation and the model failure.

A SEPARATE, explicitly new corrective user turn was sent to the SAME
session with ONLY the calculator enabled. The corrective prompt
reported the independently observed seven and three function counts
and the earlier wrong 0 + 0. The model called the REAL calculator
with 7 + 3, which returned ten. The model then generated an accurate,
source-backed final answer with both exact test names, explicit missing
file recovery and both original and repaired calculation. No files
were reread or modified by this second turn.

The corrective action model call took 139.499 seconds for 170 tokens.
Final reply generation took 222.792 seconds for 315 tokens.
The corrective process exited successfully after 391.82 seconds.
Combined sequential command wall duration was 710.33 seconds.
These measured timings, particularly the very slow model calls,
are NOT acceptable evidence of smooth interactive performance.

The final session has 341 durable events: two user turns, four model
requests, three completed model responses, one model timeout/failure,
six real tool calls, five successful results, one real tool error,
and one completed terminal turn. A new process independently replayed
this final 341-event history and confirmed the original failure and
correction remained present.

## Independent accept/reject oracle

The actual private event journal was examined with the versioned
scripts/demos/verify_real_readonly_agent_demo.py. It requires actual
file-not-found and subsequent real listing, exact two distinct source
file reads, the correct actual calculator result, required test names,
tool allowlist, one durable session, a terminal response and a final
answer supported by those receipts.

The observed verdict was exactly:
verified_after_supervisor_correction

A subsequent independent verifier-integrity regression identified and repaired false acceptance of fabricated, mismatched or duplicated journal receipts and generic unsupported final answers. The preserved live session still passes the hardened checks. See `read-only-demo-verifier-integrity-2026-11-20.md` for the real/synthetic distinction and measured first-token delays.

It was NOT verified_first_pass. Additional pure unit tests ensure this
verifier refuses a fabricated assistant claim, a changed source read,
wrong arithmetic, a missing terminal event, or false recovery status.

## Reproducibility and artifact provenance

Versioned public-safe source: this document, the separate shared-sales
handoff, scripts/demos/prepare_real_readonly_agent_demo.py,
scripts/demos/check_swaag_agent_runtime.py,
scripts/demos/verify_real_readonly_agent_demo.py,
the immutable source_audit_prompt_v1.txt and
source_audit_correction_v2.txt, and matching tests for the scripts.

The preparation script takes a known SWAAG source worktree, an unused
private evidence directory and the preauthorized model endpoint. It
records exact source revision, input/source hashes, model/tool config,
and fixed prompts without making any model call. It refuses overwrites
and model hosts outside the approved private routes. To reproduce,
first run its preflight under a compatible Python and SQLite runtime,
consult the resource broker and run a new source-audit developer session
with the generated V1 config and task text. Inspect the actual result
and durably recorded failure BEFORE deciding whether a separate,
explicitly authorized V2 corrective turn is warranted. Never replay
an uncertain request or a side effect. Run the independent oracle on
the complete session journal; a plausible answer alone does not pass.

Private original evidence remains on Nitro under
/data/var/swaag-benchmarks/agent-buyer-demo-20261120.
Main files include run-v1-console.log, run-v2-correction-console.log,
focused-py311-real-tests.log, focused-py311-sqlite3513.log,
sessions/session_1db4ece0588e/complete_history.jsonl,
oracle-after-correction.json and
verified-post-correction-durable-replay.json.
Do not commit or publish the raw journal, model prompts, credentials
or private generated user results.

This demo only proves actual read-only developer-session tool dispatch,
safe tool scope, durable failure evidence and a successful manually
directed recovery. It does NOT prove general autonomous first-pass
correctness, buyer demand, production reliability, enterprise security
approval, or the full SWAAG voice-agent/Android lifecycle. The normal
foreground voice interface, human speaker enrollment, semantic
end-of-turn, TTS and physical device acceptance remain separate.

## Cancellation evidence, separately qualified

The owning repository's isolated preemption and run-cancellation
regression suite was run under the same compatible Python 3.11/SQLite
process. All **40 tests passed**. Those tests cover preservation of
exact cancellation reasons, preempted model-request provenance,
concurrent interruption and correctly replayed requests. They use
deterministic test model transports. This live Qwen demonstration did
NOT exercise real model preemption or a live cancellation, only one
model timeout followed by an explicit corrective user turn. Do not
conflate these independent acceptance levels.

The complete final focused Python grammar, history, storage, tool,
authority, recovery and independent demo-verifier suite PASSED
**128 tests with no failures**. These 128 and the separate 40
preemption tests do not constitute SWAAG's entire test suite.

## Exact tested command shape and safe fresh reproduction

The fixed source and model backend must be provisioned, with the
independent resource broker admitting any live model requests.
On Nitro the actual successful Python 3.11/SQLite test process used:

    export D=/data/src/worktrees/swaag-agent-buyer-demo-20261120
    export PYTHONPATH=/data/src/github/swaag/.venv/lib/python3.11/site-packages:$D/src:$D
    export LD_PRELOAD=/data/var/swaag/test-deps/sqlite-3510300/libsqlite3.so.0
    export P=/data/venvs/riganything-py311/bin/python
    cd "$D"
    $P scripts/demos/check_swaag_agent_runtime.py --repo "$D"

An external reproducer must provide their own compatible Python
installation and SQLite, rather than assuming these host-local
private paths exist elsewhere. To prepare a NEW, unused private run
directory, using only the versioned fixed inputs and tool contract:

    export NEW_PRIVATE_EVIDENCE_DIR=/data/var/swaag-benchmarks/UNUSED-NEW-READONLY-DEMO
    $P scripts/demos/prepare_real_readonly_agent_demo.py --repo "$D" --evidence "$NEW_PRIVATE_EVIDENCE_DIR" --model-url http://10.8.0.7:14835

The actual real agent entrypoint for the INITIAL request is:

    $P -m swaag --config "$NEW_PRIVATE_EVIDENCE_DIR/demo-readonly-config.toml" ask --session buyer-demo-audit-v1 < "$NEW_PRIVATE_EVIDENCE_DIR/demo-task-v1.txt"

This initial attempt was a REAL failure, not authorization
to try again with the same idempotency or effects. Only after
the exact recorded model failure and an independent supervisor
check should a distinct correction be deliberately initiated
on the SAME durable session:

    $P -m swaag --config "$NEW_PRIVATE_EVIDENCE_DIR/demo-readonly-config.toml" --config "$NEW_PRIVATE_EVIDENCE_DIR/demo-correction-config.toml" ask --session buyer-demo-audit-v1 < "$NEW_PRIVATE_EVIDENCE_DIR/demo-correction-v2.txt"

Afterward, inspect the actual session ID recorded in the private
run directory. The verifier is invoked on THAT journal, not
a claimed or newly invented session:

    python3 scripts/demos/verify_real_readonly_agent_demo.py --repo "$D" --journal "$NEW_PRIVATE_EVIDENCE_DIR/sessions/ACTUAL_SESSION_ID/complete_history.jsonl" --out "$NEW_PRIVATE_EVIDENCE_DIR/oracle-result.json"

These commands reproduce the bounded procedure, not a
guarantee of the same model decisions, elapsed time or
successful correction. Never replay an uncertain action.
The original live commands, two separate console logs,
source files, actual result and independent verdict are
retained in the private recorded evidence directory.
