# Read-only SWAAG agent demo evidence: receipt-integrity regression, November 20, 2026

## Why another verification increment was necessary

The independently recorded real Qwen source-audit session was previously graded as verified **after separate supervisor correction**. On the original verification program, deliberately fabricated local test journals containing an orphan successful calculator result, a mismatched requested file path and a duplicated successful receipt each returned a false `verified_first_pass` verdict. These were controlled, private tests, **not** behavior observed in the original live run. The previous verifier trusted result payloads without matching them to a preceding exact `tool_called` event.

The original failed model run remains a failure. No new live model execution, production change, Android installation, user message to SWAAG, provider account change or publication was performed.

## Source-level repair and false-acceptance controls

`verify_real_readonly_agent_demo.py` now requires every reported tool result or tool error to reference a unique earlier `tool_called` event with matching `call_id`, tool name and exact validated input. Orphaned, duplicated, renamed and out-of-order receipts are rejected; a tool call without a valid outcome is not accepted as completion evidence. A terminal event and a source-grounded final answer must belong to the *latest* started turn; an earlier completion cannot prove a later unfinished correction. The final answer must identify both real source files, their required test functions and the correct total backed by arithmetic. A recorded model failure cannot be graded as first-pass success without a separate recovery turn.

Additional regression scenarios cover orphan receipt, wrong requested file, duplicate receipt, wrong tool name, result before call, an unfinished new turn and unsupported final-answer claims. The final focused verifier suite passed nine tests with zero failures. The broader targeted Python grammar, history, tool, storage, authority, recovery, verifier and preemption suite passed 171 tests with zero failures using the already-proven compatible Python 3.11 / SQLite 3.51.3 process. This is not the complete SWAAG repository-wide test suite. The original real live event journal still returns `verified_after_supervisor_correction`, with no receipt-integrity errors. Three preserved deliberately manipulated journals now return `failed_or_incomplete`. Private before/after fixtures and a machine-readable result remain under `.../agent-buyer-demo-20261120/oracle-integrity-20261120/` on Nitro. They are not customer data or a real model run.

This is **internal consistency validation**, not a cryptographic signature or proof that arbitrary journal bytes could not be forged by a privileged attacker. It assumes the local append-only event journal and the actual source checkout are trustworthy. It checks that accepted journal evidence is internally tied to recorded tool calls; it does not prove autonomous first-pass correctness or a secure multi-tenant external deployment.

## Independent live timing analysis, original journal only

The genuine resident model run had three successful completions. The recorded elapsed and time-to-first-token values in seconds were 142.473/65.496, 139.499/80.713 and 222.792/127.685. The observed completion-token throughput after the first token was approximately 3.27, 2.89 and 3.31 tokens per second. The maximum local inference-admission queue wait among four requests was about 0.419 seconds. These are measured source-session values, not predictions for other conversations or a separate throughput benchmark. The actual calculator and file tool latencies were under a quarter of a second apiece.

The dominant measured delay is **model preparation / first-token latency and slow decoding**, not local admission queueing or file-tool execution. The record cannot by itself distinguish GPU contention, model context prefill, constrained decoding, or remote inference serving effects. It would be incorrect to attribute the time entirely to a specific hardware bottleneck without an independent controlled probe.

## Reproduce without model or external actions

Read the current source `tests/test_real_readonly_agent_demo_oracle.py` and run its focused pytest suite in the previously documented compatible Python 3.11 and SQLite 3.51.3 process. Preserve the prior private journals; do not rerun the live model just to reproduce this grader regression. Re-evaluate the existing private real session journal with the versioned oracle and check it still reports `verified_after_supervisor_correction`. Verify each deliberately corrupted synthetic journal is rejected. For the exact source revision, model inputs, controlled failure and originally measured end-to-end times, see `real-readonly-agent-demonstration-2026-11-20.md`.

This verifier repair does not alter SWAAG agent tool permissions or the runtime protocol. The outstanding production gates are unchanged: live-model responsiveness, real voice/orchestrator integration, reliable speaker and final-turn authority, and a separately authorized Android release/physical-device acceptance.
