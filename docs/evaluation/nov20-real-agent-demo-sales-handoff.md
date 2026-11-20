# Public-safe SWAAG engineering-evaluation handoff

For the ONE registered shared sales/website worker only. No publication,
website modification, customer message, outside push or sale is performed
or authorized by this handoff.

An actual locally resident Qwen model used registered SWAAG tools in a
multi-step read-only security-source audit. An initial missing file
was honestly detected, the model recovered with real directory listing
and read two real test files. Its first calculation was wrong, and a
later model request timed out without a final answer. A NEW, separately
supervised correction in the SAME durable SWAAG session used the real
calculator, corrected the count to ten, and produced a source-backed
final answer. Independent verification of actual source and tool results
returned verified_after_supervisor_correction, never first-pass success.

The first and corrective live runs took about 319 and 392 seconds,
respectively, roughly twelve minutes combined, mainly due to slow model
inference. This is a genuine demonstration of fault recording, durable
state replay and supervised recovery, NOT proof of fast reliable
autonomous execution or any customer installation. Exactly five real
tool results were successful across six tool calls, and one error was
observed. No file write, external account change or other tool side
effect was enabled.

A real source compatibility problem with Python 3.11 was reproduced
and repaired by incorporating an already tested owning-repo fix.
A separate Python 3.12 installation lacked an OpenTelemetry module,
and the standard Python 3.11 environment's SQLite was below SWAAG's
required version. Using a preexisting compatible local SQLite runtime
without weakening SWAAG's source guards, all 112 focused security,
authority, history and recovery tests passed. Additional independent
verifier tests check that fabricated or incorrect success claims are
rejected. The original failure logs remain preserved.

Exact public-safe reproduction sources:
docs/evaluation/real-readonly-agent-demonstration-2026-11-20.md,
scripts/demos/prepare_real_readonly_agent_demo.py,
scripts/demos/check_swaag_agent_runtime.py,
scripts/demos/verify_real_readonly_agent_demo.py,
scripts/demos/source_audit_prompt_v1.txt,
scripts/demos/source_audit_correction_v2.txt and their unit tests.

The relevant existing first-party feature test sources include
tests/test_history_replay.py, tests/test_tools.py,
tests/test_storage_boundaries.py, tests/test_control_authority.py
and tests/test_agent_workspace_recovery.py.
Mere presence of these files does not establish that any runtime passed.

Permitted claim: "A real read-only SWAAG agent source audit demonstrated
durable tool receipts and verified completion after an independently
documented model failure and supervisor correction."
Not permitted without new independent evidence: "autonomous first-pass
reliability," "instant multi-step execution," "production deployment,"
"fully functioning voice agent," "enterprise certified," or
"verified customers or revenue."

The private original event journal, exact model requests, source SHA
manifest and completed run logs stay in local Nitro storage. The owning
project has not modified the public website or contacted outsiders.

The separate deterministic cancellation and preemption tests
passed all forty cases. The live model run was NOT canceled
and restarted successfully: it timed out, its state replayed,
and a new supervisor correction completed the task. The final
focused Python compatibility, history, authority, tool-scope
and evidence-verification suite passed 128 tests. Do not
represent these as a complete repository-wide test run.
