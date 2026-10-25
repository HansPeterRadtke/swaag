# Observed production SWAAG failure: repeated file reads, unsupported second-source answer, excessive model calls

Date: 2026-10-25 (Europe/Berlin). Host: Thor. This is a **real deployed-model regression**, not a mocked test or statistical accuracy estimate. Inspect the intact private event journal before changing a runtime contract.

## Test and independent oracle

An authorized **read-only** test was sent to the deployed canonical orchestrator.message API on Thor, once, with a distinct durable operation identifier. The test requested actual inspection of /data/infra/AGENTS.md and /data/src/github/swaag/README.md, return of the exact first Markdown heading of each, and determination whether the Infra owner-wide rule mandates commit/push on all projects. Tool use was allowed only for read-only local files.

Independent shell checks established:
- Infra heading: "# Instructions for agents working in /data/infra".
- SWAAG heading: "# SWAAG".
- Owner-wide commit/push requirement: yes, explicitly in Infra AGENTS.md (October 25 amendment).

The first two lightweight direct production probes around this time were correct: SWAAG Qwen Coder model identity in **5.93 s**, and "no currently active background workers" in **4.72 s**, agreeing with an independent status count of 41 terminal workers.

## Actual failure

The above multi-source test took **770.2 seconds** to reach durable state completed. The first caller timed out after **24 seconds**, but the original operation was monitored by durable ID and never resubmitted.

The control trace shows a tool discovery call followed by **four repeated successful reads of only /data/infra/AGENTS.md** (action indexes two through five). There was **no read_file call for the requested SWAAG README** in that run. Each repeated read was a real successful tool receipt containing the correct Infra heading. The model nevertheless proposed and eventually delivered the exact unsupported SWAAG README heading "# SWAAG: Self-Contained, Autonomous Agent Gateway". That is demonstrably wrong: the current source's first heading is exactly "# SWAAG". It correctly answered the owner-wide Git rule, but **failed the user's explicit requirement to inspect both files and report their literal headings**.

The independent completion evaluator made a further **258-second model call** and still allowed that unsupported proposed answer to be reported as completed. Thus both the action model and completion-verification path failed on this case. There is no evidence that the second file was read under a different tool. The tool event journal is authoritative for what was actually observed; fluent model text is not proof of tool execution.

Model_response logs for the full task include action-model durations of roughly **213, 45, 51, 52, 52 and 58 seconds**, plus the **258-second** completion evaluation. All six action requests used configured **Qwen3-Coder-30B-A3B Q4_K_M**. Their recorded input budgets grew from about **71,300 to 79,800 tokens**, with roughly **67,900** for the completion evaluator. Detailed prompt component measurements show that the evaluator received ~127 kB of historical evidence and **four ~11.7 kB copies** of the same Infra read_file tool result, alongside a short original objective and candidate. Previous completed tool evidence was demonstrably present in later action requests, so do not claim it was dropped during prompt construction. The model ignored or failed to prioritize it. The Qwen3.8 Q8 model server was concurrently busy, but that activity was not causally linked to this operation.

The full model-free regression suite on the synchronized source passed **1,729**, skipped **3**, failed **0** in **377.13 seconds** under supported Thor Python 3.13 / SQLite 3.53.1. This shows how real semantic-action failures escape existing unit coverage.

## Precise provenance

The Thor runtime was executing the source via effective PYTHONPATH=/data/src/github/swaag/src, confirmed in systemd and /proc environment. The separate installed site-package is older, but is **not evidence of this live failure** because source is ahead on the import path. Last executable source commit preceded the live process start. The user-facing model and worker model paths were independently verified from actual llama.cpp /props endpoints.

Preserved original traces are in the owner's private SWAAG runtime stores on Thor: voice_sessions/history.sqlite3 (session session_0f023cd03bf8, event type agent_action_selected, tool_result, model_request_sent, model_response_received, completion evaluator), orchestrator_operations.sqlite3 (operation acceptance-20261025-readonly-multisource-v1) and inference_requests.sqlite3. Do not commit unredacted model prompts, tool outputs or private journals into the GitHub repository.

## Engineering implications, NOT yet a validated fix

Priority is **grounded task completion and unnecessary inference cost**. The user expressly requested two independent source observations; the agent completed only one, spent many model calls rereading it and asserted an unobserved false fact. The completion verifier is not an independent guarantee. This is an actual release-acceptance failure, not merely a feature wishlist.

First inspect the current source's action-loop evidence projections, tool-observation replay, semantic decision instruction, completion tool-evidence inventory, history relevance and context assembly. Use this exact case as **known development/failure evidence**, NOT as untouched validation. A defensible repair should require evidence for all expressly required observations while preserving the original request, allow legitimate repeated observations when the world changes, distinguish progress from tool replay and keep old exact history retrievable. Do not create a brittle absolute-path regex or universal ban on repeated read/search/poll; the user can discuss paths hypothetically or legitimately recheck changed external state. Consider a bounded source-linked evidence check and independent completion acceptance, benchmarked on the **actual models** against diverse held-out realistic prompts. Test correct quoting, missing/optional observations, corrections, history, permissions, backend contention, false positives/false negatives, and total latency and tool side effects. No change should reduce context at the expense of losing governing instruction evidence. Do not mistake short-prompt schema validity for final-task correctness.

This document records a **reproduced failure**, not a repaired system. The source-level and real-model repair and independent holdout acceptance remain outstanding.
