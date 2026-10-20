# Fast SWAAG context: system description, observed state, and user message

Date: 2026-10-20. Host: Thor. This update only changes the fast-stage prompt and its context projection, not the full SWAAG orchestration/worker policies or tool grants.

## Guideline contract

Infra `docs/guidlines/agents/fast_user_facing_dispatch_and_preemption.md` explicitly specifies three principal inputs: **compact system/capability description**, **fresh authoritative current-state package**, and **the entire final user request**, with relevant exact conversation context. Detailed work is for the persistent orchestrator after the initial reply. The previous fast-stage system template mixed application description with about four hundred words of repetitive behavioral instructions/examples. The instructions defining the `answer` versus `delegate` contract and forbidding unsupported effect claims are still necessary; full tool/worker policy is not.

The revised fast-stage prompt uses a shorter protocol explaining exactly which complete conversational requests can be answered in place, which requests must go to the unchanged full controller, and that quoted/hypothetical instructions and later user corrections must be interpreted semantically. Its remaining instructions are *routing and truthfulness rules*, not a second tool authority.

## Exact input partitions

- `fast_system_description`: configured SWAAG application identity, quick interface limitations, full controller's capabilities and requested foreground model. If observed, the configured default worker model is added, correctly labeled as configured rather than weight-attested.
- `fast_system_state`: prior-message count left available in exact history and, only for a snapshot captured within three seconds, its timestamp, age, active worker/plan counts and up to four opaque IDs/statuses. A stale or invalid snapshot carries an explicit stale marker, not an authoritative liveness claim. Arbitrary worker objective/output prose is never elevated into a system instruction.
- `fast_recent_conversation`: recent entire messages, not truncated snippets, within the existing local bounded context; older history remains available to the full controller.
- `fast_authoritative_user`: the complete current finalized user request, unchanged. The separate small portable JSON output schema remains the mechanical contract.

The full orchestrator retains the canonical history, durable user policy, worker state, tools, per-action error evidence, and independent verification. The fast stage still declines when persistent non-core user instructions, notes, open questions or attachments could be omitted unsafely. There is no special keyword router that executes tools: the fast model produces only a reply and whether to forward the original request.

## Tested size and quality

Compared with the earlier isolated fast-stage production baseline, the static prompt fell from **821 to approximately 692 actual Qwen-Coder prompt tokens**. Its earlier overall small-turn input was **991 tokens**; the newer input is roughly **970 tokens** including the newly separated, richer description/state. This is primarily a simplification and clearer information architecture, not an enormous total token reduction; the measured non-model code overhead was already small.

The earliest aggressively truncated prompt (~475 system tokens) classified only 12 of 16 representative live Coder cases correctly and inappropriately claimed microphone success. After retaining compact routing/truthfulness cases, the revised prompt classified **16 of 16 representative cases correctly** in the current isolated Qwen-Coder trials. A separate 16-case held-out set scored **14 of 16**; both errors were **conservative extra delegations**, not missed actionable requests, although these still add unnecessary deep latency. These small test sets do not prove general routing accuracy. No model/server, worker, or source archive was modified by the trials, and the user's live session was not used for model generation.

Focused deterministic regression tests passed, including verified system-description/state separation, stale snapshot suppression, a bounded static prompt, original receipt and session replay, an early progress message before deep work, user corrections, retained durable user policies, and deep tool verification. The installed production communication daemon requires a restart to import any source change; no active worker should be interrupted to enable it.
